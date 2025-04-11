# -----------------------------------------------------------------------------
#
# Copyright (c) 2024 Qualcomm Innovation Center, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------

import torch
import torch.nn as nn
import torch.utils.checkpoint
from transformers.models.mistral3.modeling_mistral3 import Mistral3ForConditionalGeneration

from QEfficient.utils import constants
from QEfficient.utils._utils import IOInfo, get_padding_shape_from_config

BS = 1
NUM_CHANNEL = 3
SEQ_LEN = 3072
CTX_LEN = 4096


class QEFFMistral3EncoderWrapper(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
        self.model.vision_model = self.model.vision_tower

    def forward(self, pixel_values):
        image_sizes = torch.tensor([[pixel_values.shape[2], pixel_values.shape[3]]])
        image_features = self.model.get_image_features(
            pixel_values=pixel_values,
            vision_feature_layer=self.model.config.vision_feature_layer,
            image_sizes=image_sizes,
        )
        return image_features


class QEFFMistral3DecoderWrapper(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
        self.config = self.model.config
        self.language_model = self.model.language_model

    def forward(self, input_ids, vit_embeds, position_ids, past_key_values):
        inputs_embeds = self.model.get_input_embeddings()(input_ids)
        vit_embeds = vit_embeds.to(inputs_embeds.device, inputs_embeds.dtype)
        mask = input_ids == self.model.config.image_token_index
        indices1 = mask.to(torch.int64).cumsum(1) - 1
        indices0 = torch.arange(mask.shape[0]).view(-1, 1)
        image_features_expanded = vit_embeds.unsqueeze(0)[indices0, indices1]
        inputs_embeds = torch.where(mask.unsqueeze(-1), image_features_expanded, inputs_embeds)
        outputs = self.model.language_model(
            inputs_embeds=inputs_embeds,
            position_ids=position_ids,
            past_key_values=past_key_values,
        )

        return outputs.logits, vit_embeds, outputs.past_key_values


class QEffMistral3ForConditionalGeneration(Mistral3ForConditionalGeneration):
    def get_qeff_vision_encoder(self):
        return QEFFMistral3EncoderWrapper(self)

    def get_qeff_language_decoder(self):
        return QEFFMistral3DecoderWrapper(self)

    def forward(self, pixel_values, input_ids, position_ids, past_key_values):
        inputs_embeds = self.get_input_embeddings()(input_ids)
        # Image features
        image_sizes = torch.tensor([[pixel_values.shape[2], pixel_values.shape[3]]])
        image_features = self.get_image_features(
            pixel_values=pixel_values,
            vision_feature_layer=self.config.vision_feature_layer,
            image_sizes=image_sizes,
        )
        image_features = image_features.to(inputs_embeds.device, inputs_embeds.dtype)
        mask = input_ids == self.config.image_token_index
        indices1 = mask.to(torch.int64).cumsum(1) - 1
        indices0 = torch.arange(mask.shape[0]).view(-1, 1)
        image_features_expanded = image_features.unsqueeze(0)[indices0, indices1]
        inputs_embeds = torch.where(mask.unsqueeze(-1), image_features_expanded, inputs_embeds)
        outputs = self.language_model(
            inputs_embeds=inputs_embeds,
            position_ids=position_ids,
            past_key_values=past_key_values,
        )
        return outputs.logits, pixel_values, outputs.past_key_values

    def get_dummy_inputs(self, kv_offload: bool = False, **kwargs):
        inputs_shapes = {}
        inputs_shapes["input_ids"] = (constants.ONNX_EXPORT_EXAMPLE_BATCH_SIZE, constants.ONNX_EXPORT_EXAMPLE_SEQ_LEN)
        inputs_shapes["vit_embeds"] = (
            constants.MISTRAL3_FEATURE_SIZE,
            self.language_model.config.hidden_size,
        )
        inputs_shapes["position_ids"] = (
            constants.ONNX_EXPORT_EXAMPLE_BATCH_SIZE,
            constants.ONNX_EXPORT_EXAMPLE_SEQ_LEN,
        )
        inputs_shapes["pixel_values"] = (
            constants.ONNX_EXPORT_EXAMPLE_BATCH_SIZE,
            constants.MISTRAL3_NUM_CHANNELS,
            constants.MISTRAL3_HEIGHT,
            constants.MISTRAL3_WIDTH,
        )

        # Define inputs
        vision_inputs = {}
        lang_inputs = {}
        vision_inputs["pixel_values"] = torch.zeros((inputs_shapes["pixel_values"]), dtype=torch.float32)
        lang_inputs["input_ids"] = torch.zeros((inputs_shapes["input_ids"]), dtype=torch.int64)
        lang_inputs["vit_embeds"] = torch.zeros((inputs_shapes["vit_embeds"]), dtype=torch.float32)
        lang_inputs["position_ids"] = (
            torch.arange(constants.ONNX_EXPORT_EXAMPLE_SEQ_LEN, dtype=torch.int64)
            .view(1, constants.ONNX_EXPORT_EXAMPLE_SEQ_LEN)
            .repeat(constants.ONNX_EXPORT_EXAMPLE_BATCH_SIZE, 1)
        )

        # Add data for KV
        kv_cache_shape = get_padding_shape_from_config(
            config=self.language_model.config,
            batch_size=constants.ONNX_EXPORT_EXAMPLE_BATCH_SIZE,
            seq_len=constants.ONNX_EXPORT_EXAMPLE_SEQ_LEN,
        )

        lang_inputs["past_key_values"] = [[] for _ in range(self.language_model.config.num_hidden_layers)]
        for i in range(self.language_model.config.num_hidden_layers):
            for kv in ["key", "value"]:
                lang_inputs["past_key_values"][i].append(torch.zeros(kv_cache_shape, dtype=torch.float32))

        inputs = {}
        if kv_offload:
            inputs["vision"] = vision_inputs
            inputs["lang"] = lang_inputs
        else:
            lang_inputs.pop("vit_embeds")
            inputs = {**vision_inputs, **lang_inputs}

        return inputs

    def get_specializations(
        self,
        batch_size: int,
        prefill_seq_len: int,
        ctx_len: int,
        img_size: int,
        kv_offload: bool = False,
        **compiler_options,
    ):
        prefill_seq_len = prefill_seq_len if prefill_seq_len else SEQ_LEN
        ctx_len = ctx_len if ctx_len else CTX_LEN
        height = constants.MISTRAL3_HEIGHT
        width = constants.MISTRAL3_WIDTH

        vision = [
            {
                "batch_size": batch_size,
                "seq_len": prefill_seq_len,
                "ctx_len": ctx_len,
                "height": height,
                "width": width,
            }
        ]
        lang = [
            {
                "batch_size": batch_size,
                "seq_len": prefill_seq_len,
                "ctx_len": ctx_len,
                "height": height,
                "width": width,
            },
            {
                "batch_size": batch_size,
                "seq_len": "1",
                "ctx_len": ctx_len,
                "height": height,
                "width": width,
            },
        ]
        specializations = {}

        if kv_offload:
            specializations["vision"] = vision
            specializations["lang"] = lang
            return specializations, compiler_options
        else:
            return lang, compiler_options

    def get_onnx_dynamic_axes(self, kv_offload: bool = False):
        # Define dynamic axes
        num_layers = self.config.text_config.num_hidden_layers

        vision_dynamic_axes = {
            "pixel_values": {0: "batch_size", 2: "height", 3: "width"},
        }
        lang_dynamic_axes = {
            "input_ids": {0: "batch_size", 1: "seq_len"},
            "position_ids": {0: "batch_size", 1: "seq_len"},
        }

        for i in range(num_layers):
            lang_dynamic_axes[f"past_key.{i}"] = {0: "batch_size", 2: "ctx_len"}
            lang_dynamic_axes[f"past_value.{i}"] = {0: "batch_size", 2: "ctx_len"}

        dynamic_axes = {}
        if kv_offload:
            dynamic_axes["vision"] = vision_dynamic_axes
            dynamic_axes["lang"] = lang_dynamic_axes
        else:
            dynamic_axes = {**vision_dynamic_axes, **lang_dynamic_axes}
        return dynamic_axes

    def get_output_names(self, kv_offload: bool = False):
        vision_output_names = ["vit_embeds"]
        lang_output_names = ["logits"]
        for i in range(self.language_model.config.num_hidden_layers):
            for kv in ["key", "value"]:
                lang_output_names.append(f"past_{kv}.{i}_RetainedState")

        output_names = {}
        if kv_offload:
            lang_output_names.insert(1, "vit_embeds_RetainedState")
            output_names["vision"] = vision_output_names
            output_names["lang"] = lang_output_names
        else:
            lang_output_names.insert(1, "pixel_values_RetainedState")
            return lang_output_names
        return output_names

    def get_inputs_info(self):
        return [
            IOInfo(name="input_ids", datatype=torch.int64, shape=("batch_size", "seq_len")),
            IOInfo(name="attention_mask", datatype=torch.int64, shape=("batch_size", "seq_len")),
            IOInfo(name="pixel_values", datatype=torch.float32, shape=("batch_size", 3, "height", "width")),
        ]

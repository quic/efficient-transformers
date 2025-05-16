# -----------------------------------------------------------------------------
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------

import torch
import torch.nn as nn
import torch.utils.checkpoint
from transformers.models.llava.modeling_llava import (
    LlavaForConditionalGeneration,
)

from QEfficient.utils._utils import IOInfo
from QEfficient.utils.logging_utils import logger

BS = 1
NUM_CHANNEL = 3
SEQ_LEN = 592
CTX_LEN = 1024


class QEFFLlavaEncoderWrapper(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
        self.model.vision_model = self.model.vision_tower

    def forward(self, pixel_values):
        # Image features
        image_outputs = self.model.vision_tower(pixel_values, output_hidden_states=True)
        selected_image_feature = image_outputs.hidden_states[self.model.config.vision_feature_layer]
        vision_feature_select_strategy = self.model.config.vision_feature_select_strategy
        if vision_feature_select_strategy == "default":
            selected_image_feature = selected_image_feature[:, 1:]
        elif vision_feature_select_strategy == "full":
            selected_image_feature = selected_image_feature
        else:
            raise ValueError(f"Unexpected select feature strategy: {self.model.config.vision_feature_select_strategy}")
        vision_embeds = self.model.multi_modal_projector(selected_image_feature)

        return vision_embeds


class QEFFLlavaDecoderWrapper(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
        self.config = self.model.config
        self.language_model = self.model.language_model

    def forward(self, input_ids, vision_embeds, position_ids, past_key_values):
        inputs_embeds = self.model.get_input_embeddings()(input_ids)
        vision_embeds = vision_embeds.to(inputs_embeds.device, inputs_embeds.dtype)
        mask = input_ids == self.model.config.image_token_index
        indices1 = mask.to(torch.int64).cumsum(1) - 1
        indices0 = torch.arange(mask.shape[0]).view(-1, 1)
        vision_embeds_expanded = vision_embeds[indices0, indices1]
        inputs_embeds = torch.where(mask.unsqueeze(-1), vision_embeds_expanded, inputs_embeds)
        outputs = self.model.language_model(
            inputs_embeds=inputs_embeds,
            position_ids=position_ids,
            past_key_values=past_key_values,
        )

        return outputs.logits, vision_embeds, outputs.past_key_values


class QEffLlavaForConditionalGeneration(LlavaForConditionalGeneration):
    def get_qeff_vision_encoder(self):
        return QEFFLlavaEncoderWrapper(self)

    def get_qeff_language_decoder(self):
        return QEFFLlavaDecoderWrapper(self)

    def forward(self, input_ids, position_ids, pixel_values, past_key_values):
        inputs_embeds = self.get_input_embeddings()(input_ids)
        # Image features
        image_outputs = self.vision_tower(pixel_values, output_hidden_states=True)
        selected_image_feature = image_outputs.hidden_states[self.config.vision_feature_layer]
        vision_feature_select_strategy = self.config.vision_feature_select_strategy
        if vision_feature_select_strategy == "default":
            selected_image_feature = selected_image_feature[:, 1:]
        elif vision_feature_select_strategy == "full":
            selected_image_feature = selected_image_feature
        else:
            raise ValueError(f"Unexpected select feature strategy: {self.config.vision_feature_select_strategy}")
        vision_embeds = self.multi_modal_projector(selected_image_feature)
        vision_embeds = vision_embeds.to(inputs_embeds.device, inputs_embeds.dtype)

        mask = input_ids == self.config.image_token_index
        indices1 = mask.to(torch.int64).cumsum(1) - 1
        indices0 = torch.arange(mask.shape[0]).view(-1, 1)
        vision_embeds_expanded = vision_embeds[indices0, indices1]
        image_inputs_embeds = torch.where(mask.unsqueeze(-1), vision_embeds_expanded, inputs_embeds)
        # *where to skip image encoder for decode*
        inputs_embeds = torch.where(input_ids.shape[1] == torch.tensor(1), inputs_embeds, image_inputs_embeds)
        outputs = self.language_model(
            inputs_embeds=inputs_embeds,
            position_ids=position_ids,
            past_key_values=past_key_values,
        )
        return outputs.logits, pixel_values, outputs.past_key_values

    def get_dummy_inputs(self, kv_offload: bool = False, **kwargs):
        num_layers = self.config.text_config.num_hidden_layers
        num_key_value_heads = self.config.text_config.num_key_value_heads
        head_dim = self.config.text_config.hidden_size // self.config.text_config.num_attention_heads
        if vis_cfg := getattr(self.config, "vision_config", None):
            img_size = getattr(vis_cfg, "image_size", 336)
        else:
            img_size = 336
        if img_size != 336 and kv_offload:
            raise NotImplementedError("Image Size other than 336 is not supported for Llava models yet.")
        vision_inputs = {
            "pixel_values": torch.zeros((BS, NUM_CHANNEL, img_size, img_size), dtype=torch.float32),
        }
        lang_inputs = {
            "input_ids": torch.ones((BS, SEQ_LEN), dtype=torch.int64),
            "vision_embeds": torch.ones((BS, 576, self.language_model.config.hidden_size), dtype=torch.float32),
            "attention_mask": torch.ones((BS, SEQ_LEN), dtype=torch.int64),
        }
        lang_inputs["position_ids"] = lang_inputs.pop("attention_mask").cumsum(1)
        lang_inputs["past_key_values"] = []
        for i in range(num_layers):
            lang_inputs["past_key_values"].append(
                (
                    torch.zeros(BS, num_key_value_heads, CTX_LEN, head_dim),
                    torch.zeros(BS, num_key_value_heads, CTX_LEN, head_dim),
                )
            )
        lang_inputs["position_ids"] = torch.full(lang_inputs["position_ids"].shape, CTX_LEN - 1)
        inputs = {}

        if kv_offload:
            inputs["vision"] = vision_inputs
            inputs["lang"] = lang_inputs
        else:
            lang_inputs.pop("vision_embeds")
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
        max_num_images = compiler_options.pop("max_num_images", 1)
        prefill_seq_len = prefill_seq_len if prefill_seq_len else SEQ_LEN
        ctx_len = ctx_len if ctx_len else CTX_LEN
        if img_size is None and hasattr(self.config.vision_config, "image_size"):
            img_size = getattr(self.config.vision_config, "image_size")
        elif img_size is None:
            img_size = 336
            logger.warning("Setting img_size to be 336, as it was neither passed nor found in vision_config")
        if img_size != 336 and kv_offload:
            raise NotImplementedError("Image Size other than 336 is not supported for Llava models yet.")
        vision = [
            {
                "batch_size": batch_size,
                "max_num_images": max_num_images,
                "img_size": img_size,
                "seq_len": prefill_seq_len,
                "ctx_len": ctx_len,
            }
        ]
        lang = [
            {
                "batch_size": batch_size,
                "seq_len": prefill_seq_len,
                "ctx_len": ctx_len,
                "max_num_images": max_num_images,
                "img_size": img_size,
            },
            {
                "batch_size": batch_size,
                "seq_len": "1",
                "ctx_len": ctx_len,
                "max_num_images": max_num_images,
                "img_size": img_size,
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
            "pixel_values": {0: "batch_size", 2: "img_size", 3: "img_size"},
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
        vision_output_names = ["vision_embeds"]
        lang_output_names = ["logits"]
        for i in range(self.language_model.config.num_hidden_layers):
            for kv in ["key", "value"]:
                lang_output_names.append(f"past_{kv}.{i}_RetainedState")

        output_names = {}
        if kv_offload:
            lang_output_names.insert(1, "vision_embeds_RetainedState")
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
            IOInfo(name="pixel_values", datatype=torch.float32, shape=("batch_size", 3, "img_size", "img_size")),
        ]

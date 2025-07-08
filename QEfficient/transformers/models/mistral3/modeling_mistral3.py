# -----------------------------------------------------------------------------
#
# Copyright (c) 2024 Qualcomm Innovation Center, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------

from typing import Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.utils.checkpoint
from transformers.modeling_outputs import BaseModelOutput
from transformers.models.mistral3.modeling_mistral3 import Mistral3ForConditionalGeneration
from transformers.models.pixtral.modeling_pixtral import PixtralVisionModel, position_ids_in_meshgrid

from QEfficient.utils import constants
from QEfficient.utils._utils import IOInfo, get_padding_shape_from_config


def custom_cumsum(tensor):
    dim = 0
    result = torch.zeros_like(tensor)
    indices = [slice(None)] * tensor.dim()
    for i in range(tensor.size(dim)):
        indices[dim] = slice(0, i + 1)
        result.select(dim, i).copy_(tensor[tuple(indices)].sum(dim))
    return result


def qeff_generate_block_attention_mask(patch_embeds_list, tensor):
    dtype = tensor.dtype
    device = tensor.device
    seq_len = tensor.shape[1]
    d_min = torch.finfo(dtype).min
    causal_mask = torch.full((seq_len, seq_len), fill_value=d_min, dtype=dtype, device=device)
    block_end_idx = torch.tensor(patch_embeds_list).cumsum(-1)
    block_end_idx = custom_cumsum(torch.tensor(patch_embeds_list))
    block_start_idx = custom_cumsum(torch.tensor([0] + patch_embeds_list[:-1]))
    for start, end in zip(block_start_idx.tolist(), block_end_idx.tolist()):
        causal_mask[start:end, start:end] = 0
    causal_mask = causal_mask[None, None, :, :].expand(tensor.shape[0], 1, -1, -1)
    return causal_mask


class QEffPixtralVisionModel(PixtralVisionModel):
    def forward(
        self,
        pixel_values: torch.Tensor,
        image_sizes: torch.Tensor,
        output_hidden_states: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        *args,
        **kwargs,
    ) -> Union[Tuple, BaseModelOutput]:
        """
        Returns:
            pixel_values: tensor of token features for
                all tokens of all images of shape (N_toks, D)
        """
        # pass images through initial convolution independently
        patch_embeds = self.patch_conv(pixel_values)
        patch_embeds_list = [
            embed[..., : (size[0] // self.patch_size), : (size[1] // self.patch_size)]
            for embed, size in zip(patch_embeds, image_sizes)
        ]

        # flatten to a single sequence
        patch_embeds = torch.cat([p.flatten(1).T for p in patch_embeds_list], dim=0).unsqueeze(0)
        patch_embeds = self.ln_pre(patch_embeds)

        # positional embeddings
        position_ids = position_ids_in_meshgrid(
            patch_embeds_list, max_width=self.config.image_size // self.config.patch_size
        )
        position_embeddings = self.patch_positional_embedding(patch_embeds, position_ids)

        attention_mask = qeff_generate_block_attention_mask(
            [p.shape[-2] * p.shape[-1] for p in patch_embeds_list], patch_embeds
        )

        out = self.transformer(
            patch_embeds,
            attention_mask=attention_mask,
            position_embeddings=position_embeddings,
            output_hidden_states=output_hidden_states,
            output_attentions=output_attentions,
            return_dict=return_dict,
        )

        return out


class QEFFMistral3EncoderWrapper(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
        self.model.vision_model = self.model.vision_tower

    def forward(self, pixel_values):
        image_sizes = torch.tensor([[pixel_values.shape[2], pixel_values.shape[3]]]).repeat(pixel_values.shape[0], 1)
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

    def forward(self, input_ids, vision_embeds, position_ids, image_idx, past_key_values):
        inputs_embeds = self.model.get_input_embeddings()(input_ids)
        vision_embeds = vision_embeds.to(inputs_embeds.device, inputs_embeds.dtype)
        mask = input_ids == self.model.config.image_token_index
        indices1 = mask.to(torch.int64).cumsum(1) - 1
        indices1 = torch.where(indices1 != -1, indices1 + image_idx, indices1)
        indices0 = torch.arange(mask.shape[0]).view(-1, 1)
        image_features_expanded = vision_embeds.unsqueeze(0)[indices0, indices1]
        inputs_embeds_1 = torch.where(mask.unsqueeze(-1), image_features_expanded, inputs_embeds)
        outputs = self.model.language_model(
            inputs_embeds=inputs_embeds_1,
            position_ids=position_ids,
            past_key_values=past_key_values,
        )
        next_idx = (indices1.max() + 1).unsqueeze(0).unsqueeze(0)
        image_idx = torch.where(image_idx < next_idx, next_idx, image_idx)
        return outputs.logits, vision_embeds, image_idx, outputs.past_key_values


class QEffMistral3ForConditionalGeneration(Mistral3ForConditionalGeneration):
    def get_qeff_vision_encoder(self):
        return QEFFMistral3EncoderWrapper(self)

    def get_qeff_language_decoder(self):
        return QEFFMistral3DecoderWrapper(self)

    def forward(self, input_ids, position_ids, pixel_values, image_idx, past_key_values):
        inputs_embeds = self.get_input_embeddings()(input_ids)
        image_sizes = torch.tensor([[pixel_values.shape[2], pixel_values.shape[3]]]).repeat(pixel_values.shape[0], 1)
        image_features = self.get_image_features(
            pixel_values=pixel_values,
            vision_feature_layer=self.config.vision_feature_layer,
            image_sizes=image_sizes,
        )
        image_features = image_features.to(inputs_embeds.device, inputs_embeds.dtype)
        mask = input_ids == self.config.image_token_index
        indices1 = mask.to(torch.int64).cumsum(1) - 1
        indices1 = torch.where(indices1 != -1, indices1 + image_idx, indices1)
        indices0 = torch.arange(mask.shape[0]).view(-1, 1)
        image_features_expanded = image_features.unsqueeze(0)[indices0, indices1]
        image_embeds = torch.where(mask.unsqueeze(-1), image_features_expanded, inputs_embeds)
        inputs_embeds = torch.where(input_ids.shape[1] == torch.tensor(1), inputs_embeds, image_embeds)
        outputs = self.language_model(
            inputs_embeds=inputs_embeds,
            position_ids=position_ids,
            past_key_values=past_key_values,
        )
        next_idx = (indices1.max() + 1).unsqueeze(0).unsqueeze(0)
        image_idx = torch.where(image_idx < next_idx, next_idx, image_idx)

        return outputs.logits, pixel_values, image_idx, outputs.past_key_values

    def get_dummy_inputs(self, kv_offload: bool = False, **kwargs):
        inputs_shapes = {}
        inputs_shapes["input_ids"] = (constants.ONNX_EXPORT_EXAMPLE_BATCH_SIZE, constants.ONNX_EXPORT_EXAMPLE_SEQ_LEN)
        height = self.config.vision_config.image_size
        width = self.config.vision_config.image_size
        patch_size = self.config.vision_config.patch_size
        kernel_size = self.config.spatial_merge_size
        vision_size = (
            ((height // patch_size) * (width // patch_size))
            * (constants.ONNX_EXPORT_EXAMPLE_BATCH_SIZE)
            // (kernel_size * kernel_size)
        )
        inputs_shapes["vision_embeds"] = (
            vision_size,
            self.language_model.config.hidden_size,
        )
        inputs_shapes["position_ids"] = (
            constants.ONNX_EXPORT_EXAMPLE_BATCH_SIZE,
            constants.ONNX_EXPORT_EXAMPLE_SEQ_LEN,
        )
        inputs_shapes["pixel_values"] = (
            constants.ONNX_EXPORT_EXAMPLE_BATCH_SIZE,
            3,
            height,
            width,
        )
        inputs_shapes["image_idx"] = (1, 1)
        inputs_shapes["image_sizes"] = (constants.ONNX_EXPORT_EXAMPLE_BATCH_SIZE, 2)
        # Define inputs
        vision_inputs = {}
        lang_inputs = {}
        vision_inputs["pixel_values"] = torch.zeros((inputs_shapes["pixel_values"]), dtype=torch.float32)
        lang_inputs["input_ids"] = torch.zeros((inputs_shapes["input_ids"]), dtype=torch.int64)
        lang_inputs["vision_embeds"] = torch.zeros((inputs_shapes["vision_embeds"]), dtype=torch.float32)
        lang_inputs["position_ids"] = (
            torch.arange(constants.ONNX_EXPORT_EXAMPLE_SEQ_LEN, dtype=torch.int64)
            .view(1, constants.ONNX_EXPORT_EXAMPLE_SEQ_LEN)
            .repeat(constants.ONNX_EXPORT_EXAMPLE_BATCH_SIZE, 1)
        )
        lang_inputs["image_idx"] = torch.zeros((inputs_shapes["image_idx"]), dtype=torch.int64)
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
        height = compiler_options.pop("height", None)
        width = compiler_options.pop("width", None)
        if height is None:
            height = self.config.vision_config.image_size
        if width is None:
            width = self.config.vision_config.image_size
        prefill_seq_len = prefill_seq_len if prefill_seq_len else 128
        ctx_len = ctx_len if ctx_len else constants.INTERN_CTX_LEN
        patch_size = self.config.vision_config.patch_size
        kernel_size = self.config.spatial_merge_size
        vision_size = ((height // patch_size) * (width // patch_size)) * (batch_size) // (kernel_size * kernel_size)

        vision = [
            {
                "batch_size": batch_size,
                "seq_len": prefill_seq_len,
                "ctx_len": ctx_len,
                "height": height,
                "width": width,
                "vision_size": vision_size,
            }
        ]
        lang = [
            {
                "batch_size": batch_size,
                "seq_len": prefill_seq_len,
                "ctx_len": ctx_len,
                "height": height,
                "width": width,
                "vision_size": vision_size,
            },
            {
                "batch_size": batch_size,
                "seq_len": "1",
                "ctx_len": ctx_len,
                "height": height,
                "width": width,
                "vision_size": vision_size,
            },
        ]

        specializations = {}

        if kv_offload:
            specializations["vision"] = vision
            specializations["lang"] = lang
            return specializations, compiler_options
        else:
            # return vision, compiler_options
            lang[0].pop("vision_size")
            lang[1].pop("vision_size")
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
            "vision_embeds": {0: "vision_size"},
        }

        for i in range(num_layers):
            lang_dynamic_axes[f"past_key.{i}"] = {0: "batch_size", 2: "ctx_len"}
            lang_dynamic_axes[f"past_value.{i}"] = {0: "batch_size", 2: "ctx_len"}

        dynamic_axes = {}
        if kv_offload:
            dynamic_axes["vision"] = vision_dynamic_axes
            dynamic_axes["lang"] = lang_dynamic_axes
        else:
            lang_dynamic_axes.pop("vision_embeds")
            dynamic_axes = {**vision_dynamic_axes, **lang_dynamic_axes}
            # dynamic_axes = vision_dynamic_axes
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
            lang_output_names.insert(2, "image_idx_output")
            output_names["vision"] = vision_output_names
            output_names["lang"] = lang_output_names
        else:
            lang_output_names.insert(1, "pixel_values_RetainedState")
            lang_output_names.insert(2, "image_idx_output")
            return lang_output_names
        return output_names

    def get_inputs_info(self):
        return [
            IOInfo(name="input_ids", datatype=torch.int64, shape=("batch_size", "seq_len")),
            IOInfo(name="attention_mask", datatype=torch.int64, shape=("batch_size", "seq_len")),
            IOInfo(name="pixel_values", datatype=torch.float32, shape=("batch_size", 3, "height", "width")),
        ]

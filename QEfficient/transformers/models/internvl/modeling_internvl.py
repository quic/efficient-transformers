# -----------------------------------------------------------------------------
#
# Copyright (c) 2025 Qualcomm Innovation Center, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------

import torch
import torch.nn as nn
import torch.nn.functional as F

from QEfficient.utils import constants
from QEfficient.utils._utils import IOInfo, get_padding_shape_from_config
from QEfficient.utils.logging_utils import logger


class QEffInternEncoderWrapper(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, pixel_values):
        vit_embeds = self.model.extract_feature(pixel_values)
        return vit_embeds


class QEffInternDecoderWrapper(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
        self.config = self.model.language_model.config

    def forward(self, input_ids, vit_embeds, position_ids, past_key_values):
        # TODO: Check if Hardcoding this is okay, i.e. check if this value is common for all intern models
        IMG_CONTEXT_TOKEN = 151667

        input_embeds = self.model.language_model.get_input_embeddings()(input_ids)
        B, N, C = input_embeds.shape
        image_input_embeds = input_embeds.reshape(B * N, C)
        image_input_ids = input_ids.reshape(B * N)
        selected = image_input_ids == IMG_CONTEXT_TOKEN
        indices1 = selected.unsqueeze(0).to(torch.int64).cumsum(1) - 1
        indices0 = torch.arange(selected.unsqueeze(0).shape[0]).view(-1, 1)
        image_features_expanded = vit_embeds.reshape(-1, C).unsqueeze(0)[indices0, indices1]
        image_input_embeds = torch.where(selected.unsqueeze(0).unsqueeze(-1), image_features_expanded, input_embeds)
        inputs_embeds = torch.where(input_ids.shape[1] == torch.tensor(1), input_embeds, image_input_embeds)
        outputs = self.model.language_model(
            inputs_embeds=inputs_embeds, position_ids=position_ids, past_key_values=past_key_values, use_cache=True
        )
        return outputs.logits, vit_embeds, outputs.past_key_values


class QEffInternVLModel(nn.Module):
    def get_qeff_vision_encoder(self):
        return QEffInternEncoderWrapper(self)

    def get_qeff_language_decoder(self):
        return QEffInternDecoderWrapper(self)

    def get_specializations(
        self,
        batch_size: int,
        prefill_seq_len: int,
        ctx_len: int,
        img_size: int,
        kv_offload: bool = False,
        **compiler_options,
    ):
        # TODO: check if this should be named num_patches or something else
        num_patches = compiler_options.pop("num_patches", None)
        if num_patches is None:
            logger.warning(
                "User should pass `num_patches` to compile API to fix the dynamic axes `pixel_values`, you can get more info by calling get_inputs_info function!, Since its not found setting its value to 13"
            )
            num_patches = 13

        prefill_seq_len = prefill_seq_len if prefill_seq_len else 3840  # 4096-256
        ctx_len = ctx_len if ctx_len else 4096
        if img_size is None and hasattr(self.config.vision_config, "image_size"):
            img_size = getattr(self.config.vision_config, "image_size")
        elif img_size is None:
            img_size = 448
            logger.warning("Setting img_size to be 448, as it was neither passed nor found in vision_config")

        vision = [
            {
                "batch_size": batch_size,
                "num_patches": num_patches,
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
                "num_patches": num_patches,
                "img_size": img_size,
            },
            {
                "batch_size": batch_size,
                "seq_len": "1",
                "ctx_len": ctx_len,
                "num_patches": num_patches,
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
        vision_dynamic_axes = {}
        lang_dynamic_axes = {}
        lang_dynamic_axes["input_ids"] = {0: "batch_size", 1: "seq_len"}
        lang_dynamic_axes["position_ids"] = {0: "batch_size", 1: "seq_len"}
        vision_dynamic_axes["pixel_values"] = {0: "num_patches", 2: "img_size", 3: "img_size"}

        pkv_dynamic_axes = {0: "batch_size", 2: "ctx_len"}
        for i in range(self.language_model.config.num_hidden_layers):
            for kv in ["key", "value"]:
                lang_dynamic_axes[f"past_{kv}.{i}"] = pkv_dynamic_axes

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

    def get_dummy_inputs(self, kv_offload: bool = False):
        num_patches = 13
        C = 3
        if vis_cfg := getattr(self.config, "vision_config", None):
            img_size = getattr(vis_cfg, "image_size", 448)
        else:
            img_size = 448

        # Taken from the modeling files of OpenGVLab/InternVL2_5-1B
        feature_size = int((((self.config.vision_config.hidden_size**0.5) * self.config.downsample_ratio) ** 2))

        # Define shapes
        inputs_shapes = {}
        inputs_shapes["input_ids"] = (constants.ONNX_EXPORT_EXAMPLE_BATCH_SIZE, constants.ONNX_EXPORT_EXAMPLE_SEQ_LEN)
        inputs_shapes["vit_embeds"] = (
            num_patches,
            feature_size,
            self.language_model.config.hidden_size,
        )
        inputs_shapes["position_ids"] = (
            constants.ONNX_EXPORT_EXAMPLE_BATCH_SIZE,
            constants.ONNX_EXPORT_EXAMPLE_SEQ_LEN,
        )
        inputs_shapes["pixel_values"] = (num_patches, C, img_size, img_size)

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

    def forward(self, input_ids, pixel_values, position_ids, past_key_values):
        # TODO: Check if Hardcoding this is okay, i.e. check if this value is common for all intern models
        IMG_CONTEXT_TOKEN = 151667

        input_embeds = self.language_model.get_input_embeddings()(input_ids)
        vit_embeds = self.extract_feature(pixel_values)
        B, N, C = input_embeds.shape
        image_input_embeds = input_embeds.reshape(B * N, C)
        image_input_ids = input_ids.reshape(B * N)
        selected = image_input_ids == IMG_CONTEXT_TOKEN
        indices1 = selected.unsqueeze(0).to(torch.int64).cumsum(1) - 1
        indices0 = torch.arange(selected.unsqueeze(0).shape[0]).view(-1, 1)
        image_features_expanded = vit_embeds.reshape(-1, C).unsqueeze(0)[indices0, indices1]
        image_input_embeds = torch.where(selected.unsqueeze(0).unsqueeze(-1), image_features_expanded, input_embeds)
        inputs_embeds = torch.where(input_ids.shape[1] == torch.tensor(1), input_embeds, image_input_embeds)
        outputs = self.language_model(
            inputs_embeds=inputs_embeds, position_ids=position_ids, past_key_values=past_key_values, use_cache=True
        )
        return outputs.logits, pixel_values, outputs.past_key_values

    def get_inputs_info(self):
        return [
            IOInfo(name="input_ids", datatype=torch.int64, shape=("batch_size", "seq_len")),
            IOInfo(name="attention_mask", datatype=torch.int64, shape=("batch_size", "seq_len")),
            IOInfo(name="pixel_values", datatype=torch.float32, shape=("num_patches", 3, "img_size", "img_size")),
        ]


class QEffInternVisionEmbeddings(nn.Module):
    def forward(self, pixel_values: torch.FloatTensor) -> torch.Tensor:
        target_dtype = self.patch_embedding.weight.dtype
        patch_embeds = self.patch_embedding(pixel_values)  # shape = [*, channel, width, height]
        batch_size, _, height, width = patch_embeds.shape
        patch_embeds = patch_embeds.flatten(2).transpose(1, 2)
        class_embeds = self.class_embedding.expand(batch_size, 1, -1).to(target_dtype)
        embeddings = torch.cat([class_embeds, patch_embeds], dim=1)

        pos_embed = self.position_embedding[:, 1:, :]
        target_dtype = pos_embed.dtype
        pos_embed = (
            pos_embed.float()
            .reshape(1, self.image_size // self.patch_size, self.image_size // self.patch_size, -1)
            .permute(0, 3, 1, 2)
        )
        pos_embed = (
            F.interpolate(pos_embed, size=(height, width), mode="bilinear", align_corners=False)
            .reshape(1, -1, height * width)
            .permute(0, 2, 1)
            .to(target_dtype)
        )

        position_embedding = torch.cat([self.position_embedding[:, :1, :], pos_embed], dim=1)

        embeddings = embeddings + position_embedding.to(target_dtype)
        return embeddings

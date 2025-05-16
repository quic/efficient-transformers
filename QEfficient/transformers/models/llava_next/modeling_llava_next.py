# -----------------------------------------------------------------------------
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------


import numpy as np
import torch
import torch.nn as nn
from transformers.models.llava_next.modeling_llava_next import (
    LlavaNextForConditionalGeneration,
    get_anyres_image_grid_shape,
)

from QEfficient.utils import constants
from QEfficient.utils._utils import IOInfo
from QEfficient.utils.logging_utils import logger


class QEffLlavaNextEncoderWrapper(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
        self.model.vision_model = self.model.vision_tower

    def forward(self, pixel_values, image_sizes):
        if pixel_values.dim() == constants.GRANITEVISION_PIXEL_VALUE_DIM:
            pixel_values_new = pixel_values.squeeze(0)

        image_feature = self.model.vision_tower(pixel_values_new, output_hidden_states=True)
        if isinstance(self.model.config.vision_feature_layer, int):
            selected_image_feature = image_feature.hidden_states[self.model.config.vision_feature_layer]
        else:
            hs_pool = [image_feature.hidden_states[layer_idx] for layer_idx in self.model.config.vision_feature_layer]
            selected_image_feature = torch.cat(hs_pool, dim=-1)

        vision_feature_select_strategy = self.model.config.vision_feature_select_strategy
        if vision_feature_select_strategy == "default":
            selected_image_feature = selected_image_feature[:, 1:]
        elif vision_feature_select_strategy == "full":
            selected_image_feature = selected_image_feature
        else:
            raise ValueError(f"Unexpected select feature strategy: {self.model.config.vision_feature_select_strategy}")
        image_features = self.model.multi_modal_projector(selected_image_feature)
        image_features = torch.split(image_features, [image_features.shape[0]], dim=0)
        new_image_features = []

        # Image feature
        for image_idx, image_feature in enumerate(image_features):
            if image_feature.shape[0] > 1:
                base_image_feature = image_feature[0]
                image_feature = image_feature[1:]
                height = width = (
                    self.model.config.vision_config.image_size // self.model.config.vision_config.patch_size
                )
                num_patch_height, num_patch_width = get_anyres_image_grid_shape(
                    image_sizes[image_idx],
                    self.model.config.image_grid_pinpoints,
                    self.model.config.vision_config.image_size,
                )

                if (
                    np.prod(image_feature.shape) % (num_patch_height * num_patch_width * height * width) != 0
                    and vision_feature_select_strategy == "default"
                ):
                    logger.warning_once(
                        "Image feature shape does not line up with the provided patch size. "
                        "You may be using the `default` vision_feature_select_strategy with a"
                        " visual encoder that does not have CLS."
                    )

                image_feature = image_feature.view(num_patch_height, num_patch_width, height, width, -1)
                image_feature = image_feature.permute(4, 0, 2, 1, 3).contiguous()
                image_feature = image_feature.flatten(1, 2).flatten(2, 3)

                if not isinstance(image_sizes[image_idx], (list, tuple)):
                    if not isinstance(image_sizes[image_idx], (torch.Tensor, np.ndarray)):
                        raise TypeError(
                            f"image_size invalid type: {type(image_sizes[image_idx])} not valid, should be either list, tuple, np.ndarray or tensor"
                        )
                original_size = image_sizes[image_idx].tolist()
                original_height, original_width = original_size
                current_height, current_width = image_feature.shape[1:]

                if torch.is_tensor(current_height):
                    current_height = current_height.item()
                    current_width = current_width.item()

                scale_factor = current_width / original_width
                new_height = int(round(original_height * scale_factor, 7))
                padding = (current_height - new_height) // 2
                image_feature = image_feature[:, padding : current_height - padding, :]
                if self.model.image_newline is not None:
                    image_feature = torch.cat(
                        (
                            image_feature,
                            self.model.image_newline[:, None, None]
                            .expand(*image_feature.shape[:-1], 1)
                            .to(image_feature.device, image_feature.dtype),
                        ),
                        dim=-1,
                    )
                image_feature = image_feature.flatten(1, 2).transpose(0, 1)
                image_feature = torch.cat((base_image_feature, image_feature), dim=0)
            else:
                image_feature = image_feature[0]
                if self.model.image_newline is not None:
                    image_feature = torch.cat((image_feature, self.model.image_newline[None].to(image_feature)), dim=0)
            new_image_features.append(image_feature)
        image_features = torch.cat(new_image_features, dim=0)
        return image_features


class QEffLlavaNextDecoderWrapper(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
        self.config = self.model.config
        self.language_model = self.model.language_model

    def forward(self, input_ids, image_features, position_ids, past_key_values):
        inputs_embeds = self.model.get_input_embeddings()(input_ids)
        image_features = image_features.to(inputs_embeds.device, inputs_embeds.dtype)
        mask = input_ids == self.config.image_token_index
        indices1 = mask.to(torch.int64).cumsum(1) - 1
        image_features_expanded = image_features[indices1]
        image_inputs_embeds = torch.where(mask.unsqueeze(-1), image_features_expanded, inputs_embeds)
        # *where to skip image encoder for decode*
        inputs_embeds = torch.where(input_ids.shape[1] == torch.tensor(1), inputs_embeds, image_inputs_embeds)
        outputs = self.language_model(
            inputs_embeds=inputs_embeds,
            position_ids=position_ids,
            past_key_values=past_key_values,
        )
        return outputs.logits, image_features, outputs.past_key_values


class QEffLlavaNextForConditionalGeneration(LlavaNextForConditionalGeneration):
    def get_qeff_vision_encoder(self):
        return QEffLlavaNextEncoderWrapper(self)

    def get_qeff_language_decoder(self):
        return QEffLlavaNextDecoderWrapper(self)

    def get_dummy_inputs(self, kv_offload: bool = False, **kwargs):
        num_layers = self.config.text_config.num_hidden_layers
        num_key_value_heads = self.config.text_config.num_key_value_heads
        head_dim = self.config.text_config.hidden_size // self.config.text_config.num_attention_heads
        if vis_cfg := getattr(self.config, "vision_config", None):
            img_size = getattr(vis_cfg, "image_size", constants.GRANITEVISION_IMG_SIZE)
        else:
            img_size = constants.GRANITEVISION_IMG_SIZE
        if img_size != constants.GRANITEVISION_IMG_SIZE and kv_offload:
            raise NotImplementedError("Image Size other than 384 is not supported for LlavaNext models yet.")
        vision_inputs = {
            "pixel_values": torch.zeros(
                (
                    constants.ONNX_EXPORT_EXAMPLE_BATCH_SIZE,
                    constants.GRANITEVISION_NUM_PATCHES,
                    constants.GRANITEVISION_NUM_CHANNELS,
                    constants.GRANITEVISION_IMG_SIZE,
                    constants.GRANITEVISION_IMG_SIZE,
                ),
                dtype=torch.float32,
            ),
            "image_sizes": torch.tensor(
                [[constants.GRANITEVISION_IMG_SIZE_HEIGHT, constants.GRANITEVISION_IMG_SIZE_WIDTH]], dtype=torch.int64
            ),
        }
        lang_inputs = {
            "input_ids": torch.ones(
                (constants.ONNX_EXPORT_EXAMPLE_BATCH_SIZE, constants.GRANITEVISION_SEQ_LEN), dtype=torch.int64
            ),
            "attention_mask": torch.ones(
                (constants.ONNX_EXPORT_EXAMPLE_BATCH_SIZE, constants.GRANITEVISION_SEQ_LEN), dtype=torch.int64
            ),
            "image_features": torch.ones(
                (constants.GRANITEVISION_FEATURE_SIZE, self.language_model.config.hidden_size), dtype=torch.float32
            ),
        }
        lang_inputs["position_ids"] = lang_inputs.pop("attention_mask").cumsum(1)
        lang_inputs["past_key_values"] = []
        for i in range(num_layers):
            lang_inputs["past_key_values"].append(
                (
                    torch.zeros(
                        constants.ONNX_EXPORT_EXAMPLE_BATCH_SIZE,
                        num_key_value_heads,
                        constants.GRANITEVISION_CTX_LEN,
                        head_dim,
                    ),
                    torch.zeros(
                        constants.ONNX_EXPORT_EXAMPLE_BATCH_SIZE,
                        num_key_value_heads,
                        constants.GRANITEVISION_CTX_LEN,
                        head_dim,
                    ),
                )
            )
        lang_inputs["position_ids"] = torch.full(lang_inputs["position_ids"].shape, constants.GRANITEVISION_CTX_LEN - 1)
        inputs = {}
        if kv_offload:
            inputs["vision"] = vision_inputs
            inputs["lang"] = lang_inputs
        else:
            lang_inputs.pop("image_features")
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
        num_patches = compiler_options.pop("num_patches", None)
        image_size_height = compiler_options.pop("image_size_height", None)
        image_size_width = compiler_options.pop("image_size_width", None)

        if num_patches is None:
            num_patches = constants.GRANITEVISION_NUM_PATCHES
        if image_size_height is None:
            image_size_height = constants.GRANITEVISION_IMG_SIZE_HEIGHT
        if image_size_width is None:
            image_size_width = constants.GRANITEVISION_IMG_SIZE_WIDTH

        if num_patches != constants.GRANITEVISION_NUM_PATCHES:
            logger.warning("Image Num Patches should be set to 10")
            num_patches = constants.GRANITEVISION_NUM_PATCHES

        if image_size_height != constants.GRANITEVISION_IMG_SIZE_HEIGHT:
            logger.warning(
                "Image Size Height Should be fixed to 1109. Please Reshape the image to (w x h) (1610 x 1109)"
            )
            image_size_height = constants.GRANITEVISION_IMG_SIZE_HEIGHT

        if image_size_width != constants.GRANITEVISION_IMG_SIZE_WIDTH:
            logger.warning(
                "Image Size Width Should be fixed to 1610. Please Reshape the image to (w x h) (1610 x 1109)"
            )
            image_size_width = constants.GRANITEVISION_IMG_SIZE_WIDTH

        prefill_seq_len = prefill_seq_len if prefill_seq_len else constants.GRANITEVISION_SEQ_LEN
        ctx_len = ctx_len if ctx_len else constants.GRANITEVISION_CTX_LEN
        if not kv_offload:
            raise NotImplementedError("We currently support on Dual QPC for this model please set kv_offload to True")
        if img_size is None and hasattr(self.config.vision_config, "image_size"):
            img_size = getattr(self.config.vision_config, "image_size")
        elif img_size is None:
            img_size = constants.GRANITEVISION_IMG_SIZE
            logger.warning("Setting img_size to be 384, as it was neither passed nor found in vision_config")
        if img_size != constants.GRANITEVISION_IMG_SIZE and kv_offload:
            logger.warning("Image Size other than 384 is not supported for LlavaNext models yet.")
        vision = [
            {
                "batch_size": batch_size,
                "seq_len": prefill_seq_len,
                "ctx_len": ctx_len,
                "image_size_height": image_size_height,
                "image_size_width": image_size_width,
                "num_patches": num_patches,
                "max_num_images": max_num_images,
                "img_size": img_size,
            }
        ]
        lang = [
            {
                "batch_size": batch_size,
                "seq_len": prefill_seq_len,
                "ctx_len": ctx_len,
                "image_size_height": image_size_height,
                "image_size_width": image_size_width,
                "num_patches": num_patches,
                "max_num_images": max_num_images,
                "img_size": img_size,
            },
            {
                "batch_size": batch_size,
                "seq_len": "1",
                "ctx_len": ctx_len,
                "image_size_height": image_size_height,
                "image_size_width": image_size_width,
                "num_patches": num_patches,
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
            "pixel_values": {0: "batch_size", 1: "num_patches", 3: "img_size", 4: "img_size"},
            "image_sizes": {0: "image_size_height", 1: "image_size_width"},
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
        vision_output_names = ["image_features"]
        lang_output_names = ["logits"]
        for i in range(self.language_model.config.num_hidden_layers):
            for kv in ["key", "value"]:
                lang_output_names.append(f"past_{kv}.{i}_RetainedState")

        output_names = {}
        if kv_offload:
            lang_output_names.insert(1, "image_features_RetainedState")
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
            IOInfo(name="pixel_values", datatype=torch.float32, shape=("batch_size", 10, 3, "img_size", "img_size")),
            IOInfo(name="image_sizes", datatype=torch.int64, shape=(1109, 1610)),
        ]

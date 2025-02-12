# -----------------------------------------------------------------------------
#
# Copyright (c) 2024 Qualcomm Innovation Center, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------

import torch
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


class QEffLlavaForConditionalGeneration(LlavaForConditionalGeneration):
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
        image_features = self.multi_modal_projector(selected_image_feature)
        image_features = image_features.to(inputs_embeds.device, inputs_embeds.dtype)

        mask = input_ids == self.config.image_token_index
        indices1 = mask.to(torch.int64).cumsum(1) - 1
        indices0 = torch.arange(mask.shape[0]).view(-1, 1)
        image_features_expanded = image_features[indices0, indices1]
        image_inputs_embeds = torch.where(mask.unsqueeze(-1), image_features_expanded, inputs_embeds)
        # *where to skip image encoder for decode*
        inputs_embeds = torch.where(input_ids.shape[1] == torch.tensor(1), inputs_embeds, image_inputs_embeds)
        outputs = self.language_model(
            inputs_embeds=inputs_embeds,
            position_ids=position_ids,
            past_key_values=past_key_values,
        )
        return outputs.logits, pixel_values, outputs.past_key_values

    def get_dummy_inputs(self, **kwargs):
        num_layers = self.config.text_config.num_hidden_layers
        num_key_value_heads = self.config.text_config.num_key_value_heads
        head_dim = self.config.text_config.hidden_size // self.config.text_config.num_attention_heads
        if vis_cfg := getattr(self.config, "vision_config", None):
            img_size = getattr(vis_cfg, "image_size", 336)
        else:
            img_size = 336
        inputs = {
            "input_ids": torch.ones((BS, SEQ_LEN), dtype=torch.int64),
            "attention_mask": torch.ones((BS, SEQ_LEN), dtype=torch.int64),
            "pixel_values": torch.zeros((BS, NUM_CHANNEL, img_size, img_size), dtype=torch.float32),
        }
        inputs["position_ids"] = inputs.pop("attention_mask").cumsum(1)
        inputs["past_key_values"] = []
        for i in range(num_layers):
            inputs["past_key_values"].append(
                (
                    torch.zeros(BS, num_key_value_heads, CTX_LEN, head_dim),
                    torch.zeros(BS, num_key_value_heads, CTX_LEN, head_dim),
                )
            )
        inputs["position_ids"] = torch.full(inputs["position_ids"].shape, CTX_LEN - 1)
        return inputs

    def get_specializations(
        self, batch_size: int, prefill_seq_len: int, ctx_len: int, img_size: int, **compiler_options
    ):
        max_num_images = compiler_options.get("max_num_images", 1)
        prefill_seq_len = prefill_seq_len if prefill_seq_len else SEQ_LEN
        ctx_len = ctx_len if ctx_len else CTX_LEN
        if img_size is None and hasattr(self.config.vision_config, "image_size"):
            img_size = getattr(self.config.vision_config, "image_size")
        elif img_size is None:
            img_size = 336
            logger.warning("Setting img_size to be 336, as it was neither passed nor found in vision_config")

        return [
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

    def get_onnx_dynamic_axes(
        self,
    ):
        # Define dynamic axes
        num_layers = self.config.text_config.num_hidden_layers

        dynamic_axes = {
            "input_ids": {0: "batch_size", 1: "seq_len"},
            "position_ids": {0: "batch_size", 1: "seq_len"},
            "pixel_values": {0: "batch_size", 2: "img_size", 3: "img_size"},
        }
        for i in range(num_layers):
            dynamic_axes[f"past_key.{i}"] = {0: "batch_size", 2: "ctx_len"}
            dynamic_axes[f"past_value.{i}"] = {0: "batch_size", 2: "ctx_len"}

        return dynamic_axes

    def get_output_names(
        self,
    ):
        output_names = ["logits", "pixel_values_RetainedState"]
        for i in range(self.language_model.config.num_hidden_layers):
            for kv in ["key", "value"]:
                output_names.append(f"past_{kv}.{i}_RetainedState")
        return output_names

    def get_inputs_info(self):
        return [
            IOInfo(name="input_ids", datatype=torch.int64, shape=("batch_size", "seq_len")),
            IOInfo(name="attention_mask", datatype=torch.int64, shape=("batch_size", "seq_len")),
            IOInfo(name="pixel_values", datatype=torch.float32, shape=("batch_size", 3, "img_size", "img_size")),
        ]

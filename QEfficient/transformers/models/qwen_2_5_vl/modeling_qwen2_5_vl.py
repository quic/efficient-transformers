import torch
import torch.nn as nn
from transformers import Qwen2_5_VLForConditionalGeneration

from QEfficient.utils import constants


class QEffQwen_2_5_vl_EncoderWrapper(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
        self.model.vision_model = self.model.visual

    def forward(self, pixel_values, image_grid_thw):
        pixel_values = pixel_values.type(self.model.visual.dtype)
        image_embeds = self.model.visual(pixel_values, grid_thw=image_grid_thw)
        return image_embeds


class QEffQwen_2_5_vl_DecoderWrapper(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
        self.config = self.model.config
        self.language_model = self.model.model

    def forward(self, input_ids, vision_embeds, position_ids, image_idx, past_key_values):
        breakpoint()
        pass


class QEffQwen_2_5_vl_ForConditionalGeneration(Qwen2_5_VLForConditionalGeneration):
    def get_qeff_vision_encoder(self):
        return QEffQwen_2_5_vl_EncoderWrapper(self)

    def get_qeff_language_decoder(self):
        return QEffQwen_2_5_vl_DecoderWrapper(self)

    def get_dummy_inputs(self, kv_offload: bool = False, **kwargs):
        num_layers = self.config.num_hidden_layers
        num_key_value_heads = self.config.num_key_value_heads
        head_dim = self.config.hidden_size // self.config.num_attention_heads
        vision_inputs = {
            "pixel_values": torch.zeros(
                (11016, 1176),
                dtype=torch.float32,
            ),
            "image_grid_thw": torch.tensor([[1, 108, 102]]),
        }

        lang_inputs = {
            "input_ids": torch.ones((1, 2779), dtype=torch.int64),
            "attention_mask": torch.ones((1, 2779), dtype=torch.int64),
            "vision_embeds": torch.ones(
                (11016, 1176),
                dtype=torch.float32,
            ),
            "image_idx": torch.zeros((1, 1), dtype=torch.int64),
        }
        lang_inputs["position_ids"] = lang_inputs.pop("attention_mask").cumsum(1)
        lang_inputs["past_key_values"] = []
        for i in range(num_layers):
            lang_inputs["past_key_values"].append(
                (
                    torch.zeros(
                        1,
                        num_key_value_heads,
                        6000,
                        head_dim,
                    ),
                    torch.zeros(
                        1,
                        num_key_value_heads,
                        6000,
                        head_dim,
                    ),
                )
            )

        lang_inputs["position_ids"] = torch.full(lang_inputs["position_ids"].shape, constants.GRANITEVISION_CTX_LEN - 1)
        inputs = {}
        if kv_offload:
            inputs["vision"] = vision_inputs
            inputs["lang"] = lang_inputs
        breakpoint()
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
        pass

    def get_onnx_dynamic_axes(self, kv_offload: bool = False):
        # Define dynamic axes
        num_layers = self.config.num_hidden_layers
        vision_dynamic_axes = {
            "pixel_values": {0: "batch_size", 1: "num_patches"},
            "image_grid_thw": {0: "batch_size", 1: "batch_size"},
        }
        lang_dynamic_axes = {
            "input_ids": {0: "batch_size", 1: "seq_len"},
            "position_ids": {0: "batch_size", 1: "seq_len", 2: "seq_len"},
            "vision_embeds": {0: "batch_size", 1: "vision_size"},
        }
        for i in range(num_layers):
            lang_dynamic_axes[f"past_key.{i}"] = {0: "batch_size", 2: "ctx_len"}
            lang_dynamic_axes[f"past_value.{i}"] = {0: "batch_size", 2: "ctx_len"}
        dynamic_axes = {}
        if kv_offload:
            dynamic_axes["vision"] = vision_dynamic_axes
            dynamic_axes["lang"] = lang_dynamic_axes
        return dynamic_axes

    def get_inputs_info(self):
        pass

    def get_output_names(self, kv_offload: bool = False):
        vision_output_names = ["vision_embeds"]
        lang_output_names = ["logits"]
        # breakpoint()
        for i in range(64):
            for kv in ["key", "value"]:
                lang_output_names.append(f"past_{kv}.{i}_RetainedState")

        output_names = {}
        if kv_offload:
            lang_output_names.insert(1, "vision_embeds_RetainedState")
            lang_output_names.insert(2, "image_idx_output")
            output_names["vision"] = vision_output_names
            output_names["lang"] = lang_output_names

        return output_names

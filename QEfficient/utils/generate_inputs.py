# -----------------------------------------------------------------------------
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------
from typing import List

import numpy as np
import torch

from QEfficient.transformers.modeling_utils import DYNAMIC_SEQ_LEN_SUPPORTED_MODEL_ARCH
from QEfficient.utils import (
    get_num_layers_from_config,
    get_padding_shape_from_config,
    get_sliding_window_layers,
    get_sliding_window_shapes,
    padding_check_and_fix,
)


class InputHandler:
    def __init__(self, batch_size, tokenizer, config, prompt, prompt_len, ctx_len, full_batch_size):
        """
        Initialization

        ``Mandatory`` Args:
            :batch_size (int): Number of prompts to run in one batch.
            :tokenizer (Union[PreTrainedTokenizer, PreTrainedTokenizerFast]): Pass model tokenizer.
            :config (AutoConfig): From pretrained model.
            :prompt (List[str]): String to used as input prompt for the model.
            :prompt_len (int): Prompt length for the model to compile.
            :ctx_len (int): Maximum context length to compile the model.
            :full_batch_size (int): Continuous batching batch size
        """
        # check and fix tokenizer viability
        padding_check_and_fix(tokenizer)
        self.tokenizer = tokenizer
        self.prompt = prompt
        self.prompt_len = prompt_len
        self.ctx_len = ctx_len
        self.full_batch_size = full_batch_size
        self.config = config
        self.n_layer = get_num_layers_from_config(config)
        self.padding_shape = get_padding_shape_from_config(
            config=config, batch_size=full_batch_size if full_batch_size else batch_size, seq_len=ctx_len
        )

        self.is_chunked_attention = get_sliding_window_layers(config)
        self.global_shape, self.sliding_shape = get_sliding_window_shapes(
            config=config, batch_size=full_batch_size if full_batch_size else batch_size, seq_len=ctx_len
        )

    def prepare_pytorch_inputs(self):
        """
        Function responsible for creating Prefill stage tensor inputs for PyTorch model.

        Return:
            :Dict: input_ids, position_ids, past_key_values
        """

        inputs = self.tokenizer(
            self.prompt,
            return_tensors="pt",
            padding=True,
        )
        input_ids = inputs["input_ids"]
        batch_size, input_len = input_ids.shape
        inputs.pop("attention_mask")
        inputs.pop("token_type_ids", None)
        usable_bs = self.full_batch_size if self.full_batch_size else 1
        position_ids = torch.arange(input_len).view(1, input_len).repeat(usable_bs, 1)
        inputs["input_ids"] = torch.concat(
            [
                input_ids,
                torch.ones((batch_size, self.prompt_len - input_len), dtype=torch.int64)
                * (self.tokenizer.pad_token_id),
            ],
            1,
        )
        inputs["position_ids"] = torch.concat(
            [
                position_ids,
                torch.ones((batch_size, self.prompt_len - input_len), dtype=torch.int64) * (-1),
            ],
            1,
        )

        if self.full_batch_size:
            inputs["input_ids"] = input_ids
            inputs["position_ids"] = position_ids
            inputs["batch_index"] = torch.arange(self.full_batch_size).view(-1, 1)

        past_key_values = []
        for i in range(self.n_layer):
            if (
                all(hasattr(self.config, attr) for attr in ["sliding_window", "layer_types"])
                and self.config.layer_types[i] == "sliding_attention"
            ):
                pad_shape = self.padding_shape[:2] + [self.config.sliding_window] + [self.padding_shape[-1]]
            else:
                pad_shape = self.padding_shape
            past_key = torch.zeros((pad_shape), dtype=torch.float32)
            past_value = torch.zeros((pad_shape), dtype=torch.float32)
            pkv = (past_key, past_value)
            past_key_values.append(pkv)
        inputs["past_key_values"] = tuple(past_key_values)

        return inputs

    def update_pytorch_inputs(self, inputs, pt_outputs):
        """
        Function responsible for updating Prefill stage inputs to create decode stage inputs for PyTorch model.

        ``Mandatory`` Args:
            :inputs (Dict): Pytorch inputs from previous iteration
            :pt_outputs (Dict): Pytorch outputs from previous iteration

        Return:
            :Dict: Updated input_ids, position_ids and past_key_values
        """
        updated_inputs = {}
        if self.full_batch_size:
            input_ids = pt_outputs.logits.detach().argmax(2)
            updated_inputs["input_ids"] = torch.full((self.full_batch_size, 1), self.tokenizer.pad_token_id)
            updated_inputs["input_ids"][inputs["batch_index"].view(-1)] = input_ids

            position_ids = inputs["position_ids"].max(1, keepdim=True).values + 1
            updated_inputs["position_ids"] = torch.full((self.full_batch_size, 1), 0)
            updated_inputs["position_ids"][inputs["batch_index"].view(-1)] = position_ids

            updated_inputs["batch_index"] = inputs["batch_index"]
        else:
            updated_inputs["input_ids"] = pt_outputs["logits"].argmax(-1).reshape(-1, 1)
            updated_inputs["position_ids"] = inputs["position_ids"].max(1, keepdim=True).values + 1

        updated_inputs["past_key_values"] = tuple(
            [(key.detach(), value.detach()) for key, value in pt_outputs["past_key_values"]]
        )

        return updated_inputs

    def prepare_ort_inputs(self):
        """
        Function responsible for creating Prefill stage numpy inputs for ONNX model to be run on ONNXRT.

        Return:
            :Dict: input_ids, position_ids, past_key_values
        """

        inputs = self.tokenizer(
            self.prompt,
            return_tensors="np",
            padding=True,
        )
        input_ids = inputs["input_ids"]
        batch_size, input_len = input_ids.shape
        inputs.pop("attention_mask")
        inputs.pop("token_type_ids", None)
        position_ids = np.arange(input_len).reshape(1, -1)
        inputs["input_ids"] = np.concatenate(
            [input_ids, np.full((batch_size, self.prompt_len - input_len), self.tokenizer.pad_token_id)],
            axis=1,
        ).astype(np.int64)
        inputs["position_ids"] = np.concatenate(
            [position_ids, np.full((batch_size, self.prompt_len - input_len), -1)],
            axis=1,
        ).astype(np.int64)

        if hasattr(self.config, "model_type") and self.config.model_type in DYNAMIC_SEQ_LEN_SUPPORTED_MODEL_ARCH:
            for i in range(self.n_layer):
                cache_shape = self.global_shape if not self.is_chunked_attention[i] else self.sliding_shape
                inputs["past_key." + str(i)] = np.zeros((cache_shape), dtype=np.float32)
                inputs["past_value." + str(i)] = np.zeros((cache_shape), dtype=np.float32)
        else:
            for i in range(self.n_layer):
                if (
                    all(hasattr(self.config, attr) for attr in ["sliding_window", "layer_types"])
                    and self.config.layer_types[i] == "sliding_attention"
                ):
                    pad_shape = self.padding_shape[:2] + [self.config.sliding_window] + [self.padding_shape[-1]]
                else:
                    pad_shape = self.padding_shape
                inputs["past_key." + str(i)] = np.zeros((pad_shape), dtype=np.float32)
                inputs["past_value." + str(i)] = np.zeros((pad_shape), dtype=np.float32)
        if self.full_batch_size:
            inputs["batch_index"] = np.arange(self.full_batch_size).reshape(-1, 1)
        return inputs

    def update_ort_inputs(self, inputs, ort_outputs):
        """
        Function responsible for updating Prefill stage inputs to create inputs for decode stage inputs for ONNX model to be run on ONNXRT.

        ``Mandatory`` Args:
            :inputs (Dict): NumPy inputs of Onnx model from previous iteration
            :ort_outputs (Dict): Numpy outputs of Onnx model from previous iteration

        Return:
            :Dict: Updated input_ids, position_ids and past_key_values
        """

        updated_inputs = {}
        updated_inputs["input_ids"] = ort_outputs["logits"].argmax(-1)
        updated_inputs["position_ids"] = np.max(inputs["position_ids"], axis=1, keepdims=True) + 1
        for i in range(self.n_layer):
            updated_inputs["past_key." + str(i)] = ort_outputs["past_key_values"][i * 2]
            updated_inputs["past_value." + str(i)] = ort_outputs["past_key_values"][i * 2 + 1]
        if self.full_batch_size:
            updated_inputs["batch_index"] = inputs["batch_index"]
        return updated_inputs

    def update_ort_outputs(self, ort_outputs):
        """
        Function responsible for updating ONNXRT session outputs.

        ``Mandatory`` Args:
            :ort_outputs (Dict): Numpy outputs of Onnx model from current iteration

        Return:
            updated_outputs (Dict): Updated past_key_values, logits
        """

        present_key_values = []
        for i in range(self.n_layer):
            if "past_key." + str(i) + "_RetainedState" in ort_outputs:
                present_key_values.append(ort_outputs["past_key." + str(i) + "_RetainedState"])
            if "past_value." + str(i) + "_RetainedState" in ort_outputs:
                present_key_values.append(ort_outputs["past_value." + str(i) + "_RetainedState"])

        outputs = {}
        outputs["past_key_values"] = present_key_values
        outputs["logits"] = ort_outputs["logits"]

        return outputs


class InputHandlerVLM:
    def __init__(
        self, batch_size, config, image, conversation, processor, prompt, prompt_len, ctx_len, max_gen_len, n_layer
    ):
        self.ctx_len = ctx_len
        self.prompt_len = prompt_len
        self.max_gen_len = max_gen_len
        self.config = config
        self.image = image
        self.prompt = prompt
        self.batch_size = batch_size
        self.n_layer = n_layer
        self.processor = processor
        self.conversation = conversation

    def prepare_pytorch_inputs(self):
        """
        Function responsible for creating Prefill stage tensor inputs for PyTorch model.

        Return:
            :Dict: input_ids, position_ids, past_key_values
        """
        inputs = self.processor(images=self.image, text=self.prompt, return_tensors="pt")
        if hasattr(self.config, "text_config"):
            txt_cfg = self.config.text_config
        else:
            txt_cfg = self.config.llm_config

        num_hidden_layers = txt_cfg.num_hidden_layers
        num_key_value_heads = txt_cfg.num_key_value_heads
        head_dim = getattr(txt_cfg, "head_dim", txt_cfg.hidden_size // txt_cfg.num_attention_heads)
        if hasattr(txt_cfg, "cross_attention_layers"):
            cross_attention_layers = txt_cfg.cross_attention_layers

            vis_cfg = self.config.vision_config
            num_patches = (vis_cfg.image_size // vis_cfg.patch_size) ** 2 + 1
            image_tokens_len = vis_cfg.max_num_tiles * num_patches

        inputs["position_ids"] = inputs.pop("attention_mask").cumsum(1) - 1
        inputs["past_key_values"] = []
        for i in range(num_hidden_layers):
            # Specific to mllama as of now
            if hasattr(txt_cfg, "cross_attention_layers") and i in cross_attention_layers:
                idx = cross_attention_layers.index(i)
                assert idx == ((i - 3) // 5), f"{i}, {(i - 3) // 5}"
                inputs["past_key_values"].append(
                    (
                        torch.zeros(1, num_key_value_heads, image_tokens_len, head_dim),
                        torch.zeros(1, num_key_value_heads, image_tokens_len, head_dim),
                    )
                )
            else:
                inputs["past_key_values"].append(
                    (
                        torch.zeros(1, num_key_value_heads, self.ctx_len, head_dim),
                        torch.zeros(1, num_key_value_heads, self.ctx_len, head_dim),
                    )
                )

        return inputs

    def prepare_vlm_ort_inputs(self):
        if hasattr(self.config, "text_config"):
            txt_cfg = self.config.text_config
        else:
            txt_cfg = self.config.llm_config
        num_hidden_layers = txt_cfg.num_hidden_layers
        num_key_value_heads = txt_cfg.num_key_value_heads
        head_dim = getattr(txt_cfg, "head_dim", txt_cfg.hidden_size // txt_cfg.num_attention_heads)
        if hasattr(txt_cfg, "cross_attention_layers"):
            cross_attention_layers = txt_cfg.cross_attention_layers
            vis_cfg = self.config.vision_config
            num_patches = (vis_cfg.image_size // vis_cfg.patch_size) ** 2 + 1
            image_tokens_len = vis_cfg.max_num_tiles * num_patches

        inputs = self.processor(images=self.image, text=self.prompt, return_tensors="np")
        if "attention_mask" in inputs.keys():
            inputs["position_ids"] = inputs.pop("attention_mask").cumsum(1) - 1
        inputs["past_key_values"] = []
        inputs["image_idx"] = np.array([[0]])

        vision_inputs = {
            k: v for k, v in inputs.items() if k in {"pixel_values", "aspect_ratio_ids", "aspect_ratio_mask"}
        }

        for i in range(num_hidden_layers):
            if hasattr(txt_cfg, "cross_attention_layers") and i in cross_attention_layers:
                idx = cross_attention_layers.index(i)
                assert idx == ((i - 3) // 5), f"{i}, {(i - 3) // 5}"
                inputs["past_key." + str(i)] = np.zeros(
                    (self.batch_size, num_key_value_heads, image_tokens_len, head_dim), dtype=np.float32
                )
                inputs["past_value." + str(i)] = np.zeros(
                    (self.batch_size, num_key_value_heads, image_tokens_len, head_dim), dtype=np.float32
                )
            else:
                inputs["past_key." + str(i)] = np.zeros(
                    (self.batch_size, num_key_value_heads, self.ctx_len, head_dim), dtype=np.float32
                )
                inputs["past_value." + str(i)] = np.zeros(
                    (self.batch_size, num_key_value_heads, self.ctx_len, head_dim), dtype=np.float32
                )
        lang_inputs = {k: v for k, v in inputs.items() if k not in vision_inputs}
        return vision_inputs, lang_inputs

    def update_vlm_ort_outputs(self, ort_outputs):
        """
        Function responsible for updating ONNXRT session outputs.

        ``Mandatory`` Args:
            :ort_outputs (Dict): Numpy outputs of Onnx model from current iteration

        Return:
            updated_outputs (Dict): Updated past_key_values, logits, pixel_values
        """
        present_key_values = []
        for i in range(self.n_layer[0]):
            if "past_key." + str(i) + "_RetainedState" in ort_outputs:
                present_key_values.append(ort_outputs["past_key." + str(i) + "_RetainedState"])
            if "past_value." + str(i) + "_RetainedState" in ort_outputs:
                present_key_values.append(ort_outputs["past_value." + str(i) + "_RetainedState"])

        outputs = {}
        outputs["past_key_values"] = present_key_values
        outputs["logits"] = ort_outputs["logits"]
        outputs["pixel_values_RetainedState"] = (
            ort_outputs["pixel_values_RetainedState"] if "pixel_values_RetainedState" in ort_outputs else None
        )
        outputs["image_features_RetainedState"] = (
            ort_outputs["image_features_RetainedState"] if "image_features_RetainedState" in ort_outputs else None
        )
        outputs["image_idx"] = ort_outputs["image_idx_output"]
        return outputs

    def update_vlm_ort_inputs(self, inputs, ort_outputs):
        """
        Function responsible for updating Prefill stage inputs to create inputs for decode stage inputs for ONNX model to be run on ONNXRT.

        ``Mandatory`` Args:
            :inputs (Dict): NumPy inputs of Onnx model from previous iteration
            :ort_outputs (Dict): Numpy outputs of Onnx model from previous iteration

        Return:
            :Dict: Updated input_ids, position_ids, pixel_values and past_key_values
        """
        updated_inputs = {}
        updated_inputs["input_ids"] = ort_outputs["logits"].argmax(-1)
        updated_inputs["position_ids"] = np.max(inputs["position_ids"], axis=1, keepdims=True) + 1
        for i in range(self.n_layer[0]):
            updated_inputs["past_key." + str(i)] = ort_outputs["past_key_values"][i * 2]
            updated_inputs["past_value." + str(i)] = ort_outputs["past_key_values"][i * 2 + 1]
        if "pixel_values_RetainedState" in ort_outputs.keys():
            updated_inputs["pixel_values"] = ort_outputs["pixel_values_RetainedState"]
        if "image_features_RetainedState" in ort_outputs.keys():
            updated_inputs["image_features"] = ort_outputs["image_features_RetainedState"]

        if "cross_attention_mask" in inputs.keys():
            bs, _, num_images, img_tiles = inputs["cross_attention_mask"].shape
            updated_inputs["cross_attention_mask"] = torch.ones(
                (bs, 1, num_images, img_tiles), dtype=torch.int64
            ).numpy()

        for k, v in inputs.items():
            if k not in updated_inputs.keys():
                updated_inputs[k] = v
        return updated_inputs


class InputHandlerInternVL(InputHandlerVLM):
    def __init__(self, batch_size, config, image, processor, prompt, prompt_len, ctx_len, max_gen_len, n_layer):
        self.ctx_len = ctx_len
        self.prompt_len = prompt_len
        self.max_gen_len = max_gen_len
        self.config = config
        self.image = image
        self.prompt = prompt
        self.batch_size = batch_size
        self.n_layer = n_layer
        self.processor = processor

    def prepare_pytorch_inputs(self):
        question = "<image>\n" + self.prompt
        pixel_values = self.processor.load_image(self.image, max_num=12)
        # Chat Template information for prompt preprocessing
        messages: List[List[str]] = []
        roles = ("<|im_start|>user\n", "<|im_start|>assistant\n")
        prompt = self.processor(pixel_values, question, messages, roles)
        inputs = self.processor.tokenizer(prompt, return_tensors="pt")
        inputs["pixel_values"] = pixel_values.clone()

        if hasattr(self.config, "text_config"):
            txt_cfg = self.config.text_config
        else:
            txt_cfg = self.config.llm_config

        num_hidden_layers = txt_cfg.num_hidden_layers
        num_key_value_heads = txt_cfg.num_key_value_heads
        head_dim = getattr(txt_cfg, "head_dim", txt_cfg.hidden_size // txt_cfg.num_attention_heads)

        inputs["position_ids"] = inputs.pop("attention_mask").cumsum(1) - 1
        inputs["past_key_values"] = []
        for i in range(num_hidden_layers):
            inputs["past_key_values"].append(
                (
                    torch.zeros(1, num_key_value_heads, self.ctx_len, head_dim),
                    torch.zeros(1, num_key_value_heads, self.ctx_len, head_dim),
                )
            )

        return inputs

    def prepare_vlm_ort_inputs(self):
        if hasattr(self.config, "text_config"):
            txt_cfg = self.config.text_config
        else:
            txt_cfg = self.config.llm_config
        num_hidden_layers = txt_cfg.num_hidden_layers
        num_key_value_heads = txt_cfg.num_key_value_heads
        head_dim = getattr(txt_cfg, "head_dim", txt_cfg.hidden_size // txt_cfg.num_attention_heads)

        question = "<image>\n" + self.prompt
        pixel_values = self.processor.load_image(self.image, max_num=12)
        # Chat Template information for prompt preprocessing
        messages: List[List[str]] = []
        roles = ("<|im_start|>user\n", "<|im_start|>assistant\n")
        prompt = self.processor(pixel_values, question, messages, roles)
        inputs = self.processor.tokenizer(prompt, return_tensors="np")
        inputs["pixel_values"] = pixel_values.numpy()

        if "attention_mask" in inputs.keys():
            inputs["position_ids"] = inputs.pop("attention_mask").cumsum(1) - 1
        inputs["past_key_values"] = []
        inputs["image_idx"] = np.array([[0]])

        vision_inputs = {
            k: v for k, v in inputs.items() if k in {"pixel_values", "aspect_ratio_ids", "aspect_ratio_mask"}
        }

        for i in range(num_hidden_layers):
            inputs["past_key." + str(i)] = np.zeros(
                (self.batch_size, num_key_value_heads, self.ctx_len, head_dim), dtype=np.float32
            )
            inputs["past_value." + str(i)] = np.zeros(
                (self.batch_size, num_key_value_heads, self.ctx_len, head_dim), dtype=np.float32
            )
        lang_inputs = {k: v for k, v in inputs.items() if k not in vision_inputs}
        return vision_inputs, lang_inputs

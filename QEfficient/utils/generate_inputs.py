# -----------------------------------------------------------------------------
#
# Copyright (c)  2024 Qualcomm Innovation Center, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------

import numpy as np
import torch

from QEfficient.utils import padding_check_and_fix


class InputHandler:
    def __init__(self, tokenizer, input_str, prompt_len, ctx_len):
        """
        Initialization
        :param model_name: str
        :param input_str: List[str]
        :param prompt_len: int
        :param ctx_len: int
        """
        # check and fix tokenizer viability
        padding_check_and_fix(tokenizer)
        self.tokenizer = tokenizer
        self.input_str = input_str
        self.prompt_len = prompt_len
        self.ctx_len = ctx_len

    def prepare_pytorch_inputs(self, n_layer, padding_shape):
        """
        Function responsible for creating Prefill stage tensor inputs for PyTorch model.
        :param n_layer : int
        :param padding_shape : List[int]
        :return inputs: Dict - input_ids, position_ids, past_key_values
        """

        inputs = self.tokenizer(
            self.input_str,
            return_tensors="pt",
            padding=True,
        )
        input_ids = inputs["input_ids"]
        batch_size, input_len = input_ids.shape
        inputs.pop("attention_mask")
        position_ids = torch.arange(input_len).view(1, -1)
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

        past_key_values = []
        for i in range(n_layer):
            past_key = torch.zeros((padding_shape), dtype=torch.float32)
            past_value = torch.zeros((padding_shape), dtype=torch.float32)
            pkv = (past_key, past_value)
            past_key_values.append(pkv)
        inputs["past_key_values"] = tuple(past_key_values)

        return inputs

    def update_pytorch_inputs(self, iteration, inputs, pt_outputs):
        """
        Function responsible for updating Prefill stage inputs to create inputs for decode stage inputs for PyTorch model.
        :param iteration:int
        :param inputs: Dict
        :param pt_outputs: Dict
        :return inputs: Dict - input_ids, position_ids, past_key_values
        """

        updated_inputs = {}
        updated_inputs["input_ids"] = pt_outputs["logits"].argmax(-1).reshape(-1, 1)
        updated_inputs["position_ids"] = inputs["position_ids"].max(1, keepdim=True).values + 1
        updated_inputs["past_key_values"] = tuple(
            [(key.detach(), value.detach()) for key, value in pt_outputs["past_key_values"]]
        )
        return updated_inputs

    def prepare_ort_inputs(self, n_layer, padding_shape):
        """
        Function responsible for creating Prefill stage numpy inputs for ONNX model to be run on ONNXRT.
        :param n_layer : int
        :param padding_shape : List[int]
        :return inputs: Dict - input_ids, position_ids, past_key_values
        """

        inputs = self.tokenizer(
            self.input_str,
            return_tensors="np",
            padding=True,
        )
        input_ids = inputs["input_ids"]
        batch_size, input_len = input_ids.shape
        inputs.pop("attention_mask")
        position_ids = np.arange(input_len).reshape(1, -1)
        inputs["input_ids"] = np.concatenate(
            [input_ids, np.full((batch_size, self.prompt_len - input_len), self.tokenizer.pad_token_id)],
            axis=1,
        ).astype(np.int64)
        inputs["position_ids"] = np.concatenate(
            [position_ids, np.full((batch_size, self.prompt_len - input_len), -1)],
            axis=1,
        ).astype(np.int64)

        for i in range(n_layer):
            inputs["past_key." + str(i)] = np.zeros((padding_shape), dtype=np.float32)
            inputs["past_value." + str(i)] = np.zeros((padding_shape), dtype=np.float32)

        return inputs

    def update_ort_inputs(self, iteration, inputs, ort_outputs, n_layer):
        """
        Function responsible for updating Prefill stage inputs to create inputs for decode stage inputs for ONNX model to be run on ONNXRT.
        :param iteration:int
        :param inputs: Dict
        :param ort_outputs: Dict
        :param n_layer : int
        :return inputs: Dict - input_ids, position_ids, past_key_values
        """

        updated_inputs = {}
        updated_inputs["input_ids"] = ort_outputs["logits"].argmax(-1)
        updated_inputs["position_ids"] = np.max(inputs["position_ids"], axis=1, keepdims=True) + 1
        for i in range(n_layer):
            updated_inputs["past_key." + str(i)] = ort_outputs["past_key_values"][i * 2]
            updated_inputs["past_value." + str(i)] = ort_outputs["past_key_values"][i * 2 + 1]

        return updated_inputs

    def prepare_cloud_ai_100_inputs(self, n_layer, padding_shape):
        """
        Function responsible for creating Prefill stage numpy inputs for ONNX model to be run on Cloud AI 100.
        :param n_layer : int
        :param padding_shape : List[int]
        :return inputs: Dict - input_ids, position_ids, past_key_values
        """

        inputs = self.tokenizer(
            self.input_str,
            return_tensors="np",
            padding=True,
        )
        input_ids = inputs["input_ids"]
        batch_size, input_len = inputs["input_ids"].shape
        inputs.pop("attention_mask")
        position_ids = np.arange(input_len).reshape(1, -1)
        inputs["input_ids"] = np.concatenate(
            [input_ids, np.full((batch_size, self.prompt_len - input_len), self.tokenizer.pad_token_id)],
            axis=1,
        ).astype(np.int64)
        inputs["position_ids"] = np.concatenate(
            [position_ids, np.full((batch_size, self.prompt_len - input_len), -1)],
            axis=1,
        ).astype(np.int64)

        for i in range(n_layer):
            inputs["past_key." + str(i)] = np.zeros((padding_shape), dtype=np.float16)
            inputs["past_value." + str(i)] = np.zeros((padding_shape), dtype=np.float16)

        return inputs

    def update_cloud_ai_100_inputs(self, iteration, inputs, outputs):
        """
        Function responsible for updating Prefill stage inputs to create inputs for
        decode stage inputs for ONNX model to be run on ONNXRT.
        :param iteration:int
        :param inputs: Dict
        :param outputs: Dict
        :return inputs: Dict - input_ids, position_ids
        """

        updated_inputs = {}
        updated_inputs["input_ids"] = outputs["logits"].argmax(-1)
        updated_inputs["position_ids"] = np.max(inputs["position_ids"], axis=1, keepdims=True) + 1

        return updated_inputs

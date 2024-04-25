# -----------------------------------------------------------------------------
#
# Copyright (c)  2023-2024 Qualcomm Innovation Center, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------

import numpy as np
import torch


class InputHandler:
    def __init__(self, tokenizer, input_str, prompt_len, ctx_len):
        """
        Initialization
        :param model_name: str
        :param input_str: List[str]
        :param prompt_len: int
        :param ctx_len: int
        """
        self.tokenizer = tokenizer
        self.input_str = input_str
        self.prompt_len = prompt_len
        self.ctx_len = ctx_len

    def prepare_pytorch_inputs(self, n_layer, padding_shape):
        """
        Function responsible for creating Prefill stage tensor inputs for PyTorch model.
        :param n_layer : int
        :param padding_shape : List[int]
        :return inputs: Dict - input_ids, position_ids,attention_mask, past_key_values, cache_index
        """

        inputs = self.tokenizer(
            self.input_str,
            return_tensors="pt",
            padding="max_length",
            max_length=self.prompt_len,
        )
        batch_size, input_len = inputs["input_ids"].shape
        inputs["attention_mask"] = torch.concat(
            [
                inputs["attention_mask"],
                torch.zeros((batch_size, self.ctx_len - self.prompt_len), dtype=torch.int64),
            ],
            1,
        )
        inputs["position_ids"] = ((torch.cumsum(inputs["attention_mask"], 1) - 1) * inputs["attention_mask"])[
            :, : inputs["input_ids"].shape[1]
        ]

        past_key_values = []
        for i in range(n_layer):
            past_key = torch.zeros((padding_shape), dtype=torch.float32)
            past_value = torch.zeros((padding_shape), dtype=torch.float32)
            pkv = (past_key, past_value)
            past_key_values.append(pkv)

        inputs["past_key_values"] = tuple(past_key_values)
        inputs["cache_index"] = torch.tensor(0)

        return inputs

    def update_pytorch_inputs(self, iteration, inputs, pt_outputs):
        """
        Function responsible for updating Prefill stage inputs to create inputs for decode stage inputs for PyTorch model.
        :param iteration:int
        :param inputs: Dict
        :param pt_outputs: Dict
        :return inputs: Dict - input_ids, position_ids,attention_mask, past_key_values, cache_index
        """

        updated_inputs = {}
        updated_inputs["input_ids"] = pt_outputs["logits"].argmax(-1).reshape(-1, 1)
        if iteration == 1:
            updated_inputs["attention_mask"] = inputs["attention_mask"].bool()
        else:
            updated_inputs["attention_mask"] = pt_outputs["attention_mask_RetainedState"].bool()
        updated_inputs["position_ids"] = updated_inputs["attention_mask"].sum(1, keepdim=True)
        updated_inputs["past_key_values"] = tuple(
            [(key.detach(), value.detach()) for key, value in pt_outputs["past_key_values"]]
        )
        updated_inputs["cache_index"] = torch.tensor(iteration + self.prompt_len - 1)

        return updated_inputs

    def prepare_ort_inputs(self, n_layer, padding_shape):
        """
        Function responsible for creating Prefill stage numpy inputs for ONNX model to be run on ONNXRT.
        :param n_layer : int
        :param padding_shape : List[int]
        :return inputs: Dict - input_ids, position_ids,attention_mask, past_key_values, cache_index
        """

        model_inputs = self.tokenizer(
            self.input_str,
            return_tensors="np",
            padding="max_length",
            max_length=self.prompt_len,
        )
        input_ids = model_inputs["input_ids"]
        attention_mask = model_inputs["attention_mask"]
        position_ids = (np.cumsum(attention_mask, axis=1) - 1) * attention_mask

        inputs = {}
        inputs["input_ids"] = input_ids

        batch_size, input_len = inputs["input_ids"].shape

        if len(position_ids.shape) == 1:
            inputs["position_ids"] = position_ids.astype(np.int64)[:, np.newaxis]
        else:
            position_ids = np.concatenate(
                [
                    np.zeros((position_ids.shape[0], self.prompt_len - input_len)),
                    position_ids,
                ],
                axis=1,
            ).astype(np.int64)
            inputs["position_ids"] = position_ids.astype(np.int64)

        if attention_mask.shape[-1] != self.ctx_len:
            attention_mask = np.concatenate(
                [
                    np.zeros((input_ids.shape[0], self.prompt_len - input_len)),
                    attention_mask,
                ],
                axis=1,
            ).astype(bool)
            attention_mask = np.concatenate(
                [
                    attention_mask,
                    np.zeros((input_ids.shape[0], self.ctx_len - self.prompt_len)),
                ],
                axis=1,
            ).astype(bool)
        inputs["attention_mask"] = attention_mask.astype(bool)

        inputs["cache_index"] = np.array(0)

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
        :return inputs: Dict - input_ids, position_ids,attention_mask, past_key_values, cache_index
        """

        attention_mask = ort_outputs["attention_mask"]
        past_key_values = ort_outputs["past_key_values"]

        input_ids = ort_outputs["logits"].argmax(-1)
        position_ids = np.sum(attention_mask, axis=1)

        if len(input_ids.shape) == 1:
            inputs["input_ids"] = input_ids.astype(np.int64)[:, np.newaxis]
        else:
            inputs["input_ids"] = input_ids.astype(np.int64)

        if len(position_ids.shape) == 1:
            inputs["position_ids"] = position_ids.astype(np.int64)[:, np.newaxis]
        else:
            inputs["position_ids"] = position_ids.astype(np.int64)
        inputs["attention_mask"] = attention_mask.astype(bool)

        for i in range(n_layer):
            inputs["past_key." + str(i)] = past_key_values[i * 2]
            inputs["past_value." + str(i)] = past_key_values[i * 2 + 1]

        inputs["cache_index"] = np.array(iteration + self.prompt_len - 1)

        return inputs

    def prepare_cloud_ai_100_inputs(self, n_layer, padding_shape):
        """
        Function responsible for creating Prefill stage numpy inputs for ONNX model to be run on Cloud AI 100.
        :param n_layer : int
        :param padding_shape : List[int]
        :return inputs: Dict - input_ids, position_ids,attention_mask, past_key_values, cache_index
        """

        model_inputs = self.tokenizer(
            self.input_str,
            return_tensors="np",
            padding="max_length",
            max_length=self.prompt_len,
        )
        input_ids = model_inputs["input_ids"]
        attention_mask = model_inputs["attention_mask"]
        position_ids = (np.cumsum(attention_mask, axis=1) - 1) * attention_mask

        inputs = {}
        inputs["input_ids"] = input_ids

        batch_size, input_len = inputs["input_ids"].shape

        if len(position_ids.shape) == 1:
            inputs["position_ids"] = position_ids.astype(np.int64)[:, np.newaxis]
        else:
            position_ids = np.concatenate(
                [
                    np.zeros((batch_size, self.prompt_len - position_ids.shape[1])),
                    position_ids,
                ],
                axis=1,
            ).astype(np.int64)
            inputs["position_ids"] = position_ids.astype(np.int64)

        if attention_mask.shape[-1] != self.ctx_len:
            attention_mask = np.concatenate(
                [np.zeros((batch_size, self.prompt_len - input_len)), attention_mask],
                axis=1,
            ).astype(bool)
            attention_mask = np.concatenate(
                [
                    attention_mask,
                    np.zeros((input_ids.shape[0], self.ctx_len - self.prompt_len)),
                ],
                axis=1,
            ).astype(bool)
        inputs["attention_mask"] = attention_mask.astype(bool)

        inputs["cache_index"] = np.array([0], np.int64)

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
        :return inputs: Dict - input_ids, position_ids, cache_index
        (since attention_mask and past_key_values inputs are skipped in decode stage at Cloud AI 100)
        """

        updated_inputs = {}

        attention_mask = outputs["attention_mask_RetainedState"]

        input_ids = outputs["logits"].argmax(-1)
        position_ids = np.sum(attention_mask, axis=1)
        if len(input_ids.shape) == 1:
            updated_inputs["input_ids"] = input_ids.astype(np.int64)[:, np.newaxis]
        else:
            updated_inputs["input_ids"] = input_ids.astype(np.int64)
        if len(position_ids.shape) == 1:
            updated_inputs["position_ids"] = position_ids.astype(np.int64)[:, np.newaxis]
        else:
            updated_inputs = np.concatenate(
                [
                    np.zeros((position_ids.shape[0], self.prompt_len - position_ids.shape[1])),
                    position_ids,
                ],
                axis=1,
            ).astype(np.int64)
            updated_inputs["position_ids"] = position_ids.astype(np.int64)
        updated_inputs["cache_index"] = np.array([iteration + self.prompt_len - 1], np.int64)

        return updated_inputs

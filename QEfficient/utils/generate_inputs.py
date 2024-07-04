# -----------------------------------------------------------------------------
#
# Copyright (c) 2024 Qualcomm Innovation Center, Inc. All rights reserved.
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
        --------

        :tokenizer: Union[PreTrainedTokenizer, PreTrainedTokenizerFast]. Pass model tokenizer.
        :input_str: List[str]. String to used as input prompt for the model.
        :prompt_len: int. prompt length for the model to compile.
        :ctx_len: int. Maximum context length to compile the model.
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
        --------

        :n_layer : int. Number of layers present in the model.
        :padding_shape : List[int]. Shape of Past Key values used for initialization with zeros in first iteration.

        :return inputs: Dict. input_ids, position_ids, past_key_values
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

    def update_pytorch_inputs(self, inputs, pt_outputs):
        """
        Function responsible for updating Prefill stage inputs to create decode stage inputs for PyTorch model.
        --------

        :inputs: Dict. Pytorch inputs from previous iteration
        :pt_outputs: Dict. Pytorch outputs from previous iteration

        :return updated_inputs: Dict. Updated input_ids, position_ids and past_key_values
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
        --------

        :n_layer : int. Number of layers present in the model.
        :padding_shape : List[int]. Shape of Past Key values used for initialization with zeros in first iteration.

        :return inputs: Dict. input_ids, position_ids, past_key_values
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

    def update_ort_inputs(self, inputs, ort_outputs, n_layer):
        """
        Function responsible for updating Prefill stage inputs to create inputs for decode stage inputs for ONNX model to be run on ONNXRT.
        --------

        :inputs: Dict. NumPy inputs of Onnx model from previous iteration
        :ort_outputs: Dict. Numpy outputs of Onnx model from previous iteration
        :n_layer : int. Number of layers present in the model.

        :return updated_inputs: Dict. Updated input_ids, position_ids and past_key_values
        """

        updated_inputs = {}
        updated_inputs["input_ids"] = ort_outputs["logits"].argmax(-1)
        updated_inputs["position_ids"] = np.max(inputs["position_ids"], axis=1, keepdims=True) + 1
        for i in range(n_layer):
            updated_inputs["past_key." + str(i)] = ort_outputs["past_key_values"][i * 2]
            updated_inputs["past_value." + str(i)] = ort_outputs["past_key_values"][i * 2 + 1]

        return updated_inputs

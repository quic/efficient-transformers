# -----------------------------------------------------------------------------
#
# Copyright (c) 2024 Qualcomm Innovation Center, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------

import numpy as np
import torch

from QEfficient.utils import get_num_layers_from_config, get_padding_shape_from_config, padding_check_and_fix


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
        self.n_layer = get_num_layers_from_config(config)
        # self.padding_shape = get_padding_shape_from_config(
        #     config=config, batch_size=full_batch_size if full_batch_size else batch_size, seq_len=ctx_len
        # )
        self.past_key_values = get_padding_shape_from_config(
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

        if self.full_batch_size:
            inputs["input_ids"] = input_ids
            inputs["position_ids"] = torch.arange(input_len).view(1, input_len)
            inputs["batch_index"] = torch.arange(1).view(-1, 1)

        # past_key_values = []
        # for i in range(self.n_layer):
        #     past_key = torch.zeros((self.padding_shape), dtype=torch.float32)
        #     past_value = torch.zeros((self.padding_shape), dtype=torch.float32)
        #     pkv = (past_key, past_value)
        #     past_key_values.append(pkv)
        # inputs["past_key_values"] = tuple(past_key_values)
        inputs["past_key_values"] = tuple(self.past_key_values)

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
            batch_index = torch.arange(1).view(-1, 1)

            input_ids = pt_outputs.logits.detach().argmax(2)
            updated_inputs["input_ids"] = torch.full((self.full_batch_size, 1), self.tokenizer.pad_token_id)
            updated_inputs["input_ids"][batch_index.view(-1)] = input_ids

            position_ids = inputs["position_ids"].max(1, keepdim=True).values + 1
            updated_inputs["position_ids"] = torch.full((self.full_batch_size, 1), 0)
            updated_inputs["position_ids"][batch_index.view(-1)] = position_ids

            updated_inputs["batch_index"] = torch.arange(self.full_batch_size).view(-1, 1)

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

        for i in range(self.n_layer):
            inputs["past_key." + str(i)] = np.zeros((self.padding_shape), dtype=np.float32)
            inputs["past_value." + str(i)] = np.zeros((self.padding_shape), dtype=np.float32)

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

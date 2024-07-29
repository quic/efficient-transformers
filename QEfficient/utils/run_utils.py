# -----------------------------------------------------------------------------
#
# Copyright (c) 2024 Qualcomm Innovation Center, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------

import numpy as np
import onnx
import onnxruntime
import torch

from QEfficient.generation.text_generation_inference import cloud_ai_100_exec_kv_helper
from QEfficient.utils.generate_inputs import InputHandler


class ApiRunner:
    """
    ApiRunner class is responsible for running:
    ---------

    1. HuggingFace PyTorch model
    2. Transformed KV Pytorch Model
    3. ONNX model on ONNXRT
    4. ONNX model on Cloud AI 100
    """

    def __init__(self, batch_size, tokenizer, config, prompt, prompt_len, ctx_len):
        """
        Initialization
        --------

        :batch_size: int. Number of prompts to run in one batch.
        :tokenizer: Union[PreTrainedTokenizer, PreTrainedTokenizerFast]. Pass model tokenizer.
        :config: AutoConfig from pretrained model.
        :prompt: List[str]. input prompt for running the model.
        :prompt_len: int. prompt length to compile the model.
        :ctx_len: int. Maximum context length to compile the model.
        """
        self.input_handler = InputHandler(
            batch_size=batch_size,
            tokenizer=tokenizer,
            config=config,
            prompt=prompt,
            prompt_len=prompt_len,
            ctx_len=ctx_len,
        )

        self.gen_len = self.input_handler.ctx_len - self.input_handler.prompt_len

    @torch.no_grad()
    def run_hf_model_on_pytorch(self, model_hf):
        """
        Function responsible for running HuggingFace PyTorch model and return the output tokens
        --------

        :model_hf: torch.nn.module. Original PyTorch model

        :return generated_ids: numpy.ndarray. Generated output tokens
        """
        input_ids = self.input_handler.tokenizer.encode(self.input_handler.prompt[0], return_tensors="pt")

        input_ids_len = len(input_ids[0])

        for _ in range(self.gen_len):
            outputs = model_hf(input_ids)
            logits = outputs.logits[:, -1, :]
            predicted_token_id = torch.argmax(logits, dim=-1)
            input_ids = torch.cat([input_ids, predicted_token_id.unsqueeze(1)], dim=-1)

        generated_ids = input_ids[0][input_ids_len:].detach().numpy()
        generated_text = self.input_handler.tokenizer.decode(generated_ids, skip_special_tokens=True)
        print("Original HF Model Outputs (Torch CPU): \n")
        print("Prompt:", repr(self.input_handler.prompt))
        print("Completion:", repr(generated_text))
        return generated_ids

    def run_kv_model_on_pytorch(self, model):
        """
        Function responsible for running KV PyTorch model and return the output tokens
        --------

        :model: torch.nn.module. Transformed PyTorch model

        :return generated_ids: numpy.ndarray. Generated output tokens
        """

        generated_ids = []
        inputs = self.input_handler.prepare_pytorch_inputs()

        pt_outputs = model(**inputs)
        for _ in range(1, self.gen_len):
            generated_ids.append(pt_outputs["logits"].argmax(-1).reshape(-1, 1))
            inputs = self.input_handler.update_pytorch_inputs(inputs, pt_outputs)
            pt_outputs = model(**inputs)

        generated_ids.append(pt_outputs["logits"].argmax(-1).reshape(-1, 1))
        generated_ids = np.concatenate(generated_ids, axis=1)
        predicted_string = self.input_handler.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
        print("QEff Transformed HF Model Outputs (Torch CPU): \n")
        print("Prompt:", repr(self.input_handler.prompt))
        print("Completion:", repr(predicted_string))
        return generated_ids

    def run_ort_session(self, inputs, session):
        """
        Function responsible for running onnxrt session with given inputs and
        passing retained state outputs to be used for next iteration inputs
        --------

        :inputs: Dict. Numpy inputs of Onnx model
        :session: 'onnxruntime.capi.onnxruntime_inference_collection.InferenceSession'.

        :return outputs: Dict. Numpy outputs of Onnx model
        """
        output_names = [x.name for x in session.get_outputs()]
        session_input_names = [x.name for x in session.get_inputs()]
        session_inputs = {}
        for inp_name in session_input_names:
            if inp_name in inputs.keys():
                session_inputs[inp_name] = inputs[inp_name]
        outputs_data = session.run(output_names, session_inputs)
        ort_outputs = dict(zip(output_names, outputs_data))
        return ort_outputs

    def run_kv_model_on_ort(self, model_path):
        """
        Function responsible for running ONNX model on onnxruntime and return the output tokens
        --------

        :model_path: str. Path to the Onnx model.

        :return generated_ids: numpy.ndarray. Generated output tokens
        """

        # todo:vbaddi; find a better version to do this changes
        # Currently the gathernd invalid index is set to INT MAX(FP16) and hence fails in OnnxRuntime
        # Changing the constant value from INT MAX to -1. Fixes the issue.
        m = onnx.load(model_path, load_external_data=False)
        for node in m.graph.node:
            if node.op_type == "Constant":
                np_tensor = onnx.numpy_helper.to_array(node.attribute[0].t)
                if len(np_tensor.shape) == 0 and np_tensor.item() == 65504:
                    node.attribute[0].t.raw_data = np.array(-1).tobytes()

        onnxruntime_model = model_path[:-5] + "_ort.onnx"
        onnx.save(m, onnxruntime_model)
        session = onnxruntime.InferenceSession(onnxruntime_model)

        generated_ids = []
        inputs = self.input_handler.prepare_ort_inputs()
        ort_outputs = self.run_ort_session(inputs, session)
        ort_outputs = self.input_handler.update_ort_outputs(ort_outputs)

        for _ in range(1, self.gen_len):
            generated_ids.append(ort_outputs["logits"].argmax(-1).reshape(-1, 1))
            inputs = self.input_handler.update_ort_inputs(inputs, ort_outputs)
            ort_outputs = self.run_ort_session(inputs, session)
            ort_outputs = self.input_handler.update_ort_outputs(ort_outputs)

        generated_ids.append(ort_outputs["logits"].argmax(-1).reshape(-1, 1))
        generated_ids = np.concatenate(generated_ids, axis=1)
        predicted_string = self.input_handler.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
        print("QEff Transformed Onnx Model Outputs (OnnxRuntime CPU): \n")
        print("Prompt:", repr(self.input_handler.prompt))
        print("Completion:", repr(predicted_string))
        return generated_ids

    def run_kv_model_on_cloud_ai_100(self, qpc_path, device_group):
        """
        Function responsible for running ONNX model on Cloud AI 100 and return the output tokens
        --------

        :qpc_path: str. path to qpc generated after compilation
        :device_group: List[int]. Device Ids to be used for compilation. if len(device_group) > 1. Multiple Card setup is enabled.

        :return generated_ids: numpy.ndarray. Generated output tokens
        """
        execinfo = cloud_ai_100_exec_kv_helper(
            tokenizer=self.input_handler.tokenizer,
            qpc_path=qpc_path,
            device_id=device_group,
            ctx_len=self.input_handler.ctx_len,
            generation_len=self.gen_len,
            prompt=self.input_handler.prompt,
            stream=False,
        )
        predicted_string = self.input_handler.tokenizer.batch_decode(execinfo.generated_ids, skip_special_tokens=True)
        print("QEff Transformed Model Outputs (Cloud AI 100): \n")
        print("Prompt:", repr(self.input_handler.prompt))
        print("Completion:", repr(predicted_string))
        return execinfo.generated_ids

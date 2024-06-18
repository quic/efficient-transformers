# -----------------------------------------------------------------------------
#
# Copyright (c)  2023-2024 Qualcomm Innovation Center, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------

import numpy as np
import onnx
import onnxruntime
import torch

from .generate_inputs import InputHandler


class ApiRunner:
    """
    ApiRunner class is responsible for:

    1. Running Huggingface PyTorch model
    2. Running KV Pytorch Model
    3. Running ONNX model on ONNXRT
    4. Running ONNX Model on Cloud AI 100
    """

    def __init__(self, tokenizer, prompt, prompt_len, ctx_len):
        """
        Initialization
        :param tokenizer: tokenizer
        :param input_str: List[str]
        :param prompt_len: int
        :param ctx_len: int
        """
        
        self.tokenizer = tokenizer
        self.prompt = prompt
        self.prompt_len = prompt_len
        self.ctx_len = ctx_len
        self.gen_len = self.ctx_len - self.prompt_len

        self.input_handler = InputHandler(self.tokenizer, self.prompt, self.prompt_len, self.ctx_len)

    def run_hf_model_on_pytorch(self, model_hf):
        """
        Function responsible for running Huggingface PyTorch model and return the output tokens
        :param model_hf: pytorch model
        :return generated_ids: numpy.ndarray - output tokens
        """
        input_ids = self.tokenizer.encode(self.prompt[0], return_tensors="pt")

        input_ids_len = len(input_ids[0])

        with torch.no_grad():
            for _ in range(self.gen_len):
                outputs = model_hf(input_ids)
                logits = outputs.logits[:, -1, :]
                predicted_token_id = torch.argmax(logits, dim=-1)
                input_ids = torch.cat([input_ids, predicted_token_id.unsqueeze(1)], dim=-1)

        generated_ids = input_ids[0][input_ids_len:].detach().numpy()
        generated_text = self.tokenizer.decode(generated_ids, skip_special_tokens=True)
        print("Original HF Model Outputs (Torch CPU): \n")
        print("Prompt:", repr(self.prompt))
        print("Completion:", repr(generated_text))
        return generated_ids


    def run_kv_model_on_pytorch(self, model, n_layer, padding_shape):
        """
        Function responsible for running KV PyTorch model and return the output tokens
        :param model_hf: pytorch model
        :param n_layer : int
        :param padding_shape : List[int]
        :return generated_ids: numpy.ndarray - output tokens
        """

        generated_ids = []
        inputs = self.input_handler.prepare_pytorch_inputs(n_layer, padding_shape)

        pt_outputs = model(**inputs)
        for i in range(1, self.gen_len):
            generated_ids.append(pt_outputs["logits"].argmax(-1).reshape(-1, 1))
            inputs = self.input_handler.update_pytorch_inputs(i, inputs, pt_outputs)
            pt_outputs = model(**inputs)

        generated_ids.append(pt_outputs["logits"].argmax(-1).reshape(-1, 1))
        generated_ids = np.concatenate(generated_ids, axis=1)
        predicted_string = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
        print("QEff Transformed HF Model Outputs (Torch CPU): \n")
        print("Prompt:", repr(self.prompt))
        print("Completion:", repr(predicted_string))
        return generated_ids

    def run_ort_session(self, inputs, session, n_layer):
        """
        Function responsible for running onnxrt session with given inputs and passing retained state outputs to be used for next iteration inputs
        :param inputs: Dict
        :param session: 'onnxruntime.capi.onnxruntime_inference_collection.InferenceSession'
        :param n_layer: int
        :return outputs: Dict
        """

        outputs = {}
        output_names = [x.name for x in session.get_outputs()]
        session_input_names = [x.name for x in session.get_inputs()]
        session_inputs = {}
        for inp_name in session_input_names:
            if inp_name in inputs.keys():
                session_inputs[inp_name] = inputs[inp_name]
        outputs_data = session.run(output_names, session_inputs)
        ort_outputs = dict(zip(output_names, outputs_data))

        present_key_values = []
        for i in range(n_layer):
            if "past_key." + str(i) + "_RetainedState" in ort_outputs:
                present_key_values.append(ort_outputs["past_key." + str(i) + "_RetainedState"])
            if "past_value." + str(i) + "_RetainedState" in ort_outputs:
                present_key_values.append(ort_outputs["past_value." + str(i) + "_RetainedState"])

        outputs["past_key_values"] = present_key_values
        outputs["logits"] = ort_outputs["logits"]

        return outputs

    def run_kv_model_on_ort(self, model_path, n_layer, padding_shape):
        """
        Function responsible for running ONNX model on onnxruntime and return the output tokens
        :param model_path: str
        :param n_layer : int
        :param padding_shape : List[int]
        :return generated_ids: numpy.ndarray - output tokens
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
        inputs = self.input_handler.prepare_ort_inputs(n_layer, padding_shape)
        ort_outputs = self.run_ort_session(inputs, session, n_layer)

        for i in range(1, self.gen_len):
            generated_ids.append(ort_outputs["logits"].argmax(-1).reshape(-1, 1))
            inputs = self.input_handler.update_ort_inputs(i, inputs, ort_outputs, n_layer)
            ort_outputs = self.run_ort_session(inputs, session, n_layer)

        generated_ids.append(ort_outputs["logits"].argmax(-1).reshape(-1, 1))
        generated_ids = np.concatenate(generated_ids, axis=1)
        predicted_string = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
        print("QEff Transformed Onnx Model Outputs (OnnxRuntime CPU): \n")
        print("Prompt:", repr(self.prompt))
        print("Completion:", repr(predicted_string))
        return generated_ids

    def run_kv_model_on_cloud_ai_100(self, session, n_layer, padding_shape):
        """
        Function responsible for running ONNX model on Cloud AI 100 and return the output tokens
        :param qpc_path: str
        :param device_id: List[int]
        :param n_layer : int
        :param padding_shape : List[int]
        :return generated_ids: numpy.ndarray - output tokens
        """

        generated_ids = []
        inputs = self.input_handler.prepare_cloud_ai_100_inputs(n_layer, padding_shape)

        outputs = session.run(inputs)

        for i in range(1, self.gen_len):
            generated_ids.append(outputs["logits"].argmax(-1).reshape(-1, 1))
            inputs = self.input_handler.update_cloud_ai_100_inputs(i, inputs, outputs)
            session.skip_buffers([x for x in session.input_names + session.output_names if x.startswith("past_")])
            outputs = session.run(inputs)

        generated_ids.append(outputs["logits"].argmax(-1).reshape(-1, 1))
        generated_ids = np.concatenate(generated_ids, axis=1)
        predicted_string = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
        print("QEff Transformed Model Outputs (Cloud AI 100): \n")
        print("Prompt:", repr(self.prompt))
        print("Completion:", repr(predicted_string))
        return generated_ids

# -----------------------------------------------------------------------------
#
# Copyright (c) 2024 Qualcomm Innovation Center, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------

import os

import numpy as np
import onnx
import onnxruntime
import torch
from transformers import TextStreamer

from transformers import TextStreamer
from QEfficient.generation.text_generation_inference import TextGeneration
from QEfficient.utils.generate_inputs import InputHandler, InputHandlerVLM


# TODO: Deprecate this class and encourage the use of `QeffAutoModel...` classes
class ApiRunner:
    """
    ApiRunner class is responsible for running:
    ---------

    1. HuggingFace ``PyTorch`` model
    2. Transformed KV Pytorch Model
    3. ``ONNX`` model on ONNXRT
    4. ``ONNX`` model on Cloud AI 100
    """

    def __init__(self, batch_size, tokenizer, config, prompt, prompt_len, ctx_len, full_batch_size=None):
        """
        Initialization

        Args:
            :batch_size (int): Number of prompts to run in one batch.
            :tokenizer (Union[PreTrainedTokenizer, PreTrainedTokenizerFast]): Pass model tokenizer.
            :config (AutoConfig): From pretrained model.
            :prompt (List[str]): Input prompt for running the model.
            :prompt_len (int): Prompt length to compile the model.
            :ctx_len (int): Maximum context length to compile the model.
        """
        self.input_handler = InputHandler(
            batch_size=batch_size,
            tokenizer=tokenizer,
            config=config,
            prompt=prompt,
            prompt_len=prompt_len,
            ctx_len=ctx_len,
            full_batch_size=full_batch_size,
        )

        self.gen_len = self.input_handler.ctx_len - self.input_handler.prompt_len

    @torch.no_grad()
    def run_hf_model_on_pytorch_CB(self, model_hf):
        """
        Function responsible for running HuggingFace ``PyTorch`` model and return the output tokens

        ``Mandatory`` Args:
            :model_hf (torch.nn.module): Original ``PyTorch`` model

        Return:
            :numpy.ndarray: Generated output tokens
        """
        input_ids = [
            self.input_handler.tokenizer.encode(prompt, return_tensors="pt") for prompt in self.input_handler.prompt
        ]

        generated_ids = []

        for idx, inp_ids in enumerate(input_ids):
            gen_ids = inp_ids.clone()
            for _ in range(self.gen_len):
                outputs = model_hf(input_ids=gen_ids)
                logits = outputs.logits[:, -1, :]
                predicted_token_id = torch.argmax(logits, dim=-1)
                gen_ids = torch.cat([gen_ids, predicted_token_id.unsqueeze(-1)], dim=-1)

            gen_ids = gen_ids.detach().numpy()
            gen_ids = gen_ids[:, inp_ids.shape[1] :]
            generated_ids.append(gen_ids)

        generated_texts = [
            self.input_handler.tokenizer.decode(gen_ids.squeeze().tolist(), skip_special_tokens=True)
            for gen_ids in generated_ids
        ]
        print("Original HF Model Outputs (Torch CPU): \n")
        print("Prompt:", repr(self.input_handler.prompt))
        print("Completion:", repr(generated_texts))
        return generated_ids

    @torch.no_grad()
    def run_hf_model_on_pytorch(self, model_hf):
        """
        Function responsible for running HuggingFace ``PyTorch`` model and return the output tokens

        ``Mandatory`` Args:
            :model_hf (torch.nn.module): Original ``PyTorch`` model

        Return:
            :numpy.ndarray: Generated output tokens
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
        Function responsible for running KV ``PyTorch`` model and return the output tokens

        ``Mandatory`` Args:
        :model (torch.nn.module): Transformed ``PyTorch`` model

        Return:
            :numpy.ndarray: Generated output tokens
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

    def run_ort_session(self, inputs, session) -> dict:
        """
        Function responsible for running onnxrt session with given inputs and passing retained state outputs to be used for next iteration inputs

        ``Mandatory`` Args:
            :inputs (Dict):
            :session (onnxruntime.capi.onnxruntime_inference_collection.InferenceSession):

        Return:
            :Dict: Numpy outputs of Onnx model
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

    def run_kv_model_on_ort(self, model_path, is_tlm=False):
        """
        Function responsible for running ``ONNX`` model on onnxruntime and return the output tokens

        ``Mandatory`` Args:
            :model_path (str): Path to the Onnx model.

        Return:
            :numpy.ndarray: Generated output tokens
        """

        # Replace invalid index value for INT32 max to 0 using add_initializer
        m = onnx.load(model_path, load_external_data=False)
        # NOTE: OrtValue objects should be kept around until the session is run, hence this dict is required
        added_initializers = {}
        for node in m.graph.node:
            if node.op_type == "Constant":
                np_tensor = onnx.numpy_helper.to_array(node.attribute[0].t, os.path.dirname(model_path))
                if len(np_tensor.shape) == 0 and np_tensor.item() == 2147483647:
                    added_initializers[node.output[0]] = onnxruntime.OrtValue.ortvalue_from_numpy(
                        np.array(0, np_tensor.dtype)
                    )

        session_options = onnxruntime.SessionOptions()
        for name, value in added_initializers.items():
            session_options.add_initializer(name, value)
        session = onnxruntime.InferenceSession(model_path, session_options)

        generated_ids = []
        inputs = self.input_handler.prepare_ort_inputs()
        if is_tlm:
            nltk = np.zeros((1, 1), dtype=np.int64)
            inputs["num_logits_to_keep"] = nltk
        ort_outputs = self.run_ort_session(inputs, session)
        ort_outputs = self.input_handler.update_ort_outputs(ort_outputs)

        for _ in range(1, self.gen_len):
            generated_ids.append(ort_outputs["logits"].argmax(-1).reshape(-1, 1))
            inputs = self.input_handler.update_ort_inputs(inputs, ort_outputs)
            if is_tlm:
                inputs["num_logits_to_keep"] = nltk
            ort_outputs = self.run_ort_session(inputs, session)
            ort_outputs = self.input_handler.update_ort_outputs(ort_outputs)

        generated_ids.append(ort_outputs["logits"].argmax(-1).reshape(-1, 1))
        generated_ids = np.concatenate(generated_ids, axis=1)
        predicted_string = self.input_handler.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
        print("QEff Transformed Onnx Model Outputs (OnnxRuntime CPU): \n")
        print("Prompt:", repr(self.input_handler.prompt))
        print("Completion:", repr(predicted_string))
        return generated_ids

    def run_kv_model_on_cloud_ai_100(self, qpc_path, device_group=None):
        """
        Function responsible for running ``ONNX`` model on Cloud AI 100 and return the output tokens

        ``Mandatory`` Args:
            :qpc_path (str): path to qpc generated after compilation
            :device_group (List[int]): Device Ids to be used for compilation. if len(device_group) > 1. Multiple Card setup is enabled.

        Return:
            :numpy.ndarray: Generated output tokens
        """
        execinfo = TextGeneration(
            tokenizer=self.input_handler.tokenizer,
            qpc_path=qpc_path,
            device_id=device_group,
            ctx_len=self.input_handler.ctx_len,
            full_batch_size=self.input_handler.full_batch_size,
        ).generate(prompt=self.input_handler.prompt, generation_len=self.gen_len, stream=False)

        predicted_string = self.input_handler.tokenizer.batch_decode(execinfo.generated_ids, skip_special_tokens=True)
        print("QEff Transformed Model Outputs (Cloud AI 100): \n")
        print("Prompt:", repr(self.input_handler.prompt))
        print("Completion:", repr(predicted_string))
        return execinfo.generated_ids


class ApiRunnerVlm:
    """
    ApiRunnerVlm class is responsible for running Vision models:
    ---------

    1. HuggingFace ``PyTorch`` model
    2. Transformed KV Pytorch Model
    3. ``ONNX`` model on ONNXRT
    4. ``ONNX`` model on Cloud AI 100
    """

    def __init__(
        self, batch_size, processor, config, image, conversation, prompt, prompt_len, ctx_len, max_gen_len, n_layer
    ):
        """ """
        self.input_handler_vlm = InputHandlerVLM(
            batch_size=batch_size,
            prompt_len=prompt_len,
            ctx_len=ctx_len,
            max_gen_len=max_gen_len,
            config=config,
            image=image,
            conversation=conversation,
            processor=processor,
            n_layer=n_layer,
            prompt=prompt,
        )
        self.processor = processor
        self.ctx_len = ctx_len
        self.prompt_len = prompt_len
        self.batch_size = batch_size
        self.config = config
        self.gen_len = max_gen_len

    ### works for early + late and any kv_offload instance
    @torch.no_grad()
    def run_vlm_hf_model_on_pytorch(self, model, inputs):
        output = model.generate(**inputs, max_new_tokens=self.gen_len, do_sample=False)
        offset_output = output[0, inputs["input_ids"].shape[1] :]
        py_output = self.processor.tokenizer.decode(offset_output).strip()
        print("Original HF Model Outputs (Torch CPU):")
        print("Completion:", repr(py_output))
        return offset_output

    ### works for early + late and any kv_offload instance
    @torch.no_grad()
    def run_late_fusion_vlm_kv_model_on_pytorch(self, model):
        generation_len = self.gen_len
        generated_ids = torch.full((self.batch_size, generation_len), self.processor.tokenizer.pad_token_id)
        inputs = self.input_handler_vlm.prepare_pytorch_inputs()

        outputs = model(**inputs)
        inputs["input_ids"] = outputs[0].argmax(2)
        if "cross_attention_mask" in inputs:
            bs, _, num_images, img_tiles = inputs["cross_attention_mask"].shape
            inputs["cross_attention_mask"] = torch.ones((bs, 1, num_images, img_tiles), dtype=torch.int64)

        generated_ids[:, 0] = inputs["input_ids"].squeeze(1)
        finished_sequences = inputs["input_ids"] == self.processor.tokenizer.eos_token_id
        inputs["position_ids"] = inputs["position_ids"].max(1, keepdim=True).values + 1

        print("QEFF Model Outputs (Torch CPU):")
        streamer = TextStreamer(self.processor.tokenizer)
        streamer.put(inputs["input_ids"])
        for num_token in range(1, self.gen_len):
            outputs = model(**inputs)
            inputs["input_ids"] = outputs[0].argmax(2)
            inputs["position_ids"] += 1
            streamer.put(inputs["input_ids"])
            generated_ids[:, num_token] = inputs["input_ids"].squeeze(1)
            finished_sequences |= inputs["input_ids"] == self.processor.tokenizer.eos_token_id
            if finished_sequences.all():
                break
        streamer.end()
        return generated_ids[0]

    def late_fusion_llava_both_kv_model_on_pytorch(self, model, inputs):
        breakpoint()
        txt_cfg = self.config.get_text_config()
        num_hidden_layers = txt_cfg.num_hidden_layers
        num_key_value_heads = txt_cfg.num_key_value_heads
        head_dim = txt_cfg.hidden_size // txt_cfg.num_attention_heads
        if hasattr(txt_cfg, "cross_attention_layers"):
            cross_attention_layers = txt_cfg.cross_attention_layers

            vis_cfg = self.config.vision_config
            num_patches = (vis_cfg.image_size // vis_cfg.patch_size) ** 2 + 1
            image_tokens_len = vis_cfg.max_num_tiles * num_patches

        generation_len = self.gen_len
        generated_ids = torch.full((self.batch_size, generation_len), self.processor.tokenizer.pad_token_id)
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
        outputs = model(**inputs)
        inputs["input_ids"] = outputs[0].argmax(2)
        if "cross_attention_mask" in inputs:
            bs, _, num_images, img_tiles = inputs["cross_attention_mask"].shape
            inputs["cross_attention_mask"] = torch.ones((bs, 1, num_images, img_tiles), dtype=torch.int64)
        generated_ids[:, 0] = inputs["input_ids"].squeeze(1)
        finished_sequences = inputs["input_ids"] == self.processor.tokenizer.eos_token_id
        inputs["position_ids"] = inputs["position_ids"].max(1, keepdim=True).values + 1
        print("QEFF Model Outputs (Torch CPU):")
        streamer = TextStreamer(self.processor.tokenizer)
        streamer.put(inputs["input_ids"])
        for num_token in range(1, self.gen_len):
            outputs = model(**inputs)
            inputs["input_ids"] = outputs[0].argmax(2)
            inputs["position_ids"] += 1
            streamer.put(inputs["input_ids"])
            generated_ids[:, num_token] = inputs["input_ids"].squeeze(1)
            finished_sequences |= inputs["input_ids"] == self.processor.tokenizer.eos_token_id
            if finished_sequences.all():
                break
        streamer.end()
        return generated_ids[0]

    def run_ort_session(self, inputs, session) -> dict:
        """
        Function responsible for running onnxrt session with given inputs and passing retained state outputs to be used for next iteration inputs

        ``Mandatory`` Args:
            :inputs (Dict):
            :session (onnxruntime.capi.onnxruntime_inference_collection.InferenceSession):

        Return:
            :Dict: Numpy outputs of Onnx model
        """
        output_names = [x.name for x in session.get_outputs()]
        session_input_names = [x.name for x in session.get_inputs()]
        session_inputs = {}
        for inp_name in session_input_names:
            if inp_name in inputs.keys():
                session_inputs[inp_name] = inputs[inp_name]
        # breakpoint()
        outputs_data = session.run(output_names, session_inputs)
        ort_outputs = dict(zip(output_names, outputs_data))
        return ort_outputs

    def setup_ort_session(self, model_path):
        m = onnx.load(model_path, load_external_data=False)
        # NOTE: OrtValue objects should be kept around until the session is run, hence this dict is required
        added_initializers = {}
        for node in m.graph.node:
            if node.op_type == "Constant":
                np_tensor = onnx.numpy_helper.to_array(node.attribute[0].t, os.path.dirname(model_path))
                if len(np_tensor.shape) == 0 and np_tensor.item() == 2147483647:
                    added_initializers[node.output[0]] = onnxruntime.OrtValue.ortvalue_from_numpy(
                        np.array(0, np_tensor.dtype)
                    )
        session_options = onnxruntime.SessionOptions()
        for name, value in added_initializers.items():
            session_options.add_initializer(name, value)
        session = onnxruntime.InferenceSession(model_path, session_options)

        return added_initializers, session

    def run_late_fusion_vlm_kv_model_on_ort(self, model_path):
        vision_inputs, lang_inputs = self.input_handler_vlm.prepare_vlm_ort_inputs()
        # TODO: Make a DAG based parser to compile and run N ONNX files with dependencies
        ### If kv_offload was `True`
        if isinstance(model_path, list):
            encoder_path = model_path[0]
            decoder_path = model_path[1]

            added_initializers, encoder_session = self.setup_ort_session(encoder_path)

            encoder_ort_outputs = self.run_ort_session(vision_inputs, session=encoder_session)
            lang_inputs.update(encoder_ort_outputs)
            del added_initializers
            ### TEXT COMPONENT RUNNING

            added_initializers, decoder_session = self.setup_ort_session(decoder_path)
            generated_ids = []

            ort_outputs = self.run_ort_session(lang_inputs, session=decoder_session)
            ort_outputs = self.input_handler_vlm.update_vlm_ort_outputs(ort_outputs)
            for _ in range(1, self.gen_len):
                generated_ids.append(ort_outputs["logits"].argmax(-1).reshape(-1, 1))
                inputs = self.input_handler_vlm.update_vlm_ort_inputs(lang_inputs, ort_outputs)
                ort_outputs = self.run_ort_session(inputs, decoder_session)
                ort_outputs = self.input_handler_vlm.update_vlm_ort_outputs(ort_outputs)
            generated_ids.append(ort_outputs["logits"].argmax(-1).reshape(-1, 1))
            generated_ids = np.concatenate(generated_ids, axis=1)
            predicted_string = self.processor.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
            print("ORT KV_OFFLOAD Session Outputs:")
            print("Completion:", repr(predicted_string))
            del added_initializers

        ### IF MODELPATH IS A SINGLE POSIXPATH
        else:
            added_initializers, session = self.setup_ort_session(model_path)
            generated_ids = []
            inputs = {**vision_inputs, **lang_inputs}
            ort_outputs = self.run_ort_session(inputs, session=session)
            ort_outputs = self.input_handler_vlm.update_vlm_ort_outputs(ort_outputs)
            for _ in range(1, self.gen_len):
                generated_ids.append(ort_outputs["logits"].argmax(-1).reshape(-1, 1))
                inputs = self.input_handler_vlm.update_vlm_ort_inputs(inputs, ort_outputs)
                ort_outputs = self.run_ort_session(inputs, session)
                ort_outputs = self.input_handler_vlm.update_vlm_ort_outputs(ort_outputs)
            generated_ids.append(ort_outputs["logits"].argmax(-1).reshape(-1, 1))
            generated_ids = np.concatenate(generated_ids, axis=1)
            predicted_string = self.processor.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
            print("ORT Session Outputs:")
            print("Completion:", repr(predicted_string))
            del added_initializers
        return generated_ids

# -----------------------------------------------------------------------------
#
# Copyright (c) 2024 Qualcomm Innovation Center, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------

from peft import LoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer

from .onnx_utils import ONNXModel


class QEffModelManager:
    def __init__(self, config):
        self.config = config
        self.model = None
        self.tokenizer = None
        self.onnx_model = None
        self.qaic_compiler = None
        self.qaic_loader = None

    def initialize_model(self):
        self.tokenizer = AutoTokenizer.from_pretrained(self.config.model_id)
        self.model = AutoModelForCausalLM.from_pretrained(self.config.model_id, dropout=0.0, use_cache=False)
        lora_config = LoraConfig(**self.config.lora_config)
        self.model.add_adapter(lora_config, adapter_name="adapter_1")

    def prepare_for_training(self):
        self.export_to_onnx()
        self.apply_onnx_transforms()
        self.compile_for_qaic()
        self.load_qaic_model()

    def export_to_onnx(self):
        sample_input = self._get_sample_input()
        self.onnx_model = ONNXModel(self.config.model_id)
        self.onnx_model.export(self.model, sample_input, "model.onnx")

    def apply_onnx_transforms(self):
        trainable_params = set([name for name, param in self.model.named_parameters() if param.requires_grad])
        frozen_params = set([name for name, param in self.model.named_parameters() if not param.requires_grad])
        self.onnx_model.modify(trainable_params, frozen_params)
        self.onnx_model.save("model_transformed.onnx")

    def compile_for_qaic(self):
        # Implement QAIC compilation here
        pass

    def load_qaic_model(self):
        # Implement QAIC model loading here
        pass

    def get_model(self):
        return self.model

    def get_tokenizer(self):
        return self.tokenizer

    def generate(self, prompt: str, max_length: int = 100):
        input_ids = self.tokenizer.encode(prompt, return_tensors="pt")
        output = self.model.generate(input_ids, max_length=max_length)
        return self.tokenizer.decode(output[0], skip_special_tokens=True)

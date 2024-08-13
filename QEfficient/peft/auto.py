# -----------------------------------------------------------------------------
#
# Copyright (c) 2024 Qualcomm Innovation Center, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
#
# ----------------------------------------------------------------------------

import os
import shutil
import warnings
from typing import Dict, List, Optional

import numpy as np
import onnx
import torch
from peft import AutoPeftModelForCausalLM
from torch import nn

from QEfficient.base.modeling_qeff import QEFFBaseModel
from QEfficient.base.onnx_transforms import FP16ClipTransform, OnnxTransform, SplitTensorsTransform
from QEfficient.base.pytorch_transforms import PytorchTransform
from QEfficient.peft.onnx_transforms import AdaptersAsInputsTransform
from QEfficient.peft.pytorch_transforms import PeftModelInputsTransform
from QEfficient.transformers.pytorch_transforms import CustomOpsTransform, KVCacheTransform
from QEfficient.utils._utils import get_num_layers_from_config, get_padding_shape_from_config
from QEfficient.utils.cache_dir import QEFF_HOME


class QEffAutoPeftModelForCausalLM(QEFFBaseModel):
    pytorch_transforms: List[PytorchTransform] = [CustomOpsTransform, KVCacheTransform, PeftModelInputsTransform]
    onnx_transforms: List[OnnxTransform] = [FP16ClipTransform, AdaptersAsInputsTransform, SplitTensorsTransform]
    _hf_auto_class = AutoPeftModelForCausalLM

    def __init__(self, model: nn.Module, card_name: Optional[str] = None):
        super().__init__(model)
        self.card_name = card_name
        self.transform()

    @classmethod
    def from_pretrained(cls, pretrained_name_or_path: str, **kwargs):
        # Base class
        card_name = kwargs.pop("card_name", None)
        if not kwargs.get("use_cache"):
            warnings.warn("Overriding to use_cache=True")
        kwargs["use_cache"] = True
        model = cls._hf_auto_class.from_pretrained(pretrained_name_or_path, **kwargs)

        if card_name is None and (not os.path.exists(pretrained_name_or_path)):
            card_name = pretrained_name_or_path

        return cls(model, card_name)

    def transform(self, **kwargs):
        # Base class
        for transform in self.pytorch_transforms:
            self.model, transformed = transform.apply(self.model)

    @property
    def sample_inputs(self) -> Dict[str, torch.Tensor]:
        kv_cache_shape = get_padding_shape_from_config(self.model.config, 1, 32)
        num_layers = get_num_layers_from_config(self.model.config)
        inputs = {
            "input_ids": torch.zeros((1, 32), dtype=torch.int64),
            "position_ids": torch.arange(32, dtype=torch.int64).view((1, 32)),
            "past_key_values": [
                (
                    torch.zeros(kv_cache_shape, dtype=torch.float32),
                    torch.zeros(kv_cache_shape, dtype=torch.float32),
                )
                for _ in range(num_layers)
            ],
        }
        return inputs

    @property
    def dynamic_axes(self) -> Dict[str, Dict[int, str]]:
        num_layers = get_num_layers_from_config(self.model.config)
        dynamic_axes = {
            "input_ids": {0: "batch_size", 1: "seq_len"},
            "position_ids": {0: "batch_size", 1: "seq_len"},
        }
        for i in range(num_layers):
            dynamic_axes[f"past_key.{i}"] = {0: "batch_size", 2: "ctx_len"}
            dynamic_axes[f"past_value.{i}"] = {0: "batch_size", 2: "ctx_len"}
        return dynamic_axes

    @property
    def input_names(self) -> List[str]:
        return list(self.dynamic_axes.keys())

    @property
    def output_names(self) -> List[str]:
        num_layers = get_num_layers_from_config(self.model.config)
        outputs = ["logits"]
        for i in range(num_layers):
            outputs.append(f"past_key.{i}_RetainedState")
            outputs.append(f"past_value.{i}_RetainedState")
        return outputs

    def export(self, export_dir: Optional[str] = None) -> str:
        # Base class
        model_name = self.card_name.replace("/", "_")
        if export_dir is None:
            export_dir = os.path.join(QEFF_HOME, model_name)
        self.onnx_path = os.path.join(export_dir, f"{model_name}.onnx")
        tmp_onnx_dir = os.path.join(export_dir, "onnx_tmp")
        tmp_onnx_path = os.path.join(tmp_onnx_dir, f"{model_name}.onnx")
        os.makedirs(tmp_onnx_dir, exist_ok=True)
        torch.onnx.export(
            self.model,
            (self.sample_inputs,),
            tmp_onnx_path,
            input_names=self.input_names,
            output_names=self.output_names,
            dynamic_axes=self.dynamic_axes,
        )

        model = onnx.load(tmp_onnx_path, load_external_data=False)
        transform_kwargs = {
            "onnx_base_dir": tmp_onnx_dir,
            "model_name": model_name,
            "adapter_name": self.model.active_adapter,
        }
        for transform in self.onnx_transforms:
            model, transformed = transform.apply(model, **transform_kwargs)
        onnx.save(model, self.onnx_path)

        shutil.rmtree(tmp_onnx_dir)
        return self.onnx_path

    def compile(self, **kwargs) -> str:
        # Base class
        pass

    def run_pytorch(self, inputs: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        # Base class
        return dict(self.model(inputs))

    def run_ort(self, inputs: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        # Base class
        pass

    def run_cloud_ai_100(self, inputs: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        # Base class
        pass

    def generate(self, inputs: Dict[str, np.ndarray], streamer) -> np.ndarray:
        pass

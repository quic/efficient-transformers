# onnx_utils.py

from typing import Dict

import numpy as np
import onnx
import torch

from .onnx_transforms import FixTrainingOps, FP16Clip, LoraAdapters, SplitWeights


class ONNXModel:
    def __init__(self, model_path: str):
        self.model = onnx.load(model_path)

    def export(self, pytorch_model: torch.nn.Module, inputs: Dict[str, torch.Tensor], output_path: str):
        torch.onnx.export(
            pytorch_model,
            (dict(inputs),),
            output_path,
            input_names=list(inputs.keys()),
            dynamic_axes={name: {0: "batch_size", 1: "seq_len"} for name in inputs.keys()},
            opset_version=15,
            do_constant_folding=False,
            training=torch.onnx.TrainingMode.TRAINING,
        )
        self.model = onnx.load(output_path)

    def modify(self, trainable_params: set, frozen_params: set):
        transforms = [FP16Clip, SplitWeights, LoraAdapters, FixTrainingOps]
        for transform in transforms:
            self.model = transform.apply(self.model)

        # Additional modifications specific to trainable and frozen params
        self._handle_params(trainable_params, frozen_params)

    def _handle_params(self, trainable_params: set, frozen_params: set):
        # Implement logic to handle trainable and frozen parameters
        pass

    def validate(self, inputs: Dict[str, np.ndarray]):
        # Implement ONNX model validation here
        pass

    def save(self, output_path: str):
        onnx.save(self.model, output_path)

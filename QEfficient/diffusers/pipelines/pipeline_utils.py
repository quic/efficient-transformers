# -----------------------------------------------------------------------------
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
#
# ----------------------------------------------------------------------------

import copy
import hashlib

import torch
import torch.nn as nn

from QEfficient.base.modeling_qeff import QEFFBaseModel
from QEfficient.base.onnx_transforms import FP16ClipTransform, SplitTensorsTransform
from QEfficient.diffusers.models.pytorch_transforms import AttentionTransform, CustomOpsTransform, OnnxFunctionTransform
from QEfficient.transformers.models.pytorch_transforms import (
    T5ModelTransform,
)
from QEfficient.utils import constants
from QEfficient.utils.cache import to_hashable


class QEffTextEncoder(QEFFBaseModel):
    _pytorch_transforms = [CustomOpsTransform, T5ModelTransform]
    _onnx_transforms = [FP16ClipTransform, SplitTensorsTransform]
    """
    QEffTextEncoder is a wrapper class for text encoder models that provides ONNX export and compilation capabilities.

    This class extends QEFFBaseModel to handle text encoder models (like T5EncoderModel) with specific
    transformations and optimizations for efficient inference on Qualcomm AI hardware.
    """

    def __init__(self, model: nn.modules):
        super().__init__(model)
        self.model = copy.deepcopy(model)

    def get_onnx_config(self):
        bs = constants.ONNX_EXPORT_EXAMPLE_BATCH_SIZE
        seq_len = self.tokenizer.model_max_length

        example_inputs = {
            "input_ids": torch.zeros((bs, seq_len), dtype=torch.int64),
        }

        dynamic_axes = {"input_ids": {0: "batch_size", 1: "seq_len"}}

        output_names = ["pooler_output", "last_hidden_state"]
        if self.model.__class__.__name__ == "T5EncoderModel":
            output_names = ["last_hidden_state"]
        else:
            example_inputs["output_hidden_states"] = (True,)

        return example_inputs, dynamic_axes, output_names

    def export(self, inputs, output_names, dynamic_axes, export_dir=None):
        return self._export(inputs, output_names, dynamic_axes, export_dir)

    def get_specializations(
        self,
        batch_size: int,
        seq_len: int,
    ):
        specializations = [
            {"batch_size": batch_size, "seq_len": seq_len},
        ]

        return specializations

    def compile(
        self,
        compile_dir,
        compile_only,
        specializations,
        convert_to_fp16,
        mxfp6_matmul,
        mdp_ts_num_devices,
        aic_num_cores,
        custom_io,
        **compiler_options,
    ) -> str:
        return self._compile(
            compile_dir=compile_dir,
            compile_only=compile_only,
            specializations=specializations,
            convert_to_fp16=convert_to_fp16,
            mxfp6_matmul=mxfp6_matmul,
            mdp_ts_num_devices=mdp_ts_num_devices,
            aic_num_cores=aic_num_cores,
            custom_io=custom_io,
            **compiler_options,
        )

    @property
    def model_hash(self) -> str:
        # Compute the hash with: model_config, continuous_batching, transforms
        mhash = hashlib.sha256()
        mhash.update(to_hashable(self.model.config.to_diff_dict()))
        mhash.update(to_hashable(self._transform_names()))
        mhash = mhash.hexdigest()[:16]
        return mhash

    @property
    def model_name(self) -> str:
        mname = self.model.__class__.__name__
        if mname.startswith("QEff") or mname.startswith("QEFF"):
            mname = mname[4:]
        return mname

    @property
    def get_model_config(self) -> dict:
        return self.model.model.vision_model.config.__dict__


class QEffUNet(QEFFBaseModel):
    _pytorch_transforms = [CustomOpsTransform]
    _onnx_transforms = [FP16ClipTransform, SplitTensorsTransform]

    """
    QEffUNet is a wrapper class for UNet models that provides ONNX export and compilation capabilities.

    This class extends QEFFBaseModel to handle UNet models with specific transformations and optimizations
    for efficient inference on Qualcomm AI hardware. It is commonly used in diffusion models for image
    generation tasks.
    """

    def __init__(self, model: nn.modules):
        super().__init__(model.unet)
        self.model = model.unet

    def export(self, inputs, output_names, dynamic_axes, export_dir=None):
        return self._export(inputs, output_names, dynamic_axes, export_dir)

    def compile(
        self,
        compile_dir,
        compile_only,
        specializations,
        convert_to_fp16,
        mxfp6_matmul,
        mdp_ts_num_devices,
        aic_num_cores,
        custom_io,
        **compiler_options,
    ) -> str:
        return self._compile(
            compile_dir=compile_dir,
            compile_only=compile_only,
            specializations=specializations,
            convert_to_fp16=convert_to_fp16,
            mxfp6_matmul=mxfp6_matmul,
            mdp_ts_num_devices=mdp_ts_num_devices,
            aic_num_cores=aic_num_cores,
            custom_io=custom_io,
            **compiler_options,
        )

    @property
    def model_hash(self) -> str:
        # Compute the hash with: model_config, continuous_batching, transforms
        mhash = hashlib.sha256()
        mhash.update(to_hashable(dict(self.model.config)))
        mhash.update(to_hashable(self._transform_names()))
        mhash = mhash.hexdigest()[:16]
        return mhash

    @property
    def model_name(self) -> str:
        mname = self.model.__class__.__name__
        if mname.startswith("QEff") or mname.startswith("QEFF"):
            mname = mname[4:]
        return mname

    @property
    def get_model_config(self) -> dict:
        return self.model.model.vision_model.config.__dict__


class QEffVAE(QEFFBaseModel):
    _pytorch_transforms = [CustomOpsTransform]
    _onnx_transforms = [FP16ClipTransform, SplitTensorsTransform]

    """
    QEffVAE is a wrapper class for Variational Autoencoder (VAE) models that provides ONNX export and compilation capabilities.

    This class extends QEFFBaseModel to handle VAE models with specific transformations and optimizations
    for efficient inference on Qualcomm AI hardware. VAE models are commonly used in diffusion pipelines
    for encoding images to latent space and decoding latent representations back to images.
    """

    def __init__(self, model: nn.modules, type: str):
        super().__init__(model.vae)
        self.model = copy.deepcopy(model.vae)
        self.type = type

    def get_onnx_config(self):
        # VAE decode
        bs = constants.ONNX_EXPORT_EXAMPLE_BATCH_SIZE
        example_inputs = {
            "latent_sample": torch.randn(bs, 16, 64, 64),
            "return_dict": False,
        }

        output_names = ["sample"]

        dynamic_axes = {
            "latent_sample": {0: "batch_size", 1: "channels", 2: "height", 3: "width"},
        }
        return example_inputs, dynamic_axes, output_names

    def export(self, inputs, output_names, dynamic_axes, export_dir=None):
        return self._export(inputs, output_names, dynamic_axes, export_dir)

    def get_specializations(
        self,
        batch_size: int,
    ):
        sepcializations = [
            {
                "batch_size": batch_size,
                "channels": 16,
                "height": 128,
                "width": 128,
            }
        ]
        return sepcializations

    def compile(
        self,
        compile_dir,
        compile_only,
        specializations,
        convert_to_fp16,
        mxfp6_matmul,
        mdp_ts_num_devices,
        aic_num_cores,
        custom_io,
        **compiler_options,
    ) -> str:
        return self._compile(
            compile_dir=compile_dir,
            compile_only=compile_only,
            specializations=specializations,
            convert_to_fp16=convert_to_fp16,
            mxfp6_matmul=mxfp6_matmul,
            mdp_ts_num_devices=mdp_ts_num_devices,
            aic_num_cores=aic_num_cores,
            custom_io=custom_io,
            **compiler_options,
        )

    @property
    def model_hash(self) -> str:
        # Compute the hash with: model_config, continuous_batching, transforms
        mhash = hashlib.sha256()
        mhash.update(to_hashable(dict(self.model.config)))
        mhash.update(to_hashable(self._transform_names()))
        mhash.update(to_hashable(self.type))
        mhash = mhash.hexdigest()[:16]
        return mhash

    @property
    def model_name(self) -> str:
        mname = self.model.__class__.__name__
        if mname.startswith("QEff") or mname.startswith("QEFF"):
            mname = mname[4:]
        return mname

    @property
    def get_model_config(self) -> dict:
        return self.model.model.vision_model.config.__dict__


class QEffSafetyChecker(QEFFBaseModel):
    _pytorch_transforms = [CustomOpsTransform]
    _onnx_transforms = [FP16ClipTransform, SplitTensorsTransform]

    """
    QEffSafetyChecker is a wrapper class for safety checker models that provides ONNX export and compilation capabilities.

    This class extends QEFFBaseModel to handle safety checker models with specific transformations and optimizations
    for efficient inference on Qualcomm AI hardware. Safety checker models are commonly used in diffusion pipelines
    to filter out potentially harmful or inappropriate generated content.
    """

    def __init__(self, model: nn.modules):
        super().__init__(model.vae)
        self.model = model.safety_checker

    def export(self, inputs, output_names, dynamic_axes, export_dir=None):
        return self._export(inputs, output_names, dynamic_axes, export_dir)

    def compile(
        self,
        compile_dir,
        compile_only,
        specializations,
        convert_to_fp16,
        mxfp6_matmul,
        mdp_ts_num_devices,
        aic_num_cores,
        custom_io,
        **compiler_options,
    ) -> str:
        return self._compile(
            compile_dir=compile_dir,
            compile_only=compile_only,
            specializations=specializations,
            convert_to_fp16=convert_to_fp16,
            mxfp6_matmul=mxfp6_matmul,
            mdp_ts_num_devices=mdp_ts_num_devices,
            aic_num_cores=aic_num_cores,
            custom_io=custom_io,
            **compiler_options,
        )

    @property
    def model_hash(self) -> str:
        # Compute the hash with: model_config, continuous_batching, transforms
        mhash = hashlib.sha256()
        mhash.update(to_hashable(self.model.config.to_diff_dict()))
        mhash.update(to_hashable(self._transform_names()))
        mhash = mhash.hexdigest()[:16]
        return mhash

    @property
    def model_name(self) -> str:
        mname = self.model.__class__.__name__
        if mname.startswith("QEff") or mname.startswith("QEFF"):
            mname = mname[4:]
        return mname

    @property
    def get_model_config(self) -> dict:
        return self.model.model.vision_model.config.__dict__


class QEffSD3Transformer2DBaseModel(QEFFBaseModel):
    _pytorch_transforms = [AttentionTransform, CustomOpsTransform]
    _onnx_transforms = [FP16ClipTransform, SplitTensorsTransform]

    """
    QEffSD3Transformer2DModel is a wrapper class for Stable Diffusion 3 Transformer2D models that provides ONNX export and compilation capabilities.

    This class extends QEFFBaseModel to handle SD3 Transformer2D models with specific transformations and optimizations
    for efficient inference on Qualcomm AI hardware. It is designed for the newer Stable Diffusion 3 architecture
    that uses transformer-based diffusion models instead of traditional UNet architectures.
    """

    def __init__(self, model: nn.modules, use_onnx_function):
        super().__init__(model)
        if use_onnx_function:
            self._pytorch_transforms.append(OnnxFunctionTransform)
            model, _ = OnnxFunctionTransform.apply(model)
        self.model = model

    def get_onnx_config(self):
        example_inputs = {
            "hidden_states": torch.randn(
                2,
                self.model.config.in_channels,
                self.model.config.sample_size,
                self.model.config.sample_size,
            ),
            "encoder_hidden_states": torch.randn(2, 333, self.model.config.joint_attention_dim),
            "pooled_projections": torch.randn(2, self.model.config.pooled_projection_dim),
            "timestep": torch.randint(0, 20, (2,), dtype=torch.int64),
        }

        output_names = ["output"]

        dynamic_axes = {
            "hidden_states": {0: "batch_size", 1: "latent_channels", 2: "latent_height", 3: "latent_width"},
            "encoder_hidden_states": {0: "batch_size", 1: "seq_len"},
            "pooled_projections": {0: "batch_size"},
            "timestep": {0: "steps"},
            "output": {0: "batch_size", 1: "latent_channels", 2: "latent_height", 3: "latent_width"},
        }
        return example_inputs, dynamic_axes, output_names

    def export(
        self,
        inputs,
        output_names,
        dynamic_axes,
        export_dir=None,
        export_kwargs=None,
    ):
        return self._export(
            example_inputs=inputs,
            output_names=output_names,
            dynamic_axes=dynamic_axes,
            export_dir=export_dir,
            export_kwargs=export_kwargs,
        )

    def get_specializations(
        self,
        batch_size: int,
        seq_len: int,
    ):
        specializations = [
            {
                "batch_size": 2 * batch_size,
                "latent_channels": 16,
                "latent_height": self.model.config.sample_size,
                "latent_width": self.model.config.sample_size,
                "seq_len": seq_len,
                "steps": 1,
            }
        ]

        return specializations

    def compile(
        self,
        compile_dir,
        compile_only,
        specializations,
        convert_to_fp16,
        mxfp6_matmul,
        mdp_ts_num_devices,
        aic_num_cores,
        custom_io,
        **compiler_options,
    ) -> str:
        return self._compile(
            compile_dir=compile_dir,
            compile_only=compile_only,
            specializations=specializations,
            convert_to_fp16=convert_to_fp16,
            mxfp6_matmul=mxfp6_matmul,
            mdp_ts_num_devices=mdp_ts_num_devices,
            aic_num_cores=aic_num_cores,
            custom_io=custom_io,
            **compiler_options,
        )

    @property
    def model_hash(self) -> str:
        # Compute the hash with: model_config, continuous_batching, transforms
        mhash = hashlib.sha256()
        mhash.update(to_hashable(dict(self.model.config)))
        mhash.update(to_hashable(self._transform_names()))
        mhash = mhash.hexdigest()[:16]
        return mhash

    @property
    def model_name(self) -> str:
        mname = self.model.__class__.__name__
        if mname.startswith("QEff") or mname.startswith("QEFF"):
            mname = mname[4:]
        return mname

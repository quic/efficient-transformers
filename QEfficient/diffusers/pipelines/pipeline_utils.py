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
from QEfficient.diffusers.models.pytorch_transforms import AttentionTransform, CustomOpsTransform
from QEfficient.transformers.models.pytorch_transforms import (
    KVCacheExternalModuleMapperTransform,
    KVCacheTransform,
    T5ModelTransform,
)
from QEfficient.transformers.quantizers.quant_transforms import AwqToMatmulNbitsTransform, GPTQToMatmulNbitsTransform
from QEfficient.utils import constants
from QEfficient.utils.cache import to_hashable


class QEffTextEncoder(QEFFBaseModel):
    _pytorch_transforms = [CustomOpsTransform, T5ModelTransform]
    _onnx_transforms = [FP16ClipTransform, SplitTensorsTransform]

    def __init__(self, model: nn.modules):
        super().__init__(model)
        self.model = copy.deepcopy(model)

    def get_onnx_config(self):
        bs = constants.ONNX_EXPORT_EXAMPLE_BATCH_SIZE
        seq_len = self.tokenizer.model_max_length

        example_inputs = {
            "input_ids": torch.zeros((bs, seq_len), dtype=torch.int64),
            "output_hidden_states": True,
        }

        dynamic_axes = {"input_ids": {0: "batch_size", 1: "seq_len"}}

        output_names = ["pooler_output", "last_hidden_state"]
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
    _pytorch_transforms = [
        AwqToMatmulNbitsTransform,
        GPTQToMatmulNbitsTransform,
        CustomOpsTransform,
        KVCacheTransform,
        KVCacheExternalModuleMapperTransform,
    ]
    _onnx_transforms = [FP16ClipTransform, SplitTensorsTransform]

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
    _pytorch_transforms = [
        AwqToMatmulNbitsTransform,
        GPTQToMatmulNbitsTransform,
        CustomOpsTransform,
    ]
    _onnx_transforms = [FP16ClipTransform, SplitTensorsTransform]

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
    _pytorch_transforms = [
        AwqToMatmulNbitsTransform,
        GPTQToMatmulNbitsTransform,
        CustomOpsTransform,
    ]
    _onnx_transforms = [FP16ClipTransform, SplitTensorsTransform]

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


class QEffSD3Transformer2DModel(QEFFBaseModel):
    _pytorch_transforms = [AttentionTransform, CustomOpsTransform]
    _onnx_transforms = [FP16ClipTransform, SplitTensorsTransform]

    def __init__(self, model: nn.modules):
        super().__init__(model)
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

    def export(self, inputs, output_names, dynamic_axes, export_dir=None):
        return self._export(inputs, output_names, dynamic_axes, export_dir)

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

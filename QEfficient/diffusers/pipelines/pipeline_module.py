# -----------------------------------------------------------------------------
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
#
# ----------------------------------------------------------------------------

import copy

import torch
import torch.nn as nn

from QEfficient.base.modeling_qeff import QEFFBaseModel
from QEfficient.base.onnx_transforms import FP16ClipTransform, SplitTensorsTransform
from QEfficient.diffusers.models.pytorch_transforms import (
    AttentionTransform,
    CustomOpsTransform,
    NormalizationTransform,
    OnnxFunctionTransform,
)
from QEfficient.transformers.models.pytorch_transforms import (
    T5ModelTransform,
)
from QEfficient.utils import constants


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

        example_inputs = {
            "input_ids": torch.zeros((bs, self.model.config.max_position_embeddings), dtype=torch.int64),
        }

        dynamic_axes = {"input_ids": {0: "batch_size", 1: "seq_len"}}
        output_names = ["last_hidden_state", "pooler_output"]

        if self.model.__class__.__name__ == "T5EncoderModel":
            output_names = ["last_hidden_state"]
        else:
            example_inputs["output_hidden_states"] = False

        return example_inputs, dynamic_axes, output_names

    @property
    def get_model_config(self) -> dict:
        """
        Get the model configuration as a dictionary.

        Returns
        -------
        dict
            The configuration dictionary of the underlying HuggingFace model.
        """
        return self.model.config.__dict__

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

    def compile(self, specializations, **compiler_options):
        self._compile(specializations=specializations, **compiler_options)

    @property
    def model_name(self) -> str:
        mname = self.model.__class__.__name__
        if mname.startswith("QEff") or mname.startswith("QEFF"):
            mname = mname[4:]
        return mname


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

    @property
    def get_model_config(self) -> dict:
        """
        Get the model configuration as a dictionary.

        Returns
        -------
        dict
            The configuration dictionary of the underlying HuggingFace model.
        """
        return self.model.config.__dict__

    def compile(self, specializations, **compiler_options):
        self._compile(specializations=specializations, **compiler_options)

    @property
    def model_name(self) -> str:
        mname = self.model.__class__.__name__
        if mname.startswith("QEff") or mname.startswith("QEFF"):
            mname = mname[4:]
        return mname


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

    def get_onnx_config(self, latent_height=32, latent_width=32):
        # VAE decode
        bs = constants.ONNX_EXPORT_EXAMPLE_BATCH_SIZE
        example_inputs = {
            "latent_sample": torch.randn(bs, 16, latent_height, latent_width),
            "return_dict": False,
        }

        output_names = ["sample"]

        dynamic_axes = {
            "latent_sample": {0: "batch_size", 1: "channels", 2: "latent_height", 3: "latent_width"},
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

    def compile(self, specializations, **compiler_options):
        self._compile(specializations=specializations, **compiler_options)

    @property
    def get_model_config(self) -> dict:
        """
        Get the model configuration as a dictionary.

        Returns
        -------
        dict
            The configuration dictionary of the underlying HuggingFace model.
        """
        return self.model.config.__dict__

    @property
    def model_name(self) -> str:
        mname = self.model.__class__.__name__
        if mname.startswith("QEff") or mname.startswith("QEFF"):
            mname = mname[4:]
        return mname


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

    def compile(self, specializations, **compiler_options):
        self._compile(specializations=specializations, **compiler_options)

    @property
    def get_model_config(self) -> dict:
        """
        Get the model configuration as a dictionary.

        Returns
        -------
        dict
            The configuration dictionary of the underlying HuggingFace model.
        """
        return self.model.config.__dict__

    @property
    def model_name(self) -> str:
        mname = self.model.__class__.__name__
        if mname.startswith("QEff") or mname.startswith("QEFF"):
            mname = mname[4:]
        return mname


class QEffFluxTransformerModel(QEFFBaseModel):
    _pytorch_transforms = [AttentionTransform, CustomOpsTransform, NormalizationTransform]
    _onnx_transforms = [FP16ClipTransform, SplitTensorsTransform]
    """
    QEffFluxTransformerModel is a wrapper class for Flux Transformer2D models that provides ONNX export and compilation capabilities.

    This class extends QEFFBaseModel to handle Flux Transformer2D models with specific transformations and optimizations
    for efficient inference on Qualcomm AI hardware. It is designed for the newer Flux transformer architecture
    that uses transformer-based diffusion models instead of traditional UNet architectures.
    """

    def __init__(self, model: nn.modules, use_onnx_function):
        super().__init__(model)
        if use_onnx_function:
            self._pytorch_transforms.append(OnnxFunctionTransform)
            model, _ = OnnxFunctionTransform.apply(model)
        # Ensure the model and all its submodules are on CPU to avoid meta device issues
        self.model = model.to("cpu")

    def get_onnx_config(self, batch_size=1, seq_length=256, cl=4096):
        example_inputs = {
            "hidden_states": torch.randn(batch_size, cl, self.model.config.in_channels, dtype=torch.float32),
            "encoder_hidden_states": torch.randn(
                batch_size, seq_length, self.model.config.joint_attention_dim, dtype=torch.float32
            ),
            "pooled_projections": torch.randn(batch_size, self.model.config.pooled_projection_dim, dtype=torch.float32),
            "timestep": torch.tensor([1.0], dtype=torch.float32),
            "img_ids": torch.randn(cl, 3, dtype=torch.float32),
            "txt_ids": torch.randn(seq_length, 3, dtype=torch.float32),
            "adaln_emb": torch.randn(
                self.model.config.num_layers, 12, 3072, dtype=torch.float32
            ),  # num_layers, #chunks, # Adalan_hidden_dim
            "adaln_single_emb": torch.randn(self.model.config.num_single_layers, 3, 3072, dtype=torch.float32),
            "adaln_out": torch.randn(batch_size, 6144, dtype=torch.float32),
        }

        output_names = ["output"]

        dynamic_axes = {
            "hidden_states": {0: "batch_size", 1: "cl"},
            "encoder_hidden_states": {0: "batch_size", 1: "seq_len"},
            "pooled_projections": {0: "batch_size"},
            "timestep": {0: "steps"},
            "img_ids": {0: "cl"},
        }

        return example_inputs, dynamic_axes, output_names

    @property
    def get_model_config(self) -> dict:
        """
        Get the model configuration as a dictionary.

        Returns
        -------
        dict
            The configuration dictionary of the underlying HuggingFace model.
        """
        return self.model.config.__dict__

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

    def get_specializations(self, batch_size: int, seq_len: int, cl: int):
        specializations = [
            {
                "batch_size": batch_size,
                "stats-batchsize": batch_size,
                "num_layers": self.model.config.num_layers,
                "num_single_layers": self.model.config.num_single_layers,
                "seq_len": seq_len,
                "cl": cl,
                "steps": 1,
            }
        ]

        return specializations

    def compile(self, specializations, **compiler_options):
        self._compile(specializations=specializations, **compiler_options)

    @property
    def model_name(self) -> str:
        mname = self.model.__class__.__name__
        if mname.startswith("QEff") or mname.startswith("QEFF"):
            mname = mname[4:]
        return mname

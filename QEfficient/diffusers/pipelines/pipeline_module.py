# -----------------------------------------------------------------------------
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
#
# ----------------------------------------------------------------------------

import copy
from typing import Dict, List, Tuple

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
    """
    Wrapper for text encoder models with ONNX export and QAIC compilation capabilities.

    This class handles text encoder models (CLIP, T5) with specific transformations and
    optimizations for efficient inference on Qualcomm AI hardware. It applies custom
    PyTorch and ONNX transformations to prepare models for deployment.

    Attributes:
        model (nn.Module): The wrapped text encoder model (deep copy of original)
        _pytorch_transforms (List): PyTorch transformations applied before ONNX export
        _onnx_transforms (List): ONNX transformations applied after export
    """

    _pytorch_transforms = [CustomOpsTransform, T5ModelTransform]
    _onnx_transforms = [FP16ClipTransform, SplitTensorsTransform]

    def __init__(self, model: nn.Module) -> None:
        """
        Initialize the text encoder wrapper.

        Args:
            model (nn.Module): The text encoder model to wrap (CLIP or T5)
        """
        super().__init__(model)
        self.model = copy.deepcopy(model)

    def get_onnx_config(self) -> Tuple[Dict, Dict, List[str]]:
        """
        Generate ONNX export configuration for the text encoder.

        Creates example inputs, dynamic axes specifications, and output names
        tailored to the specific text encoder type (CLIP vs T5).

        Returns:
            Tuple containing:
                - example_inputs (Dict): Sample inputs for ONNX export
                - dynamic_axes (Dict): Specification of dynamic dimensions
                - output_names (List[str]): Names of model outputs
        """
        bs = constants.ONNX_EXPORT_EXAMPLE_BATCH_SIZE

        # Create example input with max sequence length
        example_inputs = {
            "input_ids": torch.zeros((bs, self.model.config.max_position_embeddings), dtype=torch.int64),
        }

        # Define which dimensions can vary at runtime
        dynamic_axes = {"input_ids": {0: "batch_size", 1: "seq_len"}}

        # T5 only outputs hidden states, CLIP outputs both hidden states and pooled output
        if self.model.__class__.__name__ == "T5EncoderModel":
            output_names = ["last_hidden_state"]
        else:
            output_names = ["last_hidden_state", "pooler_output"]
            example_inputs["output_hidden_states"] = False

        return example_inputs, dynamic_axes, output_names

    def export(
        self,
        inputs: Dict,
        output_names: List[str],
        dynamic_axes: Dict,
        export_dir: str = None,
        export_kwargs: Dict = None,
    ) -> str:
        """
        Export the text encoder model to ONNX format.

        Args:
            inputs (Dict): Example inputs for ONNX export
            output_names (List[str]): Names of model outputs
            dynamic_axes (Dict): Specification of dynamic dimensions
            export_dir (str, optional): Directory to save ONNX model
            export_kwargs (Dict, optional): Additional export arguments

        Returns:
            str: Path to the exported ONNX model
        """
        return self._export(
            example_inputs=inputs,
            output_names=output_names,
            dynamic_axes=dynamic_axes,
            export_dir=export_dir,
            export_kwargs=export_kwargs,
        )

    def compile(self, specializations: List[Dict], **compiler_options) -> None:
        """
        Compile the ONNX model for Qualcomm AI hardware.

        Args:
            specializations (List[Dict]): Model specialization configurations
            **compiler_options: Additional compiler options (e.g., num_cores, aic_num_of_activations)
        """
        self._compile(specializations=specializations, **compiler_options)


class QEffUNet(QEFFBaseModel):
    """
    Wrapper for UNet models with ONNX export and QAIC compilation capabilities.

    This class handles UNet models with specific transformations and optimizations
    for efficient inference on Qualcomm AI hardware. UNet is commonly used in
    diffusion models for image generation tasks.

    Attributes:
        model (nn.Module): The wrapped UNet model
        _pytorch_transforms (List): PyTorch transformations applied before ONNX export
        _onnx_transforms (List): ONNX transformations applied after export
    """

    _pytorch_transforms = [CustomOpsTransform]
    _onnx_transforms = [FP16ClipTransform, SplitTensorsTransform]

    def __init__(self, model: nn.Module) -> None:
        """
        Initialize the UNet wrapper.

        Args:
            model (nn.Module): The pipeline model containing the UNet
        """
        super().__init__(model.unet)
        self.model = model.unet

    def export(
        self,
        inputs: Dict,
        output_names: List[str],
        dynamic_axes: Dict,
        export_dir: str = None,
        export_kwargs: Dict = None,
    ) -> str:
        """
        Export the UNet model to ONNX format.

        Args:
            inputs (Dict): Example inputs for ONNX export
            output_names (List[str]): Names of model outputs
            dynamic_axes (Dict): Specification of dynamic dimensions
            export_dir (str, optional): Directory to save ONNX model
            export_kwargs (Dict, optional): Additional export arguments

        Returns:
            str: Path to the exported ONNX model
        """
        return self._export(
            example_inputs=inputs,
            output_names=output_names,
            dynamic_axes=dynamic_axes,
            export_dir=export_dir,
            export_kwargs=export_kwargs,
        )

    def compile(self, specializations: List[Dict], **compiler_options) -> None:
        """
        Compile the ONNX model for Qualcomm AI hardware.

        Args:
            specializations (List[Dict]): Model specialization configurations
            **compiler_options: Additional compiler options
        """
        self._compile(specializations=specializations, **compiler_options)


class QEffVAE(QEFFBaseModel):
    """
    Wrapper for Variational Autoencoder (VAE) models with ONNX export and QAIC compilation.

    This class handles VAE models with specific transformations and optimizations
    for efficient inference on Qualcomm AI hardware. VAE models are used in diffusion
    pipelines for encoding images to latent space and decoding latents back to images.

    Attributes:
        model (nn.Module): The wrapped VAE model (deep copy of original)
        type (str): VAE operation type ("encoder" or "decoder")
        _pytorch_transforms (List): PyTorch transformations applied before ONNX export
        _onnx_transforms (List): ONNX transformations applied after export
    """

    _pytorch_transforms = [CustomOpsTransform]
    _onnx_transforms = [FP16ClipTransform, SplitTensorsTransform]

    def __init__(self, model: nn.Module, type: str) -> None:
        """
        Initialize the VAE wrapper.

        Args:
            model (nn.Module): The pipeline model containing the VAE
            type (str): VAE operation type ("encoder" or "decoder")
        """
        super().__init__(model.vae)
        self.model = copy.deepcopy(model.vae)
        self.type = type

    def get_onnx_config(self, latent_height: int = 32, latent_width: int = 32) -> Tuple[Dict, Dict, List[str]]:
        """
        Generate ONNX export configuration for the VAE decoder.

        Args:
            latent_height (int): Height of latent representation (default: 32)
            latent_width (int): Width of latent representation (default: 32)

        Returns:
            Tuple containing:
                - example_inputs (Dict): Sample inputs for ONNX export
                - dynamic_axes (Dict): Specification of dynamic dimensions
                - output_names (List[str]): Names of model outputs
        """
        bs = constants.ONNX_EXPORT_EXAMPLE_BATCH_SIZE

        # VAE decoder takes latent representation as input
        example_inputs = {
            "latent_sample": torch.randn(bs, 16, latent_height, latent_width),
            "return_dict": False,
        }

        output_names = ["sample"]

        # All dimensions except channels can be dynamic
        dynamic_axes = {
            "latent_sample": {0: "batch_size", 1: "channels", 2: "latent_height", 3: "latent_width"},
        }

        return example_inputs, dynamic_axes, output_names

    def export(
        self,
        inputs: Dict,
        output_names: List[str],
        dynamic_axes: Dict,
        export_dir: str = None,
        export_kwargs: Dict = None,
    ) -> str:
        """
        Export the VAE model to ONNX format.

        Args:
            inputs (Dict): Example inputs for ONNX export
            output_names (List[str]): Names of model outputs
            dynamic_axes (Dict): Specification of dynamic dimensions
            export_dir (str, optional): Directory to save ONNX model
            export_kwargs (Dict, optional): Additional export arguments

        Returns:
            str: Path to the exported ONNX model
        """
        return self._export(
            example_inputs=inputs,
            output_names=output_names,
            dynamic_axes=dynamic_axes,
            export_dir=export_dir,
            export_kwargs=export_kwargs,
        )

    def compile(self, specializations: List[Dict], **compiler_options) -> None:
        """
        Compile the ONNX model for Qualcomm AI hardware.

        Args:
            specializations (List[Dict]): Model specialization configurations
            **compiler_options: Additional compiler options
        """
        self._compile(specializations=specializations, **compiler_options)


class QEffSafetyChecker(QEFFBaseModel):
    """
    Wrapper for safety checker models with ONNX export and QAIC compilation capabilities.

    This class handles safety checker models with specific transformations and optimizations
    for efficient inference on Qualcomm AI hardware. Safety checkers are used in diffusion
    pipelines to filter out potentially harmful or inappropriate generated content.

    Attributes:
        model (nn.Module): The wrapped safety checker model
        _pytorch_transforms (List): PyTorch transformations applied before ONNX export
        _onnx_transforms (List): ONNX transformations applied after export
    """

    _pytorch_transforms = [CustomOpsTransform]
    _onnx_transforms = [FP16ClipTransform, SplitTensorsTransform]

    def __init__(self, model: nn.Module) -> None:
        """
        Initialize the safety checker wrapper.

        Args:
            model (nn.Module): The pipeline model containing the safety checker
        """
        super().__init__(model.safety_checker)
        self.model = model.safety_checker

    def export(
        self,
        inputs: Dict,
        output_names: List[str],
        dynamic_axes: Dict,
        export_dir: str = None,
        export_kwargs: Dict = None,
    ) -> str:
        """
        Export the safety checker model to ONNX format.

        Args:
            inputs (Dict): Example inputs for ONNX export
            output_names (List[str]): Names of model outputs
            dynamic_axes (Dict): Specification of dynamic dimensions
            export_dir (str, optional): Directory to save ONNX model
            export_kwargs (Dict, optional): Additional export arguments

        Returns:
            str: Path to the exported ONNX model
        """
        return self._export(
            example_inputs=inputs,
            output_names=output_names,
            dynamic_axes=dynamic_axes,
            export_dir=export_dir,
            export_kwargs=export_kwargs,
        )

    def compile(self, specializations: List[Dict], **compiler_options) -> None:
        """
        Compile the ONNX model for Qualcomm AI hardware.

        Args:
            specializations (List[Dict]): Model specialization configurations
            **compiler_options: Additional compiler options
        """
        self._compile(specializations=specializations, **compiler_options)


class QEffFluxTransformerModel(QEFFBaseModel):
    """
    Wrapper for Flux Transformer2D models with ONNX export and QAIC compilation capabilities.

    This class handles Flux Transformer2D models with specific transformations and optimizations
    for efficient inference on Qualcomm AI hardware. Flux uses a transformer-based diffusion
    architecture instead of traditional UNet, with dual transformer blocks and adaptive layer
    normalization (AdaLN) for conditioning.

    Attributes:
        model (nn.Module): The wrapped Flux transformer model
        _pytorch_transforms (List): PyTorch transformations applied before ONNX export
        _onnx_transforms (List): ONNX transformations applied after export
    """

    _pytorch_transforms = [AttentionTransform, NormalizationTransform, CustomOpsTransform]
    _onnx_transforms = [FP16ClipTransform, SplitTensorsTransform]

    def __init__(self, model: nn.Module, use_onnx_function: bool) -> None:
        """
        Initialize the Flux transformer wrapper.

        Args:
            model (nn.Module): The Flux transformer model to wrap
            use_onnx_function (bool): Whether to export transformer blocks as ONNX functions
                                     for better modularity and potential optimization
        """

        # Optionally apply ONNX function transform for modular export

        if use_onnx_function:
            model, _ = OnnxFunctionTransform.apply(model)

        super().__init__(model)

        if use_onnx_function:
            self._pytorch_transforms.append(OnnxFunctionTransform)

        # Ensure model is on CPU to avoid meta device issues
        self.model = model.to("cpu")

    def get_onnx_config(
        self, batch_size: int = 1, seq_length: int = 256, cl: int = 4096
    ) -> Tuple[Dict, Dict, List[str]]:
        """
        Generate ONNX export configuration for the Flux transformer.

        Creates example inputs for all Flux-specific inputs including hidden states,
        text embeddings, timestep conditioning, and AdaLN embeddings.

        Args:
            batch_size (int): Batch size for example inputs (default: 1)
            seq_length (int): Text sequence length (default: 256)
            cl (int): Compressed latent dimension (default: 4096)

        Returns:
            Tuple containing:
                - example_inputs (Dict): Sample inputs for ONNX export
                - dynamic_axes (Dict): Specification of dynamic dimensions
                - output_names (List[str]): Names of model outputs
        """
        example_inputs = {
            # Latent representation of the image
            "hidden_states": torch.randn(batch_size, cl, self.model.config.in_channels, dtype=torch.float32),
            # Text embeddings from T5 encoder
            "encoder_hidden_states": torch.randn(
                batch_size, seq_length, self.model.config.joint_attention_dim, dtype=torch.float32
            ),
            # Pooled text embeddings from CLIP encoder
            "pooled_projections": torch.randn(batch_size, self.model.config.pooled_projection_dim, dtype=torch.float32),
            # Diffusion timestep (normalized to [0, 1])
            "timestep": torch.tensor([1.0], dtype=torch.float32),
            # Position IDs for image patches
            "img_ids": torch.randn(cl, 3, dtype=torch.float32),
            # Position IDs for text tokens
            "txt_ids": torch.randn(seq_length, 3, dtype=torch.float32),
            # AdaLN embeddings for dual transformer blocks
            # Shape: [num_layers, 12 chunks (6 for norm1 + 6 for norm1_context), hidden_dim]
            "adaln_emb": torch.randn(
                self.model.config.num_layers,
                12,  # 6 chunks for norm1 + 6 chunks for norm1_context
                3072,  # AdaLN hidden dimension
                dtype=torch.float32,
            ),
            # AdaLN embeddings for single transformer blocks
            # Shape: [num_single_layers, 3 chunks, hidden_dim]
            "adaln_single_emb": torch.randn(
                self.model.config.num_single_layers,
                3,  # 3 chunks for single block norm
                3072,  # AdaLN hidden dimension
                dtype=torch.float32,
            ),
            # Output AdaLN embedding
            # Shape: [batch_size, 2 * hidden_dim] for final projection
            "adaln_out": torch.randn(batch_size, 6144, dtype=torch.float32),  # 2 * 3072
        }

        output_names = ["output"]

        # Define dynamic dimensions for runtime flexibility
        dynamic_axes = {
            "hidden_states": {0: "batch_size", 1: "cl"},
            "encoder_hidden_states": {0: "batch_size", 1: "seq_len"},
            "pooled_projections": {0: "batch_size"},
            "timestep": {0: "steps"},
            "img_ids": {0: "cl"},
        }

        return example_inputs, dynamic_axes, output_names

    def export(
        self,
        inputs: Dict,
        output_names: List[str],
        dynamic_axes: Dict,
        export_dir: str = None,
        export_kwargs: Dict = None,
    ) -> str:
        """
        Export the Flux transformer model to ONNX format.

        Args:
            inputs (Dict): Example inputs for ONNX export
            output_names (List[str]): Names of model outputs
            dynamic_axes (Dict): Specification of dynamic dimensions
            export_dir (str, optional): Directory to save ONNX model
            export_kwargs (Dict, optional): Additional export arguments (e.g., export_modules_as_functions)

        Returns:
            str: Path to the exported ONNX model
        """
        return self._export(
            example_inputs=inputs,
            output_names=output_names,
            dynamic_axes=dynamic_axes,
            export_dir=export_dir,
            export_kwargs=export_kwargs,
        )

    def get_specializations(self, batch_size: int, seq_len: int, cl: int) -> List[Dict]:
        """
        Generate specialization configuration for compilation.

        Specializations define fixed values for certain dimensions to enable
        compiler optimizations specific to the target use case.

        Args:
            batch_size (int): Batch size for inference
            seq_len (int): Text sequence length
            cl (int): Compressed latent dimension

        Returns:
            List[Dict]: Specialization configurations for the compiler
        """
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

    def compile(self, specializations: List[Dict], **compiler_options) -> None:
        """
        Compile the ONNX model for Qualcomm AI hardware.

        Args:
            specializations (List[Dict]): Model specialization configurations
            **compiler_options: Additional compiler options (e.g., num_cores, aic_num_of_activations)
        """
        self._compile(specializations=specializations, **compiler_options)


class QEffQwenImageTransformer2DModel(QEFFBaseModel):
    _pytorch_transforms = [AttentionTransform, CustomOpsTransform]
    _onnx_transforms = [FP16ClipTransform, SplitTensorsTransform]

    """
    QEffQwenImageTransformer2DModel is a wrapper class for QwenImage Transformer2D models that provides ONNX export and compilation capabilities.

    This class extends QEFFBaseModel to handle QwenImage Transformer2D models with specific transformations and optimizations
    for efficient inference on Qualcomm AI hardware. It is designed for the QwenImage architecture that uses
    transformer-based diffusion models with unique latent packing and attention mechanisms.
    """

    def __init__(self, model: nn.modules):
        super().__init__(model)
        self.model = model

    def get_onnx_config(self):
        bs = constants.ONNX_EXPORT_EXAMPLE_BATCH_SIZE

        # For testing purpose I have set this to constant values from the original models
        latent_seq_len = 6032
        text_seq_len = 126
        hidden_dim = 64
        encoder_hidden_dim = 3584
        example_inputs = {
            "hidden_states": torch.randn(bs, latent_seq_len, hidden_dim, dtype=torch.float32),
            "encoder_hidden_states": torch.randn(bs, text_seq_len, encoder_hidden_dim, dtype=torch.float32),
            "encoder_hidden_states_mask": torch.ones(bs, text_seq_len, dtype=torch.int64),
            "timestep": torch.tensor([1000.0], dtype=torch.float32),
            "frame": torch.tensor([1], dtype=torch.int64),
            "height": torch.tensor([58], dtype=torch.int64),
            "width": torch.tensor([104], dtype=torch.int64),
            "txt_seq_lens": torch.tensor([126], dtype=torch.int64),
        }

        output_names = ["output"]

        dynamic_axes = {
            "hidden_states": {0: "batch_size", 1: "latent_seq_len"},
            "encoder_hidden_states": {0: "batch_size", 1: "text_seq_len"},
            "encoder_hidden_states_mask": {0: "batch_size", 1: "text_seq_len"},
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
        latent_seq_len: int,
        text_seq_len: int,
    ):
        specializations = [
            {
                "batch_size": batch_size,
                "latent_seq_len": latent_seq_len,
                "text_seq_len": text_seq_len,
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
    def model_name(self) -> str:
        mname = self.model.__class__.__name__
        if mname.startswith("QEff") or mname.startswith("QEFF"):
            mname = mname[4:]
        return mname

    @property
    def get_model_config(self) -> dict:
        return self.model.config.__dict__

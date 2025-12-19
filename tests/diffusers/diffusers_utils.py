# -----------------------------------------------------------------------------
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------

"""
Common utilities for diffusion pipeline testing.
Provides essential functions for MAD validation, image validation
hash verification, and other testing utilities.
"""

import os
from typing import Any, Dict, Tuple, Union

import numpy as np
import torch
from PIL import Image


class DiffusersTestUtils:
    """Essential utilities for diffusion pipeline testing"""

    @staticmethod
    def validate_image_generation(
        image: Image.Image, expected_size: Tuple[int, int], min_variance: float = 1.0
    ) -> Dict[str, Any]:
        """
        Validate generated image properties.
        Args:
            image: Generated PIL Image
            expected_size: Expected (width, height) tuple
            min_variance: Minimum pixel variance to ensure image is not blank

        Returns:
            Dict containing validation results
        Raises:
            AssertionError: If image validation fails
        """
        # Basic image validation
        assert isinstance(image, Image.Image), f"Expected PIL Image, got {type(image)}"
        assert image.size == expected_size, f"Expected size {expected_size}, got {image.size}"
        assert image.mode in ["RGB", "RGBA"], f"Unexpected image mode: {image.mode}"

        # Variance check (ensure image is not blank)
        img_array = np.array(image)
        image_variance = float(img_array.std())
        assert image_variance > min_variance, f"Generated image appears blank (variance: {image_variance:.2f})"

        return {
            "size": image.size,
            "mode": image.mode,
            "variance": image_variance,
            "mean_pixel_value": float(img_array.mean()),
            "min_pixel": int(img_array.min()),
            "max_pixel": int(img_array.max()),
            "valid": True,
        }

    @staticmethod
    def check_file_exists(file_path: str, file_type: str = "file") -> bool:
        """
        Check if file exists and log result.
        Args:
            file_path: Path to check
            file_type: Description of file type for logging
        Returns:
            bool: True if file exists
        """
        exists = os.path.exists(file_path)
        print(f"file exist: {exists}; {file_type}: {file_path}")
        return exists

    @staticmethod
    def print_test_header(title: str, config: Dict[str, Any]) -> None:
        """
        Print formatted test header with configuration details.

        Args:
            title: Test title
            config: Test configuration dictionary
        """
        print(f"\n{'=' * 80}")
        print(f"{title}")
        print(f"{'=' * 80}")

        if "model_setup" in config:
            setup = config["model_setup"]
            for k, v in setup.items():
                print(f"{k} : {v}")

        if "functional_testing" in config:
            func = config["functional_testing"]
            print(f"Test Prompt: {func.get('test_prompt', 'N/A')}")
            print(f"Inference Steps: {func.get('num_inference_steps', 'N/A')}")
            print(f"Guidance Scale: {func.get('guidance_scale', 'N/A')}")

        print(f"{'=' * 80}")


class MADValidator:
    """Specialized class for MAD validation - always enabled, always reports, always fails on exceed"""

    def __init__(self, tolerances: Dict[str, float] = None):
        """
        Initialize MAD validator.
        MAD validation is always enabled, always reports values, and always fails if tolerance is exceeded.

        Args:
            tolerances: Dictionary of module_name -> tolerance mappings
        """
        self.tolerances = tolerances
        self.results = {}

    def calculate_mad(
        self, tensor1: Union[torch.Tensor, np.ndarray], tensor2: Union[torch.Tensor, np.ndarray]
    ) -> float:
        """
        Calculate Max Absolute Deviation between two tensors.

        Args:
            tensor1: First tensor (PyTorch or NumPy)
            tensor2: Second tensor (PyTorch or NumPy)

        Returns:
            float: Maximum absolute difference between tensors
        """
        if isinstance(tensor1, torch.Tensor):
            tensor1 = tensor1.detach().numpy()
        if isinstance(tensor2, torch.Tensor):
            tensor2 = tensor2.detach().numpy()

        return float(np.max(np.abs(tensor1 - tensor2)))

    def validate_module_mad(
        self,
        pytorch_output: Union[torch.Tensor, np.ndarray],
        qaic_output: Union[torch.Tensor, np.ndarray],
        module_name: str,
        step_info: str = "",
    ) -> bool:
        """
        Validate MAD for a specific module.
        Always validates, always reports, always fails if tolerance exceeded.

        Args:
            pytorch_output: PyTorch reference output
            qaic_output: QAIC inference output
            module_name: Name of the module
            step_info: Additional step information for logging

        Returns:
            bool: True if validation passed

        Raises:
            AssertionError: If MAD exceeds tolerance
        """
        mad_value = self.calculate_mad(pytorch_output, qaic_output)

        # Always report MAD value
        step_str = f" {step_info}" if step_info else ""
        print(f"{module_name.upper()} MAD{step_str}: {mad_value:.8f}")

        # Always validate - fail if tolerance exceeded
        tolerance = self.tolerances.get(module_name, 1e-2)
        if mad_value > tolerance:
            raise AssertionError(f"{module_name} MAD {mad_value:.6f} exceeds tolerance {tolerance:.6f}")

        # Store result
        if module_name not in self.results:
            self.results[module_name] = []
        self.results[module_name].append({"mad": mad_value, "step_info": step_info, "tolerance": tolerance})
        return True

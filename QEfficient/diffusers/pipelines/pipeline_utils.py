# -----------------------------------------------------------------------------
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
#
# ----------------------------------------------------------------------------

import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Union

import numpy as np
import PIL.Image
from tqdm import tqdm

from QEfficient.utils._utils import load_json
from QEfficient.utils.logging_utils import logger


def calculate_compressed_latent_dimension(height: int, width: int, vae_scale_factor: int) -> int:
    """
    Calculate the compressed latent dimension.
    Args:
        height (int): Target image height in pixels
        width (int): Target image width in pixels
        vae_scale_factor (int): VAE downsampling factor (typically 8 for Flux)

    Returns:
        int: Compressed latent dimension (cl) for transformer input buffer allocation
    """
    latent_height = height // vae_scale_factor
    latent_width = width // vae_scale_factor
    # cl = compressed latent dimension (divided by 4 for Flux's 2x2 packing)
    cl = (latent_height * latent_width) // 4
    return cl, latent_height, latent_width


def config_manager(cls, config_source: Optional[str] = None):
    """
    JSON-based compilation configuration manager for diffusion pipelines.

    Supports loading configuration from JSON files only. Automatically detects
    model type and handles model-specific requirements.
    Initialize the configuration manager.

    Args:
        config_source: Path to JSON configuration file. If None, uses default config.
    """
    if config_source is None:
        config_source = cls.get_default_config_path()

    if not isinstance(config_source, str):
        raise ValueError("config_source must be a path to JSON configuration file")

    # Direct use of load_json utility - no wrapper needed
    if not os.path.exists(config_source):
        raise FileNotFoundError(f"Configuration file not found: {config_source}")

    cls.custom_config = load_json(config_source)


def set_module_device_ids(cls):
    """
    Set device IDs for each module based on the custom configuration.

    Iterates through all modules in the pipeline and assigns device IDs
    from the configuration file to each module's device_ids attribute.
    """
    config_modules = cls.custom_config["modules"]
    for module_name, module_obj in cls.modules.items():
        module_obj.device_ids = config_modules[module_name]["execute"]["device_ids"]


def compile_modules_parallel(
    modules: Dict[str, Any],
    config: Dict[str, Any],
    specialization_updates: Dict[str, Dict[str, Any]] = None,
) -> None:
    """
    Compile multiple pipeline modules in parallel using ThreadPoolExecutor.

    Args:
        modules: Dictionary of module_name -> module_object pairs to compile
        config: Configuration dictionary containing module-specific compilation settings
        specialization_updates: Optional dictionary of module_name -> specialization_updates
                               to apply dynamic values (e.g., image dimensions)
    """

    def _prepare_and_compile(module_name: str, module_obj: Any) -> None:
        """Prepare specializations and compile a single module."""
        specializations = config["modules"][module_name]["specializations"].copy()
        compile_kwargs = config["modules"][module_name]["compilation"]

        if specialization_updates and module_name in specialization_updates:
            specializations.update(specialization_updates[module_name])

        module_obj.compile(specializations=[specializations], **compile_kwargs)

    # Execute compilations in parallel
    with ThreadPoolExecutor(max_workers=len(modules)) as executor:
        futures = {executor.submit(_prepare_and_compile, name, obj): name for name, obj in modules.items()}

        with tqdm(total=len(futures), desc="Compiling modules", unit="module") as pbar:
            for future in as_completed(futures):
                try:
                    future.result()
                except Exception as e:
                    logger.error(f"Compilation failed for {futures[future]}: {e}")
                    raise
                pbar.update(1)


def compile_modules_sequential(
    modules: Dict[str, Any],
    config: Dict[str, Any],
    specialization_updates: Dict[str, Dict[str, Any]] = None,
) -> None:
    """
    Compile multiple pipeline modules sequentially.

    This function provides a generic way to compile diffusion pipeline modules
    sequentially, which is the default behavior for backward compatibility.

    Args:
        modules: Dictionary of module_name -> module_object pairs to compile
        config: Configuration dictionary containing module-specific compilation settings
        specialization_updates: Optional dictionary of module_name -> specialization_updates
                               to apply dynamic values (e.g., image dimensions)

    """
    for module_name, module_obj in tqdm(modules.items(), desc="Compiling modules", unit="module"):
        module_config = config["modules"]
        specializations = module_config[module_name]["specializations"].copy()
        compile_kwargs = module_config[module_name]["compilation"]

        # Apply dynamic specialization updates if provided
        if specialization_updates and module_name in specialization_updates:
            specializations.update(specialization_updates[module_name])

        # Compile the module to QPC format
        module_obj.compile(specializations=[specializations], **compile_kwargs)


@dataclass(frozen=True)
class ModulePerf:
    """
    Data class to store performance metrics for a pipeline module.

    Attributes:
        module_name: Name of the pipeline module (e.g., 'text_encoder', 'transformer', 'vae_decoder')
        perf: Performance metric in seconds. Can be a single float for modules that run once,
              or a list of floats for modules that run multiple times (e.g., transformer steps)
    """

    module_name: str
    perf: int


@dataclass(frozen=True)
class QEffPipelineOutput:
    """
    Data class to store the output of a QEfficient diffusion pipeline.

    Attributes:
        pipeline_module: List of ModulePerf objects containing performance metrics for each module
        images: Generated images as either a list of PIL Images or numpy array
    """

    pipeline_module: list[ModulePerf]
    images: Union[List[PIL.Image.Image], np.ndarray]

    def __repr__(self):
        output_str = "=" * 60 + "\n"
        output_str += "QEfficient Diffusers Pipeline Inference Report\n"
        output_str += "=" * 60 + "\n\n"

        # Module-wise inference times
        output_str += "Module-wise Inference Times:\n"
        output_str += "-" * 60 + "\n"

        # Calculate E2E time while iterating
        e2e_time = 0
        for module_perf in self.pipeline_module:
            module_name = module_perf.module_name
            inference_time = module_perf.perf

            # Add to E2E time
            e2e_time += sum(inference_time) if isinstance(inference_time, list) else inference_time

            # Format module name for display
            display_name = module_name.replace("_", " ").title()

            # Handle transformer specially as it has a list of times
            if isinstance(inference_time, list) and len(inference_time) > 0:
                total_time = sum(inference_time)
                avg_time = total_time / len(inference_time)
                output_str += f"  {display_name:25s} {total_time:.4f} s\n"
                output_str += f"    - Total steps: {len(inference_time)}\n"
                output_str += f"    - Average per step:    {avg_time:.4f} s\n"
                output_str += f"    - Min step time:       {min(inference_time):.4f} s\n"
                output_str += f"    - Max step time:       {max(inference_time):.4f} s\n"
            else:
                # Single inference time value
                output_str += f"  {display_name:25s} {inference_time:.4f} s\n"

        output_str += "-" * 60 + "\n\n"

        # Print E2E time after all modules
        output_str += f"End-to-End Inference Time: {e2e_time:.4f} s\n\n"
        output_str += "=" * 60 + "\n"

        return output_str


# List of module name that require special handling during export
# when use_onnx_subfunctions is enabled
ONNX_SUBFUNCTION_MODULE = ["transformer"]

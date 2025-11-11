# -----------------------------------------------------------------------------
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
#
# ----------------------------------------------------------------------------

import os
from dataclasses import dataclass
from typing import TYPE_CHECKING, List, Optional, Union

import numpy as np
import PIL.Image

from QEfficient.utils._utils import load_json

if TYPE_CHECKING:
    from QEfficient.diffusers.pipelines.flux.pipeline_flux import QEFFFluxPipeline


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
    for module_name, module_obj in cls.has_module:
        module_obj.device_ids = config_modules[module_name]["execute"]["device_ids"]


@dataclass
class QEffPipelineOutput:
    pipeline: "QEFFFluxPipeline"
    images: Union[List[PIL.Image.Image], np.ndarray]
    E2E_time: int

    def __repr__(self):
        output_str = "=" * 60 + "\n"
        output_str += "QEfficient Diffusers Pipeline Inference Report\n"
        output_str += "=" * 60 + "\n\n"

        # End-to-End time
        output_str += f"End-to-End Inference Time: {self.E2E_time:.4f} s\n\n"

        # Module-wise inference times
        output_str += "Module-wise Inference Times:\n"
        output_str += "-" * 60 + "\n"

        # Iterate through all modules using has_module
        for module_name, module_obj in self.pipeline.has_module:
            if hasattr(module_obj, "inference_time"):
                inference_time = module_obj.inference_time

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

        output_str += "=" * 60 + "\n"

        return output_str

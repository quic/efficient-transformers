# -----------------------------------------------------------------------------
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------

import os
import warnings

from QEfficient.utils import custom_format_warning

# For faster downloads via hf_transfer
# This code is put above import statements as this needs to be executed before
# hf_transfer is imported (will happen on line 15 via leading imports)
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"
# Placeholder for all non-transformer models registered in QEfficient
import QEfficient.utils.model_registery  # noqa: F401
from QEfficient.utils.logging_utils import logger

# custom warning for the better logging experience
warnings.formatwarning = custom_format_warning


def check_qaic_sdk():
    """Check if QAIC SDK is installed"""
    try:
        import platform
        import sys

        sys.path.append(f"/opt/qti-aic/dev/lib/{platform.machine()}")
        import qaicrt  # noqa: F401

        return True
    except ImportError:
        return False


# Conditionally import QAIC-related modules if the SDK is installed
__version__ = "0.0.1.dev0"

if check_qaic_sdk():
    from QEfficient.base import (
        QEFFAutoModel,
        QEFFAutoModelForCausalLM,
        QEFFAutoModelForImageTextToText,
        QEFFAutoModelForSpeechSeq2Seq,
        QEFFCommonLoader,
    )
    from QEfficient.compile.compile_helper import compile
    from QEfficient.exporter.export_hf_to_cloud_ai_100 import qualcomm_efficient_converter
    from QEfficient.generation.text_generation_inference import cloud_ai_100_exec_kv
    from QEfficient.peft import QEffAutoPeftModelForCausalLM
    from QEfficient.transformers.transform import transform
    
    
    # Imports for the diffusers
    
    from QEfficient.diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion import QEFFStableDiffusionPipeline
    # Users can use QEfficient.export for exporting models to ONNX
    export = qualcomm_efficient_converter

    __all__ = [
        "transform",
        "export",
        "compile",
        "cloud_ai_100_exec_kv",
        "QEFFAutoModel",
        "QEFFAutoModelForCausalLM",
        "QEffAutoPeftModelForCausalLM",
        "QEFFAutoModelForImageTextToText",
        "QEFFAutoModelForSpeechSeq2Seq",
        "QEFFCommonLoader",
        "QEFFStableDiffusionPipeline"
    ]

else:
    logger.warning("QAIC SDK is not installed, eager mode features won't be available!")

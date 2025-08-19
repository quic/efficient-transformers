# -----------------------------------------------------------------------------
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------

import os
import warnings

# For faster downloads via hf_transfer
# This code is put above import statements as this needs to be executed before
# hf_transfer is imported (will happen on line 14 via leading imports)
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"

# Placeholder for all non-transformer models registered in QEfficient
import QEfficient.utils.model_registery  # noqa: F401
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
from QEfficient.utils import custom_format_warning

# custom warning for the better logging experience
warnings.formatwarning = custom_format_warning

# Conditionally import QAIC-related modules if the SDK is installed
__version__ = "0.0.1.dev0"

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
]

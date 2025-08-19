# -----------------------------------------------------------------------------
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------

import os
import warnings

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

# Set environment variable before hf_transfer is imported
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"

# Custom warning formatting
warnings.formatwarning = custom_format_warning

# Exported utility
export = qualcomm_efficient_converter

__version__ = "0.0.1.dev0"

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

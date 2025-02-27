# -----------------------------------------------------------------------------
#
# Copyright (c) 2025 Qualcomm Innovation Center, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------

import os

# For faster downloads via hf_transfer
# This code is put above import statements as this needs to be executed before
# hf_transfer is imported (will happen on line 15 via leading imports)
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"

from transformers import AutoConfig

from QEfficient.transformers.modeling_utils import (
    MODEL_TYPE_TO_CONFIG_CLS_AND_ARCH_CLS,
    get_auto_model_class,
    get_model_class_type_from_model_type,
)
from QEfficient.utils.logging_utils import logger
from transformers import AutoConfig
from QEfficient.transformers.modeling_utils import (
    get_model_class_type_from_model_type,
    get_auto_model_class,
    MODEL_TYPE_TO_CONFIG_CLS_AND_ARCH_CLS
)

# loop over all the model types which are not present in transformers and register them
for model_type, model_cls in MODEL_TYPE_TO_CONFIG_CLS_AND_ARCH_CLS.items():
    # Register the model config class based on the model type. This will be first element in the tuple
    AutoConfig.register(model_type, model_cls[0])

    model_class_type = get_model_class_type_from_model_type(model_type)
    AutoModelClassName = get_auto_model_class(model_class_type, model_cls[1])

    # Register the non transformer library Class and config class using AutoModelClass
    AutoModelClassName.register(model_cls[0], model_cls[1])


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

else:
    logger.warning("QAIC SDK is not installed, eager mode features won't be available!")

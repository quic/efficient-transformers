# -----------------------------------------------------------------------------
#
# Copyright (c) 2024 Qualcomm Innovation Center, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------

try:
    import qaicrt  # noqa: F401

    qaic_sdk_installed = True
except ModuleNotFoundError:
    qaic_sdk_installed = False

if qaic_sdk_installed:
    from QEfficient.base import QEffAutoModel, QEFFAutoModelForCausalLM, QEFFCommonLoader
    from QEfficient.compile.compile_helper import compile
    from QEfficient.exporter.export_hf_to_cloud_ai_100 import qualcomm_efficient_converter
    from QEfficient.generation.text_generation_inference import cloud_ai_100_exec_kv
    from QEfficient.peft import QEffAutoPeftModelForCausalLM
    from QEfficient.transformers.transform import transform

    # Users can use QEfficient.export for exporting models to ONNX
    export = qualcomm_efficient_converter
    __version__ = "0.0.1.dev0"

    __all__ = [
        "transform",
        "export",
        "compile",
        "cloud_ai_100_exec_kv",
        "QEffAutoModel",
        "QEFFAutoModelForCausalLM",
        "QEffAutoPeftModelForCausalLM",
        "QEFFCommonLoader",
    ]
else:
    print("QAIC SDK is not found, skipping QEfficient imports.")

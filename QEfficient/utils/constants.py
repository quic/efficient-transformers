# -----------------------------------------------------------------------------
#
# Copyright (c) 2025 Qualcomm Innovation Center, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------

import os
from dataclasses import dataclass

UTILS_DIR = os.path.dirname(os.path.abspath(__file__))
QEFF_DIR = os.path.dirname(UTILS_DIR)
ROOT_DIR = os.path.dirname(QEFF_DIR)
QEFF_CACHE_DIR_NAME = "qeff_cache"

ONNX_EXPORT_EXAMPLE_BATCH_SIZE = 1
ONNX_EXPORT_EXAMPLE_SEQ_LEN = 32
ONNX_EXPORT_EXAMPLE_FBS = 4
ONNX_EXPORT_EXAMPLE_NLK = 2  # Number of Logits to Keep
ONNX_EXPORT_OPSET = 13
ONNX_EXPORT_MAX_NUM_IMAGES = 1
ONNX_EXPORT_MAX_IMAGE_TILES = 4
ONNX_EXPORT_IMAGE_WIDTH = 560
ONNX_EXPORT_IMAGE_LENGHT = 560
ONNX_EXPORT_IMAGE_DEPTH = 3
ONNX_EXPORT_CTX_LEN = 1024


# Store the qeff_models inside the ~/.cache directory or over-ride with an env variable.
def get_models_dir():
    """
    Determine the directory for storing QEFF models.
    Priority:
    1. Use $XDG_CACHE_HOME/qeff_models if XDG_CACHE_HOME is set.
    2. Use QEFF_HOME if set in environment.
    3. Default to ~/.cache/qeff_models.
    Sets QEFF_MODELS_DIR environment variable if not already set.
    Returns:
        str: Path to the QEFF models directory.
    """
    qeff_cache_home = os.environ.get("QEFF_HOME")
    # Check if XDG_CACHE_HOME is set
    xdg_cache_home = os.environ.get("XDG_CACHE_HOME")
    if qeff_cache_home:
        qeff_models_dir = os.path.join(qeff_cache_home, QEFF_CACHE_DIR_NAME)
    # Check if QEFF_MODELS_DIR is set
    elif xdg_cache_home:
        qeff_models_dir = os.path.join(xdg_cache_home, QEFF_CACHE_DIR_NAME)
    else:
        # Use ~/.cache/qeff_models as the default
        qeff_models_dir = os.path.join(os.path.expanduser("~"), ".cache", QEFF_CACHE_DIR_NAME)

    # Set QEFF_MODELS_DIR environment variable
    return qeff_models_dir


QEFF_MODELS_DIR = get_models_dir()

ONNX_EXPORT_EXAMPLE_BATCH_SIZE = 1
ONNX_EXPORT_EXAMPLE_SEQ_LEN = 32
ONNX_EXPORT_EXAMPLE_FBS = 4
ONNX_EXPORT_EXAMPLE_NLK = 2  # Number of Logits to Keep
ONNX_EXPORT_OPSET = 13

COMPILER = ["/opt/qti-aic/exec/qaic-exec", "-aic-hw", "-aic-hw-version=2.0"]

# InternVL constants
# Fixing the feature size with reference to OpenGVLab/InternVL2_5-1B, OpenGVLab/InternVL2_5-38B and OpenGVLab/InternVL2_5-78B
INTERN_FEATURE_SIZE = 256
INTERN_NUM_PATCHES = 13
INTERN_IMG_SIZE = 448
INTERN_CTX_LEN = 4096
INTERN_PREFILL_SEQ_LEN = INTERN_CTX_LEN - 256  # 4096-256
INTERN_NUM_CHANNELS = 3
INTERN_IMG_CONTEXT_TOKEN = 151667

# Granite Vision Constants
# Fixing the feature size with reference to ibm-granite/granite-vision-3.2-2b
GRANITEVISION_FEATURE_SIZE = 5239
GRANITEVISION_NUM_PATCHES = 10
GRANITEVISION_IMG_SIZE = 384
GRANITEVISION_IMG_SIZE_HEIGHT = 1109
GRANITEVISION_IMG_SIZE_WIDTH = 1610
GRANITEVISION_PIXEL_VALUE_DIM = 5
GRANITEVISION_PREFIL_SEQ_LEN = GRANITEVISION_SEQ_LEN = 5500
GRANITEVISION_CTX_LEN = 6000
GRANITEVISION_NUM_CHANNELS = 3


class Constants:
    # Export Constants.
    SEQ_LEN = 32
    CTX_LEN = 32
    PROMPT_LEN = 8
    INPUT_STR = ["My name is"]
    GB = 2**30
    MAX_QPC_LIMIT = 30
    MAX_RETRIES = 10  # This constant will be used set the maximum number of retry attempts for downloading a model using huggingface_hub snapshot_download
    NUM_SPECULATIVE_TOKENS = 2
    SDK_APPS_XML = "/opt/qti-aic/versions/apps.xml"  # This xml file is parsed to find out the SDK version.


@dataclass
class QnnConstants:
    # QNN PATH to be read from environment variable.
    QNN_SDK_PATH_ENV_VAR_NAME = "QNN_SDK_ROOT"
    QNN_SDK_YAML = "sdk.yaml"

    # QNN Compilation tools
    QAIRT_CONVERTER = "{}/bin/{}/qairt-converter"
    QNN_CONTEXT_BIN = "{}/bin/{}/qnn-context-binary-generator"

    # QNN Libraries required for compilation
    QNN_CONTEXT_LIB_BACKEND = "{}/lib/{}/libQnnAic.so"
    QNN_CONTEXT_LIB_NET_RUN_EXTENSIONS = "{}/lib/{}/libQnnAicNetRunExtensions.so"

    # QNN Compilation target names
    MODEL_NAME = "model"
    QNN_DATA_FORMAT_CONFIG_NAME = "qnn_data_format_config.json"
    CONTEXT_BIN_NAME = "qnngraph.serialized"
    CONTEXT_BIN_QPC_NAME = "programqpc.bin"

    # TARGET System Architecture
    TARGET = "x86_64-linux-clang"  # TODO add support in infer to be override

    # Converter Arguments
    FLOAT_BITWIDTH = 16
    FLOAT_BIAS_BITWIDTH = 32
    CONVERTER_DEFAULT_ARGS = "--preserve_io_datatype --onnx_skip_simplification "

    # Context-Binary-Generator Arguments
    LOG_LEVEL = "error"

    # qnn_compilation_backend default Arguments
    COMPILER_COMPILATION_TARGET = "hardware"
    COMPILER_CONVERT_TO_FP16 = True
    COMPILER_DO_DDR_TO_MULTICAST = True
    COMPILER_HARDWARE_VERSION = "2.0"
    COMPILER_PERF_WARNINGS = False
    COMPILER_PRINT_DDR_STATS = False
    COMPILER_PRINT_PERF_METRICS = False
    COMPILER_RETAINED_STATE = True
    COMPILER_STAT_LEVEL = 10
    COMPILER_STATS_BATCH_SIZE = 1
    COMPILER_TIME_PASSES = False
    GRAPH_NAMES = [f"{MODEL_NAME}_configuration_1", f"{MODEL_NAME}_configuration_2"]
    GRAPH_NAMES_PREFILL_ONLY = [f"{MODEL_NAME}"]

    # qnn_config JSON file supported Keys
    CONVERTER_ARGS_EXTENSION_STR = "converter_args_extension"
    CONTEXT_BIN_ARGS_EXTENSION_STR = "context_binary_generator_args_extension"
    QNN_COMPILATION_BACKEND_STR = "qnn_compilation_backend"
    SKIP_QNN_CONVERTER_STEP_STR = "SKIP_QNN_CONVERTER_STEP"

    IMMUTABLE_CONVERTER_ARGS = [
        "--input_network ",
        "--output_path ",
        "--config ",
        "--float_bias_bitwidth ",
        "--float_bitwidth ",
        "--preserve_io_datatype",
        "--onnx_skip_simplification",
    ]

    IMMUTABLE_CONTEXT_BIN_GEN_ARGS = [
        "--binary_file ",
        "--backend_binary ",
        "--output_dir ",
        "--backend ",
        "--model ",
        "--dlc_path ",
        "--config_file ",
    ]

    QNN_SAMPLE_CONFIG = {
        "converter_args_extension": "--onnx_defer_loading",
        "context_binary_generator_args_extension": "--log_level debug",
        "qnn_compilation_backend": {
            "compiler_enable_depth_first": True,
            "compiler_printDDRStats": False,
            "compiler_printPerfMetrics": False,
        },
        "SKIP_QNN_CONVERTER_STEP": False,
    }

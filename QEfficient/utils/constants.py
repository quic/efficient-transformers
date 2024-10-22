# -----------------------------------------------------------------------------
#
# Copyright (c) 2024 Qualcomm Innovation Center, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------

import os
from dataclasses import dataclass

UTILS_DIR = os.path.dirname(os.path.abspath(__file__))
QEFF_DIR = os.path.dirname(UTILS_DIR)
ROOT_DIR = os.path.dirname(QEFF_DIR)
QEFF_CACHE_DIR_NAME = "qeff_cache"


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


class Constants:
    # Export Constants.
    SEQ_LEN = 32
    CTX_LEN = 32
    PROMPT_LEN = 8
    INPUT_STR = ["My name is"]
    GB = 2**30
    MAX_QPC_LIMIT = 30


@dataclass
class QnnConstants:
    # QNN PATH to be read from environment variable.
    QNN_SDK_PATH_ENV_VAR_NAME = "QNN_SDK_ROOT"

    # QNN Compilation tools
    QAIRT_CONVERTER = "{}/bin/{}/qairt-converter"
    QNN_CONTEXT_BIN = "{}/bin/{}/qnn-context-binary-generator"

    # QNN Libraries required for compilation
    QNN_CONTEXT_LIB_BACKEND = "{}/lib/{}/libQnnAicCC.so"
    QNN_CONTEXT_LIB_MODEL = "{}/lib/{}/libQnnModelDlc.so"
    QNN_CONTEXT_LIB_NET_RUN_EXTENSIONS = "{}/lib/{}/libQnnAicNetRunExtensions.so"

    # QNN Compilation target names
    MODEL_NAME = "model"
    CONTEXT_BIN_NAME = "qnngraph.serialized"
    CONTEXT_BIN_QPC_NAME = "programqpc.bin"

    # TARGET System Architecture
    TARGET = "x86_64-linux-clang"  # TODO add support in infer to be override

    # Convertor Arguments
    FLOAT_BITWIDTH = 16
    FLOAT_BIAS_BITWIDTH = 32
    CONVERTOR_DEFAULT_ARGS = "--keep_int64_inputs --onnx_no_simplification "

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

    # qnn_config JSON file supported Keys
    CONVERTOR_ARGS_EXTENSION_STR = "convertor_args_extension"
    CONTEXT_BIN_ARGS_EXTENSION_STR = "context_binary_generator_args_extension"
    QNN_COMPILATION_BACKEND_STR = "qnn_compilation_backend"
    SKIP_QNN_CONVERTOR_STEP_STR = "SKIP_QNN_CONVERTOR_STEP"

    IMMUTABLE_CONVERTOR_ARGS = [
        "--input_network ",
        "--output_path ",
        "--io_config ",
        "--float_bias_bitwidth ",
        "--float_bitwidth ",
        "--keep_int64_inputs",
        "--onnx_no_simplification",
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

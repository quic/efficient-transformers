# -----------------------------------------------------------------------------
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------

import importlib
import logging
from pathlib import Path

from QEfficient.finetune.utils.logging_utils import logger


def load_module_from_py_file(py_file: str) -> object:
    """
    This method loads a module from a py file which is not in the Python path
    """
    module_name = Path(py_file).name
    loader = importlib.machinery.SourceFileLoader(module_name, py_file)
    spec = importlib.util.spec_from_loader(module_name, loader)
    module = importlib.util.module_from_spec(spec)

    loader.exec_module(module)

    return module


def get_custom_dataset(dataset_config, tokenizer, split: str, context_length=None):
    if not hasattr(dataset_config, "preproc_file"):
        logger.raise_error("Can not find preproc_file key in dataset_config file.", RuntimeError)

    if ":" in dataset_config.preproc_file:
        module_path, func_name = dataset_config.preproc_file.split(":")
    else:
        module_path, func_name = dataset_config.preproc_file, "get_custom_dataset"
        logger.log_rank_zero(
            f"Using '{func_name}' function from "
            f"{dataset_config.preproc_file} as preprocessing function in "
            "dataset preprocessing.",
            logging.WARNING,
        )

    if not module_path.endswith(".py"):
        logger.raise_error(f"Custom dataset preprocessing file {module_path} is not a .py file.", ValueError)

    module_path = Path(module_path)
    if not module_path.is_file():
        logger.raise_error(
            f"Custom dataset file {module_path.as_posix()} does not exist or is not a file.", FileNotFoundError
        )

    module = load_module_from_py_file(module_path.as_posix())
    try:
        return getattr(module, func_name)(dataset_config, tokenizer, split, context_length)
    except AttributeError:
        logger.raise_error(
            f"For custom dataset preprocessing, the method ({func_name}) is not "
            f"present in the file ({module_path.as_posix()}).",
            AttributeError,
        )


def get_data_collator(dataset_processer, dataset_config):
    if not hasattr(dataset_config, "collate_file"):
        logger.log_rank_zero(
            "Can not find collate_file key in dataset_config file. Using the default data collator function instead.",
            logging.WARNING,
        )
        return None

    if ":" in dataset_config.collate_file:
        module_path, func_name = dataset_config.collate_file.split(":")
    else:
        module_path, func_name = dataset_config.collate_file, "get_data_collator"
        logger.log_rank_zero(
            f"Using '{func_name}' function from {dataset_config.collate_file} as collate_fn in dataset preprocessing.",
            logging.WARNING,
        )

    if not module_path.endswith(".py"):
        logger.raise_error(f"Custom dataset collate file {module_path} is not a .py file.", ValueError)

    module_path = Path(module_path)
    if not module_path.is_file():
        logger.raise_error(
            f"Custom dataset collate file {module_path.as_posix()} does not exist or is not a file.", FileNotFoundError
        )

    module = load_module_from_py_file(module_path.as_posix())
    try:
        return getattr(module, func_name)(dataset_processer)
    except AttributeError:
        logger.log_rank_zero(
            f"Can not find the function {func_name} in file "
            f"({module_path.as_posix()}). Using the default data collator "
            "function instead."
        )
        return None

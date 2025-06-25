# -----------------------------------------------------------------------------
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------

import importlib
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
    if ":" in dataset_config.file:
        module_path, func_name = dataset_config.file.split(":")
    else:
        module_path, func_name = dataset_config.file, "get_custom_dataset"

    if not module_path.endswith(".py"):
        logger.raise_runtimeerror(f"Dataset file {module_path} is not a .py file.")

    module_path = Path(module_path)
    if not module_path.is_file():
        logger.raise_runtimeerror(f"Dataset py file {module_path.as_posix()} does not exist or is not a file.")

    module = load_module_from_py_file(module_path.as_posix())
    try:
        return getattr(module, func_name)(dataset_config, tokenizer, split, context_length)
    except AttributeError:
        logger.raise_runtimeerror(
            f"It seems like the given method name ({func_name}) is not present in the dataset .py file ({module_path.as_posix()})."
        )


def get_data_collator(dataset_processer, dataset_config):
    if ":" in dataset_config.file:
        module_path, func_name = dataset_config.file.split(":")
    else:
        module_path, func_name = dataset_config.file, "get_data_collator"

    if not module_path.endswith(".py"):
        logger.raise_runtimeerror(f"Dataset file {module_path} is not a .py file.")

    module_path = Path(module_path)
    if not module_path.is_file():
        logger.raise_runtimeerror(f"Dataset py file {module_path.as_posix()} does not exist or is not a file.")

    module = load_module_from_py_file(module_path.as_posix())
    try:
        return getattr(module, func_name)(dataset_processer)
    except AttributeError:
        logger.log_rank_zero(
            f"Can not find the custom data_collator in the dataset.py file ({module_path.as_posix()})."
        )
        logger.log_rank_zero("Using the default data_collator instead.")
        return None

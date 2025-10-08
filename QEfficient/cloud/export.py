# -----------------------------------------------------------------------------
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------

import argparse
import os
from typing import Optional

from QEfficient.base.common import QEFFCommonLoader
from QEfficient.utils import check_and_assign_cache_dir
from QEfficient.utils.logging_utils import logger

# Specifically for Docker images.
ROOT_DIR = os.path.dirname(os.path.abspath(""))


def get_onnx_model_path(
    model_name: str,
    cache_dir: Optional[str] = None,
    hf_token: Optional[str] = None,
    full_batch_size: Optional[int] = None,
    local_model_dir: Optional[str] = None,
):
    """
    Exports the PyTorch model to ONNX format if a pre-exported file is not found,
    and returns the path to the ONNX model.

    This function loads a Hugging Face model via QEFFCommonLoader, then calls
    its export method to generate the ONNX graph.

    Parameters
    ----------
    model_name : str
        Hugging Face Model Card name (e.g., ``gpt2``).

    Other Parameters
    ----------------
    cache_dir : str, optional
        Cache directory where downloaded HuggingFace files are stored. Default is None.
    hf_token : str, optional
        HuggingFace login token to access private repositories. Default is None.
    full_batch_size : int, optional
        Sets the full batch size to enable continuous batching mode. Default is None.
    local_model_dir : str, optional
        Path to custom model weights and config files. Default is None.

    Returns
    -------
    str
        Path of the generated ONNX graph file.
    """
    logger.info(f"Exporting Pytorch {model_name} model to ONNX...")

    qeff_model = QEFFCommonLoader.from_pretrained(
        pretrained_model_name_or_path=model_name,
        cache_dir=cache_dir,
        hf_token=hf_token,
        full_batch_size=full_batch_size,
        local_model_dir=local_model_dir,
    )
    onnx_model_path = qeff_model.export()
    logger.info(f"Generated onnx_path: {onnx_model_path}")
    return onnx_model_path


def main(
    model_name: str,
    cache_dir: Optional[str] = None,
    hf_token: Optional[str] = None,
    local_model_dir: Optional[str] = None,
    full_batch_size: Optional[int] = None,
) -> None:
    """
    Main function for the QEfficient ONNX export CLI application.

    This function serves as the entry point for exporting a PyTorch model, loaded
    via QEFFCommonLoader, to the ONNX format. It prepares the necessary
    paths and calls `get_onnx_model_path`.

    Parameters
    ----------
    model_name : str
        Hugging Face Model Card name (e.g., ``gpt2``).

    Other Parameters
    ----------------
    cache_dir : str, optional
        Cache directory where downloaded HuggingFace files are stored. Default is None.
    hf_token : str, optional
        HuggingFace login token to access private repositories. Default is None.
    local_model_dir : str, optional
        Path to custom model weights and config files. Default is None.
    full_batch_size : int, optional
        Sets the full batch size to enable continuous batching mode. Default is None.

    Example
    -------
    To export a model from the command line:

    .. code-block:: bash

        python -m QEfficient.cloud.export --model-name gpt2 --cache-dir /path/to/cache

    """
    cache_dir = check_and_assign_cache_dir(local_model_dir, cache_dir)
    get_onnx_model_path(
        model_name=model_name,
        cache_dir=cache_dir,
        hf_token=hf_token,
        full_batch_size=full_batch_size,
        local_model_dir=local_model_dir,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Export script.")
    parser.add_argument("--model_name", "--model-name", required=True, help="HF Model card name/id")
    parser.add_argument(
        "--local-model-dir", "--local_model_dir", required=False, help="Path to custom model weights and config files"
    )
    parser.add_argument(
        "--cache_dir",
        "--cache-dir",
        required=False,
        help="Cache_dir to store the HF files",
    )
    parser.add_argument(
        "--hf-token", "--hf_token", default=None, type=str, required=False, help="HF token id for private HF models"
    )
    parser.add_argument(
        "--full_batch_size",
        "--full-batch-size",
        type=int,
        default=None,
        help="Set full batch size to enable continuous batching mode, default is None",
    )
    args = parser.parse_args()
    main(**args.__dict__)

# -----------------------------------------------------------------------------
#
# Copyright (c)  2023-2024 Qualcomm Innovation Center, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------
from QEfficient.utils.logging_utils import logger
from typing import Union
from transformers import PreTrainedTokenizer, PreTrainedTokenizerFast

from QEfficient.utils._utils import (  # noqa: F401
    get_qpc_dir_name_infer,
    hf_download,
    load_hf_tokenizer,
    login_and_download_hf_lm,
    onnx_exists,
    qpc_exists,
)


def padding_check_and_fix(
    tokenizer: Union[PreTrainedTokenizer, PreTrainedTokenizerFast]
):
    """
    Checks Tokenizer paddding side and pad_token_id viability.
    --------
    
    Tokeinizer: Union[PreTrainedTokenizer, PreTrainedTokenizerFast]. Pass model tokenizer to check and fix.
    """
    
    if tokenizer.padding_side != "left":
        logger.warning("Please use padding_side='left' while initializing the tokenizer")
        tokenizer.padding_side = "left"
    
    if tokenizer.pad_token_id is None:
        # If Pad token is out of range of vocab size
        if tokenizer.eos_token_id < tokenizer.vocab_size:
            tokenizer.pad_token_id = tokenizer.eos_token_id
        else:
            tokenizer.pad_token_id = tokenizer.vocab_size - 1
            
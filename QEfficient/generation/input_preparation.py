# -----------------------------------------------------------------------------
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------

from typing import Dict, List, Tuple, Union

import numpy as np
from transformers import PreTrainedTokenizer, PreTrainedTokenizerFast

from QEfficient.utils.logging_utils import logger


def prepare_tokenizer(tokenizer: Union[PreTrainedTokenizer, PreTrainedTokenizerFast]) -> None:
    """Configure tokenizer padding consistently for prefill generation."""
    if tokenizer.padding_side != "right":
        logger.warning("Please use padding_side='right' while initializing the tokenizer")
        tokenizer.padding_side = "right"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id


def build_prefill_inputs(
    tokenizer: Union[PreTrainedTokenizer, PreTrainedTokenizerFast],
    prompt: Union[str, List[str]],
    prefill_seq_len: int,
) -> Tuple[Dict[str, np.ndarray], np.ndarray, int]:
    """Tokenize and pad host inputs to a whole number of prefill chunks."""
    unpadded_inputs = tokenizer(prompt, return_tensors="np", padding=True)
    position_ids = unpadded_inputs["attention_mask"].sum(1, keepdims=True)
    input_length = unpadded_inputs["input_ids"].shape[1]
    num_chunks = -(input_length // -prefill_seq_len)
    padded_length = num_chunks * prefill_seq_len

    inputs = tokenizer(prompt, return_tensors="np", padding="max_length", max_length=padded_length)
    inputs["position_ids"] = np.where(inputs.pop("attention_mask"), np.arange(padded_length), -1)
    inputs.pop("token_type_ids", None)
    return inputs, position_ids, num_chunks


def slice_prefill_inputs(
    inputs: Dict[str, np.ndarray], chunk_index: int, prefill_seq_len: int
) -> Dict[str, np.ndarray]:
    """Return one prefill chunk while preserving non-sequence inputs."""
    chunk_inputs = inputs.copy()
    start = chunk_index * prefill_seq_len
    end = start + prefill_seq_len
    chunk_inputs["input_ids"] = inputs["input_ids"][:, start:end]
    chunk_inputs["position_ids"] = inputs["position_ids"][:, start:end]
    return chunk_inputs

# -----------------------------------------------------------------------------
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------

from transformers import AutoTokenizer

from QEfficient import QEFFAutoModelForCausalLM
from QEfficient.utils.constants import Constants

model_name = "Qwen/Qwen3-30B-A3B-Instruct-2507"
"""
# For CB inference, set continuous_batching to True and add full_batch_size,mxfp6,mxint8 argument in compile function
# We will use prompt_len=1 for compilation for both cb and non-cb inference
"""

## Activate Compute-Context-Length (CCL) feature by setting ccl_enabled=True when loading the model with from_pretrained().
## Use the optional comp_ctx_lengths argument to provide two lists of context lengths for the prefilling and decoding processes. If comp_ctx_lengths=None, the model will run with its default context length.
##   - The first list, comp_ctx_lengths_prefill, defines the compute-context-length values for the prefilling process.
##           -- The process starts with the first value in the list and gradually increases the context length based on the position_id of the current prompt chunk.
##   - The second list, comp_ctx_lengths_decode, defines the compute-context-length values for the decoding process.
##           -- During decoding, the model selects an appropriate context length from the list based on the input prompt length and cache index.
##           -- It starts from the correct value in the list and increases the context length dynamically when the cache index exceeds the current threshold.

ctx_len = 1024
prefill_seq_len = 1
# In moe models when compiling with prefill_seq_len=1 and non-continuous-batching mode, prefill and decode will share the same ccl specializations.
comp_ctx_lengths_prefill = comp_ctx_lengths_decode = [256, 512, ctx_len]

model = QEFFAutoModelForCausalLM.from_pretrained(
    model_name,
    continuous_batching=False,
    qaic_config={
        "ccl_enabled": True,
    },
)

model.compile(
    prefill_seq_len=prefill_seq_len,
    ctx_len=ctx_len,
    batch_size=1,
    num_cores=16,
    num_devices=4,
    mxfp6_matmul=True,
    mxint8_kv_cache=True,
    mos=1,
    comp_ctx_lengths_prefill=comp_ctx_lengths_prefill,
    comp_ctx_lengths_decode=comp_ctx_lengths_decode,
)

tokenizer = AutoTokenizer.from_pretrained(model_name)
exec_info = model.generate(prompts=Constants.INPUT_STR, tokenizer=tokenizer)

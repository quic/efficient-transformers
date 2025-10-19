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
# For CB inference, set continuous_batching to True and add full_batch_size,mxfp6,mint8 argument in compile function
# We will use prompt_len=1 for compilation for both cb and non-cb inference
"""

ctx_len = 65536
prefill_seq_len = 1
# In moe models when compiling with prefill_seq_len=1 and non-continuous-batching mode, prefill and decode will share the same specializations.
comp_ctx_lengths_prefill = [4096,8192,16384,32768,ctx_len]
comp_ctx_lengths_decode = [4096,8192,16384,32768,ctx_len]

model = QEFFAutoModelForCausalLM.from_pretrained(
    model_name,
    comp_ctx_lengths_prefill=comp_ctx_lengths_prefill,
    comp_ctx_lengths_decode=comp_ctx_lengths_decode,
    ctx_len=ctx_len,
    continuous_batching=False,
    prefill_seq_len=prefill_seq_len,
)
    # prefill_seq_len=prefill_seq_len,
model.compile(
    prefill_seq_len=prefill_seq_len,
    ctx_len=ctx_len,
    batch_size=1,
    num_cores=16,
    num_devices=4,
    mxfp6_matmul=True,
    mxint8_kv_cache=True,
    mos=1,
)
    # mos=1,
tokenizer = AutoTokenizer.from_pretrained(model_name)
exec_info = model.generate(prompts=Constants.INPUT_STR, tokenizer=tokenizer)

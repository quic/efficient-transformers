# -----------------------------------------------------------------------------
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------

from transformers import AutoTokenizer

from QEfficient import QEFFAutoModelForCausalLM
from QEfficient.utils.constants import Constants

model_name = "ibm-research/PowerMoE-3b"
"""
# For CB inference, set continuous_batching to True and add full_batch_size,mxfp6,mint8 argument in compile function
# We will use prompt_len=1 for compilation for both cb and non-cb inference
"""

comp_ctx_lengths = [256, 512, 1024, 2048]  # None

## Prefill_ccl_len shows how many numbers in the comp_ctx_lengths list is related to prefilling and the rest would be for decoding. The default value is 1.
prefill_ccl_len = 2

model = QEFFAutoModelForCausalLM.from_pretrained(
    model_name, comp_ctx_lengths=comp_ctx_lengths, prefill_ccl_len=prefill_ccl_len, continuous_batching=False
)
model.compile(
    prefill_seq_len=1,
    ctx_len=2048,
    full_batch_size=1,
    num_cores=16,
    num_devices=4,
    mxfp6_matmul=False,
    mxint8_kv_cache=False,
)
tokenizer = AutoTokenizer.from_pretrained(model_name)
exec_info = model.generate(prompts=Constants.INPUT_STR, tokenizer=tokenizer, device_id=[16, 17, 18, 19])

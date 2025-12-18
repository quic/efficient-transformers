# -----------------------------------------------------------------------------
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------

from transformers import AutoTokenizer, TextStreamer

from QEfficient import QEFFAutoModelForCausalLM

model_id = "openai/gpt-oss-20b"  # weights are not required to convert to fp32

## Activate Compute-Context-Length (CCL) feature by setting ccl_enabled=True when loading the model with from_pretrained().
## Use the optional comp_ctx_lengths argument to provide two lists of context lengths for the prefilling and decoding processes. If comp_ctx_lengths=None, the model will run with its default context length.
##   - The first list, comp_ctx_lengths_prefill, defines the compute-context-length values for the prefilling process.
##           -- The process starts with the first value in the list and gradually increases the context length based on the position_id of the current prompt chunk.
##   - The second list, comp_ctx_lengths_decode, defines the compute-context-length values for the decoding process.
##           -- During decoding, the model selects an appropriate context length from the list based on the input prompt length and cache index.
##           -- It starts from the correct value in the list and increases the context length dynamically when the cache index exceeds the current threshold.

ctx_len = 4096
# In moe models like gpt-oss, since prefill_seq_len=1 both comp_ctx_lengths_prefill and comp_ctx_lengths_decode can share similar lists.
# Set the list of ccl during prefilling and decoding processes
comp_ctx_lengths_prefill = comp_ctx_lengths_decode = [1024, ctx_len]

qeff_model = QEFFAutoModelForCausalLM.from_pretrained(
    model_id,
    qaic_config={
        "ccl_enabled": True,
    },
)
tokenizer = AutoTokenizer.from_pretrained(model_id)

qpc_path = qeff_model.compile(
    prefill_seq_len=1,  # Currently we can get best perf using PL=1 i.e. decode-only model, prefill optimizations are being worked on.
    ctx_len=ctx_len,
    num_cores=16,
    mxfp6_matmul=True,
    mxint8_kv_cache=True,
    num_devices=4,
    mos=1,
    aic_enable_depth_first=True,
    num_speculative_tokens=None,
    comp_ctx_lengths_prefill=comp_ctx_lengths_prefill,
    comp_ctx_lengths_decode=comp_ctx_lengths_decode,
)
print(f"qpc path is {qpc_path}")
streamer = TextStreamer(tokenizer)
exec_info = qeff_model.generate(
    tokenizer,
    prompts="Who is your creator? and What all you are allowed to do?",
    generation_len=256,
)

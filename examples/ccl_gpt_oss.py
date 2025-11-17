# -----------------------------------------------------------------------------
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------

from transformers import AutoTokenizer, TextStreamer

from QEfficient import QEFFAutoModelForCausalLM

model_id = "openai/gpt-oss-20b"  # weights are not required to convert to fp32

ctx_len = 4096
# In moe models like gpt-oss, since prefill_seq_len=1 both comp_ctx_lengths_prefill and comp_ctx_lengths_decode can share similar lists.
# Set the list of ccl during prefilling process
comp_ctx_lengths_prefill = [512, ctx_len]
# Set the list of ccl during decoding process
comp_ctx_lengths_decode = [512, ctx_len]


qeff_model = QEFFAutoModelForCausalLM.from_pretrained(
    model_id,
    qaic_config={
        "comp_ctx_lengths_prefill": comp_ctx_lengths_prefill,
        "comp_ctx_lengths_decode": comp_ctx_lengths_decode,
        "ctx_len": ctx_len,
        "prefill_seq_len": 1,  # Passing prefill_seq_len is mandatory for CCL goal in moe models. Currently we can get best perf using PL=1.
    },
)
tokenizer = AutoTokenizer.from_pretrained(model_id)

onnx_model_path = qeff_model.export()
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
)
print(f"qpc path is {qpc_path}")
streamer = TextStreamer(tokenizer)
exec_info = qeff_model.generate(
    tokenizer,
    prompts="Who is your creator? and What all you are allowed to do?",
    generation_len=256,
)

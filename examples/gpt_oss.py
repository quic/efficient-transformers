# -----------------------------------------------------------------------------
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------

from transformers import AutoTokenizer, TextStreamer

from QEfficient import QEFFAutoModelForCausalLM

model_id = "openai/gpt-oss-20b"

qeff_model = QEFFAutoModelForCausalLM.from_pretrained(model_id)
tokenizer = AutoTokenizer.from_pretrained(model_id)

onnx_model_path = qeff_model.export()
qpc_path = qeff_model.compile(
    prefill_seq_len=1,  # Currently we can get best perf using PL=1 i.e. decode-only model, prefill optimizations are being worked on.
    ctx_len=256,
    num_cores=16,
    mxfp6_matmul=True,
    mxint8_kv_cache=True,
    num_devices=8,
    mos=1,
    aic_enable_depth_first=True,
    num_speculative_tokens=None,
)
print(f"qpc path is {qpc_path}")
streamer = TextStreamer(tokenizer)
exec_info = qeff_model.generate(
    tokenizer,
    streamer=streamer,
    prompts="Who is your creator? and What all you are allowed to do?",
    device_ids=[0, 1, 2, 3],
)

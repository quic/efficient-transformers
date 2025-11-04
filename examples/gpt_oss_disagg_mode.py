# -----------------------------------------------------------------------------
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------

from transformers import AutoTokenizer, TextStreamer

from QEfficient import QEFFAutoModelForCausalLM

model_id = "openai/gpt-oss-20b"  # weights are not required to convert to fp32

qeff_model = QEFFAutoModelForCausalLM.from_pretrained(model_id, num_hidden_layers=2)
tokenizer = AutoTokenizer.from_pretrained(model_id)

decode_qpc_path = qeff_model.compile(
    prefill_seq_len=1,  # Currently we can get best perf using PL=1 i.e. decode-only model, prefill optimizations are being worked on.
    ctx_len=256,
    num_cores=16,
    mxfp6_matmul=True,
    mxint8_kv_cache=True,
    num_devices=1,
    mos=1,
    aic_enable_depth_first=True,
    num_speculative_tokens=None,
    offload_pt_weights=False,
)
prefill_qpc_path = qeff_model.compile(
    prefill_seq_len=256,  # Currently we can get best perf using PL=1 i.e. decode-only model, prefill optimizations are being worked on.
    ctx_len=256,
    num_cores=16,
    mxfp6_matmul=True,
    mxint8_kv_cache=True,
    num_devices=1,
    mos=1,
    aic_enable_depth_first=True,
    num_speculative_tokens=None,
    prefill_only=True,
)
# print(f"qpc path is {qpc_path}")
# streamer = TextStreamer(tokenizer)
# exec_info = qeff_model.generate(
#     tokenizer,
#     prompts="Who is your creator? and What all you are allowed to do?",
#     device_id=[0, 1, 2, 3],
# )

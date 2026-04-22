# -----------------------------------------------------------------------------
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
#
# ----------------------------------------------------------------------------

import os

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from QEfficient import QEFFAutoModelForCausalLM

# parameters to be configured

prompt = "Once upon a time,"
num_kv_heads_repeat = 1  # When using KIMI_BLOCKING="kv", make sure num_kv_heads_repeat is set to 1. Use only for KIMI_BLOCKING="h" when using MLA and this should be equal to TS in that case.
num_hidden_layers = 2
TS = 4
enable_mla = True
mla_absorption_config = {"enable": False, "online": False}
qaic_config = None

if os.environ.get("KIMI_BLOCKING", "0") == "kv":
    qaic_config = {"enable_blocking": True, "blocking_mode": "kv"}
    num_kv_heads_repeat = 1

if os.environ.get("KIMI_BLOCKING", "0") == "h":
    if enable_mla:
        num_kv_heads_repeat = TS
    else:
        num_kv_heads_repeat = 1  # head replication is not needed if MLA is not enabled
    qaic_config = {"enable_blocking": True, "blocking_mode": "h", "head_block_size": TS}

model = AutoModelForCausalLM.from_pretrained(
    "moonshotai/Kimi-K2-Thinking",
    torch_dtype=torch.float32,
    num_hidden_layers=num_hidden_layers,
    trust_remote_code=True,
)
tokenizer = AutoTokenizer.from_pretrained("moonshotai/Kimi-K2-Thinking", trust_remote_code=True)

qeff_model = QEFFAutoModelForCausalLM(model, num_kv_heads_repeat=num_kv_heads_repeat, qaic_config=qaic_config)

qpc_path = qeff_model.compile(
    prefill_seq_len=1,
    ctx_len=16 * 1024,
    enable_mla=enable_mla,
    mla_absorption_config=mla_absorption_config,
    mxfp6_matmul=True,
    mxint8_kv_cache=False,
    num_devices=TS,
    num_cores=16,
    # use_onnx_subfunctions=True,
    qaic_config=qaic_config,
)

qeff_model.generate(prompts=["Once upon a time,"], tokenizer=tokenizer)

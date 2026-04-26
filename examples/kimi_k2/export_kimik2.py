# -----------------------------------------------------------------------------
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
#
# ----------------------------------------------------------------------------

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from QEfficient import QEFFAutoModelForCausalLM

# parameters to be configured
prompt = "Once upon a time,"
num_hidden_layers = 2
TS = 4
enable_mla = True
mla_absorption_config = {"enable": False, "online": False}
# qaic_config = None #for orig_forward
# qaic_config = {"num_kv_heads_repeat": TS}  #with head replication for orig_forward
# qaic_config = {"enable_blocking": True, "blocking_mode": "h", "num_kv_heads_repeat": TS} # for h blocking, it internally sets head_block_size equal to num_devices/num_kv_heads_repeat
qaic_config = {"enable_blocking": True, "blocking_mode": "kv"}  # for KV blocking

# model_path = "/home/ochougul/.cache/huggingface/hub/models--moonshotai--Kimi-K2-Thinking/snapshots/a51ccc050d73dab088bf7b0e2dd9b30ae85a4e55/"
model_path = (
    "/home/huggingface_hub/models--moonshotai--Kimi-K2-Thinking/snapshots/612681931a8c906ddb349f8ad0f582cb552189cd"
)
model = AutoModelForCausalLM.from_pretrained(
    model_path, torch_dtype=torch.float32, num_hidden_layers=num_hidden_layers, trust_remote_code=True
)
tokenizer = AutoTokenizer.from_pretrained("moonshotai/Kimi-K2-Thinking", trust_remote_code=True)

qeff_model = QEFFAutoModelForCausalLM(model)

prefill_seq_len = 1
ctx_len = 16 * 1024

qpc_path = qeff_model.compile(
    prefill_seq_len=prefill_seq_len,
    ctx_len=ctx_len,
    enable_mla=enable_mla,
    mla_absorption_config=mla_absorption_config,
    mxfp6_matmul=True,
    mxint8_kv_cache=False,
    num_devices=TS,
    num_cores=16,
    use_onnx_subfunctions=True,
    qaic_config=qaic_config,
)

qeff_model.generate(prompts=["Once upon a time,"], tokenizer=tokenizer)

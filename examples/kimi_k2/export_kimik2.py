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
mla_absorption = {"cache_compressed": True, "absorption": False, "online": False}
# qaic_config = {"mla_absorption": mla_absorption} # for No Blocking
# qaic_config = {"mla_absorption": mla_absorption, "num_kv_heads_repeat": TS}  # No blocking with kv head replication
# qaic_config = {"mla_absorption": mla_absorption, "enable_blocking": True, "blocking_mode": "kv"}  # for KV blocking
qaic_config = {
    "mla_absorption": mla_absorption,
    "enable_blocking": True,
    "blocking_mode": "h",
    "num_kv_heads_repeat": TS,
}
# for h blocking, it internally sets head_block_size equal to num_devices/num_kv_heads_repeat

model_name = "moonshotai/Kimi-K2-Thinking"
model = AutoModelForCausalLM.from_pretrained(
    model_name, torch_dtype=torch.float32, num_hidden_layers=num_hidden_layers, trust_remote_code=True
)
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

qeff_model = QEFFAutoModelForCausalLM(model, qaic_config=qaic_config)

prefill_seq_len = 1
ctx_len = 16 * 1024

qpc_path = qeff_model.compile(
    prefill_seq_len=prefill_seq_len,
    ctx_len=ctx_len,
    mla_absorption=mla_absorption,
    mxfp6_matmul=True,
    mxint8_kv_cache=False,
    num_devices=TS,
    num_cores=16,
    use_onnx_subfunctions=True,
    qaic_config=qaic_config,
)

qeff_model.generate(prompts=["Once upon a time,"], tokenizer=tokenizer)

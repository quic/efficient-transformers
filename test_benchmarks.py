# -----------------------------------------------------------------------------
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------

from transformers import AutoConfig, AutoModelForCausalLM

from QEfficient import QEFFAutoModelForCausalLM
from QEfficient.blocking.attention_blocking import AttentionBlockingConfig, BlockingMode

config = AutoConfig.from_pretrained("tiny-random/gpt-oss-bf16", trust_remote_code=True)
text_model = AutoModelForCausalLM.from_config(config, trust_remote_code=True, attn_implementation="eager")

# Benchmark the decoder directly — no full VLM weights needed
m = QEFFAutoModelForCausalLM(text_model, enable_benchmark=True)
bc = AttentionBlockingConfig(mode=BlockingMode.KV, num_kv_blocks=16)
m.compile(prefill_seq_len=32, ctx_len=16384, blocking_config=bc, use_onnx_subfunctions=True)
m.generate(tokenizer=None, prompts=[])

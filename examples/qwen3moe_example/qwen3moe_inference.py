# -----------------------------------------------------------------------------
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------

from transformers import AutoTokenizer

from QEfficient import QEFFAutoModelForCausalLM

model_name = "Qwen/Qwen3-30B-A3B-Instruct-2507"
"""
# For CB inference, set continuous_batching to True and add full_batch_size argument in compile function
# We will use prompt_len=1 for compilation for both cb and non-cb inference
"""
model = QEFFAutoModelForCausalLM.from_pretrained(model_name, continuous_batching=False, num_hidden_layers=2)
model.compile(prefill_seq_len=1, ctx_len=256, num_cores=16, num_devices=4, mxfp6_matmul=False, mxint8_kv_cache=False)
tokenizer = AutoTokenizer.from_pretrained(model_name)
exec_info = model.generate(prompts=["My name is"], tokenizer=tokenizer)
gen_length = 50
cloud_ai_100_tokens = exec_info.generated_ids[0][:, :gen_length]

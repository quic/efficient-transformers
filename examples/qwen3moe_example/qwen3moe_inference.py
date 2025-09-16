# -----------------------------------------------------------------------------
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------

from transformers import AutoTokenizer

from QEfficient import QEFFAutoModelForCausalLM

model_name = "Qwen/Qwen3-30B-A3B-Instruct-2507"
model = QEFFAutoModelForCausalLM.from_pretrained(model_name, num_hidden_layers=48)
model.compile(prefill_seq_len=1, ctx_len=256, num_cores=16, num_devices=4)

tokenizer = AutoTokenizer.from_pretrained(model_name)
exec_info = model.generate(prompts=["My name is"], tokenizer=tokenizer)
cloud_ai_100_tokens = exec_info.generated_ids[0][:, :50]

# -----------------------------------------------------------------------------
#
# Copyright (c) 2025 Qualcomm Innovation Center, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------

# This is the work example of the GGUF models with the AI 100

from transformers import AutoTokenizer

from QEfficient import QEFFAutoModelForCausalLM as AutoModelForCausalLM

# Load the model and tokenizer
model_name = "MaziyarPanahi/Mistral-7B-Instruct-v0.3-GGUF"
gguf_file = "Mistral-7B-Instruct-v0.3.fp16.gguf"
# org_model_name = "mistralai/Mistral-7B-Instruct-v0.3"

tokenizer = AutoTokenizer.from_pretrained(model_name, gguf_file=gguf_file)
model = AutoModelForCausalLM.from_pretrained(model_name, gguf_file=gguf_file)

generated_qpc_path = model.compile(prefill_seq_len=32, ctx_len=128, num_cores=16, num_devices=1)
model.generate(prompts=["How are you?"], tokenizer=tokenizer)

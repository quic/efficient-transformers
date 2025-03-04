# -----------------------------------------------------------------------------
#
# Copyright (c) 2024 Qualcomm Innovation Center, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------

import torch
from transformers import AutoConfig, AutoTokenizer

from QEfficient import QEFFAutoModelForCausalLM

# Select model and load it.
MODEL_ID = "unsloth/DeepSeek-V3-bf16"

config = AutoConfig.from_pretrained(MODEL_ID)
# del config.quantization_config
qeff_model = QEFFAutoModelForCausalLM.from_pretrained(MODEL_ID, num_hidden_layers=4)
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

print("\n\n")
print("========== SAMPLE GENERATION ==============")

prompt = "Hello my name is"
inputs = tokenizer(
    prompt,
    return_tensors="pt",
    padding=True,
)
input_ids = inputs["input_ids"]
batch_size, input_len = input_ids.shape
prompt_len = 32
inputs.pop("attention_mask")
inputs.pop("token_type_ids", None)
position_ids = torch.arange(input_len).view(1, -1)
inputs["input_ids"] = torch.concat(
    [
        input_ids,
        torch.ones((batch_size, prompt_len - input_len), dtype=torch.int64) * (tokenizer.pad_token_id),
    ],
    1,
)
inputs["position_ids"] = torch.concat(
    [
        position_ids,
        torch.ones((batch_size, prompt_len - input_len), dtype=torch.int64) * (-1),
    ],
    1,
)
qeff_model.compile(prefill_seq_len=32, ctx_len=128, num_cores=16, num_devices=1)
output = qeff_model.generate(prompts=prompt, tokenizer=tokenizer)
print(tokenizer.decode(output[0]))

print("==========================================\n\n")

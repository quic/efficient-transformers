# -----------------------------------------------------------------------------
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------

import os
import warnings

import torch
from peft import AutoPeftModelForCausalLM
from transformers import AutoModelForCausalLM, AutoTokenizer

from QEfficient.finetune.configs.training import TrainConfig

# Suppress all warnings
warnings.filterwarnings("ignore")

try:
    import torch_qaic  # noqa: F401

    device = "qaic:0"
except ImportError as e:
    print(f"Warning: {e}. Moving ahead without these qaic modules.")
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

train_config = TrainConfig()
model = AutoModelForCausalLM.from_pretrained(
    train_config.model_name,
    use_cache=False,
    attn_implementation="sdpa",
    torch_dtype=torch.float16 if torch.cuda.is_available() or device == "qaic:0" else None,
)

# Load the tokenizer and add special tokens
tokenizer = AutoTokenizer.from_pretrained(
    train_config.model_name if train_config.tokenizer_name is None else train_config.tokenizer_name
)
if not tokenizer.pad_token_id:
    tokenizer.pad_token_id = tokenizer.eos_token_id

# This prompt template is specific to alpaca dataset, please change it according to your dataset.
eval_prompt = """"Below is an instruction that describes a task. Write a response that appropriately completes the request.

### Instruction:
Give three tips for staying healthy.

### Response:"""

model_input = tokenizer(eval_prompt, return_tensors="pt")

model.to(device)
model_input.to(device)
model.eval()

with torch.inference_mode():
    print(
        tokenizer.decode(
            model.generate(**model_input, max_new_tokens=50, do_sample=False)[0],
            skip_special_tokens=True,
        )
    )

# Load the pre-trained model from latest checkpoint
trained_weights_path = os.path.join(train_config.output_dir, "trained_weights")
epoch_max_index = max([int(name.split("_")[-1]) for name in os.listdir(trained_weights_path)])
epochs_path = os.path.join(trained_weights_path, "epoch_" + str(epoch_max_index))
step_max_index = max([int(name.split("_")[-1]) for name in os.listdir(epochs_path)])
save_dir = os.path.join(epochs_path, "step_" + str(step_max_index))

# Load PEFT model on CPU
model = AutoPeftModelForCausalLM.from_pretrained(save_dir)
# Merge LoRA and base model and save
merged_model = model.merge_and_unload()
merged_model.save_pretrained(train_config.output_dir, safe_serialization=True)
model_id = train_config.output_dir

# Load Model with PEFT adapter
model_peft = AutoModelForCausalLM.from_pretrained(model_id, use_cache=False, attn_implementation="sdpa")

model_peft.to(device)
model_peft.eval()
with torch.inference_mode():
    print(
        tokenizer.decode(
            model_peft.generate(**model_input, max_new_tokens=50, do_sample=False)[0],
            skip_special_tokens=True,
        )
    )

# -----------------------------------------------------------------------------
#
# Copyright (c) 2024 Qualcomm Innovation Center, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------

import os
import warnings

import torch
from peft import AutoPeftModelForCausalLM
from transformers import AutoModelForCausalLM, AutoTokenizer

from QEfficient.finetune.configs.training import train_config as TRAIN_CONFIG

# Suppress all warnings
warnings.filterwarnings("ignore")

try:
    import torch_qaic  # noqa: F401

    device = "qaic:0"
except ImportError as e:
    print(f"Warning: {e}. Moving ahead without these qaic modules.")
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

train_config = TRAIN_CONFIG()
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

eval_prompt = """
    Summarize this dialog:
    A: Hi Tom, are you busy tomorrow’s afternoon?
    B: I’m pretty sure I am. What’s up?
    A: Can you go with me to the animal shelter?.
    B: What do you want to do?
    A: I want to get a puppy for my son.
    B: That will make him so happy.
    ---
    Summary:
    """

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

trained_weights_path = os.path.join(train_config.output_dir, "trained_weights")
list_paths = [d for d in os.listdir(trained_weights_path) if os.path.isdir(os.path.join(trained_weights_path, d))]
max_index = max([int(path[5:]) for path in list_paths])

save_dir = os.path.join(trained_weights_path, "step_" + str(max_index))

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

# -----------------------------------------------------------------------------
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------

import logging
import os
import warnings

import torch
from peft import AutoPeftModelForCausalLM
from transformers import AutoModelForCausalLM, AutoTokenizer

from QEfficient.finetune.configs.training import TrainConfig
from QEfficient.finetune.utils.logging_utils import logger

# Suppress all warnings
warnings.filterwarnings("ignore")

try:
    import torch_qaic  # noqa: F401

    device = "qaic:0"
except ImportError as e:
    logger.log_rank_zero(
        f"Unable to import 'torch_qaic' package due to exception: {e}. Moving ahead without the torch_qaic extension.",
        logging.WARNING,
    )
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

train_config = TrainConfig()
model = AutoModelForCausalLM.from_pretrained(
    train_config.model_name,
    use_cache=False,
    attn_implementation="sdpa",
    torch_dtype=torch.float16 if torch.cuda.is_available() or device == "qaic:0" else None,
)

# Load the tokenizer and add special tokens
tokenizer = AutoTokenizer.from_pretrained(train_config.model_name if train_config.tokenizer_name is None else train_config.tokenizer_name)
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
    logger.log_rank_zero(
        tokenizer.decode(
            model.generate(**model_input, max_new_tokens=50, do_sample=False)[0],
            skip_special_tokens=True,
        )
    )

# Load the pre-trained model from latest checkpoint
save_dir = os.path.join(train_config.output_dir, "complete_epoch_" + str(train_config.num_epochs))

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
    logger.log_rank_zero(
        tokenizer.decode(
            model_peft.generate(**model_input, max_new_tokens=50, do_sample=False)[0],
            skip_special_tokens=True,
        )
    )

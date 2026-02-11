# -----------------------------------------------------------------------------
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
#
# ----------------------------------------------------------------------------

"""
Simple example: How to enable proxy model and generate IO files.
"""

from transformers import AutoTokenizer

from QEfficient import QEFFAutoModelForCausalLM

model_name = "openai-community/gpt2"

# Load proxy model (enable_proxy=True replaces embedding and LM head with proxy implementations)
model = QEFFAutoModelForCausalLM.from_pretrained(model_name, enable_proxy=True)

# Compile model
model.compile(num_cores=16)

# Generate with IO files
tokenizer = AutoTokenizer.from_pretrained(model_name)
model.generate(
    prompts=["Hi there!!"],
    tokenizer=tokenizer,
    write_io=True,  # Saves input/output tensors to files
)

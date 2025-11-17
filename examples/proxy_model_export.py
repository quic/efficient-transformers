# -----------------------------------------------------------------------------
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------

from transformers import AutoTokenizer

from QEfficient import QEFFAutoModelForCausalLM

model = QEFFAutoModelForCausalLM.from_pretrained(
    "gpt2", num_hidden_layers=2, enable_proxy=True
)  # enable_proxy=True to use proxy model export i.e., export model disable the embedding and LM head layers
model.compile(num_cores=16)
tokenizer = AutoTokenizer.from_pretrained("gpt2")
model.generate(prompts=["Hi there!!"], tokenizer=tokenizer, write_io=True)  # write_io = True to save io files

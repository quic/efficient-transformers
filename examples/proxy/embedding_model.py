# -----------------------------------------------------------------------------
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
#
# ----------------------------------------------------------------------------

"""
Simple example: How to enable proxy model for embeddings and generate IO files.
"""

from transformers import AutoTokenizer

from QEfficient import QEFFAutoModel

# Model configuration
model_name = "BAAI/bge-base-en-v1.5"
test_text = "My name is John"

# Load proxy model (enable_proxy=True replaces embeddings with proxy implementations)
model = QEFFAutoModel.from_pretrained(model_name, pooling="mean", enable_proxy=True)

# Compile model
model.compile(num_cores=16)

# Tokenize input
tokenizer = AutoTokenizer.from_pretrained(model_name)
inputs = tokenizer(test_text, return_tensors="pt")

# Generate embeddings with IO files
output = model.generate(inputs, write_io=True)
print(output)  # Output will be a dictionary containing extracted features

# -----------------------------------------------------------------------------

# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause

# -----------------------------------------------------------------------------

# This is the work example of the Embedding model with the AI 100
# For more information, visit: https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2

from transformers import AutoTokenizer

from QEfficient import QEFFAutoModel as AutoModel

# Sentences we want sentence embeddings for
sentences = "This is an example sentence"

# Load model from HuggingFace Hub
tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")

# If pooling is not set, model will generate default output
qeff_model = AutoModel.from_pretrained("sentence-transformers/all-MiniLM-L6-v2", pooling="mean")

# Here list of seq_len also can be used
qeff_model.compile(num_cores=16, seq_len=32)

# Tokenize sentences
encoded_input = tokenizer(sentences, return_tensors="pt")

sentence_embeddings = qeff_model.generate(encoded_input)

print("Sentence embeddings:", sentence_embeddings)

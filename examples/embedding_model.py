# -----------------------------------------------------------------------------
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------

# This is the work example of the Embedding model with the AI 100
# For more information, visit: https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2

from transformers import AutoTokenizer

from QEfficient import QEFFAutoModel as AutoModel

# Sentences we want sentence embeddings for
sentences = "This is an example sentence"

# Load model from HuggingFace Hub
tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")


qeff_model = AutoModel.from_pretrained("sentence-transformers/all-MiniLM-L6-v2", pooling="mean")
qeff_model.compile(num_cores=14)

# Tokenize sentences
encoded_input = tokenizer(sentences, return_tensors="pt")
sentence_embeddings = qeff_model.generate(encoded_input)

print("Sentence embeddings:", sentence_embeddings)

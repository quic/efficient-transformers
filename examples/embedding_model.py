# -----------------------------------------------------------------------------
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------

# This is the work example of the Embedding model with the AI 100
# For more information, visit: https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer

from QEfficient import QEFFAutoModel as AutoModel


def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output  # First element of model_output contains all token embeddings
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)



# Sentences we want sentence embeddings for
sentences = "This is an example sentence"

# Load model from HuggingFace Hub
tokenizer = AutoTokenizer.from_pretrained("intfloat/e5-large", token="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")


qeff_model = AutoModel.from_pretrained("intfloat/e5-large", token="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")
qeff_model.compile(num_cores=14)

# Tokenize sentences
encoded_input = tokenizer(sentences, return_tensors="pt")
qeff_output = torch.tensor(qeff_model.generate(encoded_input))

# Perform pooling
sentence_embeddings = mean_pooling(qeff_output, encoded_input["attention_mask"])

# Normalize embeddings
sentence_embeddings = F.normalize(sentence_embeddings, p=2, dim=1)

print("Sentence embeddings:")
print(sentence_embeddings)

# -----------------------------------------------------------------------------

# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause

# -----------------------------------------------------------------------------

# This is the work example of the Embedding model with the AI 100
# For more information, visit: https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2

import torch
from transformers import AutoTokenizer

from QEfficient import QEFFAutoModel as AutoModel


# def max_pooling(last_hidden_states: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
#     input_mask_expanded = attention_mask.unsqueeze(-1).expand(last_hidden_states.size()).float()
#     last_hidden_states[input_mask_expanded == 0] = -1e9
#     return torch.max(last_hidden_states, 1)[0]

import torch

# def max_pooling(last_hidden_states: torch.Tensor, position_ids: torch.Tensor) -> torch.Tensor:
#     # Expand position_ids to match the shape of last_hidden_states
#     position_mask_expanded = position_ids.unsqueeze(-1).expand(last_hidden_states.size()).float()
    
#     # Mask out positions with a special value (e.g., -1e9) where position_id is 0
#     last_hidden_states[position_mask_expanded == 0] = -1e9
    
#     # Apply max pooling across the sequence length dimension
#     return torch.max(last_hidden_states, dim=1)[0]
def max_pooling(last_hidden_states: torch.Tensor, position_ids: torch.Tensor) -> torch.Tensor:
    # Create a mask where position_ids > 0 (or use a different condition based on your data)
    position_mask = (position_ids > 0).unsqueeze(-1).expand(last_hidden_states.size()).float()
    last_hidden_states[position_mask == 0] = -1e9
    return torch.max(last_hidden_states, 1)[0]

# Sentences we want sentence embeddings for
sentences = "This is an example sentence"

model_name="jinaai/jina-embeddings-v2-base-code"
# Load model from HuggingFace Hub
tokenizer = AutoTokenizer.from_pretrained(model_name)


# You can specify the pooling strategy either as a string (e.g., "max") or by passing a custom pooling function.
# If no pooling is specified, the model will return its default output (typically token embeddings).
qeff_model = AutoModel.from_pretrained(model_name, pooling=max_pooling, trust_remote_code=True, num_hidden_layers=1)
# qeff_model = AutoModel.from_pretrained("sentence-transformers/all-MiniLM-L6-v2", pooling="max")
# qeff_model = AutoModel.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")

# Here seq_len can be list of seq_len or single int
qeff_model.compile(num_cores=16, seq_len=32)
# qeff_model.compile(num_cores=16, seq_len=32)


# Tokenize sentences
encoded_input = tokenizer(sentences, return_tensors="pt")

sentence_embeddings = qeff_model.generate(encoded_input)

print("Sentence embeddings:", sentence_embeddings)

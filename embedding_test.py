# -----------------------------------------------------------------------------

# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause

# -----------------------------------------------------------------------------

import argparse

import torch
from transformers import AutoTokenizer

from QEfficient import QEFFAutoModel as AutoModel


testing_models = [
    # "BAAI/bge-base-en-v1.5",
    # "BAAI/bge-large-en-v1.5",
    # "BAAI/bge-small-en-v1.5",
    # "intfloat/e5-large-v2",
    # "sentence-transformers/multi-qa-mpnet-base-cos-v1",
    "intfloat/e5-mistral-7b-instruct", #RuntimeError: _Map_base::at
    # "nomic-ai/nomic-embed-text-v1.5", #trust remote code
    # "NovaSearch/stella_en_1.5B_v5", #RuntimeError: _Map_base::at
    # "ibm-granite/granite-embedding-30m-english",
    # # "ibm-granite/granite-embedding-125m-english",
    # "BAAI/bge-reranker-v2-m3",
    # "ibm-granite/granite-embedding-107m-multilingual",
    # "ibm-granite/granite-embedding-278m-multilingual"
]


def max_pooling(last_hidden_states: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    """Apply max pooling to the last hidden states."""
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(last_hidden_states.size()).float()
    last_hidden_states[input_mask_expanded == 0] = -1e9
    return torch.max(last_hidden_states, 1)[0]

pooling = "min"
seq_len = 32
sentences = "This is an example sentence"
num_cores = 16

for model_name in testing_models:

    print(f"Loading embedding model: {model_name}")
    print(f"Pooling strategy: {pooling}")
    print(f"Sequence length(s): {seq_len}")

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Load model with pooling strategy
    # You can specify the pooling strategy either as a string (e.g., "max") or by passing a custom pooling function.
    # If no pooling is specified, the model will return its default output (typically token embeddings).
    if pooling == "max":
        qeff_model = AutoModel.from_pretrained(model_name, pooling=max_pooling)
    elif pooling == "mean":
        qeff_model = AutoModel.from_pretrained(model_name, pooling="mean")
    else:
        qeff_model = AutoModel.from_pretrained(model_name)

    print(qeff_model)
    # Compile the model
    # seq_len can be a list of seq_len or single int
    qeff_model.compile(num_cores=num_cores, seq_len=seq_len)

    # # Tokenize sentences
    encoded_input = tokenizer(sentences, return_tensors="pt")

    # # Run the generation
    sentence_embeddings = qeff_model.generate(encoded_input)

    print(f"\nInput: {sentences}")
    print(f"Sentence embeddings shape: {sentence_embeddings['output'].shape}")
    print(f"Sentence embeddings: {sentence_embeddings}")


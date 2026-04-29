# -----------------------------------------------------------------------------

# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause

# -----------------------------------------------------------------------------


import json
import os

import pytest
import torch
from transformers import AutoTokenizer

from QEfficient import QEFFAutoModel as AutoModel

from ..nightly_utils import NIGHTLY_SKIPPED_MODELS, get_onnx_and_qpc_size


def max_pooling(last_hidden_states: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    """Apply max pooling to the last hidden states."""
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(last_hidden_states.size()).float()
    last_hidden_states[input_mask_expanded == 0] = -1e9
    return torch.max(last_hidden_states, 1)[0]


model_config_path = os.path.join(os.path.dirname(__file__), "../configs/validated_models.json")
with open(model_config_path, "r") as f:
    config = json.load(f)

test_models = config["embedding_models"]


@pytest.mark.parametrize("model_name", test_models)
@pytest.mark.parametrize("pooling", [None])
def test_generate_causal_lm(model_name, pooling, get_model_config, embedding_model_artifacts):

    if model_name in NIGHTLY_SKIPPED_MODELS:
        pytest.skip(f"Skipping {model_name} as it is in nightly skipped models list.")

    config, pipeline_configs = get_model_config
    compile_params = pipeline_configs["embedding_model_configs"][0].get("compile_params", {})
    generate_params = pipeline_configs["embedding_model_configs"][0].get("generate_params", {})
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Retrieve onnx_path from previous stage
    if model_name not in embedding_model_artifacts or "onnx_path" not in embedding_model_artifacts[model_name]:
        pytest.skip(f"ONNX path not available for {model_name}. Run test_export.py first.")

    # Retrieve qpc_path from previous stage
    if model_name not in embedding_model_artifacts or "qpc_path" not in embedding_model_artifacts[model_name]:
        pytest.skip(f"QPC path not available for {model_name}. Run test_compile.py first.")

    if pooling == "max":
        qeff_model = AutoModel.from_pretrained(model_name, pooling=max_pooling)
    elif pooling == "mean":
        qeff_model = AutoModel.from_pretrained(model_name, pooling="mean")
    else:
        qeff_model = AutoModel.from_pretrained(model_name)

    onnx_path = embedding_model_artifacts[model_name].get("onnx_path")
    _ = qeff_model.compile(onnx_path=onnx_path, **compile_params)

    # Tokenize sentences
    sentences = generate_params.get("prompts", ["This is an example sentence"])
    encoded_input = tokenizer(sentences, return_tensors="pt")

    # Run the generation
    sentence_embeddings = qeff_model.generate(encoded_input)

    onnx_and_qpc_dir = os.path.dirname(onnx_path)
    size = get_onnx_and_qpc_size(onnx_and_qpc_dir)
    # Store all metrics and execution info
    embedding_model_artifacts[model_name].update(
        {
            "onnx_and_qpc_dir": onnx_and_qpc_dir,
            "size": size,
            "embedding": sentence_embeddings["output"].shape,
        }
    )

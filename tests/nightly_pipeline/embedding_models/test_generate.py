# -----------------------------------------------------------------------------

# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause

# -----------------------------------------------------------------------------


import json
import os

import pytest
from transformers import AutoTokenizer

from QEfficient import QEFFAutoModel as AutoModel

from ..nightly_utils import get_onnx_and_qpc_size, max_pooling, pre_generate_utils

model_config_path = os.path.join(os.path.dirname(__file__), "../configs/validated_models.json")
with open(model_config_path, "r") as f:
    config = json.load(f)

test_models = config["embedding_models"]


@pytest.mark.parametrize("model_name", test_models)
@pytest.mark.parametrize("pooling", [None])
def test_generate_causal_lm(model_name, pooling, get_pipeline_config, embedding_model_artifacts):

    compile_params, generate_params = pre_generate_utils(
        model_name, "embedding_model_configs", get_pipeline_config, embedding_model_artifacts
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if pooling == "max":
        qeff_model = AutoModel.from_pretrained(model_name, pooling=max_pooling, trust_remote_code=True)
    elif pooling == "mean":
        qeff_model = AutoModel.from_pretrained(model_name, pooling="mean", trust_remote_code=True)
    else:
        qeff_model = AutoModel.from_pretrained(model_name, trust_remote_code=True)

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

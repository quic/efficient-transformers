# -----------------------------------------------------------------------------
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------
import json
import os
from typing import Optional

import numpy as np
import onnxruntime as ort
import pytest
import torch
from transformers import AutoModel, AutoTokenizer

from QEfficient.transformers.embeddings.embedding_utils import POOLING_MAP
from QEfficient.transformers.models.modeling_auto import QEFFAutoModel
from QEfficient.utils._utils import create_json
from QEfficient.utils.constants import Constants, QnnConstants

CONFIG_PATH = "tests/configs/embedding_model_configs.json"

with open(CONFIG_PATH, "r") as f:
    config_data = json.load(f)
    embed_test_models = config_data["embedding_models"]


def check_embed_pytorch_vs_ort_vs_ai100(
    model_name: str,
    seq_len: int = Constants.CTX_LEN,
    n_layer: int = 1,
    enable_qnn: Optional[bool] = False,
    qnn_config: Optional[str] = None,
    pooling: Optional[str] = None,
):
    # Prepare input
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    inputs = tokenizer("My name is", return_tensors="pt")

    # Original PyTorch model
    pt_model = AutoModel.from_pretrained(
        model_name,
        num_hidden_layers=n_layer,
        attn_implementation="eager",
        trust_remote_code=True,
    )

    # Original PyTorch model output
    pt_outputs = pt_model(**inputs)
    pooling_method = POOLING_MAP[pooling] if pooling else None
    pt_embeddings = (
        pooling_method(pt_outputs.last_hidden_state, inputs["attention_mask"])
        if pooling
        else pt_outputs.last_hidden_state
    )

    # QEff transformed PyTorch model
    qeff_model = QEFFAutoModel(pt_model, pretrained_model_name_or_path=model_name, pooling=pooling)

    # QEff transformed PyTorch model output
    qeff_pt_outputs = qeff_model.generate(inputs=inputs, runtime_ai100=False)
    qeff_pt_embeddings = qeff_pt_outputs if pooling else qeff_pt_outputs[0]

    mad = torch.mean(torch.abs(pt_embeddings - qeff_pt_embeddings))
    print("Mad for PyTorch and PyTorch transformed qeff_model is ", mad)
    assert mad <= 0, f"MAD is too high for onnx and Pytorch: {mad}"

    # ONNX session load
    onnx_model = qeff_model.export()
    ort_session = ort.InferenceSession(str(onnx_model))

    # Prepare the inputs for ONNX Runtime
    input_ids = np.array(inputs["input_ids"])
    attention_mask = np.array(inputs["attention_mask"])

    onnx_inputs = {"input_ids": input_ids, "attention_mask": attention_mask}

    # Run inference
    onnx_outputs = ort_session.run(None, onnx_inputs)

    # Compare Transformed PyTorch and ONNX outputs
    mad = torch.mean(torch.abs(pt_embeddings - torch.tensor(onnx_outputs[0])))
    print("Mad for onnx and PyTorch is ", mad)
    assert mad <= 10**-5, f"MAD is too high for onnx and Pytorch: {mad}"

    qeff_model.compile(
        num_cores=14,
        enable_qnn=enable_qnn,
        qnn_config=qnn_config,
    )
    ai100_output = qeff_model.generate(inputs=inputs)
    qeff_ai100_embeddings = (
        ai100_output["output"] if pooling else ai100_output["output"][:, : inputs["input_ids"].shape[1], :]
    )

    # Compare ONNX and AI 100 outputs
    mad = np.mean(np.abs(qeff_ai100_embeddings - onnx_outputs[0]))
    print("Mad for onnx and AI 100 output is ", mad)
    assert mad <= 10**-2, f"MAD is too high for onnx and Pytorch: {mad}"
    assert os.path.isfile(os.path.join(os.path.dirname(qeff_model.qpc_path), "qconfig.json"))


@pytest.mark.on_qaic
@pytest.mark.llm_model
@pytest.mark.parametrize("model", embed_test_models)
def test_embed_model_pytorch_vs_onnx_vs_ai100(model):
    """
    Test function to validate output of the Pytorch, ONNX and AI 100 runtime model output.
    """
    check_embed_pytorch_vs_ort_vs_ai100(model_name=model["model_name"], seq_len=32, n_layer=1)


@pytest.mark.on_qaic
@pytest.mark.llm_model
@pytest.mark.parametrize("model", embed_test_models)
def test_embed_model_pytorch_vs_onnx_vs_ai100_pooling(model):
    """
    Test function to validate output of the Pytorch, ONNX and AI 100 runtime model output with pooling.
    """
    check_embed_pytorch_vs_ort_vs_ai100(model_name=model["model_name"], seq_len=32, n_layer=1, pooling=model["pooling"])


@pytest.mark.on_qaic
@pytest.mark.llm_model
@pytest.mark.parametrize("model", embed_test_models[:1])
def test_embed_model_pytorch_vs_onnx_vs_ai100_multiple_seq_len(model):
    """
    Test function to validate output of the Pytorch, ONNX and AI 100 runtime model output with multiple seq_len.
    """
    check_embed_pytorch_vs_ort_vs_ai100(model_name=model["model_name"], seq_len=[32, 20], n_layer=1)


##########  QNN TESTS ##############


@pytest.mark.on_qaic
@pytest.mark.llm_model
@pytest.mark.qnn
@pytest.mark.parametrize("model_name", embed_test_models)
def test_embed_model_pytorch_vs_onnx_vs_ai100_qnn(model_name):
    """
    QNN Compilation path test.
    Test function to validate output of the Pytorch, ONNX and AI 100 runtime model output.
    """
    qnn_config_json_path = os.path.join(os.getcwd(), "qnn_config.json")
    create_json(qnn_config_json_path, QnnConstants.QNN_SAMPLE_CONFIG)

    check_embed_pytorch_vs_ort_vs_ai100(
        model_name=model_name["model_name"], seq_len=32, n_layer=1, enable_qnn=True, qnn_config=qnn_config_json_path
    )


@pytest.mark.on_qaic
@pytest.mark.llm_model
@pytest.mark.qnn
@pytest.mark.parametrize("model", embed_test_models)
def test_embed_model_pytorch_vs_onnx_vs_ai100_pooling_qnn(model):
    """
    QNN Compilation path test.
    Test function to validate output of the Pytorch, ONNX and AI 100 runtime model output with pooling.
    """
    qnn_config_json_path = os.path.join(os.getcwd(), "qnn_config.json")
    create_json(qnn_config_json_path, QnnConstants.QNN_SAMPLE_CONFIG)

    check_embed_pytorch_vs_ort_vs_ai100(
        model_name=model["model_name"],
        seq_len=32,
        n_layer=1,
        pooling=model["pooling"],
        enable_qnn=True,
        qnn_config=qnn_config_json_path,
    )


@pytest.mark.on_qaic
@pytest.mark.llm_model
@pytest.mark.qnn
@pytest.mark.parametrize("model", [embed_test_models[0]])
def test_embed_model_pytorch_vs_onnx_vs_ai100_multiple_seq_len_qnn(model):
    """
    QNN Compilation path test.
    Test function to validate output of the Pytorch, ONNX and AI 100 runtime model output with multiple seq_len.
    """
    qnn_config_json_path = os.path.join(os.getcwd(), "qnn_config.json")
    create_json(qnn_config_json_path, QnnConstants.QNN_SAMPLE_CONFIG)

    check_embed_pytorch_vs_ort_vs_ai100(
        model_name=model["model_name"], seq_len=[32, 20], n_layer=1, enable_qnn=True, qnn_config=qnn_config_json_path
    )

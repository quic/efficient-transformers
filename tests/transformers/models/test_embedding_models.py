# -----------------------------------------------------------------------------
#
# Copyright (c) 2024 Qualcomm Innovation Center, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------

import os
from typing import Optional

import numpy as np
import onnxruntime as ort
import pytest
from transformers import AutoModel, AutoTokenizer

from QEfficient.transformers.models.modeling_auto import QEFFAutoModel
from QEfficient.utils._utils import create_json
from QEfficient.utils.constants import Constants, QnnConstants

embed_test_models = [
    # model_name, architecture
    "sentence-transformers/multi-qa-mpnet-base-cos-v1",  # MPNetForMaskedLM
    "BAAI/bge-reranker-v2-m3",  # XLMRobertaForSequenceClassification
    "BAAI/bge-small-en-v1.5",  # BertModel
]


def check_embed_pytorch_vs_ort_vs_ai100(
    model_name: str,
    seq_len: int = Constants.CTX_LEN,
    n_layer: int = 1,
    enable_qnn: Optional[bool] = False,
    qnn_config: Optional[str] = None,
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

    pt_outputs = pt_model(**inputs)
    pt_embeddings = pt_outputs[0][0].detach().numpy()
    # Pytorch transformed model
    qeff_model = QEFFAutoModel(pt_model)
    qeff_pt_outputs = qeff_model.generate(inputs=inputs, runtime_ai100=False)
    qeff_pt_embeddings = qeff_pt_outputs[0][0].detach().numpy()
    mad = np.mean(np.abs(pt_embeddings - qeff_pt_embeddings))
    print("Mad for PyTorch and PyTorch transformed qeff_model is ", mad)
    assert mad <= 0, f"MAD is too high for onnx and Pytorch: {mad}"

    onnx_model = qeff_model.export()
    ort_session = ort.InferenceSession(str(onnx_model))

    # Prepare the inputs for ONNX Runtime
    input_ids = np.array(inputs["input_ids"])
    attention_mask = np.array(inputs["attention_mask"])

    onnx_inputs = {"input_ids": input_ids, "attention_mask": attention_mask}
    # Run inference
    onnx_outputs = ort_session.run(None, onnx_inputs)

    # Compare Transformed PyTorch and ONNX outputs

    pt_embeddings = pt_outputs[0][0].detach().numpy()
    onnx_embeddings = onnx_outputs[0]
    mad = np.mean(np.abs(pt_embeddings - onnx_embeddings))
    print("Mad for onnx and PyTorch is ", mad)
    assert mad <= 10**-5, f"MAD is too high for onnx and Pytorch: {mad}"

    qeff_model.compile(
        num_cores=14,
        enable_qnn=enable_qnn,
        qnn_config=qnn_config,
    )
    ai100_output = qeff_model.generate(inputs=inputs)

    # Compare ONNX and AI 100 outputs
    mad = np.mean(np.abs(ai100_output - onnx_outputs[0]))
    print("Mad for onnx and AI 100 output is ", mad)
    assert mad <= 10**-3, f"MAD is too high for onnx and Pytorch: {mad}"
    assert os.path.isfile(os.path.join(os.path.dirname(qeff_model.qpc_path), "qconfig.json"))


@pytest.mark.on_qaic
@pytest.mark.parametrize("model_name", embed_test_models)
def test_embed_model_pytorch_vs_onnx_vs_ai100(model_name):
    """
    Test function to validate output of the Pytorch, ONNX and AI 100 runtime model output.
    """
    check_embed_pytorch_vs_ort_vs_ai100(model_name=model_name, seq_len=32, n_layer=1)


@pytest.mark.on_qaic
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
        model_name=model_name, seq_len=32, n_layer=1, enable_qnn=True, qnn_config=qnn_config_json_path
    )

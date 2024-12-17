# -----------------------------------------------------------------------------
#
# Copyright (c) 2024 Qualcomm Innovation Center, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------


import numpy as np
import onnxruntime as ort
import pytest

from QEfficient.transformers.models.modeling_auto import QEffAutoModel
from QEfficient.utils import hf_download
from QEfficient.utils._utils import load_hf_tokenizer
from QEfficient.utils.constants import Constants

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
):
    model_path = hf_download(
        repo_id=model_name,
        ignore_patterns=["*.onnx", "*.ot", "*.md", "*.tflite", "*.pdf", "*.h5", "*.msgpack"],
    )

    qeff_model = QEffAutoModel.from_pretrained(
        pretrained_model_name_or_path=model_path,
        num_hidden_layers=n_layer,
        attn_implementation="eager",
        trust_remote_code=True,
    )

    prompt = "My name is"
    pt_outputs = qeff_model.generate(prompts=["My name is"], runtime_ai100=False)

    onnx_model = qeff_model.export()
    ort_session = ort.InferenceSession(str(onnx_model))
    # Prepare the inputs for ONNX Runtime
    tokenizer = load_hf_tokenizer(model_path)
    inputs = tokenizer(prompt, return_tensors="pt", padding="max_length", max_length=seq_len)
    onnx_inputs = {"input_ids": inputs["input_ids"].numpy(), "attention_mask": inputs["attention_mask"].numpy()}
    # Run inference
    onnx_outputs = ort_session.run(None, onnx_inputs)

    # Compare PyTorch and ONNX outputs
    pt_embeddings = pt_outputs[0][0].detach().numpy()
    onnx_embeddings = onnx_outputs[0]
    mad = np.mean(np.abs(pt_embeddings - onnx_embeddings))
    print("Mad for onnx and pytorch is ", mad)
    assert mad <= 10**-5, f"MAD is too high for onnx and Pytorch: {mad}"

    qeff_model.compile(
        num_cores=14,
    )
    ai100_output = qeff_model.generate(prompts=["My name is"])

    # Compare ONNX and AI 100 outputs
    mad = np.mean(np.abs(ai100_output[0]["output"] - onnx_outputs[0]))
    print("Mad for onnx and AI 100 output is ", mad)
    assert mad <= 10**-3, f"MAD is too high for onnx and Pytorch: {mad}"


@pytest.mark.on_qaic
@pytest.mark.parametrize("model_name", embed_test_models)
def test_embed_model_pytorch_vs_onnx_vs_ai100(model_name):
    """
    Test function to validate output of the Pytorch, ONNX and AI 100 runtime model output.
    """
    check_embed_pytorch_vs_ort_vs_ai100(model_name=model_name, seq_len=32, n_layer=1)

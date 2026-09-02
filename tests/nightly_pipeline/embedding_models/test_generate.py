# -----------------------------------------------------------------------------

# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause

# -----------------------------------------------------------------------------


import json
import os
from pathlib import Path

import pytest
import torch
from transformers import AutoTokenizer

from QEfficient import QEFFAutoModel

from ..model_age_utils import filter_models_for_nightly
from ..nightly_utils import (
    compare_with_golden,
    make_golden_key,
    measure_peak_ram,
    run_or_load_golden,
    get_nightly_skip_reason,
)

model_config_path = os.path.join(os.path.dirname(__file__), "../configs/validated_models.json")
with open(model_config_path, "r") as f:
    config = json.load(f)

PIPELINE_CONFIG_FP = os.path.join(os.path.dirname(__file__), "../configs/pipeline_configs.json")

test_models = filter_models_for_nightly(config["embedding_models"], "embedding_models")

poolings = ["mean", "max", "cls", "avg", None]


def _generate_embedding_model(model_name, pooling, get_pipeline_config, embedding_model_artifacts, torch_dtype, seq_len=32, dtype_key="fp32"):
    """Common generate logic for embedding models.

    Reads artifacts from nested structure:
        artifacts[model_name][dtype_key][pooling_key]
    """
    # embedding uses a 3-level nested structure (model → dtype → pooling)
    skip_reason = get_nightly_skip_reason(model_name, "embedding_model_configs")
    if skip_reason:
        pytest.skip(skip_reason)

    if model_name == "nomic-ai/nomic-embed-text-v1.5" and pooling is None and isinstance(seq_len, list):
        pytest.xfail("nomic-embed-text-v1.5 pooling=None multiseqlen fails with `IndexError: list index out of range`")

    pipeline_configs = get_pipeline_config
    compile_params = pipeline_configs["embedding_model_configs"][0].get("compile_params", {})
    generate_params = pipeline_configs["embedding_model_configs"][0].get("generate_params", {})

    pooling_key = str(pooling) if pooling is not None else "None"

    # Check nested artifacts exist
    pooling_artifacts = (
        embedding_model_artifacts
        .get(model_name, {})
        .get(dtype_key, {})
        .get(pooling_key, {})
    )
    if "onnx_path" not in pooling_artifacts:
        pytest.skip(f"ONNX path not available for {model_name} [{dtype_key}][{pooling_key}]. Run export and compile first.")
    if "qpc_path" not in pooling_artifacts:
        pytest.skip(f"QPC path not available for {model_name} [{dtype_key}][{pooling_key}]. Run export and compile first.")

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    qeff_model = QEFFAutoModel.from_pretrained(model_name, pooling=pooling, torch_dtype=torch_dtype, attn_implementation="eager", trust_remote_code=True)

    onnx_path = pooling_artifacts["onnx_path"]
    qeff_model.qpc_path = Path(pooling_artifacts["qpc_path"])
    qeff_model.onnx_path = Path(onnx_path)

    # Tokenize sentences
    sentences = generate_params.get("prompts", ["This is an example sentence"])
    encoded_input = tokenizer(sentences, return_tensors="pt")

    with measure_peak_ram() as ram:
        sentence_embeddings = qeff_model.generate(inputs=encoded_input, dtype=torch_dtype)

    # Golden output: run PyTorch reference if not already stored, then compare
    golden_key = make_golden_key(
        dtype=dtype_key,
        config_params=compile_params,
        extra_tags={"pooling": pooling_key, "seq_len": seq_len},
    )

    def _run_pytorch():
        """Run HF PyTorch embedding inference and return golden output dict."""
        from QEfficient.transformers.embeddings.embedding_utils import POOLING_MAP
        from transformers import AutoModel as HFAutoModel
        hf_model = HFAutoModel.from_pretrained(
            model_name,
            attn_implementation="eager",
            trust_remote_code=True,
            )
        hf_model.eval()
        with torch.no_grad():
            outputs = hf_model(**encoded_input)

        # Apply same pooling as QPC model
        if pooling and pooling in POOLING_MAP:
            pooling_fn = POOLING_MAP[pooling]
            embeddings = pooling_fn(outputs.last_hidden_state, encoded_input["attention_mask"])
        else:
            embeddings = outputs.last_hidden_state
        return {"pytorch_embeddings": embeddings.tolist()}

    golden = run_or_load_golden(
        category="embedding_models",
        model_name=model_name,
        golden_key=golden_key,
        run_pytorch_fn=_run_pytorch,
        config_fp=PIPELINE_CONFIG_FP,
    )

    tolerance = (
        pipeline_configs["embedding_model_configs"][0]
        .get("golden_mad_validation", {})
        .get("tolerance", 1e-2)
    )
    qpc_emb = sentence_embeddings["output"]
    if pooling is None:
        qpc_emb = qpc_emb[:, : encoded_input["input_ids"].shape[1], :]
    comparison = compare_with_golden(
        qpc_output={"pytorch_embeddings": qpc_emb.tolist()},
        golden={"pytorch_embeddings": golden.get("pytorch_embeddings", [])},
        tolerance=tolerance,
    )
    print(f"\n[GOLDEN COMPARISON] passed={comparison['passed']} details={comparison['per_key']}")

    onnx_and_qpc_dir = os.path.dirname(onnx_path)
    embedding_model_artifacts[model_name][dtype_key][pooling_key].update(
        {
            "embedding_shape": list(sentence_embeddings["output"].shape),
            "embedding_mean": round(float(sentence_embeddings["output"].mean()), 6),
            "embedding_max": round(float(sentence_embeddings["output"].max()), 6),
            "seq_len": seq_len,
            "generate_peak_ram_mb": round(ram["peak_mb"], 2),
            "golden_comparison": comparison,
        }
    )

    assert comparison["passed"], f"QPC output differs from golden PyTorch: {comparison['per_key']}"


# Config 1: FP32, all poolings, single seq_len
@pytest.mark.parametrize("model_name", test_models)
@pytest.mark.parametrize("pooling", poolings)
def test_generate_embedding_model(model_name, pooling, get_pipeline_config, embedding_model_artifacts):
    """FP32 generate, all pooling variants."""
    _generate_embedding_model(model_name, pooling, get_pipeline_config, embedding_model_artifacts, torch.float32, dtype_key="fp32")


# Config 2: FP32, all poolings, multi seq_len
@pytest.mark.parametrize("model_name", test_models)
@pytest.mark.parametrize("pooling", poolings)
def test_generate_embedding_model_multiseqlen(model_name, pooling, get_pipeline_config, embedding_model_artifacts):
    """FP32 generate, multi seq_len."""
    _generate_embedding_model(model_name, pooling, get_pipeline_config, embedding_model_artifacts, torch.float32, seq_len=[32, 20], dtype_key="fp32_multiseqlen")


# Config 3: FP16, all poolings, single seq_len
@pytest.mark.parametrize("model_name", test_models)
@pytest.mark.parametrize("pooling", poolings)
def test_generate_embedding_model_fp16(model_name, pooling, get_pipeline_config, embedding_model_artifacts):
    """FP16 generate, all pooling variants."""
    _generate_embedding_model(model_name, pooling, get_pipeline_config, embedding_model_artifacts, torch.float16, dtype_key="fp16")


# Config 4: FP16, all poolings, multi seq_len
@pytest.mark.parametrize("model_name", test_models)
@pytest.mark.parametrize("pooling", poolings)
def test_generate_embedding_model_fp16_multiseqlen(model_name, pooling, get_pipeline_config, embedding_model_artifacts):
    """FP16 generate, multi seq_len."""
    _generate_embedding_model(model_name, pooling, get_pipeline_config, embedding_model_artifacts, torch.float16, seq_len=[32, 20], dtype_key="fp16_multiseqlen")

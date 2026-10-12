# -----------------------------------------------------------------------------
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------

"""
Weight-free export ONNX structure and ORT parity tests.

All tests export with weight_free=True (set on the QEff model) and use_onnx_subfunctions=True.
Each test validates:
  - ONNX graph structure: _RetainedState outputs, subfunctions, naming
  - Weight-free-specific: unique graph input names, no int64 KV cache inputs
  - HF PT == ORT token parity (weights injected from HF cache via load_weight_free_ort_inputs)

CPU-only. No QAIC hardware required.
"""

from __future__ import annotations

import json

import numpy as np
import onnx
import pytest
import torch
from transformers import AutoConfig

from QEfficient.exporter.weight_free import resolve_weight_spec_path
from QEfficient.transformers.models.modeling_auto import QEFFAutoModelForCausalLM
from QEfficient.utils import get_num_layers_from_config
from QEfficient.utils.run_utils import ApiRunner

from ._helpers import (
    BATCH_SIZE,
    CTX_LEN,
    PROMPT_LEN,
    WEIGHT_FREE_CAUSAL_LM_MODEL_IDS,
    assert_has_subfunctions,
    assert_no_int64_kv_cache_inputs,
    assert_retained_state_outputs,
    assert_subfunction_names_match_decoder_class,
    assert_unique_graph_input_names,
    exported_onnx_path,
    load_hf_model,
    load_tokenizer,
    run_weight_free_ort,
    skip_on_model_fetch_error,
)


@pytest.mark.weight_free
@pytest.mark.weight_free_export
@pytest.mark.parametrize(
    "model_type,model_id",
    sorted(WEIGHT_FREE_CAUSAL_LM_MODEL_IDS.items()),
    ids=sorted(WEIGHT_FREE_CAUSAL_LM_MODEL_IDS),
)
def test_weight_free_export_onnx_structure(model_type, model_id, tmp_export_dir):
    """Export with weight_free=True and use_onnx_subfunctions=True, then validate:
    - Correct _RetainedState output count (2 x num_layers)
    - Subfunctions renamed to decoder class names
    - No duplicate graph input names (guards position_ids aliasing regression)
    - No int64 tensor named past_key.X / past_value.X (same regression, dtype check)

    CPU-only. No QAIC hardware or weight injection required.
    """
    try:
        # Build meta-device model — no weights loaded, only shapes.
        # pretrained_model_name_or_path is carried in the QEff model so the export
        # can write weight_spec.json pointing at the HF cache checkpoint.
        config = AutoConfig.from_pretrained(model_id, trust_remote_code=True)
        config.num_hidden_layers = 2
        qeff_model = QEFFAutoModelForCausalLM.from_pretrained(model_id, config=config, weight_free=True)
    except Exception as exc:
        skip_on_model_fetch_error(exc, model_id)

    onnx_path = exported_onnx_path(
        qeff_model.export(
            tmp_export_dir,
            use_onnx_subfunctions=True,
            offload_pt_weights=False,
        )
    )

    # Shared structure checks
    num_layers = qeff_model.model.config.num_hidden_layers
    assert_retained_state_outputs(onnx_path, expected_count=2 * num_layers)
    assert_has_subfunctions(onnx_path, qeff_model)
    assert_subfunction_names_match_decoder_class(onnx_path, qeff_model)

    # Weight-free-specific regression guards
    assert_unique_graph_input_names(onnx_path)
    assert_no_int64_kv_cache_inputs(onnx_path)


@pytest.mark.weight_free
@pytest.mark.weight_free_export
def test_qwen4_exp_weight_free_four_layer_export_and_ort_execution(tmp_path):
    """Export and execute a locally generated four-layer Qwen4-Exp checkpoint."""
    from tests.unit_test.models.test_new_arch_accuracy import make_tiny_qwen4_exp

    from QEfficient.exporter.weight_free import load_weight_free_ort_inputs
    from QEfficient.transformers.models.pytorch_transforms import KVCacheTransform

    source_dir = tmp_path / "qwen4_exp_checkpoint"
    source_model, _ = make_tiny_qwen4_exp()
    source_model.save_pretrained(source_dir, safe_serialization=True)

    qeff_weight_free = QEFFAutoModelForCausalLM.from_pretrained(
        str(source_dir), weight_free=True, torch_dtype=torch.float32
    )
    onnx_path = exported_onnx_path(
        qeff_weight_free.export(
            str(tmp_path / "export"),
            use_onnx_subfunctions=False,
            offload_pt_weights=False,
        )
    )
    weight_spec_path = resolve_weight_spec_path(onnx_path)
    assert weight_spec_path.is_file()
    assert json.loads(weight_spec_path.read_text())["inputs"]

    graph = onnx.load(str(onnx_path), load_external_data=False)
    assert_retained_state_outputs(onnx_path, expected_count=11)
    computed_suffixes = {"sin_cached", "cos_cached", "inv_freq", "original_inv_freq", "_qeff_conv_positions"}
    assert all(initializer.name.rsplit(".", 1)[-1] in computed_suffixes for initializer in graph.graph.initializer)

    source_model = source_model.__class__.from_pretrained(str(source_dir)).eval()
    source_model, _ = KVCacheTransform.apply(source_model)
    qeff_inputs = source_model.get_dummy_inputs(batch_size=2)
    with torch.no_grad():
        pytorch_output = source_model(**qeff_inputs)

    retained = [state for layer in qeff_inputs["past_key_values"] for state in layer]
    runtime_inputs = {
        name: value.detach().cpu().numpy()
        for name, value in zip(
            source_model.get_onnx_dynamic_axes(),
            [qeff_inputs["input_ids"], qeff_inputs["ngram_embeddings"], qeff_inputs["position_ids"], *retained],
            strict=True,
        )
    }
    ort_inputs = load_weight_free_ort_inputs(weight_spec_path, runtime_inputs)
    import onnxruntime as ort

    ort_outputs = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"]).run(None, ort_inputs)
    expected_outputs = [
        pytorch_output.logits.detach().cpu().numpy(),
        *[state.detach().cpu().numpy() for layer in pytorch_output.past_key_values for state in layer],
    ]
    for expected, actual in zip(expected_outputs, ort_outputs, strict=True):
        np.testing.assert_allclose(expected, actual, rtol=1e-4, atol=1e-5)


@pytest.mark.weight_free
@pytest.mark.weight_free_export
@pytest.mark.parametrize(
    "model_type,model_id",
    sorted(WEIGHT_FREE_CAUSAL_LM_MODEL_IDS.items()),
    ids=sorted(WEIGHT_FREE_CAUSAL_LM_MODEL_IDS),
)
def test_weight_free_export_ort_parity(model_type, model_id, tmp_export_dir):
    """Export with weight_free=True, inject real weights into ORT, then validate
    HF PyTorch == ORT token parity.

    Unlike regular ONNX (which embeds weights), weight-free ONNX stores weights externally
    in a weight_spec.json. load_weight_free_ort_inputs reads the checkpoint from HF cache
    and injects the tensors as ORT inputs at each generation step.

    CPU-only. No QAIC hardware required.
    """
    try:
        model_hf = load_hf_model(model_id)
        tokenizer = load_tokenizer(model_id)
    except Exception as exc:
        skip_on_model_fetch_error(exc, model_id)

    api_runner = ApiRunner(
        batch_size=BATCH_SIZE,
        tokenizer=tokenizer,
        config=model_hf.config,
        prompt=["hello world"],
        prompt_len=PROMPT_LEN,
        ctx_len=CTX_LEN,
        full_batch_size=None,
    )

    # HF PyTorch reference tokens (real weights)
    hf_tokens = api_runner.run_hf_model_on_pytorch(model_hf)

    # Weight-free export uses a meta-device model — no weights in the ONNX.
    # The real weights are read from the HF cache at ORT inference time via
    # load_weight_free_ort_inputs(weight_spec_path, inputs).
    # The exported model must have the same layer count as model_hf, or the
    # HF PT and ORT token streams come from architecturally different models.
    try:
        config = AutoConfig.from_pretrained(model_id, trust_remote_code=True)
        config.num_hidden_layers = get_num_layers_from_config(model_hf.config)
        qeff_model = QEFFAutoModelForCausalLM.from_pretrained(model_id, config=config, weight_free=True)
    except Exception as exc:
        skip_on_model_fetch_error(exc, model_id)

    onnx_path = exported_onnx_path(
        qeff_model.export(
            tmp_export_dir,
            use_onnx_subfunctions=True,
            offload_pt_weights=False,
        )
    )
    weight_spec_path = resolve_weight_spec_path(onnx_path)

    # ORT inference with weights injected from HF cache
    ort_tokens = run_weight_free_ort(api_runner, onnx_path, weight_spec_path)

    assert hf_tokens is not None and ort_tokens is not None
    assert hf_tokens.flatten().tolist() == ort_tokens.flatten().tolist(), (
        f"HF PT vs weight-free ORT parity failed for {model_hf.__class__.__name__}: "
        f"HF={hf_tokens.flatten().tolist()}, ORT={ort_tokens.flatten().tolist()}"
    )

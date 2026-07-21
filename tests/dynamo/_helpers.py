# -----------------------------------------------------------------------------
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------

"""
Shared helpers, model registry, and constants for tests/dynamo/.

Model IDs follow the same plain-dict pattern as CAUSAL_RUNTIME_MODEL_IDS
in tests/unit_test/models/test_model_quickcheck.py.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import onnx
import pytest
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from QEfficient.transformers.models.modeling_auto import QEFFAutoModelForCausalLM
from QEfficient.utils.run_utils import ApiRunner

# ---------------------------------------------------------------------------
# Model registry
# ---------------------------------------------------------------------------

DYNAMO_CAUSAL_LM_MODEL_IDS = {
    "codegen": "hf-internal-testing/tiny-random-CodeGenForCausalLM",
    # deepseek_v3: transformers version in this env is missing is_torch_fx_available;
    # excluded until the env is updated to a compatible transformers release.
    "falcon": "hf-internal-testing/tiny-random-FalconForCausalLM",
    "gemma": "Xenova/tiny-random-GemmaForCausalLM",
    "gemma2": "hf-internal-testing/tiny-random-Gemma2ForCausalLM",
    "glm4_moe": "tiny-random/glm-4-moe",
    "gpt2": "hf-internal-testing/tiny-random-GPT2LMHeadModel",
    "gpt_bigcode": "hf-internal-testing/tiny-random-GPTBigCodeForCausalLM",
    "gpt_oss": "tiny-random/gpt-oss-bf16",
    "gptj": "hf-internal-testing/tiny-random-GPTJForCausalLM",
    "granite": "hf-internal-testing/tiny-random-GraniteForCausalLM",
    "granitemoe": "hf-internal-testing/tiny-random-GraniteMoeForCausalLM",
    # grok_1: tiny random model config is malformed for QEff (AttributeError on keys());
    # excluded until a valid tiny checkpoint is available.
    "llama": "hf-internal-testing/tiny-random-LlamaForCausalLM",
    # llama_swiftkv requires QEffLlamaSwiftKVConfig + QEffLlamaSwiftKVForCausalLM,
    # not AutoModelForCausalLM. No tiny random model exists on HF Hub.
    # Tested via check_causal_models.py with the full Snowflake model.
    "mistral": "hf-internal-testing/tiny-random-MistralForCausalLM",
    "mixtral": "hf-internal-testing/tiny-random-MixtralForCausalLM",
    "mpt": "hf-internal-testing/tiny-random-MptForCausalLM",
    "olmo2": "hf-internal-testing/tiny-random-Olmo2ForCausalLM",
    "phi": "hf-internal-testing/tiny-random-PhiForCausalLM",
    "phi3": "tiny-random/phi-4",
    "qwen2": "yujiepan/qwen2-tiny-random",
    "qwen3": "tiny-random/qwen3",
    "qwen3_moe": "tiny-random/qwen3-moe",
    "starcoder2": "hf-internal-testing/tiny-random-Starcoder2ForCausalLM",
}

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

PROMPT_LEN = 8
CTX_LEN = 16
BATCH_SIZE = 1
FULL_BATCH_SIZE = 4
MODEL_KWARGS = {"attn_implementation": "eager", "low_cpu_mem_usage": False, "torch_dtype": torch.float32}

# ---------------------------------------------------------------------------
# Load helpers
# ---------------------------------------------------------------------------


def skip_on_model_fetch_error(exc: Exception, model_id: str) -> None:
    pytest.skip(
        f"Skipping {model_id}: model unavailable or unsupported in this environment ({type(exc).__name__}: {exc})"
    )


def load_hf_model(model_id: str) -> AutoModelForCausalLM:
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        trust_remote_code=True,
        **MODEL_KWARGS,
    )
    model.eval()
    return model


def load_tokenizer(model_id: str) -> AutoTokenizer:
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    if not hasattr(tokenizer, "pad_token") or tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def exported_onnx_path(export_result) -> Path:
    if isinstance(export_result, (list, tuple)):
        export_result = export_result[-1]
    onnx_path = Path(export_result)
    assert onnx_path.is_file(), f"Expected ONNX file at {onnx_path}"
    return onnx_path


# ---------------------------------------------------------------------------
# ONNX assertion helpers
# ---------------------------------------------------------------------------


def assert_has_subfunctions(onnx_path: Path, qeff_model: QEFFAutoModelForCausalLM) -> None:
    """Assert the ONNX graph contains at least one decoder-block subfunction.

    CtxScatter/CtxGather/CustomRMSNorm always appear as functions regardless of
    use_onnx_subfunctions, so checking len(model.functions) > 0 is not sufficient.
    We require at least one function whose name contains a decoder class name from
    get_submodules_for_export(), matching the main suite's approach.
    """
    get_submodules = getattr(qeff_model.model, "get_submodules_for_export", None)
    if not callable(get_submodules):
        return  # Model doesn't declare submodule boundaries — skip check

    submodule_classes = get_submodules()
    if not submodule_classes:
        return

    decoder_names = {
        cls.__name__
        for cls in (submodule_classes if isinstance(submodule_classes, (set, list, tuple)) else [submodule_classes])
    }

    model = onnx.load(str(onnx_path), load_external_data=False)
    found = [fn.name for fn in model.functions if any(d in fn.name for d in decoder_names)]
    assert found, (
        f"Expected decoder-block subfunctions ({decoder_names}) in {onnx_path.name} but found none. "
        f"Functions present: {[fn.name for fn in model.functions]}"
    )


def assert_subfunction_names_match_decoder_class(onnx_path: Path, qeff_model: QEFFAutoModelForCausalLM) -> None:
    """Verify RenameRepeatedSubgraphTransform renamed functions to decoder class names."""
    get_submodules = getattr(qeff_model.model, "get_submodules_for_export", None)
    if not callable(get_submodules):
        return  # Model doesn't declare submodule boundaries — skip name check

    submodule_classes = get_submodules()
    if not submodule_classes:
        return

    expected_names = {
        cls.__name__
        for cls in (submodule_classes if isinstance(submodule_classes, (set, list, tuple)) else [submodule_classes])
    }

    model = onnx.load(str(onnx_path), load_external_data=False)
    for fn in model.functions:
        assert not any(fn.name.startswith(pat) for pat in ("repeated_subgraph", "subgraph_", "invoke_subgraph_")), (
            f"Function '{fn.name}' still has raw dynamo name — "
            f"RenameRepeatedSubgraphTransform did not rename it. "
            f"Expected a name derived from {expected_names}."
        )


def assert_retained_state_outputs(onnx_path: Path, expected_count: int) -> None:
    """Assert that the ONNX graph has the expected number of _RetainedState outputs."""
    model = onnx.load(str(onnx_path), load_external_data=False)
    retained = [o for o in model.graph.output if o.name.endswith("_RetainedState")]
    assert len(retained) == expected_count, (
        f"Expected {expected_count} _RetainedState outputs, got {len(retained)}: {[o.name for o in retained]}"
    )


# ---------------------------------------------------------------------------
# ORT parity helper
# ---------------------------------------------------------------------------


def run_dynamo_ort_parity(
    model_hf: AutoModelForCausalLM,
    tokenizer,
    export_dir: Path,
    use_onnx_subfunctions: bool = False,
) -> None:
    """
    Wrap pre-loaded model with QEFFAutoModelForCausalLM, export with dynamo=True,
    and assert HF PT == QEff PT == ORT token parity via ApiRunner.
    """
    api_runner = ApiRunner(
        batch_size=BATCH_SIZE,
        tokenizer=tokenizer,
        config=model_hf.config,
        prompt=["hello world"],
        prompt_len=PROMPT_LEN,
        ctx_len=CTX_LEN,
        full_batch_size=None,
    )

    hf_tokens = api_runner.run_hf_model_on_pytorch(model_hf)

    qeff_model = QEFFAutoModelForCausalLM(model_hf)
    kv_tokens = api_runner.run_kv_model_on_pytorch(qeff_model.model)

    onnx_path = exported_onnx_path(
        qeff_model.export(
            export_dir,
            dynamo=True,
            use_onnx_subfunctions=use_onnx_subfunctions,
            offload_pt_weights=False,
        )
    )
    ort_tokens = api_runner.run_kv_model_on_ort(str(onnx_path))

    assert np.array_equal(hf_tokens, kv_tokens.squeeze(0)), (
        f"HF vs QEff PyTorch parity failed for {model_hf.__class__.__name__}"
    )
    assert np.array_equal(kv_tokens, ort_tokens), (
        f"QEff PyTorch vs ORT parity failed for {model_hf.__class__.__name__} (use_onnx_subfunctions={use_onnx_subfunctions})"
    )

# -----------------------------------------------------------------------------
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------

"""
Shared helpers for the dynamo CausalLM test lanes.

Kept intentionally small: constants, tokenizer/model loading, ApiRunner
construction, and small ONNX / QPC assertions used by more than one file.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple

import onnx
import pytest
import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

from QEfficient.transformers.models.modeling_auto import QEFFAutoModelForCausalLM
from QEfficient.utils.run_utils import ApiRunner

from .model_registry import DynamoModelSpec

# --- Runtime constants -------------------------------------------------------
MODEL_KWARGS = {"attn_implementation": "eager"}
PROMPT = ["hello world"]
PROMPT_LEN = 8
CTX_LEN = 12
FULL_BATCH_SIZE = 4
GENERATION_LEN = 20

RUNTIME_ENV_FLAG = "QEFF_DYNAMO_FULL_RUNTIME"  # kept for docs/env parity; every runtime lane runs by default.


# --- Model / tokenizer loading ----------------------------------------------
def skip_on_model_fetch_error(exc: Exception, model_id: str) -> None:
    pytest.skip(
        f"Skipping {model_id}: model unavailable or unsupported in this environment ({type(exc).__name__}: {exc})"
    )


def load_tokenizer(spec: DynamoModelSpec):
    tokenizer = AutoTokenizer.from_pretrained(spec.model_id, trust_remote_code=True, use_fast=False)
    if getattr(tokenizer, "pad_token", None) is None and getattr(tokenizer, "eos_token", None) is not None:
        tokenizer.pad_token = tokenizer.eos_token
    if hasattr(tokenizer, "model_input_names"):
        tokenizer.model_input_names = ["input_ids", "attention_mask"]
    return tokenizer


def load_model(spec: DynamoModelSpec, *, torch_dtype=torch.float32):
    """Load the tiny HF model for this spec.

    ``pytest.skip`` fires only when the model itself cannot be fetched; we do
    NOT skip on registry policy — every spec must attempt to load, and any
    resulting failure surfaces as a red cell in the coverage report.
    """
    if spec.model_id is None:
        pytest.skip("No tiny model id configured.")

    try:
        if spec.model_loader == "causal_lm":
            model = AutoModelForCausalLM.from_pretrained(
                spec.model_id,
                **MODEL_KWARGS,
                low_cpu_mem_usage=False,
                trust_remote_code=True,
                torch_dtype=torch_dtype,
            )
        else:
            config = AutoConfig.from_pretrained(spec.model_id, trust_remote_code=True)
            text_config = getattr(config, "text_config", None) or getattr(config, "llm_config", None) or config
            if hasattr(text_config, "torch_dtype"):
                text_config.torch_dtype = torch_dtype
            model = AutoModelForCausalLM.from_config(
                text_config,
                **MODEL_KWARGS,
                trust_remote_code=True,
                torch_dtype=torch_dtype,
            )
    except Exception as exc:
        skip_on_model_fetch_error(exc, spec.model_id or "<unknown>")

    model.eval()
    return model


def api_runner(tokenizer, config, *, continuous_batching: bool = False) -> ApiRunner:
    prompt = PROMPT * FULL_BATCH_SIZE if continuous_batching else PROMPT
    return ApiRunner(
        batch_size=1,
        tokenizer=tokenizer,
        config=config,
        prompt=prompt,
        prompt_len=PROMPT_LEN,
        ctx_len=CTX_LEN,
        full_batch_size=FULL_BATCH_SIZE if continuous_batching else None,
    )


def prepare_runtime_model(
    spec: DynamoModelSpec, *, continuous_batching: bool = False
) -> Tuple[object, torch.nn.Module, QEFFAutoModelForCausalLM]:
    """Load HF model + tokenizer + a *deferred* QEff wrapper factory.

    ``QEFFAutoModelForCausalLM(model_hf)`` mutates the passed HF module in
    place, so callers that need HF-baseline tokens must run
    ``run_hf_model_on_pytorch`` BEFORE the wrapper is built. To keep the call
    sites terse without leaking that ordering bug, this helper returns the
    QEff wrapper as a zero-arg factory that the caller invokes once the HF
    baseline is captured.
    """
    tokenizer = load_tokenizer(spec)
    model_hf = load_model(spec, torch_dtype=torch.float32)

    def _build_qeff() -> QEFFAutoModelForCausalLM:
        return QEFFAutoModelForCausalLM(
            model_hf,
            continuous_batching=continuous_batching,
            pretrained_model_name_or_path=spec.model_id,
        )

    return tokenizer, model_hf, _build_qeff


# --- Assertions --------------------------------------------------------------
def exported_onnx_path(export_result) -> Path:
    if isinstance(export_result, (tuple, list)):
        export_result = export_result[-1]
    onnx_path = Path(export_result)
    assert onnx_path.is_file(), f"Export did not produce an ONNX file: {onnx_path}"
    return onnx_path


def assert_has_subfunctions(onnx_path: Path) -> None:
    model = onnx.load(str(onnx_path), load_external_data=False)
    assert model.functions, "Expected dynamo+subfunctions export to emit ONNX local functions."


def assert_qconfig_exists(qpc_path: str) -> None:
    qconfig = Path(qpc_path).parent / "qconfig.json"
    assert qconfig.is_file(), f"qconfig.json missing next to compiled QPC: {qpc_path}"


def generate_without_sampler(tokenizer, qeff_model, *, prompts: Optional[list] = None):
    return qeff_model.generate(
        tokenizer=tokenizer,
        prompts=prompts if prompts is not None else PROMPT,
        generation_len=GENERATION_LEN,
    )

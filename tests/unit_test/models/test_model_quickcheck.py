# -----------------------------------------------------------------------------
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------

import copy
from pathlib import Path

import numpy as np
import onnx
import pytest
import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoModelForImageTextToText, AutoTokenizer

from QEfficient.transformers.models.modeling_auto import QEFFAutoModelForCausalLM
from QEfficient.utils.run_utils import ApiRunner

MODEL_KWARGS = {"attn_implementation": "eager"}
GEMMA4_TEXT_MODEL_IDS = {
    "gemma4_dense": "tiny-random/gemma-4-dense",
    "gemma4_moe": "tiny-random/gemma-4-moe",
}
PARITY_PROMPT = ["hello world"]
PARITY_PROMPT_LEN = 4
PARITY_CTX_LEN = 8
GENERATE_PROMPT = ["hello world hello world"]
GENERATE_PROMPT_LEN = 4
GENERATE_CTX_LEN = 8


def _skip_on_model_fetch_error(exc: Exception, model_id: str) -> None:
    pytest.skip(
        f"Skipping {model_id}: model unavailable or unsupported in this environment ({type(exc).__name__}: {exc})"
    )


def _exported_onnx_path(export_result) -> Path:
    if isinstance(export_result, (list, tuple)):
        export_result = export_result[-1]
    onnx_path = Path(export_result)
    assert onnx_path.is_file()
    return onnx_path


def _load_gemma4_text_model(model_id: str):
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
        if hasattr(tokenizer, "model_input_names"):
            tokenizer.model_input_names = ["input_ids", "attention_mask"]

        full_model = AutoModelForImageTextToText.from_pretrained(
            model_id,
            trust_remote_code=True,
            low_cpu_mem_usage=False,
            torch_dtype=torch.float32,
            **MODEL_KWARGS,
        ).to(torch.float32)
        full_model.eval()

        text_config = AutoConfig.from_pretrained(model_id, trust_remote_code=True).text_config
        text_model = AutoModelForCausalLM.from_config(
            text_config,
            trust_remote_code=True,
            **MODEL_KWARGS,
        ).to(torch.float32)
        text_model.model.load_state_dict(full_model.model.language_model.state_dict())
        text_model.lm_head.load_state_dict(full_model.lm_head.state_dict())
        text_model.eval()
        return tokenizer, text_model
    except Exception as exc:
        _skip_on_model_fetch_error(exc, model_id)


def _build_api_runner(tokenizer, config, prompt, prompt_len, ctx_len):
    return ApiRunner(
        batch_size=1,
        tokenizer=tokenizer,
        config=config,
        prompt=prompt,
        prompt_len=prompt_len,
        ctx_len=ctx_len,
        full_batch_size=None,
    )


@pytest.mark.llm_model
@pytest.mark.parametrize(
    ("model_type", "model_id"),
    sorted(GEMMA4_TEXT_MODEL_IDS.items()),
    ids=sorted(GEMMA4_TEXT_MODEL_IDS),
)
def test_gemma4_text_runtime_parity_and_subfunction_export(model_type, model_id, tmp_path):
    del model_type
    tokenizer, model_hf = _load_gemma4_text_model(model_id)
    api_runner = _build_api_runner(tokenizer, model_hf.config, PARITY_PROMPT, PARITY_PROMPT_LEN, PARITY_CTX_LEN)

    hf_tokens = api_runner.run_hf_model_on_pytorch(model_hf)
    qeff_model = QEFFAutoModelForCausalLM(copy.deepcopy(model_hf), pretrained_model_name_or_path=model_id)
    kv_tokens = api_runner.run_kv_model_on_pytorch(qeff_model.model)
    onnx_path = _exported_onnx_path(
        qeff_model.export(
            tmp_path / "with-subfunctions",
            use_onnx_subfunctions=True,
            offload_pt_weights=False,
        )
    )
    ort_tokens = api_runner.run_kv_model_on_ort(str(onnx_path))

    onnx_model = onnx.load(onnx_path, load_external_data=False)
    assert any("QEffGemma4TextDecoderLayer" in func.name for func in onnx_model.functions)
    assert np.array_equal(hf_tokens, kv_tokens.squeeze(0))
    assert np.array_equal(kv_tokens, ort_tokens)


@pytest.mark.llm_model
@pytest.mark.parametrize(
    ("model_type", "model_id"),
    sorted(GEMMA4_TEXT_MODEL_IDS.items()),
    ids=sorted(GEMMA4_TEXT_MODEL_IDS),
)
def test_gemma4_text_compile_and_generate_smoke_with_subfunctions(model_type, model_id, tmp_path):
    del model_type
    tokenizer, model_hf = _load_gemma4_text_model(model_id)
    api_runner = _build_api_runner(
        tokenizer,
        model_hf.config,
        GENERATE_PROMPT,
        GENERATE_PROMPT_LEN,
        GENERATE_CTX_LEN,
    )
    qeff_model = QEFFAutoModelForCausalLM(copy.deepcopy(model_hf), pretrained_model_name_or_path=model_id)

    onnx_path = _exported_onnx_path(
        qeff_model.export(
            tmp_path / "with-subfunctions",
            use_onnx_subfunctions=True,
            offload_pt_weights=False,
        )
    )

    try:
        qpc_path = qeff_model.compile(
            onnx_path=str(onnx_path),
            compile_dir=tmp_path / "compile-with-subfunctions",
            prefill_seq_len=GENERATE_PROMPT_LEN,
            ctx_len=GENERATE_CTX_LEN,
            use_onnx_subfunctions=True,
        )
    except Exception as exc:
        pytest.skip(
            f"Skipping compile for {model_id}: compile backend unavailable or unsupported in this environment "
            f"({type(exc).__name__}: {exc})"
        )

    assert Path(qpc_path).name == "qpc"
    assert Path(qpc_path).is_dir()

    try:
        exec_info = qeff_model.generate(
            tokenizer,
            prompts=GENERATE_PROMPT,
            generation_len=api_runner.gen_len,
        )
    except Exception as exc:
        pytest.skip(
            f"Skipping QAic generate for {model_id}: runtime unavailable or unsupported in this environment "
            f"({type(exc).__name__}: {exc})"
        )

    cloud_ai_100_tokens = exec_info.generated_ids[0]
    assert cloud_ai_100_tokens.shape[0] == 1
    assert cloud_ai_100_tokens.shape[-1] >= api_runner.gen_len

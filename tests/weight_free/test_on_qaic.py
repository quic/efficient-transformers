# -----------------------------------------------------------------------------
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------

"""
Weight-free on-QAIC tests.

All tests require QAIC hardware (marked @pytest.mark.on_qaic).
All tests run with weight_free=True (set on the QEff model, which forces
dynamo=True internally) and use_onnx_subfunctions=True.

Covers:
  - Generate smoke test (weight-free export -> compile -> generate on QAIC)
  - HF PT vs QAIC HW parity (HF PT tokens == weight-free QAIC top-1 token)
  - Weight-free vs legacy/dynamo QAIC parity (two independently compiled QPCs
    of the same model, generated output compared directly)
"""

from __future__ import annotations

import numpy as np
import pytest
import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoModelForImageTextToText, AutoProcessor

from QEfficient.transformers.models.modeling_auto import QEFFAutoModelForCausalLM
from QEfficient.utils import get_num_layers_from_config

from ._helpers import (
    BATCH_SIZE,
    CTX_LEN,
    PROMPT_LEN,
    WEIGHT_FREE_QAIC_MODEL_PARAMS,
    WEIGHT_FREE_VLM_MODEL_PARAMS,
    exported_onnx_path,
    load_hf_model,
    load_tokenizer,
    load_weight_free_vlm_model,
    skip_on_model_fetch_error,
)

VLM_PREFILL_SEQ_LEN = 64
VLM_CTX_LEN = 512
VLM_IMAGE_HEIGHT = 354
VLM_IMAGE_WIDTH = 536
VLM_NUM_CORES = 16
VLM_NUM_DEVICES = 1
VLM_GENERATION_LEN = 2
VLM_PROMPT = "Describe the image."
VLM_IMAGE_URL = "https://wallup.net/wp-content/uploads/2017/03/28/351036-San_Francisco-USA-bridge-sunset-Golden_Gate_Bridge-lights.jpg"


def _load_vlm_hf_reference(model_id: str):
    config = AutoConfig.from_pretrained(model_id, trust_remote_code=True)
    try:
        model = AutoModelForImageTextToText.from_pretrained(
            model_id,
            config=config,
            attn_implementation="eager",
            trust_remote_code=True,
            torch_dtype=torch.float32,
        )
    except ValueError:
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            config=config,
            attn_implementation="eager",
            trust_remote_code=True,
            torch_dtype=torch.float32,
        )
    return model.eval()


def _prepare_vlm_inputs(processor) -> dict:
    process_vision_info = pytest.importorskip("qwen_vl_utils").process_vision_info
    messages = [
        [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": VLM_IMAGE_URL},
                    {"type": "text", "text": VLM_PROMPT},
                ],
            }
        ]
    ]
    texts = [processor.apply_chat_template(message, tokenize=False, add_generation_prompt=True) for message in messages]
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(text=texts, images=image_inputs, videos=video_inputs, padding=True, return_tensors="pt")
    return dict(inputs)


def _run_vlm_hf_reference(model, inputs: dict) -> np.ndarray:
    with torch.inference_mode():
        outputs = model.generate(
            **inputs,
            max_new_tokens=VLM_GENERATION_LEN,
            min_new_tokens=VLM_GENERATION_LEN,
            do_sample=False,
            temperature=None,
            top_p=None,
        )
    prompt_len = inputs["input_ids"].shape[-1]
    return outputs[:, prompt_len:].detach().cpu().numpy()


@pytest.mark.weight_free
@pytest.mark.on_qaic
@pytest.mark.multimodal
@pytest.mark.parametrize("model_type,model_id", WEIGHT_FREE_VLM_MODEL_PARAMS)
def test_weight_free_vlm_combined_compile(model_type, model_id, tmp_export_dir):
    """Run a weight-free dual-QPC VLM and compare generated tokens with HF."""
    try:
        hf_model = _load_vlm_hf_reference(model_id)
        processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
        qeff_model = load_weight_free_vlm_model(model_id)
    except Exception as exc:
        skip_on_model_fetch_error(exc, model_id)

    hf_tokens = _run_vlm_hf_reference(hf_model, _prepare_vlm_inputs(processor))
    qeff_inputs = _prepare_vlm_inputs(processor)
    common = {
        "batch_size": 1,
        "ctx_len": VLM_CTX_LEN,
        "height": VLM_IMAGE_HEIGHT,
        "prefill_seq_len": VLM_PREFILL_SEQ_LEN,
        "width": VLM_IMAGE_WIDTH,
        "num_cores": VLM_NUM_CORES,
        "num_devices": VLM_NUM_DEVICES,
        "mxfp6_matmul": False,
        "mxint8_kv_cache": False,
        "aic_enable_depth_first": True,
        "mos": 1,
        "split_model_io": True,
        "use_onnx_subfunctions": True,
    }
    if model_type == "gemma4_moe":
        common.pop("height")
        common.pop("width")

    qpc_paths = qeff_model.compile(
        compile_dir=str(tmp_export_dir / f"{model_type}_combined"),
        **common,
    )

    assert qpc_paths.get("vision_qpc_path")
    assert qpc_paths.get("lang_qpc_path")

    qaic_output = qeff_model.generate(inputs=qeff_inputs, generation_len=VLM_GENERATION_LEN)
    assert qaic_output is not None, "Weight-free VLM QAIC generate returned None"
    assert qaic_output.generated_ids is not None, "Weight-free VLM QAIC generate returned no token IDs"

    qaic_tokens = qaic_output.generated_ids[:, :VLM_GENERATION_LEN]
    assert qaic_tokens.shape == hf_tokens.shape == (1, VLM_GENERATION_LEN)
    assert np.array_equal(qaic_tokens, hf_tokens), (
        f"Weight-free VLM QAIC/HF parity failed for {model_id}: HF={hf_tokens.tolist()}, QAIC={qaic_tokens.tolist()}"
    )


@pytest.mark.weight_free
@pytest.mark.on_qaic
@pytest.mark.llm_model
@pytest.mark.parametrize("model_type,model_id", WEIGHT_FREE_QAIC_MODEL_PARAMS)
def test_weight_free_generate_fp16(model_type, model_id, tmp_export_dir):
    """End-to-end weight-free export -> compile -> generate on real QAIC hardware."""
    try:
        config = AutoConfig.from_pretrained(model_id, trust_remote_code=True)
        config.num_hidden_layers = 2
        qeff_model = QEFFAutoModelForCausalLM.from_pretrained(model_id, config=config, weight_free=True)
    except Exception as exc:
        skip_on_model_fetch_error(exc, model_id)

    onnx_path = exported_onnx_path(
        qeff_model.export(
            tmp_export_dir / "wf_gen_export",
            use_onnx_subfunctions=True,
            offload_pt_weights=False,
        )
    )
    qeff_model.compile(
        onnx_path=str(onnx_path),
        compile_dir=str(tmp_export_dir / "wf_gen_compile"),
        prefill_seq_len=PROMPT_LEN,
        ctx_len=CTX_LEN,
        num_cores=16,
        batch_size=BATCH_SIZE,
        use_onnx_subfunctions=True,
    )
    tokenizer = load_tokenizer(model_id)
    output = qeff_model.generate(
        tokenizer=tokenizer,
        prompts=["hello world"],
    )
    assert output is not None
    assert output.generated_texts is not None


@pytest.mark.weight_free
@pytest.mark.on_qaic
@pytest.mark.llm_model
@pytest.mark.parametrize("model_type,model_id", WEIGHT_FREE_QAIC_MODEL_PARAMS)
def test_weight_free_hw_hf_parity(model_type, model_id, tmp_export_dir):
    """HF PT tokens == weight-free QAIC FP16 tokens (exact equality)."""
    from QEfficient.utils.run_utils import ApiRunner

    try:
        tokenizer = load_tokenizer(model_id)
        model_hf = load_hf_model(model_id)
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
    hf_tokens = api_runner.run_hf_model_on_pytorch(model_hf)
    assert hf_tokens is not None, "HF PT inference returned None"

    try:
        config = AutoConfig.from_pretrained(model_id, trust_remote_code=True)
        config.num_hidden_layers = get_num_layers_from_config(model_hf.config)
        qeff_model = QEFFAutoModelForCausalLM.from_pretrained(model_id, config=config, weight_free=True)
    except Exception as exc:
        skip_on_model_fetch_error(exc, model_id)

    onnx_path = exported_onnx_path(
        qeff_model.export(
            tmp_export_dir / "wf_hw_parity_export",
            use_onnx_subfunctions=True,
            offload_pt_weights=False,
        )
    )
    qeff_model.compile(
        onnx_path=str(onnx_path),
        compile_dir=str(tmp_export_dir / "wf_hw_parity_compile"),
        prefill_seq_len=PROMPT_LEN,
        ctx_len=CTX_LEN,
        num_cores=16,
        batch_size=BATCH_SIZE,
        use_onnx_subfunctions=True,
    )
    qaic_output = qeff_model.generate(
        tokenizer=tokenizer,
        prompts=["hello world"],
    )

    assert qaic_output is not None, "QAIC generate returned None"
    if hasattr(qaic_output, "generated_ids") and qaic_output.generated_ids is not None:
        gen_len = CTX_LEN - PROMPT_LEN
        qaic_tokens = qaic_output.generated_ids[0].flatten()[:gen_len]
        assert np.array_equal(hf_tokens, qaic_tokens), (
            f"Weight-free HW/HF parity failed for {model_id}: HF={hf_tokens.tolist()}, QAIC={qaic_tokens.tolist()}"
        )


@pytest.mark.weight_free
@pytest.mark.on_qaic
@pytest.mark.llm_model
@pytest.mark.parametrize("model_type,model_id", WEIGHT_FREE_QAIC_MODEL_PARAMS)
def test_weight_free_vs_legacy_qaic_parity(model_type, model_id, tmp_export_dir):
    """Weight-free-compiled and legacy dynamo-compiled QPCs produce identical tokens on QAIC."""
    try:
        tokenizer = load_tokenizer(model_id)
        model_hf = load_hf_model(model_id)
        if model_type == "gpt_oss":
            model_hf = model_hf.to(torch.float32)
            model_hf.config.torch_dtype = torch.float32
            model_hf.config.dtype = torch.float32
    except Exception as exc:
        skip_on_model_fetch_error(exc, model_id)

    # Legacy/dynamo leg — real weights, no weight-free export.
    qeff_legacy = QEFFAutoModelForCausalLM(model_hf)

    legacy_onnx_path = exported_onnx_path(
        qeff_legacy.export(
            tmp_export_dir / "legacy_export",
            dynamo=True,
            use_onnx_subfunctions=True,
            offload_pt_weights=False,
        )
    )
    qeff_legacy.compile(
        onnx_path=str(legacy_onnx_path),
        compile_dir=str(tmp_export_dir / "legacy_compile"),
        prefill_seq_len=PROMPT_LEN,
        ctx_len=CTX_LEN,
        num_cores=16,
        batch_size=BATCH_SIZE,
        use_onnx_subfunctions=True,
    )
    legacy_output = qeff_legacy.generate(
        tokenizer=tokenizer,
        prompts=["hello world"],
    )
    assert legacy_output is not None, "Legacy QAIC generate returned None"

    # Weight-free leg — meta-device model, matching layer count.
    try:
        config = AutoConfig.from_pretrained(model_id, trust_remote_code=True)
        config.num_hidden_layers = get_num_layers_from_config(model_hf.config)
        qeff_weight_free = QEFFAutoModelForCausalLM.from_pretrained(model_id, config=config, weight_free=True)
    except Exception as exc:
        skip_on_model_fetch_error(exc, model_id)

    weight_free_onnx_path = exported_onnx_path(
        qeff_weight_free.export(
            tmp_export_dir / "wf_vs_legacy_export",
            use_onnx_subfunctions=True,
            offload_pt_weights=False,
        )
    )
    qeff_weight_free.compile(
        onnx_path=str(weight_free_onnx_path),
        compile_dir=str(tmp_export_dir / "wf_vs_legacy_compile"),
        prefill_seq_len=PROMPT_LEN,
        ctx_len=CTX_LEN,
        num_cores=16,
        batch_size=BATCH_SIZE,
        use_onnx_subfunctions=True,
    )
    weight_free_output = qeff_weight_free.generate(
        tokenizer=tokenizer,
        prompts=["hello world"],
    )
    assert weight_free_output is not None, "Weight-free QAIC generate returned None"

    if (
        hasattr(legacy_output, "generated_ids")
        and legacy_output.generated_ids is not None
        and hasattr(weight_free_output, "generated_ids")
        and weight_free_output.generated_ids is not None
    ):
        legacy_tokens = legacy_output.generated_ids[0].flatten()
        weight_free_tokens = weight_free_output.generated_ids[0].flatten()
        assert np.array_equal(legacy_tokens, weight_free_tokens), (
            f"Weight-free vs legacy QAIC parity failed for {model_id}: "
            f"legacy={legacy_tokens.tolist()}, weight_free={weight_free_tokens.tolist()}"
        )

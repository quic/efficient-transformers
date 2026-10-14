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

import json
from pathlib import Path

import numpy as np
import onnx
import pytest
import torch
from PIL import Image
from transformers import (
    AutoConfig,
    AutoModel,
    AutoModelForCausalLM,
    AutoModelForImageTextToText,
    AutoProcessor,
    AutoTokenizer,
)

from QEfficient import QEFFAutoModelForImageTextToText
from QEfficient.generation.cloud_infer import QAICInferenceSession
from QEfficient.transformers.embeddings.embedding_utils import POOLING_MAP
from QEfficient.transformers.models.modeling_auto import QEFFAutoModel, QEFFAutoModelForCausalLM
from QEfficient.utils import get_num_layers_from_config
from QEfficient.utils.test_utils import load_vlm_model_from_config

from ._helpers import (
    BATCH_SIZE,
    CTX_LEN,
    PROMPT_LEN,
    WEIGHT_FREE_EMBEDDING_MODEL_PARAMS,
    WEIGHT_FREE_QAIC_MODEL_PARAMS,
    WEIGHT_FREE_VLM_MODEL_PARAMS,
    exported_onnx_path,
    load_hf_model,
    load_tokenizer,
    load_weight_free_vlm_model,
    skip_on_model_fetch_error,
)

_VLM_CONFIG_PATH = Path(__file__).resolve().parents[1] / "configs" / "image_text_model_configs.json"
with _VLM_CONFIG_PATH.open() as config_file:
    _VLM_MODEL_CONFIGS = {entry["model_type"]: entry for entry in json.load(config_file)["image_text_models"]}

WEIGHT_FREE_VLM_QAIC_MODEL_PARAMS = [
    pytest.param("qwen2_5_vl", marks=pytest.mark.xdist_group(name="qaic-vlm-qwen2-5-vl"), id="qwen2.5-vl"),
    pytest.param("qwen3_vl", marks=pytest.mark.xdist_group(name="qaic-vlm-qwen3-vl"), id="qwen3-vl"),
    pytest.param("qwen3_vl_moe", marks=pytest.mark.xdist_group(name="qaic-vlm-qwen3-vl-moe"), id="qwen3-vl-moe"),
    pytest.param("qwen3_5", marks=pytest.mark.xdist_group(name="qaic-vlm-qwen3-5"), id="qwen3.5"),
    pytest.param("qwen3_5_moe", marks=pytest.mark.xdist_group(name="qaic-vlm-qwen3-5-moe"), id="qwen3.5-moe"),
]
WEIGHT_FREE_VLM_QAIC_MODEL_IDS = {
    "qwen2_5_vl": "optimum-intel-internal-testing/tiny-random-qwen2.5-vl",
    "qwen3_vl": "tiny-random/qwen3-vl",
    "qwen3_vl_moe": "tiny-random/qwen3-vl-moe",
    "qwen3_5": "tiny-random/qwen3.5",
    "qwen3_5_moe": "tiny-random/qwen3.5-moe",
}
VLM_GENERATION_LEN = 2
VLM_PREFILL_SEQ_LEN = 64
VLM_CTX_LEN = 512
VLM_IMAGE_HEIGHT = 354
VLM_IMAGE_WIDTH = 536
VLM_NUM_CORES = 4
VLM_PROMPT = "Describe this image."


def _build_vlm_test_checkpoint(model_type: str, checkpoint_dir: Path, model_id: str | None = None):
    model_config = _VLM_MODEL_CONFIGS[model_type]
    if model_id is None:
        config = AutoConfig.for_model(
            model_type,
            trust_remote_code=True,
            **model_config.get("additional_params", {}),
        )
        config.name_or_path = model_config["model_name"]
    else:
        config = AutoConfig.from_pretrained(model_id, trust_remote_code=True)
        config.name_or_path = model_id
        if model_type == "qwen3_5" and hasattr(config.text_config, "layer_types"):
            config.text_config.layer_types[-1] = "full_attention"
        for nested_config in (config, getattr(config, "text_config", None), getattr(config, "vision_config", None)):
            if nested_config is not None:
                nested_config.dtype = "float32"
                nested_config.torch_dtype = torch.float32
    torch.manual_seed(42)
    model = load_vlm_model_from_config(config)
    model.save_pretrained(checkpoint_dir, safe_serialization=True)
    return model, model_config


def _prepare_vlm_test_inputs(
    processor,
    model,
    qeff_model,
    model_config,
    image_size=(536, 354),
    skip_vision=False,
):
    if skip_vision:
        hf_inputs = processor(text=model_config["text_prompt"], return_tensors="pt")
        qeff_inputs = {
            key: value.clone() if isinstance(value, torch.Tensor) else value for key, value in hf_inputs.items()
        }
        if qeff_model is not None:
            qeff_inputs = qeff_model.prepare_inputs_for_generation(
                inputs=qeff_inputs,
                prefill_seq_len=model_config["prompt_len"],
                batch_size=model_config["batch_size"],
            )
        return hf_inputs, qeff_inputs

    image_width, image_height = image_size
    image = Image.new("RGB", image_size, color=(32, 96, 160))
    conversation = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": model_config["text_prompt"]},
                {"type": "image"},
            ],
        }
    ]
    prompt = processor.apply_chat_template(conversation, add_generation_prompt=True)
    pixel_count = image_width * image_height
    hf_inputs = processor(
        images=image,
        text=prompt,
        return_tensors="pt",
        min_pixels=pixel_count,
        max_pixels=pixel_count,
    )
    qeff_inputs = {key: value.clone() if isinstance(value, torch.Tensor) else value for key, value in hf_inputs.items()}
    if qeff_model is not None:
        qeff_inputs = qeff_model.prepare_inputs_for_generation(
            inputs=qeff_inputs,
            prefill_seq_len=model_config["prompt_len"],
            batch_size=model_config["batch_size"],
        )
    if "pixel_values" in hf_inputs:
        hf_inputs["pixel_values"] = hf_inputs["pixel_values"].to(torch.float32)
    if "pixel_values" in qeff_inputs:
        qeff_inputs["pixel_values"] = qeff_inputs["pixel_values"].to(torch.float32)
    return hf_inputs, qeff_inputs


def _assert_vlm_weight_free_graph(model_type: str, language_onnx: Path) -> None:
    graph = onnx.load(str(language_onnx), load_external_data=False)
    function_names = [function.name for function in graph.functions]
    expected_functions = {
        "qwen2_5_vl": ("QEffQwen2_5_VLDecoderLayer",),
        "qwen3_vl": ("QEffQwen3VLTextDecoderLayer",),
        "qwen3_vl_moe": ("QEffQwen3VLMoeTextDecoderLayer",),
        "qwen3_5": (
            "QEffQwen3_5LinearAttentionDecoderLayer",
            "QEffQwen3_5FullAttentionDecoderLayer",
        ),
        "qwen3_5_moe": (
            "QEffQwen3_5MoeLinearAttentionDecoderLayer",
            "QEffQwen3_5MoeFullAttentionDecoderLayer",
        ),
    }[model_type]
    missing_functions = [
        expected for expected in expected_functions if not any(expected in name for name in function_names)
    ]
    assert not missing_functions, f"Missing subfunctions {missing_functions} in {language_onnx}: {function_names}"

    retained_outputs = {output.name for output in graph.graph.output if output.name.endswith("_RetainedState")}
    assert retained_outputs, f"No retained-state outputs found in {language_onnx}"
    if model_type in {"qwen3_5", "qwen3_5_moe"}:
        assert any(name.startswith("conv_state.") for name in retained_outputs)
        assert any(name.startswith("recurrent_state.") for name in retained_outputs)
    else:
        assert any(name.startswith("past_key.") for name in retained_outputs)
        assert any(name.startswith("past_value.") for name in retained_outputs)


def _assert_weight_spec(onnx_path: Path) -> None:
    weight_spec = onnx_path.with_name("weight_spec.json")
    assert weight_spec.is_file(), f"Missing weight-free spec next to {onnx_path}"


def _update_retained_states(target_inputs, source_outputs, layer_types):
    for layer_idx, layer_type in enumerate(layer_types):
        state_names = (
            (f"past_key.{layer_idx}", f"past_value.{layer_idx}")
            if layer_type == "full_attention"
            else (f"conv_state.{layer_idx}", f"recurrent_state.{layer_idx}")
        )
        for state_name in state_names:
            retained_name = f"{state_name}_RetainedState"
            if retained_name in source_outputs:
                target_inputs[state_name] = np.array(source_outputs[retained_name], copy=True)


def _run_qwen35_three_qpc(qeff_model, qeff_inputs, qpc_paths, layer_types, generation_len):
    vision_session = QAICInferenceSession(qpc_paths["vision_qpc_path"])
    prefill_session = None
    decode_session = None
    try:
        inputs = {
            name: np.asarray(value.detach().cpu() if isinstance(value, torch.Tensor) else value)
            for name, value in qeff_inputs.items()
        }
        vision_inputs = {name: inputs[name] for name in vision_session.input_names if name in inputs}
        for name in ("pixel_values", "image_embeds"):
            if name in vision_inputs:
                vision_inputs[name] = vision_inputs[name].astype(np.float16)
        vision_outputs = vision_session.run(vision_inputs)
        vision_session.deactivate()
        vision_session = None

        prefill_session = QAICInferenceSession(qpc_paths["lang_prefill_qpc_path"])

        prefill_binding = prefill_session.binding_index_map["input_ids"]
        prefill_seq_len = prefill_session.bindings[prefill_binding].dims[1]
        input_ids = inputs["input_ids"]
        padded_len = int(np.ceil(input_ids.shape[1] / prefill_seq_len) * prefill_seq_len)
        if padded_len != input_ids.shape[1]:
            pad_token_id = int(input_ids[0, -1])
            inputs["input_ids"] = np.pad(
                input_ids, ((0, 0), (0, padded_len - input_ids.shape[1])), constant_values=pad_token_id
            )
            inputs["position_ids"] = np.pad(
                inputs["position_ids"], ((0, 0), (0, 0), (0, padded_len - input_ids.shape[1])), constant_values=-1
            )

        lang_inputs = {
            name: inputs[name] for name in prefill_session.input_names if name in inputs and name not in vision_inputs
        }
        lang_inputs.pop("attention_mask", None)
        lang_inputs["image_idx"] = np.array([[0]], dtype=np.int64)
        for name, value in vision_outputs.items():
            if name in prefill_session.input_names:
                lang_inputs[name] = value

        prefill_session.set_buffers(vision_outputs)
        chunk_inputs = dict(lang_inputs)
        for chunk_idx in range(padded_len // prefill_seq_len):
            start = chunk_idx * prefill_seq_len
            end = start + prefill_seq_len
            chunk_inputs["input_ids"] = lang_inputs["input_ids"][:, start:end]
            chunk_inputs["position_ids"] = lang_inputs["position_ids"][..., start:end]
            prefill_outputs = prefill_session.run(chunk_inputs)
            _update_retained_states(chunk_inputs, prefill_outputs, layer_types)
            if "image_idx_output" in prefill_outputs:
                chunk_inputs["image_idx"] = prefill_outputs["image_idx_output"]

        prefill_session.deactivate()
        prefill_session = None

        decode_session = QAICInferenceSession(qpc_paths["lang_decode_qpc_path"])
        generated_tokens = [np.argmax(prefill_outputs["logits"], axis=-1)[:, -1]]
        decode_inputs = {
            "input_ids": generated_tokens[-1].reshape(1, 1),
            "position_ids": np.max(inputs["position_ids"], axis=-1, keepdims=True) + 1,
        }
        _update_retained_states(decode_inputs, prefill_outputs, layer_types)
        for name in ("image_idx_output", "vision_embeds_RetainedState"):
            if name in prefill_outputs:
                decode_inputs[
                    name.removesuffix("_RetainedState") if name.endswith("_RetainedState") else "image_idx"
                ] = prefill_outputs[name]

        for _ in range(generation_len - 1):
            decode_outputs = decode_session.run(decode_inputs)
            generated_tokens.append(np.argmax(decode_outputs["logits"], axis=-1)[:, -1])
            decode_inputs["input_ids"] = generated_tokens[-1].reshape(1, 1)
            decode_inputs["position_ids"] = decode_inputs["position_ids"] + 1
            _update_retained_states(decode_inputs, decode_outputs, layer_types)
            if "image_idx_output" in decode_outputs:
                decode_inputs["image_idx"] = decode_outputs["image_idx_output"]
            if "vision_embeds_RetainedState" in decode_outputs:
                decode_inputs["vision_embeds"] = decode_outputs["vision_embeds_RetainedState"]
        return np.stack(generated_tokens[:generation_len], axis=1)
    finally:
        for session in (vision_session, prefill_session, decode_session):
            if session is not None:
                session.deactivate()


@pytest.mark.weight_free
@pytest.mark.weight_free_export
@pytest.mark.parametrize("model_type", WEIGHT_FREE_VLM_QAIC_MODEL_PARAMS)
def test_weight_free_vlm_dummy_cpu_export_and_multispec_structure(model_type, tmp_export_dir):
    """Export every supported VLM from a synthesized checkpoint on CPU."""
    model_config = _VLM_MODEL_CONFIGS[model_type]
    checkpoint_dir = tmp_export_dir / f"{model_type}_cpu_checkpoint"
    tiny_model_types = {"qwen2_5_vl", "qwen3_vl"}
    dummy_model_id = WEIGHT_FREE_VLM_QAIC_MODEL_IDS.get(model_type) if model_type in tiny_model_types else None
    model, model_config = _build_vlm_test_checkpoint(model_type, checkpoint_dir, dummy_model_id)
    qeff_model = QEFFAutoModelForImageTextToText.from_pretrained(
        str(checkpoint_dir),
        kv_offload=True,
        weight_free=True,
        dtype=torch.float32,
    )
    skip_vision = model_type == "qwen2_5_vl"
    # Skipped due to an ONNX issue in the vision module.

    export_paths = qeff_model.export(
        tmp_export_dir / f"{model_type}_cpu_export",
        use_onnx_subfunctions=True,
        offload_pt_weights=False,
        prefill_seq_len=model_config["prompt_len"],
        ctx_len=model_config["ctx_len"],
        skip_vision=skip_vision,
    )
    language_onnx = Path(export_paths[-1])
    assert language_onnx.is_file()
    _assert_weight_spec(language_onnx)
    _assert_vlm_weight_free_graph(model_type, language_onnx)
    if skip_vision:
        return

    vision_onnx = Path(export_paths[0])
    assert vision_onnx.is_file()
    _assert_weight_spec(vision_onnx)

    specializations, _ = qeff_model.model.get_specializations(
        batch_size=model_config["batch_size"],
        prefill_seq_len=model_config["prompt_len"],
        ctx_len=model_config["ctx_len"],
        height=[224, VLM_IMAGE_HEIGHT],
        width=[224, VLM_IMAGE_WIDTH],
        num_frames=[1, 1],
        kv_offload=True,
    )
    assert len(specializations["vision"]) == 2
    assert len({(spec["grid_height"], spec["grid_width"]) for spec in specializations["vision"]}) == 2


def _load_vlm_hf_reference(model_id: str):
    config = AutoConfig.from_pretrained(model_id, trust_remote_code=True)
    config.text_config.num_hidden_layers = 1
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


def _prepare_vlm_inputs(processor: AutoProcessor, image: Image.Image) -> dict:
    process_vision_info = pytest.importorskip("qwen_vl_utils").process_vision_info
    messages = [
        [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": VLM_PROMPT},
                ],
            }
        ]
    ]
    texts = [processor.apply_chat_template(message, tokenize=False, add_generation_prompt=True) for message in messages]
    image_inputs, video_inputs = process_vision_info(messages)
    return dict(processor(text=texts, images=image_inputs, videos=video_inputs, padding=True, return_tensors="pt"))


def _run_vlm_hf_reference(model, inputs: dict) -> np.ndarray:
    inputs = {
        name: value.to(dtype=torch.float32) if torch.is_floating_point(value) else value
        for name, value in inputs.items()
    }
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

    image = Image.new("RGB", (VLM_IMAGE_WIDTH, VLM_IMAGE_HEIGHT), color=(127, 127, 127))
    hf_tokens = _run_vlm_hf_reference(hf_model, _prepare_vlm_inputs(processor, image))
    qeff_inputs = _prepare_vlm_inputs(processor, image)
    qeff_inputs = qeff_model.model.prepare_inputs_for_generation(
        inputs=qeff_inputs,
        prefill_seq_len=VLM_PREFILL_SEQ_LEN,
        batch_size=1,
    )

    qpc_paths = qeff_model.compile(
        compile_dir=str(tmp_export_dir / f"{model_type}_combined"),
        batch_size=1,
        prefill_seq_len=VLM_PREFILL_SEQ_LEN,
        ctx_len=VLM_CTX_LEN,
        height=VLM_IMAGE_HEIGHT,
        width=VLM_IMAGE_WIDTH,
        num_cores=VLM_NUM_CORES,
        num_devices=1,
        mxfp6_matmul=False,
        mxint8_kv_cache=False,
        use_onnx_subfunctions=True,
        offload_pt_weights=False,
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
@pytest.mark.embedding_audio_model
@pytest.mark.parametrize("model_type,model_id,pooling", WEIGHT_FREE_EMBEDDING_MODEL_PARAMS)
def test_weight_free_embedding_hw_hf_parity(model_type, model_id, pooling, tmp_export_dir):
    """HF embedding output matches weight-free embedding output on QAIC."""
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
        model_hf = AutoModel.from_pretrained(
            model_id,
            trust_remote_code=True,
            attn_implementation="eager",
            low_cpu_mem_usage=False,
            torch_dtype=torch.float32,
        )
        model_hf.eval()
    except (ImportError, OSError, RuntimeError, ValueError) as exc:
        if model_type in {"jina", "nomic"}:
            raise
        skip_on_model_fetch_error(exc, model_id)

    inputs = tokenizer("hello world", return_tensors="pt")
    with torch.no_grad():
        hf_hidden_states = model_hf(**inputs).last_hidden_state
        hf_output = POOLING_MAP[pooling](hf_hidden_states, inputs["attention_mask"]).cpu().numpy()

    qeff_model = QEFFAutoModel.from_pretrained(model_id, pooling=pooling, weight_free=True)
    onnx_path = exported_onnx_path(
        qeff_model.export(
            tmp_export_dir / f"{model_type}_export",
            dynamo=True,
            offload_pt_weights=False,
        )
    )
    qeff_model.compile(
        onnx_path=str(onnx_path),
        compile_dir=str(tmp_export_dir / f"{model_type}_compile"),
        seq_len=32,
        batch_size=1,
        num_cores=16,
    )
    qaic_output = qeff_model.generate(inputs=inputs)

    assert qaic_output is not None, f"Weight-free QAIC generate returned None for {model_type}"
    qaic_embeddings = qaic_output["output"]
    if qaic_embeddings.ndim == 3:
        qaic_embeddings = qaic_embeddings[:, 0, :]
    mad = np.mean(np.abs(hf_output - qaic_embeddings))
    assert mad <= 1e-2, f"HF PT vs weight-free QAIC parity failed for {model_type}: MAD={mad}"


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


@pytest.mark.weight_free
@pytest.mark.on_qaic
@pytest.mark.multimodal
@pytest.mark.parametrize("model_type", WEIGHT_FREE_VLM_QAIC_MODEL_PARAMS)
def test_weight_free_vlm_hf_qaic_parity(model_type, tmp_export_dir):
    """Validate one compact weight-free VLM export, compile, and HF/QAIC parity run."""
    model_config = _VLM_MODEL_CONFIGS[model_type]
    model_id = WEIGHT_FREE_VLM_QAIC_MODEL_IDS[model_type]
    checkpoint_dir = tmp_export_dir / f"{model_type}_checkpoint"

    try:
        processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
        hf_model, model_config = _build_vlm_test_checkpoint(model_type, checkpoint_dir, model_id)
    except Exception as exc:
        skip_on_model_fetch_error(exc, model_id)

    image_size = (VLM_IMAGE_WIDTH, VLM_IMAGE_HEIGHT)
    skip_vision = model_type == "qwen2_5_vl"
    compile_prefill_seq_len = model_config["prompt_len"]
    hf_inputs, _ = _prepare_vlm_test_inputs(
        processor,
        hf_model,
        None,
        model_config,
        image_size,
        skip_vision=skip_vision,
    )

    with torch.inference_mode():
        hf_output = hf_model.generate(
            **hf_inputs,
            min_new_tokens=VLM_GENERATION_LEN,
            max_new_tokens=VLM_GENERATION_LEN,
            do_sample=False,
        )
    hf_tokens = hf_output[0, hf_inputs["input_ids"].shape[1] :].cpu().numpy()

    qeff_model = QEFFAutoModelForImageTextToText.from_pretrained(
        str(checkpoint_dir),
        kv_offload=True,
        weight_free=True,
        dtype=torch.float32,
    )
    hf_inputs, qeff_inputs = _prepare_vlm_test_inputs(
        processor,
        hf_model,
        qeff_model.model,
        model_config,
        image_size,
        skip_vision=skip_vision,
    )
    export_paths = qeff_model.export(
        tmp_export_dir / "weight_free_vlm_export",
        use_onnx_subfunctions=True,
        offload_pt_weights=False,
        prefill_seq_len=compile_prefill_seq_len,
        ctx_len=model_config["ctx_len"],
        skip_vision=skip_vision,
    )
    language_onnx = Path(export_paths[-1])
    assert language_onnx.is_file()
    _assert_weight_spec(language_onnx)
    _assert_vlm_weight_free_graph(model_type, language_onnx)
    if not skip_vision:
        vision_onnx = Path(export_paths[0])
        assert vision_onnx.is_file()
        _assert_weight_spec(vision_onnx)

    common_compile_kwargs = {
        "img_size": model_config["img_size"],
        "ctx_len": model_config["ctx_len"],
        "num_devices": 1,
        "num_cores": 16,
        "mxfp6_matmul": False,
        "use_onnx_subfunctions": True,
    }
    if skip_vision:
        common_compile_kwargs["skip_vision"] = True
    else:
        common_compile_kwargs.update(
            {
                "height": image_size[1],
                "width": image_size[0],
                "mm_processor_kwargs": {
                    "min_pixels": image_size[0] * image_size[1],
                    "max_pixels": image_size[0] * image_size[1],
                },
            }
        )

    if model_type in {"qwen3_5", "qwen3_5_moe"}:
        vision_qpc = qeff_model.compile(
            **common_compile_kwargs,
            aic_enable_depth_first=False,
            split_model_io=True,
            prefill_seq_len=compile_prefill_seq_len,
            skip_lang=True,
            skip_vision=False,
        )
        prefill_qpc = qeff_model.compile(
            **common_compile_kwargs,
            retain_full_kv=True,
            split_model_io=True,
            aic_enable_depth_first=False,
            prefill_seq_len=compile_prefill_seq_len,
            prefill_only=True,
            enable_chunking=True,
            skip_vision=True,
        )
        decode_qpc = qeff_model.compile(
            **common_compile_kwargs,
            retain_full_kv=True,
            split_model_io=True,
            aic_enable_depth_first=True,
            prefill_seq_len=1,
            prefill_only=False,
            skip_vision=True,
        )
        layer_types = list(qeff_model.model.config.text_config.layer_types)
        qaic_tokens = _run_qwen35_three_qpc(
            qeff_model,
            qeff_inputs,
            {
                "vision_qpc_path": vision_qpc["vision_qpc_path"],
                "lang_prefill_qpc_path": prefill_qpc["lang_prefill_qpc_path"],
                "lang_decode_qpc_path": decode_qpc["lang_decode_qpc_path"],
            },
            layer_types,
            VLM_GENERATION_LEN,
        )[0]
    else:
        compile_kwargs = {**common_compile_kwargs, "prefill_seq_len": compile_prefill_seq_len}
        qeff_model.compile(**compile_kwargs)
        qaic_output = qeff_model.generate(inputs=qeff_inputs, generation_len=VLM_GENERATION_LEN)
        assert qaic_output.generated_ids is not None
        qaic_tokens = np.asarray(qaic_output.generated_ids)[0, :VLM_GENERATION_LEN]
    assert np.array_equal(hf_tokens[:VLM_GENERATION_LEN], qaic_tokens), (
        f"Weight-free HF/QAIC parity failed for {model_type}: HF={hf_tokens.tolist()}, QAIC={qaic_tokens.tolist()}"
    )


@pytest.mark.weight_free
@pytest.mark.on_qaic
@pytest.mark.multimodal
@pytest.mark.parametrize("model_type", WEIGHT_FREE_VLM_QAIC_MODEL_PARAMS)
def test_weight_free_vlm_multi_specialization_hf_qaic_parity(model_type, tmp_export_dir):
    """Compile two vision resolutions in one QPC and run the dummy VLM on QAIC."""
    if model_type == "qwen2_5_vl":
        pytest.skip("Qwen 2.5-VL vision QPC is deferred because QAIC reports a Conv3D bias-shape error")

    model_config = _VLM_MODEL_CONFIGS[model_type]
    model_id = WEIGHT_FREE_VLM_QAIC_MODEL_IDS[model_type]
    checkpoint_dir = tmp_export_dir / f"{model_type}_multi_spec_checkpoint"
    image_size = (VLM_IMAGE_WIDTH, VLM_IMAGE_HEIGHT)
    specialization_heights = [224, VLM_IMAGE_HEIGHT]
    specialization_widths = [224, VLM_IMAGE_WIDTH]
    specialization_frames = [1, 1]

    try:
        processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
        hf_model, model_config = _build_vlm_test_checkpoint(model_type, checkpoint_dir, model_id)
    except Exception as exc:
        skip_on_model_fetch_error(exc, model_id)

    qeff_model = QEFFAutoModelForImageTextToText.from_pretrained(
        str(checkpoint_dir),
        kv_offload=True,
        weight_free=True,
        dtype=torch.float32,
    )
    hf_inputs, qeff_inputs = _prepare_vlm_test_inputs(
        processor,
        hf_model,
        qeff_model.model,
        model_config,
        image_size,
    )

    with torch.inference_mode():
        hf_output = hf_model.generate(
            **hf_inputs,
            min_new_tokens=VLM_GENERATION_LEN,
            max_new_tokens=VLM_GENERATION_LEN,
            do_sample=False,
        )
    hf_tokens = hf_output[0, hf_inputs["input_ids"].shape[1] :].cpu().numpy()

    export_paths = qeff_model.export(
        tmp_export_dir / f"{model_type}_multi_spec_export",
        use_onnx_subfunctions=True,
        offload_pt_weights=False,
        prefill_seq_len=model_config["prompt_len"],
        ctx_len=model_config["ctx_len"],
    )
    vision_onnx, language_onnx = (Path(export_paths[0]), Path(export_paths[-1]))
    _assert_weight_spec(vision_onnx)
    _assert_weight_spec(language_onnx)
    _assert_vlm_weight_free_graph(model_type, language_onnx)

    specializations, _ = qeff_model.model.get_specializations(
        batch_size=model_config["batch_size"],
        prefill_seq_len=model_config["prompt_len"],
        ctx_len=model_config["ctx_len"],
        height=specialization_heights,
        width=specialization_widths,
        num_frames=specialization_frames,
        kv_offload=True,
    )
    assert len(specializations["vision"]) == 2
    assert len({(spec["grid_height"], spec["grid_width"]) for spec in specializations["vision"]}) == 2

    compile_kwargs = {
        "img_size": model_config["img_size"],
        "prefill_seq_len": model_config["prompt_len"],
        "ctx_len": model_config["ctx_len"],
        "height": specialization_heights,
        "width": specialization_widths,
        "num_frames": specialization_frames,
        "mm_processor_kwargs": {
            "min_pixels": 224 * 224,
            "max_pixels": VLM_IMAGE_WIDTH * VLM_IMAGE_HEIGHT,
        },
        "num_devices": 1,
        "num_cores": 16,
        "mxfp6_matmul": False,
        "use_onnx_subfunctions": True,
    }
    if model_type in {"qwen3_5", "qwen3_5_moe"}:
        vision_qpc = qeff_model.compile(
            **compile_kwargs,
            aic_enable_depth_first=False,
            split_model_io=True,
            skip_lang=True,
            skip_vision=False,
        )
        prefill_qpc = qeff_model.compile(
            **compile_kwargs,
            retain_full_kv=True,
            split_model_io=True,
            aic_enable_depth_first=False,
            prefill_only=True,
            enable_chunking=True,
            skip_vision=True,
        )
        decode_qpc = qeff_model.compile(
            **compile_kwargs,
            retain_full_kv=True,
            split_model_io=True,
            aic_enable_depth_first=True,
            prefill_seq_len=1,
            prefill_only=False,
            skip_vision=True,
        )
        qaic_tokens = _run_qwen35_three_qpc(
            qeff_model,
            qeff_inputs,
            {
                "vision_qpc_path": vision_qpc["vision_qpc_path"],
                "lang_prefill_qpc_path": prefill_qpc["lang_prefill_qpc_path"],
                "lang_decode_qpc_path": decode_qpc["lang_decode_qpc_path"],
            },
            list(qeff_model.model.config.text_config.layer_types),
            VLM_GENERATION_LEN,
        )[0]
    else:
        qeff_model.compile(**compile_kwargs)
        qaic_output = qeff_model.generate(
            inputs=qeff_inputs,
            tokenizer=getattr(processor, "tokenizer", None),
            generation_len=VLM_GENERATION_LEN,
            multi_specs=True,
            num_frames=1,
        )
        qaic_tokens = np.asarray(qaic_output.generated_ids)[0, :VLM_GENERATION_LEN]
    assert np.array_equal(hf_tokens[:VLM_GENERATION_LEN], qaic_tokens), (
        f"Weight-free multi-specialization HF/QAIC parity failed for {model_type}: "
        f"HF={hf_tokens.tolist()}, QAIC={qaic_tokens.tolist()}"
    )

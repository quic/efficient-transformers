# -----------------------------------------------------------------------------
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
#
# ----------------------------------------------------------------------------

"""Token-level parity test for the Gemma4-MoE disaggregated prefill/decode DMA path.
pytest -m "on_qaic and multimodal" tests/transformers/disaggregated/test_gemma4_moe_disagg_kv_share_w_hf_fp32.py
"""

import copy
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image
from transformers import AutoConfig, AutoModelForCausalLM, AutoModelForImageTextToText, AutoProcessor

from QEfficient import QEFFAutoModelForImageTextToText
from QEfficient.base.onnx_transforms import FP16ClipTransform
from QEfficient.generation.cloud_infer import QAICInferenceSession

MODEL_NAME = "google/gemma-4-26B-A4B-it"
SYSTEM_PROMPT = "You are a helpful assistant."
NUM_HIDDEN_LAYERS = 6
VISION_DEPTH = 2
PREFILL_SEQ_LEN = 296
CTX_LEN = 4096
BATCH_SIZE = 1
GENERATION_LEN = 30
IMAGE_SIZE = (536, 354)
TEXT_PROMPT = "Can you describe this image in detail?"

PREFILL_NUM_DEVICES = 4
DECODE_NUM_DEVICES = 2
PREFILL_MDP_PARTITIONS = 2

# The Gemma4 vision encoder forward binds exactly (pixel_values, image_position_ids);
# everything else feeds the lang QPCs. Extra keys are tolerated (routed if the processor
# emits them, ignored otherwise).
VISION_INPUT_KEYS = {
    "pixel_values",
    "image_position_ids",
    "image_masks",
    "image_input_idx",
    "valid_idx",
    "aspect_ratio_ids",
    "aspect_ratio_mask",
}
VISION_FP16_KEYS = {"pixel_values", "image_masks"}
VISION_OUTPUTS = ("vision_embeds",)

# Minimal Gemma4 chat template (from the sample's gemma4_utils); only used if neither the
# processor nor the tokenizer ships one.
CHAT_TEMPLATE = """
{%- for message in messages %}
    {%- if loop.index0 == 0 %}
        {{- bos_token }}
    {%- endif %}
    {{- '<|turn|>' + message['role'] + '\n' }}
    {%- if message['content'] is string %}
        {{- message['content'] }}
    {%- else %}
        {%- for content in message['content'] %}
            {%- if content['type'] == 'image' %}
                {{- image_token }}
            {%- elif content['type'] == 'text' %}
                {{- content['text'] }}
            {%- endif %}
        {%- endfor %}
    {%- endif %}
    {{- '<turn|>\n' }}
{%- endfor %}
{%- if add_generation_prompt %}
    {{- '<|turn|>assistant\n' }}
{%- endif %}
"""


def _assert_onnx_path(onnx_path, label: str) -> Path:
    assert onnx_path is not None, f"{label} compile did not set an ONNX path"
    onnx_path = Path(onnx_path)
    assert onnx_path.is_file(), f"{label} ONNX path does not exist: {onnx_path}"
    assert onnx_path.suffix == ".onnx", f"{label} path is not an ONNX file: {onnx_path}"
    return onnx_path.resolve()


def _assert_distinct_onnx_paths(onnx_paths: dict[str, Path]):
    unique_paths = {str(path) for path in onnx_paths.values()}
    assert len(unique_paths) == len(onnx_paths), f"Expected distinct ONNX paths per compile, got: {onnx_paths}"


def _load_hf_model_from_pretrained(config):
    try:
        model = AutoModelForImageTextToText.from_pretrained(
            MODEL_NAME,
            config=config,
            attn_implementation="eager",
            trust_remote_code=True,
            torch_dtype=config.torch_dtype,
        )
    except ValueError:
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME,
            config=config,
            attn_implementation="eager",
            trust_remote_code=True,
            torch_dtype=config.torch_dtype,
        )
    model.eval()
    return model


def _apply_reduced_layer_config(config, num_lang_layers: int, num_vision_layers: int):
    """Truncate depth so the compile/run stays cheap.

    Mirrors ``gemma4_example._apply_reduced_layer_config``: slice the pretrained
    ``layer_types`` (rather than regenerate the sliding/full pattern) so the truncated
    config stays consistent with whatever pattern the checkpoint actually ships.
    """
    config.text_config.num_hidden_layers = num_lang_layers
    config.vision_config.num_hidden_layers = num_vision_layers

    if hasattr(config.text_config, "layer_types") and config.text_config.layer_types:
        config.text_config.layer_types = config.text_config.layer_types[:num_lang_layers]

    if hasattr(config.text_config, "num_kv_shared_layers"):
        # KV sharing to avoid invalid first_kv_shared_layer_idx=0 edge cases.
        config.text_config.num_kv_shared_layers = 0

    return config


def _build_config(dtype: str = "float32"):
    """Load the real config, then truncate depth so the compile/run stays cheap."""
    config = AutoConfig.from_pretrained(MODEL_NAME, trust_remote_code=True)
    config.dtype = dtype
    config.torch_dtype = getattr(torch, dtype)

    config = _apply_reduced_layer_config(
        config,
        num_lang_layers=NUM_HIDDEN_LAYERS,
        num_vision_layers=VISION_DEPTH,
    )

    text_config = config.text_config if hasattr(config, "text_config") else config
    text_config.dtype = dtype
    text_config.torch_dtype = getattr(torch, dtype)
    return config


def _remove_fp16clip_transform(qeff_model: QEFFAutoModelForImageTextToText):
    """Strip FP16ClipTransform from both sub-models so the fp32 export stays true fp32.

    Mirrors ``gemma4_utils.remove_fp16clip_transform_if_disabled`` for the disabled case;
    without it the fp32 QAIC path clips activations and drifts from the HF fp32 reference.
    """
    for sub in (getattr(qeff_model, "lang_model", None), getattr(qeff_model, "vision_model", None)):
        if sub is not None and hasattr(sub, "_onnx_transforms"):
            sub._onnx_transforms = [t for t in sub._onnx_transforms if t is not FP16ClipTransform]


def _prepare_messages(image: Image.Image) -> list:
    return [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": TEXT_PROMPT},
            ],
        }
    ]


def _resolve_chat_template(processor, tokenizer) -> str:
    return getattr(processor, "chat_template", None) or getattr(tokenizer, "chat_template", None) or CHAT_TEMPLATE


def _prepare_processor_inputs(processor, chat_template: str, messages: list) -> dict:
    return dict(
        processor.apply_chat_template(
            messages,
            chat_template=chat_template,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        )
    )


def _get_next_token_ids(logits: np.ndarray) -> np.ndarray:
    logits = np.asarray(logits)
    return logits[:, -1, :].argmax(axis=-1).astype(np.int64)


def _run_hf_torch_fp32(model, inputs: dict) -> np.ndarray:
    model = model.to(dtype=torch.float32).eval()
    gen_inputs = {
        name: value.to(dtype=torch.float32) if torch.is_floating_point(value) else value
        for name, value in inputs.items()
    }

    with torch.inference_mode():
        # min_new_tokens == max_new_tokens forces exactly GENERATION_LEN tokens: HF would
        # otherwise stop early at an EOS token (max_new_tokens is only an upper bound), while
        # the QAIC decode loop always runs a fixed GENERATION_LEN with no EOS check -- the
        # length mismatch would break the token-for-token shape/parity comparison.
        outputs = model.generate(
            **gen_inputs,
            max_new_tokens=GENERATION_LEN,
            min_new_tokens=GENERATION_LEN,
            do_sample=False,
        )

    prompt_len = gen_inputs["input_ids"].shape[-1]
    return outputs[:, prompt_len:].detach().cpu().numpy()


def _run_disagg_kv_share_qaic_generation(
    processor,
    common_inputs: dict,
    vision_session: QAICInferenceSession,
    prefill_session: QAICInferenceSession,
    decode_session: QAICInferenceSession,
) -> np.ndarray:
    inputs = {
        name: value.clone() if isinstance(value, torch.Tensor) else copy.deepcopy(value)
        for name, value in common_inputs.items()
    }

    pad_token_id = processor.tokenizer.pad_token_id or 1
    input_ids_length = inputs["input_ids"].shape[1]
    num_chunks = -(input_ids_length // -PREFILL_SEQ_LEN)
    padded_len = num_chunks * PREFILL_SEQ_LEN
    inputs["input_ids"] = torch.nn.functional.pad(
        inputs["input_ids"],
        (0, padded_len - input_ids_length),
        "constant",
        pad_token_id,
    )
    inputs["attention_mask"] = torch.nn.functional.pad(
        inputs["attention_mask"],
        (0, padded_len - input_ids_length),
        "constant",
        0,
    )
    if "mm_token_type_ids" in inputs:
        inputs["mm_token_type_ids"] = torch.nn.functional.pad(
            inputs["mm_token_type_ids"],
            (0, padded_len - input_ids_length),
            "constant",
            0,
        )
    inputs = {name: np.array(value) for name, value in inputs.items()}
    vision_inputs = {name: value for name, value in inputs.items() if name in VISION_INPUT_KEYS}
    vision_inputs.update(
        {name: vision_inputs[name].astype("float16") for name in VISION_FP16_KEYS if name in vision_inputs}
    )
    vision_outputs = vision_session.run(vision_inputs)
    vision_session.deactivate()

    lang_inputs = {name: value for name, value in inputs.items() if name not in vision_inputs}
    if "position_ids" in inputs:
        lang_inputs["position_ids"] = inputs["position_ids"]
        lang_inputs.pop("attention_mask", None)
    else:
        lang_inputs["position_ids"] = np.where(lang_inputs.pop("attention_mask"), np.arange(padded_len), -1)

    # mm_token_type_ids is a per-chunk lang input; synthesize zeros if the processor omitted
    # it so prefill slicing and the binding are always satisfied.
    if "mm_token_type_ids" not in lang_inputs:
        lang_inputs["mm_token_type_ids"] = np.zeros((BATCH_SIZE, padded_len), dtype=np.int64)

    lang_inputs["image_idx"] = np.array([[0]])

    assert "image_idx" in prefill_session.binding_index_map, "image_idx not a compiled prefill input binding"
    decode_has_image_idx = "image_idx" in decode_session.binding_index_map

    vision_persist = {name: vision_outputs[name] for name in VISION_OUTPUTS if name in vision_outputs}
    prefill_session.set_persistent_inputs(vision_persist)

    # The decode QPC binds mm_token_type_ids (seq_len=1 ignores its value, but the pooled
    # np_run path wires only the bindings it is handed), so register constant zeros to satisfy
    # it; register vision_embeds on decode too when it binds the name.
    decode_persist = {"mm_token_type_ids": np.zeros((BATCH_SIZE, 1), dtype=np.int64), **vision_persist}
    decode_session.set_persistent_inputs(
        {name: value for name, value in decode_persist.items() if name in decode_session.binding_index_map}
    )

    # Hybrid: kv_cache_info carries mixed sliding-window and full-attention 4-D shapes.
    kv_caches = [np.zeros(shape, dtype=dtype) for (shape, dtype) in decode_session.kv_cache_info]

    # ---- Prefill (producer, SERIAL): image_idx threads chunk-to-chunk ----
    # Only the LAST chunk wires the DMA handoff into kv_caches (earlier chunks just accumulate
    # KV on-device). np_run_pipeline selects the hybrid full slicing spec internally.
    chunk_inputs = dict(lang_inputs)
    exec_idx = None
    for chunk_idx in range(num_chunks):
        chunk_inputs["input_ids"] = lang_inputs["input_ids"][
            :, chunk_idx * PREFILL_SEQ_LEN : (chunk_idx + 1) * PREFILL_SEQ_LEN
        ]
        chunk_inputs["position_ids"] = lang_inputs["position_ids"][
            ..., chunk_idx * PREFILL_SEQ_LEN : (chunk_idx + 1) * PREFILL_SEQ_LEN
        ]
        chunk_inputs["mm_token_type_ids"] = lang_inputs["mm_token_type_ids"][
            ..., chunk_idx * PREFILL_SEQ_LEN : (chunk_idx + 1) * PREFILL_SEQ_LEN
        ]
        last_chunk = chunk_idx == num_chunks - 1
        exec_idx = prefill_session.np_run_pipeline(
            chunk_inputs,
            last_chunk=last_chunk,
            kv_cache_buffers=kv_caches if last_chunk else None,
        )
        prefill_session.complete_inf(exec_idx, is_prefill=True)
        chunk_inputs["image_idx"] = prefill_session.get_outputs(index=exec_idx)["image_idx_output"]

    prefill_out = prefill_session.get_outputs(index=exec_idx)
    generated_ids = [_get_next_token_ids(prefill_out["logits"])]

    # ---- Decode (consumer): re-point DMA descriptor at kv_caches EVERY step ----
    decode_kv_map = decode_session.decode_buff_map + decode_session.decode_rs_kv_only_buff_map
    position_ids = np.max(lang_inputs["position_ids"], axis=-1, keepdims=True) + 1
    decode_inputs = {
        "input_ids": generated_ids[-1].reshape(BATCH_SIZE, 1),
        "position_ids": position_ids,
    }
    if decode_has_image_idx:
        decode_inputs["image_idx"] = prefill_out["image_idx_output"]

    for _ in range(GENERATION_LEN - 1):
        decode_session.set_data_for_kv_handoff(
            kv_caches + kv_caches,
            [("batch_index", 0), ("ctx_start", 0)],
            index=decode_session.decode_execObj_idx,
            buff_map=decode_kv_map,
        )
        exec_idx = decode_session.np_run(decode_inputs, is_prefill=False)
        decode_session.complete_inf(exec_idx, is_prefill=False)
        decode_outputs = decode_session.get_outputs(index=exec_idx)
        generated_ids.append(_get_next_token_ids(decode_outputs["logits"]))
        position_ids = position_ids + 1
        decode_inputs = {
            "input_ids": generated_ids[-1].reshape(BATCH_SIZE, 1),
            "position_ids": position_ids,
        }
        if decode_has_image_idx:
            decode_inputs["image_idx"] = decode_outputs["image_idx_output"]

    return np.stack(generated_ids, axis=1)


def _run_disagg_baseline_numpy_copy_generation(
    processor,
    common_inputs: dict,
    vision_session: QAICInferenceSession,
    prefill_session: QAICInferenceSession,
    decode_session: QAICInferenceSession,
) -> np.ndarray:
    """Baseline disaggregated generation with an explicit numpy KV copy each step."""
    inputs = {
        name: value.clone() if isinstance(value, torch.Tensor) else copy.deepcopy(value)
        for name, value in common_inputs.items()
    }

    pad_token_id = processor.tokenizer.pad_token_id or 1
    input_ids_length = inputs["input_ids"].shape[1]
    num_chunks = -(input_ids_length // -PREFILL_SEQ_LEN)
    padded_len = num_chunks * PREFILL_SEQ_LEN
    inputs["input_ids"] = torch.nn.functional.pad(
        inputs["input_ids"],
        (0, padded_len - input_ids_length),
        "constant",
        pad_token_id,
    )
    inputs["attention_mask"] = torch.nn.functional.pad(
        inputs["attention_mask"],
        (0, padded_len - input_ids_length),
        "constant",
        0,
    )
    if "mm_token_type_ids" in inputs:
        inputs["mm_token_type_ids"] = torch.nn.functional.pad(
            inputs["mm_token_type_ids"],
            (0, padded_len - input_ids_length),
            "constant",
            0,
        )
    inputs = {name: np.array(value) for name, value in inputs.items()}

    vision_inputs = {name: value for name, value in inputs.items() if name in VISION_INPUT_KEYS}
    vision_inputs.update(
        {name: vision_inputs[name].astype("float16") for name in VISION_FP16_KEYS if name in vision_inputs}
    )
    vision_outputs = vision_session.run(vision_inputs)

    lang_inputs = {name: value for name, value in inputs.items() if name not in vision_inputs}
    if "position_ids" in inputs:
        lang_inputs["position_ids"] = inputs["position_ids"]
        lang_inputs.pop("attention_mask", None)
    else:
        lang_inputs["position_ids"] = np.where(lang_inputs.pop("attention_mask"), np.arange(padded_len), -1)

    if "mm_token_type_ids" not in lang_inputs:
        lang_inputs["mm_token_type_ids"] = np.zeros((BATCH_SIZE, padded_len), dtype=np.int64)

    lang_inputs["image_idx"] = np.array([[0]])
    for name in VISION_OUTPUTS:
        if name in vision_outputs and name in prefill_session.binding_index_map:
            lang_inputs[name] = vision_outputs[name]

    assert "image_idx" in prefill_session.binding_index_map, "image_idx not a compiled prefill input binding"
    decode_has_image_idx = "image_idx" in decode_session.binding_index_map
    decode_has_mm_token_type_ids = "mm_token_type_ids" in decode_session.binding_index_map

    # ---- Prefill (SERIAL): image_idx and the retained KV thread chunk-to-chunk via host copy ----
    chunk_inputs = dict(lang_inputs)
    outputs = None
    for chunk_idx in range(num_chunks):
        chunk_inputs["input_ids"] = lang_inputs["input_ids"][
            :, chunk_idx * PREFILL_SEQ_LEN : (chunk_idx + 1) * PREFILL_SEQ_LEN
        ]
        chunk_inputs["position_ids"] = lang_inputs["position_ids"][
            ..., chunk_idx * PREFILL_SEQ_LEN : (chunk_idx + 1) * PREFILL_SEQ_LEN
        ]
        chunk_inputs["mm_token_type_ids"] = lang_inputs["mm_token_type_ids"][
            ..., chunk_idx * PREFILL_SEQ_LEN : (chunk_idx + 1) * PREFILL_SEQ_LEN
        ]
        outputs = prefill_session.run(chunk_inputs)
        for layer_idx in range(NUM_HIDDEN_LAYERS):
            chunk_inputs[f"past_key.{layer_idx}"] = outputs[f"past_key.{layer_idx}_RetainedState"]
            chunk_inputs[f"past_value.{layer_idx}"] = outputs[f"past_value.{layer_idx}_RetainedState"]
        chunk_inputs["image_idx"] = outputs["image_idx_output"]

    generated_ids = [_get_next_token_ids(outputs["logits"])]

    # ---- Decode (consumer): feed each step's RetainedState back as the next step's KV input ----
    position_ids = np.max(lang_inputs["position_ids"], axis=-1, keepdims=True) + 1
    decode_inputs = {
        "input_ids": generated_ids[-1].reshape(BATCH_SIZE, 1),
        "position_ids": position_ids,
    }
    for layer_idx in range(NUM_HIDDEN_LAYERS):
        decode_inputs[f"past_key.{layer_idx}"] = outputs[f"past_key.{layer_idx}_RetainedState"]
        decode_inputs[f"past_value.{layer_idx}"] = outputs[f"past_value.{layer_idx}_RetainedState"]
    if decode_has_image_idx:
        decode_inputs["image_idx"] = outputs["image_idx_output"]
    for name in VISION_OUTPUTS:
        rs_name = f"{name}_RetainedState"
        if rs_name in outputs and name in decode_session.binding_index_map:
            decode_inputs[name] = outputs[rs_name]
    if decode_has_mm_token_type_ids:
        decode_inputs["mm_token_type_ids"] = np.zeros((BATCH_SIZE, 1), dtype=np.int64)

    for _ in range(GENERATION_LEN - 1):
        decode_outputs = decode_session.run(decode_inputs)
        generated_ids.append(_get_next_token_ids(decode_outputs["logits"]))
        position_ids = position_ids + 1
        decode_inputs["input_ids"] = generated_ids[-1].reshape(BATCH_SIZE, 1)
        decode_inputs["position_ids"] = position_ids
        for layer_idx in range(NUM_HIDDEN_LAYERS):
            decode_inputs[f"past_key.{layer_idx}"] = decode_outputs[f"past_key.{layer_idx}_RetainedState"]
            decode_inputs[f"past_value.{layer_idx}"] = decode_outputs[f"past_value.{layer_idx}_RetainedState"]
        if decode_has_image_idx:
            decode_inputs["image_idx"] = decode_outputs["image_idx_output"]
        for name in VISION_OUTPUTS:
            rs_name = f"{name}_RetainedState"
            if rs_name in decode_outputs and name in decode_session.binding_index_map:
                decode_inputs[name] = decode_outputs[rs_name]

    return np.stack(generated_ids, axis=1)


@pytest.mark.on_qaic
@pytest.mark.multimodal
def test_gemma4_moe_disagg_kv_share_qaic_vs_hf_fp32(manual_cleanup):
    torch.manual_seed(42)

    hf_model = _load_hf_model_from_pretrained(_build_config(dtype="float32"))
    processor = AutoProcessor.from_pretrained(MODEL_NAME, trust_remote_code=True)

    image = Image.new("RGB", IMAGE_SIZE, color=(127, 127, 127))
    messages = _prepare_messages(image)
    chat_template = _resolve_chat_template(processor, processor.tokenizer)
    common_inputs = _prepare_processor_inputs(processor, chat_template, messages)
    hf_tokens = _run_hf_torch_fp32(hf_model, common_inputs)

    hf_model.config.dtype = "float32"
    hf_model.config.torch_dtype = torch.float32
    if hasattr(hf_model.config, "text_config"):
        hf_model.config.text_config.dtype = "float32"
        hf_model.config.text_config.torch_dtype = torch.float32
    qeff_model = QEFFAutoModelForImageTextToText(
        hf_model,
        attn_implementation="eager",
        kv_offload=True,
        config=hf_model.config,
        dtype=torch.float32,
        trust_remote_code=True,
    )
    # Disable FP16 clipping so the fp32 export stays true fp32 and can match HF fp32 exactly.
    _remove_fp16clip_transform(qeff_model)

    sessions = []
    compiled_onnx_paths = {}
    try:
        vision_qpc_path = qeff_model.compile(
            batch_size=BATCH_SIZE,
            prefill_seq_len=PREFILL_SEQ_LEN,
            ctx_len=CTX_LEN,
            num_cores=16,
            num_devices=1,
            mos=1,
            aic_enable_depth_first=True,
            skip_vision=False,
            split_model_io=True,
            skip_lang=True,
            use_onnx_subfunctions=True,
            offload_pt_weights=False,
        )
        compiled_onnx_paths["vision"] = _assert_onnx_path(qeff_model.vision_model.onnx_path, "vision")

        decode_qpc_path = qeff_model.compile(
            batch_size=BATCH_SIZE,
            prefill_seq_len=1,
            ctx_len=CTX_LEN,
            num_cores=16,
            num_devices=DECODE_NUM_DEVICES,
            retain_full_kv=True,  # required for DMA slice writes into full KV
            split_retained_state_io=True,
            mos=1,
            aic_enable_depth_first=True,
            prefill_only=False,
            skip_vision=True,
            use_onnx_subfunctions=True,
            offload_pt_weights=False,
        )
        compiled_onnx_paths["decode"] = _assert_onnx_path(qeff_model.lang_model.onnx_path, "decode")

        prefill_qpc_path = qeff_model.compile(
            batch_size=BATCH_SIZE,
            prefill_seq_len=PREFILL_SEQ_LEN,
            ctx_len=CTX_LEN,
            num_cores=16,
            num_devices=PREFILL_NUM_DEVICES,
            retain_full_kv=True,
            split_retained_state_io=True,
            mos=1,
            aic_enable_depth_first=True,
            mdp_num_partitions=PREFILL_MDP_PARTITIONS,
            prefill_only=True,
            enable_chunking=True,
            skip_vision=True,
            use_onnx_subfunctions=True,
        )
        compiled_onnx_paths["prefill"] = _assert_onnx_path(qeff_model.lang_model.onnx_path, "prefill")
        # _assert_distinct_onnx_paths(compiled_onnx_paths)
        print(f"Disagg ONNX paths: {compiled_onnx_paths}")

        vision_session = QAICInferenceSession(vision_qpc_path.get("vision_qpc_path"))
        prefill_session = QAICInferenceSession(prefill_qpc_path.get("lang_prefill_qpc_path"), kv_dma_share=True)
        decode_session = QAICInferenceSession(decode_qpc_path.get("lang_decode_qpc_path"), kv_dma_share=True)
        sessions.extend([vision_session, prefill_session, decode_session])

        qaic_tokens = _run_disagg_kv_share_qaic_generation(
            processor=processor,
            common_inputs=common_inputs,
            vision_session=vision_session,
            prefill_session=prefill_session,
            decode_session=decode_session,
        )
    finally:
        for session in sessions:
            session.deactivate()
        cleanup_paths = list(compiled_onnx_paths.values()) or [
            getattr(qeff_model.vision_model, "onnx_path", None),
            getattr(qeff_model.lang_model, "onnx_path", None),
        ]
        manual_cleanup([path for path in cleanup_paths if path is not None])

    assert qaic_tokens.shape == (BATCH_SIZE, GENERATION_LEN)
    assert hf_tokens.shape == (BATCH_SIZE, GENERATION_LEN)
    assert np.issubdtype(qaic_tokens.dtype, np.integer)
    assert np.issubdtype(hf_tokens.dtype, np.integer)

    matches = hf_tokens == qaic_tokens
    num_matched = int(matches.all(axis=0).cumprod().sum())  # leading run matched across all rows
    hf_text = processor.tokenizer.batch_decode(hf_tokens, skip_special_tokens=True)
    qaic_text = processor.tokenizer.batch_decode(qaic_tokens, skip_special_tokens=True)
    print(f"HF Torch fp32 tokens   : {hf_tokens.tolist()}")
    print(f"Disagg QAIC DMA tokens : {qaic_tokens.tolist()}")
    print(f"HF Torch fp32 text     : {hf_text}")
    print(f"Disagg QAIC DMA text   : {qaic_text}")
    print(f"Matched leading tokens : {num_matched}/{GENERATION_LEN}")

    if not matches.all():
        first_mismatch = int(np.argmin(matches.all(axis=0)))
        raise AssertionError(
            "Tokens don't match for HF Torch fp32 output and disagg QAIC DMA output; "
            f"first mismatch at token index {first_mismatch} "
            f"(matched {num_matched}/{GENERATION_LEN} leading tokens): "
            f"HF={hf_tokens[:, first_mismatch].tolist()} vs QAIC={qaic_tokens[:, first_mismatch].tolist()}"
        )


def _build_qeff_model(hf_model) -> QEFFAutoModelForImageTextToText:
    hf_model.config.dtype = "float32"
    hf_model.config.torch_dtype = torch.float32
    if hasattr(hf_model.config, "text_config"):
        hf_model.config.text_config.dtype = "float32"
        hf_model.config.text_config.torch_dtype = torch.float32
    qeff_model = QEFFAutoModelForImageTextToText(
        hf_model,
        attn_implementation="eager",
        kv_offload=True,
        config=hf_model.config,
        dtype=torch.float32,
        trust_remote_code=True,
    )
    # Disable FP16 clipping so the fp32 export stays true fp32 (identical for both handoffs).
    _remove_fp16clip_transform(qeff_model)
    return qeff_model


def _compile_vision(qeff_model) -> str:
    vision_qpc_path = qeff_model.compile(
        batch_size=BATCH_SIZE,
        prefill_seq_len=PREFILL_SEQ_LEN,
        ctx_len=CTX_LEN,
        num_cores=16,
        num_devices=1,
        mos=1,
        aic_enable_depth_first=True,
        skip_vision=False,
        split_model_io=True,
        skip_lang=True,
        use_onnx_subfunctions=True,
        offload_pt_weights=False,
    )
    return vision_qpc_path.get("vision_qpc_path")


def _compile_kv_share_lang(qeff_model) -> tuple[str, str, dict]:
    """Compile the DMA KV-share lang QPCs (split_retained_state_io + retain_full_kv)."""
    onnx_paths = {}
    decode_qpc_path = qeff_model.compile(
        batch_size=BATCH_SIZE,
        prefill_seq_len=1,
        ctx_len=CTX_LEN,
        num_cores=16,
        num_devices=DECODE_NUM_DEVICES,
        retain_full_kv=True,  # required for DMA slice writes into full KV
        split_retained_state_io=True,
        mos=1,
        aic_enable_depth_first=True,
        prefill_only=False,
        skip_vision=True,
        use_onnx_subfunctions=True,
        offload_pt_weights=False,
    )
    onnx_paths["kv_share_decode"] = _assert_onnx_path(qeff_model.lang_model.onnx_path, "kv_share decode")

    prefill_qpc_path = qeff_model.compile(
        batch_size=BATCH_SIZE,
        prefill_seq_len=PREFILL_SEQ_LEN,
        ctx_len=CTX_LEN,
        num_cores=16,
        num_devices=PREFILL_NUM_DEVICES,
        retain_full_kv=True,
        split_retained_state_io=True,
        mos=1,
        aic_enable_depth_first=True,
        mdp_num_partitions=PREFILL_MDP_PARTITIONS,
        prefill_only=True,
        enable_chunking=True,
        skip_vision=True,
        use_onnx_subfunctions=True,
        offload_pt_weights=False,
    )
    onnx_paths["kv_share_prefill"] = _assert_onnx_path(qeff_model.lang_model.onnx_path, "kv_share prefill")
    return prefill_qpc_path.get("lang_prefill_qpc_path"), decode_qpc_path.get("lang_decode_qpc_path"), onnx_paths


def _compile_baseline_lang(qeff_model) -> tuple[str, str, dict]:
    """Compile the numpy-copy baseline lang QPCs."""
    onnx_paths = {}
    decode_qpc_path = qeff_model.compile(
        batch_size=BATCH_SIZE,
        prefill_seq_len=1,
        ctx_len=CTX_LEN,
        num_cores=16,
        num_devices=DECODE_NUM_DEVICES,
        split_model_io=True,
        mos=1,
        aic_enable_depth_first=True,
        prefill_only=False,
        skip_vision=True,
        use_onnx_subfunctions=True,
        offload_pt_weights=False,
    )
    onnx_paths["baseline_decode"] = _assert_onnx_path(qeff_model.lang_model.onnx_path, "baseline decode")

    prefill_qpc_path = qeff_model.compile(
        batch_size=BATCH_SIZE,
        prefill_seq_len=PREFILL_SEQ_LEN,
        ctx_len=CTX_LEN,
        num_cores=16,
        num_devices=PREFILL_NUM_DEVICES,
        retain_full_kv=True,
        split_model_io=True,
        mos=1,
        aic_enable_depth_first=True,
        mdp_num_partitions=PREFILL_MDP_PARTITIONS,
        prefill_only=True,
        enable_chunking=True,
        skip_vision=True,
        use_onnx_subfunctions=True,
        offload_pt_weights=False,
    )
    onnx_paths["baseline_prefill"] = _assert_onnx_path(qeff_model.lang_model.onnx_path, "baseline prefill")
    return prefill_qpc_path.get("lang_prefill_qpc_path"), decode_qpc_path.get("lang_decode_qpc_path"), onnx_paths


@pytest.mark.on_qaic
@pytest.mark.multimodal
def test_gemma4_moe_disagg_kv_share_matches_numpy_copy_baseline(manual_cleanup):
    """The DMA KV-share handoff must reproduce the numpy-copy baseline token for token."""
    torch.manual_seed(42)

    hf_model = _load_hf_model_from_pretrained(_build_config(dtype="float32"))
    processor = AutoProcessor.from_pretrained(MODEL_NAME, trust_remote_code=True)

    image = Image.new("RGB", IMAGE_SIZE, color=(127, 127, 127))
    messages = _prepare_messages(image)
    chat_template = _resolve_chat_template(processor, processor.tokenizer)
    common_inputs = _prepare_processor_inputs(processor, chat_template, messages)

    qeff_model = _build_qeff_model(hf_model)

    sessions = []
    compiled_onnx_paths = {}
    try:
        vision_qpc_path = _compile_vision(qeff_model)
        compiled_onnx_paths["vision"] = _assert_onnx_path(qeff_model.vision_model.onnx_path, "vision")

        share_prefill_qpc, share_decode_qpc, share_onnx = _compile_kv_share_lang(qeff_model)
        compiled_onnx_paths.update(share_onnx)

        base_prefill_qpc, base_decode_qpc, base_onnx = _compile_baseline_lang(qeff_model)
        compiled_onnx_paths.update(base_onnx)

        # _assert_distinct_onnx_paths(compiled_onnx_paths)
        print(f"Disagg ONNX paths: {compiled_onnx_paths}")

        share_vision_session = QAICInferenceSession(vision_qpc_path)
        share_prefill_session = QAICInferenceSession(share_prefill_qpc, kv_dma_share=True)
        share_decode_session = QAICInferenceSession(share_decode_qpc, kv_dma_share=True)
        sessions.extend([share_vision_session, share_prefill_session, share_decode_session])

        share_tokens = _run_disagg_kv_share_qaic_generation(
            processor=processor,
            common_inputs=common_inputs,
            vision_session=share_vision_session,
            prefill_session=share_prefill_session,
            decode_session=share_decode_session,
        )
        for session in (share_vision_session, share_prefill_session, share_decode_session):
            session.deactivate()

        base_vision_session = QAICInferenceSession(vision_qpc_path)
        base_prefill_session = QAICInferenceSession(base_prefill_qpc)
        base_decode_session = QAICInferenceSession(base_decode_qpc)
        sessions.extend([base_vision_session, base_prefill_session, base_decode_session])

        baseline_tokens = _run_disagg_baseline_numpy_copy_generation(
            processor=processor,
            common_inputs=common_inputs,
            vision_session=base_vision_session,
            prefill_session=base_prefill_session,
            decode_session=base_decode_session,
        )
    finally:
        for session in sessions:
            session.deactivate()
        cleanup_paths = list(compiled_onnx_paths.values()) or [
            getattr(qeff_model.vision_model, "onnx_path", None),
            getattr(qeff_model.lang_model, "onnx_path", None),
        ]
        manual_cleanup([path for path in cleanup_paths if path is not None])

    assert share_tokens.shape == (BATCH_SIZE, GENERATION_LEN)
    assert baseline_tokens.shape == (BATCH_SIZE, GENERATION_LEN)

    matches = share_tokens == baseline_tokens
    num_matched = int(matches.all(axis=0).cumprod().sum())  # leading run matched across all rows
    share_text = processor.tokenizer.batch_decode(share_tokens, skip_special_tokens=True)
    baseline_text = processor.tokenizer.batch_decode(baseline_tokens, skip_special_tokens=True)
    print(f"KV-share DMA tokens   : {share_tokens.tolist()}")
    print(f"Numpy-copy baseline   : {baseline_tokens.tolist()}")
    print(f"KV-share DMA text     : {share_text}")
    print(f"Numpy-copy text       : {baseline_text}")
    print(f"Matched leading tokens: {num_matched}/{GENERATION_LEN}")

    if not matches.all():
        first_mismatch = int(np.argmin(matches.all(axis=0)))
        raise AssertionError(
            "Tokens don't match between KV-share DMA and numpy-copy baseline; "
            f"first mismatch at token index {first_mismatch} "
            f"(matched {num_matched}/{GENERATION_LEN} leading tokens): "
            f"kv_share={share_tokens[:, first_mismatch].tolist()} vs "
            f"baseline={baseline_tokens[:, first_mismatch].tolist()}"
        )

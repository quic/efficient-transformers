# -----------------------------------------------------------------------------
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
#
# ----------------------------------------------------------------------------

"""Token-level parity test for the Qwen3.5-MoE disaggregated prefill/decode DMA path.

pytest -m "on_qaic and multimodal" tests/transformers/disaggregated/test_qwen3_5_disagg_kv_share_w_hf_fp32.py
"""

import copy
import os
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image
from transformers import AutoConfig, AutoModelForCausalLM, AutoModelForImageTextToText, AutoProcessor

from QEfficient import QEFFAutoModelForImageTextToText
from QEfficient.generation.cloud_infer import QAICInferenceSession

MODEL_NAME = "Qwen/Qwen3.6-35B-A3B"
NUM_HIDDEN_LAYERS = 4
VISION_DEPTH = 2
PREFILL_SEQ_LEN = 64
CTX_LEN = 1024
BATCH_SIZE = 1
GENERATION_LEN = 50
IMAGE_SIZE = (536, 354)
TEXT_PROMPT = "Describe all the colors seen in the image."

# Vision-tower inputs are routed to the vision QPC; everything else feeds the lang QPCs.
# The Qwen3.5-MoE vision encoder forward binds exactly (pixel_values, image_grid_thw).
VISION_INPUT_KEYS = {
    "pixel_values",
    "image_masks",
    "image_input_idx",
    "valid_idx",
    "aspect_ratio_ids",
    "aspect_ratio_mask",
}
VISION_FP16_KEYS = {"pixel_values", "image_masks"}
VISION_OUTPUTS = ("vision_embeds",)


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


def _build_config(dtype: str = "float16"):
    """Load the real (full-depth) config; keep the pretrained layer_types / vision depth."""
    config = AutoConfig.from_pretrained(MODEL_NAME, trust_remote_code=True)
    config.dtype = dtype
    config.torch_dtype = getattr(torch, dtype)

    text_config = config.text_config if hasattr(config, "text_config") else config
    # --- Config truncation: run a shallow model so the compile/run stays cheap. ---
    text_config.num_hidden_layers = NUM_HIDDEN_LAYERS
    # Pin the interval to the depth so both sources of truth agree for any
    # NUM_HIDDEN_LAYERS: one full_attention layer at the end (a full-KV slice for the DMA
    # handoff), all others linear.
    interval = NUM_HIDDEN_LAYERS
    text_config.full_attention_interval = interval
    text_config.layer_types = [
        "full_attention" if (i + 1) % interval == 0 else "linear_attention" for i in range(NUM_HIDDEN_LAYERS)
    ]
    text_config.dtype = dtype
    text_config.torch_dtype = getattr(torch, dtype)

    # Reduce the vision tower depth too so the vision QPC compile stays cheap.
    if hasattr(config, "vision_config"):
        config.vision_config.depth = VISION_DEPTH
    return config


def _prepare_messages(image: Image.Image) -> list:
    return [
        [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": TEXT_PROMPT},
                ],
            }
        ]
        for _ in range(BATCH_SIZE)
    ]


def _prepare_processor_inputs(processor: AutoProcessor, messages: list) -> dict:
    process_vision_info = pytest.importorskip("qwen_vl_utils").process_vision_info

    texts = [processor.apply_chat_template(message, tokenize=False, add_generation_prompt=True) for message in messages]
    image_inputs, video_inputs = process_vision_info(messages)
    return dict(processor(text=texts, images=image_inputs, videos=video_inputs, padding=True, return_tensors="pt"))


def _get_next_token_ids(logits: np.ndarray) -> np.ndarray:
    logits = np.asarray(logits)
    return logits[:, -1, :].argmax(axis=-1).astype(np.int64)


def _last_step_logits(logits: np.ndarray) -> np.ndarray:
    """The (B, vocab) logit slice that produced the next token for one step."""
    return np.asarray(logits, dtype=np.float32)[:, -1, :]


def _run_hf_torch_fp32(model, processor: AutoProcessor, messages: list, collect_logits: bool = False):
    model = model.to(dtype=torch.float32).eval()
    inputs = _prepare_processor_inputs(processor, messages)
    inputs = {
        name: value.to(dtype=torch.float32) if torch.is_floating_point(value) else value
        for name, value in inputs.items()
    }

    with torch.inference_mode():
        outputs = model.generate(
            **inputs,
            max_new_tokens=GENERATION_LEN,
            do_sample=False,
            output_logits=collect_logits,
            return_dict_in_generate=collect_logits,
        )

    if collect_logits:
        # outputs.logits is a per-step tuple of (B, vocab) raw (pre-warp) logits, one entry
        # per generated token -> stack to (B, GEN, vocab) aligned with the returned tokens.
        sequences = outputs.sequences
        step_logits = np.stack([step.float().cpu().numpy() for step in outputs.logits], axis=1)
        prompt_len = inputs["input_ids"].shape[-1]
        return sequences[:, prompt_len:].detach().cpu().numpy(), step_logits

    prompt_len = inputs["input_ids"].shape[-1]
    return outputs[:, prompt_len:].detach().cpu().numpy()


def _run_disagg_kv_share_qaic_generation(
    qeff_model: QEFFAutoModelForImageTextToText,
    processor: AutoProcessor,
    common_inputs: dict,
    vision_session: QAICInferenceSession,
    prefill_session: QAICInferenceSession,
    decode_session: QAICInferenceSession,
    collect_logits: bool = False,
):
    inputs = {
        name: value.clone() if isinstance(value, torch.Tensor) else copy.deepcopy(value)
        for name, value in common_inputs.items()
    }
    inputs = qeff_model.model.prepare_inputs_for_generation(
        inputs=inputs,
        prefill_seq_len=PREFILL_SEQ_LEN,
        batch_size=BATCH_SIZE,
    )

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

    lang_inputs["image_idx"] = np.array([[0]])

    # image_idx must be a compiled prefill input binding; the KV-share path silently drops
    # unknown input names (warn + skip), so assert it up front. Qwen3.5 also binds it on decode.
    assert "image_idx" in prefill_session.binding_index_map, "image_idx not a compiled prefill input binding"
    decode_has_image_idx = "image_idx" in decode_session.binding_index_map

    # vision_embeds is constant across every prefill chunk and decode
    vision_persist = {name: vision_outputs[name] for name in VISION_OUTPUTS if name in vision_outputs}
    prefill_session.set_persistent_inputs(vision_persist)
    decode_session.set_persistent_inputs(
        {name: value for name, value in vision_persist.items() if name in decode_session.binding_index_map}
    )

    # Hybrid: kv_cache_info carries mixed 4-D (full) and 3-D (linear) shapes.
    kv_caches = [np.zeros(shape, dtype=dtype) for (shape, dtype) in decode_session.kv_cache_info]

    # ---- Prefill (producer, SERIAL): image_idx threads chunk-to-chunk ----
    # Only the LAST chunk wires the DMA handoff into kv_caches (earlier chunks just accumulate
    # KV on-device).
    chunk_inputs = dict(lang_inputs)
    exec_idx = None
    for chunk_idx in range(num_chunks):
        chunk_inputs["input_ids"] = lang_inputs["input_ids"][
            :, chunk_idx * PREFILL_SEQ_LEN : (chunk_idx + 1) * PREFILL_SEQ_LEN
        ]
        chunk_inputs["position_ids"] = lang_inputs["position_ids"][
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
    step_logits = [_last_step_logits(prefill_out["logits"])] if collect_logits else None

    decode_kv_map = decode_session.decode_buff_map + decode_session.decode_rs_kv_only_buff_map
    num_pos_sections = lang_inputs["position_ids"].shape[0]
    phys_pos = int(lang_inputs["position_ids"][0].max()) + 1
    mrope_pos = int(lang_inputs["position_ids"][1:].max()) + 1

    def _decode_position_ids(next_phys: int, next_mrope: int) -> np.ndarray:
        pos = np.empty((num_pos_sections, BATCH_SIZE, 1), dtype=np.int64)
        pos[0] = next_phys
        pos[1:] = next_mrope
        return pos

    decode_inputs = {
        "input_ids": generated_ids[-1].reshape(BATCH_SIZE, 1),
        "position_ids": _decode_position_ids(phys_pos, mrope_pos),
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
        if collect_logits:
            step_logits.append(_last_step_logits(decode_outputs["logits"]))
        phys_pos += 1
        mrope_pos += 1
        decode_inputs = {
            "input_ids": generated_ids[-1].reshape(BATCH_SIZE, 1),
            "position_ids": _decode_position_ids(phys_pos, mrope_pos),
        }
        if decode_has_image_idx:
            decode_inputs["image_idx"] = decode_outputs["image_idx_output"]

    tokens = np.stack(generated_ids, axis=1)
    if collect_logits:
        # (B, GEN, vocab) aligned with tokens: index 0 is the prefill/first-token logits.
        return tokens, np.stack(step_logits, axis=1)
    return tokens


def _wrap_qeff_model(hf_model) -> QEFFAutoModelForImageTextToText:
    hf_model.config.dtype = "float32"
    hf_model.config.torch_dtype = torch.float32
    if hasattr(hf_model.config, "text_config"):
        hf_model.config.text_config.dtype = "float32"
        hf_model.config.text_config.torch_dtype = torch.float32
    return QEFFAutoModelForImageTextToText(
        hf_model,
        attn_implementation="eager",
        kv_offload=True,
        config=hf_model.config,
        dtype=torch.float32,
        layerwise=False,
    )


def _compile_disagg_sessions(qeff_model, image, sessions: list, compiled_onnx_paths: dict):
    """Compile the vision/decode/prefill QPCs (fp32, no quantization) and open pooled sessions.

    Appends the opened sessions to ``sessions`` and records ONNX paths in
    ``compiled_onnx_paths`` so the caller's ``finally`` can deactivate + clean up. Returns
    ``(vision_session, prefill_session, decode_session)``.
    """
    vision_qpc_path = qeff_model.compile(
        batch_size=BATCH_SIZE,
        prefill_seq_len=PREFILL_SEQ_LEN,
        ctx_len=CTX_LEN,
        height=image.height,
        width=image.width,
        num_cores=16,
        num_devices=1,
        mos=1,
        aic_enable_depth_first=True,
        skip_vision=False,
        split_model_io=True,
        skip_lang=True,
        use_onnx_subfunctions=True,
        layerwise=False,
        offload_pt_weights=False,
    )
    compiled_onnx_paths["vision"] = _assert_onnx_path(qeff_model.vision_model.onnx_path, "vision")
    decode_qpc_path = qeff_model.compile(
        batch_size=BATCH_SIZE,
        prefill_seq_len=1,
        ctx_len=CTX_LEN,
        height=image.height,
        width=image.width,
        num_cores=16,
        num_devices=2,
        retain_full_kv=True,  # required for DMA slice writes into full KV
        split_retained_state_io=True,
        mos=1,
        aic_enable_depth_first=True,
        prefill_only=False,
        skip_vision=True,
        use_onnx_subfunctions=True,
        layerwise=False,
        offload_pt_weights=False,
    )
    compiled_onnx_paths["decode"] = _assert_onnx_path(qeff_model.lang_model.onnx_path, "decode")

    prefill_qpc_path = qeff_model.compile(
        batch_size=BATCH_SIZE,
        prefill_seq_len=PREFILL_SEQ_LEN,
        ctx_len=CTX_LEN,
        height=image.height,
        width=image.width,
        num_cores=16,
        num_devices=4,
        retain_full_kv=True,
        split_retained_state_io=True,
        mos=1,
        aic_enable_depth_first=True,
        mdp_num_partitions=2,
        prefill_only=True,
        enable_chunking=True,
        skip_vision=True,
        use_onnx_subfunctions=True,
        layerwise=False,
    )
    compiled_onnx_paths["prefill"] = _assert_onnx_path(qeff_model.lang_model.onnx_path, "prefill")
    _assert_distinct_onnx_paths(compiled_onnx_paths)
    print(f"Disagg ONNX paths: {compiled_onnx_paths}")

    vision_session = QAICInferenceSession(vision_qpc_path.get("vision_qpc_path"))
    prefill_session = QAICInferenceSession(prefill_qpc_path.get("lang_prefill_qpc_path"), kv_dma_share=True)
    decode_session = QAICInferenceSession(decode_qpc_path.get("lang_decode_qpc_path"), kv_dma_share=True)
    sessions.extend([vision_session, prefill_session, decode_session])
    return vision_session, prefill_session, decode_session


@pytest.mark.on_qaic
@pytest.mark.multimodal
def test_qwen3_5_disagg_kv_share_qaic_vs_hf_fp32(manual_cleanup):
    pytest.importorskip("qwen_vl_utils")
    torch.manual_seed(42)

    hf_model = _load_hf_model_from_pretrained(_build_config(dtype="float32"))
    processor = AutoProcessor.from_pretrained(MODEL_NAME, trust_remote_code=True)

    image = Image.new("RGB", IMAGE_SIZE, color=(127, 127, 127))
    messages = _prepare_messages(image)
    common_inputs = _prepare_processor_inputs(processor, messages)
    hf_tokens = _run_hf_torch_fp32(hf_model, processor, messages)

    qeff_model = _wrap_qeff_model(hf_model)

    sessions = []
    compiled_onnx_paths = {}
    try:
        vision_session, prefill_session, decode_session = _compile_disagg_sessions(
            qeff_model, image, sessions, compiled_onnx_paths
        )

        qaic_tokens = _run_disagg_kv_share_qaic_generation(
            qeff_model=qeff_model,
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


def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    a = a.reshape(-1).astype(np.float64)
    b = b.reshape(-1).astype(np.float64)
    denom = (np.linalg.norm(a) * np.linalg.norm(b)) or 1.0
    return float(np.dot(a, b) / denom)


def _report_step_logits(step: int, hf_logits: np.ndarray, qaic_logits: np.ndarray, tokenizer, top_k: int = 5):
    """Print a per-step logit comparison for row 0 and classify the disagreement.

    Prints top-k candidates for both engines, the engines' logit gap between their two
    contested top tokens, and the max-abs-diff / cosine-similarity of the full vocab logit
    vectors. A near-tie (small contested gap + high cosine) is benign fp32 drift; a gross
    disagreement (low cosine) points at a structural/wiring bug.
    """
    hf_row = hf_logits[0].astype(np.float64)
    qaic_row = qaic_logits[0].astype(np.float64)
    hf_top = np.argsort(hf_row)[::-1][:top_k]
    qaic_top = np.argsort(qaic_row)[::-1][:top_k]
    hf_tok, qaic_tok = int(hf_top[0]), int(qaic_top[0])

    max_abs_diff = float(np.max(np.abs(hf_row - qaic_row)))
    cos = _cosine_similarity(hf_row, qaic_row)
    # Gap between the two *contested* tokens, measured in each engine's own logits: how
    # decisively each engine preferred its own winner over the other's winner.
    hf_gap = float(hf_row[hf_tok] - hf_row[qaic_tok])
    qaic_gap = float(qaic_row[qaic_tok] - qaic_row[hf_tok])

    def _fmt(row, idx):
        return [(int(i), round(float(row[i]), 4), repr(tokenizer.decode([int(i)]))) for i in idx]

    print(f"\n--- logit diff @ step {step} (HF tok={hf_tok} vs QAIC tok={qaic_tok}) ---")
    print(f"  HF   top{top_k}: {_fmt(hf_row, hf_top)}")
    print(f"  QAIC top{top_k}: {_fmt(qaic_row, qaic_top)}")
    print(f"  full-vocab max|Δlogit|={max_abs_diff:.4e}  cosine={cos:.8f}")
    print(f"  contested gap: HF prefers its token by {hf_gap:.4e}; QAIC by {qaic_gap:.4e}")
    verdict = "NEAR-TIE (benign fp32 drift)" if min(abs(hf_gap), abs(qaic_gap)) < 1e-2 else "DECISIVE (investigate)"
    print(f"  verdict: {verdict}")


@pytest.mark.on_qaic
@pytest.mark.multimodal
@pytest.mark.skipif(
    os.environ.get("QEFF_RUN_LOGIT_DIAG") != "1",
    reason="opt-in logit-level diagnostic; set QEFF_RUN_LOGIT_DIAG=1 to run",
)
def test_qwen3_5_disagg_kv_share_logit_diff_diagnostic(manual_cleanup):
    """Optional diagnostic (does NOT assert token parity): compares HF fp32 vs disagg-QAIC
    *logits* step-by-step to classify a post-N divergence as benign fp32 drift (near-tie
    argmax flip, high cosine similarity) versus a structural bug (gross logit disagreement).

    Run with:
      QEFF_RUN_LOGIT_DIAG=1 pytest -s -m "on_qaic and multimodal" \
        tests/transformers/disaggregated/test_qwen3_5_disagg_kv_share_w_hf_fp32.py \
        -k logit_diff_diagnostic
    """
    pytest.importorskip("qwen_vl_utils")
    torch.manual_seed(42)

    hf_model = _load_hf_model_from_pretrained(_build_config(dtype="float32"))
    processor = AutoProcessor.from_pretrained(MODEL_NAME, trust_remote_code=True)

    image = Image.new("RGB", IMAGE_SIZE, color=(127, 127, 127))
    messages = _prepare_messages(image)
    common_inputs = _prepare_processor_inputs(processor, messages)
    hf_tokens, hf_logits = _run_hf_torch_fp32(hf_model, processor, messages, collect_logits=True)

    qeff_model = _wrap_qeff_model(hf_model)

    sessions = []
    compiled_onnx_paths = {}
    try:
        vision_session, prefill_session, decode_session = _compile_disagg_sessions(
            qeff_model, image, sessions, compiled_onnx_paths
        )
        qaic_tokens, qaic_logits = _run_disagg_kv_share_qaic_generation(
            qeff_model=qeff_model,
            processor=processor,
            common_inputs=common_inputs,
            vision_session=vision_session,
            prefill_session=prefill_session,
            decode_session=decode_session,
            collect_logits=True,
        )
    finally:
        for session in sessions:
            session.deactivate()
        cleanup_paths = list(compiled_onnx_paths.values()) or [
            getattr(qeff_model.vision_model, "onnx_path", None),
            getattr(qeff_model.lang_model, "onnx_path", None),
        ]
        manual_cleanup([path for path in cleanup_paths if path is not None])

    assert hf_logits.shape[:2] == (BATCH_SIZE, GENERATION_LEN)
    assert qaic_logits.shape[:2] == (BATCH_SIZE, GENERATION_LEN)

    matches = (hf_tokens == qaic_tokens).all(axis=0)
    num_matched = int(matches.cumprod().sum())
    print(f"\nMatched leading tokens : {num_matched}/{GENERATION_LEN}")
    print(f"HF Torch fp32 tokens   : {hf_tokens.tolist()}")
    print(f"Disagg QAIC DMA tokens : {qaic_tokens.tolist()}")

    # Whole-run logit trajectory summary (per step: max-abs-diff and cosine over full vocab).
    per_step_maxdiff = np.max(np.abs(hf_logits - qaic_logits), axis=(0, 2))
    per_step_cos = np.array([_cosine_similarity(hf_logits[:, s], qaic_logits[:, s]) for s in range(GENERATION_LEN)])
    print(f"per-step max|Δlogit| : {np.round(per_step_maxdiff, 4).tolist()}")
    print(f"per-step cosine      : {np.round(per_step_cos, 6).tolist()}")

    # Detailed breakdown at the first token that diverged (if any) plus the prefill/first token.
    _report_step_logits(0, hf_logits[:, 0], qaic_logits[:, 0], processor.tokenizer)
    if num_matched < GENERATION_LEN:
        first_mismatch = int(np.argmin(matches))
        _report_step_logits(
            first_mismatch, hf_logits[:, first_mismatch], qaic_logits[:, first_mismatch], processor.tokenizer
        )
    else:
        print("\nAll tokens matched; no divergence step to break down.")

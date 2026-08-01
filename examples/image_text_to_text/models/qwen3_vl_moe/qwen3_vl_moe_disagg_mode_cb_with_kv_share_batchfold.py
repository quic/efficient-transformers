# -----------------------------------------------------------------------------
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------

"""Disaggregated prefill/decode for Qwen3-VL-MoE — DMA KV handoff, with a BATCH-FOLDED decode.

This is the batch_fold sibling of ``qwen3_vl_moe_disagg_mode_cb_with_kv_share.py``, but with
continuous batching turned OFF (see "Compile knobs" below). Each prompt is chunk-prefilled into
its own KV slot of a shared host cache ``[N, Hkv, ctx, D]`` via a per-slot DMA handoff (the host
buffer view ``kv[slot:slot+1]`` selects the slot). Only the DECODE side folds — it runs with
``batch_fold=True``.

Why no numpy KV copy is needed for the fold
--------------------------------------------
The modeling folds decode KV with ``key_states.reshape(B*Hkv, ctx, D)`` (cache_utils
``write_only_batch``), i.e. folded row index ``= b*Hkv + h``. That is exactly what
``np.reshape([N, Hkv, ctx, D] -> [1, N*Hkv, ctx, D])`` does, and on a C-contiguous array
reshape returns a *view over the same bytes* — not a copy. So one host allocation serves both
sides:

    host kv_caches[layer] : [N, Hkv, ctx, D]           (allocated once, contiguous)
      prefill writes row `slot`  -> view kv[slot:slot+1] = [1, Hkv, ctx, D]  (host-view selects slot)
      decode reads/writes all    -> view kv.reshape(1, N*Hkv, ctx, D)        (identity DMA)

Why no new slicing spec / DimSpec is needed
-------------------------------------------
The DimSpec template (``FULL_ATTN_DIMSPEC = [batch_index, 0, ctx_start, 0]``) is the same 4-D
rule for both sides. The compiled *spec handle* is built per session from that session's own
device bindings, so the decode session automatically builds a handle matching its folded
``[1, N*Hkv, ctx, D]`` binding. At handoff we pass ``batch_index=0, ctx_start=0`` — the
DimSpec resolves to ``(0, 0, 0, 0)``, an identity DMA of the whole fused cache over the whole
host view. No per-slot addressing (there is none under fold; all N slots decode as one
tensor), no host copy.

Compile knobs that make the fold line up
----------------------------------------
``BH = batch_size * num_key_value_heads`` (get_specializations) uses the compile ``batch_size``
param. To fold across all N prompts the decode QPC is compiled with ``batch_size=N`` and the
prefill QPC with ``batch_size=1`` (one prompt per exec).

"""

import argparse
from time import perf_counter

import numpy as np
import requests
import torch
import transformers
from PIL import Image
from qwen_vl_utils import process_vision_info
from transformers import AutoConfig, AutoProcessor

from QEfficient import QEFFAutoModelForImageTextToText
from QEfficient.generation.cloud_infer import QAICInferenceSession

DEFAULT_MODEL_ID = "Qwen/Qwen3-VL-30B-A3B-Instruct"
DEFAULT_PROMPTS = [
    "Tell me about yourself.",
    "What is the capital of France?",
    "Explain photosynthesis in one sentence.",
    "Name three primary colors.",
]
DEFAULT_IMAGE_PROMPTS = [
    "Describe all the colors seen in the image",
    "What are the objects in the image?",
    "What is the main subject of the image?",
    "describe the image?",
]
DEFAULT_IMAGE_URLS = [
    "https://picsum.photos/id/237/536/354",
    "https://picsum.photos/id/230/536/354",
    "https://picsum.photos/id/234/536/354",
    "https://picsum.photos/id/235/536/354",
]
DEFAULT_PREFILL_SEQ_LEN = 1024
DEFAULT_CTX_LEN = 4096
DEFAULT_GENERATION_LEN = 100
DEFAULT_BATCH_SIZE = 1

STAGES = 4
PREFILL_NUM_DEVICES = 4
DECODE_NUM_DEVICES = 16

# Decode-side attention blocking. batch_fold rides on the "kv" blocking path: the cached K/V
# is streamed through a running softmax in NUM_KV_BLOCKS chunks, but the batch/head axes are
# folded into a single [1, N*Hkv, ctx, D] cache (write_only_batch / read_only_blocked_K_batch).
# There is no batch_index binding under fold — slot identity is implicit in the folded row block
# b*Hkv+h.
NUM_KV_BLOCKS = 4

# Prefill-side head-parallel "online" blocking, matching qwen3_vl_disagg_mode.py. The prefill
# qaic_config is the decode config PLUS these online-prefill keys: because prefill_blocking_mode
# is set, dispatch goes through the online-prefill forward and batch_fold is inert during the
# prefill trace. PREFILL_BLOCK_CHUNKS = ceil(prefill_seq_len / PREFILL_QL_CHUNK).
PREFILL_BLOCKING_MODE = "online"
PREFILL_QL_CHUNK = 128
PREFILL_N_REP_CHUNK = 4
MOE_PREFILL_PACKED_CHUNK_SIZE = 256

VISION_INPUT_KEYS = {
    "pixel_values",
    "image_masks",
    "image_input_idx",
    "valid_idx",
    "aspect_ratio_ids",
    "aspect_ratio_mask",
    "image_grid_thw",
}
VISION_FP16_KEYS = {"pixel_values", "image_masks"}
VISION_OUTPUT_KEYS = ("vision_embeds", "deepstack_features")


def _build_config(model_id: str):
    """Load the model config, pinned to float16 (matches the single-request baseline)."""
    config = AutoConfig.from_pretrained(model_id)
    config.dtype = "float16"
    config.torch_dtype = torch.float16

    # For faster execution user can run with fewer layers. For testing purposes only.
    # config.vision_config.depth = 9
    # config.text_config.num_hidden_layers = 2
    # config.vision_config.deepstack_visual_indexes = [8]
    return config


def _decode_qaic_config(ctx_len: int, num_kv_blocks: int) -> dict:
    """Batch-folded KV blocking for the decode QPC.
     drops per-slot ``batch_index``
    addressing — all N slots decode together as one fused tensor. It rides on the ``"kv"``
    blocking path, which streams the cached K/V through a running softmax in ``num_kv_blocks``
    chunks. The prefill QPC (``prefill_only=True``) ignores ``batch_fold``, so prefill keeps
    the un-folded ``[N, Hkv, ctx, D]`` layout and its per-slot DMA handoff.
    """
    return {
        "blocking_mode": "kv",
        "num_kv_blocks": num_kv_blocks,
        "batch_fold": True,
        "ctx_len": ctx_len,
    }


def _prefill_qaic_config(ctx_len: int, num_kv_blocks: int, prefill_seq_len: int) -> dict:
    """Head-parallel "online" prefill blocking — same qaic_config as qwen3_vl_disagg_mode.py.

    This mirrors the reference driver exactly: the prefill config is ``_decode_qaic_config``
    (including ``batch_fold=True``) PLUS the online-prefill keys ``prefill_blocking_mode="online"``,
    ``prefill_block_chunks``, and ``prefill_n_rep_chunk``.

    Prefill MUST pass an explicit ``qaic_config`` because ``BlockingAttentionTransform.apply`` only
    *sets* ``module.attn_blocking_config`` — it never clears it — and decode compiles first,
    installing its config on every attention module. Passing this config overwrites the stale
    decode config. Because ``prefill_blocking_mode`` is set, prefill dispatches through the
    online-prefill forward (``prefill_blocked_attention_interface``) rather than the folded decode
    forward, so ``batch_fold`` is inert during the prefill trace (no ``seq_len == 1`` assert) and
    the prefill KV layout stays the un-folded per-slot ``[1, Hkv, ctx, D]`` the DMA handoff writes
    into ``kv[slot:slot+1]``.
    """
    cfg = _decode_qaic_config(ctx_len, num_kv_blocks)
    cfg["prefill_blocking_mode"] = PREFILL_BLOCKING_MODE
    cfg["prefill_block_chunks"] = -(-prefill_seq_len // PREFILL_QL_CHUNK)  # ceil divide
    cfg["prefill_n_rep_chunk"] = PREFILL_N_REP_CHUNK
    return cfg


def run(
    model_id: str = DEFAULT_MODEL_ID,
    prompts=None,
    prefill_seq_len: int = DEFAULT_PREFILL_SEQ_LEN,
    ctx_len: int = DEFAULT_CTX_LEN,
    generation_len: int = DEFAULT_GENERATION_LEN,
    batch_size: int = DEFAULT_BATCH_SIZE,
    skip_vision: bool = False,
    image_urls=None,
    stages: int = STAGES,
    prefill_num_devices: int = PREFILL_NUM_DEVICES,
    decode_num_devices: int = DECODE_NUM_DEVICES,
    num_kv_blocks: int = NUM_KV_BLOCKS,
):
    """Run chunked-prefill + batch-folded decode over ``prompts`` with the DMA KV handoff.

    ``skip_vision=False`` (default) pairs each prompt with the image at the same index of
    ``image_urls`` (cycled if shorter) and runs it as an image+text turn through the vision
    QPC; ``skip_vision=True`` runs text-only prompts. Returns a dict with, per prompt, the
    ``first_tokens`` (prefill argmax) and the full decoded ``tokens`` list, for parity
    comparison against the single-request driver.
    """
    prompts = list(prompts) if prompts else list(DEFAULT_PROMPTS)
    image_urls = list(image_urls) if image_urls else list(DEFAULT_IMAGE_URLS)
    # Repeat a single prompt/image across all batch_size slots (replicating one turn to fill the
    # batch): pass exactly one prompt (and, with vision, one image) to run every slot with the
    # same turn. A multi-element list is used as-is (distinct prompts per slot).
    if len(prompts) == 1:
        prompts = prompts * batch_size
    if len(image_urls) == 1:
        image_urls = image_urls * batch_size
    config = _build_config(model_id)

    # NOTE: continuous_batching is intentionally OFF. `continuous_batching` is a
    # from_pretrained-level flag shared by every compile() call — prefill and decode cannot
    # differ on it — and the CB + batch_fold export path is broken (get_dummy_inputs builds the
    # query dummy from `bs` but the folded KV from `fbs`, so the traced decode matmul
    # query[.,bs*Hkv,.] × k[.,fbs*Hkv,.] mismatches). Without CB, get_dummy_inputs builds BOTH
    # the query and the folded KV from `bs`, so BH matches and the fold export succeeds. Slot
    # identity is carried by the host-side DMA buffer view instead of a batch_index binding.
    qeff_model = QEFFAutoModelForImageTextToText.from_pretrained(
        model_id,
        attn_implementation="eager",
        kv_offload=True,
        config=config,
        dtype=torch.float16,
        layerwise=False,
    )
    tokenizer = transformers.AutoTokenizer.from_pretrained(model_id)
    processor = AutoProcessor.from_pretrained(model_id)

    vision_session = None
    if not skip_vision:
        vision_qpc_path = qeff_model.compile(
            batch_size=1,
            prefill_seq_len=prefill_seq_len,
            ctx_len=ctx_len,
            height=354,
            width=536,
            num_cores=16,
            num_devices=2,
            mos=1,
            aic_enable_depth_first=True,
            skip_vision=False,
            split_model_io=True,
            skip_lang=True,
            use_onnx_subfunctions=True,
            offload_pt_weights=False,
        )
        vision_session = QAICInferenceSession(vision_qpc_path.get("vision_qpc_path"))

    # Decode is compiled with batch_size = the number of prompts to decode together, and NO
    # continuous_batching, so BH = batch_size * num_key_value_heads and the folded KV cache is
    # [1, N*Hkv, ctx, D]. Without CB, get_dummy_inputs builds both the query and the folded KV
    # from this same batch, so the export traces cleanly. Compiled before prefill with
    # offload_pt_weights=False so the PyTorch weights stay resident for the prefill export/compile
    # below.
    decode_qpc_path = qeff_model.compile(
        batch_size=batch_size,  # drives BH = batch_size * num_kv_heads for the fold
        prefill_seq_len=1,
        ctx_len=ctx_len,
        height=354,
        width=536,
        num_cores=16,
        num_devices=decode_num_devices,
        mxfp6_matmul=True,
        mxint8_kv_cache=True,
        user_tiled=True,
        expert_parallel=True,  # This forces the model to use expert parallelism for the MoE layers
        tree_reduce=True,  # This enables tree reduction for the MoE layers, which can improve performance when using multiple devices
        cores_per_expert=2,  # number_of_parallelized_experts_per_device = total_experts * cores_per_expert / total_cores , total_cores = num_devices * num_cores, number_of_pipline_stages = total_experts / number_of_parallelized_experts_per_device
        split_retained_state_io=True,
        split_model_io=True,
        mos=1,
        prefill_only=False,
        skip_vision=True,
        use_onnx_subfunctions=True,
        layerwise=False,
        offload_pt_weights=False,
        qaic_config=_decode_qaic_config(ctx_len, num_kv_blocks),
    )

    # Prefill compiles batch_size=1 (one prompt per exec) with NO continuous_batching, so the
    # prefill KV cache is per-slot [1, Hkv, ctx, D] — exactly the per-slot shape the DMA handoff
    # writes into kv[slot:slot+1]. enable_chunking is mandatory for a prefill-only compile. The
    # head-parallel "online" prefill qaic_config (same as qwen3_vl_disagg_mode.py) overwrites the
    # stale batch_fold config the decode compile installed and routes prefill through the
    # online-prefill forward, so batch_fold is inert during the prefill trace.
    prefill_qpc_path = qeff_model.compile(
        batch_size=1,
        prefill_seq_len=prefill_seq_len,
        ctx_len=ctx_len,
        moe_prefill_packed_chunk_size=MOE_PREFILL_PACKED_CHUNK_SIZE,
        height=354,
        width=536,
        num_cores=16,
        num_devices=prefill_num_devices,
        mxfp6_matmul=True,
        retain_full_kv=True,
        mxint8_kv_cache=True,
        split_retained_state_io=True,
        mos=1,
        aic_enable_depth_first=True,
        # mdp_num_partitions=stages,
        prefill_only=True,
        enable_chunking=True,
        skip_vision=True,
        use_onnx_subfunctions=True,
        qaic_config=_prefill_qaic_config(ctx_len, num_kv_blocks, prefill_seq_len),
    )

    prefill_session = QAICInferenceSession(prefill_qpc_path.get("lang_prefill_qpc_path"), kv_dma_share=True)
    decode_session = QAICInferenceSession(decode_qpc_path.get("lang_decode_qpc_path"), kv_dma_share=True)

    # image_idx must be a compiled input binding; the KV-share path silently drops unknown
    # input names (warn + skip), so assert it up front. NOTE: with continuous_batching OFF there
    # is NO batch_index binding on either QPC — prefill targets a slot purely via the host buffer
    # view kv[slot:slot+1] (handoff offset 0), and decode writes the whole folded cache.
    assert "image_idx" in prefill_session.binding_index_map, "image_idx not a compiled prefill input binding"
    assert "batch_index" not in decode_session.binding_index_map, (
        "unexpected batch_index binding on the non-CB folded decode QPC"
    )
    decode_has_image_idx = "image_idx" in decode_session.binding_index_map

    # Shared host KV arrays. The decode session reports its cache folded as [1, N*Hkv, ctx, D];
    # we allocate the SAME bytes in the UN-folded [N, Hkv, ctx, D] shape so prefill's per-slot
    # kv[slot:slot+1] writes land correctly. A contiguous [N, Hkv, ctx, D] array and its
    # [1, N*Hkv, ctx, D] reshape are the same bytes, so decode reads/writes the fold as a
    # zero-copy view (see module docstring). Hybrid caches (mixed 4-D families) keep per-family
    # shapes; linear/recurrent (3-D) states are not folded and are unsupported here.
    kv_caches = []
    for shape, dtype in decode_session.kv_cache_info:
        assert len(shape) == 4, f"batch_fold expects 4-D KV families, got shape {shape}"
        folded_batch, bh, c, d = shape
        assert folded_batch == 1, f"folded decode KV dim0 must be 1, got {folded_batch}"
        assert bh % batch_size == 0, f"folded BH {bh} not divisible by batch_size {batch_size}"
        hkv = bh // batch_size
        kv_caches.append(np.zeros((batch_size, hkv, c, d), dtype=dtype))
    assert kv_caches and kv_caches[0].shape[0] == batch_size, (
        f"host KV batch dim {kv_caches[0].shape[0] if kv_caches else None} != batch_size {batch_size}"
    )
    # Folded views over the SAME bytes, [1, N*Hkv, ctx, D], to hand the decode handoff.
    decode_kv_views = [kv.reshape(1, kv.shape[0] * kv.shape[1], kv.shape[2], kv.shape[3]) for kv in kv_caches]
    decode_kv_map = decode_session.decode_buff_map + decode_session.decode_rs_kv_only_buff_map

    def _prepare_prompt(prompt: str, image_url: str):
        """Tokenise + (optionally) run the vision QPC for one prompt.

        ``image_url`` is used only when ``skip_vision=False``. Returns
        ``(lang_inputs, vision_outputs, num_chunks, num_pos_sections)`` where ``lang_inputs``
        is padded to a multiple of ``prefill_seq_len`` and carries ``position_ids`` /
        ``image_idx``, and ``vision_outputs`` is a dict with ``vision_embeds`` /
        ``deepstack_features`` (empty when ``skip_vision``).
        """
        if skip_vision:
            content = [{"type": "text", "text": prompt}]
        else:
            image = Image.open(requests.get(image_url, stream=True).raw)
            content = [{"type": "image", "image": image}, {"type": "text", "text": prompt}]
        messages = [{"role": "user", "content": content}]

        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs, video_inputs = process_vision_info([messages])
        inputs = processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        inputs = qeff_model.model.prepare_inputs_for_generation(
            inputs=inputs, prefill_seq_len=prefill_seq_len, batch_size=1
        )

        pad_token_id = 1
        input_ids_length = inputs["input_ids"].shape[1]
        num_chunks = -(input_ids_length // -prefill_seq_len)  # ceil divide without float
        padded_len = num_chunks * prefill_seq_len

        inputs["input_ids"] = torch.nn.functional.pad(
            inputs["input_ids"], (0, padded_len - input_ids_length), "constant", pad_token_id
        )
        inputs["attention_mask"] = torch.nn.functional.pad(
            inputs["attention_mask"], (0, padded_len - input_ids_length), "constant", 0
        )
        for k, v in inputs.items():
            inputs[k] = np.array(v)

        vision_inputs = {k: v for k, v in inputs.items() if k in VISION_INPUT_KEYS}
        vision_inputs.update({k: vision_inputs[k].astype("float16") for k in VISION_FP16_KEYS if k in vision_inputs})
        vision_outputs = {}
        if not skip_vision and vision_inputs:
            raw = vision_session.run(vision_inputs)
            vision_outputs = {k: raw[k] for k in VISION_OUTPUT_KEYS if k in raw}

        lang_inputs = {k: v for k, v in inputs.items() if k not in vision_inputs}
        if "position_ids" in inputs:
            lang_inputs["position_ids"] = inputs["position_ids"]
            lang_inputs.pop("attention_mask", None)
        else:
            lang_inputs["position_ids"] = np.where(
                lang_inputs.pop("attention_mask"), np.arange(padded_len), -1
            )  # -1 marks invalid positions
        lang_inputs["image_idx"] = np.array([[0]])
        num_pos_sections = lang_inputs["position_ids"].shape[0]
        return lang_inputs, vision_outputs, num_chunks, num_pos_sections

    def _prefill_slot(lang_inputs, vision_outputs, num_chunks, slot: int):
        """Chunked prefill of one prompt into KV ``slot`` (un-folded [N, Hkv, ctx, D] cache).

        With CB off the prefill device KV is per-slot [1, Hkv, ctx, D] and there is no
        batch_index binding: the on-device scatter always fills device row 0. Slot targeting is
        done purely by the host buffer view — the last chunk wires the DMA handoff of that single
        device row into ``kv_caches[*][slot]`` (the ``kv[slot:slot+1]`` view). The handoff offset
        must therefore stay 0, so batch_index is NOT added to the chunk inputs. Returns
        ``(first_token, phys_pos, mrope_pos)``.
        """
        chunk_inputs = dict(lang_inputs)
        if not skip_vision:
            for k in VISION_OUTPUT_KEYS:
                if k in vision_outputs:
                    chunk_inputs[k] = vision_outputs[k]
        slot_kv_view = [kv[slot : slot + 1] for kv in kv_caches]
        exec_idx = None
        for i in range(num_chunks):
            chunk_inputs["input_ids"] = lang_inputs["input_ids"][:, i * prefill_seq_len : (i + 1) * prefill_seq_len]
            chunk_inputs["position_ids"] = lang_inputs["position_ids"][
                ..., i * prefill_seq_len : (i + 1) * prefill_seq_len
            ]
            last_chunk = i == num_chunks - 1
            exec_idx = prefill_session.np_run_pipeline(
                chunk_inputs,
                last_chunk=last_chunk,
                kv_cache_buffers=slot_kv_view if last_chunk else None,
            )
            prefill_session.complete_inf(exec_idx, is_prefill=True)
            chunk_inputs["image_idx"] = prefill_session.get_outputs(index=exec_idx)["image_idx_output"]

        prefill_out = prefill_session.get_outputs(index=exec_idx)
        first_token = int(np.argmax(prefill_out["logits"]))
        phys_pos = int(lang_inputs["position_ids"][0].max()) + 1
        mrope_pos = (
            int(lang_inputs["position_ids"][1:].max()) + 1 if lang_inputs["position_ids"].shape[0] > 1 else phys_pos
        )
        return first_token, phys_pos, mrope_pos

    # Per-slot decode state. Pure batch folding: every slot is prefilled once and all slots
    # decode together in lockstep — there is NO continuous-batching queue or mid-decode refill.
    ongoing = [False] * batch_size
    last_token = [0] * batch_size
    phys_pos = [0] * batch_size
    mrope_pos = [0] * batch_size
    gen_count = [0] * batch_size
    slot_tokens = [None] * batch_size
    num_pos_sections = 1
    vision_outputs_ref = None

    def _seed_slot(slot, first_token, phys, mrope):
        slot_tokens[slot] = [first_token]
        gen_count[slot] = 1
        last_token[slot] = first_token
        phys_pos[slot] = phys
        mrope_pos[slot] = mrope
        ongoing[slot] = True

    # One prompt per slot, up to batch_size slots. Slot i serves prompt i (no queue). If more
    # prompts than slots are supplied, only the first batch_size are served — pure batch folding
    # has no request-refill path; raise batch_size (or run again) for the rest.
    num_active = min(batch_size, len(prompts))
    if len(prompts) > batch_size:
        print(f"NOTE: {len(prompts)} prompts > batch_size {batch_size}; only the first {batch_size} are served.")
    results = [None] * len(prompts)

    prefill_start = perf_counter()
    for slot in range(num_active):
        prompt = prompts[slot]
        prompt_image_url = image_urls[slot % len(image_urls)]
        lang_inputs, vision_outputs, num_chunks, num_pos_sections = _prepare_prompt(prompt, prompt_image_url)
        vision_outputs_ref = vision_outputs if vision_outputs else vision_outputs_ref
        ft, phys, mrope = _prefill_slot(lang_inputs, vision_outputs, num_chunks, slot)
        _seed_slot(slot, ft, phys, mrope)
    print(f"Initial prefill time : {perf_counter() - prefill_start:.2f} secs")

    # Decode does not re-gather image tokens (image_idx has advanced past them), but the
    # vision_embeds / deepstack_features bindings must still be satisfied every step. Bind
    # constant zeros buffers of the compiled shapes; their values are never used by the
    # text-token decode path. The buffers MUST match the DECODE binding's own compiled dims
    # (the export-dummy shape, e.g. deepstack_features = [num_layers, fbs, vision_size, C]),
    # NOT the vision QPC output shape — the two differ in vision_size and batch dim, which is
    # what triggers the "input buffer deepstack_features is expected to have size ..." error.
    if not skip_vision and vision_outputs_ref:
        persistent = {}
        for k in VISION_OUTPUT_KEYS:
            if k not in decode_session.binding_index_map:
                continue
            binding = decode_session.bindings[decode_session.binding_index_map[k]]
            dtype = decode_session.aic_to_np_dtype_mapping[binding.type]
            persistent[k] = np.zeros(tuple(binding.dims), dtype=dtype)
        decode_session.set_persistent_inputs(persistent)

    def _build_decode_inputs():
        input_ids = np.full((batch_size, 1), -1, dtype=np.int64)
        position_ids = np.full((num_pos_sections, batch_size, 1), -1, dtype=np.int64)
        for slot in range(num_active):
            if not ongoing[slot]:
                continue
            input_ids[slot, 0] = last_token[slot]
            position_ids[0, slot, 0] = phys_pos[slot]
            position_ids[1:, slot, 0] = mrope_pos[slot]
        # No batch_index: the folded decode has no batch_index binding (CB is off). Slot identity
        # lives in the folded row block b*Hkv+h, and the handoff is a whole-cache identity DMA.
        decode_inputs = {
            "input_ids": input_ids,
            "position_ids": position_ids,
        }
        if decode_has_image_idx:
            # image_idx is a fixed (1,1) binding: decode does not re-gather image tokens (they
            # were merged into KV during prefill), so it is inert here. A static [[0]] satisfies it.
            decode_inputs["image_idx"] = np.array([[0]], dtype=np.int64)
        return decode_inputs

    st = perf_counter()
    decode_steps = 0
    # Fixed-length lockstep decode. All active slots step together; a slot goes inactive once it
    # emits EOS or reaches generation_len, and the loop ends when every slot is done (or the step
    # budget is exhausted). The first token was produced by prefill (gen_count seeded to 1).
    for _ in range(generation_len):
        if not any(ongoing):
            break
        # Wire the folded [1, N*Hkv, ctx, D] views (same host bytes as the [N, Hkv, ctx, D]
        # cache). batch_index=0, ctx_start=0 -> identity DMA of the whole fused cache over the
        # whole host view; the fold row block b*Hkv+h keeps each slot in place. No numpy copy.
        decode_session.set_data_for_kv_handoff(
            decode_kv_views + decode_kv_views,
            [("batch_index", 0), ("ctx_start", 0)],
            index=decode_session.decode_execObj_idx,
            buff_map=decode_kv_map,
        )
        decode_inputs = _build_decode_inputs()
        exec_idx = decode_session.np_run(decode_inputs, is_prefill=False)
        decode_session.complete_inf(exec_idx, is_prefill=False)
        out = decode_session.get_outputs(index=exec_idx)
        decode_steps += 1

        logits = out["logits"]
        logits = logits.reshape(batch_size, -1, logits.shape[-1])[:, -1, :]
        next_tokens = np.argmax(logits, axis=-1)

        for slot in range(num_active):
            if not ongoing[slot]:
                continue
            tok = int(next_tokens[slot])
            slot_tokens[slot].append(tok)
            gen_count[slot] += 1
            last_token[slot] = tok
            phys_pos[slot] += 1
            mrope_pos[slot] += 1
            if tok == tokenizer.eos_token_id or gen_count[slot] >= generation_len:
                ongoing[slot] = False
    ft = perf_counter()

    # Slot i served prompt i (no queue), so results index == slot index for the active slots.
    for slot in range(num_active):
        results[slot] = slot_tokens[slot]

    total_tokens = sum(len(t) for t in results if t)
    print(f"decode steps={decode_steps} tok/sec={total_tokens / (ft - st):.2f}")
    first_tokens = []
    for idx, prompt in enumerate(prompts):
        toks = results[idx] or []
        first_tokens.append(toks[0] if toks else None)
        print(f"\ninput [{idx}]\n{prompt}\noutput\n{tokenizer.decode(toks)}")

    return {"first_tokens": first_tokens, "tokens": results}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model-id", default=DEFAULT_MODEL_ID, help="HF model id")
    parser.add_argument("--prompt", action="append", dest="prompts", help="prompt (repeatable); defaults to a set of 4")
    parser.add_argument("--prefill-seq-len", type=int, default=DEFAULT_PREFILL_SEQ_LEN)
    parser.add_argument("--ctx-len", type=int, default=DEFAULT_CTX_LEN)
    parser.add_argument("--generation-len", type=int, default=DEFAULT_GENERATION_LEN)
    parser.add_argument(
        "--batch-size", type=int, default=DEFAULT_BATCH_SIZE, help="folded decode batch width (N prompts)"
    )
    parser.add_argument("--stages", type=int, default=STAGES, help="prefill pipeline depth (mdp_num_partitions)")
    parser.add_argument(
        "--prefill-num-devices", type=int, default=PREFILL_NUM_DEVICES, help="num devices for the prefill QPC"
    )
    parser.add_argument(
        "--decode-num-devices", type=int, default=DECODE_NUM_DEVICES, help="num devices for the decode QPC"
    )
    parser.add_argument(
        "--num-kv-blocks",
        type=int,
        default=NUM_KV_BLOCKS,
        help="decode KV-blocking chunk count (batch-folded 'kv' blocking)",
    )
    vision = parser.add_mutually_exclusive_group()
    vision.add_argument("--skip-vision", dest="skip_vision", action="store_true", help="text-only lang path")
    vision.add_argument(
        "--with-vision",
        dest="skip_vision",
        action="store_false",
        help="image+text: compile and run the vision QPC, pairing each prompt with an --image-url (default)",
    )
    parser.set_defaults(skip_vision=False)
    parser.add_argument(
        "--image-url",
        action="append",
        dest="image_urls",
        help="image URL (repeatable); paired with prompts by index. Defaults to a set of 4",
    )
    args = parser.parse_args()

    # With vision the default text prompts are a poor fit; use the image prompts unless the
    # user supplied their own --prompt.
    prompts = args.prompts
    image_urls = args.image_urls
    if not args.skip_vision:
        if not prompts:
            prompts = list(DEFAULT_IMAGE_PROMPTS)
        if not image_urls:
            image_urls = list(DEFAULT_IMAGE_URLS)

    run(
        model_id=args.model_id,
        prompts=prompts,
        prefill_seq_len=args.prefill_seq_len,
        ctx_len=args.ctx_len,
        generation_len=args.generation_len,
        batch_size=args.batch_size,
        skip_vision=args.skip_vision,
        image_urls=image_urls,
        stages=args.stages,
        prefill_num_devices=args.prefill_num_devices,
        decode_num_devices=args.decode_num_devices,
        num_kv_blocks=args.num_kv_blocks,
    )

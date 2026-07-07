# -----------------------------------------------------------------------------
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------

"""Disaggregated prefill/decode for Gemma4 — DMA-based KV handoff.

Gemma4 uses a **hybrid** KV cache (``layer_types`` mixes ``sliding_attention`` and
``full_attention``), so ``decode_session.kv_cache_info`` carries more than one 4-D shape
family and the session auto-selects the full (per-binding) slicing spec — no example-level
branching is needed. The lang forward also threads a per-chunk ``mm_token_type_ids`` (sliced
alongside ``input_ids``/``position_ids``) and a constant ``vision_embeds``; the latter is
registered once via ``set_persistent_inputs`` rather than re-supplied every step. ``image_idx``
threads serially from each step's ``image_idx_output`` into the next step's input, so prefill
runs as a simple serial loop; only the last chunk wires the DMA handoff into the shared host
arrays, after all KV has accumulated on-device.

Supports both text-only (``skip_vision=True``) and image+text (``skip_vision=False``, the
default) inputs; the latter compiles a vision QPC and runs it once.

The body is exposed as ``run(...)`` returning the prefill ``logits``, the ``first_token``
and the full decoded ``tokens`` list.
"""

import argparse
from time import perf_counter

import numpy as np
import torch
import transformers
from gemma4_utils import (
    CHAT_TEMPLATE,
    build_messages,
    remove_fp16clip_transform_if_disabled,
)
from transformers import AutoConfig, AutoProcessor

from QEfficient import QEFFAutoModelForImageTextToText
from QEfficient.generation.cloud_infer import QAICInferenceSession

DEFAULT_MODEL_ID = "google/gemma-4-26B-A4B-it"
SYSTEM_PROMPT = "You are a helpful assistant."
DEFAULT_PROMPT = "Tell me about Taj Mahal?"
DEFAULT_IMAGE_PROMPT = "Can you Describe this image in detail?"
DEFAULT_IMAGE_URL = (
    "https://wallup.net/wp-content/uploads/2017/03/28/"
    "351036-San_Francisco-USA-bridge-sunset-Golden_Gate_Bridge-lights.jpg"
)
DEFAULT_PREFILL_SEQ_LEN = 296
DEFAULT_CTX_LEN = 4096
DEFAULT_GENERATION_LEN = 200

ENABLE_FP16_CLIP = True
STAGES = 4
PREFILL_NUM_DEVICES = 8
DECODE_NUM_DEVICES = 4
BS = 1

# Vision-tower inputs are routed to the vision QPC; everything else feeds the lang QPCs.
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


def _build_config(model_id: str):
    """Load the model config (gemma4 defaults to float32 weights)."""
    config = AutoConfig.from_pretrained(model_id)

    # For faster execution user can run with fewer layers. For testing purposes only.
    # config.text_config.num_hidden_layers = 2
    # if getattr(config.text_config, "layer_types", None):
    #     config.text_config.layer_types = config.text_config.layer_types[: config.text_config.num_hidden_layers]
    # config.vision_config.num_hidden_layers = 2
    return config


def _resolve_lang_qpc_path(qpc_obj, preferred_keys):
    if isinstance(qpc_obj, dict):
        for key in preferred_keys:
            if key in qpc_obj:
                return qpc_obj[key]
        raise KeyError(f"Could not find any of {preferred_keys} in compile output keys: {list(qpc_obj.keys())}")
    if isinstance(qpc_obj, (list, tuple)):
        # Backward-compat: some codepaths return (vision_qpc, lang_qpc)
        return qpc_obj[1]
    return qpc_obj


def _resolve_vision_qpc_path(qpc_obj, preferred_keys=("vision_qpc_path",)):
    if isinstance(qpc_obj, dict):
        for key in preferred_keys:
            if key in qpc_obj:
                return qpc_obj[key]
        raise KeyError(f"Could not find any of {preferred_keys} in compile output keys: {list(qpc_obj.keys())}")
    if isinstance(qpc_obj, (list, tuple)):
        # Backward-compat: some codepaths return (vision_qpc, lang_qpc)
        return qpc_obj[0]
    return qpc_obj


def run(
    model_id: str = DEFAULT_MODEL_ID,
    prompt: str = DEFAULT_PROMPT,
    prefill_seq_len: int = DEFAULT_PREFILL_SEQ_LEN,
    ctx_len: int = DEFAULT_CTX_LEN,
    generation_len: int = DEFAULT_GENERATION_LEN,
    skip_vision: bool = True,
    image_url: str = DEFAULT_IMAGE_URL,
    stages: int = STAGES,
    prefill_num_devices: int = PREFILL_NUM_DEVICES,
    decode_num_devices: int = DECODE_NUM_DEVICES,
):
    """Run chunked-prefill + decode with the DMA-based KV handoff.

    ``skip_vision=False`` (default) fetches ``image_url`` and runs an image+text prompt
    through the vision QPC; ``skip_vision=True`` runs a text-only prompt. Returns a dict
    with the prefill ``logits``, the ``first_token`` (argmax over those logits) and the
    full decoded ``tokens`` list, for parity comparison.
    """
    config = _build_config(model_id)

    qeff_model = QEFFAutoModelForImageTextToText.from_pretrained(
        model_id, attn_implementation="eager", kv_offload=True, config=config, dtype="float32", trust_remote_code=True
    )
    tokenizer = transformers.AutoTokenizer.from_pretrained(model_id)
    processor = AutoProcessor.from_pretrained(model_id)

    remove_fp16clip_transform_if_disabled(qeff_model, ENABLE_FP16_CLIP)

    vision_session = None
    if not skip_vision:
        vision_qpc_path = qeff_model.compile(
            batch_size=BS,
            prefill_seq_len=prefill_seq_len,
            ctx_len=ctx_len,
            num_cores=16,
            num_devices=1,
            mos=1,
            mxfp6_matmul=True,
            aic_enable_depth_first=True,
            skip_vision=False,
            split_model_io=True,
            use_onnx_subfunctions=True,
            skip_lang=True,
        )
        vision_session = QAICInferenceSession(_resolve_vision_qpc_path(vision_qpc_path))

    # split_retained_state_io=True gives distinct KV input vs. *_RetainedState output
    # buffers — required by the DMA handoff, which wires both sides at the shared host
    # arrays. (The baseline's split_model_io is an orthogonal VLM vision/lang split.)
    prefill_qpc_path = qeff_model.compile(
        batch_size=BS,
        prefill_seq_len=prefill_seq_len,
        ctx_len=ctx_len,
        num_cores=16,
        num_devices=prefill_num_devices,
        mxfp6_matmul=True,
        mxint8_kv_cache=True,
        retain_full_kv=True,
        split_retained_state_io=True,
        mos=1,
        aic_enable_depth_first=True,
        node_precision_info=True,
        prefill_only=True,
        enable_chunking=True,
        use_onnx_subfunctions=True,
        skip_vision=True,
        mdp_num_partitions=stages,
    )

    decode_qpc_path = qeff_model.compile(
        batch_size=BS,
        prefill_seq_len=1,
        ctx_len=ctx_len,
        num_cores=16,
        num_devices=decode_num_devices,
        mxfp6_matmul=True,
        mxint8_kv_cache=True,
        retain_full_kv=True,  # required for DMA slice writes into full KV
        split_retained_state_io=True,
        mos=1,
        aic_enable_depth_first=True,
        node_precision_info=True,
        prefill_only=False,
        use_onnx_subfunctions=True,
        skip_vision=True,
    )

    prefill_session = QAICInferenceSession(
        _resolve_lang_qpc_path(prefill_qpc_path, ("lang_prefill_qpc_path", "lang_qpc_path")), kv_dma_share=True
    )
    decode_session = QAICInferenceSession(
        _resolve_lang_qpc_path(decode_qpc_path, ("lang_decode_qpc_path", "lang_qpc_path")), kv_dma_share=True
    )

    # image_idx must be a compiled prefill input binding; the KV-share path silently
    # drops unknown input names (warn + skip), so assert it up front. The decode QPC may
    # not bind it, so treat it as optional there and only wire it when present.
    assert "image_idx" in prefill_session.binding_index_map, "image_idx not a compiled prefill input binding"
    decode_has_image_idx = "image_idx" in decode_session.binding_index_map

    chat_template = (
        getattr(processor, "chat_template", None) or getattr(tokenizer, "chat_template", None) or CHAT_TEMPLATE
    )
    if skip_vision:
        messages = build_messages(SYSTEM_PROMPT, prompt, use_image=False)
    else:
        messages = build_messages(SYSTEM_PROMPT, prompt, use_image=True)
        messages[-1]["content"][0]["url"] = image_url

    inputs = processor.apply_chat_template(
        messages,
        chat_template=chat_template,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
    )

    pad_token_id = 1
    input_ids_length = inputs["input_ids"].shape[1]
    num_chunks = -(input_ids_length // -prefill_seq_len)  # ceil divide without float
    padded_len = num_chunks * prefill_seq_len  # Convert to a multiple of prompt_len

    inputs["input_ids"] = torch.nn.functional.pad(
        inputs["input_ids"],
        (0, padded_len - input_ids_length),
        "constant",
        pad_token_id,
    )
    inputs["attention_mask"] = torch.nn.functional.pad(
        inputs["attention_mask"], (0, padded_len - input_ids_length), "constant", 0
    )
    if "mm_token_type_ids" in inputs:
        inputs["mm_token_type_ids"] = torch.nn.functional.pad(
            inputs["mm_token_type_ids"], (0, padded_len - input_ids_length), "constant", 0
        )

    for k, v in inputs.items():
        inputs[k] = np.array(v)

    # ---- Vision (producer): run the vision QPC once; its output feeds every lang step ----
    vision_inputs = {k: v for k, v in inputs.items() if k in VISION_INPUT_KEYS}
    vision_inputs.update({k: vision_inputs[k].astype("float16") for k in VISION_FP16_KEYS if k in vision_inputs})
    vision_embeds = None
    if not skip_vision and vision_inputs:
        vision_embeds = vision_session.run(vision_inputs)["vision_embeds"]

    lang_inputs = {k: v for k, v in inputs.items() if k not in vision_inputs}
    if "position_ids" in inputs:
        lang_inputs["position_ids"] = inputs["position_ids"]
        lang_inputs.pop("attention_mask", None)
    else:
        lang_inputs["position_ids"] = np.where(
            lang_inputs.pop("attention_mask"), np.arange(padded_len), -1
        )  # Need to use -1 as position_ids for invalid tokens

    # mm_token_type_ids is a per-chunk lang input; synthesize zeros if the processor
    # omitted it (e.g. text-only) so prefill slicing and the binding are always satisfied.
    if "mm_token_type_ids" not in lang_inputs:
        lang_inputs["mm_token_type_ids"] = np.zeros((BS, padded_len), dtype=np.int64)

    lang_inputs["image_idx"] = np.array([[0]])
    # vision_embeds is constant across every prefill chunk and decode step, so register it
    # once as a persistent input instead of re-supplying it in each step's dict (a large
    # per-step host->device copy). set_persistent_inputs appends it to each enqueue's tuple
    # list automatically (warning+skipping any name a session does not bind).
    if not skip_vision:
        prefill_session.set_persistent_inputs({"vision_embeds": vision_embeds})

    # The decode QPC binds mm_token_type_ids (seq_len=1 ignores its value, but the pooled
    # np_run path wires only the bindings it is handed), so register a constant zeros to
    # satisfy it; register vision_embeds on decode too when present.
    decode_persist = {"mm_token_type_ids": np.zeros((BS, 1), dtype=np.int64)}
    if not skip_vision:
        decode_persist["vision_embeds"] = vision_embeds
    decode_session.set_persistent_inputs(
        {k: v for k, v in decode_persist.items() if k in decode_session.binding_index_map}
    )

    # Shared host KV arrays, allocated once in decode-map order (per-slot shape[0] == 1).
    # Hybrid: kv_cache_info carries mixed sliding-window and full-attention 4-D shapes.
    kv_caches = [np.zeros(shape, dtype=dtype) for (shape, dtype) in decode_session.kv_cache_info]

    # ---- Prefill (producer, SERIAL): image_idx threads chunk-to-chunk ----
    # Only the LAST chunk wires the DMA handoff into kv_caches (earlier chunks just
    # accumulate KV on-device). np_run_pipeline selects decode_rs_full_buff_map
    # internally for the hybrid cache.
    prefill_start = perf_counter()
    chunk_inputs = dict(lang_inputs)
    exec_idx = None
    for i in range(num_chunks):
        chunk_inputs["input_ids"] = lang_inputs["input_ids"][:, i * prefill_seq_len : (i + 1) * prefill_seq_len]
        chunk_inputs["position_ids"] = lang_inputs["position_ids"][..., i * prefill_seq_len : (i + 1) * prefill_seq_len]
        chunk_inputs["mm_token_type_ids"] = lang_inputs["mm_token_type_ids"][
            ..., i * prefill_seq_len : (i + 1) * prefill_seq_len
        ]
        last_chunk = i == num_chunks - 1
        exec_idx = prefill_session.np_run_pipeline(
            chunk_inputs,
            last_chunk=last_chunk,
            kv_cache_buffers=kv_caches if last_chunk else None,
        )
        prefill_session.complete_inf(exec_idx, is_prefill=True)
        chunk_inputs["image_idx"] = prefill_session.get_outputs(index=exec_idx)["image_idx_output"]
    print(f"Prefill time : {perf_counter() - prefill_start:.2f} secs")

    prefill_out = prefill_session.get_outputs(index=exec_idx)
    prefill_logits = prefill_out["logits"]
    first_token = int(np.argmax(prefill_logits))
    all_outputs = [first_token]

    # ---- Decode (consumer): re-point DMA descriptor at kv_caches EVERY step ----
    # With split_retained_state_io=True the KV *input* binding (decode_buff_map) and the
    # *_RetainedState output* (decode_rs_kv_only_buff_map) are distinct device buffers —
    # exactly the two sides the baseline bridges with a host copy each step. Wire BOTH at
    # kv_caches: the input for the read, the RS output for the write-back of the newly
    # decoded position. Both maps share the same layer-sorted family order, so the two
    # concatenated maps line up with kv_caches repeated.
    decode_kv_map = decode_session.decode_buff_map + decode_session.decode_rs_kv_only_buff_map
    pos = int(np.max(lang_inputs["position_ids"])) + 1
    decode_inputs = {
        "input_ids": np.array(first_token, dtype=np.int64).reshape(1, 1),
        "position_ids": np.array([[pos]], dtype=np.int64),
    }
    if decode_has_image_idx:
        decode_inputs["image_idx"] = prefill_out["image_idx_output"]
    st = perf_counter()
    for _ in range(generation_len - 1):
        decode_session.set_data_for_kv_handoff(
            kv_caches + kv_caches,
            [("batch_index", 0), ("ctx_start", 0)],
            index=decode_session.decode_execObj_idx,
            buff_map=decode_kv_map,
        )
        exec_idx = decode_session.np_run(decode_inputs, is_prefill=False)
        decode_session.complete_inf(exec_idx, is_prefill=False)
        out = decode_session.get_outputs(index=exec_idx)
        tok = int(np.argmax(out["logits"]))
        all_outputs.append(tok)
        pos += 1
        decode_inputs = {
            "input_ids": np.array(tok, dtype=np.int64).reshape(1, 1),
            "position_ids": np.array([[pos]], dtype=np.int64),
        }
        if decode_has_image_idx:
            decode_inputs["image_idx"] = out["image_idx_output"]
    ft = perf_counter()

    print(f"decode tok/sec={(generation_len - 1) / (ft - st)}")
    print(f"input\n{prompt}\noutput\n{tokenizer.decode(all_outputs)}")

    return {"logits": prefill_logits, "first_token": first_token, "tokens": all_outputs}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model-id", default=DEFAULT_MODEL_ID, help="HF model id")
    parser.add_argument("--prompt", default=DEFAULT_PROMPT, help="text prompt")
    parser.add_argument("--prefill-seq-len", type=int, default=DEFAULT_PREFILL_SEQ_LEN)
    parser.add_argument("--ctx-len", type=int, default=DEFAULT_CTX_LEN)
    parser.add_argument("--generation-len", type=int, default=DEFAULT_GENERATION_LEN)
    parser.add_argument("--stages", type=int, default=STAGES, help="prefill pipeline depth (mdp_num_partitions)")
    parser.add_argument(
        "--prefill-num-devices", type=int, default=PREFILL_NUM_DEVICES, help="num devices for the prefill QPC"
    )
    parser.add_argument(
        "--decode-num-devices", type=int, default=DECODE_NUM_DEVICES, help="num devices for the decode QPC"
    )
    vision = parser.add_mutually_exclusive_group()
    vision.add_argument(
        "--skip-vision",
        dest="skip_vision",
        action="store_true",
        help="text-only lang path",
    )
    vision.add_argument(
        "--with-vision",
        dest="skip_vision",
        action="store_false",
        help="image+text: compile and run the vision QPC on --image-url (default)",
    )
    parser.set_defaults(skip_vision=False)
    parser.add_argument("--image-url", default=DEFAULT_IMAGE_URL, help="image URL when --with-vision")
    args = parser.parse_args()

    # With vision the default text prompt is a poor fit; use the image prompt unless
    # the user overrode --prompt.
    prompt = args.prompt
    if not args.skip_vision and prompt == DEFAULT_PROMPT:
        prompt = DEFAULT_IMAGE_PROMPT

    run(
        model_id=args.model_id,
        prompt=prompt,
        prefill_seq_len=args.prefill_seq_len,
        ctx_len=args.ctx_len,
        generation_len=args.generation_len,
        skip_vision=args.skip_vision,
        image_url=args.image_url,
        stages=args.stages,
        prefill_num_devices=args.prefill_num_devices,
        decode_num_devices=args.decode_num_devices,
    )

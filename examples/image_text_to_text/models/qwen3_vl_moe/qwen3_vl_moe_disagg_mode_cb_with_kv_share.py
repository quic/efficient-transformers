# -----------------------------------------------------------------------------
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------

"""Continuous-batching disaggregated prefill/decode for Qwen3-VL-MoE — DMA KV handoff."""

import argparse
from collections import deque
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
DEFAULT_PREFILL_SEQ_LEN = 128
DEFAULT_CTX_LEN = 2048
DEFAULT_GENERATION_LEN = 100
DEFAULT_FULL_BATCH_SIZE = 32

STAGES = 4
PREFILL_NUM_DEVICES = 8
DECODE_NUM_DEVICES = 4
# NOTE: the prefill-exec (input_ids) batch is always 1 under CB — each prompt is prefilled
# individually into its own KV slot; the batch dimension is `full_batch_size` (KV slots).

VISION_INPUT_KEYS = {
    "pixel_values",
    "image_masks",
    "image_input_idx",
    "valid_idx",
    "aspect_ratio_ids",
    "aspect_ratio_mask",
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
    # config.text_config.num_hidden_layers = 6
    # config.vision_config.deepstack_visual_indexes = [8]
    return config


def run(
    model_id: str = DEFAULT_MODEL_ID,
    prompts=None,
    prefill_seq_len: int = DEFAULT_PREFILL_SEQ_LEN,
    ctx_len: int = DEFAULT_CTX_LEN,
    generation_len: int = DEFAULT_GENERATION_LEN,
    full_batch_size: int = DEFAULT_FULL_BATCH_SIZE,
    skip_vision: bool = False,
    image_urls=None,
    stages: int = STAGES,
    prefill_num_devices: int = PREFILL_NUM_DEVICES,
    decode_num_devices: int = DECODE_NUM_DEVICES,
):
    """Run CB (chunked-prefill + batched decode) over ``prompts`` with the DMA KV handoff.

    ``skip_vision=False`` (default) pairs each prompt with the image at the same index of
    ``image_urls`` (cycled if shorter) and runs it as an image+text turn through the vision
    QPC; ``skip_vision=True`` runs text-only prompts. Returns a dict with, per prompt, the
    ``first_tokens`` (prefill argmax) and the full decoded ``tokens`` list, for parity
    comparison against the single-request driver.
    """
    prompts = list(prompts) if prompts else list(DEFAULT_PROMPTS)
    image_urls = list(image_urls) if image_urls else list(DEFAULT_IMAGE_URLS)
    # Repeat a single prompt/image across all full_batch_size slots (the CB analog of
    # replicating one turn to fill the batch): pass exactly one prompt (and, with vision,
    # one image) to run every slot with the same turn. A multi-element list is used as-is
    # (distinct prompts per slot).
    if len(prompts) == 1:
        prompts = prompts * full_batch_size
    if len(image_urls) == 1:
        image_urls = image_urls * full_batch_size
    config = _build_config(model_id)

    qeff_model = QEFFAutoModelForImageTextToText.from_pretrained(
        model_id,
        attn_implementation="eager",
        kv_offload=True,
        continuous_batching=True,
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
            full_batch_size=full_batch_size,  # required by the CB compile guard; vision spec stays batch_size=1
            prefill_seq_len=prefill_seq_len,
            ctx_len=ctx_len,
            height=354,
            width=536,
            num_cores=16,
            num_devices=2,
            mos=1,
            mxfp6_matmul=True,
            aic_enable_depth_first=True,
            skip_vision=False,
            split_model_io=True,
            skip_lang=True,
            use_onnx_subfunctions=True,
            offload_pt_weights=False,
        )
        vision_session = QAICInferenceSession(vision_qpc_path.get("vision_qpc_path"))

    # full_batch_size pins the KV/RetainedState batch axis to N on BOTH QPCs. Decode is
    # compiled before prefill with offload_pt_weights=False so the PyTorch weights stay
    # resident for the prefill export/compile below.
    decode_qpc_path = qeff_model.compile(
        batch_size=1,
        full_batch_size=full_batch_size,
        prefill_seq_len=1,
        ctx_len=ctx_len,
        height=354,
        width=536,
        num_cores=16,
        num_devices=decode_num_devices,
        mxfp6_matmul=True,
        mxint8_kv_cache=True,
        retain_full_kv=True,  # required for DMA slice writes into full KV
        split_retained_state_io=True,
        mos=1,
        aic_enable_depth_first=True,
        prefill_only=False,
        skip_vision=True,
        use_onnx_subfunctions=True,
        offload_pt_weights=False,
    )

    # enable_chunking is mandatory for a CB prefill-only compile.
    prefill_qpc_path = qeff_model.compile(
        batch_size=1,
        full_batch_size=full_batch_size,
        prefill_seq_len=prefill_seq_len,
        ctx_len=ctx_len,
        height=354,
        width=536,
        num_cores=16,
        num_devices=prefill_num_devices,
        mxfp6_matmul=True,
        mxint8_kv_cache=True,
        retain_full_kv=True,
        split_retained_state_io=True,
        mos=1,
        aic_enable_depth_first=True,
        mdp_num_partitions=stages,
        prefill_only=True,
        enable_chunking=True,
        skip_vision=True,
        use_onnx_subfunctions=True,
    )

    prefill_session = QAICInferenceSession(
        prefill_qpc_path.get("lang_prefill_qpc_path"), kv_dma_share=True, full_batch_size=full_batch_size
    )
    decode_session = QAICInferenceSession(
        decode_qpc_path.get("lang_decode_qpc_path"), kv_dma_share=True, full_batch_size=full_batch_size
    )

    # image_idx / batch_index must be compiled input bindings; the KV-share path silently drops
    # unknown input names (warn + skip), so assert the ones we always rely on up front.
    assert "image_idx" in prefill_session.binding_index_map, "image_idx not a compiled prefill input binding"
    assert "batch_index" in decode_session.binding_index_map, "batch_index not a compiled decode input binding"
    decode_has_image_idx = "image_idx" in decode_session.binding_index_map

    # Shared host KV arrays, allocated once in decode-map order. Under CB the leading batch
    # dim is full_batch_size, so each family is [N, ...]: prefill writes one row, decode
    # reads/writes all N. Hybrid caches carry mixed 4-D (full) and 3-D (linear) shapes.
    kv_caches = [np.zeros(shape, dtype=dtype) for (shape, dtype) in decode_session.kv_cache_info]
    assert kv_caches and kv_caches[0].shape[0] == full_batch_size, (
        f"decode KV batch dim {kv_caches[0].shape[0] if kv_caches else None} != full_batch_size {full_batch_size}"
    )
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
        """Chunked prefill of one prompt into KV ``slot``.

        Every chunk carries ``batch_index=slot`` so the on-device scatter accumulates into
        row ``slot``; the last chunk wires the DMA handoff of that single row into
        ``kv_caches[*][slot]``. Returns ``(first_token, phys_pos, mrope_pos)``.
        """
        chunk_inputs = dict(lang_inputs)
        chunk_inputs["batch_index"] = np.array([[slot]], dtype=np.int64)
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

    # Per-slot decode state.
    ongoing = [False] * full_batch_size
    last_token = [0] * full_batch_size
    phys_pos = [0] * full_batch_size
    mrope_pos = [0] * full_batch_size
    gen_count = [0] * full_batch_size
    slot_prompt_idx = [-1] * full_batch_size
    slot_tokens = [None] * full_batch_size
    results = [None] * len(prompts)
    num_pos_sections = 1
    vision_outputs_ref = None

    def _seed_slot(slot, prompt_idx, first_token, phys, mrope):
        slot_prompt_idx[slot] = prompt_idx
        slot_tokens[slot] = [first_token]
        gen_count[slot] = 1
        last_token[slot] = first_token
        phys_pos[slot] = phys
        mrope_pos[slot] = mrope
        ongoing[slot] = True

    # Prompt queue: each entry is (prompt_idx, prompt, image_url). Everything beyond the
    # first N slots waits here and refills on completion. image_urls is cycled if shorter.
    prompt_queue = deque((idx, prompt, image_urls[idx % len(image_urls)]) for idx, prompt in enumerate(prompts))

    prefill_start = perf_counter()
    for slot in range(full_batch_size):
        if not prompt_queue:
            break
        prompt_idx, prompt, prompt_image_url = prompt_queue.popleft()
        lang_inputs, vision_outputs, num_chunks, num_pos_sections = _prepare_prompt(prompt, prompt_image_url)
        vision_outputs_ref = vision_outputs if vision_outputs else vision_outputs_ref
        ft, phys, mrope = _prefill_slot(lang_inputs, vision_outputs, num_chunks, slot)
        _seed_slot(slot, prompt_idx, ft, phys, mrope)
    print(f"Initial prefill time : {perf_counter() - prefill_start:.2f} secs")

    # Decode does not re-gather image tokens (image_idx has advanced past them), but the
    # vision_embeds / deepstack_features bindings must still be satisfied every step. Bind
    # constant zeros buffers of the compiled shapes; their values are never used by the
    # text-token decode path. Only wire keys that are actual decode bindings.
    if not skip_vision and vision_outputs_ref:
        decode_session.set_persistent_inputs(
            {k: np.zeros_like(v) for k, v in vision_outputs_ref.items() if k in decode_session.binding_index_map}
        )

    def _build_decode_inputs():
        input_ids = np.full((full_batch_size, 1), -1, dtype=np.int64)
        position_ids = np.full((num_pos_sections, full_batch_size, 1), -1, dtype=np.int64)
        batch_index = np.full((full_batch_size, 1), -1, dtype=np.int64)
        for slot in range(full_batch_size):
            if not ongoing[slot]:
                continue
            input_ids[slot, 0] = last_token[slot]
            position_ids[0, slot, 0] = phys_pos[slot]
            position_ids[1:, slot, 0] = mrope_pos[slot]
            batch_index[slot, 0] = slot
        decode_inputs = {
            "input_ids": input_ids,
            "position_ids": position_ids,
            "batch_index": batch_index,
        }
        if decode_has_image_idx:
            # image_idx is a fixed (1,1) binding, NOT widened to full_batch_size: decode does
            # not re-gather image tokens (they were merged into KV during prefill), so it is
            # inert here. A static [[0]] satisfies the binding.
            decode_inputs["image_idx"] = np.array([[0]], dtype=np.int64)
        return decode_inputs

    st = perf_counter()
    decode_steps = 0
    while any(ongoing):
        # Wire the full [N, ...] KV buffers once (identity: device row i <-> host row i);
        # per-slot addressing is carried by the decode batch_index input above.
        decode_session.set_data_for_kv_handoff(
            kv_caches + kv_caches,
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
        logits = logits.reshape(full_batch_size, -1, logits.shape[-1])[:, -1, :]
        next_tokens = np.argmax(logits, axis=-1)

        for slot in range(full_batch_size):
            if not ongoing[slot]:
                continue
            tok = int(next_tokens[slot])
            if tok == tokenizer.eos_token_id or gen_count[slot] >= generation_len:
                # Slot finished: record its output, then refill from the queue or retire.
                results[slot_prompt_idx[slot]] = slot_tokens[slot]
                if prompt_queue:
                    prompt_idx, prompt, prompt_image_url = prompt_queue.popleft()
                    lang_inputs, vision_outputs, num_chunks, num_pos_sections = _prepare_prompt(
                        prompt, prompt_image_url
                    )
                    ft, phys, mrope = _prefill_slot(lang_inputs, vision_outputs, num_chunks, slot)
                    _seed_slot(slot, prompt_idx, ft, phys, mrope)
                else:
                    ongoing[slot] = False
            else:
                slot_tokens[slot].append(tok)
                gen_count[slot] += 1
                last_token[slot] = tok
                phys_pos[slot] += 1
                mrope_pos[slot] += 1
    ft = perf_counter()

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
    parser.add_argument("--full-batch-size", type=int, default=DEFAULT_FULL_BATCH_SIZE, help="CB decode width (N)")
    parser.add_argument("--stages", type=int, default=STAGES, help="prefill pipeline depth (mdp_num_partitions)")
    parser.add_argument(
        "--prefill-num-devices", type=int, default=PREFILL_NUM_DEVICES, help="num devices for the prefill QPC"
    )
    parser.add_argument(
        "--decode-num-devices", type=int, default=DECODE_NUM_DEVICES, help="num devices for the decode QPC"
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
        full_batch_size=args.full_batch_size,
        skip_vision=args.skip_vision,
        image_urls=image_urls,
        stages=args.stages,
        prefill_num_devices=args.prefill_num_devices,
        decode_num_devices=args.decode_num_devices,
    )

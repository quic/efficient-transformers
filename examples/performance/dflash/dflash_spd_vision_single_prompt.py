# -----------------------------------------------------------------------------
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------

import argparse
import time
from typing import Optional

import numpy as np
import torch
import transformers
from rich.console import Console
from rich.markup import escape
from utils import format_prompt

from QEfficient.generation.cloud_infer import QAICInferenceSession

torch.manual_seed(42)
np.random.seed(42)

console = Console()


# ===== METRICS =====


class SpecDecodingMetrics:
    def __init__(self, block_size: int = 10):
        self.block_size = block_size
        self.total_prefill_time = 0.0
        self.tlm_decode_time = 0.0
        self.dlm_decode_time = 0.0
        self.total_accepted_tokens = 0
        self.total_rejected_tokens = 0
        self.total_generated_tokens = 0
        self.num_total_iters = 0
        self.acceptance_history = []
        self.generated_ids: list = []
        self.generated_sources: list = []

    def acceptance_rate(self) -> float:
        if self.num_total_iters == 0:
            return 0.0
        return self.total_generated_tokens / self.num_total_iters

    def dlm_tok_rate(self) -> float:
        if self.dlm_decode_time <= 0:
            return 0.0
        return (self.block_size * self.num_total_iters) / self.dlm_decode_time

    def tlm_tok_rate(self) -> float:
        if self.tlm_decode_time <= 0:
            return 0.0
        ar = self.acceptance_rate()
        num_tok_tlm = self.total_generated_tokens / (1 + ar) if (1 + ar) > 0 else 0.0
        return num_tok_tlm / self.tlm_decode_time

    def spd_tok_rate(self) -> float:
        total_decode_s = self.tlm_decode_time + self.dlm_decode_time
        if total_decode_s <= 0:
            return 0.0
        return self.total_generated_tokens / total_decode_s


# ===== INFERENCE =====


def run_spd_inference_single(
    prompt_text: str,
    tokenizer,
    dlm_session: QAICInferenceSession,
    tlm_session: QAICInferenceSession,
    mask_token_id: int,
    vocab_size: int,
    prompt_chunk_size: int,
    ctx_len: int = 4096,
    block_size: int = 16,
    max_iterations: int = 300,
    hidden_size: int = 4096,
    generation_len: int = 256,
    input_ids: Optional[np.ndarray] = None,
    mm_token_type_ids: Optional[np.ndarray] = None,
    vision_embeds: Optional[np.ndarray] = None,
) -> SpecDecodingMetrics:
    eos_token_ids = {tokenizer.eos_token_id} if tokenizer.eos_token_id is not None else set()

    prompt = [prompt_text]
    batch_size = 1
    metrics = SpecDecodingMetrics(block_size=block_size)

    # Tokenize (or reuse pre-built input_ids, e.g. a gemma processor prompt). Either
    # way we pad up to a multiple of prompt_chunk_size and derive position_ids so the
    # TLM and DLM see identical shapes.
    #
    # mm_full holds the padded multimodal token-type ids (1 at image-placeholder
    # positions) aligned to input_ids; it is sliced per prefill chunk so the TLM's
    # image gather picks up the real vision_embeds. None => text-only (zeros).
    mm_full = None
    if input_ids is not None:
        raw_ids = np.asarray(input_ids, dtype=np.int64).reshape(1, -1)
        unpadded_len = raw_ids.shape[1]
        num_chunks = -(unpadded_len // -prompt_chunk_size)  # ceil divide without float
        padded_len = num_chunks * prompt_chunk_size
        padded_ids = np.full((1, padded_len), tokenizer.pad_token_id, dtype=np.int64)
        padded_ids[:, :unpadded_len] = raw_ids
        attention_mask = np.zeros((1, padded_len), dtype=np.int64)
        attention_mask[:, :unpadded_len] = 1
        tlm_inputs = {
            "input_ids": padded_ids,
            "position_ids": np.where(attention_mask, np.arange(padded_len), -1),
        }
        if mm_token_type_ids is not None:
            raw_mm = np.asarray(mm_token_type_ids, dtype=np.int64).reshape(1, -1)
            mm_full = np.zeros((1, padded_len), dtype=np.int64)
            # mm_token_type_ids comes from the same processor call as input_ids, so it is
            # the same length; guard the slice in case a template diverges.
            copy_len = min(unpadded_len, raw_mm.shape[1])
            mm_full[:, :copy_len] = raw_mm[:, :copy_len]
    else:
        tlm_inputs = tokenizer(prompt, return_tensors="np", padding=True)
        padded_len = tlm_inputs["input_ids"].shape[1]
        num_chunks = -(padded_len // -prompt_chunk_size)  # ceil divide without float
        padded_len = num_chunks * prompt_chunk_size  # Convert to a multiple of padded_len
        tlm_inputs = tokenizer(prompt, return_tensors="np", padding="max_length", max_length=padded_len)
        tlm_inputs["position_ids"] = np.where(tlm_inputs.pop("attention_mask"), np.arange(padded_len), -1)

        tlm_inputs.pop("token_type_ids", None)
        tlm_inputs = {k: torch.from_numpy(v) for k, v in tlm_inputs.items()}
        tlm_inputs.pop("past_key_values", None)
        tlm_inputs = {k: v.detach().numpy() for k, v in tlm_inputs.items()}
    prompt_len = padded_len

    generated_ids = np.full((batch_size, ctx_len - prompt_len), tokenizer.pad_token_id)

    # The TLM's fused target-layer feature output is named "hidden_states" on some
    # builds and "target_hidden_states" on others (gemma4 exported from final-gemma).
    # Detect whichever the compiled TLM QPC actually exposes — same idea as
    # kv_offload_generate keying off the session's real output_names — so the SPD loop
    # binds and reads the correct buffer regardless of which graph was compiled.
    tlm_hidden_name = "target_hidden_states" if "target_hidden_states" in tlm_session.output_names else "hidden_states"

    # ── TLM graph-contract detection ─────────────────────────────────────────────
    # Two gemma4 TLM flavors exist:
    #  (a) text-only DFlash graph: inputs {input_ids, position_ids}; logits output is
    #      already-argmax'd token ids of shape [1, seq].
    #  (b) full multimodal graph (Gemma4DecoderWrapper exported with vision path): inputs
    #      additionally include vision_embeds / image_idx / mm_token_type_ids, and the
    #      logits output is FULL logits [1, seq, vocab] (host must argmax).
    # Detect both from the session so the same loop drives either.
    tlm_in = set(tlm_session.input_names)
    tlm_needs_vision = "vision_embeds" in tlm_in
    tlm_needs_image_idx = "image_idx" in tlm_in
    tlm_needs_mm_ids = "mm_token_type_ids" in tlm_in
    # logits rank: full logits = [b, seq, vocab] (3D); token ids = [b, seq] (2D).
    # NOTE: binding.dims is the flat default shape, so read the real rank from
    # allowed_shapes (where logits shows as e.g. [1, 128, 262144] for full logits).
    _logits_idx = tlm_session.binding_index_map["logits"]
    _logits_ranks = {len(sh[_logits_idx][1]) for sh in tlm_session.allowed_shapes}
    tlm_logits_is_full = max(_logits_ranks) == 3 if _logits_ranks else False
    console.print(
        f"TLM contract: vision_embeds={tlm_needs_vision} image_idx={tlm_needs_image_idx} "
        f"mm_token_type_ids={tlm_needs_mm_ids} full_logits={tlm_logits_is_full}"
    )

    # Static zero vision_embeds (text-only prompt => no real image), OR the real
    # vision_embeds produced by the vision QPC for an --image prompt. Sized from the
    # binding so it matches whatever the graph expects (e.g. [1, 256, 5376]).
    tlm_vision_embeds = None
    if tlm_needs_vision:
        ve_b = tlm_session.bindings[tlm_session.binding_index_map["vision_embeds"]]
        ve_shape = tuple(int(d) for d in ve_b.dims)
        ve_dtype = tlm_session.aic_to_np_dtype_mapping.get(ve_b.type, np.dtype(np.float32))
        if vision_embeds is not None:
            tlm_vision_embeds = np.asarray(vision_embeds, dtype=ve_dtype)
        else:
            tlm_vision_embeds = np.zeros(ve_shape, dtype=ve_dtype)

    # image_idx is threaded across prefill chunks: the TLM graph consumes image_idx and
    # emits image_idx_output (how many image tokens it has consumed so far). Start at 0
    # and advance it from each run's output so the image gather stays aligned when the
    # placeholder span crosses a chunk boundary.
    tlm_image_idx = np.array([[0]], dtype=np.int64)

    def tlm_run(ids, pos, mm=None):
        """Run the TLM with whatever inputs this graph requires.

        Returns a dict with normalized keys so downstream code is contract-agnostic:
          "logits"          -> token ids [1, seq]  (argmax'd on host if graph emits full logits)
          tlm_hidden_name   -> target hidden states [1, seq, hidden]
        `mm` (optional) is the mm_token_type_ids slice aligned to `ids`; when None a zero
        slice is used (text tokens / decode). image_idx is advanced from image_idx_output.
        """
        nonlocal tlm_image_idx
        feeds = {"input_ids": ids, "position_ids": pos}
        if tlm_needs_vision:
            feeds["vision_embeds"] = tlm_vision_embeds
        if tlm_needs_image_idx:
            feeds["image_idx"] = tlm_image_idx
        if tlm_needs_mm_ids:
            # mm_token_type_ids MUST have the same seq_len as input_ids, or no compiled
            # specialization matches (the QPC only allows mm shaped like input_ids). Callers
            # pass a slice aligned to `ids`, but chunk/sub-block boundaries can hand us a
            # slice that is shorter/longer than `ids` (e.g. multi-chunk prefill of a prompt
            # longer than prompt_chunk_size). Normalize here so the shapes always agree.
            seq = ids.shape[1]
            mm_ids = np.zeros((ids.shape[0], seq), dtype=ids.dtype)
            if mm is not None:
                mm = np.asarray(mm, dtype=ids.dtype).reshape(ids.shape[0], -1)
                copy = min(seq, mm.shape[1])
                mm_ids[:, :copy] = mm[:, :copy]
            feeds["mm_token_type_ids"] = mm_ids
        out = tlm_session.run(feeds)
        if tlm_needs_image_idx and "image_idx_output" in out:
            tlm_image_idx = out["image_idx_output"]
        logits = out["logits"]
        token_ids = logits.argmax(axis=-1).astype(np.int64) if tlm_logits_is_full else logits
        return {"logits": token_ids, tlm_hidden_name: out[tlm_hidden_name]}

    # Set output buffers. logits buffer shape depends on the contract.
    if tlm_logits_is_full:
        tlm_session.set_buffers({"logits": np.zeros((batch_size, prompt_chunk_size, vocab_size), dtype=np.float32)})
    else:
        tlm_session.set_buffers({"logits": np.zeros((batch_size, prompt_chunk_size), dtype=np.int32)})
    tlm_session.set_buffers(
        {tlm_hidden_name: np.zeros((batch_size, prompt_chunk_size, hidden_size), dtype=np.float32)}
    )
    dlm_session.set_buffers({"logits": np.zeros((batch_size, block_size, vocab_size), dtype=np.float32)})

    # vision_embeds is bound per-run inside tlm_run; image_idx/mm ids likewise.

    tlm_cache_index = np.array([0])
    dlm_cache_index = np.array([0])
    dlm_inputs = {}

    # ===== PREFILL =====
    prefill_start = time.time()
    num_sub_blocks = prompt_chunk_size // block_size
    remainder = prompt_chunk_size % block_size

    for pi in range(num_chunks - 1):
        chunk_inputs = {
            "input_ids": tlm_inputs["input_ids"][:, tlm_cache_index[0] : tlm_cache_index[0] + prompt_chunk_size],
            "position_ids": tlm_inputs["position_ids"][:, tlm_cache_index[0] : tlm_cache_index[0] + prompt_chunk_size],
        }
        tlm_prefill_outputs = tlm_run(
            chunk_inputs["input_ids"],
            chunk_inputs["position_ids"],
            mm=None
            if mm_full is None
            else mm_full[:, tlm_cache_index[0] : tlm_cache_index[0] + prompt_chunk_size],
        )
        ## Add support for when the prefill_seq_len is more than block_size
        for sub_i in range(num_sub_blocks):
            sub_start = sub_i * block_size
            dlm_inputs["target_hidden"] = tlm_prefill_outputs[tlm_hidden_name][:, sub_start : sub_start + block_size, :]
            dlm_inputs["position_ids_target"] = tlm_inputs["position_ids"][
                :, tlm_cache_index[0] + sub_start : tlm_cache_index[0] + sub_start + block_size
            ]
            dlm_inputs["position_ids"] = dlm_inputs["position_ids_target"] + block_size
            dlm_inputs["input_ids"] = np.full((1, block_size), mask_token_id, dtype=np.int64)
            dlm_session.run(dlm_inputs)

        ## Add support when prefill_seq_len is not a multiple of block_size
        if remainder > 0:
            sub_start = num_sub_blocks * block_size
            target_hidden_rem = np.zeros((1, block_size, hidden_size), dtype=np.float32)
            target_hidden_rem[:, :remainder, :] = tlm_prefill_outputs[tlm_hidden_name][:, sub_start:, :]
            pos_ids_target_rem = np.full((1, block_size), -1, dtype=tlm_inputs["position_ids"].dtype)
            pos_ids_target_rem[:, :remainder] = tlm_inputs["position_ids"][
                :, tlm_cache_index[0] + sub_start : tlm_cache_index[0] + sub_start + remainder
            ]
            dlm_inputs["target_hidden"] = target_hidden_rem
            dlm_inputs["position_ids_target"] = pos_ids_target_rem
            dlm_inputs["position_ids"] = pos_ids_target_rem + block_size
            dlm_inputs["input_ids"] = np.full((1, block_size), mask_token_id, dtype=np.int64)
            dlm_session.run(dlm_inputs)
        tlm_cache_index[0] += prompt_chunk_size
        dlm_cache_index[0] += prompt_chunk_size

    # Last prefill chunk
    chunk_inputs = {
        "input_ids": tlm_inputs["input_ids"][:, tlm_cache_index[0] : tlm_cache_index[0] + prompt_chunk_size],
        "position_ids": tlm_inputs["position_ids"][:, tlm_cache_index[0] : tlm_cache_index[0] + prompt_chunk_size],
    }
    tlm_last_prefill_outputs = tlm_run(
        chunk_inputs["input_ids"],
        chunk_inputs["position_ids"],
        mm=None
        if mm_full is None
        else mm_full[:, tlm_cache_index[0] : tlm_cache_index[0] + prompt_chunk_size],
    )
    last_prefill_pos_in_chunk = chunk_inputs["position_ids"].argmax()
    new_tlm_token = tlm_last_prefill_outputs["logits"][:, last_prefill_pos_in_chunk]

    ## Add support for when the prefill_seq_len is more than block_size
    last_sub = last_prefill_pos_in_chunk // block_size
    for sub_i in range(last_sub):
        sub_start = sub_i * block_size
        dlm_inputs["target_hidden"] = tlm_last_prefill_outputs[tlm_hidden_name][
            :, sub_start : sub_start + block_size, :
        ]
        dlm_inputs["position_ids_target"] = tlm_inputs["position_ids"][
            :, tlm_cache_index[0] + sub_start : tlm_cache_index[0] + sub_start + block_size
        ]
        dlm_inputs["position_ids"] = dlm_inputs["position_ids_target"] + block_size
        dlm_inputs["input_ids"] = np.full((1, block_size), mask_token_id, dtype=np.int64)
        dlm_session.run(dlm_inputs)

    input_ids = np.full((1, block_size), mask_token_id, dtype=np.int64)
    input_ids[:, 0] = new_tlm_token
    sub_start = last_sub * block_size

    ## Add support when prefill_seq_len is not a multiple of block_size
    if last_sub < num_sub_blocks:
        target_hidden = tlm_last_prefill_outputs[tlm_hidden_name][:, sub_start : sub_start + block_size, :]
        dlm_inputs["position_ids_target"] = tlm_inputs["position_ids"][
            :, tlm_cache_index[0] + sub_start : tlm_cache_index[0] + sub_start + block_size
        ]
    else:
        target_hidden = np.zeros((1, block_size, hidden_size), dtype=np.float32)
        target_hidden[:, :remainder, :] = tlm_last_prefill_outputs[tlm_hidden_name][:, sub_start:, :]
        pos_ids_target = np.full((1, block_size), -1, dtype=tlm_inputs["position_ids"].dtype)
        pos_ids_target[:, :remainder] = tlm_inputs["position_ids"][
            :, tlm_cache_index[0] + sub_start : tlm_cache_index[0] + sub_start + remainder
        ]
        dlm_inputs["position_ids_target"] = pos_ids_target

    dlm_inputs["position_ids"] = np.arange(
        tlm_cache_index[0] + last_prefill_pos_in_chunk + 1,
        tlm_cache_index[0] + last_prefill_pos_in_chunk + 1 + block_size,
    ).reshape(1, -1)
    dlm_inputs["input_ids"] = input_ids
    dlm_inputs["target_hidden"] = target_hidden
    dlm_outputs = dlm_session.run(dlm_inputs)

    metrics.total_prefill_time += time.time() - prefill_start
    dlm_candidates = dlm_outputs["logits"].argmax(axis=-1)

    # ===== DECODE =====
    spd_counter_idx = tlm_cache_index[0] + last_prefill_pos_in_chunk
    gen_idx = 0
    iteration_count = 0
    continue_generation = True

    # Decode output buffers (block-sized). logits shape follows the contract.
    if tlm_logits_is_full:
        tlm_session.set_buffers({"logits": np.zeros((batch_size, block_size, vocab_size), dtype=np.float32)})
    else:
        tlm_session.set_buffers({"logits": np.zeros((batch_size, block_size), dtype=np.int32)})
    tlm_session.set_buffers({tlm_hidden_name: np.zeros((batch_size, block_size, hidden_size), dtype=np.float32)})

    while gen_idx < generation_len and iteration_count < max_iterations and continue_generation:
        iteration_count += 1
        dlm_candidates[:, 0] = new_tlm_token

        tlm_decode_start = time.time()
        tlm_decode_outputs = tlm_run(dlm_candidates, dlm_inputs["position_ids"])
        metrics.tlm_decode_time += time.time() - tlm_decode_start

        tlm_logits = tlm_decode_outputs["logits"]
        target_hidden = tlm_decode_outputs[tlm_hidden_name]

        accepted_length = 0
        rejected_flag = False

        for spec_idx in range(block_size - 1):
            tlm_token = tlm_logits[:, spec_idx]
            dlm_token = dlm_candidates[:, spec_idx + 1]
            if tlm_token == dlm_token:
                accepted_length += 1
                metrics.total_accepted_tokens += 1
                if gen_idx < len(generated_ids[0]):
                    generated_ids[0, gen_idx] = dlm_token[0]
                    gen_idx += 1
                    metrics.generated_ids.append(int(dlm_token[0]))
                    metrics.generated_sources.append("dlm")
            else:
                metrics.total_rejected_tokens += block_size - spec_idx - 1
                rejected_flag = True
                new_tlm_token = tlm_token
                if gen_idx < len(generated_ids[0]):
                    generated_ids[0, gen_idx] = tlm_token[0]
                    gen_idx += 1
                    metrics.generated_ids.append(int(tlm_token[0]))
                    metrics.generated_sources.append("tlm")
                break

        metrics.acceptance_history.append(accepted_length)
        metrics.total_generated_tokens += accepted_length + 1

        if not rejected_flag:
            new_tlm_token = tlm_logits[:, block_size - 1]
            if gen_idx < len(generated_ids[0]):
                generated_ids[0, gen_idx] = new_tlm_token[0]
                gen_idx += 1
                metrics.generated_ids.append(int(new_tlm_token[0]))
                metrics.generated_sources.append("tlm")

        dlm_candidate_ids = list(dlm_candidates[0, 1 : accepted_length + 1])
        this_iter_gen_ids = dlm_candidate_ids + [new_tlm_token[0]]
        for tok_id in this_iter_gen_ids:
            if tok_id in eos_token_ids:
                continue_generation = False
                break

        if not continue_generation:
            break

        dlm_decode_start = time.time()
        dlm_inputs["position_ids_target"] = np.arange(spd_counter_idx + 1, spd_counter_idx + block_size + 1).reshape(
            1, -1
        )
        spd_counter_idx += accepted_length + 1
        dlm_inputs["position_ids_target"][:, accepted_length + 1 :] = -1
        dlm_inputs["position_ids"] = np.arange(spd_counter_idx + 1, spd_counter_idx + block_size + 1).reshape(1, -1)
        input_ids[:, 0] = new_tlm_token
        dlm_inputs["input_ids"] = input_ids
        dlm_inputs["target_hidden"] = target_hidden
        dlm_outputs = dlm_session.run(dlm_inputs)
        metrics.dlm_decode_time += time.time() - dlm_decode_start

        dlm_candidates = dlm_outputs["logits"].argmax(axis=-1)

    metrics.num_total_iters = iteration_count
    return metrics


# ===== GEMMA IMAGE/TEXT PROMPT (multimodal TLM) =====
# Mirrors examples/image_text_to_text/models/gemma_vision/gemma4/gemma4_example.py: the
# prompt is built with the gemma processor's chat template.
#
# The gemma4 vision build (SKIP_VISION=False, kv_offload=True) produces TWO QPCs:
#   * a vision-encoder QPC:  pixel_values, image_position_ids -> vision_embeds
#   * a language (DFlash TLM) QPC: input_ids, vision_embeds, position_ids, image_idx,
#     mm_token_type_ids -> logits, image_idx_output, target_hidden_states
# For --image we run the vision QPC once to produce real vision_embeds, then feed those
# (plus mm_token_type_ids marking the image-placeholder span) through the SPD TLM so the
# image tokens attend to real image features. Text-only prompts bind zero vision_embeds.
SYSTEM_PROMPT = "You are a helpful assistant."
IMAGE_PROMPT = "Can you Describe this image in detail?"
IMAGE_URL = "https://wallup.net/wp-content/uploads/2017/03/28/351036-San_Francisco-USA-bridge-sunset-Golden_Gate_Bridge-lights.jpg"


def build_gemma_inputs(processor, tokenizer, user_prompt, image_url=None, system_prompt=SYSTEM_PROMPT, pil_image=None):
    """Build processor inputs for a gemma text or image+text prompt.

    Returns (input_ids [1, L] int64, mm_token_type_ids [1, L] int64 or None,
    processor_inputs dict). With image_url=None and pil_image=None this is the text-only
    path (no pixel_values / image tokens), matching gemma4_example.py at input image = false.

    Image source precedence: `pil_image` (an in-memory PIL.Image, e.g. a HF dataset sample)
    is used via the chat-template "image" key; otherwise `image_url` via the "url" key.
    """
    chat_template = getattr(processor, "chat_template", None) or getattr(tokenizer, "chat_template", None)
    if pil_image is not None:
        messages = [
            {"role": "user", "content": [{"type": "image", "image": pil_image}, {"type": "text", "text": user_prompt}]}
        ]
    elif image_url is not None:
        messages = [
            {"role": "user", "content": [{"type": "image", "url": image_url}, {"type": "text", "text": user_prompt}]}
        ]
    else:
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": [{"type": "text", "text": system_prompt}]})
        messages.append({"role": "user", "content": [{"type": "text", "text": user_prompt}]})
    inputs = processor.apply_chat_template(
        messages,
        chat_template=chat_template,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
    )
    input_ids = inputs["input_ids"].detach().cpu().numpy()
    mm_ids = None
    if "mm_token_type_ids" in inputs:
        mm_ids = inputs["mm_token_type_ids"].detach().cpu().numpy().astype(np.int64)
    return input_ids, mm_ids, inputs


def build_gemma_input_ids(processor, tokenizer, user_prompt, image_url=None, system_prompt=SYSTEM_PROMPT):
    """Back-compat shim: return only input_ids (text-only callers)."""
    input_ids, _, _ = build_gemma_inputs(processor, tokenizer, user_prompt, image_url, system_prompt)
    return input_ids


def run_gemma_vision_encoder(vision_session, processor_inputs):
    """Run the gemma4 vision-encoder QPC to produce vision_embeds.

    Feeds pixel_values (fp16) + image_position_ids exactly as kv_offload_generate does
    (see QEfficient/transformers/models/modeling_auto.py and embedding_handler.py).
    Returns the vision_embeds numpy array, or None if there are no pixel_values.
    """
    if "pixel_values" not in processor_inputs:
        return None

    vision_feeds = {}
    for key in ("pixel_values", "image_position_ids"):
        if key in processor_inputs:
            vision_feeds[key] = np.asarray(processor_inputs[key])
    # pixel_values feeds the vision encoder in fp16 (matches the compiled custom-io dtype).
    if "pixel_values" in vision_feeds:
        vision_feeds["pixel_values"] = vision_feeds["pixel_values"].astype("float16")

    vision_session.skip_buffers(
        set(
            [
                x
                for x in vision_session.input_names + vision_session.output_names
                if x.startswith("past_") or x.endswith("_RetainedState")
            ]
        )
    )
    vision_outputs = vision_session.run(vision_feeds)
    return vision_outputs["vision_embeds"]


# ===== ARGUMENT PARSING =====


def parse_args():
    parser = argparse.ArgumentParser(description="SPD single-prompt inference (text or --image gemma prompt)")
    parser.add_argument("--prompt", help="Input prompt text. Required unless --image is given.")
    parser.add_argument("--tlm_qpc", required=True)
    parser.add_argument("--dlm_qpc", required=True)
    parser.add_argument("--tlm_model_name", required=True)
    parser.add_argument("--dlm_model_name", required=True)
    parser.add_argument("--iteration", type=int, default=300)
    parser.add_argument("--ctx_len", type=int, default=4096)
    parser.add_argument("--generation_len", type=int, default=256)
    parser.add_argument("--tlm_devices", nargs="+", type=int, required=True)
    parser.add_argument("--dlm_devices", nargs="+", type=int, required=True)
    parser.add_argument("--hf_token", default=None)
    parser.add_argument(
        "--category",
        default="",
        help="Prompt category for formatting (math, coding, reasoning, …). Defaults to the general reasoning format.",
    )
    parser.add_argument(
        "--format_prompt",
        action="store_true",
        help="If set, wrap the prompt with the category-specific template from utils.format_prompt. "
        "Off by default — the prompt is used verbatim.",
    )
    # ---- Gemma (multimodal) single-prompt modes ----
    parser.add_argument(
        "--image",
        action="store_true",
        help="Run a single image+text prompt (gemma processor) through SPD instead of a text prompt.",
    )
    parser.add_argument("--image_url", default=IMAGE_URL, help="Image URL for --image mode.")
    parser.add_argument("--image_prompt", default=IMAGE_PROMPT, help="Prompt text for --image mode.")
    parser.add_argument(
        "--vision_qpc",
        default=None,
        help="Path to the gemma4 vision-encoder QPC (the second QPC produced by "
        "gemma4_example.py). Required for --image so pixel_values -> vision_embeds runs.",
    )
    parser.add_argument(
        "--vision_devices",
        nargs="+",
        type=int,
        default=None,
        help="Device IDs for the vision QPC (defaults to --tlm_devices).",
    )
    return parser.parse_args()


# ===== MAIN =====


def main():
    args = parse_args()
    if not args.image and not args.prompt:
        raise SystemExit("Provide --prompt <text>, or --image for a single gemma image prompt.")

    console.print("[bold blue]Loading tokenizer and config...[/bold blue]")
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        args.tlm_model_name, token=args.hf_token, trust_remote_code=True
    )
    config = transformers.AutoConfig.from_pretrained(args.dlm_model_name, token=args.hf_token, trust_remote_code=True)
    vocab_size = config.vocab_size
    hidden_size = config.hidden_size
    block_size = config.block_size
    dflash_cfg = getattr(config, "dflash_config", None) or config.to_dict().get("dflash_config", {})
    mask_token_id = dflash_cfg["mask_token_id"] if isinstance(dflash_cfg, dict) else dflash_cfg.mask_token_id

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    console.print("[bold blue]Loading QAIC inference sessions...[/bold blue]")
    dlm_session = QAICInferenceSession(args.dlm_qpc, args.dlm_devices)
    tlm_session = QAICInferenceSession(args.tlm_qpc, args.tlm_devices)
    dlm_session.skip_buffers(
        set([x for x in dlm_session.input_names + dlm_session.output_names if x.startswith("past_")])
    )
    tlm_session.skip_buffers(
        set([x for x in tlm_session.input_names + tlm_session.output_names if x.startswith("past_")])
    )

    prompt_chunk_size = max(
        [x[tlm_session.binding_index_map["input_ids"]][1][1] for x in tlm_session.allowed_shapes]
        + [tlm_session.bindings[tlm_session.binding_index_map["input_ids"]].dims[1]]
    )
    console.print(f"prompt_chunk_size = {prompt_chunk_size}")

    # SPD feeds the TLM a full block of `block_size` tokens per decode step, so the TLM
    # QPC MUST expose a decode specialization with input_ids seq_len == block_size. A QPC
    # compiled WITHOUT dflash_block_size only has seq_len=1 decode → SPD would silently run
    # wrong. Verify here and fail loudly with a clear message.
    tlm_seq_lens = sorted({x[tlm_session.binding_index_map["input_ids"]][1][1] for x in tlm_session.allowed_shapes})
    console.print(f"TLM input_ids seq_lens (allowed shapes): {tlm_seq_lens}")
    if block_size not in tlm_seq_lens:
        raise SystemExit(
            f"TLM QPC has no decode specialization with seq_len={block_size} "
            f"(found {tlm_seq_lens}). Recompile the TLM with dflash_block_size={block_size} "
            "(e.g. gemma4_example.py with USE_DFLASH_BLOCK_SIZE=True), then point --tlm_qpc at it."
        )

    # For gemma4 the prompt must be built with the processor chat template (gemma4 needs
    # the processor template, not the bare tokenizer). Load it best-effort; if it fails
    # (e.g. a non-gemma TLM), fall back to the tokenizer chat template.
    processor = None
    try:
        from transformers import AutoProcessor

        processor = AutoProcessor.from_pretrained(args.tlm_model_name, token=args.hf_token, trust_remote_code=True)
    except Exception as e:
        console.print(f"[yellow]⚠ No processor for {args.tlm_model_name} ({e}); using tokenizer template.[/yellow]")

    use_image = bool(args.image)
    image_url = args.image_url if use_image else None
    mm_token_type_ids = None
    vision_embeds = None
    vision_encode_time_s = None
    if use_image:
        if processor is None:
            raise SystemExit("--image requires a gemma processor, which failed to load. See warning above.")
        user_prompt = args.image_prompt
        input_ids, mm_token_type_ids, processor_inputs = build_gemma_inputs(
            processor, tokenizer, user_prompt, image_url=image_url
        )
        prompt_text = ""
        console.print(f"[cyan]Image:[/cyan] {args.image_url}")

        # Run the gemma4 vision-encoder QPC (pixel_values -> vision_embeds) and feed the
        # real embeds through SPD. Without it the image placeholders would attend to zero
        # vision features (text-only fallback) and the TLM's gather would raise.
        if not args.vision_qpc:
            raise SystemExit(
                "--image needs the gemma4 vision-encoder QPC. Pass --vision_qpc <path> "
                "(the second QPC produced by gemma4_example.py with SKIP_VISION=False)."
            )
        vision_devices = args.vision_devices if args.vision_devices is not None else args.tlm_devices
        console.print(f"[bold blue]Loading vision QPC session on devices {vision_devices}...[/bold blue]")
        vision_session = QAICInferenceSession(args.vision_qpc, vision_devices)
        # Time ONLY the vision encode (the .run() inside run_gemma_vision_encoder), not the
        # session load above — this is the standalone vision-encoder latency, which is NOT
        # part of the LM "Prefill time" reported below (that covers only TLM/DLM prefill).
        _vision_start = time.perf_counter()
        vision_embeds = run_gemma_vision_encoder(vision_session, processor_inputs)
        vision_encode_time_s = time.perf_counter() - _vision_start
        if vision_embeds is None:
            raise SystemExit("Vision QPC produced no vision_embeds (no pixel_values in processor output).")
        if "vision_embeds" not in tlm_session.input_names:
            raise SystemExit(
                "--image requires a TLM QPC compiled WITH the vision path (vision_embeds input). "
                "The provided --tlm_qpc is text-only. Recompile gemma4_example.py with SKIP_VISION=False."
            )
        console.print(
            f"[green]✓ vision_embeds {tuple(vision_embeds.shape)} from vision QPC[/green]  "
            f"(mm_token_type_ids sum={0 if mm_token_type_ids is None else int(mm_token_type_ids.sum())}, "
            f"vision_encode={vision_encode_time_s:.3f}s)"
        )
        console.print(f"[cyan]Prompt:[/cyan] {user_prompt}  (prompt_len={input_ids.shape[1]})")
    else:
        user_content = format_prompt(args.prompt, args.category) if args.format_prompt else args.prompt
        if processor is not None:
            # gemma4: build input_ids via the processor chat template (text-only path).
            input_ids = build_gemma_input_ids(processor, tokenizer, user_content, image_url=None)
            prompt_text = ""
            console.print("[cyan]Mode:[/cyan] gemma text-only (input image = false)")
            console.print(f"[cyan]Input:[/cyan] {args.prompt[:120].strip()}  (prompt_len={input_ids.shape[1]})")
        else:
            input_ids = None
            messages = [{"role": "user", "content": user_content}]
            prompt_text = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
            )
            console.print(f"[cyan]Input:[/cyan] {args.prompt[:120].strip()}")

    metrics = run_spd_inference_single(
        prompt_text=prompt_text,
        tokenizer=tokenizer,
        dlm_session=dlm_session,
        tlm_session=tlm_session,
        vocab_size=vocab_size,
        prompt_chunk_size=prompt_chunk_size,
        ctx_len=args.ctx_len,
        block_size=block_size,
        max_iterations=args.iteration,
        hidden_size=hidden_size,
        generation_len=args.generation_len,
        mask_token_id=mask_token_id,
        input_ids=input_ids,
        mm_token_type_ids=mm_token_type_ids,
        vision_embeds=vision_embeds,
    )

    output_parts = ["Output: "]
    for tok_id, source in zip(metrics.generated_ids, metrics.generated_sources):
        text = escape(tokenizer.decode([tok_id], skip_special_tokens=True))
        if source == "dlm":
            output_parts.append(f"[blue]{text}[/blue]")
        else:
            output_parts.append(f"[white]{text}[/white]")
    console.print("".join(output_parts))

    ar = metrics.acceptance_rate()
    dlm_tps = metrics.dlm_tok_rate()
    tlm_tps = metrics.tlm_tok_rate()
    spd_tps = metrics.spd_tok_rate()

    w = 46
    print("\n" + "=" * w)
    print("  SPD Inference — Metrics")
    print("=" * w)
    print(f"  {'Acceptance Rate (tok/iter)':<30} {ar:>6.2f}")
    print(f"  {'DLM Throughput  (tok/s)':<30} {dlm_tps:>6.1f}")
    print(f"  {'TLM Throughput  (tok/s)':<30} {tlm_tps:>6.1f}")
    print(f"  {'SPD Decode Speed (tok/s)':<30} {spd_tps:>6.1f}")
    print(f"  {'Generated tokens':<30} {metrics.total_generated_tokens:>6}")
    print(f"  {'Iterations':<30} {metrics.num_total_iters:>6}")
    print(f"  {'Prefill time (s)':<30} {metrics.total_prefill_time:>6.3f}")
    if vision_encode_time_s is not None:
        # Standalone vision-encoder latency (pixel_values -> vision_embeds). This is a
        # one-shot per-image cost and is NOT included in the LM "Prefill time" above.
        print(f"  {'Vision Encode (s)':<30} {vision_encode_time_s:>6.3f}")
    print("=" * w + "\n")


if __name__ == "__main__":
    main()

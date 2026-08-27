# -----------------------------------------------------------------------------
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------

"""
DFlash speculative-decoding (SPD) generation: text-only and VLM (gemma4 / qwen3-vl).

Owns the on-device SPD loops (TLM verify + DLM draft, block-wise accept/reject) and the
two top-level entry points, ``run_text_inference`` and ``run_vision_inference``, that load
tokenizer/config/processor, open the QAIC sessions, and drive one prompt end to end. These
are called directly (in-process) by the ``examples/performance/dflash/basic_inference_*.py``
front-end scripts.
"""

import time

import numpy as np
import torch
import transformers
from qwen_vl_utils import process_vision_info
from rich.console import Console

from QEfficient.generation.cloud_infer import QAICInferenceSession
from QEfficient.transformers.models.qwen3_vl.modeling_qwen3_vl import QEffQwen3VLForConditionalGeneration

console = Console()

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


# ===== TEXT-ONLY SPD INFERENCE =====


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
) -> SpecDecodingMetrics:
    eos_token_ids = {tokenizer.eos_token_id} if tokenizer.eos_token_id is not None else set()

    prompt = [prompt_text]
    batch_size = 1
    metrics = SpecDecodingMetrics(block_size=block_size)

    # Tokenize
    tlm_inputs = tokenizer(prompt, return_tensors="np", padding=True)
    padded_len = tlm_inputs["input_ids"].shape[1]
    num_chunks = -(padded_len // -prompt_chunk_size)
    padded_len = num_chunks * prompt_chunk_size
    tlm_inputs = tokenizer(prompt, return_tensors="np", padding="max_length", max_length=padded_len)
    tlm_inputs["position_ids"] = np.where(tlm_inputs.pop("attention_mask"), np.arange(padded_len), -1)

    tlm_inputs.pop("token_type_ids", None)
    tlm_inputs = {k: torch.from_numpy(v) for k, v in tlm_inputs.items()}
    tlm_inputs.pop("past_key_values", None)
    tlm_inputs = {k: v.detach().numpy() for k, v in tlm_inputs.items()}

    generated_ids = np.full((batch_size, ctx_len - padded_len), tokenizer.pad_token_id)

    # Set output buffers
    tlm_session.set_buffers({"logits": np.zeros((batch_size, prompt_chunk_size), dtype=np.int32)})
    tlm_session.set_buffers({"hidden_states": np.zeros((batch_size, prompt_chunk_size, hidden_size), dtype=np.float32)})
    dlm_session.set_buffers({"logits": np.zeros((batch_size, block_size, vocab_size), dtype=np.float32)})

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
        tlm_prefill_outputs = tlm_session.run(chunk_inputs)
        for sub_i in range(num_sub_blocks):
            sub_start = sub_i * block_size
            dlm_inputs["target_hidden"] = tlm_prefill_outputs["hidden_states"][:, sub_start : sub_start + block_size, :]
            dlm_inputs["position_ids_target"] = tlm_inputs["position_ids"][
                :, tlm_cache_index[0] + sub_start : tlm_cache_index[0] + sub_start + block_size
            ]
            dlm_inputs["position_ids"] = dlm_inputs["position_ids_target"] + block_size
            dlm_inputs["input_ids"] = np.full((1, block_size), mask_token_id, dtype=np.int64)
            dlm_session.run(dlm_inputs)
        if remainder > 0:
            sub_start = num_sub_blocks * block_size
            target_hidden_rem = np.zeros((1, block_size, hidden_size), dtype=np.float32)
            target_hidden_rem[:, :remainder, :] = tlm_prefill_outputs["hidden_states"][:, sub_start:, :]
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
    tlm_last_prefill_outputs = tlm_session.run(chunk_inputs)
    last_prefill_pos_in_chunk = chunk_inputs["position_ids"].argmax()
    new_tlm_token = tlm_last_prefill_outputs["logits"][:, last_prefill_pos_in_chunk]

    last_sub = last_prefill_pos_in_chunk // block_size
    for sub_i in range(last_sub):
        sub_start = sub_i * block_size
        dlm_inputs["target_hidden"] = tlm_last_prefill_outputs["hidden_states"][
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
    if last_sub < num_sub_blocks:
        target_hidden = tlm_last_prefill_outputs["hidden_states"][:, sub_start : sub_start + block_size, :]
        dlm_inputs["position_ids_target"] = tlm_inputs["position_ids"][
            :, tlm_cache_index[0] + sub_start : tlm_cache_index[0] + sub_start + block_size
        ]
    else:
        target_hidden = np.zeros((1, block_size, hidden_size), dtype=np.float32)
        target_hidden[:, :remainder, :] = tlm_last_prefill_outputs["hidden_states"][:, sub_start:, :]
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

    tlm_session.set_buffers({"logits": np.zeros((batch_size, block_size), dtype=np.int32)})
    tlm_session.set_buffers({"hidden_states": np.zeros((batch_size, block_size, hidden_size), dtype=np.float32)})

    while gen_idx < generation_len and iteration_count < max_iterations and continue_generation:
        iteration_count += 1
        dlm_candidates[:, 0] = new_tlm_token

        tlm_decode_start = time.time()
        tlm_decode_outputs = tlm_session.run(
            {
                "input_ids": dlm_candidates,
                "position_ids": dlm_inputs["position_ids"],
            }
        )
        metrics.tlm_decode_time += time.time() - tlm_decode_start

        tlm_logits = tlm_decode_outputs["logits"]
        target_hidden = tlm_decode_outputs["hidden_states"]

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


def run_text_inference(
    prompt: str,
    tlm_qpc: str,
    dlm_qpc: str,
    tlm_model_name: str,
    dlm_model_name: str,
    tlm_devices: list,
    dlm_devices: list,
    iteration: int = 300,
    ctx_len: int = 4096,
    generation_len: int = 256,
    hf_token: str | None = None,
):
    """Load tokenizer/config/sessions and run one text SPD generation end to end.

    ``prompt`` is used verbatim (apply any category/format-prompt templating before
    calling this, e.g. via ``utils.format_prompt`` in the caller).

    Returns ``(metrics, tokenizer)`` — the caller decodes ``metrics.generated_ids`` with
    the returned tokenizer to render the generated text.
    """
    console.print("[bold blue]Loading tokenizer and config...[/bold blue]")
    tokenizer = transformers.AutoTokenizer.from_pretrained(tlm_model_name, token=hf_token, trust_remote_code=True)
    config = transformers.AutoConfig.from_pretrained(dlm_model_name, token=hf_token, trust_remote_code=True)
    vocab_size = config.vocab_size
    hidden_size = config.hidden_size
    block_size = config.block_size
    dflash_cfg = getattr(config, "dflash_config", None) or config.to_dict().get("dflash_config", {})
    mask_token_id = dflash_cfg["mask_token_id"] if isinstance(dflash_cfg, dict) else dflash_cfg.mask_token_id

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    console.print("[bold blue]Loading QAIC inference sessions...[/bold blue]")
    dlm_session = QAICInferenceSession(dlm_qpc, dlm_devices)
    tlm_session = QAICInferenceSession(tlm_qpc, tlm_devices)
    dlm_session.skip_buffers(
        {x for x in dlm_session.input_names + dlm_session.output_names if x.startswith("past_")}
    )
    tlm_session.skip_buffers(
        {x for x in tlm_session.input_names + tlm_session.output_names if x.startswith("past_")}
    )

    prompt_chunk_size = max(
        [x[tlm_session.binding_index_map["input_ids"]][1][1] for x in tlm_session.allowed_shapes]
        + [tlm_session.bindings[tlm_session.binding_index_map["input_ids"]].dims[1]]
    )
    console.print(f"prompt_chunk_size = {prompt_chunk_size}")

    messages = [{"role": "user", "content": prompt}]
    prompt_text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
    )
    console.print(f"[cyan]Input:[/cyan] {prompt[:120].strip()}")

    metrics = run_spd_inference_single(
        prompt_text=prompt_text,
        tokenizer=tokenizer,
        dlm_session=dlm_session,
        tlm_session=tlm_session,
        vocab_size=vocab_size,
        prompt_chunk_size=prompt_chunk_size,
        ctx_len=ctx_len,
        block_size=block_size,
        max_iterations=iteration,
        hidden_size=hidden_size,
        generation_len=generation_len,
        mask_token_id=mask_token_id,
    )
    return metrics, tokenizer


# ===== VISION SPD INFERENCE — GEMMA4 =====


def run_spd_inference_gemma4(
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
    input_ids: np.ndarray | None = None,
    mm_token_type_ids: np.ndarray | None = None,
    vision_embeds: np.ndarray | None = None,
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


def build_inputs_gemma4(processor, tokenizer, user_prompt, image_url=None, system_prompt=SYSTEM_PROMPT, pil_image=None):
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


def build_input_ids_gemma4(processor, tokenizer, user_prompt, image_url=None, system_prompt=SYSTEM_PROMPT):
    """Back-compat shim: return only input_ids (text-only callers)."""
    input_ids, _, _ = build_inputs_gemma4(processor, tokenizer, user_prompt, image_url, system_prompt)
    return input_ids


def run_vision_encoder_gemma4(vision_session, processor_inputs):
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
        {
            x
            for x in vision_session.input_names + vision_session.output_names
            if x.startswith("past_") or x.endswith("_RetainedState")
        }
    )
    vision_outputs = vision_session.run(vision_feeds)
    return vision_outputs["vision_embeds"]


def _run_vision_inference_gemma4(
    tokenizer,
    processor,
    dlm_session,
    tlm_session,
    vocab_size,
    hidden_size,
    block_size,
    mask_token_id,
    prompt_chunk_size,
    prompt,
    image,
    image_url,
    image_prompt,
    tlm_devices,
    vision_qpc,
    vision_devices,
    ctx_len,
    iteration,
    generation_len,
):
    use_image = bool(image)
    resolved_image_url = (image_url or IMAGE_URL) if use_image else None
    mm_token_type_ids = None
    vision_embeds = None
    vision_encode_time_s = None
    if use_image:
        if processor is None:
            raise SystemExit("--image requires a gemma processor, which failed to load. See warning above.")
        user_prompt = image_prompt or IMAGE_PROMPT
        input_ids, mm_token_type_ids, processor_inputs = build_inputs_gemma4(
            processor, tokenizer, user_prompt, image_url=resolved_image_url
        )
        prompt_text = ""
        console.print(f"[cyan]Image:[/cyan] {resolved_image_url}")

        # Run the gemma4 vision-encoder QPC (pixel_values -> vision_embeds) and feed the
        # real embeds through SPD. Without it the image placeholders would attend to zero
        # vision features (text-only fallback) and the TLM's gather would raise.
        if not vision_qpc:
            raise SystemExit(
                "--image needs the gemma4 vision-encoder QPC. Pass --vision_qpc <path> "
                "(the second QPC produced by gemma4_example.py with SKIP_VISION=False)."
            )
        vd = vision_devices if vision_devices is not None else tlm_devices
        console.print(f"[bold blue]Loading vision QPC session on devices {vd}...[/bold blue]")
        vision_session = QAICInferenceSession(vision_qpc, vd)
        # Time ONLY the vision encode (the .run() inside run_vision_encoder_gemma4), not the
        # session load above — this is the standalone vision-encoder latency, which is NOT
        # part of the LM "Prefill time" reported below (that covers only TLM/DLM prefill).
        _vision_start = time.perf_counter()
        vision_embeds = run_vision_encoder_gemma4(vision_session, processor_inputs)
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
        user_content = prompt
        if processor is not None:
            # gemma4: build input_ids via the processor chat template (text-only path).
            input_ids = build_input_ids_gemma4(processor, tokenizer, user_content, image_url=None)
            prompt_text = ""
            console.print("[cyan]Mode:[/cyan] gemma text-only (input image = false)")
            console.print(f"[cyan]Input:[/cyan] {prompt[:120].strip()}  (prompt_len={input_ids.shape[1]})")
        else:
            input_ids = None
            messages = [{"role": "user", "content": user_content}]
            prompt_text = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
            )
            console.print(f"[cyan]Input:[/cyan] {prompt[:120].strip()}")

    metrics = run_spd_inference_gemma4(
        prompt_text=prompt_text,
        tokenizer=tokenizer,
        dlm_session=dlm_session,
        tlm_session=tlm_session,
        vocab_size=vocab_size,
        prompt_chunk_size=prompt_chunk_size,
        ctx_len=ctx_len,
        block_size=block_size,
        max_iterations=iteration,
        hidden_size=hidden_size,
        generation_len=generation_len,
        mask_token_id=mask_token_id,
        input_ids=input_ids,
        mm_token_type_ids=mm_token_type_ids,
        vision_embeds=vision_embeds,
    )
    return metrics, {"vision_encode_time_s": vision_encode_time_s}


# ===== VISION SPD INFERENCE — QWEN3-VL =====
# qwen3-vl's compiled DFlash TLM QPC always requires vision_embeds/deepstack_features/
# image_idx as inputs (unlike gemma4, which has separate text-only and multimodal TLM
# graphs) — kept as its own inference function since the M-RoPE position-id handling
# and vision I/O contract are different enough from the gemma path to not share a loop.
_rope_model_cache_qwen3_vl = {}


def _get_rope_model_qwen3_vl(config):
    """QEffQwen3VLForConditionalGeneration, built once per config on the meta device.
    get_rope_index (invoked via prepare_inputs_for_generation below) is a pure function
    of input_ids/config/grid_thw and never touches the weights, so a meta-device model
    is sufficient and avoids paying full-size construction cost on every call.
    """
    key = id(config)
    if key not in _rope_model_cache_qwen3_vl:
        with torch.device("meta"):
            _rope_model_cache_qwen3_vl[key] = QEffQwen3VLForConditionalGeneration._from_config(config)
    return _rope_model_cache_qwen3_vl[key]


def compute_position_ids_qwen3_vl(input_ids, attention_mask, image_grid_thw, config):
    """Delegates to QEffQwen3VLForConditionalGeneration.prepare_inputs_for_generation
    for M-RoPE construction. With image_grid_thw=None (text-only) this reduces to a
    plain arange position broadcast to all 4 rows with rope_deltas=0.
    Returns (position_ids [4, batch, seq_len] torch tensor, rope_deltas [batch, 1] torch tensor).
    """
    rope_model = _get_rope_model_qwen3_vl(config)
    batch_size, seq_len = input_ids.shape
    inputs = rope_model.prepare_inputs_for_generation(
        inputs={"input_ids": input_ids, "attention_mask": attention_mask, "image_grid_thw": image_grid_thw},
        prefill_seq_len=seq_len,
        batch_size=batch_size,
    )
    return inputs["position_ids"], rope_model.model.rope_deltas


def build_decode_position_ids_qwen3_vl(flat_row: np.ndarray, rope_delta: int) -> np.ndarray:
    """flat_row: (batch, seq_len) plain causal positions for a decode block.
    Returns the (4, batch, seq_len) M-RoPE tensor to feed the lang QPC during decode:
    row 0 stays the plain causal position (used for KV-cache indexing), rows 1-3 all
    carry the same flat_row + rope_delta (post-image text has no spatial variation left,
    per qwen3-vl's M-RoPE convention).
    """
    mrope_row = flat_row + rope_delta
    return np.concatenate([flat_row[None], np.repeat(mrope_row[None], 3, axis=0)], axis=0)


def build_inputs_qwen3_vl(processor, user_prompt, image=None):
    """Build processor inputs for a qwen3-vl text or image+text prompt.

    Returns (processor_inputs dict). With image=None this is the text-only path (no
    pixel_values / image_grid_thw keys).
    """
    content = []
    if image is not None:
        content.append({"type": "image", "image": image})
    content.append({"type": "text", "text": user_prompt})
    messages = [{"role": "user", "content": content}]
    chat_text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages) if image is not None else (None, None)
    proc_inputs = processor(
        text=[chat_text], images=image_inputs, videos=video_inputs, padding=True, return_tensors="pt"
    )
    proc_inputs.pop("token_type_ids", None)
    return proc_inputs


def run_vision_encoder_qwen3_vl(vision_session, pixel_values, image_grid_thw):
    """Run the qwen3-vl vision-encoder QPC to produce (vision_embeds, deepstack_features)."""
    vision_outputs = vision_session.run(
        {"pixel_values": pixel_values.astype(np.float16), "image_grid_thw": image_grid_thw}
    )
    return vision_outputs["vision_embeds"], vision_outputs["deepstack_features"]


def run_spd_inference_qwen3_vl(
    prompt_text: str,
    tokenizer,
    processor,
    tlm_config,
    dlm_session: QAICInferenceSession,
    tlm_session: QAICInferenceSession,
    mask_token_id: int,
    vocab_size: int,
    prompt_chunk_size: int,
    ctx_len: int = 4096,
    block_size: int = 16,
    max_iterations: int = 300,
    hidden_size: int = 5120,
    generation_len: int = 256,
    image=None,
    vision_session: QAICInferenceSession | None = None,
    compiled_height: int = 354,
    compiled_width: int = 536,
) -> SpecDecodingMetrics:
    """SPD inference for the qwen3-vl DFlash TLM, text-only or with a single image.

    Unlike gemma4's TLM (which has a separate text-only graph), the compiled qwen3-vl
    TLM QPC always requires vision_embeds/deepstack_features/image_idx inputs. When no
    image is given, zero buffers (sized from the TLM session's own bindings) are bound
    instead of running the vision encoder.

    `tlm_config` must be the TLM's own Qwen3VLConfig (not the DLM's plain Qwen3Config) --
    it is passed to compute_position_ids_qwen3_vl, which builds a QEffQwen3VLForConditionalGeneration
    from it to derive M-RoPE position ids and needs `vision_config`/`text_config`.
    """
    eos_token_ids = {tokenizer.eos_token_id} if tokenizer.eos_token_id is not None else set()
    batch_size = 1
    metrics = SpecDecodingMetrics(block_size=block_size)
    tlm_hidden_name = "hidden_states"

    image_processing_start = time.time()
    user_prompt = prompt_text
    if image is not None:
        # The vision QPC is compiled for one fixed (height, width) specialization; force
        # every input image to that exact resolution so the processor produces a matching
        # pixel_values/image_grid_thw shape (a mismatch fails as a hard shape error at
        # vision_session.run(), not silently).
        image = image.resize((compiled_width, compiled_height))
    proc_inputs = build_inputs_qwen3_vl(processor, user_prompt, image=image)

    input_ids_length = proc_inputs["input_ids"].shape[1]
    num_chunks = -(input_ids_length // -prompt_chunk_size)
    padded_len = num_chunks * prompt_chunk_size

    image_grid_thw = proc_inputs.get("image_grid_thw")
    position_ids, rope_deltas = compute_position_ids_qwen3_vl(
        proc_inputs["input_ids"], proc_inputs["attention_mask"], image_grid_thw, tlm_config
    )
    position_ids = torch.nn.functional.pad(position_ids, (0, padded_len - input_ids_length), value=-1)
    rope_delta = int(rope_deltas.reshape(-1)[0])

    pad_token_id = tokenizer.pad_token_id
    input_ids = torch.nn.functional.pad(
        proc_inputs["input_ids"], (0, padded_len - input_ids_length), value=pad_token_id
    )

    input_ids_np = input_ids.numpy()
    position_ids_np = position_ids.numpy()  # (4, batch, padded_len)
    generated_ids = np.full((batch_size, ctx_len - padded_len), tokenizer.pad_token_id)
    metrics.image_processing_time = time.time() - image_processing_start

    # ===== VISION ENCODER (or zero buffers when text-only) =====
    prefill_start = time.time()
    ve_binding = tlm_session.bindings[tlm_session.binding_index_map["vision_embeds"]]
    ve_dtype = tlm_session.aic_to_np_dtype_mapping.get(ve_binding.type, np.dtype(np.float32))
    df_binding = tlm_session.bindings[tlm_session.binding_index_map["deepstack_features"]]
    df_dtype = tlm_session.aic_to_np_dtype_mapping.get(df_binding.type, np.dtype(np.float32))

    if image is not None:
        if vision_session is None:
            raise ValueError("An image was given but no vision_session was provided.")
        pixel_values_np = proc_inputs["pixel_values"].numpy()
        image_grid_thw_np = image_grid_thw.numpy()
        vision_embeds, deepstack_features = run_vision_encoder_qwen3_vl(
            vision_session, pixel_values_np, image_grid_thw_np
        )
        metrics.vision_prefill_time = time.time() - prefill_start
    else:
        vision_embeds = np.zeros(tuple(int(d) for d in ve_binding.dims), dtype=ve_dtype)
        deepstack_features = np.zeros(tuple(int(d) for d in df_binding.dims), dtype=df_dtype)
        metrics.vision_prefill_time = 0.0

    vision_outputs = {
        "vision_embeds": vision_embeds.astype(ve_dtype),
        "deepstack_features": deepstack_features.astype(df_dtype),
    }
    tlm_session.set_buffers(vision_outputs)
    lang_extra = {"image_idx": np.array([[0]], dtype=np.int64)}

    # Set output buffers
    tlm_session.set_buffers({"logits": np.zeros((batch_size, prompt_chunk_size), dtype=np.int32)})
    tlm_session.set_buffers({tlm_hidden_name: np.zeros((batch_size, prompt_chunk_size, hidden_size), dtype=np.float32)})
    dlm_session.set_buffers({"logits": np.zeros((batch_size, block_size, vocab_size), dtype=np.float32)})

    tlm_cache_index = 0
    dlm_inputs = {}

    # ===== PREFILL (chunked, image-aware) =====
    num_sub_blocks = prompt_chunk_size // block_size
    remainder = prompt_chunk_size % block_size

    def feed_dlm_from_hidden(hidden_states_chunk, flat_pos_chunk, sub_start, sub_len):
        target_hidden = np.zeros((1, block_size, hidden_size), dtype=np.float32)
        pos_ids_target = np.full((1, block_size), -1, dtype=flat_pos_chunk.dtype)
        target_hidden[:, :sub_len, :] = hidden_states_chunk[:, sub_start : sub_start + sub_len, :]
        pos_ids_target[:, :sub_len] = flat_pos_chunk[:, sub_start : sub_start + sub_len]
        dlm_inputs["target_hidden"] = target_hidden
        dlm_inputs["position_ids_target"] = pos_ids_target
        dlm_inputs["position_ids"] = pos_ids_target + block_size
        dlm_inputs["input_ids"] = np.full((1, block_size), mask_token_id, dtype=np.int64)
        dlm_session.run(dlm_inputs)

    tlm_last_prefill_outputs = None
    chunk_inputs = {}
    for pi in range(num_chunks - 1):
        chunk_inputs = {
            "input_ids": input_ids_np[:, tlm_cache_index : tlm_cache_index + prompt_chunk_size],
            "position_ids": position_ids_np[:, :, tlm_cache_index : tlm_cache_index + prompt_chunk_size],
            **lang_extra,
        }
        tlm_prefill_outputs = tlm_session.run(chunk_inputs)
        lang_extra["image_idx"] = tlm_prefill_outputs["image_idx_output"]

        flat_pos_chunk = position_ids_np[0, :, tlm_cache_index : tlm_cache_index + prompt_chunk_size]
        for sub_i in range(num_sub_blocks):
            sub_start = sub_i * block_size
            feed_dlm_from_hidden(tlm_prefill_outputs[tlm_hidden_name], flat_pos_chunk, sub_start, block_size)
        if remainder > 0:
            sub_start = num_sub_blocks * block_size
            feed_dlm_from_hidden(tlm_prefill_outputs[tlm_hidden_name], flat_pos_chunk, sub_start, remainder)
        tlm_cache_index += prompt_chunk_size
        tlm_last_prefill_outputs = tlm_prefill_outputs

    # Last prefill chunk
    chunk_inputs = {
        "input_ids": input_ids_np[:, tlm_cache_index : tlm_cache_index + prompt_chunk_size],
        "position_ids": position_ids_np[:, :, tlm_cache_index : tlm_cache_index + prompt_chunk_size],
        **lang_extra,
    }
    tlm_last_prefill_outputs = tlm_session.run(chunk_inputs)
    lang_extra["image_idx"] = tlm_last_prefill_outputs["image_idx_output"]

    flat_pos_last_chunk = position_ids_np[0, :, tlm_cache_index : tlm_cache_index + prompt_chunk_size]
    last_prefill_pos_in_chunk = int(flat_pos_last_chunk.argmax())
    new_tlm_token = tlm_last_prefill_outputs["logits"][:, last_prefill_pos_in_chunk]

    last_sub = last_prefill_pos_in_chunk // block_size
    for sub_i in range(last_sub):
        sub_start = sub_i * block_size
        feed_dlm_from_hidden(tlm_last_prefill_outputs[tlm_hidden_name], flat_pos_last_chunk, sub_start, block_size)

    input_ids_block = np.full((1, block_size), mask_token_id, dtype=np.int64)
    input_ids_block[:, 0] = new_tlm_token
    sub_start = last_sub * block_size
    sub_len = min(block_size, prompt_chunk_size - sub_start)
    target_hidden = np.zeros((1, block_size, hidden_size), dtype=np.float32)
    pos_ids_target = np.full((1, block_size), -1, dtype=flat_pos_last_chunk.dtype)
    target_hidden[:, :sub_len, :] = tlm_last_prefill_outputs[tlm_hidden_name][:, sub_start : sub_start + sub_len, :]
    pos_ids_target[:, :sub_len] = flat_pos_last_chunk[:, sub_start : sub_start + sub_len]
    dlm_inputs["position_ids_target"] = pos_ids_target
    dlm_inputs["position_ids"] = np.arange(
        tlm_cache_index + last_prefill_pos_in_chunk + 1,
        tlm_cache_index + last_prefill_pos_in_chunk + 1 + block_size,
    ).reshape(1, -1)
    dlm_inputs["input_ids"] = input_ids_block
    dlm_inputs["target_hidden"] = target_hidden
    dlm_outputs = dlm_session.run(dlm_inputs)

    metrics.total_prefill_time += time.time() - prefill_start
    dlm_candidates = dlm_outputs["logits"].argmax(axis=-1)

    # ===== DECODE =====
    spd_counter_idx = tlm_cache_index + last_prefill_pos_in_chunk
    gen_idx = 0
    iteration_count = 0
    continue_generation = True

    tlm_session.set_buffers({"logits": np.zeros((batch_size, block_size), dtype=np.int32)})
    tlm_session.set_buffers({tlm_hidden_name: np.zeros((batch_size, block_size, hidden_size), dtype=np.float32)})
    # No more image tokens are ever consumed past prefill -- stop transferring the (unchanged)
    # vision buffers to/from the device on every decode call.
    tlm_session.skip_buffers(vision_outputs.keys())

    while gen_idx < generation_len and iteration_count < max_iterations and continue_generation:
        iteration_count += 1
        dlm_candidates[:, 0] = new_tlm_token

        tlm_lang_position_ids = build_decode_position_ids_qwen3_vl(dlm_inputs["position_ids"], rope_delta)

        tlm_decode_start = time.time()
        tlm_decode_outputs = tlm_session.run(
            {
                "input_ids": dlm_candidates,
                "position_ids": tlm_lang_position_ids,
                **lang_extra,
            }
        )
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
        input_ids_block[:, 0] = new_tlm_token
        dlm_inputs["input_ids"] = input_ids_block
        dlm_inputs["target_hidden"] = target_hidden
        dlm_outputs = dlm_session.run(dlm_inputs)
        metrics.dlm_decode_time += time.time() - dlm_decode_start

        dlm_candidates = dlm_outputs["logits"].argmax(axis=-1)

    metrics.num_total_iters = iteration_count
    return metrics


def _run_vision_inference_qwen3_vl(
    tokenizer,
    processor,
    tlm_config,
    config,
    dlm_session,
    tlm_session,
    mask_token_id,
    prompt,
    image,
    image_url,
    image_prompt,
    tlm_devices,
    vision_qpc,
    vision_devices,
    height,
    width,
    ctx_len,
    iteration,
    generation_len,
):
    if processor is None:
        raise SystemExit("qwen3-vl requires a processor, which failed to load. See warning above.")

    use_image = bool(image)
    img = None
    vision_session = None
    if use_image:
        if not vision_qpc:
            raise SystemExit("--image needs the qwen3-vl vision-encoder QPC. Pass --vision_qpc <path>.")
        resolved_image_url = image_url or IMAGE_URL
        import requests
        from PIL import Image

        console.print(f"[cyan]Image:[/cyan] {resolved_image_url}")
        img = Image.open(requests.get(resolved_image_url, stream=True).raw).convert("RGB")

        vd = vision_devices if vision_devices is not None else tlm_devices
        console.print(f"[bold blue]Loading vision QPC session on devices {vd}...[/bold blue]")
        vision_session = QAICInferenceSession(vision_qpc, vd)
        prompt_text = image_prompt or IMAGE_PROMPT
    else:
        prompt_text = prompt
    console.print(f"[cyan]Prompt:[/cyan] {prompt_text[:120].strip()}")

    resolved_height = height if height is not None else 354
    resolved_width = width if width is not None else 536

    prompt_chunk_size = max(
        [x[tlm_session.binding_index_map["input_ids"]][1][1] for x in tlm_session.allowed_shapes]
        + [tlm_session.bindings[tlm_session.binding_index_map["input_ids"]].dims[1]]
    )

    metrics = run_spd_inference_qwen3_vl(
        prompt_text=prompt_text,
        tokenizer=tokenizer,
        processor=processor,
        tlm_config=tlm_config,
        dlm_session=dlm_session,
        tlm_session=tlm_session,
        mask_token_id=mask_token_id,
        vocab_size=config.vocab_size,
        prompt_chunk_size=prompt_chunk_size,
        ctx_len=ctx_len,
        block_size=config.block_size,
        max_iterations=iteration,
        hidden_size=config.hidden_size,
        generation_len=generation_len,
        image=img,
        vision_session=vision_session,
        compiled_height=resolved_height,
        compiled_width=resolved_width,
    )
    # metrics.vision_prefill_time is the standalone vision-encoder latency (0.0 when
    # text-only) — same semantics as the gemma path's vision_encode_time_s.
    vision_encode_time_s = metrics.vision_prefill_time if use_image else None
    return metrics, {"vision_encode_time_s": vision_encode_time_s}


def run_vision_inference(
    tlm_qpc: str,
    dlm_qpc: str,
    vision_qpc: str,
    tlm_model_name: str,
    dlm_model_name: str,
    tlm_devices: list,
    dlm_devices: list,
    vision_devices: list | None = None,
    prompt: str | None = None,
    image: bool = False,
    image_url: str | None = None,
    image_prompt: str | None = None,
    height: int | None = None,
    width: int | None = None,
    iteration: int = 300,
    ctx_len: int = 2048,
    generation_len: int = 256,
    hf_token: str | None = None,
):
    """Load tokenizer/config/processor/sessions and run one VLM SPD generation end to end
    (gemma4 or qwen3-vl, auto-detected from ``tlm_model_name``'s config.model_type).

    ``prompt``/``image``/``image_url``/``image_prompt`` mirror the CLI flags of the
    now-removed ``dflash_spd_vision_single_prompt.py``. When ``image`` is set and
    ``image_url``/``image_prompt`` are not given, they fall back to ``IMAGE_URL``/
    ``IMAGE_PROMPT`` (matching that script's argparse defaults).

    Returns ``(metrics, tokenizer, extra)`` where ``extra`` carries
    ``{"vision_encode_time_s": ...}``.
    """
    if not image and not prompt:
        raise SystemExit("Provide prompt=<text>, or image=True for a single image prompt.")

    console.print("[bold blue]Loading tokenizer and config...[/bold blue]")
    tokenizer = transformers.AutoTokenizer.from_pretrained(tlm_model_name, token=hf_token, trust_remote_code=True)
    tlm_config = transformers.AutoConfig.from_pretrained(tlm_model_name, token=hf_token, trust_remote_code=True)
    is_qwen3vl = tlm_config.model_type == "qwen3_vl"

    config = transformers.AutoConfig.from_pretrained(dlm_model_name, token=hf_token, trust_remote_code=True)
    vocab_size = config.vocab_size
    hidden_size = config.hidden_size
    block_size = config.block_size
    dflash_cfg = getattr(config, "dflash_config", None) or config.to_dict().get("dflash_config", {})
    mask_token_id = dflash_cfg["mask_token_id"] if isinstance(dflash_cfg, dict) else dflash_cfg.mask_token_id

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    console.print("[bold blue]Loading QAIC inference sessions...[/bold blue]")
    dlm_session = QAICInferenceSession(dlm_qpc, dlm_devices)
    tlm_session = QAICInferenceSession(tlm_qpc, tlm_devices)
    dlm_session.skip_buffers(
        {x for x in dlm_session.input_names + dlm_session.output_names if x.startswith("past_")}
    )
    tlm_session.skip_buffers(
        {x for x in tlm_session.input_names + tlm_session.output_names if x.startswith("past_")}
    )

    prompt_chunk_size = max(
        [x[tlm_session.binding_index_map["input_ids"]][1][1] for x in tlm_session.allowed_shapes]
        + [tlm_session.bindings[tlm_session.binding_index_map["input_ids"]].dims[1]]
    )
    console.print(f"prompt_chunk_size = {prompt_chunk_size}")

    # SPD feeds the TLM a full block of `block_size` tokens per decode step, so the TLM
    # QPC MUST expose a decode specialization with input_ids seq_len == block_size. A QPC
    # compiled WITHOUT dflash_block_size only has seq_len=1 decode -> SPD would silently run
    # wrong. Verify here and fail loudly with a clear message.
    tlm_seq_lens = sorted({x[tlm_session.binding_index_map["input_ids"]][1][1] for x in tlm_session.allowed_shapes})
    console.print(f"TLM input_ids seq_lens (allowed shapes): {tlm_seq_lens}")
    if block_size not in tlm_seq_lens:
        raise SystemExit(
            f"TLM QPC has no decode specialization with seq_len={block_size} "
            f"(found {tlm_seq_lens}). Recompile the TLM with dflash_block_size={block_size} "
            "(e.g. gemma4_example.py with USE_DFLASH_BLOCK_SIZE=True), then point --tlm_qpc at it."
        )

    processor = None
    try:
        from transformers import AutoProcessor

        processor = AutoProcessor.from_pretrained(tlm_model_name, token=hf_token, trust_remote_code=True)
    except Exception as e:
        console.print(f"[yellow]⚠ No processor for {tlm_model_name} ({e}); using tokenizer template.[/yellow]")

    if is_qwen3vl:
        metrics, output_extra = _run_vision_inference_qwen3_vl(
            tokenizer=tokenizer,
            processor=processor,
            tlm_config=tlm_config,
            config=config,
            dlm_session=dlm_session,
            tlm_session=tlm_session,
            mask_token_id=mask_token_id,
            prompt=prompt,
            image=image,
            image_url=image_url,
            image_prompt=image_prompt,
            tlm_devices=tlm_devices,
            vision_qpc=vision_qpc,
            vision_devices=vision_devices,
            height=height,
            width=width,
            ctx_len=ctx_len,
            iteration=iteration,
            generation_len=generation_len,
        )
    else:
        metrics, output_extra = _run_vision_inference_gemma4(
            tokenizer=tokenizer,
            processor=processor,
            dlm_session=dlm_session,
            tlm_session=tlm_session,
            vocab_size=vocab_size,
            hidden_size=hidden_size,
            block_size=block_size,
            mask_token_id=mask_token_id,
            prompt_chunk_size=prompt_chunk_size,
            prompt=prompt,
            image=image,
            image_url=image_url,
            image_prompt=image_prompt,
            tlm_devices=tlm_devices,
            vision_qpc=vision_qpc,
            vision_devices=vision_devices,
            ctx_len=ctx_len,
            iteration=iteration,
            generation_len=generation_len,
        )

    return metrics, tokenizer, output_extra

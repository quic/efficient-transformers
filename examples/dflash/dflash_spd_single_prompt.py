# -----------------------------------------------------------------------------
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------

import argparse
import time

import numpy as np
import torch
import transformers
from qaic_infer import QAICInferenceSession
from rich.console import Console
from rich.markup import escape
from utils import format_prompt

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
    mask_token_embed,
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
    tlm_session.set_buffers({"output_embeds": np.zeros((batch_size, prompt_chunk_size, hidden_size), dtype=np.float32)})
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
            dlm_inputs["noise_embeds"] = np.full((1, block_size, hidden_size), 1, dtype=np.float32)
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
            dlm_inputs["noise_embeds"] = np.full((1, block_size, hidden_size), 1, dtype=np.float32)
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
        dlm_inputs["noise_embeds"] = np.full((1, block_size, hidden_size), 1, dtype=np.float32)
        dlm_session.run(dlm_inputs)

    noise_embeds = np.tile(mask_token_embed, (1, block_size, 1))
    noise_embeds[:, 0, :] = tlm_last_prefill_outputs["output_embeds"][:, last_prefill_pos_in_chunk, :]
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
    dlm_inputs["noise_embeds"] = noise_embeds
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
    tlm_session.set_buffers({"output_embeds": np.zeros((batch_size, block_size, hidden_size), dtype=np.float32)})

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
        noise_embeds[:, 0, :] = tlm_decode_outputs["output_embeds"][:, accepted_length, :]
        dlm_inputs["noise_embeds"] = noise_embeds
        dlm_inputs["target_hidden"] = target_hidden
        dlm_outputs = dlm_session.run(dlm_inputs)
        metrics.dlm_decode_time += time.time() - dlm_decode_start

        dlm_candidates = dlm_outputs["logits"].argmax(axis=-1)

    metrics.num_total_iters = iteration_count
    return metrics


# ===== ARGUMENT PARSING =====


def parse_args():
    parser = argparse.ArgumentParser(description="SPD single-prompt inference")
    parser.add_argument("--prompt", required=True, help="Input prompt text")
    parser.add_argument("--tlm_qpc", required=True)
    parser.add_argument("--dlm_qpc", required=True)
    parser.add_argument("--tlm_model_name", required=True)
    parser.add_argument("--dlm_model_name", required=True)
    parser.add_argument("--noise_embed_path", required=True)
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
    return parser.parse_args()


# ===== MAIN =====


def main():
    args = parse_args()

    console.print("[bold blue]Loading tokenizer and config...[/bold blue]")
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        args.tlm_model_name, token=args.hf_token, trust_remote_code=True
    )
    config = transformers.AutoConfig.from_pretrained(args.dlm_model_name, token=args.hf_token, trust_remote_code=True)
    vocab_size = config.vocab_size
    hidden_size = config.hidden_size
    block_size = config.block_size
    mask_token_embed = np.load(args.noise_embed_path)

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

    messages = [{"role": "user", "content": format_prompt(args.prompt, args.category)}]
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
        mask_token_embed=mask_token_embed,
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
    print("=" * w + "\n")


if __name__ == "__main__":
    main()

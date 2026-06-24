# -----------------------------------------------------------------------------
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------

import argparse
import csv
import os
import time
from typing import Optional

import numpy as np
import torch
import transformers
from rich.console import Console
from rich.markup import escape
from utils import load_and_process_dataset

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
        self.generated_sources: list = []  # "dlm" or "tlm" per token

    def acceptance_rate(self) -> float:
        if self.num_total_iters == 0:
            return 0.0
        return self.total_generated_tokens / self.num_total_iters

    def dlm_tok_rate(self) -> float:
        if self.dlm_decode_time <= 0:
            return 0.0
        num_tok_drafted = self.block_size * self.num_total_iters
        return num_tok_drafted / self.dlm_decode_time

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
) -> SpecDecodingMetrics:
    eos_token_ids = {tokenizer.eos_token_id} if tokenizer.eos_token_id is not None else set()

    prompt = [prompt_text]
    batch_size = 1
    metrics = SpecDecodingMetrics(block_size=block_size)

    # Tokenize
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
        ## Add support for when the prefill_seq_len is more than block_size
        for sub_i in range(num_sub_blocks):
            sub_start = sub_i * block_size
            dlm_inputs["target_hidden"] = tlm_prefill_outputs["hidden_states"][:, sub_start : sub_start + block_size, :]
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

    ## Add support for when the prefill_seq_len is more than block_size
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

    ## Add support when prefill_seq_len is not a multiple of block_size
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

        # EOS check
        dlm_candidate_ids = list(dlm_candidates[0, 1 : accepted_length + 1])
        this_iter_gen_ids = dlm_candidate_ids + [new_tlm_token[0]]
        for tok_id in this_iter_gen_ids:
            if tok_id in eos_token_ids:
                continue_generation = False
                break

        if not continue_generation:
            break

        # Next DLM iteration
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


# ===== DATASET CONFIG =====

DATASET_CONFIG = {
    "gsm8k": {
        "hf_path": "openai/gsm8k",
        "hf_name": "main",
        "split": "test",
        "prompt_field": "question",
        "label": "GSM8K",
    },
    "math500": {
        "hf_path": "HuggingFaceH4/MATH-500",
        "hf_name": None,
        "split": "test",
        "prompt_field": "problem",
        "label": "MATH-500",
    },
    "humaneval": {
        "hf_path": "openai/openai_humaneval",
        "hf_name": None,
        "split": "test",
        "prompt_field": "prompt",
        "label": "HumanEval",
    },
}


# ===== EVALUATION LOOP =====

PER_SAMPLE_FIELDS = [
    "dataset",
    "sample_idx",
    "acceptance_rate",
    "dlm_tps",
    "tlm_tps",
    "spd_tps",
    "total_generated_tokens",
    "num_iters",
    "prefill_time_s",
    "tlm_decode_time_s",
    "dlm_decode_time_s",
]

SUMMARY_FIELDS = [
    "dataset",
    "num_evaluated",
    "num_total",
    "avg_acceptance_rate",
    "min_acceptance_rate",
    "max_acceptance_rate",
    "avg_dlm_tps",
    "min_dlm_tps",
    "max_dlm_tps",
    "avg_tlm_tps",
    "min_tlm_tps",
    "max_tlm_tps",
    "avg_spd_tps",
    "min_spd_tps",
    "max_spd_tps",
]


def _write_per_sample_csv(rows: list, path: str):
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=PER_SAMPLE_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    console.print(f"[green]Per-sample CSV → {path}[/green]")


def _append_summary_csv(row: dict, path: str):
    write_header = not os.path.exists(path)
    with open(path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=SUMMARY_FIELDS)
        if write_header:
            writer.writeheader()
        writer.writerow(row)
    console.print(f"[green]Summary CSV    → {path}[/green]")


def evaluate_dataset(
    dataset_name: str,
    tokenizer,
    dlm_session,
    tlm_session,
    mask_token_id: int,
    vocab_size: int,
    prompt_chunk_size: int,
    ctx_len: int = 4096,
    block_size: int = 10,
    max_iterations: int = 300,
    hidden_size: int = 4096,
    generation_len: int = 1024,
    num_samples: Optional[int] = None,
    output_dir: str = "./results",
):
    cfg = DATASET_CONFIG[dataset_name]
    console.print(f"[bold blue]Loading {cfg['label']} dataset...[/bold blue]")
    dataset = load_and_process_dataset(dataset_name)

    if num_samples is not None:
        # dataset = dataset.shuffle(seed=0).select(range(min(num_samples, len(dataset))))
        dataset = dataset.select(range(min(num_samples, len(dataset))))
    console.print(f"[green]✓ Loaded {len(dataset)} {cfg['label']} problems[/green]")

    all_ar, all_dlm_tps, all_tlm_tps, all_spd_tps = [], [], [], []
    per_sample_rows = []

    for i, sample in enumerate(dataset):
        user_content = sample["turns"][0]
        messages = [{"role": "user", "content": user_content}]
        prompt_text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
        )
        console.print(f"[cyan]({i + 1}/{len(dataset)})[/cyan] Input: {user_content[:80].strip()}")

        # try:
        metrics = run_spd_inference_single(
            prompt_text=prompt_text,
            tokenizer=tokenizer,
            dlm_session=dlm_session,
            tlm_session=tlm_session,
            vocab_size=vocab_size,
            prompt_chunk_size=prompt_chunk_size,
            ctx_len=ctx_len,
            block_size=block_size,
            max_iterations=max_iterations,
            hidden_size=hidden_size,
            generation_len=generation_len,
            mask_token_id=mask_token_id,
        )

        ar = metrics.acceptance_rate()
        dlm_tps = metrics.dlm_tok_rate()
        tlm_tps = metrics.tlm_tok_rate()
        spd_tps = metrics.spd_tok_rate()

        all_ar.append(ar)
        all_dlm_tps.append(dlm_tps)
        all_tlm_tps.append(tlm_tps)
        all_spd_tps.append(spd_tps)

        per_sample_rows.append(
            {
                "dataset": dataset_name,
                "sample_idx": i,
                "acceptance_rate": round(ar, 4),
                "dlm_tps": round(dlm_tps, 2),
                "tlm_tps": round(tlm_tps, 2),
                "spd_tps": round(spd_tps, 2),
                "total_generated_tokens": metrics.total_generated_tokens,
                "num_iters": metrics.num_total_iters,
                "prefill_time_s": round(metrics.total_prefill_time, 4),
                "tlm_decode_time_s": round(metrics.tlm_decode_time, 4),
                "dlm_decode_time_s": round(metrics.dlm_decode_time, 4),
            }
        )

        console.print(f"  AR={ar:.2f}  DLM={dlm_tps:.1f} tok/s  TLM={tlm_tps:.1f} tok/s  SPD={spd_tps:.1f} tok/s")

        output_parts = ["Output: "]
        for tok_id, source in zip(metrics.generated_ids, metrics.generated_sources):
            text = escape(tokenizer.decode([tok_id], skip_special_tokens=True))
            if source == "dlm":
                output_parts.append(f"[blue]{text}[/blue]")
            else:
                output_parts.append(f"[white]{text}[/white]")
        console.print("".join(output_parts))

        # except Exception as e:
        #     console.print(f"[red]  ✗ Error on sample {i}: {e}[/red]")

    # ===== SUMMARY =====
    if all_ar:
        w = 46
        print("\n" + "=" * w)
        print(f"  {cfg['label']} SPD Evaluation — Averages")
        print("=" * w)
        print(f"  {'Metric':<30} {'Avg':>6}  {'Min':>6}  {'Max':>6}")
        print("-" * w)
        for name, vals in [
            ("Acceptance Rate (tok/iter)", all_ar),
            ("DLM Throughput  (tok/s)", all_dlm_tps),
            ("TLM Throughput  (tok/s)", all_tlm_tps),
            ("SPD Decode Speed (tok/s)", all_spd_tps),
        ]:
            print(f"  {name:<30} {np.mean(vals):>6.2f}  {np.min(vals):>6.2f}  {np.max(vals):>6.2f}")
        print("=" * w)
        print(f"  Evaluated {len(all_ar)} / {len(dataset)} samples successfully.")
        print("=" * w + "\n")

        # ===== SAVE CSV =====
        os.makedirs(output_dir, exist_ok=True)
        _write_per_sample_csv(
            per_sample_rows,
            os.path.join(output_dir, f"{dataset_name}_per_sample.csv"),
        )
        _append_summary_csv(
            {
                "dataset": dataset_name,
                "num_evaluated": len(all_ar),
                "num_total": len(dataset),
                "avg_acceptance_rate": round(float(np.mean(all_ar)), 4),
                "min_acceptance_rate": round(float(np.min(all_ar)), 4),
                "max_acceptance_rate": round(float(np.max(all_ar)), 4),
                "avg_dlm_tps": round(float(np.mean(all_dlm_tps)), 2),
                "min_dlm_tps": round(float(np.min(all_dlm_tps)), 2),
                "max_dlm_tps": round(float(np.max(all_dlm_tps)), 2),
                "avg_tlm_tps": round(float(np.mean(all_tlm_tps)), 2),
                "min_tlm_tps": round(float(np.min(all_tlm_tps)), 2),
                "max_tlm_tps": round(float(np.max(all_tlm_tps)), 2),
                "avg_spd_tps": round(float(np.mean(all_spd_tps)), 2),
                "min_spd_tps": round(float(np.min(all_spd_tps)), 2),
                "max_spd_tps": round(float(np.max(all_spd_tps)), 2),
            },
            os.path.join(output_dir, "summary.csv"),
        )
    else:
        print("No successful results.")


# ===== ARGUMENT PARSING =====


def parse_args():
    parser = argparse.ArgumentParser(description="SPD benchmark — gsm8k / math500 / humaneval")
    parser.add_argument("--dataset", required=True, choices=list(DATASET_CONFIG.keys()))
    parser.add_argument("--tlm_qpc", required=True)
    parser.add_argument("--dlm_qpc", required=True)
    parser.add_argument("--tlm_model_name", required=True)
    parser.add_argument("--dlm_model_name", required=True)
    parser.add_argument("--iteration", type=int, default=300)
    parser.add_argument("--ctx_len", type=int, default=4096)
    parser.add_argument("--generation_len", type=int, default=1024)
    parser.add_argument("--tlm_devices", nargs="+", type=int, required=True)
    parser.add_argument("--dlm_devices", nargs="+", type=int, required=True)
    parser.add_argument("--hf_token", default=None)
    parser.add_argument("--num_samples", type=int, default=0, help="Number of samples to run (0 = all)")
    parser.add_argument("--output_dir", default="./results", help="Directory for CSV output (default: ./results)")
    return parser.parse_args()


# ===== MAIN =====


def main():
    args = parse_args()
    num_samples = args.num_samples if args.num_samples > 0 else None

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

    evaluate_dataset(
        dataset_name=args.dataset,
        tokenizer=tokenizer,
        dlm_session=dlm_session,
        tlm_session=tlm_session,
        vocab_size=vocab_size,
        prompt_chunk_size=prompt_chunk_size,
        mask_token_id=mask_token_id,
        ctx_len=args.ctx_len,
        block_size=block_size,
        max_iterations=args.iteration,
        hidden_size=hidden_size,
        generation_len=args.generation_len,
        num_samples=num_samples,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()

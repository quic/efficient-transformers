# -----------------------------------------------------------------------------
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------

"""
DFlash SPD benchmark over a TEXT dataset for a vision-language model (VLM), language-only.

This is the language-only counterpart of dflash_spd_vision_benchmark.py (which runs the
MathVision image dataset). It drives the VLM's language decoder over a text dataset
(humaneval / gsm8k / math500) with NO image: vision_embeds is zero-bound and the vision
encoder QPC is not loaded, so only the language part of the model runs. This is the
validated language-only path (the vision-capable lang QPC fed a text prompt with zero
vision_embeds) — it does NOT require a separate SKIP_VISION=True build.

Reuses:
  * build_gemma_input_ids  -> gemma chat-template input_ids for a text prompt
  * run_spd_inference_single (vision_embeds=None) -> the TLM+DLM speculative decode loop
  * utils.load_and_process_dataset -> the text dataset

Example:
    python dflash_spd_vision_text_benchmark.py \
        --tlm_qpc <lang qpc> --dlm_qpc <draft qpc> \
        --tlm_model_name google/gemma-4-31B-it --dlm_model_name z-lab/gemma-4-31B-it-DFlash \
        --tlm_devices 40 41 42 43 --dlm_devices 44 45 46 47 \
        --ctx_len 2048 --dataset humaneval --num_samples 20
"""

import argparse
import csv
import os

import numpy as np
import torch
import transformers
from rich.console import Console

from QEfficient.generation.cloud_infer import QAICInferenceSession

# Reuse the SPD loop + gemma text-prompt builder from the single-prompt script.
from dflash_spd_vision_single_prompt import build_gemma_input_ids, run_spd_inference_single
from utils import format_prompt, load_and_process_dataset

torch.manual_seed(42)
np.random.seed(42)

console = Console()

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
]

SUMMARY_FIELDS = [
    "dataset",
    "num_evaluated",
    "num_total",
    "avg_acceptance_rate",
    "min_acceptance_rate",
    "max_acceptance_rate",
    "avg_dlm_tps",
    "avg_tlm_tps",
    "avg_spd_tps",
]


def _write_per_sample_csv(rows, path):
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=PER_SAMPLE_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    console.print(f"[green]Per-sample CSV → {path}[/green]")


def _append_summary_csv(row, path):
    write_header = not os.path.exists(path)
    with open(path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=SUMMARY_FIELDS)
        if write_header:
            writer.writeheader()
        writer.writerow(row)
    console.print(f"[green]Summary CSV    → {path}[/green]")


def evaluate_text(
    *,
    tokenizer,
    processor,
    dlm_session,
    tlm_session,
    mask_token_id,
    vocab_size,
    prompt_chunk_size,
    ctx_len,
    block_size,
    max_iterations,
    hidden_size,
    generation_len,
    dataset_name,
    num_samples,
    category,
    do_format_prompt,
    output_dir,
):
    console.print(f"[bold blue]Loading text dataset '{dataset_name}'...[/bold blue]")
    dataset = load_and_process_dataset(dataset_name)
    if num_samples is not None:
        dataset = dataset.select(range(min(num_samples, len(dataset))))
    console.print(f"[green]✓ Loaded {len(dataset)} samples[/green]")

    all_ar, all_dlm, all_tlm, all_spd = [], [], [], []
    per_sample_rows = []

    for i, sample in enumerate(dataset):
        prompt = sample["turns"][0]
        if do_format_prompt:
            prompt = format_prompt(prompt, category)
        console.print(f"[cyan]({i + 1}/{len(dataset)})[/cyan] {prompt[:80].strip().replace(chr(10), ' ')}")

        try:
            # Text prompt through the VLM language decoder (no image → vision_embeds zeros).
            input_ids = build_gemma_input_ids(processor, tokenizer, prompt)
            metrics = run_spd_inference_single(
                prompt_text="",
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
                input_ids=input_ids,
                mm_token_type_ids=None,
                vision_embeds=None,
            )
        except Exception as e:
            console.print(f"[red]  ✗ Error on sample {i}: {e}[/red]")
            continue

        ar = metrics.acceptance_rate()
        dlm_tps = metrics.dlm_tok_rate()
        tlm_tps = metrics.tlm_tok_rate()
        spd_tps = metrics.spd_tok_rate()
        all_ar.append(ar)
        all_dlm.append(dlm_tps)
        all_tlm.append(tlm_tps)
        all_spd.append(spd_tps)
        console.print(f"  AR={ar:.2f}  DLM={dlm_tps:.1f}  TLM={tlm_tps:.1f}  SPD={spd_tps:.1f} tok/s")

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
            }
        )

    if not all_ar:
        console.print("[red]No samples evaluated successfully.[/red]")
        return

    os.makedirs(output_dir, exist_ok=True)
    _write_per_sample_csv(per_sample_rows, os.path.join(output_dir, f"{dataset_name}_per_sample.csv"))
    summary = {
        "dataset": dataset_name,
        "num_evaluated": len(all_ar),
        "num_total": len(dataset),
        "avg_acceptance_rate": round(float(np.mean(all_ar)), 4),
        "min_acceptance_rate": round(float(np.min(all_ar)), 4),
        "max_acceptance_rate": round(float(np.max(all_ar)), 4),
        "avg_dlm_tps": round(float(np.mean(all_dlm)), 2),
        "avg_tlm_tps": round(float(np.mean(all_tlm)), 2),
        "avg_spd_tps": round(float(np.mean(all_spd)), 2),
    }
    _append_summary_csv(summary, os.path.join(output_dir, "summary.csv"))

    console.print("\n" + "=" * 50)
    console.print(f"  VLM language-only SPD ({dataset_name}) — Averages")
    console.print("=" * 50)
    console.print(f"  {'Metric':<32}{'Avg':>8}{'Min':>8}{'Max':>8}")
    console.print("-" * 50)
    console.print(f"  {'Acceptance Rate (tok/iter)':<32}{np.mean(all_ar):>8.2f}{np.min(all_ar):>8.2f}{np.max(all_ar):>8.2f}")
    console.print(f"  {'DLM Throughput  (tok/s)':<32}{np.mean(all_dlm):>8.2f}")
    console.print(f"  {'TLM Throughput  (tok/s)':<32}{np.mean(all_tlm):>8.2f}")
    console.print(f"  {'SPD Decode Speed (tok/s)':<32}{np.mean(all_spd):>8.2f}")
    console.print("=" * 50)
    console.print(f"  Evaluated {len(all_ar)} / {len(dataset)} samples successfully.")
    console.print("=" * 50)


def _qpc_num_devices(qpc_dir):
    import json as _json

    for cand in (
        os.path.join(qpc_dir, "qconfig.json"),
        os.path.join(os.path.dirname(qpc_dir), "qconfig.json"),
    ):
        if os.path.exists(cand):
            try:
                cfg = _json.load(open(cand))
                return int(cfg["qpc_config"]["compiler_config"]["mdp_ts_num_devices"])
            except Exception:
                return None
    return None


def _qpc_ctx_len(qpc_dir):
    import json as _json

    spec_path = os.path.join(os.path.dirname(qpc_dir), "specializations.json")
    if os.path.exists(spec_path):
        try:
            specs = _json.load(open(spec_path))["specializations"]
            vals = {int(s["symbols"]["ctx_len"]) for s in specs if "ctx_len" in s.get("symbols", {})}
            return min(vals) if vals else None
        except Exception:
            return None
    return None


def parse_args():
    p = argparse.ArgumentParser(description="DFlash SPD benchmark over a TEXT dataset for a VLM (language-only).")
    p.add_argument("--tlm_qpc", required=True, help="VLM language-decoder QPC (built for SPD, block-sized decode).")
    p.add_argument("--dlm_qpc", required=True)
    p.add_argument("--tlm_model_name", required=True)
    p.add_argument("--dlm_model_name", required=True)
    p.add_argument("--tlm_devices", nargs="+", type=int, required=True)
    p.add_argument("--dlm_devices", nargs="+", type=int, required=True)
    p.add_argument("--dataset", default="humaneval", choices=["humaneval", "gsm8k", "math500"])
    p.add_argument("--num_samples", type=int, default=0, help="0 = all")
    p.add_argument("--ctx_len", type=int, default=2048)
    p.add_argument("--generation_len", type=int, default=256)
    p.add_argument("--iteration", type=int, default=300)
    p.add_argument("--category", default="", help="Prompt category for --format_prompt (math / coding / reasoning).")
    p.add_argument("--format_prompt", action="store_true", help="Wrap the prompt with the category template.")
    p.add_argument("--output_dir", default="./results-text-vlm")
    p.add_argument("--hf_token", default=None)
    return p.parse_args()


def main():
    args = parse_args()
    num_samples = args.num_samples if args.num_samples > 0 else None

    for name, qpc, devs in (("TLM", args.tlm_qpc, args.tlm_devices), ("DLM", args.dlm_qpc, args.dlm_devices)):
        nd = _qpc_num_devices(qpc)
        if nd is not None and len(devs) != nd:
            raise SystemExit(f"{name} QPC was compiled for {nd} device(s) but got {len(devs)} ({devs}).")

    tlm_ctx = _qpc_ctx_len(args.tlm_qpc)
    dlm_ctx = _qpc_ctx_len(args.dlm_qpc)
    compiled_ctx = min([c for c in (tlm_ctx, dlm_ctx) if c is not None], default=None)
    if compiled_ctx is not None and args.ctx_len > compiled_ctx:
        console.print(f"[yellow]⚠ --ctx_len {args.ctx_len} > compiled {compiled_ctx}; clamping.[/yellow]")
        args.ctx_len = compiled_ctx

    console.print("[bold blue]Loading tokenizer, processor and config...[/bold blue]")
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        args.tlm_model_name, token=args.hf_token, trust_remote_code=True
    )
    processor = transformers.AutoProcessor.from_pretrained(
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

    console.print("[bold blue]Loading QAIC inference sessions (language-only, no vision QPC)...[/bold blue]")
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
    tlm_seq_lens = sorted({x[tlm_session.binding_index_map["input_ids"]][1][1] for x in tlm_session.allowed_shapes})
    if block_size not in tlm_seq_lens:
        raise SystemExit(
            f"TLM QPC has no decode specialization with seq_len={block_size} (found {tlm_seq_lens}). "
            f"Recompile the TLM with dflash_block_size={block_size}."
        )

    evaluate_text(
        tokenizer=tokenizer,
        processor=processor,
        dlm_session=dlm_session,
        tlm_session=tlm_session,
        mask_token_id=mask_token_id,
        vocab_size=vocab_size,
        prompt_chunk_size=prompt_chunk_size,
        ctx_len=args.ctx_len,
        block_size=block_size,
        max_iterations=args.iteration,
        hidden_size=hidden_size,
        generation_len=args.generation_len,
        dataset_name=args.dataset,
        num_samples=num_samples,
        category=args.category,
        do_format_prompt=args.format_prompt,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()

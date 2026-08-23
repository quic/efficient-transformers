# -----------------------------------------------------------------------------
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------

"""
DFlash SPD benchmark over an IMAGE dataset (MathVision).

Runs the gemma4 vision + DFlash speculative-decoding pipeline over many image+text prompts
from MathLLMs/MathVision and reports per-sample + summary SPD metrics (acceptance rate,
DLM/TLM/SPD throughput, prefill time) PLUS a standalone vision-encode latency.

Reuses the vision-capable pieces from dflash_spd_vision_single_prompt.py:
  * build_gemma_inputs (with pil_image=) -> input_ids + mm_token_type_ids + processor inputs
  * run_gemma_vision_encoder            -> vision_embeds from the vision QPC
  * run_spd_inference_single            -> the TLM+DLM speculative decode loop

This is the multimodal counterpart of dflash_spd_benchmark.py (which is text-only). It does
NOT score answer correctness — it measures speed / acceptance only.

Example:
    python dflash_spd_vision_benchmark.py \
        --tlm_qpc <lang qpc> --dlm_qpc <draft qpc> --vision_qpc <encoder qpc> \
        --tlm_model_name google/gemma-4-31B-it --dlm_model_name z-lab/gemma-4-31B-it-DFlash \
        --tlm_devices 32 33 34 35 --dlm_devices 32 33 34 35 --vision_devices 36 37 38 39 \
        --ctx_len 2048 --split testmini --num_samples 20
"""

import argparse
import csv
import os
from time import perf_counter
from typing import Optional

import numpy as np
import torch
import transformers
from rich.console import Console
from rich.markup import escape

from QEfficient.generation.cloud_infer import QAICInferenceSession

# Reuse the vision-capable SPD pipeline from the single-prompt script (same directory).
from dflash_spd_vision_single_prompt import (
    build_gemma_inputs,
    run_gemma_vision_encoder,
    run_spd_inference_single,
)

torch.manual_seed(42)
np.random.seed(42)

console = Console()

DATASET_REPO = "MathLLMs/MathVision"


# ===== PROMPT BUILDING =====


def build_mathvision_prompt(sample) -> str:
    """Build the user text for a MathVision sample: question + choices (if any).

    MathVision free-response items have an empty `options` list; multiple-choice items have
    options we render as "(A) ... (B) ...". The image is supplied separately (decoded_image).
    """
    question = (sample.get("question") or "").strip()
    options = sample.get("options") or []
    if options:
        letters = "ABCDEFGHIJKLMNOP"
        choices = "\n".join(f"({letters[i]}) {opt}" for i, opt in enumerate(options))
        return f"{question}\n{choices}"
    return question


# ===== CSV FIELDS =====

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
    "vision_encode_time_s",
]

SUMMARY_FIELDS = [
    "dataset",
    "split",
    "num_evaluated",
    "num_total",
    "avg_acceptance_rate",
    "min_acceptance_rate",
    "max_acceptance_rate",
    "avg_dlm_tps",
    "avg_tlm_tps",
    "avg_spd_tps",
    "avg_vision_encode_s",
    "min_vision_encode_s",
    "max_vision_encode_s",
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


# ===== EVALUATION LOOP =====


def evaluate_mathvision(
    *,
    tokenizer,
    processor,
    dlm_session,
    tlm_session,
    vision_session,
    mask_token_id: int,
    vocab_size: int,
    prompt_chunk_size: int,
    ctx_len: int,
    block_size: int,
    max_iterations: int,
    hidden_size: int,
    generation_len: int,
    split: str,
    num_samples: Optional[int],
    output_dir: str,
):
    from datasets import load_dataset

    console.print(f"[bold blue]Loading {DATASET_REPO} (split={split})...[/bold blue]")
    try:
        dataset = load_dataset(DATASET_REPO, split=split)
    except Exception as e:
        raise SystemExit(
            f"Failed to load {DATASET_REPO} (split={split}): {e}\n"
            "The dataset must be present in HF_HUB_CACHE or downloadable. If the box is "
            "offline and it is not cached, pre-fetch it once online (or unset HF_HUB_OFFLINE)."
        )

    if num_samples is not None:
        dataset = dataset.select(range(min(num_samples, len(dataset))))
    console.print(f"[green]✓ Loaded {len(dataset)} MathVision samples[/green]")

    all_ar, all_dlm, all_tlm, all_spd, all_vis = [], [], [], [], []
    per_sample_rows = []

    for i, sample in enumerate(dataset):
        prompt = build_mathvision_prompt(sample)
        pil_image = sample.get("decoded_image")
        console.print(f"[cyan]({i + 1}/{len(dataset)})[/cyan] {prompt[:80].strip().replace(chr(10), ' ')}")

        try:
            if pil_image is None:
                console.print("[yellow]  ⚠ no decoded_image; skipping[/yellow]")
                continue

            input_ids, mm_token_type_ids, processor_inputs = build_gemma_inputs(
                processor, tokenizer, prompt, pil_image=pil_image
            )

            # Vision encode (timed standalone; not part of the LM prefill time).
            vis_start = perf_counter()
            vision_embeds = run_gemma_vision_encoder(vision_session, processor_inputs)
            vision_encode_time_s = perf_counter() - vis_start
            if vision_embeds is None:
                console.print("[yellow]  ⚠ vision QPC produced no vision_embeds; skipping[/yellow]")
                continue

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
                mm_token_type_ids=mm_token_type_ids,
                vision_embeds=vision_embeds,
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
        all_vis.append(vision_encode_time_s)

        per_sample_rows.append(
            {
                "dataset": "mathvision",
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
                "vision_encode_time_s": round(vision_encode_time_s, 4),
            }
        )

        # Print the generated answer (DLM-drafted tokens in blue, TLM tokens in white),
        # same coloring as dflash_spd_single_prompt.py.
        output_parts = ["  [bold]Output:[/bold] "]
        for tok_id, source in zip(metrics.generated_ids, metrics.generated_sources):
            text = escape(tokenizer.decode([tok_id], skip_special_tokens=True))
            output_parts.append(f"[blue]{text}[/blue]" if source == "dlm" else f"[white]{text}[/white]")
        console.print("".join(output_parts))

        console.print(
            f"  AR={ar:.2f}  DLM={dlm_tps:.1f}  TLM={tlm_tps:.1f}  SPD={spd_tps:.1f} tok/s  "
            f"vision_encode={vision_encode_time_s:.3f}s"
        )

    # ===== SUMMARY =====
    if all_ar:
        w = 50
        print("\n" + "=" * w)
        print("  MathVision SPD Evaluation — Averages")
        print("=" * w)
        print(f"  {'Metric':<32} {'Avg':>6}  {'Min':>6}  {'Max':>6}")
        print("-" * w)
        for name, vals in [
            ("Acceptance Rate (tok/iter)", all_ar),
            ("DLM Throughput  (tok/s)", all_dlm),
            ("TLM Throughput  (tok/s)", all_tlm),
            ("SPD Decode Speed (tok/s)", all_spd),
            ("Vision Encode (s)", all_vis),
        ]:
            print(f"  {name:<32} {np.mean(vals):>6.2f}  {np.min(vals):>6.2f}  {np.max(vals):>6.2f}")
        print("=" * w)
        print(f"  Evaluated {len(all_ar)} / {len(dataset)} samples successfully.")
        print("=" * w + "\n")

        os.makedirs(output_dir, exist_ok=True)
        _write_per_sample_csv(per_sample_rows, os.path.join(output_dir, "mathvision_per_sample.csv"))
        _append_summary_csv(
            {
                "dataset": "mathvision",
                "split": split,
                "num_evaluated": len(all_ar),
                "num_total": len(dataset),
                "avg_acceptance_rate": round(float(np.mean(all_ar)), 4),
                "min_acceptance_rate": round(float(np.min(all_ar)), 4),
                "max_acceptance_rate": round(float(np.max(all_ar)), 4),
                "avg_dlm_tps": round(float(np.mean(all_dlm)), 2),
                "avg_tlm_tps": round(float(np.mean(all_tlm)), 2),
                "avg_spd_tps": round(float(np.mean(all_spd)), 2),
                "avg_vision_encode_s": round(float(np.mean(all_vis)), 4),
                "min_vision_encode_s": round(float(np.min(all_vis)), 4),
                "max_vision_encode_s": round(float(np.max(all_vis)), 4),
            },
            os.path.join(output_dir, "summary.csv"),
        )
    else:
        print("No successful results.")


# ===== ARGUMENT PARSING =====


def parse_args():
    parser = argparse.ArgumentParser(description="DFlash SPD benchmark over MathVision (image+text).")
    parser.add_argument("--tlm_qpc", required=True)
    parser.add_argument("--dlm_qpc", required=True)
    parser.add_argument("--vision_qpc", required=True, help="gemma4 vision-encoder QPC (pixel_values -> vision_embeds).")
    parser.add_argument("--tlm_model_name", required=True)
    parser.add_argument("--dlm_model_name", required=True)
    parser.add_argument("--tlm_devices", nargs="+", type=int, required=True)
    parser.add_argument("--dlm_devices", nargs="+", type=int, required=True)
    parser.add_argument(
        "--vision_devices",
        nargs="+",
        type=int,
        default=None,
        help="Device IDs for the vision QPC (defaults to --tlm_devices). Use separate cards: "
        "TLM+DLM fill their cards, so vision needs its own (e.g. 36 37 38 39).",
    )
    parser.add_argument("--iteration", type=int, default=300)
    parser.add_argument("--ctx_len", type=int, default=4096)
    parser.add_argument("--generation_len", type=int, default=256)
    parser.add_argument("--hf_token", default=None)
    parser.add_argument("--split", default="testmini", help="MathVision split: testmini (304) or test (3040).")
    parser.add_argument("--num_samples", type=int, default=20, help="Number of samples (0 = all).")
    parser.add_argument("--output_dir", default="./results-mathvision", help="Directory for CSV output.")
    return parser.parse_args()


# ===== QPC VALIDATION (mirrors dflash_spd_benchmark.py) =====


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


# ===== MAIN =====


def main():
    args = parse_args()
    num_samples = args.num_samples if args.num_samples > 0 else None

    # Validate device counts against how each QPC was compiled (clear message on mismatch).
    for name, qpc, devs in (
        ("TLM", args.tlm_qpc, args.tlm_devices),
        ("DLM", args.dlm_qpc, args.dlm_devices),
        ("Vision", args.vision_qpc, args.vision_devices if args.vision_devices is not None else args.tlm_devices),
    ):
        nd = _qpc_num_devices(qpc)
        if nd is not None and len(devs) != nd:
            raise SystemExit(
                f"{name} QPC was compiled for {nd} device(s) but got {len(devs)} ({devs}). "
                f"Pass exactly {nd} device ids."
            )

    # Clamp ctx_len to the smaller compiled ctx_len (avoid KV overflow), same as benchmark.
    tlm_ctx = _qpc_ctx_len(args.tlm_qpc)
    dlm_ctx = _qpc_ctx_len(args.dlm_qpc)
    compiled_ctx = min([c for c in (tlm_ctx, dlm_ctx) if c is not None], default=None)
    if compiled_ctx is not None and args.ctx_len > compiled_ctx:
        console.print(
            f"[yellow]⚠ --ctx_len {args.ctx_len} exceeds compiled ctx_len "
            f"(TLM={tlm_ctx}, DLM={dlm_ctx}); clamping to {compiled_ctx}.[/yellow]"
        )
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

    console.print("[bold blue]Loading QAIC inference sessions...[/bold blue]")
    dlm_session = QAICInferenceSession(args.dlm_qpc, args.dlm_devices)
    tlm_session = QAICInferenceSession(args.tlm_qpc, args.tlm_devices)
    vision_devices = args.vision_devices if args.vision_devices is not None else args.tlm_devices
    vision_session = QAICInferenceSession(args.vision_qpc, vision_devices)
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
    console.print(f"TLM input_ids seq_lens (allowed shapes): {tlm_seq_lens}")
    if block_size not in tlm_seq_lens:
        raise SystemExit(
            f"TLM QPC has no decode specialization with seq_len={block_size} (found {tlm_seq_lens}). "
            f"Recompile the TLM with dflash_block_size={block_size}."
        )
    if "vision_embeds" not in tlm_session.input_names:
        raise SystemExit(
            "This benchmark needs a TLM QPC compiled WITH the vision path (vision_embeds input). "
            "The provided --tlm_qpc is text-only; recompile gemma4_example.py with SKIP_VISION=False."
        )

    evaluate_mathvision(
        tokenizer=tokenizer,
        processor=processor,
        dlm_session=dlm_session,
        tlm_session=tlm_session,
        vision_session=vision_session,
        mask_token_id=mask_token_id,
        vocab_size=vocab_size,
        prompt_chunk_size=prompt_chunk_size,
        ctx_len=args.ctx_len,
        block_size=block_size,
        max_iterations=args.iteration,
        hidden_size=hidden_size,
        generation_len=args.generation_len,
        split=args.split,
        num_samples=num_samples,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()

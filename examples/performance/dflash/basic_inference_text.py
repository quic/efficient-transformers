# -----------------------------------------------------------------------------
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------

"""
Single-entry SPD single-prompt runner.

Given a TLM model_name (short name OR full HF repo path) and a prompt, this
script:
  1. Looks up the matching DFlash DLM repo on Hugging Face.
  2. Compiles TLM + DLM QPCs (only the side(s) not provided via
     --tlm_qpc / --dlm_qpc).
  3. Runs SPD single-prompt inference in-process via
     QEfficient.generation.dflash_generation.run_spd_inference_single.

Examples:
    # Compile + run with all defaults
    python basic_inference_text.py --model_name Qwen3-4B \\
        --prompt "Explain speculative decoding in two sentences."

    # Full HF path also accepted
    python basic_inference_text.py --model_name Qwen/Qwen3-4B \\
        --prompt "Hello"

    # Reuse pre-compiled QPCs
    python basic_inference_text.py --model_name Qwen3-4B \\
        --tlm_qpc /path/to/tlm/qpc --dlm_qpc /path/to/dlm/qpc \\
        --prompt "What is 17 * 23?"
"""

import argparse
import os
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from rich.console import Console  # noqa: E402
from rich.markup import escape  # noqa: E402
from transformers import AutoConfig, AutoTokenizer  # noqa: E402

from examples.performance.dflash.utils import (  # noqa: E402
    MODEL_MAP,
    compile_dlm_qpc,
    compile_tlm_qpc,
    get_spd_prompt_chunk_size,
    load_spd_sessions,
    resolve_model_name,
    validate_spd_decode_specialization,
)
from examples.performance.dflash.utils import format_prompt as format_prompt_text  # noqa: E402
from QEfficient.generation.dflash_generation import run_spd_inference_single  # noqa: E402
from QEfficient.utils.logging_utils import logger  # noqa: E402

console = Console()


def parse_device_list(s):
    return [int(x) for x in s.split(",") if x.strip() != ""]


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument(
        "--model_name",
        required=True,
        type=resolve_model_name,
        help="TLM name — either the short key (e.g. 'Qwen3-4B') or "
        "the full HF repo path (e.g. 'Qwen/Qwen3-4B'). "
        f"Supported: {', '.join(MODEL_MAP.keys())}",
    )
    p.add_argument("--prompt", required=True, help="Input prompt text.")
    p.add_argument(
        "--category",
        default="",
        help="Prompt category for formatting (math, coding, reasoning, …).",
    )
    p.add_argument(
        "--format_prompt",
        action="store_true",
        help="If set, wrap the prompt with the category-specific reasoning/coding template before sending to the model. "
        "Off by default — the prompt is used verbatim.",
    )
    p.add_argument(
        "--tlm_hf_path",
        default=None,
        help="Override TLM HF repo (required if mapping has None).",
    )

    # Optional pre-built QPCs (skip compilation)
    p.add_argument("--tlm_qpc", default=None, help="Pre-compiled TLM qpc dir (skip TLM compile).")
    p.add_argument("--dlm_qpc", default=None, help="Pre-compiled DLM qpc dir (skip DLM compile).")

    # Devices / cores
    p.add_argument(
        "--tlm_devices",
        type=parse_device_list,
        default=[40, 41, 42, 43],
        help="Comma-separated device IDs, e.g. '0,1,2,3' or '0'.",
    )
    p.add_argument(
        "--dlm_devices",
        type=parse_device_list,
        default=[40, 41, 42, 43],
        help="Comma-separated device IDs, e.g. '0,1,2,3' or '0'.",
    )
    p.add_argument("--tlm_cores", type=int, default=8)
    p.add_argument("--dlm_cores", type=int, default=8)

    # Compile / run knobs
    p.add_argument("--ctx_len", type=int, default=4096)
    p.add_argument("--prefill_seq_len", type=int, default=128)
    p.add_argument("--generation_len", type=int, default=256)
    p.add_argument("--iteration", type=int, default=300)
    p.add_argument(
        "--batch_size",
        type=int,
        default=1,
        help="Requested batch size. QEfficient currently supports batch_size=1 only; higher values fall back to 1.",
    )

    p.add_argument("--hf_token", default=os.environ.get("HF_TOKEN"))
    return p.parse_args()


def _resolve_batch_size(batch_size):
    if batch_size < 1:
        raise ValueError("--batch_size must be at least 1.")
    if batch_size > 1:
        logger.warning(
            "QEfficient currently does not support batch sizes greater than 1. "
            "Falling back to batch_size=1; use vLLM for higher batch sizes."
        )
        return 1
    return batch_size


def run_text_dflash(
    *,
    model_name: str,
    prompt: str,
    ctx_len: int,
    prefill_seq_len: int,
    generation_len: int,
    iteration: int,
    tlm_devices: list[int],
    dlm_devices: list[int],
    tlm_cores: int = 8,
    dlm_cores: int = 8,
    tlm_hf_path: str | None = None,
    tlm_qpc: str | None = None,
    dlm_qpc: str | None = None,
    category: str = "",
    format_prompt: bool = False,
    hf_token: str | None = None,
    batch_size: int = 1,
    compile_only: bool = False,
    compile_dir: str | None = None,
):
    """Compile/reuse a text target/draft pair and run the shared DFlash loop.

    The standalone CLI and canonical text example share this orchestration.
    ``compile_only`` stops before loading the tokenizer or runtime sessions.
    """
    model_name = resolve_model_name(model_name)
    if model_name not in ("Qwen3-4B", "Qwen3-8B", "Llama-3.1-8B-Instruct"):
        raise ValueError(f"{model_name} requires the vision DFlash example; this runner supports text models only.")
    batch_size = _resolve_batch_size(batch_size)
    tlm_repo_default, dlm_repo = MODEL_MAP[model_name]
    tlm_repo = tlm_hf_path or tlm_repo_default
    if tlm_repo is None:
        raise ValueError(f"No default TLM HF path for '{model_name}'. Supply tlm_hf_path.")

    def stage_compile_dir(stage):
        if compile_dir is None:
            return None
        directory = Path(compile_dir) / stage
        directory.mkdir(parents=True, exist_ok=True)
        return str(directory)

    if tlm_qpc:
        logger.info(f"[skip compile] using provided TLM qpc: {tlm_qpc}")
    else:
        tlm_qpc = compile_tlm_qpc(
            tlm_repo,
            dlm_repo,
            prefill_seq_len=prefill_seq_len,
            ctx_len=ctx_len,
            num_cores=tlm_cores,
            num_devices=len(tlm_devices),
            hf_token=hf_token,
            compile_dir=stage_compile_dir("tlm"),
        )

    if dlm_qpc:
        logger.info(f"[skip compile] using provided DLM qpc: {dlm_qpc}")
    else:
        dlm_qpc = compile_dlm_qpc(
            tlm_repo,
            dlm_repo,
            ctx_len=ctx_len,
            num_cores=dlm_cores,
            num_devices=len(dlm_devices),
            hf_token=hf_token,
            compile_dir=stage_compile_dir("dlm"),
        )
    logger.info(f"TLM qpc        : {tlm_qpc}")
    logger.info(f"DLM qpc        : {dlm_qpc}")
    if compile_only:
        return None

    prompt_text = format_prompt_text(prompt, category) if format_prompt else prompt
    tokenizer = AutoTokenizer.from_pretrained(tlm_repo, token=hf_token, trust_remote_code=True)
    config = AutoConfig.from_pretrained(dlm_repo, token=hf_token, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    dlm_session, tlm_session = load_spd_sessions(tlm_qpc, dlm_qpc, tlm_devices, dlm_devices)
    prompt_chunk_size = get_spd_prompt_chunk_size(tlm_session)
    validate_spd_decode_specialization(tlm_session, config.block_size)

    dflash_config = getattr(config, "dflash_config", None) or config.to_dict().get("dflash_config", {})
    mask_token_id = dflash_config["mask_token_id"] if isinstance(dflash_config, dict) else dflash_config.mask_token_id
    messages = [{"role": "user", "content": prompt_text}]
    formatted_prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
    )
    metrics = run_spd_inference_single(
        prompt_text=formatted_prompt,
        tokenizer=tokenizer,
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
        batch_size=batch_size,
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
    return metrics


def main():
    run_text_dflash(**vars(parse_args()))


if __name__ == "__main__":
    main()

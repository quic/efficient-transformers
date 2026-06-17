# -----------------------------------------------------------------------------
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------

"""MiniMax-M3 decode-throughput benchmark on AI100 (dual-QPC, skip_vision=True).

This script compiles decode-oriented language QPCs (prefill_seq_len=1) and compares
AI100 decode throughput between:
1) FP16
2) FP16 + MXFP6 matmul

It uses the same prompt for both variants and writes an MD report.
"""

from __future__ import annotations

import argparse
import os
import traceback
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter

import numpy as np
import torch
import transformers
from transformers import AutoConfig

from QEfficient import QEFFAutoModelForImageTextToText
from QEfficient.generation.cloud_infer import QAICInferenceSession

MODEL_ID = "MiniMaxAI/MiniMax-M3"
PROMPT = "tell me about yourself, Mini!"
GROUND_TRUTH = """Hello! Great to meet you! 👋

While I appreciate the friendly nickname "Mini," my full name is MiniMax-M3, and I was developed by MiniMax — a global AI foundation model company founded in early 2022. We're committed to advancing the frontiers of AI on the path toward AGI (Artificial General Intelligence).

Here's a bit about me:

🧠 What I am:

A large language model (LLM) — essentially a very capable text-based AI assistant.
My knowledge extends up to January 2026, so I have a broad understanding of the world up to that point.
💡 What I can do:

Answer questions on a wide range of topics (science, history, coding, writing, etc.)
Help with creative tasks like brainstorming, storytelling, and content creation
Assist with analysis, summarization, and problem-solving
Have conversations, explain complex concepts, and help you learn new things
Write code, debug, and discuss technical topics
🤖 What I can't do:

I don't have personal experiences, feelings, or consciousness
I can't browse the internet or access real-time information
I don't have memories of past conversations — each chat starts fresh for me
I'm here to be as helpful as I can, so feel free to ask me anything! What would you like to know or talk about? 😊
"""


@dataclass
class VariantResult:
    name: str
    mxfp6: bool
    compile_ok: bool = False
    run_ok: bool = False
    compile_s: float | None = None
    prefill_s: float | None = None
    decode_s: float | None = None
    decode_tok_s: float | None = None
    generated_text: str = ""
    qpc_path: str = ""
    error: str = ""


def _setup_hf_cache() -> None:
    hf_home = os.environ.setdefault("HF_HOME", "/tmp/hf_minimax_cache")
    hub_cache = os.environ.setdefault("HUGGINGFACE_HUB_CACHE", f"{hf_home}/hub")
    os.environ.setdefault("HF_HUB_CACHE", hub_cache)
    os.environ.setdefault("TRANSFORMERS_CACHE", f"{hf_home}/transformers")
    Path(hf_home).mkdir(parents=True, exist_ok=True)
    Path(hub_cache).mkdir(parents=True, exist_ok=True)
    Path(os.environ["TRANSFORMERS_CACHE"]).mkdir(parents=True, exist_ok=True)


def _build_chat_prompt(tokenizer, prompt: str) -> str:
    messages = [
        {
            "role": "user",
            "content": [{"type": "text", "text": prompt}],
        }
    ]
    if hasattr(tokenizer, "apply_chat_template"):
        try:
            return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        except Exception:
            return prompt
    return prompt


def _extract_lang_qpc_path(qpc_paths: dict) -> str:
    for key in ("lang_decode_qpc_path", "lang_qpc_path", "lang_prefill_qpc_path"):
        if qpc_paths.get(key):
            return qpc_paths[key]
    raise RuntimeError(f"No language QPC path found in compile result: {qpc_paths}")


def _prepare_state_inputs(session: QAICInferenceSession) -> dict[str, np.ndarray]:
    state_inputs: dict[str, np.ndarray] = {}

    # Use basename aliases so callers are robust to onnx-subfunction prefixes.
    input_names = sorted({name.rsplit("/", 1)[-1] for name in session.input_names})

    def _zeros_for(name: str) -> np.ndarray:
        idx = session.binding_index_map[name]
        dtype = session.aic_to_np_dtype_mapping[session.bindings[idx].type]
        shape = tuple(session.buf_dims[idx][1])
        return np.zeros(shape, dtype=dtype)

    for name in input_names:
        if name.startswith(("past_key.", "past_value.", "conv_state.", "recurrent_state.")):
            state_inputs[name] = _zeros_for(name)
        elif name in {"vision_embeds", "deepstack_features", "image_idx", "batch_index", "comp_ctx_lengths"}:
            state_inputs[name] = _zeros_for(name)

    return state_inputs


def _update_state_from_outputs(state_inputs: dict[str, np.ndarray], outputs: dict[str, np.ndarray]) -> None:
    for state_name in list(state_inputs.keys()):
        retained_name = f"{state_name}_RetainedState"
        if retained_name in outputs:
            state_inputs[state_name] = outputs[retained_name]

    if "image_idx" in state_inputs and "image_idx_output" in outputs:
        state_inputs["image_idx"] = outputs["image_idx_output"]


def _simple_similarity(a: str, b: str) -> float:
    a_tokens = set(a.lower().split())
    b_tokens = set(b.lower().split())
    if not a_tokens or not b_tokens:
        return 0.0
    return len(a_tokens & b_tokens) / len(a_tokens | b_tokens)


def _run_variant(
    *,
    qeff_model,
    tokenizer,
    model_id: str,
    prompt: str,
    variant_name: str,
    mxfp6: bool,
    ctx_len: int,
    num_cores: int,
    num_devices: int,
    generation_len: int,
    compile_dir: Path,
) -> VariantResult:
    result = VariantResult(name=variant_name, mxfp6=mxfp6)
    compile_dir.mkdir(parents=True, exist_ok=True)

    try:
        t0 = perf_counter()
        qpc_paths = qeff_model.compile(
            compile_dir=str(compile_dir),
            batch_size=1,
            prefill_seq_len=1,
            ctx_len=ctx_len,
            num_cores=num_cores,
            num_devices=num_devices,
            mxfp6_matmul=mxfp6,
            mxint8_kv_cache=False,
            skip_vision=True,
            split_model_io=True,
            use_onnx_subfunctions=False,
            mos=1,
        )
        result.compile_s = perf_counter() - t0
        result.compile_ok = True
        result.qpc_path = _extract_lang_qpc_path(qpc_paths)
    except Exception:
        result.error = traceback.format_exc()
        return result

    try:
        session = QAICInferenceSession(result.qpc_path, device_ids=list(range(num_devices)))
        state_inputs = _prepare_state_inputs(session)

        prompt_text = _build_chat_prompt(tokenizer, prompt)
        prompt_ids = tokenizer(prompt_text, return_tensors="np")["input_ids"][0].tolist()
        if not prompt_ids:
            raise RuntimeError("Prompt tokenization produced an empty token list.")

        # Prompt ingestion through decode graph token-by-token (decode-only benchmark mode).
        position = 0
        prefill_start = perf_counter()
        outputs = None
        for tok in prompt_ids:
            step_inputs = dict(state_inputs)
            step_inputs["input_ids"] = np.array([[tok]], dtype=np.int64)
            step_inputs["position_ids"] = np.array([[position]], dtype=np.int64)
            outputs = session.run(step_inputs)
            _update_state_from_outputs(state_inputs, outputs)
            position += 1
        result.prefill_s = perf_counter() - prefill_start

        assert outputs is not None
        next_token = int(np.argmax(outputs["logits"]))

        generated_ids: list[int] = []
        decode_start = perf_counter()
        for _ in range(generation_len):
            step_inputs = dict(state_inputs)
            step_inputs["input_ids"] = np.array([[next_token]], dtype=np.int64)
            step_inputs["position_ids"] = np.array([[position]], dtype=np.int64)
            outputs = session.run(step_inputs)
            _update_state_from_outputs(state_inputs, outputs)
            next_token = int(np.argmax(outputs["logits"]))
            generated_ids.append(next_token)
            position += 1
        result.decode_s = perf_counter() - decode_start
        result.decode_tok_s = (generation_len / result.decode_s) if result.decode_s and result.decode_s > 0 else None
        result.generated_text = tokenizer.decode(generated_ids, skip_special_tokens=False)
        result.run_ok = True
    except Exception:
        result.error = traceback.format_exc()

    return result


def _fmt(v: float | None, ndigits: int = 3) -> str:
    if v is None:
        return "NA"
    return f"{v:.{ndigits}f}"


def _write_report(
    report_path: Path,
    *,
    model_id: str,
    prompt: str,
    ctx_len: int,
    num_cores: int,
    num_devices: int,
    generation_len: int,
    results: list[VariantResult],
) -> None:
    report_path.parent.mkdir(parents=True, exist_ok=True)
    run_utc = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")

    lines: list[str] = []
    lines.append("# MiniMax-M3 Decode Throughput Report (AI100)")
    lines.append("")
    lines.append("## Setup")
    lines.append("")
    lines.append(f"- Run timestamp: `{run_utc}`")
    lines.append(f"- Model: `{model_id}`")
    lines.append(f"- Prompt: `{prompt}`")
    lines.append("- Compile mode: `kv_offload=True`, `skip_vision=True`, decode-oriented (`prefill_seq_len=1`)")
    lines.append(f"- Context Length (CL): `{ctx_len}`")
    lines.append(f"- TS (`num_devices`): `{num_devices}`")
    lines.append(f"- AI100 cores: `{num_cores}`")
    lines.append(f"- Generated tokens for throughput: `{generation_len}`")
    lines.append("")
    lines.append("## Ground Truth")
    lines.append("")
    lines.append("```text")
    lines.append(GROUND_TRUTH.strip())
    lines.append("```")
    lines.append("")
    lines.append("## Throughput Comparison")
    lines.append("")
    lines.append(
        "| Variant | Compile | Run | Prefill time (s) | Decode time (s) | Decode tok/s | Output similarity vs GT | Lang QPC | Notes |"
    )
    lines.append("|---|---|---|---:|---:|---:|---:|---|---|")

    fp16_val = None
    mxfp6_val = None
    for r in results:
        sim = _simple_similarity(r.generated_text, GROUND_TRUTH) if r.generated_text else 0.0
        note = "OK" if r.run_ok else (r.error.replace("\n", " ")[:180] if r.error else "Failed")
        lines.append(
            f"| {r.name} | {'PASS' if r.compile_ok else 'FAIL'} | {'PASS' if r.run_ok else 'FAIL'} | "
            f"{_fmt(r.prefill_s)} | {_fmt(r.decode_s)} | {_fmt(r.decode_tok_s)} | {_fmt(sim)} | "
            f"`{r.qpc_path or 'NA'}` | {note} |"
        )
        if r.name == "FP16":
            fp16_val = r.decode_tok_s
        if r.name == "FP16+MXFP6":
            mxfp6_val = r.decode_tok_s

    lines.append("")
    lines.append("## Delta")
    lines.append("")
    if fp16_val is not None and mxfp6_val is not None and fp16_val > 0:
        speedup = mxfp6_val / fp16_val
        pct = (mxfp6_val - fp16_val) * 100.0 / fp16_val
        lines.append(f"- FP16 decode tok/s: `{fp16_val:.3f}`")
        lines.append(f"- FP16+MXFP6 decode tok/s: `{mxfp6_val:.3f}`")
        lines.append(f"- Relative speedup: `{speedup:.3f}x` (`{pct:+.2f}%`)")
    else:
        lines.append("- Unable to compute delta due to failed or missing throughput values.")

    lines.append("")
    lines.append("## Generated Outputs")
    lines.append("")
    for r in results:
        lines.append(f"### {r.name}")
        lines.append("")
        if r.generated_text:
            lines.append("```text")
            lines.append(r.generated_text.strip())
            lines.append("```")
        else:
            lines.append("No generated output captured.")
        lines.append("")

    report_path.write_text("\n".join(lines), encoding="utf-8")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="MiniMax-M3 decode throughput benchmark on AI100")
    parser.add_argument("--model-id", default=MODEL_ID)
    parser.add_argument("--prompt", default=PROMPT)
    parser.add_argument("--ctx-len", type=int, default=1024)
    parser.add_argument("--num-cores", type=int, default=16)
    parser.add_argument("--num-devices", type=int, default=16, help="TS value; TS16 => 16")
    parser.add_argument("--generation-len", type=int, default=128)
    parser.add_argument("--compile-root", type=Path, default=Path("artifacts/minimax_m3_decode_ai100"))
    parser.add_argument(
        "--report-path",
        type=Path,
        default=Path("artifacts/minimax_m3_decode_ai100/minimax_m3_decode_throughput_report.md"),
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    _setup_hf_cache()

    config = AutoConfig.from_pretrained(args.model_id, trust_remote_code=True)
    config.torch_dtype = torch.float16

    tokenizer = transformers.AutoTokenizer.from_pretrained(args.model_id, trust_remote_code=True)
    qeff_model = QEFFAutoModelForImageTextToText.from_pretrained(
        args.model_id,
        kv_offload=True,
        config=config,
        torch_dtype=torch.float16,
        attn_implementation="eager",
        layerwise=False,
        trust_remote_code=True,
    )

    variants = [("FP16", False), ("FP16+MXFP6", True)]
    results: list[VariantResult] = []

    for name, mxfp6 in variants:
        compile_dir = args.compile_root / name.lower().replace("+", "_")
        print(f"[run] {name}: compiling + benchmarking")
        result = _run_variant(
            qeff_model=qeff_model,
            tokenizer=tokenizer,
            model_id=args.model_id,
            prompt=args.prompt,
            variant_name=name,
            mxfp6=mxfp6,
            ctx_len=args.ctx_len,
            num_cores=args.num_cores,
            num_devices=args.num_devices,
            generation_len=args.generation_len,
            compile_dir=compile_dir,
        )
        results.append(result)
        if result.error:
            print(f"[run] {name} failed:\n{result.error}")
        else:
            print(f"[run] {name} decode_tok_s={_fmt(result.decode_tok_s)}")

    _write_report(
        args.report_path,
        model_id=args.model_id,
        prompt=args.prompt,
        ctx_len=args.ctx_len,
        num_cores=args.num_cores,
        num_devices=args.num_devices,
        generation_len=args.generation_len,
        results=results,
    )
    print(f"[run] report: {args.report_path}")


if __name__ == "__main__":
    main()

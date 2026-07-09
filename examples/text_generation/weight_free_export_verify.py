# -----------------------------------------------------------------------------
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------
"""
Weight-Free Export Verification Script
=======================================
End-to-end smoke-test for the weight-free dynamo export pipeline:

  1. Weight-free ONNX export  (no weights in the ONNX file)
  2. FP32 checkpoint materialisation  (one-time, skipped if already done)
  3. QAIC compile
  4. OnnxRuntime inference
  5. PyTorch inference
  6. QPC inference (on-device)
  7. Per-token comparison table  (ORT vs PT vs QPC)

Usage
-----
# Minimal — uses a local safetensors checkpoint
python examples/text_generation/weight_free_export_verify.py \\
    --model_name google/gemma-2b

# With all options
python examples/text_generation/weight_free_export_verify.py \\
    --model_name /path/to/model \\
    --prompt "Hello, how are you?" \\
    --prompt_len 32 \\
    --ctx_len 256 \\
    --num_cores 16 \\
    --output_dir test_models/weightfree \\
    --no_subfunctions
"""

import argparse
import json
import os
import threading
import time
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
import psutil
import torch
from accelerate import init_empty_weights
from safetensors import safe_open
from safetensors.torch import load_file, save_file
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

from QEfficient.exporter.weight_free import _default_weights_roots, load_weight_free_ort_inputs
from QEfficient.exporter.weight_spec import (
    ExternalDataFile,
    load_weight_spec,
    resolve_weight_spec_path,
    save_weight_spec,
)
from QEfficient.transformers.models.modeling_auto import QEFFAutoModelForCausalLM
from QEfficient.utils.run_utils import ApiRunner


# ── Peak RAM tracker ──────────────────────────────────────────────────────────

class PeakRAMTracker:
    """Background-thread peak RSS tracker for the current process + optional children."""

    def __init__(self, interval: float = 0.2):
        self._proc = psutil.Process(os.getpid())
        self._interval = interval
        self._peak_bytes = 0
        self._running = False
        self._thread = None
        self._include_children = False

    def start(self, include_children: bool = False):
        self._include_children = include_children
        self._peak_bytes = self._rss()
        self._running = True
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def _rss(self) -> int:
        try:
            rss = self._proc.memory_info().rss
            if self._include_children:
                for child in self._proc.children(recursive=True):
                    try:
                        rss += child.memory_info().rss
                    except (psutil.NoSuchProcess, psutil.AccessDenied):
                        pass
            return rss
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            return 0

    def _loop(self):
        while self._running:
            rss = self._rss()
            if rss > self._peak_bytes:
                self._peak_bytes = rss
            time.sleep(self._interval)

    def stop(self) -> float:
        """Stop tracking and return peak RAM in GB."""
        self._running = False
        if self._thread:
            self._thread.join(timeout=1.0)
        return self._peak_bytes / 1024 ** 3


# ── FP32 checkpoint materialisation ──────────────────────────────────────────

def convert_checkpoint_to_fp32(onnx_path: Path, weight_spec_path: Path) -> None:
    """Extract weight_spec tensors, cast to FP32, save alongside the ONNX.

    Only the tensors actually referenced by the exported graph are written.
    Already-FP32 shards are referenced in place without copying.
    Idempotent: skips if local FP32 files are already present.
    """
    spec = load_weight_spec(weight_spec_path)
    export_dir = onnx_path.parent
    candidate_roots = _default_weights_roots(weight_spec_path, spec)

    if spec.files and all(
        not Path(f.path).is_absolute() and (export_dir / f.path).is_file()
        for f in spec.files
    ):
        print("Reusing existing local FP32 safetensors.")
        return

    needed: dict[int, set[str]] = {}
    for inp in spec.inputs:
        file_idx = int(inp.location.file)
        needed.setdefault(file_idx, set()).add(inp.location.key)

    sorted_old_idxs = sorted(needed.keys())
    old_to_new = {old: new for new, old in enumerate(sorted_old_idxs)}

    new_files = []
    for old_idx in sorted_old_idxs:
        ext_file = spec.files[old_idx]
        rel_path = Path(ext_file.path)
        abs_path = rel_path if rel_path.is_absolute() else None
        if abs_path is None:
            for root in candidate_roots:
                candidate = root / rel_path
                if candidate.exists():
                    abs_path = candidate
                    break
        if abs_path is None or not abs_path.exists():
            raise FileNotFoundError(f"Cannot resolve external data file: {ext_file.path}")

        keys_needed = needed[old_idx]
        with safe_open(str(abs_path), framework="pt") as f:
            already_fp32 = all(f.get_slice(k).get_dtype() == "F32" for k in keys_needed)

        if already_fp32:
            new_files.append(ExternalDataFile(path=str(abs_path), format="safetensors"))
            print(f"  {abs_path.name}  ({len(keys_needed)} tensors)  ->  referenced in place (already fp32)")
        else:
            tensors = load_file(str(abs_path))
            fp32_tensors = {k: v.to(torch.float32) for k, v in tensors.items() if k in keys_needed}
            out_name = f"model_{old_to_new[old_idx]:04d}.safetensors"
            save_file(fp32_tensors, str(export_dir / out_name))
            new_files.append(ExternalDataFile(path=out_name, format="safetensors"))
            print(f"  {abs_path.name}  ({len(keys_needed)}/{len(tensors)} tensors)  ->  {out_name}  (float32)")

    for inp in spec.inputs:
        inp.location.file = old_to_new[int(inp.location.file)]

    spec.files = new_files
    save_weight_spec(weight_spec_path, spec)
    _sync_embedded_extdata(onnx_path, weight_spec_path)


def _sync_embedded_extdata(onnx_path: Path, weight_spec_path: Path) -> None:
    """Keep the ONNX-embedded com.qti.aisw.extdata metadata in sync with weight_spec.json."""
    updated_json = json.dumps(
        json.loads(weight_spec_path.read_text()), separators=(",", ":"), sort_keys=True
    )
    onnx_model = onnx.load(str(onnx_path), load_external_data=False)
    for entry in onnx_model.metadata_props:
        if entry.key == "com.qti.aisw.extdata":
            entry.value = updated_json
            break
    tmp = onnx_path.with_suffix(onnx_path.suffix + ".tmp")
    onnx.save(onnx_model, str(tmp))
    tmp.replace(onnx_path)


# ── Main ──────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model_name", required=True,
                   help="HuggingFace model ID or local path to a safetensors checkpoint directory.")
    p.add_argument("--prompt", default="What is faith?",
                   help="Prompt string for inference comparison.")
    p.add_argument("--prompt_len", type=int, default=32,
                   help="Padded prompt length (tokens).")
    p.add_argument("--ctx_len", type=int, default=256,
                   help="KV cache context length (tokens).")
    p.add_argument("--num_cores", type=int, default=16,
                   help="Number of QAIC cores for compilation.")
    p.add_argument("--output_dir", default="test_models/weightfree",
                   help="Directory to write ONNX and QPC artefacts.")
    p.add_argument("--no_subfunctions", action="store_true",
                   help="Disable ONNX subfunction extraction (use for models where "
                        "-sub-functions causes a compiler segfault, e.g. MPT).")
    p.add_argument("--mxfp6_matmul", action="store_true",
                   help="Enable MXFP6 matmul quantisation during compile.")
    p.add_argument("--mxint8_kv_cache", action="store_true",
                   help="Enable MXINT8 KV cache quantisation during compile.")
    return p.parse_args()


def main():
    args = parse_args()
    use_subfunctions = not args.no_subfunctions

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    config = AutoConfig.from_pretrained(args.model_name)
    config.dtype = torch.float32
    print(config)

    runner = ApiRunner(
        batch_size=1,
        tokenizer=tokenizer,
        config=config,
        prompt=[args.prompt],
        prompt_len=args.prompt_len,
        ctx_len=args.ctx_len,
    )

    with init_empty_weights():
        meta_model = AutoModelForCausalLM.from_config(config, attn_implementation="eager")

    qeff_model = QEFFAutoModelForCausalLM(
        meta_model,
        pretrained_model_name_or_path=args.model_name,
    )

    # ── Export ────────────────────────────────────────────────────────────────
    export_dir = Path(args.output_dir)
    print("Exporting ...")
    ram_export = PeakRAMTracker()
    ram_export.start()
    t0 = time.perf_counter()
    onnx_path = Path(
        qeff_model.export(
            export_dir=export_dir,
            use_dynamo=True,
            use_onnx_subfunctions=use_subfunctions,
            use_weight_free_export=True,
            offload_pt_weights=False,
        )
    )
    export_elapsed = time.perf_counter() - t0
    print(f"Weight-free export time : {export_elapsed:.3f} sec")
    print(f"Export peak RAM         : {ram_export.stop():.2f} GB")

    weight_spec_path = resolve_weight_spec_path(onnx_path)

    # ── FP32 materialisation ──────────────────────────────────────────────────
    print("Converting checkpoint to FP32 (one-time local materialisation) ...")
    ram_fp32 = PeakRAMTracker()
    ram_fp32.start()
    t0 = time.perf_counter()
    convert_checkpoint_to_fp32(onnx_path, weight_spec_path)
    print(f"fp32 convert time: {time.perf_counter() - t0:.3f} sec")
    print(f"Export peak fp32 RAM  : {ram_fp32.stop():.2f} GB")

    # ── Compile ───────────────────────────────────────────────────────────────
    print("Compiling weight-free ONNX ...")
    ram_compile = PeakRAMTracker()
    ram_compile.start(include_children=True)
    t0 = time.perf_counter()
    qpc_path = qeff_model.compile(
        onnx_path=str(onnx_path),
        compile_dir=str(onnx_path.parent / "qpc"),
        prefill_seq_len=1,
        ctx_len=args.ctx_len,
        num_devices=1,
        num_cores=args.num_cores,
        mxfp6_matmul=args.mxfp6_matmul,
        mxint8_kv_cache=args.mxint8_kv_cache,
        use_dynamo=True,
        use_onnx_subfunctions=use_subfunctions,
        use_weight_free_export=True,
    )
    print(f"Compile time            : {time.perf_counter() - t0:.3f} sec")
    print(f"Compile peak RAM        : {ram_compile.stop():.2f} GB")
    print(f"QPC: {qpc_path}")

    # ── OnnxRT inference ──────────────────────────────────────────────────────
    print("\n--- OnnxRuntime inference ---")
    session = ort.InferenceSession(str(onnx_path))
    ort_inputs = load_weight_free_ort_inputs(weight_spec_path, runner.input_handler.prepare_ort_inputs())
    ort_outputs = runner.run_ort_session(ort_inputs, session)
    ort_outputs = runner.input_handler.update_ort_outputs(ort_outputs)

    ort_generated_ids = []
    for _ in range(1, runner.gen_len):
        ort_generated_ids.append(ort_outputs["logits"].argmax(-1).reshape(-1, 1))
        ort_inputs = runner.input_handler.update_ort_inputs(ort_inputs, ort_outputs)
        ort_inputs = load_weight_free_ort_inputs(weight_spec_path, ort_inputs)
        ort_outputs = runner.run_ort_session(ort_inputs, session)
        ort_outputs = runner.input_handler.update_ort_outputs(ort_outputs)
    ort_generated_ids.append(ort_outputs["logits"].argmax(-1).reshape(-1, 1))
    ort_generated_ids = np.concatenate(ort_generated_ids, axis=1)
    ort_generated_text = tokenizer.batch_decode(ort_generated_ids, skip_special_tokens=True)

    # ── PyTorch inference ─────────────────────────────────────────────────────
    print("\n--- PyTorch inference ---")
    pt_model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        config=AutoConfig.from_pretrained(args.model_name),
        torch_dtype=torch.float32,
        ignore_mismatched_sizes=True,
    ).eval()

    input_ids = tokenizer(args.prompt, return_tensors="pt").input_ids
    with torch.no_grad():
        pt_out = pt_model.generate(
            input_ids,
            max_new_tokens=runner.gen_len,
            do_sample=False,
            temperature=None,
            top_p=None,
        )
    pt_generated_ids = pt_out[:, input_ids.shape[1]:].numpy()
    pt_generated_text = tokenizer.batch_decode(pt_generated_ids, skip_special_tokens=True)

    # ── QPC inference ─────────────────────────────────────────────────────────
    print("\n--- QPC inference ---")
    qpc_generated_ids = None
    qpc_generated_text = None
    try:
        exec_info = qeff_model.generate(
            prompts=[args.prompt],
            tokenizer=tokenizer,
            automation=True,
            generation_len=runner.gen_len,
        )
        qpc_generated_ids = np.asarray(exec_info.generated_ids[0]).reshape(1, -1)
        qpc_generated_text = tokenizer.batch_decode(qpc_generated_ids, skip_special_tokens=True)
        print(exec_info)
    except RuntimeError as exc:
        print(f"Skipping QPC generate: {exc}")

    # ── Results ───────────────────────────────────────────────────────────────
    print(f"\nORT  generated_ids : {ort_generated_ids}")
    print(f"ORT  generated_text: {ort_generated_text}")
    print(f"\nPT   generated_ids : {pt_generated_ids}")
    print(f"PT   generated_text: {pt_generated_text}")
    if qpc_generated_ids is not None:
        print(f"\nQPC  generated_ids : {qpc_generated_ids}")
        print(f"QPC  generated_text: {qpc_generated_text}")
    else:
        print("\nQPC  generated_ids : (skipped — compile failed or no hardware)")

    # ── Per-token match table ─────────────────────────────────────────────────
    print("\n--- Per-token match (ORT vs PT vs QPC) ---")
    max_len = max(
        pt_generated_ids.shape[1],
        qpc_generated_ids.shape[1] if qpc_generated_ids is not None else 0,
    )
    header = f"{'Step':>5}  {'ORT':>8}  {'PT':>8}  {'QPC':>8}  {'ORT==PT':>8}  {'ORT==QPC':>9}"
    print(header)
    print("-" * len(header))
    for i in range(max_len):
        ort_tok = int(ort_generated_ids[0, i]) if i < ort_generated_ids.shape[1] else -1
        pt_tok  = int(pt_generated_ids[0, i])  if i < pt_generated_ids.shape[1]  else -1
        qpc_tok = int(qpc_generated_ids[0, i]) if (qpc_generated_ids is not None and i < qpc_generated_ids.shape[1]) else -1
        ort_eq_pt  = "✓" if ort_tok == pt_tok  else "✗"
        ort_eq_qpc = (
            "✓" if (qpc_generated_ids is not None and ort_tok == qpc_tok)
            else ("N/A" if qpc_generated_ids is None else "✗")
        )
        print(f"{i:>5}  {ort_tok:>8}  {pt_tok:>8}  {qpc_tok:>8}  {ort_eq_pt:>8}  {ort_eq_qpc:>9}")


if __name__ == "__main__":
    main()

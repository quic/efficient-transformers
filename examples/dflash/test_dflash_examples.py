# -----------------------------------------------------------------------------
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------

"""
Test runner for examples/dflash/ that catches issues across the SPD pipeline.

Phases (each can be skipped via flags):
  1. static   — import the modules, validate MODEL_MAP / resolve_model_name,
                check make_models.py wiring without compiling.
  2. compile  — call benchmark.py with TLM/DLM build subprocesses for each model
                using 2 devices and num_cores=16.
  3. dataset  — run benchmark.py against humaneval / gsm8k / math500 for each
                model, parse the printed acceptance rate, compare to the known
                reference within --tolerance.
  4. reuse    — re-run benchmark.py supplying --tlm_qpc/--dlm_qpc to confirm
                the skip-compile path works.
  5. prompt   — run basic_inference.py on a single prompt for each model.

Reference acceptance rates (Avg tok/iter, mxfp6 TLM + mxfp6 DLM, b16/b10):
    Qwen3-4B           humaneval=6.55  gsm8k=6.46  math500=7.96
    Qwen3-8B           humaneval=6.53  gsm8k=6.37  math500=7.88
    Llama-3.1-8B-Instr humaneval=5.01  gsm8k=4.28  math500=4.00

Everything (stdout + stderr) is tee'd to --log; a JSON summary is written next
to it. Exit code is non-zero on any failure.

Usage:
    python test_dflash_examples.py \\
        --tlm_devices 0 1 --dlm_devices 2 3 --num_cores 16 \\
        --models Qwen3-4B Qwen3-8B Llama-3.1-8B-Instruct \\
        --datasets humaneval gsm8k math500 \\
        --log results-test/run.log

Smoke run (no hardware needed):
    python test_dflash_examples.py --only static
"""

import argparse
import json
import os
import re
import subprocess
import sys
import time
import traceback
from pathlib import Path

THIS_DIR = Path(__file__).resolve().parent
REPO_ROOT = THIS_DIR.parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(THIS_DIR))


# ─────────────────────────────────────────────────────────────────────────────
# Reference acceptance rates (Avg tok/iter)
# Source: res_Data_fixing_TLM_mxfp6_DLM_mxfp6_ALL_NEW.log
# ─────────────────────────────────────────────────────────────────────────────
REFERENCE_RATES = {
    "Qwen3-4B": {"humaneval": 6.55, "gsm8k": 6.46, "math500": 7.96},
    "Qwen3-8B": {"humaneval": 6.53, "gsm8k": 6.37, "math500": 7.88},
    "Llama-3.1-8B-Instruct": {"humaneval": 5.01, "gsm8k": 4.28, "math500": 4.00},
}

# ────────────────────────────────────────────────────────────────────────────
# Output parsing — benchmark.py prints a table containing:
#   "  Acceptance Rate (tok/iter)       6.55    3.55   10.57"
# ─────────────────────────────────────────────────────────────────────────────
ACCEPT_RE = re.compile(r"Acceptance Rate \(tok/iter\)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)")
QPC_TLM_RE = re.compile(r"^TLM_QPC=(.+)$", re.MULTILINE)
QPC_DLM_RE = re.compile(r"^DLM_QPC=(.+)$", re.MULTILINE)


# ─────────────────────────────────────────────────────────────────────────────
# Tee — duplicate writes to stdout and the log file.
# ─────────────────────────────────────────────────────────────────────────────
class Tee:
    def __init__(self, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self._fp = open(path, "a", buffering=1)
        self._stdout = sys.stdout

    def write(self, s):
        self._stdout.write(s)
        self._fp.write(s)

    def flush(self):
        self._stdout.flush()
        self._fp.flush()

    def close(self):
        self._fp.close()


def log(msg):
    print(f"[test] {msg}", flush=True)


def section(title):
    line = "=" * 78
    print(f"\n{line}\n {title}\n{line}", flush=True)


# ─────────────────────────────────────────────────────────────────────────────
# Test result tracking
# ─────────────────────────────────────────────────────────────────────────────
class Results:
    def __init__(self):
        self.cases = []  # list of {name, status, detail, duration_s}

    def add(self, name, status, detail="", duration_s=0.0):
        self.cases.append({"name": name, "status": status, "detail": detail, "duration_s": round(duration_s, 2)})
        marker = {"PASS": "✓", "FAIL": "✗", "SKIP": "·"}.get(status, "?")
        log(f"  {marker} [{status}] {name}  ({duration_s:.1f}s)  {detail}")

    @property
    def failed(self):
        return [c for c in self.cases if c["status"] == "FAIL"]

    @property
    def passed(self):
        return [c for c in self.cases if c["status"] == "PASS"]

    def write_json(self, path):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as fp:
            json.dump(
                {
                    "summary": {
                        "total": len(self.cases),
                        "passed": len(self.passed),
                        "failed": len(self.failed),
                    },
                    "cases": self.cases,
                },
                fp,
                indent=2,
            )


# ─────────────────────────────────────────────────────────────────────────────
# Subprocess wrapper — stream output line-by-line so the user sees progress
# during long runs. Returns (rc, combined output, duration_s).
# ─────────────────────────────────────────────────────────────────────────────
def run_cmd(cmd, env=None, timeout=None):
    log(f"$ {' '.join(str(c) for c in cmd)}")
    t0 = time.time()
    proc = subprocess.Popen(
        [str(c) for c in cmd],
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    chunks = []
    last_heartbeat = t0
    try:
        for line in proc.stdout:
            chunks.append(line)
            sys.stdout.write(line)
            now = time.time()
            if now - last_heartbeat > 60:
                log(f"[heartbeat] still running ({int(now - t0)}s elapsed)")
                last_heartbeat = now
            if timeout is not None and (now - t0) > timeout:
                proc.kill()
                raise subprocess.TimeoutExpired(cmd, timeout)
        proc.wait()
    except KeyboardInterrupt:
        proc.kill()
        raise
    dt = time.time() - t0
    return proc.returncode, "".join(chunks), dt


# ─────────────────────────────────────────────────────────────────────────────
# Phase 1 — static checks (no hardware needed)
# ─────────────────────────────────────────────────────────────────────────────
def phase_static(args, results):
    section("Phase 1 / static")

    # --- import benchmark and check MODEL_MAP / resolve_model_name --------
    t0 = time.time()
    try:
        import benchmark  # noqa: I001 - sys.path adjusted at module top

        for short in args.models:
            if short not in benchmark.MODEL_MAP:
                raise AssertionError(f"{short} missing from MODEL_MAP")
        # resolve_model_name handles short / HF / basename
        for short, (hf_repo, _) in benchmark.MODEL_MAP.items():
            assert benchmark.resolve_model_name(short) == short
            if hf_repo is not None:
                assert benchmark.resolve_model_name(hf_repo) == short
                assert benchmark.resolve_model_name(hf_repo.split("/", 1)[-1]) == short
        # rejects unknown name
        try:
            benchmark.resolve_model_name("DoesNotExist-99T")
            raise AssertionError("resolve_model_name accepted unknown name")
        except argparse.ArgumentTypeError:
            pass
        results.add("static.benchmark.MODEL_MAP+resolver", "PASS", duration_s=time.time() - t0)
    except Exception:
        results.add(
            "static.benchmark.MODEL_MAP+resolver",
            "FAIL",
            traceback.format_exc().splitlines()[-1],
            duration_s=time.time() - t0,
        )

    # --- import basic_inference / make_models / utils -----------------
    for mod in ["basic_inference", "make_models", "utils", "dflash_spd_benchmark", "dflash_spd_single_prompt"]:
        t0 = time.time()
        try:
            __import__(mod)
            results.add(f"static.import.{mod}", "PASS", duration_s=time.time() - t0)
        except Exception:
            results.add(
                f"static.import.{mod}",
                "FAIL",
                traceback.format_exc().splitlines()[-1],
                duration_s=time.time() - t0,
            )

    # --- noise embeddings present for each requested model ----------------
    for short in args.models:
        path = THIS_DIR / "noise_embedding" / f"{short}_noise_embeds.npy"
        results.add(
            f"static.noise_embed.{short}",
            "PASS" if path.exists() else "FAIL",
            f"path={path}",
        )


# ─────────────────────────────────────────────────────────────────────────────
# Phase 2/3 — compile + dataset run via benchmark.py
# Returns (tlm_qpc, dlm_qpc, acceptance_rate_avg) on success, or None.
# ─────────────────────────────────────────────────────────────────────────────
def run_benchmark(args, model, dataset, results, tlm_qpc=None, dlm_qpc=None, phase="dataset"):
    cmd = [
        sys.executable,
        str(THIS_DIR / "benchmark.py"),
        "--model_name",
        model,
        "--dataset",
        dataset,
        "--tlm_devices",
        *[str(d) for d in args.tlm_devices],
        "--dlm_devices",
        *[str(d) for d in args.dlm_devices],
        "--tlm_cores",
        str(args.num_cores),
        "--dlm_cores",
        str(args.num_cores),
        "--ctx_len",
        str(args.ctx_len),
        "--prefill_seq_len",
        str(args.prefill_seq_len),
        "--generation_len",
        str(args.generation_len),
        "--iteration",
        str(args.iteration),
    ]
    if args.num_samples > 0:
        cmd += ["--num_samples", str(args.num_samples)]
    if args.hf_token:
        cmd += ["--hf_token", args.hf_token]
    if tlm_qpc:
        cmd += ["--tlm_qpc", tlm_qpc]
    if dlm_qpc:
        cmd += ["--dlm_qpc", dlm_qpc]

    label = f"{phase}.{model}.{dataset}"
    rc, out, dt = run_cmd(cmd, timeout=args.dataset_timeout_s)

    if rc != 0:
        results.add(label, "FAIL", f"rc={rc}", duration_s=dt)
        return None

    # capture qpc paths if compile happened
    tlm_match = QPC_TLM_RE.search(out)
    dlm_match = QPC_DLM_RE.search(out)
    new_tlm = tlm_match.group(1).strip() if tlm_match else tlm_qpc
    new_dlm = dlm_match.group(1).strip() if dlm_match else dlm_qpc

    accept = ACCEPT_RE.search(out)
    if accept is None:
        results.add(
            label,
            "FAIL",
            "no acceptance-rate line in output",
            duration_s=dt,
        )
        return None

    avg = float(accept.group(1))
    expected = REFERENCE_RATES.get(model, {}).get(dataset)
    if expected is None:
        results.add(
            label,
            "PASS",
            f"avg={avg:.2f} (no reference)",
            duration_s=dt,
        )
    else:
        delta_pct = abs(avg - expected) / expected * 100
        ok = delta_pct <= args.tolerance_pct
        results.add(
            label,
            "PASS" if ok else "FAIL",
            f"avg={avg:.2f} expected={expected:.2f} Δ={delta_pct:.1f}% tol={args.tolerance_pct:.1f}%",
            duration_s=dt,
        )
    return new_tlm, new_dlm, avg


# ─────────────────────────────────────────────────────────────────────────────
# Phase 5 — single-prompt via basic_inference.py
# ─────────────────────────────────────────────────────────────────────────────
def run_basic_inference(args, model, tlm_qpc, dlm_qpc, results):
    cmd = [
        sys.executable,
        str(THIS_DIR / "basic_inference.py"),
        "--model_name",
        model,
        "--prompt",
        args.test_prompt,
        "--tlm_devices",
        *[str(d) for d in args.tlm_devices],
        "--dlm_devices",
        *[str(d) for d in args.dlm_devices],
        "--tlm_cores",
        str(args.num_cores),
        "--dlm_cores",
        str(args.num_cores),
        "--ctx_len",
        str(args.ctx_len),
        "--prefill_seq_len",
        str(args.prefill_seq_len),
        "--generation_len",
        "64",
        "--iteration",
        "20",
    ]
    if tlm_qpc:
        cmd += ["--tlm_qpc", tlm_qpc]
    if dlm_qpc:
        cmd += ["--dlm_qpc", dlm_qpc]
    if args.hf_token:
        cmd += ["--hf_token", args.hf_token]

    rc, _, dt = run_cmd(cmd, timeout=args.prompt_timeout_s)
    results.add(
        f"prompt.basic_inference.{model}",
        "PASS" if rc == 0 else "FAIL",
        f"rc={rc}",
        duration_s=dt,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Phase orchestration
# ─────────────────────────────────────────────────────────────────────────────
def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--models", nargs="+", default=["Qwen3-4B", "Qwen3-8B", "Llama-3.1-8B-Instruct"])
    p.add_argument(
        "--datasets", nargs="+", default=["humaneval", "gsm8k", "math500"], choices=["humaneval", "gsm8k", "math500"]
    )
    p.add_argument("--tlm_devices", nargs="+", type=int, default=[0, 1])
    p.add_argument("--dlm_devices", nargs="+", type=int, default=[2, 3])
    p.add_argument("--num_cores", type=int, default=16)
    p.add_argument("--ctx_len", type=int, default=4096)
    p.add_argument("--prefill_seq_len", type=int, default=128)
    p.add_argument("--generation_len", type=int, default=1024)
    p.add_argument("--iteration", type=int, default=300)
    p.add_argument("--num_samples", type=int, default=0, help="0 = full dataset (matches the reference log).")
    p.add_argument("--tolerance_pct", type=float, default=5.0, help="Acceptance-rate tolerance (percent of expected).")
    p.add_argument("--test_prompt", default="Explain speculative decoding in two sentences.")
    p.add_argument("--hf_token", default=os.environ.get("HF_TOKEN"))
    p.add_argument("--log", default=str(THIS_DIR / "results-test" / "run.log"))
    p.add_argument("--summary_json", default=None, help="Defaults to <log>.summary.json")
    p.add_argument(
        "--only",
        nargs="+",
        default=None,
        choices=["static", "compile", "dataset", "reuse", "prompt"],
        help="If set, run only these phases.",
    )
    p.add_argument("--skip", nargs="+", default=[], choices=["static", "compile", "dataset", "reuse", "prompt"])
    p.add_argument("--dataset_timeout_s", type=int, default=60 * 90)
    p.add_argument("--prompt_timeout_s", type=int, default=60 * 30)
    p.add_argument(
        "--prebuilt_qpcs",
        default=None,
        help='JSON: {"<model>": {"tlm": "<path>", "dlm": "<path>"}}. Skips the compile phase for those models.',
    )
    args = p.parse_args()

    sys.stdout = Tee(args.log)
    summary_json = args.summary_json or args.log + ".summary.json"

    enabled = set(args.only) if args.only else {"static", "compile", "dataset", "reuse", "prompt"}
    enabled -= set(args.skip)

    results = Results()
    log(f"models={args.models}  datasets={args.datasets}")
    log(f"tlm_devices={args.tlm_devices}  dlm_devices={args.dlm_devices}  num_cores={args.num_cores}")
    log(f"phases={sorted(enabled)}  log={args.log}")

    prebuilt = {}
    if args.prebuilt_qpcs:
        with open(args.prebuilt_qpcs) as fp:
            prebuilt = json.load(fp)

    if "static" in enabled:
        phase_static(args, results)

    qpcs_per_model = {}  # model -> (tlm_qpc, dlm_qpc)

    if "dataset" in enabled or "compile" in enabled:
        section("Phase 2/3 / compile + dataset")
        for model in args.models:
            tlm_qpc = prebuilt.get(model, {}).get("tlm")
            dlm_qpc = prebuilt.get(model, {}).get("dlm")
            for dataset in args.datasets:
                res = run_benchmark(args, model, dataset, results, tlm_qpc, dlm_qpc, phase="dataset")
                if res is None:
                    continue
                tlm_qpc, dlm_qpc, _ = res  # reuse compiled qpcs across datasets
            qpcs_per_model[model] = (tlm_qpc, dlm_qpc)

    if "reuse" in enabled and qpcs_per_model:
        section("Phase 4 / qpc reuse (skip compile)")
        # Re-run only the first dataset with explicit --tlm_qpc/--dlm_qpc to
        # confirm benchmark.py honors them and skips compile.
        for model, (tlm_qpc, dlm_qpc) in qpcs_per_model.items():
            if not (tlm_qpc and dlm_qpc):
                results.add(f"reuse.{model}", "SKIP", "no qpcs available")
                continue
            run_benchmark(args, model, args.datasets[0], results, tlm_qpc, dlm_qpc, phase="reuse")

    if "prompt" in enabled:
        section("Phase 5 / single-prompt (basic_inference.py)")
        for model in args.models:
            tlm_qpc, dlm_qpc = qpcs_per_model.get(model, (None, None))
            run_basic_inference(args, model, tlm_qpc, dlm_qpc, results)

    section("Summary")
    log(f"total={len(results.cases)}  passed={len(results.passed)}  failed={len(results.failed)}")
    for c in results.failed:
        log(f"  FAIL  {c['name']}  {c['detail']}")
    results.write_json(summary_json)
    log(f"summary written to {summary_json}")

    sys.stdout.flush()
    raise SystemExit(0 if not results.failed else 1)


if __name__ == "__main__":
    main()

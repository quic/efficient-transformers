import argparse
import contextlib
import gc
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import onnx

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from benchmark_onnx_export_memory import EXPORT_SNIPPET, MODEL_SPECS, mode_env_value  # noqa: E402


def value_info_signature(value_info):
    tensor_type = value_info.type.tensor_type
    shape = []
    if tensor_type.HasField("shape"):
        for dim in tensor_type.shape.dim:
            if dim.HasField("dim_value"):
                shape.append(dim.dim_value)
            elif dim.HasField("dim_param"):
                shape.append(dim.dim_param)
            else:
                shape.append("?")
    return {
        "name": value_info.name,
        "elem_type": tensor_type.elem_type,
        "shape": shape,
    }


def graph_signature(onnx_path):
    model = onnx.load(onnx_path, load_external_data=False)
    initializer_names = {initializer.name for initializer in model.graph.initializer}
    return {
        "inputs": [value_info_signature(value_info) for value_info in model.graph.input],
        "outputs": [value_info_signature(value_info) for value_info in model.graph.output],
        "initializer_count": len(model.graph.initializer),
        "external_initializer_count": sum(
            1 for initializer in model.graph.initializer if initializer.data_location == onnx.TensorProto.EXTERNAL
        ),
        "graph_inputs_that_are_initializers": sorted(
            value_info.name for value_info in model.graph.input if value_info.name in initializer_names
        ),
    }


@contextlib.contextmanager
def temp_home(prefix):
    root = Path(tempfile.mkdtemp(prefix=prefix))
    try:
        yield root
    finally:
        shutil.rmtree(root, ignore_errors=True)


def export_one(model_key, spec, mode, python_exe, repo_root, keep_logs):
    with temp_home(f"qeff_io_{model_key}_{mode}_") as workdir:
        env = os.environ.copy()
        env["PYTHONPATH"] = str(repo_root)
        env["QEFF_HOME"] = str(workdir / "qeff_home")
        env["QEFF_EXPORT_ONLY"] = "1"
        env["TOKENIZERS_PARALLELISM"] = "false"
        lowmem_env = mode_env_value(mode)
        if lowmem_env is None:
            env.pop("QEFF_LOW_MEMORY_ONNX_EXPORT", None)
        else:
            env["QEFF_LOW_MEMORY_ONNX_EXPORT"] = lowmem_env
        export_dir = workdir / "export"
        cmd = [python_exe, "-c", EXPORT_SNIPPET, model_key, spec["id"], spec["task"], str(export_dir)]
        proc = subprocess.run(cmd, cwd=repo_root, env=env, text=True, capture_output=True)
        if keep_logs:
            log_dir = repo_root / "docs" / "memory_profile" / "io_logs"
            log_dir.mkdir(parents=True, exist_ok=True)
            (log_dir / f"{model_key}_{mode}.stderr.log").write_text(proc.stderr)
            (log_dir / f"{model_key}_{mode}.stdout.log").write_text(proc.stdout)
        payload = None
        for line in proc.stdout.splitlines()[::-1]:
            line = line.strip()
            if line.startswith("{") and "onnx_path" in line:
                payload = json.loads(line)
                break
        result = {
            "mode": mode,
            "returncode": proc.returncode,
            "error_tail": proc.stderr[-4000:] if proc.returncode != 0 else "",
        }
        if payload and proc.returncode == 0:
            onnx_path = Path(payload["onnx_path"])
            result["onnx_path_name"] = onnx_path.name
            result["signature"] = graph_signature(onnx_path)
        del proc
        gc.collect()
        return result


def compare_pair(legacy, after):
    if legacy.get("returncode") != 0 or after.get("returncode") != 0:
        return {
            "status": "export_failed",
            "inputs_match": False,
            "outputs_match": False,
            "extra_inputs": [],
            "missing_inputs": [],
            "output_diff": [],
        }
    legacy_sig = legacy["signature"]
    after_sig = after["signature"]
    legacy_inputs = legacy_sig["inputs"]
    after_inputs = after_sig["inputs"]
    legacy_outputs = legacy_sig["outputs"]
    after_outputs = after_sig["outputs"]
    legacy_input_names = [item["name"] for item in legacy_inputs]
    after_input_names = [item["name"] for item in after_inputs]
    legacy_output_names = [item["name"] for item in legacy_outputs]
    after_output_names = [item["name"] for item in after_outputs]
    inputs_match = legacy_inputs == after_inputs
    outputs_match = legacy_outputs == after_outputs
    extra_inputs = [name for name in after_input_names if name not in legacy_input_names]
    missing_inputs = [name for name in legacy_input_names if name not in after_input_names]
    extra_outputs = [name for name in after_output_names if name not in legacy_output_names]
    missing_outputs = [name for name in legacy_output_names if name not in after_output_names]
    status = (
        "match"
        if inputs_match and outputs_match and not after_sig["graph_inputs_that_are_initializers"]
        else "mismatch"
    )
    return {
        "status": status,
        "inputs_match": inputs_match,
        "outputs_match": outputs_match,
        "legacy_input_count": len(legacy_inputs),
        "after_input_count": len(after_inputs),
        "legacy_output_count": len(legacy_outputs),
        "after_output_count": len(after_outputs),
        "extra_inputs": extra_inputs,
        "missing_inputs": missing_inputs,
        "extra_outputs": extra_outputs,
        "missing_outputs": missing_outputs,
        "after_graph_inputs_that_are_initializers": after_sig["graph_inputs_that_are_initializers"],
        "after_external_initializer_count": after_sig["external_initializer_count"],
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", nargs="+", default=list(MODEL_SPECS))
    parser.add_argument("--out", default="docs/memory_profile/onnx_io_signature_results.json")
    parser.add_argument("--keep-logs", action="store_true")
    parser.add_argument("--continue-on-error", action="store_true")
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[2]
    results = []
    for model_key in args.models:
        spec = MODEL_SPECS[model_key]
        print(f"[io] {model_key}", flush=True)
        legacy = export_one(model_key, spec, "legacy", sys.executable, repo_root, args.keep_logs)
        after = export_one(model_key, spec, "after", sys.executable, repo_root, args.keep_logs)
        comparison = compare_pair(legacy, after)
        row = {
            "model_key": model_key,
            "model_id": spec["id"],
            "task": spec["task"],
            "legacy": legacy,
            "after": after,
            "comparison": comparison,
        }
        results.append(row)
        print(json.dumps({"model_key": model_key, **comparison}, sort_keys=True), flush=True)
        if comparison["status"] not in {"match", "export_failed"} and not args.continue_on_error:
            raise SystemExit(1)
        if comparison["status"] == "export_failed" and not args.continue_on_error:
            raise SystemExit(1)
    out = repo_root / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"results": results}, indent=2, sort_keys=True))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()

import argparse
import os
import re
import signal
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

# =====================================================
# CONFIG
# =====================================================

MAX_RETRIES = 1  # retries don't help for long compiles
RETRY_SLEEP = 5
TIMEOUT = 90 * 60  # 90 minutes

# =====================================================
# WORKER CONFIG (CPU-BASED)
# =====================================================

MAX_WORKERS = 8
NUM_DEVICES = 1  # default, will be overridden by CLI


# =====================================================
# DISCOVERY
# =====================================================


def _discover_onnx_jobs(base_onnx_dir: str):
    # agent: defer discovery to runtime and require explicit export path.
    onnx_jobs = []
    base_dir_path = Path(base_onnx_dir)
    layerwise_dir = base_dir_path / "onnx_layerwise_tmp"
    if layerwise_dir.is_dir():
        scan_dir = layerwise_dir
    elif base_dir_path.is_dir():
        scan_dir = base_dir_path
    else:
        raise RuntimeError(f"BASE_ONNX_DIR does not exist: {base_onnx_dir}")

    layer_dir_pat = re.compile(r"^layer_(\d+)_(\d+)$")
    for layer_dir in sorted(scan_dir.iterdir()):
        if not layer_dir.is_dir():
            continue

        m = layer_dir_pat.match(layer_dir.name)
        if not m:
            continue

        layer_start = int(m.group(1))
        layer_end = int(m.group(2))
        if layer_end <= layer_start:
            continue

        layer_indices = [str(i) for i in range(layer_start, layer_end)]
        layer_window = (layer_start, layer_end)

        # build device_group from NUM_DEVICES
        device_group = ",".join(str(i) for i in range(NUM_DEVICES))

        for f in layer_dir.iterdir():
            if f.name.startswith("DeepseekV3ForCausalLM_layer_tmp_") and f.suffix == ".onnx":
                onnx_jobs.append((f, layer_dir, layer_window, layer_indices, device_group))

    if not onnx_jobs:
        raise RuntimeError(f"No valid ONNX files found under: {scan_dir}")

    return onnx_jobs


# =====================================================
# CUSTOM IO YAML WRITER
# =====================================================


def write_custom_io_yaml(path: Path, indices):
    with open(path, "w") as fp:
        # agent: write cache entries for all layers in each discovered window.
        for idx in indices:
            fp.write(f" - IOName: k_pe.{idx}\n")
            fp.write("   Precision: mxint8\n\n")
            fp.write(f" - IOName: compressed_kv.{idx}\n")
            fp.write("   Precision: mxint8\n\n")

        for idx in indices:
            fp.write(f" - IOName: k_pe.{idx}_RetainedState\n")
            fp.write("   Precision: mxint8\n\n")
            fp.write(f" - IOName: compressed_kv.{idx}_RetainedState\n")
            fp.write("   Precision: mxint8\n\n")


# =====================================================
# COMPILE FUNCTION
# =====================================================


def compile_one(job):
    onnx_path, layer_dir, layer_window, layer_indices, device_group = job

    layer_tag = onnx_path.stem.replace("DeepseekV3ForCausalLM_layer_tmp_", "")

    qpc_dir = layer_dir / f"qpc_{layer_tag}"
    log_file = layer_dir / f"qpc_{layer_tag}.log"
    qpc_dir.mkdir(parents=True, exist_ok=True)

    custom_io_yaml = layer_dir / "custom_io_fp16.yaml"
    if not custom_io_yaml.exists():
        write_custom_io_yaml(custom_io_yaml, layer_indices)

    cmd = [
        "python",
        "-m",
        "QEfficient.cloud.compile",
        "--onnx_path",
        str(onnx_path),
        "--qpc-path",
        str(qpc_dir),
        "--batch_size",
        "1",
        "--prompt_len",
        "1",
        "--ctx_len",
        "128",
        "--mxfp6",
        "mxint8_kv_cache",
        "--num_cores",
        "16",
        "--device_group",
        device_group,
        "--mos",
        "1",
        "--aic_enable_depth_first",
        f"-custom-IO-list-file={custom_io_yaml}",
    ]

    total_start = time.time()
    last_status = "FAILED"

    for attempt in range(1, MAX_RETRIES + 1):
        print(
            f"[START ] layer {layer_window[0]}_{layer_window[1]} "
            f"device {device_group} (attempt {attempt}/{MAX_RETRIES})"
        )

        proc = None
        try:
            with open(log_file, "a") as lf:
                lf.write(f"\n===== ATTEMPT {attempt} =====\n")
                proc = subprocess.Popen(
                    cmd,
                    stdout=lf,
                    stderr=subprocess.STDOUT,
                    start_new_session=True,
                )
                proc.wait(timeout=TIMEOUT)

            if proc.returncode == 0:
                last_status = "OK"
                break
            else:
                last_status = f"FAILED(rc={proc.returncode})"

        except subprocess.TimeoutExpired:
            last_status = "TIMEOUT"
            if proc:
                os.killpg(proc.pid, signal.SIGTERM)
            break  # do not retry timeouts

        except KeyboardInterrupt:
            if proc:
                os.killpg(proc.pid, signal.SIGTERM)
            raise

        except Exception as e:
            last_status = f"ERROR({e})"
            if proc:
                os.killpg(proc.pid, signal.SIGTERM)
            break

        time.sleep(RETRY_SLEEP)

    total_elapsed = time.time() - total_start

    print(f"[DONE  ] layer {layer_window[0]}_{layer_window[1]} {last_status} | {total_elapsed:.1f}s")

    return layer_tag, last_status, total_elapsed


# =====================================================
# MAIN
# =====================================================


def run_compile_layerwise(base_onnx_dir: str):
    onnx_jobs = _discover_onnx_jobs(base_onnx_dir)
    print(f"MAX_WORKERS set to     : {MAX_WORKERS}")
    print(f"NUM_DEVICES            : {NUM_DEVICES}")
    print(f"Found {len(onnx_jobs)} ONNX files\n")

    start_time = time.time()
    results = []
    interrupted = False

    try:
        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            futures = [executor.submit(compile_one, job) for job in onnx_jobs]

            for fut in as_completed(futures):
                results.append(fut.result())

    except KeyboardInterrupt:
        interrupted = True
        print("\n[INTERRUPT] KeyboardInterrupt received")

    finally:
        total_time = time.time() - start_time

        success = sum(1 for _, s, _ in results if s == "OK")
        failed = sum(1 for _, s, _ in results if s != "OK")
        completed = len(results)
        pending = len(onnx_jobs) - completed

        print("\n============================================")
        print(f"TOTAL FILES   : {len(onnx_jobs)}")
        print(f"COMPLETED     : {completed}")
        print(f"SUCCESS       : {success}")
        print(f"FAILED        : {failed}")
        print(f"PENDING       : {pending}")
        print(f"TOTAL TIME    : {total_time:.1f} seconds")
        print(f"INTERRUPTED   : {interrupted}")
        print("============================================")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compile layerwise ONNX windows into QPC artifacts.")
    parser.add_argument("--base-onnx-dir", required=True, help="Export root containing onnx_layerwise_tmp/")
    parser.add_argument("--num-devices", type=int, default=1, help="Number of devices to use (e.g., 2 -> device_group=0,1)")

    args = parser.parse_args()

    NUM_DEVICES = args.num_devices

    run_compile_layerwise(args.base_onnx_dir)

import os
import signal
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

# =====================================================
# USER INPUT
# =====================================================

BASE_ONNX_DIR = (
    "/home/abhishek/.cache/qeff_models/DeepseekV3ForCausalLM/DeepseekV3ForCausalLM-54e8bf6964765090/onnx_layerwise_tmp"
)

# =====================================================
# CONFIG
# =====================================================

MAX_DEVICE_PROBE = 55
MAX_RETRIES = 1  # retries don’t help for long compiles
RETRY_SLEEP = 5
TIMEOUT = 90 * 60  # 90 minutes

# =====================================================
# WORKER CONFIG (CPU-BASED)
# =====================================================

MAX_WORKERS = 8

# print(f"CPU cores detected     : {CPU_CORES}")
print(f"MAX_WORKERS set to     : {MAX_WORKERS}\n")

# =====================================================
# DISCOVER ONNX FILES
# =====================================================

onnx_jobs = []

for layer_dir in sorted(Path(BASE_ONNX_DIR).iterdir()):
    if not layer_dir.is_dir():
        continue

    parts = layer_dir.name.split("_")
    if len(parts) != 3:
        continue

    layer_indices = [parts[1], parts[2]]

    for f in layer_dir.iterdir():
        if f.name.startswith("DeepseekV3ForCausalLM_layer_tmp_") and f.suffix == ".onnx":
            # device_group fixed to single device "0"
            onnx_jobs.append((f, layer_dir, layer_indices, "0"))

if not onnx_jobs:
    raise RuntimeError("No valid ONNX files found")

print(f"Found {len(onnx_jobs)} ONNX files\n")

# =====================================================
# CUSTOM IO YAML WRITER
# =====================================================


def write_custom_io_yaml(path: Path, indices):
    with open(path, "w") as fp:
        # for idx in indices:
        fp.write(f" - IOName: k_pe.{indices[0]}\n")
        fp.write("   Precision: mxint8\n\n")
        fp.write(f" - IOName: compressed_kv.{indices[0]}\n")
        fp.write("   Precision: mxint8\n\n")

        # for idx in indices:
        fp.write(f" - IOName: k_pe.{indices[0]}_RetainedState\n")
        fp.write("   Precision: mxint8\n\n")
        fp.write(f" - IOName: compressed_kv.{indices[0]}_RetainedState\n")
        fp.write("   Precision: mxint8\n\n")


# =====================================================
# COMPILE FUNCTION
# =====================================================


def compile_one(job):
    onnx_path, layer_dir, layer_indices, device_group = job

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
            f"[START ] layer {layer_indices[0]}_{layer_indices[1]} "
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
            break

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

    print(f"[DONE  ] layer {layer_indices[0]}_{layer_indices[1]} {last_status} | {total_elapsed:.1f}s")

    return layer_tag, last_status, total_elapsed


# =====================================================
# MAIN
# =====================================================


def main():
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
    main()

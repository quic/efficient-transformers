import argparse
import os
import re
import signal
import subprocess
import time
from pathlib import Path

# =====================================================
# CONFIG
# =====================================================

TIMEOUT = 500 * 60  # 90 minutes
DEVICE_GROUP = ",".join(str(i) for i in range(1))  # 0..15

# =====================================================
# CUSTOM IO YAML WRITER
# =====================================================

def write_custom_io_yaml(path: Path, layer_start: int, layer_end: int):
    indices = [str(i) for i in range(layer_start, layer_end+1)]

    with open(path, "w") as fp:
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
# HELPERS
# =====================================================

def parse_layer_window(onnx_path: Path):
    """
    Expect: merged_<start>-<end>.onnx
    """
    m = re.search(r"merged_(\d+)-(\d+)\.onnx$", onnx_path.name)
    if not m:
        raise ValueError(
            "ONNX filename must follow: merged_<start>-<end>.onnx"
        )
    return int(m.group(1)), int(m.group(2))

# =====================================================
# COMPILE
# =====================================================

def compile_merged_onnx(onnx_path: Path):
    layer_start, layer_end = parse_layer_window(onnx_path)

    output_dir = onnx_path.parent
    tag = f"{layer_start}_{layer_end}"

    qpc_dir = output_dir / f"qpc_merged_{tag}"
    log_file = output_dir / f"qpc_merged_{tag}.log"
    qpc_dir.mkdir(parents=True, exist_ok=True)

    custom_io_yaml = output_dir / "custom_io_fp16.yaml"
    write_custom_io_yaml(custom_io_yaml, layer_start, layer_end)

    cmd = [
        "python",
        "-m",
        "QEfficient.cloud.compile",
        "--onnx_path", str(onnx_path),
        "--qpc-path", str(qpc_dir),
        "--batch_size", "1",
        "--prompt_len", "1",
        "--ctx_len", "1000",
        "--mxfp6", "mxint8_kv_cache",
        "--num_cores", "16",
        "--device_group", DEVICE_GROUP,
        "--mos", "1",
        "--aic_enable_depth_first",
        f"-custom-IO-list-file={custom_io_yaml}",
    ]

    print(f"[START ] merged layers {layer_start}-{layer_end} | devices [{DEVICE_GROUP}]")

    start = time.time()
    proc = None

    try:
        with open(log_file, "w") as lf:
            proc = subprocess.Popen(
                cmd,
                stdout=lf,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
            proc.wait(timeout=TIMEOUT)

        if proc.returncode != 0:
            raise RuntimeError(f"Compile failed rc={proc.returncode}")

    except subprocess.TimeoutExpired:
        if proc:
            os.killpg(proc.pid, signal.SIGTERM)
        raise RuntimeError("Compile TIMEOUT")

    elapsed = time.time() - start
    print(f"[DONE  ] SUCCESS | {elapsed:.1f}s")

# =====================================================
# MAIN
# =====================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Compile merged_<start>-<end>.onnx into QPC"
    )
    parser.add_argument("--onnx-path", required=True, type=Path)
    args = parser.parse_args()

    compile_merged_onnx(args.onnx_path)
import argparse
import json
import subprocess
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
MXFP6 = True
MXINT8_KV_CACHE = True

COMPILER = ["/opt/qti-aic/exec/qaic-compile", "-aic-hw", "-sub-functions"]
# =====================================================
# DISCOVERY
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



def create_specializations_json(path: Path, batch_size: int, seq_len: int, ctx_len: int,
                                kv_cache_batch_size:int = None, cb: bool=False, prefill_only:bool=False) -> Path:
    payload = {"specializations": [{"batch_size": str(batch_size), "seq_len": str(seq_len), "ctx_len": str(ctx_len)}]}
    if prefill_only and cb and kv_cache_batch_size is not None:
        payload['specializations'][0]['full_batch_size'] = str(kv_cache_batch_size)
    path.write_text(json.dumps(payload, indent=2))
    return path


def create_mdp_ts_json(path: Path, device_ids: list[int], num_cores: int) -> Path | None:
    if len(device_ids) <= 1:
        return None
    payload = {
        "connections": [{"devices": list(range(len(device_ids))), "type": "p2p"}],
        "partitions": [
            {
                "name": "Partition0",
                "devices": [{"deviceId": idx, "numCores": num_cores} for idx in range(len(device_ids))],
            }
        ],
    }
    path.write_text(json.dumps(payload, indent=2))
    return path


def create_custom_io_yaml(path: Path, kv_cache_dtype: str, num_layers: int):
    custom_io = {}
    for suffix in ["", "_RetainedState"]:
        for i in range(num_layers):
            custom_io[f"compressed_kv.{i}{suffix}"] = kv_cache_dtype
            custom_io[f"k_pe.{i}{suffix}"] = kv_cache_dtype

    with open(path, "w") as fp:
        for io_name, dtype in custom_io.items():
            fp.write(f" - IOName: {io_name}\n   Precision: {dtype}\n\n")
    return path


def build_compile_command(
    onnx_path: Path,
    qpc_dir: Path,
    specialization_json: Path,
    num_cores: int,
    enable_mxfp6: bool,
    mdp_ts_json: Path | None,
    custom_io_yaml: str,
    aic_version: str,
) -> list[str]:
    command = [
        COMPILER[0],
        COMPILER[1],
        COMPILER[2],
        f"-aic-hw-version={aic_version}",
        f"-m={onnx_path}",
        f"-network-specialization-config={specialization_json}",
        "-convert-to-fp16",
        f"-aic-num-cores={num_cores}",
        "-compile-only",
        f"-aic-binary-dir={qpc_dir}",
        f"-custom-IO-list-file={custom_io_yaml}",
        "-retained-state",
    ]
    if mdp_ts_json is not None:
        command.append(f"-mdp-load-partition-config={mdp_ts_json}")
    if enable_mxfp6:
        command.append("-mxfp6-matmul")
    return command


def run_full_model_compile(
    onnx_path: Path | str,
    num_devices: int = 1,
    batch_size: int = 1,
    seq_len: int = 1,
    ctx_len: int = 128,
    num_cores: int = 16,
    num_layers: int = 61,
    mxfp6: bool = True,
    mxint8_kv_cache: bool = True,
    aic_version: str = "ai100",
    kv_cache_batch_size: int = None,
    cb: bool = False,
    prefill_only:bool=False
):
    onnx_path = Path(onnx_path)
    qpc_binaries_dir = onnx_path.parent / "qpc_binaries"
    qpc_binaries_dir.mkdir(exist_ok=True)

    specialization_json = create_specializations_json(
        qpc_binaries_dir / "specializations.json",
        batch_size=batch_size,
        seq_len=seq_len,
        ctx_len=ctx_len,
        kv_cache_batch_size=kv_cache_batch_size,
        cb=cb,
        prefill_only=prefill_only,
    )
    custom_io_file = create_custom_io_yaml(
        qpc_binaries_dir / "custom_io_fp16.yaml",
        kv_cache_dtype="mxint8" if mxint8_kv_cache else "float16",
        num_layers=num_layers,
    )
    mdp_ts_json = create_mdp_ts_json(
        qpc_binaries_dir / "mdp_ts.json",
        device_ids=list(range(num_devices)),
        num_cores=num_cores,
    )
    command = build_compile_command(
        onnx_path=onnx_path,
        qpc_dir=qpc_binaries_dir / "qpc",
        specialization_json=specialization_json,
        num_cores=num_cores,
        enable_mxfp6=mxfp6,
        mdp_ts_json=mdp_ts_json,
        custom_io_yaml=custom_io_file,
        aic_version=aic_version,
    )
    print(" ".join(command))
    result = subprocess.run(command, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"Compilation Failed!!\n\nSTDOUT\n{result.stdout}\n\nSTDERR\n{result.stderr}")
    return qpc_binaries_dir / "qpc"


def _parse_args():
    parser = argparse.ArgumentParser(description="Compile layerwise ONNX windows into QPC artifacts.")
    parser.add_argument("--onnx_path", required=True, help="Path to .onnx file")
    parser.add_argument("--aic_hw_version", dest="aic_hw_version", type=str, default="ai100")
    parser.add_argument("--num-devices", type=int, default=1, help="Number of devices to use (e.g., 2 -> device_group=0,1)")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size for specialization config (default: 1)")
    parser.add_argument("--seq_len", type=int, default=1, help="Sequence length for specialization config (default: 1)")
    parser.add_argument("--ctx_len", type=int, default=128, help="Context length for specialization config (default: 128)")
    parser.add_argument("--num_cores", type=int, default=16, help="Number of accelerator cores to compile for (default: 16)")
    parser.add_argument("--num_layers", type=int, default=61, help="Total number of transformer layers for custom IO yaml")
    parser.add_argument("--mxfp6", dest="mxfp6", action="store_true", default=True, help="Enable mxfp6 compile flag (default: True)")
    parser.add_argument("--no-mxfp6", dest="mxfp6", action="store_false", help="Disable mxfp6 compile flag")
    parser.add_argument(
        "--mxint8_kv_cache",
        dest="mxint8_kv_cache",
        action="store_true",
        default=True,
        help="Enable mxint8 kv-cache compile flag (default: True)",
    )
    parser.add_argument(
        "--no-mxint8_kv_cache",
        dest="mxint8_kv_cache",
        action="store_false",
        help="Disable mxint8 kv-cache compile flag",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    run_full_model_compile(
        onnx_path=args.onnx_path,
        num_devices=args.num_devices,
        batch_size=args.batch_size,
        seq_len=args.seq_len,
        ctx_len=args.ctx_len,
        num_cores=args.num_cores,
        num_layers=args.num_layers,
        mxfp6=args.mxfp6,
        mxint8_kv_cache=args.mxint8_kv_cache,
        aic_version=args.aic_hw_version,
    )

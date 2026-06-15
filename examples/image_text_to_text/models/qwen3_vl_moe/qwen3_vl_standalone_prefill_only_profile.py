"""Standalone prefill-only layerwise compile profiler for Qwen3-VL MoE.

This intentionally skips vision compilation and decode compilation. It measures
`torch.onnx.export` calls and the whole prefill compile path.

Useful environment variables:
  HF_HUB_CACHE=/home/huggingface_hub
  QEFF_HOME=/tmp/qeff_qwen3_vl_prefill_only_profile
  QEFF_PROFILE_FORCE_REEXPORT=1  # default; set 0 to allow export cache reuse
"""

import contextlib
import os
import time

import torch
from transformers import AutoConfig

from QEfficient import QEFFAutoModelForImageTextToText

MODEL_ID = "Qwen/Qwen3-VL-30B-A3B-Instruct"
LANGUAGE_NUM_LAYERS = 2

PREFILL_SEQ_LEN = 128
CTX_LEN = 4096
BATCH_SIZE = 1
HEIGHT = 354
WIDTH = 536
NUM_CORES = 16
NUM_DEVICES = 1


@contextlib.contextmanager
def _time_block(name):
    start_time = time.perf_counter()
    try:
        yield
    finally:
        elapsed_seconds = time.perf_counter() - start_time
        print(f"[profile] {name}: time={elapsed_seconds:.2f}s", flush=True)


@contextlib.contextmanager
def _profile_torch_onnx_export():
    original_export = torch.onnx.export
    export_count = 0

    def timed_export(*args, **kwargs):
        nonlocal export_count
        export_count += 1
        output_path = kwargs.get("f")
        if output_path is None and len(args) >= 3:
            output_path = args[2]
        with _time_block(f"torch.onnx.export #{export_count} path={output_path}"):
            return original_export(*args, **kwargs)

    torch.onnx.export = timed_export
    try:
        yield
    finally:
        torch.onnx.export = original_export


@contextlib.contextmanager
def _force_fresh_export_if_requested():
    force_reexport = os.environ.get("QEFF_PROFILE_FORCE_REEXPORT", "1") != "0"
    if not force_reexport:
        yield
        return

    import QEfficient.utils.export_utils as qeff_export_utils

    original_create_export_hash = qeff_export_utils.create_export_hash
    profile_run_id = str(time.time_ns())

    def profiled_create_export_hash(*args, **kwargs):
        kwargs = dict(kwargs)
        export_kwargs = dict(kwargs.get("export_kwargs") or {})
        export_kwargs["_qeff_profile_run_id"] = profile_run_id
        kwargs["export_kwargs"] = export_kwargs
        return original_create_export_hash(*args, **kwargs)

    qeff_export_utils.create_export_hash = profiled_create_export_hash
    try:
        yield
    finally:
        qeff_export_utils.create_export_hash = original_create_export_hash


def main():
    print(
        "[profile config] "
        f"model_id={MODEL_ID}, layers={LANGUAGE_NUM_LAYERS}, "
        f"prefill_seq_len={PREFILL_SEQ_LEN}, ctx_len={CTX_LEN}, "
        "skip_vision=True, skip_lang=False, prefill_only=True, "
        f"HF_HUB_CACHE={os.environ.get('HF_HUB_CACHE')}, "
        f"QEFF_HOME={os.environ.get('QEFF_HOME')}, "
        f"force_reexport={os.environ.get('QEFF_PROFILE_FORCE_REEXPORT', '1') != '0'}",
        flush=True,
    )

    config = AutoConfig.from_pretrained(MODEL_ID)
    config.dtype = "float16"
    config.text_config.dtype = "float16"
    config.text_config.num_hidden_layers = LANGUAGE_NUM_LAYERS

    with _time_block("from_pretrained"):
        qeff_model = QEFFAutoModelForImageTextToText.from_pretrained(
            MODEL_ID,
            attn_implementation="eager",
            kv_offload=True,
            config=config,
            dtype=torch.float16,
            layerwise=True,
        )

    compile_kwargs = dict(
        batch_size=BATCH_SIZE,
        ctx_len=CTX_LEN,
        height=HEIGHT,
        width=WIDTH,
        num_cores=NUM_CORES,
        num_devices=NUM_DEVICES,
        mos=1,
        mxfp6_matmul=True,
        aic_enable_depth_first=True,
        split_model_io=True,
        use_onnx_subfunctions=True,
        layerwise=True,
        layerwise_window_size=1,
        prefill_seq_len=PREFILL_SEQ_LEN,
        mxint8_kv_cache=True,
        retain_full_kv=True,
        prefill_only=True,
        enable_chunking=True,
        skip_vision=True,
        skip_lang=False,
    )

    with _force_fresh_export_if_requested(), _profile_torch_onnx_export(), _time_block("prefill_only_compile"):
        qpc_path = qeff_model.compile(**compile_kwargs)

    print(f"Prefill QPC path: {qpc_path}", flush=True)


if __name__ == "__main__":
    main()

# -----------------------------------------------------------------------------
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------

import argparse
import copy
import shutil
import subprocess
import tempfile
from io import BytesIO
from pathlib import Path

import numpy as np
import onnxruntime as ort
import requests
import torch
from export_kimi_k25_vision import (
    LOADED_EXPERT_IDS,
    MODEL_PATH,
    NUM_EXPERTS_PER_TOKEN,
    NUM_TEXT_LAYERS,
    NUM_VISION_LAYERS,
    _ensure_torch_fx_import_compatibility,
    _load_layer_subset_model,
    _patch_deepseek_init_weights_compat,
    _patch_kimi_tie_weights_compat,
    _prepare_config,
)
from PIL import Image
from transformers.dynamic_module_utils import get_class_from_dynamic_module

from QEfficient.generation.cloud_infer import QAICInferenceSession
from QEfficient.transformers.models.modeling_auto import QEffVisionEncoderForTextImageToTextModel


def parse_args():
    parser = argparse.ArgumentParser(
        description="Verify Kimi K2.5 vision parity: HF PyTorch vs QEff PyTorch vs ONNXRuntime."
    )
    parser.add_argument("--model-path", type=Path, default=MODEL_PATH)
    parser.add_argument("--num-vision-layers", type=int, default=NUM_VISION_LAYERS)
    parser.add_argument("--num-text-layers", type=int, default=NUM_TEXT_LAYERS)
    parser.add_argument("--expert-ids", type=str, default=",".join(str(x) for x in LOADED_EXPERT_IDS))
    parser.add_argument("--num-experts-per-token", type=int, default=NUM_EXPERTS_PER_TOKEN)
    parser.add_argument(
        "--image-url",
        type=str,
        default="https://huggingface.co/moonshotai/Kimi-K2.5/resolve/main/figures/kimi-logo.png",
    )
    parser.add_argument("--prompt", type=str, default="Tell me about yourself.")
    parser.add_argument("--atol", type=float, default=1e-3)
    parser.add_argument("--rtol", type=float, default=1e-3)
    parser.add_argument(
        "--keep-onnx",
        action="store_true",
        help="Keep exported ONNX under ./aisand/onnx_export_verify instead of temp directory.",
    )
    parser.add_argument(
        "--run-qpc", action="store_true", help="Compile and execute QPC, then compare QPC output vs HF."
    )
    parser.add_argument(
        "--qaic-compile-bin",
        type=str,
        default="/opt/qti-aic/exec/qaic-compile",
        help="Path to qaic-compile binary.",
    )
    parser.add_argument(
        "--keep-qpc",
        action="store_true",
        help="Keep compiled QPC under --qpc-dir (used only with --run-qpc).",
    )
    parser.add_argument(
        "--qpc-dir",
        type=Path,
        default=Path("aisand/qpc_verify"),
        help="Output directory for QPC when --keep-qpc is set.",
    )
    parser.add_argument("--aic-num-cores", type=int, default=16, help="Number of QAIC cores for qpc compile.")
    parser.add_argument("--mos", type=int, default=1, help="-mos value for qpc compile.")
    parser.add_argument(
        "--no-convert-to-fp16",
        action="store_true",
        help="Disable -convert-to-fp16 during qpc compile (defaults to enabled).",
    )
    return parser.parse_args()


def _parse_expert_ids(value: str):
    return tuple(int(x.strip()) for x in value.split(",") if x.strip())


def _stats(a: torch.Tensor, b: torch.Tensor, atol: float, rtol: float):
    diff = (a - b).abs()
    return {
        "max_abs": float(diff.max().item()),
        "mean_abs": float(diff.mean().item()),
        "rmse": float(torch.sqrt(torch.mean((a - b) ** 2)).item()),
        "allclose": bool(torch.allclose(a, b, atol=atol, rtol=rtol)),
    }


def _hf_image_embeds(model, pixel_values, grid_thws):
    target_dtype = model.vision_tower.patch_embed.proj.weight.dtype
    pixel_values = pixel_values.to(target_dtype)
    image_features = model.vision_tower(pixel_values, grid_thws)
    if model.mm_projector:
        image_features = model.mm_projector(image_features)
    return image_features[0] if isinstance(image_features, list) else image_features


def _run_onnx(qeff_vision_wrapper, qeff_base_model, pixel_values, h_shape, w_shape, keep_onnx: bool):
    output_names = qeff_base_model.get_output_names(kv_offload=True)["vision"]
    dynamic_axes = qeff_base_model.get_onnx_dynamic_axes(kv_offload=True)["vision"]

    if keep_onnx:
        export_dir = Path("aisand/onnx_export_verify")
        export_dir.mkdir(parents=True, exist_ok=True)
    else:
        export_dir = Path(tempfile.mkdtemp(prefix="kimi_k25_verify_onnx_"))

    onnx_path = qeff_vision_wrapper.export(
        {"pixel_values": pixel_values, "h_shape": h_shape, "w_shape": w_shape},
        output_names,
        dynamic_axes,
        export_dir=export_dir,
        offload_pt_weights=False,
    )

    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    ort_inputs = {
        "pixel_values": pixel_values.detach().cpu().numpy().astype(np.float32),
        "h_shape": h_shape.detach().cpu().numpy().astype(np.int64),
        "w_shape": w_shape.detach().cpu().numpy().astype(np.int64),
    }
    ort_out = session.run(None, ort_inputs)[0]
    return torch.from_numpy(ort_out).float(), Path(onnx_path)


def _run_qpc(onnx_path: Path, pixel_values, h_shape, w_shape, args):
    if args.keep_qpc:
        qpc_dir = args.qpc_dir
        if qpc_dir.exists():
            shutil.rmtree(qpc_dir)
        qpc_dir.parent.mkdir(parents=True, exist_ok=True)
    else:
        qpc_dir = Path(tempfile.mkdtemp(prefix="kimi_k25_verify_qpc_"))
        shutil.rmtree(qpc_dir)

    compile_cmd = [
        args.qaic_compile_bin,
        f"-m={onnx_path}",
        f"-onnx-define-symbol=num_patches,{pixel_values.shape[0]}",
        f"-onnx-define-symbol=h,{h_shape.shape[0]}",
        f"-onnx-define-symbol=w,{w_shape.shape[0]}",
        f"-aic-num-cores={args.aic_num_cores}",
        f"-mos={args.mos}",
        f"-aic-binary-dir={qpc_dir}",
    ]
    if not args.no_convert_to_fp16:
        compile_cmd.insert(1, "-convert-to-fp16")
    compile_result = subprocess.run(compile_cmd, capture_output=True, text=True)
    if compile_result.returncode != 0:
        raise RuntimeError(
            "QPC compile failed.\n"
            f"Command: {' '.join(compile_cmd)}\n"
            f"stdout:\n{compile_result.stdout}\n"
            f"stderr:\n{compile_result.stderr}"
        )

    session = QAICInferenceSession(str(qpc_dir))
    qpc_inputs = {}
    for name in session.input_names:
        if name == "pixel_values":
            qpc_inputs[name] = pixel_values.detach().cpu().numpy().astype(np.float32)
        elif name == "h_shape":
            qpc_inputs[name] = h_shape.detach().cpu().numpy().astype(np.int64)
        elif name == "w_shape":
            qpc_inputs[name] = w_shape.detach().cpu().numpy().astype(np.int64)
        else:
            raise RuntimeError(f"Unexpected QPC input name: {name}")

    qpc_outputs = session.run(qpc_inputs)
    output_name = session.output_names[0]
    qpc_out = qpc_outputs[output_name]
    return torch.from_numpy(qpc_out).float(), qpc_dir


if __name__ == "__main__":
    args = parse_args()
    expert_ids = _parse_expert_ids(args.expert_ids)

    torch.manual_seed(123)
    torch.set_grad_enabled(False)

    _ensure_torch_fx_import_compatibility()
    config = _prepare_config(args.model_path)
    kimi_cls = get_class_from_dynamic_module("modeling_kimi_k25.KimiK25ForConditionalGeneration", str(args.model_path))
    _patch_kimi_tie_weights_compat(kimi_cls)
    _patch_deepseek_init_weights_compat(kimi_cls)

    hf_model, _, processor = _load_layer_subset_model(
        model_path=args.model_path,
        kimi_cls=kimi_cls,
        config=config,
        num_vision_layers=args.num_vision_layers,
        num_text_layers=args.num_text_layers,
        loaded_expert_ids=expert_ids,
        num_experts_per_tok=args.num_experts_per_token,
        dtype=torch.float32,
    )

    # Required for HW compatibility and parity in this flow.
    hf_model.vision_tower.patch_embed.pos_emb.interpolation_mode = "bilinear"
    hf_model.eval()

    qeff_base_model = copy.deepcopy(hf_model).eval()
    qeff_vision_wrapper = QEffVisionEncoderForTextImageToTextModel(qeff_base_model)
    qeff_encoder = qeff_vision_wrapper.model.eval()

    image = Image.open(BytesIO(requests.get(args.image_url, timeout=30).content)).convert("RGB")
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": image},
                {"type": "text", "text": args.prompt},
            ],
        }
    ]
    inputs = processor(messages=messages, add_generation_prompt=True, tokenize=False, return_tensors="pt")

    pixel_values = inputs["pixel_values"]
    grid_thws = inputs["grid_thws"]
    h_shape = torch.ones(int(grid_thws[0, 1].item()), dtype=torch.int64)
    w_shape = torch.ones(int(grid_thws[0, 2].item()), dtype=torch.int64)

    hf_out = _hf_image_embeds(hf_model, pixel_values, grid_thws).detach().cpu().float()
    qeff_out = qeff_encoder(pixel_values, h_shape, w_shape).detach().cpu().float()
    ort_out, onnx_path = _run_onnx(qeff_vision_wrapper, qeff_base_model, pixel_values, h_shape, w_shape, args.keep_onnx)

    hf_vs_qeff = _stats(hf_out, qeff_out, args.atol, args.rtol)
    hf_vs_ort = _stats(hf_out, ort_out, args.atol, args.rtol)
    qeff_vs_ort = _stats(qeff_out, ort_out, args.atol, args.rtol)

    print(f"ONNX path: {onnx_path}")
    print("hf_vs_qeff", hf_vs_qeff)
    print("hf_vs_onnxrt", hf_vs_ort)
    print("qeff_vs_onnxrt", qeff_vs_ort)

    all_ok = hf_vs_qeff["allclose"] and hf_vs_ort["allclose"] and qeff_vs_ort["allclose"]

    if args.run_qpc:
        atol = 1e-2
        rtol = 1e-2
        qpc_out, qpc_dir = _run_qpc(onnx_path, pixel_values, h_shape, w_shape, args)
        hf_vs_qpc = _stats(hf_out, qpc_out, atol, rtol)
        qeff_vs_qpc = _stats(qeff_out, qpc_out, atol, rtol)
        ort_vs_qpc = _stats(ort_out, qpc_out, atol, rtol)
        print(f"QPC dir: {qpc_dir}")
        print("hf_vs_qpc", hf_vs_qpc)
        print("qeff_vs_qpc", qeff_vs_qpc)
        print("onnxrt_vs_qpc", ort_vs_qpc)
        all_ok = all_ok and hf_vs_qpc["allclose"] and qeff_vs_qpc["allclose"] and ort_vs_qpc["allclose"]

    if not all_ok:
        raise SystemExit(f"Parity check failed for atol={atol}, rtol={rtol}. See metrics above for details.")

    print("Parity check passed.")

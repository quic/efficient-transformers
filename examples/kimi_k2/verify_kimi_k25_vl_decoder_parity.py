# -----------------------------------------------------------------------------
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------

import argparse
import copy
import os
import shutil
import subprocess
import sys
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
    _ensure_torch_fx_import_compatibility,
    _load_layer_subset_model,
    _patch_deepseek_init_weights_compat,
    _patch_kimi_tie_weights_compat,
    _prepare_config,
)
from huggingface_hub import snapshot_download
from PIL import Image
from transformers.dynamic_module_utils import get_class_from_dynamic_module

from QEfficient import QEFFAutoModelForImageTextToText
from QEfficient.generation.cloud_infer import QAICInferenceSession
from QEfficient.transformers.models.modeling_auto import QEffCausalLMForTextImageToTextModel


def parse_args():
    parser = argparse.ArgumentParser(
        description=("Verify Kimi-K2.5 V+L decoder parity on CPU with a 2-layer vision + 2-layer text subset.")
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default="moonshotai/Kimi-K2.5",
        help=(
            "Local model path or HF repo id (defaults to moonshotai/Kimi-K2.5; "
            "downloads/resolves using HF_HUB_CACHE when needed)."
        ),
    )
    parser.add_argument("--num-text-layers", type=int, default=2)
    parser.add_argument("--num-vision-layers", type=int, default=2)
    parser.add_argument("--expert-ids", type=str, default=",".join(str(x) for x in LOADED_EXPERT_IDS))
    parser.add_argument("--num-experts-per-token", type=int, default=NUM_EXPERTS_PER_TOKEN)
    parser.add_argument(
        "--image-url",
        type=str,
        default="https://huggingface.co/moonshotai/Kimi-K2.5/resolve/main/figures/kimi-logo.png",
    )
    parser.add_argument("--prompt", type=str, default="Describe this image in one sentence.")
    parser.add_argument("--atol", type=float, default=1e-4)
    parser.add_argument("--rtol", type=float, default=1e-4)
    parser.add_argument("--skip-onnx", action="store_true", help="Skip ONNX export/runtime parity check.")
    parser.add_argument(
        "--keep-onnx",
        action="store_true",
        help="Keep exported ONNX under ./aisand/onnx_export_verify_vl instead of a temporary directory.",
    )
    parser.add_argument("--run-qpc", action="store_true", help="Compile and execute QPC, then compare vs references.")
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
        default=Path("aisand/qpc_verify_vl"),
        help="Output directory for QPC when --keep-qpc is set.",
    )
    parser.add_argument("--aic-num-cores", type=int, default=16, help="Number of QAIC cores for qpc compile.")
    parser.add_argument("--mos", type=int, default=1, help="-mos value for qpc compile.")
    parser.add_argument(
        "--no-convert-to-fp16",
        action="store_true",
        help="Disable -convert-to-fp16 during qpc compile (defaults to enabled).",
    )
    parser.add_argument("--qpc-atol", type=float, default=1e-2)
    parser.add_argument("--qpc-rtol", type=float, default=1e-2)
    return parser.parse_args()


def _parse_expert_ids(value: str):
    return tuple(int(x.strip()) for x in value.split(",") if x.strip())


def _resolve_model_path(model_path_or_id: str) -> Path:
    candidate = Path(model_path_or_id)
    if candidate.exists():
        return candidate

    cache_dir = os.environ.get("HF_HUB_CACHE")
    return Path(snapshot_download(repo_id=model_path_or_id, cache_dir=cache_dir))


def _stats(a: torch.Tensor, b: torch.Tensor, atol: float, rtol: float):
    diff = (a - b).abs()
    return {
        "shape": tuple(a.shape),
        "max_abs": float(diff.max().item()),
        "mean_abs": float(diff.mean().item()),
        "rmse": float(torch.sqrt(torch.mean((a - b) ** 2)).item()),
        "allclose": bool(torch.allclose(a, b, atol=atol, rtol=rtol)),
    }


def _hf_image_embeds(model, pixel_values: torch.Tensor, grid_thws: torch.Tensor) -> torch.Tensor:
    target_dtype = model.vision_tower.patch_embed.proj.weight.dtype
    pixel_values = pixel_values.to(target_dtype)
    image_features = model.vision_tower(pixel_values, grid_thws)
    if model.mm_projector:
        image_features = model.mm_projector(image_features)
    return image_features[0] if isinstance(image_features, list) else image_features


def _get_onnx_export_dir(keep_onnx: bool):
    if keep_onnx:
        base_dir = Path("aisand")
        base_dir.mkdir(parents=True, exist_ok=True)
        for old in base_dir.glob("onnx_export_verify_vl-*"):
            shutil.rmtree(old, ignore_errors=True)
        export_dir = base_dir / "onnx_export_verify_vl"
        if export_dir.exists():
            shutil.rmtree(export_dir, ignore_errors=True)
        export_dir.mkdir(parents=True, exist_ok=True)
        return export_dir
    return Path(tempfile.mkdtemp(prefix="kimi_k25_verify_vl_onnx_"))


def _export_and_run_lang_onnx(
    transformed_model,
    input_ids: torch.Tensor,
    position_ids: torch.Tensor,
    past_key_values,
    keep_onnx: bool,
):
    lang_exporter = QEffCausalLMForTextImageToTextModel(copy.deepcopy(transformed_model))
    lang_inputs = transformed_model.get_dummy_inputs(kv_offload=True)["lang"]
    output_names = transformed_model.get_output_names(kv_offload=True)["lang"]
    dynamic_axes = transformed_model.get_onnx_dynamic_axes(kv_offload=True)["lang"]

    export_dir = _get_onnx_export_dir(keep_onnx)
    onnx_path = lang_exporter.export(
        inputs=lang_inputs,
        output_names=output_names,
        dynamic_axes=dynamic_axes,
        export_dir=export_dir,
        offload_pt_weights=False,
    )

    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    ort_inputs = {
        "input_ids": input_ids.detach().cpu().numpy().astype(np.int64),
        "position_ids": position_ids.detach().cpu().numpy().astype(np.int64),
    }
    for layer_idx, (past_key, past_value) in enumerate(past_key_values):
        ort_inputs[f"past_key.{layer_idx}"] = past_key.detach().cpu().numpy().astype(np.float32)
        ort_inputs[f"past_value.{layer_idx}"] = past_value.detach().cpu().numpy().astype(np.float32)

    ort_logits = torch.from_numpy(session.run(["logits"], ort_inputs)[0]).float()
    return ort_logits, Path(onnx_path)


def _prepare_qpc_dir(args, suffix: str):
    if args.keep_qpc:
        qpc_dir = args.qpc_dir / suffix
        if qpc_dir.exists():
            shutil.rmtree(qpc_dir)
        qpc_dir.parent.mkdir(parents=True, exist_ok=True)
    else:
        qpc_dir = Path(tempfile.mkdtemp(prefix=f"kimi_k25_verify_vl_{suffix}_qpc_"))
        shutil.rmtree(qpc_dir)
    return qpc_dir


def _run_vision_qpc(onnx_path: Path, pixel_values: torch.Tensor, h_shape: torch.Tensor, w_shape: torch.Tensor, args):
    qpc_dir = _prepare_qpc_dir(args, "vision")
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
            "Vision QPC compile failed.\n"
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
            raise RuntimeError(f"Unexpected vision QPC input name: {name}")

    qpc_outputs = session.run(qpc_inputs)
    output_name = session.output_names[0]
    qpc_out = qpc_outputs[output_name]
    return torch.from_numpy(qpc_out).float(), qpc_dir


def _run_lang_qpc(onnx_path: Path, input_ids: torch.Tensor, position_ids: torch.Tensor, past_key_values, args):
    qpc_dir = _prepare_qpc_dir(args, "lang")
    compile_cmd = [
        args.qaic_compile_bin,
        f"-m={onnx_path}",
        f"-onnx-define-symbol=batch_size,{input_ids.shape[0]}",
        f"-onnx-define-symbol=seq_len,{input_ids.shape[1]}",
        f"-onnx-define-symbol=ctx_len,{past_key_values[0][0].shape[2]}",
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
    qpc_inputs = {
        "input_ids": input_ids.detach().cpu().numpy().astype(np.int64),
        "position_ids": position_ids.detach().cpu().numpy().astype(np.int64),
    }
    for layer_idx, (past_key, past_value) in enumerate(past_key_values):
        qpc_inputs[f"past_key.{layer_idx}"] = past_key.detach().cpu().numpy().astype(np.float32)
        qpc_inputs[f"past_value.{layer_idx}"] = past_value.detach().cpu().numpy().astype(np.float32)

    qpc_feed = {name: qpc_inputs[name] for name in session.input_names}
    qpc_outputs = session.run(qpc_feed)
    output_name = "logits" if "logits" in qpc_outputs else session.output_names[0]
    qpc_logits = torch.from_numpy(qpc_outputs[output_name]).float()
    return qpc_logits, qpc_dir


if __name__ == "__main__":
    args = parse_args()
    if args.model_path is None:
        args.model_path = MODEL_PATH
    args.model_path = _resolve_model_path(str(args.model_path))
    expert_ids = _parse_expert_ids(args.expert_ids)

    torch.set_grad_enabled(False)
    _ensure_torch_fx_import_compatibility()

    config = _prepare_config(args.model_path)
    kimi_cls = get_class_from_dynamic_module("modeling_kimi_k25.KimiK25ForConditionalGeneration", str(args.model_path))
    _patch_kimi_tie_weights_compat(kimi_cls)
    _patch_deepseek_init_weights_compat(kimi_cls)

    hf_model, tokenizer, processor = _load_layer_subset_model(
        model_path=args.model_path,
        kimi_cls=kimi_cls,
        config=config,
        num_vision_layers=args.num_vision_layers,
        num_text_layers=args.num_text_layers,
        loaded_expert_ids=expert_ids,
        num_experts_per_tok=args.num_experts_per_token,
        dtype=torch.float32,
    )
    # Required for HW compatibility and stable vision parity in this flow.
    hf_model.vision_tower.patch_embed.pos_emb.interpolation_mode = "bilinear"
    hf_model = hf_model.eval()

    transformed_wrapper = QEFFAutoModelForImageTextToText(copy.deepcopy(hf_model))
    transformed_model = transformed_wrapper.model.eval()
    decoder_wrapper = transformed_model.get_qeff_language_decoder().eval()

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

    input_ids = inputs["input_ids"].to(torch.long)
    attention_mask = inputs["attention_mask"].to(torch.long)
    pixel_values = inputs["pixel_values"].to(torch.float32)
    grid_thws = inputs["grid_thws"].to(torch.long)
    h_shape = torch.ones(int(grid_thws[0, 1].item()), dtype=torch.int64)
    w_shape = torch.ones(int(grid_thws[0, 2].item()), dtype=torch.int64)
    position_ids = torch.where(attention_mask.bool(), attention_mask.cumsum(-1) - 1, -1)

    image_features = transformed_model._extract_image_features(pixel_values, grid_thws)
    if transformed_model.mm_projector:
        image_features = transformed_model.mm_projector(image_features)
    image_embeds = image_features[0] if isinstance(image_features, list) else image_features
    vision_reference = _hf_image_embeds(hf_model, pixel_values, grid_thws).detach().cpu().float()

    vision_onnx_path = None
    if not args.skip_onnx or args.run_qpc:
        vision_cmd = [
            str(Path(sys.executable)),
            str(Path(__file__).with_name("verify_kimi_k25_vision_parity.py")),
            "--model-path",
            str(args.model_path),
            "--num-vision-layers",
            str(args.num_vision_layers),
            "--num-text-layers",
            str(args.num_text_layers),
            "--expert-ids",
            args.expert_ids,
            "--num-experts-per-token",
            str(args.num_experts_per_token),
            "--image-url",
            args.image_url,
            "--prompt",
            args.prompt,
            "--atol",
            str(args.atol),
            "--rtol",
            str(args.rtol),
        ]
        keep_vision_onnx = args.keep_onnx or args.run_qpc
        if keep_vision_onnx:
            vision_cmd.append("--keep-onnx")
        subprocess.run(vision_cmd, check=True)

        if keep_vision_onnx:
            vision_matches = sorted(Path("aisand").glob("onnx_export_verify-*/KimiK25EncoderWrapper.onnx"))
            if not vision_matches:
                raise RuntimeError("Vision ONNX not found under aisand/onnx_export_verify-*/KimiK25EncoderWrapper.onnx")
            vision_onnx_path = vision_matches[-1]

    merged_inputs_embeds, merged_attention_mask, _, merged_position_ids = (
        transformed_model._merge_input_ids_with_image_features(
            image_features=image_features,
            inputs_embeds=transformed_model.get_input_embeddings()(input_ids),
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=None,
        )
    )

    past_key_values = transformed_model.language_model.get_dummy_pkv_cache(
        transformed_model.config.text_config,
        input_ids.shape[0],
        merged_inputs_embeds.shape[1],
    )

    model_logits = (
        transformed_model.language_model(
            inputs_embeds=merged_inputs_embeds,
            attention_mask=merged_attention_mask,
            position_ids=merged_position_ids,
            past_key_values=copy.deepcopy(past_key_values),
            use_cache=True,
        )[0]
        .detach()
        .cpu()
        .float()
    )

    decoder_logits = (
        decoder_wrapper(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            image_embeds=image_embeds,
            image_idx=torch.zeros((input_ids.shape[0], 1), dtype=torch.int64),
            past_key_values=copy.deepcopy(past_key_values),
            use_cache=True,
        )[0]
        .detach()
        .cpu()
        .float()
    )

    decoder_vs_model = _stats(model_logits, decoder_logits, args.atol, args.rtol)

    print(
        "Loaded subset:",
        f"vision_layers={transformed_model.config.vision_config.vt_num_hidden_layers}",
        f"text_layers={transformed_model.config.text_config.num_hidden_layers}",
    )
    print("Tokenizer vocab size:", tokenizer.vocab_size)
    print("Prompt:", repr(args.prompt))
    print("Input shape:", tuple(input_ids.shape))
    print("Model logits vs decoder-wrapper logits:", decoder_vs_model)

    if not decoder_vs_model["allclose"]:
        raise SystemExit(
            "Parity check failed: decoder-wrapper logits do not match transformed model logits for image+text input."
        )

    # ONNX/QPC parity runs the language decoder in decode mode (seq_len=1)
    # with multimodal context represented in past_key_values.
    onnx_input_ids = input_ids[:, -1:]
    onnx_position_ids = merged_position_ids[:, -1:]
    onnx_reference_logits = (
        transformed_model.language_model(
            inputs_embeds=transformed_model.get_input_embeddings()(onnx_input_ids),
            position_ids=onnx_position_ids,
            past_key_values=copy.deepcopy(past_key_values),
            use_cache=True,
        )[0]
        .detach()
        .cpu()
        .float()
    )

    lang_ort = None
    lang_onnx_path = None
    if not args.skip_onnx or args.run_qpc:
        lang_ort, lang_onnx_path = _export_and_run_lang_onnx(
            transformed_model,
            input_ids=onnx_input_ids,
            position_ids=onnx_position_ids,
            past_key_values=copy.deepcopy(past_key_values),
            keep_onnx=args.keep_onnx,
        )

    if vision_onnx_path is not None:
        print("Vision ONNX path:", vision_onnx_path)

    if lang_ort is not None:
        lang_ref_vs_onnx = _stats(onnx_reference_logits, lang_ort, args.atol, args.rtol)
        print("Language ONNX path:", lang_onnx_path)
        print("Decode-step reference vs ONNXRuntime:", lang_ref_vs_onnx)

        if not lang_ref_vs_onnx["allclose"]:
            raise SystemExit("Parity check failed: language ONNXRuntime logits do not match decode-step reference.")

    if args.run_qpc:
        vision_qpc, vision_qpc_dir = _run_vision_qpc(
            vision_onnx_path,
            pixel_values=pixel_values,
            h_shape=h_shape,
            w_shape=w_shape,
            args=args,
        )
        vision_ref_vs_qpc = _stats(vision_reference, vision_qpc, args.qpc_atol, args.qpc_rtol)
        print("Vision QPC dir:", vision_qpc_dir)
        print(f"Vision reference vs QPC (atol={args.qpc_atol}, rtol={args.qpc_rtol}):", vision_ref_vs_qpc)

        if not vision_ref_vs_qpc["allclose"]:
            raise SystemExit("Parity check failed: vision QPC output does not match vision reference.")

        lang_qpc, lang_qpc_dir = _run_lang_qpc(
            lang_onnx_path,
            input_ids=onnx_input_ids,
            position_ids=onnx_position_ids,
            past_key_values=copy.deepcopy(past_key_values),
            args=args,
        )
        lang_ref_vs_qpc = _stats(onnx_reference_logits, lang_qpc, args.qpc_atol, args.qpc_rtol)
        print("Language QPC dir:", lang_qpc_dir)
        print(f"Decode-step reference vs QPC (atol={args.qpc_atol}, rtol={args.qpc_rtol}):", lang_ref_vs_qpc)

        if not lang_ref_vs_qpc["allclose"]:
            raise SystemExit("Parity check failed: language QPC logits do not match decode-step reference.")

    print("Parity check passed.")

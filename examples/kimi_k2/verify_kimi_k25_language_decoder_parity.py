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
from pathlib import Path

import numpy as np
import onnxruntime as ort
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
from transformers.dynamic_module_utils import get_class_from_dynamic_module

from QEfficient import QEFFAutoModelForImageTextToText
from QEfficient.generation.cloud_infer import QAICInferenceSession
from QEfficient.transformers.models.modeling_auto import QEffCausalLMForTextImageToTextModel


def parse_args():
    parser = argparse.ArgumentParser(
        description="Verify Kimi-K2.5 language decoder parity: HF LM logits vs transformed decoder-wrapper logits."
    )
    parser.add_argument("--model-path", type=Path, default=MODEL_PATH)
    parser.add_argument("--num-text-layers", type=int, default=2)
    parser.add_argument("--num-vision-layers", type=int, default=1)
    parser.add_argument("--expert-ids", type=str, default=",".join(str(x) for x in LOADED_EXPERT_IDS))
    parser.add_argument("--num-experts-per-token", type=int, default=NUM_EXPERTS_PER_TOKEN)
    parser.add_argument("--prompt", type=str, default="Describe this image.")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--atol", type=float, default=1e-4)
    parser.add_argument("--rtol", type=float, default=1e-4)
    parser.add_argument("--skip-onnx", action="store_true", help="Skip ONNX export/runtime parity check.")
    parser.add_argument(
        "--keep-onnx",
        action="store_true",
        help="Keep exported ONNX under ./aisand/onnx_export_verify_lang instead of a temporary directory.",
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
        default=Path("aisand/qpc_verify_lang"),
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


def _stats(a: torch.Tensor, b: torch.Tensor, atol: float, rtol: float):
    diff = (a - b).abs()
    return {
        "shape": tuple(a.shape),
        "max_abs": float(diff.max().item()),
        "mean_abs": float(diff.mean().item()),
        "rmse": float(torch.sqrt(torch.mean((a - b) ** 2)).item()),
        "allclose": bool(torch.allclose(a, b, atol=atol, rtol=rtol)),
    }


def _language_model_reference_logits(model, input_ids, attention_mask, position_ids, past_key_values):
    inputs_embeds = model.get_input_embeddings()(input_ids)
    outputs = model.language_model(
        inputs_embeds=inputs_embeds,
        attention_mask=attention_mask,
        position_ids=position_ids,
        past_key_values=past_key_values,
        use_cache=True,
    )
    return outputs[0].float()


def _create_prompt_inputs(tokenizer, prompt: str, batch_size: int):
    prompts = [prompt] * batch_size
    tokenized = tokenizer(prompts, return_tensors="pt", padding=True)

    input_ids = tokenized["input_ids"].to(torch.long)
    attention_mask = tokenized["attention_mask"].to(torch.long)
    position_ids = torch.where(attention_mask.bool(), attention_mask.cumsum(-1) - 1, -1)
    return input_ids, attention_mask, position_ids


def _export_and_run_onnx(
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

    if keep_onnx:
        export_dir = Path("aisand/onnx_export_verify_lang")
        export_dir.mkdir(parents=True, exist_ok=True)
    else:
        export_dir = Path(tempfile.mkdtemp(prefix="kimi_k25_verify_lang_onnx_"))

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


def _run_qpc(onnx_path: Path, input_ids: torch.Tensor, position_ids: torch.Tensor, past_key_values, args):
    if args.keep_qpc:
        qpc_dir = args.qpc_dir
        if qpc_dir.exists():
            shutil.rmtree(qpc_dir)
        qpc_dir.parent.mkdir(parents=True, exist_ok=True)
    else:
        qpc_dir = Path(tempfile.mkdtemp(prefix="kimi_k25_verify_lang_qpc_"))
        shutil.rmtree(qpc_dir)

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
    expert_ids = _parse_expert_ids(args.expert_ids)

    torch.set_grad_enabled(False)
    _ensure_torch_fx_import_compatibility()

    config = _prepare_config(args.model_path)
    kimi_cls = get_class_from_dynamic_module("modeling_kimi_k25.KimiK25ForConditionalGeneration", str(args.model_path))
    _patch_kimi_tie_weights_compat(kimi_cls)
    _patch_deepseek_init_weights_compat(kimi_cls)

    hf_model, tokenizer, _ = _load_layer_subset_model(
        model_path=args.model_path,
        kimi_cls=kimi_cls,
        config=config,
        num_vision_layers=args.num_vision_layers,
        num_text_layers=args.num_text_layers,
        loaded_expert_ids=expert_ids,
        num_experts_per_tok=args.num_experts_per_token,
        dtype=torch.float32,
    )
    hf_model = hf_model.eval()

    transformed_wrapper = QEFFAutoModelForImageTextToText(copy.deepcopy(hf_model))
    transformed_model = transformed_wrapper.model.eval()
    decoder_wrapper = transformed_model.get_qeff_language_decoder().eval()

    input_ids, attention_mask, position_ids = _create_prompt_inputs(
        tokenizer,
        prompt=args.prompt,
        batch_size=args.batch_size,
    )

    past_key_values = transformed_model.language_model.get_dummy_pkv_cache(
        transformed_model.config.text_config,
        args.batch_size,
        input_ids.shape[1],
    )

    reference_logits = (
        _language_model_reference_logits(
            transformed_model,
            input_ids,
            attention_mask,
            position_ids,
            past_key_values=copy.deepcopy(past_key_values),
        )
        .detach()
        .cpu()
    )
    transformed_decoder_logits = (
        decoder_wrapper(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            image_embeds=None,
            image_idx=None,
            past_key_values=copy.deepcopy(past_key_values),
            use_cache=True,
        )[0]
        .detach()
        .cpu()
        .float()
    )

    lm_vs_wrapper = _stats(reference_logits, transformed_decoder_logits, args.atol, args.rtol)
    print(
        "Loaded subset:",
        f"vision_layers={transformed_model.config.vision_config.vt_num_hidden_layers}",
        f"text_layers={transformed_model.config.text_config.num_hidden_layers}",
    )
    print("Prompt:", repr(args.prompt))
    print("Input shape:", tuple(input_ids.shape))
    print("Language model vs decoder-wrapper:", lm_vs_wrapper)

    if not lm_vs_wrapper["allclose"]:
        raise SystemExit(
            "Parity check failed: transformed decoder-wrapper logits do not match transformed language-model logits."
        )

    ort_logits = None
    onnx_path = None
    if not args.skip_onnx or args.run_qpc:
        ort_logits, onnx_path = _export_and_run_onnx(
            transformed_model,
            input_ids=input_ids,
            position_ids=position_ids,
            past_key_values=copy.deepcopy(past_key_values),
            keep_onnx=args.keep_onnx,
        )

    if ort_logits is not None:
        lm_vs_onnx = _stats(reference_logits, ort_logits, args.atol, args.rtol)
        wrapper_vs_onnx = _stats(transformed_decoder_logits, ort_logits, args.atol, args.rtol)
        print("Language ONNX path:", onnx_path)
        print("Language model vs ONNXRuntime:", lm_vs_onnx)
        print("Decoder-wrapper vs ONNXRuntime:", wrapper_vs_onnx)

        if not lm_vs_onnx["allclose"] or not wrapper_vs_onnx["allclose"]:
            raise SystemExit("Parity check failed: ONNXRuntime logits do not match transformed language references.")

    if args.run_qpc:
        qpc_logits, qpc_dir = _run_qpc(
            onnx_path,
            input_ids=input_ids,
            position_ids=position_ids,
            past_key_values=copy.deepcopy(past_key_values),
            args=args,
        )
        lm_vs_qpc = _stats(reference_logits, qpc_logits, args.qpc_atol, args.qpc_rtol)
        wrapper_vs_qpc = _stats(transformed_decoder_logits, qpc_logits, args.qpc_atol, args.qpc_rtol)
        print("Language QPC dir:", qpc_dir)
        print(f"Language model vs QPC (atol={args.qpc_atol}, rtol={args.qpc_rtol}):", lm_vs_qpc)
        print(f"Decoder-wrapper vs QPC (atol={args.qpc_atol}, rtol={args.qpc_rtol}):", wrapper_vs_qpc)

        if not lm_vs_qpc["allclose"] or not wrapper_vs_qpc["allclose"]:
            raise SystemExit("Parity check failed: QPC logits do not match transformed language references.")

    print("Parity check passed.")

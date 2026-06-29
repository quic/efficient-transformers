# -----------------------------------------------------------------------------
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------

import json
import time
from pathlib import Path

import onnx
import torch
from accelerate import init_empty_weights
from safetensors.torch import load_file, save_file
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

from QEfficient.exporter.weight_free import _default_weights_roots
from QEfficient.exporter.weight_spec import (
    ExternalDataFile,
    load_weight_spec,
    resolve_weight_spec_path,
    save_weight_spec,
)
from QEfficient.transformers.models.modeling_auto import QEFFAutoModelForCausalLM
from QEfficient.utils.run_utils import ApiRunner


def convert_checkpoint_to_fp32(onnx_path: Path, weight_spec_path: Path) -> None:
    """
    Load each safetensors checkpoint file, cast all tensors to FP32,
    save next to the ONNX, and update weight_spec.json to point there.

    This ensures the compiler sees matching dtypes between the ONNX (FLOAT)
    and the safetensors files (also FLOAT after conversion).
    """
    spec = load_weight_spec(weight_spec_path)
    export_dir = onnx_path.parent
    candidate_roots = _default_weights_roots(weight_spec_path, spec)
    local_files = [
        ExternalDataFile(
            path=f"model_{idx:04d}.safetensors" if len(spec.files) > 1 else "model.safetensors",
            format="safetensors",
        )
        for idx, _ in enumerate(spec.files)
    ]

    # Reuse previously materialized local safetensors even if a fresh export
    # rewrote the spec back to the original checkpoint paths.
    if local_files and all((export_dir / ext_file.path).is_file() for ext_file in local_files):
        print("Reusing existing local FP32 safetensors.")
        spec.files = local_files
        save_weight_spec(weight_spec_path, spec)
        _sync_embedded_extdata(onnx_path, weight_spec_path)
        return

    new_files = []
    for idx, ext_file in enumerate(spec.files):
        rel_path = Path(ext_file.path)
        abs_path = rel_path if rel_path.is_absolute() else None
        if abs_path is None:
            for root in candidate_roots:
                candidate = root / rel_path
                if candidate.exists():
                    abs_path = candidate
                    break
        if abs_path is None or not abs_path.exists():
            raise FileNotFoundError(f"Cannot resolve external data file: {ext_file.path}")

        tensors = load_file(str(abs_path))
        fp32_tensors = {k: v.to(torch.float32) for k, v in tensors.items()}

        out_name = f"model_{idx:04d}.safetensors" if len(spec.files) > 1 else "model.safetensors"
        save_file(fp32_tensors, str(export_dir / out_name))
        new_files.append(ExternalDataFile(path=out_name, format="safetensors"))
        print(f"  {abs_path.name}  ({next(iter(tensors.values())).dtype})  ->  {out_name}  (float32)")

    spec.files = new_files
    save_weight_spec(weight_spec_path, spec)
    _sync_embedded_extdata(onnx_path, weight_spec_path)


def _sync_embedded_extdata(onnx_path: Path, weight_spec_path: Path) -> None:
    # Keep the embedded external-data metadata aligned with weight_spec.json so
    # compiler and ORT verification resolve the same files.
    updated_json = json.dumps(json.loads(weight_spec_path.read_text()), separators=(",", ":"), sort_keys=True)
    onnx_model = onnx.load(str(onnx_path), load_external_data=False)
    for entry in onnx_model.metadata_props:
        if entry.key == "com.qti.aisw.extdata":
            entry.value = updated_json
            break
    tmp = onnx_path.with_suffix(onnx_path.suffix + ".tmp")
    onnx.save(onnx_model, str(tmp))
    tmp.replace(onnx_path)


# model_name = "meta-llama/Llama-3.3-70B-Instruct"
model_name = "meta-llama/Llama-3.2-1B"
# model_name = "gpt2"
# model_name = "hf-internal-testing/tiny-random-LlamaForCausalLM"

tokenizer = AutoTokenizer.from_pretrained(model_name)
config = AutoConfig.from_pretrained(model_name)
# config.num_hidden_layers = 2
config.dtype = torch.float32
print(config)

CONTINUOUS_BATCHING = False
FULL_BATCH_SIZE = 4  # slots in the KV cache; active batch_size stays at 1 here

runner = ApiRunner(
    batch_size=1,
    tokenizer=tokenizer,
    config=config,
    prompt=["My name is"],
    prompt_len=8,
    ctx_len=32,
    full_batch_size=FULL_BATCH_SIZE if CONTINUOUS_BATCHING else None,
)

with init_empty_weights():
    meta_model = AutoModelForCausalLM.from_config(config, attn_implementation="eager")

qeff_model = QEFFAutoModelForCausalLM(
    meta_model,
    pretrained_model_name_or_path=model_name,
    continuous_batching=CONTINUOUS_BATCHING,
)

export_dir = Path("test_models/weightfree_from_config")
export_start = time.perf_counter()
onnx_path = Path(
    qeff_model.export(
        export_dir=export_dir,
        use_dynamo=True,
        use_onnx_subfunctions=True,
        use_weight_free_export=True,
        offload_pt_weights=False,
    )
)
export_elapsed = time.perf_counter() - export_start
weight_spec_path = resolve_weight_spec_path(onnx_path)

print(f"Weight-free export time: {export_elapsed:.3f} sec")

# Uncomment to convert checkpoint weights to FP32 for compilation
# print("Converting checkpoint to FP32 (one-time local materialization) ...")
# convert_checkpoint_to_fp32(onnx_path, weight_spec_path)

# Uncomment to compile the exported ONNX to a QPC binary
# print("Compiling weight-free ONNX ...")
# qpc_path = qeff_model.compile(
#     onnx_path=str(onnx_path),
#     compile_dir=str(onnx_path.parent / "qpc"),
#     prefill_seq_len=8,
#     ctx_len=32,
#     use_dynamo=True,
#     use_onnx_subfunctions=True,
#     use_weight_free_export=True,
# )
# print(f"QPC: {qpc_path}")

# Uncomment for OnnxRuntime inference
# from QEfficient.exporter.weight_free import load_weight_free_ort_inputs
# import onnxruntime as ort
# import numpy as np
# session = ort.InferenceSession(str(onnx_path))
# ort_inputs = load_weight_free_ort_inputs(weight_spec_path, runner.input_handler.prepare_ort_inputs())
# ort_outputs = runner.run_ort_session(ort_inputs, session)
# ort_outputs = runner.input_handler.update_ort_outputs(ort_outputs)
# generated_ids = []
# for _ in range(1, runner.gen_len):
#     generated_ids.append(ort_outputs["logits"].argmax(-1).reshape(-1, 1))
#     ort_inputs = runner.input_handler.update_ort_inputs(ort_inputs, ort_outputs)
#     ort_inputs = load_weight_free_ort_inputs(weight_spec_path, ort_inputs)
#     ort_outputs = runner.run_ort_session(ort_inputs, session)
#     ort_outputs = runner.input_handler.update_ort_outputs(ort_outputs)
# generated_ids.append(ort_outputs["logits"].argmax(-1).reshape(-1, 1))
# generated_ids = np.concatenate(generated_ids, axis=1)
# generated_text = runner.input_handler.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
# print(generated_text)

# Uncomment for on-device QPC inference
# print("Running QPC generate ...")
# try:
#     exec_info = qeff_model.generate(
#         prompts=["My name is"],
#         tokenizer=tokenizer,
#         automation=True,
#         generation_len=runner.gen_len,
#     )
#     qpc_generated_ids = np.asarray(exec_info.generated_ids[0]).reshape(1, -1)
#     qpc_generated_text = tokenizer.batch_decode(qpc_generated_ids, skip_special_tokens=True)
#     print(exec_info)
#     print(qpc_generated_text)
# except RuntimeError as exc:
#     print(f"Skipping QPC generate: {exc}")

print(f"Weight-free ONNX: {onnx_path}")
print(f"Weight spec:      {weight_spec_path}")

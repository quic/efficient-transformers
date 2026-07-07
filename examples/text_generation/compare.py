# -----------------------------------------------------------------------------
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------

import json
import sys
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
import torch
from accelerate import init_empty_weights
from safetensors import safe_open
from safetensors.torch import load_file, save_file
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

from QEfficient.exporter.weight_free import _default_weights_roots, load_weight_free_ort_inputs
from QEfficient.exporter.weight_spec import (
    ExternalDataFile,
    load_weight_spec,
    resolve_weight_spec_path,
    save_weight_spec,
)
from QEfficient.transformers.models.modeling_auto import QEFFAutoModelForCausalLM
from QEfficient.utils.run_utils import ApiRunner

# Memory profiler — lives in scripts/memory_profiling relative to repo root
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "scripts"))
from memory_profiling import QEffMemoryProfiler


def convert_checkpoint_to_fp32(onnx_path: Path, weight_spec_path: Path) -> None:
    """
    Extract only the tensors referenced by weight_spec inputs, cast to FP32,
    save next to the ONNX, and update weight_spec.json to point there.

    This ensures the compiler sees matching dtypes between the ONNX (FLOAT)
    and the safetensors files (also FLOAT after conversion).
    Only files/tensors actually used by the exported graph are written —
    for a 2-layer model sliced from a 30-shard checkpoint this avoids
    loading the irrelevant shards entirely.
    """
    spec = load_weight_spec(weight_spec_path)
    export_dir = onnx_path.parent
    candidate_roots = _default_weights_roots(weight_spec_path, spec)

    if spec.files and all(not Path(f.path).is_absolute() and (export_dir / f.path).is_file() for f in spec.files):
        print("Reusing existing local FP32 safetensors.")
        return

    needed: dict[int, set[str]] = {}
    for inp in spec.inputs:
        file_idx = int(inp.location.file)
        needed.setdefault(file_idx, set()).add(inp.location.key)

    sorted_old_idxs = sorted(needed.keys())
    old_to_new = {old: new for new, old in enumerate(sorted_old_idxs)}

    new_files = []
    for old_idx in sorted_old_idxs:
        ext_file = spec.files[old_idx]
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

        keys_needed = needed[old_idx]
        with safe_open(str(abs_path), framework="pt") as f:
            already_fp32 = all(f.get_slice(k).get_dtype() == "F32" for k in keys_needed)

        if already_fp32:
            new_files.append(ExternalDataFile(path=str(abs_path), format="safetensors"))
            print(f"  {abs_path.name}  ({len(keys_needed)} tensors)  ->  referenced in place (already fp32)")
        else:
            tensors = load_file(str(abs_path))
            fp32_tensors = {k: v.to(torch.float32) for k, v in tensors.items() if k in keys_needed}
            out_name = f"model_{old_to_new[old_idx]:04d}.safetensors"
            save_file(fp32_tensors, str(export_dir / out_name))
            new_files.append(ExternalDataFile(path=out_name, format="safetensors"))
            print(f"  {abs_path.name}  ({len(keys_needed)}/{len(tensors)} tensors)  ->  {out_name}  (float32)")

    for inp in spec.inputs:
        inp.location.file = old_to_new[int(inp.location.file)]

    spec.files = new_files
    save_weight_spec(weight_spec_path, spec)
    _sync_embedded_extdata(onnx_path, weight_spec_path)


def _sync_embedded_extdata(onnx_path: Path, weight_spec_path: Path) -> None:
    updated_json = json.dumps(json.loads(weight_spec_path.read_text()), separators=(",", ":"), sort_keys=True)
    onnx_model = onnx.load(str(onnx_path), load_external_data=False)
    for entry in onnx_model.metadata_props:
        if entry.key == "com.qti.aisw.extdata":
            entry.value = updated_json
            break
    tmp = onnx_path.with_suffix(onnx_path.suffix + ".tmp")
    onnx.save(onnx_model, str(tmp))
    tmp.replace(onnx_path)


# ── Config ────────────────────────────────────────────────────────────────────

PROMPT = "what is faith ?"

# Four prompts of different lengths for continuous-batching (FBS) testing.
PROMPTS_FBS = [
    "what is faith ?",  # short   (~4 tokens)
    "Explain the concept of machine learning in simple terms.",  # medium  (~9 tokens)
    "The history of artificial intelligence dates back to the 1950s when",  # longer  (~14 tokens)
    "Once upon a time in a land far away, there lived a wise old wizard who",  # longest (~16 tokens)
]

# model_name = "Qwen/Qwen3-235B-A22B-Instruct-2507"
# model_name = "ibm-granite/granite-3.0-3b-a800m-instruct"
# model_name="Qwen/Qwen3-8B-A3B-Instruct-2507"
model_name = "openai-community/gpt2"
# model_name = "hpcai-tech/grok-1"
# model_name = "zai-org/GLM-4.7"
# model_name="meta-llama/Llama-3.2-1B"
# model_name="Qwen/Qwen3-32B"
# model_name="meta-llama/Llama-3.3-70B-Instruct"
# model_name = "Qwen/Qwen3-235B-A22B-Instruct-2507"
# model_name="Qwen/Qwen3-30B-A3B-Instruct-2507"
# model_name="/home/amarshar/qwen3-30b-prepared"
# model_name="/home/huggingface_hub/glm51-fp32-stacked"
# model_name="tiny-random/glm-4-moe"
# model_name="openai/gpt-oss-20b"

CONTINUOUS_BATCHING = True
FULL_BATCH_SIZE = 4

# ── Profiler ──────────────────────────────────────────────────────────────────

model_short = model_name.split("/")[-1]
profiler = QEffMemoryProfiler(
    sampling_interval=0.05,
    output_file=f"compare_profile_{model_short}.png",
    verbose=False,
    track_child_processes=True,
)
profiler.start_monitoring()

# ── Phase 1: Initialization ───────────────────────────────────────────────────

profiler.mark_operation("Initialization")
print(f"Model: {model_name}")

tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
config.dtype = torch.float32
# config.num_hidden_layers = 6
print(config)

# Apply chat template if the tokenizer has one, so instruction models get proper input format
if tokenizer.chat_template is not None:
    messages = [{"role": "user", "content": PROMPT}]
    formatted_prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
else:
    formatted_prompt = PROMPT

# For continuous batching use the 4 variable-length prompts; otherwise just the single prompt.
active_prompts = PROMPTS_FBS if CONTINUOUS_BATCHING else [formatted_prompt]

runner = ApiRunner(
    batch_size=1,
    tokenizer=tokenizer,
    config=config,
    prompt=active_prompts,
    prompt_len=32,
    ctx_len=256,
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

# ── Phase 2: Weight-free export ───────────────────────────────────────────────

profiler.mark_operation("Model Export")
print("\nExporting (weight-free) ...")

onnx_path = Path(
    qeff_model.export(
        export_dir=export_dir,
        use_dynamo=True,
        use_onnx_subfunctions=True,
        use_weight_free_export=True,
        offload_pt_weights=False,
    )
)
weight_spec_path = resolve_weight_spec_path(onnx_path)
print(f"ONNX: {onnx_path}")
print(f"Weight spec: {weight_spec_path}")

# ── Phase 3: Compilation ──────────────────────────────────────────────────────

profiler.mark_operation("Model Compilation")
print("\nCompiling weight-free ONNX ...")
qpc_path = qeff_model.compile(
    onnx_path=str(onnx_path),
    compile_dir=str(onnx_path.parent / "qpc"),
    prefill_seq_len=16,
    ctx_len=256,
    num_devices=1,
    mxfp6_matmul=False,
    mxint8_kv_cache=False,
    use_dynamo=True,
    use_onnx_subfunctions=True,
    use_weight_free_export=True,
    full_batch_size=FULL_BATCH_SIZE if CONTINUOUS_BATCHING else None,
)
print(f"QPC: {qpc_path}")

# # ── Phase 4: ORT inference ────────────────────────────────────────────────────

profiler.mark_operation("ORT Inference")
print("\n--- OnnxRT inference ---")

session = ort.InferenceSession(str(onnx_path))
ort_inputs = load_weight_free_ort_inputs(weight_spec_path, runner.input_handler.prepare_ort_inputs())
ort_outputs = runner.run_ort_session(ort_inputs, session)
ort_outputs = runner.input_handler.update_ort_outputs(ort_outputs)

ort_generated_ids = []
for _ in range(1, runner.gen_len):
    ort_generated_ids.append(ort_outputs["logits"].argmax(-1).reshape(-1, 1))
    ort_inputs = runner.input_handler.update_ort_inputs(ort_inputs, ort_outputs)
    ort_inputs = load_weight_free_ort_inputs(weight_spec_path, ort_inputs)
    ort_outputs = runner.run_ort_session(ort_inputs, session)
    ort_outputs = runner.input_handler.update_ort_outputs(ort_outputs)

ort_generated_ids.append(ort_outputs["logits"].argmax(-1).reshape(-1, 1))
ort_generated_ids = np.concatenate(ort_generated_ids, axis=1)
ort_generated_text = tokenizer.batch_decode(ort_generated_ids, skip_special_tokens=True)

# PT inference variables — set to None when PT section is skipped.
pt_generated_ids = None
pt_generated_text = None

# ── Phase 5: PT model loading ─────────────────────────────────────────────────

profiler.mark_operation("Model Loading")
print("\n--- PyTorch model loading ---")

pt_config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
pt_model = AutoModelForCausalLM.from_pretrained(
    model_name,
    config=pt_config,
    torch_dtype=torch.float32,
    ignore_mismatched_sizes=True,
    trust_remote_code=True,
)
pt_model.eval()

# ── Phase 5: PT inference ─────────────────────────────────────────────────────

profiler.mark_operation("Text Generation")
print("\n--- PyTorch inference ---")

input_ids = tokenizer(formatted_prompt, return_tensors="pt").input_ids
with torch.no_grad():
    pt_out = pt_model.generate(
        input_ids,
        max_new_tokens=runner.gen_len,
        do_sample=False,
        temperature=None,
        top_p=None,
    )

pt_generated_ids = pt_out[:, input_ids.shape[1] :].numpy()
pt_generated_text = tokenizer.batch_decode(pt_generated_ids, skip_special_tokens=True)
print("PT output:", pt_generated_text)

# ── Phase 6: QPC inference ────────────────────────────────────────────────────

profiler.mark_operation("Generation")
print("\n--- QPC inference ---")

qpc_generated_ids = None
qpc_generated_text = None
try:
    exec_info = qeff_model.generate(
        prompts=active_prompts,
        tokenizer=tokenizer,
        automation=True,
        generation_len=runner.gen_len,
    )
    qpc_generated_ids = np.asarray(exec_info.generated_ids[0]).reshape(1, -1)
    qpc_generated_text = tokenizer.batch_decode(qpc_generated_ids, skip_special_tokens=True)
    print(exec_info)
except RuntimeError as exc:
    print(f"Skipping QPC generate: {exc}")

# ── Stop profiler and report ──────────────────────────────────────────────────

profiler.stop_monitoring()
graph_file = f"compare_profile_{model_short}.png"
profiler.generate_memory_graph(graph_file)
print(profiler.get_memory_report())
print(f"\nMemory profile graph saved: {graph_file}")

# # ── Token comparison ──────────────────────────────────────────────────────────

print(f"\nPrompt: {PROMPT!r}")
print()
print(f"ORT  generated_ids : {ort_generated_ids}")
print(f"ORT  generated_text: {ort_generated_text}")
print()
if pt_generated_ids is not None:
    print(f"PT   generated_ids : {pt_generated_ids}")
    print(f"PT   generated_text: {pt_generated_text}")
else:
    print("PT   generated_ids : (skipped)")
print()
if qpc_generated_ids is not None:
    print(f"QPC  generated_ids : {qpc_generated_ids}")
    print(f"QPC  generated_text: {qpc_generated_text}")
else:
    print("QPC  generated_ids : (skipped)")

print("\n--- Per-token match (ORT vs QPC) ---")
max_len = max(
    ort_generated_ids.shape[1],
    qpc_generated_ids.shape[1] if qpc_generated_ids is not None else 0,
)
header = f"{'Step':>5}  {'ORT':>8}  {'PT':>8}  {'QPC':>8}  {'ORT==PT':>8}  {'ORT==QPC':>9}"
print(header)
print("-" * len(header))
# ort_tok='$'
for i in range(max_len):
    ort_tok = int(ort_generated_ids[0, i]) if i < ort_generated_ids.shape[1] else -1
    pt_tok = int(pt_generated_ids[0, i]) if i < pt_generated_ids.shape[1] else -1
    qpc_tok = int(qpc_generated_ids[0, i]) if (qpc_generated_ids is not None and i < qpc_generated_ids.shape[1]) else -1
    ort_eq_pt = "✓" if ort_tok == pt_tok else "✗"
    ort_eq_qpc = (
        "✓" if (qpc_generated_ids is not None and ort_tok == qpc_tok) else ("N/A" if qpc_generated_ids is None else "✗")
    )
    print(f"{i:>5}  {ort_tok:>8}  {pt_tok:>8}  {qpc_tok:>8}  {ort_eq_pt:>8}  {ort_eq_qpc:>9}")

---
name: qeff-layerwise-fp16-sensitivity
description: Debug numerical sensitivity for QEff models (LLM, VLM, diffusion) using CPU fp32-vs-fp16 MAD localization, boundary caching for fast restart, targeted fp32 promotion for non-matmul ops, mathematically equivalent scaling/rescaling for matmul-path instability, and validated NPI YAML generation from exact ONNX output tensor names.
---

# QEff Layerwise FP16 Sensitivity

Use this skill when a model has accuracy/numerical issues and the user provides only a model card (or model id) and task context. The agent should derive next steps autonomously.

## Contract

- CPU numerical analysis is source of truth before AIC compile/runtime decisions.
- Always localize first bad region by fp32 vs low-precision MAD/non-finite checks.
- Cache intermediate boundaries so reruns restart near failure, not from layer 0.
- If non-matmul ops are the issue, promote those op outputs to fp32 via NPI.
- If matmul/MLP path is the issue, prefer mathematically equivalent scaling/rescaling over blanket fp32.
- Final output must include a validated NPI YAML when fp32 promotion is part of the fix.

## Inputs Expected

- Required: model id/model card, modality/task (LLM, VLM, diffusion), and failure symptom.
- Optional: known bad prompts, seeds, expected outputs, existing debug artifacts.

If only model card is provided, proceed with defaults and report assumptions.

## Universal Workflow

### 1) Build deterministic reference and low-precision runs

- Fix seed(s), prompt/input, and inference settings.
- Collect fp32 reference activations/logits.
- Collect fp16/bf16 (and if relevant fp8 path) with same inputs.

### 2) Localize sensitivity region

- Compare block-wise outputs (MAD, max-abs, finite ratio).
- Stop at first block/layer/timestep where drift spikes or non-finite appears.

### 3) Enable restart cache

- Persist boundary states before sensitive region.
- On reruns, reload nearest cached boundary and continue from there.

### 4) Mitigate

- Non-matmul instability: promote affected post-ops (e.g. `Sigmoid/Mul/Add/Div/Softmax`) to fp32 via NPI.
- Matmul/MLP instability: apply algebraically equivalent scaling/rescaling so functional mapping is preserved.

### 5) Re-run end-to-end and verify

- Verify finite ratios restored.
- Verify MAD reduction vs fp32 reference.
- Verify output quality/token behavior/sample quality recovered.

### 6) Emit and validate NPI

- NPI entries must be exact ONNX output tensor names.
- For multi-output operators, use first output (`node.output[0]`).
- Validate every NPI entry exists in the target ONNX graph before handing off.

## Scale Factor Selection (MAD vs fp32)

Use this only when fp32 promotion of non-matmul post-ops is insufficient or too expensive.

Per problematic layer/stage:

1. Cache boundary state immediately before the region.
2. Run baseline fp16 from cache and record:
   - `finite_ratio`
   - `MAD(layer_out_fp16, layer_out_fp32)`
3. Sweep candidate scales (descending powers of 2 is preferred):
   - `1.0, 0.5, 0.25, ...`
4. For each candidate, re-run only from cached boundary and score:
   - hard gate: `finite_ratio == 1.0`
   - quality gate: MAD must drop versus baseline
5. Select the first fully finite candidate with lowest MAD (or earliest finite candidate if runtime budget is tight).

Do not accept a scale that restores finite values but worsens MAD materially versus fp32.

## Where To Scale And How To Compensate Back

Rule: apply scale on linear branches only, and compensate at linear merge points. Do not place inverse compensation across non-linear ops.

Decoder layer form:

- original: `h_l = r_l + m_l`
- scaled checkpoint exports `m_l' = s_l * m_l`
- runtime computes scaled sum:
  - `h_l' = s_l * r_l + m_l' = s_l * (r_l + m_l)`
- next layer input restores equivalence:
  - `x_{l+1} = h_l' / s_l = r_l + m_l`

Windowed decoding chain:

- pre-layer compensation: `x_l = h_{l-1}' / s_{l-1}` before layer `l` input norm.
- final-window compensation: divide by last layer scale before final norm/head.

For Qwen3.5-MoE recovery path, the offline scaling targets are MLP up branches (experts/shared-expert), with residual-branch compensation at runtime to preserve exact functional mapping.

## QEff Wiring (End-to-End)

1. Generate recovery evidence and scales from debug scripts:
   - `scripts/debug/qwen3_5_moe_full_depth_mlp_scale_recovery.py`
2. Emit recipe YAML from recovery result:
   - `QEfficient/utils/layer_scale_checkpoint.py` via `build_layer_scale_recipe_from_recovery_result(...)`
3. Apply recipe to checkpoint snapshot:
   - `apply_layer_scale_recipe_to_snapshot(...)`
   - writes scaled tensors + `qeff_layer_scales` metadata in `config.json`
4. Runtime consumes metadata automatically in QEff wrapper:
   - compensation logic in `QEffQwen3_5MoeDecoderLayer.forward` and `QEffQwen3_5MoeTextModel.forward`
5. Export merged ONNX from scaled runtime path (layerwise or full graph).
6. Generate/validate NPI YAML from merged ONNX output tensor names.
7. Compile and run AIC path; confirm outputs and drift versus fp32 reference.

## Family Tracks

### A) LLM/VLM (layerwise-capable)

For Qwen3.5-MoE/Qwen3-VL-MoE style models, use existing debug scripts:

```bash
python scripts/debug/qwen3_5_moe_chunked_precision_compare.py \
  --model-id <MODEL_ID> --chunk-size 1 --output-json scripts/debug/artifacts/<run>.json

python scripts/debug/qwen3_5_moe_window23_zoom.py \
  --model-id <MODEL_ID> --target-layer <LAYER> \
  --cache-dir scripts/debug/artifacts/<cache_dir> \
  --output-json scripts/debug/artifacts/<zoom>.json

python scripts/debug/qwen3_5_moe_full_depth_iterative_fp32_recovery.py \
  --model-id <MODEL_ID> --start-layer <LAYER> \
  --boundary-cache-dir scripts/debug/artifacts/<cache_dir> \
  --continuation-cache-dir scripts/debug/artifacts/<cont_cache> \
  --output-json scripts/debug/artifacts/<recovery>.json
```

For matmul-path recovery via scaling:

```bash
python scripts/debug/qwen3_5_moe_full_depth_mlp_scale_recovery.py \
  --model-id <MODEL_ID> --start-layer <LAYER> \
  --emit-recipe-yaml QEfficient/transformers/models/qwen3_5_moe/configs/<recipe>.yaml \
  --output-json scripts/debug/artifacts/<scale_recovery>.json
```

### B) Diffusion (UNet/Transformer/VAE/TextEncoder)

If no model-specific debug script exists, create one under `scripts/debug/` and follow same loop:

- compare fp32 vs fp16 per module/stage (UNet down/mid/up blocks, transformer blocks, VAE encode/decode, text encoder)
- cache latent/hidden boundaries per stage and (if needed) per timestep bucket
- apply non-matmul fp32 promotion first
- for matmul-path instability, apply branch scaling/rescaling with equivalence checks
- re-run generation with fixed seed and compare latent/image metrics vs fp32 baseline

Keep scheduler/noise seed fixed while debugging diffusion.

## NPI Generation And Validation

Use bundled helper script.

### Pattern 1: templated layer+ops (layerwise merged ONNX)

```bash
python \
  docs/skills/qeff-layerwise-fp16-sensitivity/scripts/emit_npi_from_merged_onnx.py \
  --onnx /path/to/final_data/merged_0-*.onnx \
  --layers 22,26,30,54 \
  --ops shared_expert/gate_proj/MatMul,shared_expert/up_proj/MatMul,shared_expert/down_proj/MatMul,Sigmoid,Mul_4,Add \
  --output-yaml /path/to/fp32_nodes.yaml
```

### Pattern 2: explicit node names (works for diffusion/all models)

```bash
python \
  docs/skills/qeff-layerwise-fp16-sensitivity/scripts/emit_npi_from_merged_onnx.py \
  --onnx /path/to/model.onnx \
  --node-names "/unet/down_blocks.0/resnets.0/Add,/unet/mid_block/attn/Softmax" \
  --output-yaml /path/to/fp32_nodes.yaml
```

### Validate an existing NPI YAML against ONNX

```bash
python \
  docs/skills/qeff-layerwise-fp16-sensitivity/scripts/emit_npi_from_merged_onnx.py \
  --onnx /path/to/model.onnx \
  --validate-yaml /path/to/fp32_nodes.yaml \
  --validate-only
```

## Required Deliverables

- First problematic region with MAD/non-finite evidence file paths
- Chosen mitigation and why (fp32 promotion vs scaling/rescaling)
- Exact ONNX used for NPI extraction
- NPI YAML path
- Validation summary: `missing=0` against ONNX outputs

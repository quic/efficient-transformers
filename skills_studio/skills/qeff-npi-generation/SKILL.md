---
name: qeff-npi-generation
description: Generate, refine, or review QEfficient node-precision-info (NPI) YAML files for QAIC compiles, especially when isolating fp32 islands for fp16-converted ONNX/QPC parity. Use for NPI node selection, Clip+producer handling, HF/QPC parity experiments, and Qwen3.5-MoE layerwise debugging.
---

# QEff NPI Generation

## Purpose
Use this skill when creating or debugging `node_precision_info` YAML for QEfficient / QAIC compilation. The goal is to keep the smallest required set of ONNX node outputs in fp32 while the compiler otherwise converts the graph to fp16.

## Core Rules
- First check open-source reports/issues for the target model's fp16 behavior. Public issues often point to the fragile op family (for example SSM/Mamba exponentials, MoE routing/accumulation, norm/reduction, or attention gates) and should guide the first NPI hypotheses.
- Treat NPI as a compiler precision hint for **node output names**: the compiler keeps nodes whose output name is listed under `FP32NodeInstanceNames` in fp32.
- The goal of NPI generation is to make QPC outputs match the PyTorch reference, not merely to make compile succeed.
- QPC normally executes with fp16 conversion for these flows. If `mxfp6_matmul=True` is passed to compile, MatMul ops are quantized to MXFP6 while the remaining ops stay in fp16 unless NPI keeps selected nodes in fp32.
- If adding a modeling `Clip` before returning to fp16, include **both**:
  - the inserted `Clip` node output
  - the output of the node immediately feeding that `Clip`
- Do not use `custom_io` for internal fp32 nodes; reserve it for graph inputs/outputs and retained-state IO precision.
- Avoid broad NPI lists first. Broad non-MatMul or all-node NPI can exceed QAIC VA/static memory limits.
- Prefer proving HF/QPC parity without NPI first. If baseline QPC mismatches HF, NPI is not yet the right lever.

## Required Setup
- Use the user-provided venv and cache if supplied.
- Set a scratch `QEFF_HOME` for experiments unless the user specifies one:
  - `export QEFF_HOME=$PWD/agent_space/<experiment>/qeff_home`
- Do not download/load a model without confirming `HF_HUB_CACHE` if it was not already provided.

## Workflow
1. Open-source triage first.
   - Search model/runtime issues for fp16 accuracy failures, overflow, NaNs, repeated-token output, or recommended dtype workarounds.
   - Translate those reports into first-pass suspect ops before generating broad NPI.
2. Export a minimal graph first.
   - For huge models, use layerwise export and start with a small `num_hidden_layers`.
   - For Qwen3.5-MoE, one-layer compile is not reliable; use at least 2 layers and `layerwise_window_size=1`.
3. Establish baseline parity before NPI.
   - Compare HF float32 against QPC generated tokens.
   - Run at least the requested number of output tokens; if the user does not specify, start with 32 greedy tokens.
   - HF should use plain model inputs; do not pass QEff retained-state `past_key_values` to HF.
   - QPC should follow the model example path: `prepare_inputs_for_generation(...)` then `qeff_model.generate(...)`.
4. If baseline compile fails, fix compile/export issues before accuracy NPI.
   - NPI cannot prove accuracy if the QPC cannot compile or run.
   - Record exact compiler command and stderr.
5. If baseline compiles but mismatches HF, create focused NPI candidates.
   - Start with known high-risk ops: `Exp`, norm/reduction outputs, recurrent-state update outputs, MoE router/expert combine outputs, and any inserted `Clip` + producer pairs.
   - If we can't control the fact that an operations output will go beyond fp16 range, then we can add a `torch.clip(out, fp32_min, fp32_max)` operation in pytorch modeling file if by doing this the output of final model doesn't change a lot and we can be within e-1, e-2 range of MAD between pytorch an QPC execution.
   - Avoid MatMul outputs unless there is direct evidence; MatMul-heavy NPI often causes VA space failures.
6. Iterate narrowly.
   - Generate one candidate NPI per hypothesis. The candidate need not be just single node it can be a pattern found in the ONNX graph.
   - Compile and run the same HF/QPC token comparison.
   - Keep the smallest NPI that matches.


## NPI File Shape
Use this YAML shape:

```yaml
FP32NodeInstanceNames:
  - <onnx-output-name-1>
  - <onnx-output-name-2>
```

For an intentionally empty matching NPI:

```yaml
FP32NodeInstanceNames: []
```

## Node Name Discovery
- Load ONNX with `onnx.load(path, load_external_data=False)`.
- Inspect both `model.graph.node` and `model.functions[*].node`; subfunction exports hide many useful node names in functions.
- Match by output names, op type, and surrounding producer/consumer edges.
- For compiler-reported instantiated names, note that names may include prefixes such as `Decode_<FunctionName>_<instance>_<output>`. If direct ONNX output names do not work, first verify whether the issue is a compiler/export bug rather than missing NPI.

## Validation Checklist
- NPI YAML exists and is passed as `node_precision_info=<path>`; confirm the compile command contains `-node-precision-info=<path>`.
- QPC compiles successfully before claiming an NPI candidate.
- HF/QPC comparison uses identical prompt, tokenizer/processor, model config layer count, and generation length.
- For Qwen3.5-MoE QPC input prep, use the example-style `prepare_inputs_for_generation`; do not hand-build retained states unless testing low-level raw QPC IO.
- Report:
  - ONNX path
  - QPC path
  - NPI path
  - exact token match result
  - any compiler warnings/errors that constrain NPI size or node choice

# Llama 3.2 1B Export/Compile Timing and Output Match Test

## Summary
- Use `examples/text_generation/basic_inference.py` as the behavioral test path.
- Instrument only `QEfficient/base/modeling_qeff.py`, where both relevant calls live:
  - `torch.onnx.export(...)`
  - `subprocess.run(command, ...)` for `/opt/qti-aic/exec/qaic-compile`
- Compare timings and generated output with ONNX passes enabled vs disabled.

## Key Changes
- In `QEfficient/base/modeling_qeff.py`, add precise timing around both `torch.onnx.export(...)` call sites:
  - Print `[onnx export subprocess-independent profile] path=... time=...s`.
  - Preserve existing behavior and exceptions.
- Keep or add exact compiler timing around `subprocess.run(...)`:
  - Print `[compiler subprocess profile] time=...s`.
- Use the existing env-controlled ONNX pass disablement:
  - Enabled baseline: `QEFF_ONNX_DISABLE_SAFE_EXPORT_PASSES=0`
  - Disabled experiment: `QEFF_ONNX_DISABLE_SAFE_EXPORT_PASSES=1`
- Ensure export hashing includes the env setting so enabled/disabled runs do not reuse the same ONNX/QPC artifacts.
- Do not assume layerwise `final_data` layout for Llama; record the exact ONNX/export directory and exact `qpc_path` returned by `model.compile(...)`.

## Test Procedure
- Run `examples/text_generation/basic_inference.py` twice with identical args except ONNX pass env:
  - Model: `meta-llama/Llama-3.2-1B`
  - Prompt: `Hello, how are you?`
  - `prefill_seq_len=32`
  - `ctx_len=128`
  - `generation_len=32`
  - `num_cores=16`
  - `aic_hw_version=ai100`
- Save logs:
  - `/tmp/llama32_1b_passes_enabled.log`
  - `/tmp/llama32_1b_passes_disabled.log`
- Save generated outputs:
  - `agent_space/llama32_1b_passes_enabled_output.txt`
  - `agent_space/llama32_1b_passes_disabled_output.txt`
- Save a concise JSON report:
  - `agent_space/llama32_1b_pass_disable_timing_report.json`
- For each mode, verify the returned `qpc_path` directly with `(Path(qpc_path) / "programqpc.bin").exists()`; do not infer the QPC path from a hardcoded directory structure.

## Acceptance Criteria
- Both runs generate a QPC.
- Both logs contain:
  - ONNX export timing from `torch.onnx.export(...)`
  - exact compiler subprocess timing from `qaic-compile`
- The generated text from `.generate(...)` matches exactly between passes-enabled and passes-disabled runs.
- The report includes:
  - ONNX export time
  - compiler subprocess time
  - total compile-profile time if present
  - exact returned QPC path
  - actual ONNX/export path or export root observed in logs
  - generated text
  - `outputs_match`

## Assumptions
- `meta-llama/Llama-3.2-1B` is available through local cache or configured HF credentials.
- Exact generated text equality is sufficient to detect accuracy differences for this smoke test.
- The two runs should use fresh artifacts or distinct hash inputs to avoid timing cached exports/compiles.
- For this non-layerwise Llama path, expected cache layout may resemble `<architecture>/<export_hash>/qpc-<compile_hash>/qpc`; the agent must not assume `final_data`.

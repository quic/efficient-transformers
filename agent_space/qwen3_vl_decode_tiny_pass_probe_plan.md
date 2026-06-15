# Find Safely Disableable ONNX Export Passes

## Summary
- Create a copy of `examples/image_text_to_text/models/qwen3_vl_moe/qwen3_vl_moe_layerwise_decode.py` for experiments.
- Use `tiny-random/qwen3-vl-moe` and decode-only settings (`prefill_seq_len=1`, `skip_vision=True`) for fast QPC validation.
- Automatically disable candidate export passes greedily; keep a pass disabled only if `qaic-compile` still produces a valid QPC.

## Key Changes
- Add new script `examples/image_text_to_text/models/qwen3_vl_moe/qwen3_vl_moe_layerwise_decode_tiny_pass_probe.py`.
- Configure the copied script for:
  - `MODEL_ID = "tiny-random/qwen3-vl-moe"`
  - `num_devices=1`, `num_cores=16`, `layerwise_window_size=1`
  - `skip_vision=True`, `use_onnx_subfunctions=True`, `prefill_seq_len=1`
  - no post-compile `generate()` call; success criterion is QPC creation only.
- Add script-local pass monkeypatching around `torch.onnx.export`; restore all patched functions after each compile attempt.
- Patch QEff export hashing in the script so each attempted pass set gets a unique ONNX/QPC cache path and never reuses stale artifacts.
- Treat compile success as:
  - returned path or dict contains a QPC directory, and
  - `programqpc.bin` exists inside that QPC directory.

## Pass Search
- Run a baseline first with no disabled passes; abort the experiment if tiny decode baseline does not compile.
- Use greedy search:
  - Start with no disabled passes.
  - Try disabling one additional candidate pass.
  - If QPC compiles, keep that pass disabled.
  - If QPC fails, restore that pass and record the compiler error.
  - Continue until all candidates are tested.
- Candidate passes to test:
  - `_jit_pass_constant_propagation`
  - `_jit_pass_dce`
  - `_jit_pass_cse`
  - `_jit_pass_canonicalize_graph_fuser_ops`
  - `_jit_pass_peephole`
  - `_jit_pass_fuse_addmm`
  - `_jit_pass_onnx_peephole`
  - `_jit_pass_onnx_eval_peephole`
  - `_jit_pass_onnx_constant_fold`
  - `_jit_pass_dce_allow_deleting_nodes_with_side_effects`
  - `_jit_pass_canonicalize`
  - `_jit_pass_onnx_graph_shape_type_inference`
  - `_jit_pass_onnx_set_dynamic_input_shape`
  - `_jit_pass_onnx_deduplicate_initializers`
- Do not disable required lowering/conversion passes in the first sweep:
  - `_jit_pass_onnx`
  - `_jit_pass_onnx_preprocess`
  - `_jit_pass_onnx_remove_inplace_ops_for_onnx`
  - `_jit_pass_lower_all_tuples`
  - `_jit_pass_erase_number_types`
  - `_jit_pass_onnx_scalar_type_analysis`
  - `_jit_pass_onnx_assign_output_shape`

## Reporting
- Print one summary row per attempt:
  - attempted pass, kept/failed, export time, compile time, QPC path or compiler error.
- Print final safe disable set and unsafe pass list.
- Save full logs with:
  - `HF_HUB_CACHE=/home/huggingface_hub ~/envs/qeff/bin/python examples/image_text_to_text/models/qwen3_vl_moe/qwen3_vl_moe_layerwise_decode_tiny_pass_probe.py 2>&1 | tee /tmp/qwen3_vl_decode_tiny_pass_probe.log`

## Test Plan
- Syntax check the new script:
  - `~/envs/qeff/bin/python -m py_compile examples/image_text_to_text/models/qwen3_vl_moe/qwen3_vl_moe_layerwise_decode_tiny_pass_probe.py`
- Run baseline-only mode first to confirm the tiny decode script can produce a QPC.
- Run greedy mode and collect the final safe/unsafe pass sets.
- If the final safe set is non-empty, rerun once with all safe passes disabled together to confirm the final QPC still builds.

## Assumptions
- `tiny-random/qwen3-vl-moe` is the intended tiny model; it exists on Hugging Face and uses `qwen3_vl_moe`.
- A pass is “safe to disable” only if `qaic-compile` generates `programqpc.bin`.
- This remains an experiment script; production QEff export behavior is not changed until the safe pass set is known.

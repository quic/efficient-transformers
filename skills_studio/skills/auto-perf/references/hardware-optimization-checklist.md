# Hardware Optimization Checklist (AI 100 + compiler-glow)

Use this checklist to choose the next PyTorch-side optimization hypothesis for
the `export -> compile -> run` pipeline targeting AI 100.

## Scope and Assumptions
- Target hardware: AI 100 Standard card.
- Execution mode: AI 100 hardware (`-aic-hw`, `-aic-hw-version=2.0/ai100`).
- Precision policy: MXFP6 for MatMul weights (`-mxfp6-matmul`) and MXINT8 for
  KV/cache paths when enabled.
- Core topology in practice: 14 NSP used in standard-card runs, while compiler
  code paths support up to 16 AI100 cores.

## 1. Hardware Facts to Design Around
- AI100 HMX supports Int8 and FP16 compute types for matmul-like workloads.
- Per-core fast memory is constrained:
  - L2TCM options: 256KB or 512KB.
  - VTCM options on AI100: 5/6/7/8MB (AI100 is limited to 8MB max).
  - Compiler defaults to 8MB VTCM on AI100 when not overridden.
- VTCM working-set pressure is a first-order limiter:
  - `-vtcm-working-set-limit-ratio` caps per-instruction VTCM usage.
  - Oversized live ranges spill to DDR and add copy/multicast traffic.
- Data-path alignment matters:
  - HVX vector size is 128 bytes.
  - HMX tile/alignment is 2048 bytes.
- PMU recipes can expose whether kernels are compute-bound or memory-bound:
  `KernelUtil` (HMX/HVX/UDMA utilization), `HmxMacs`, `AxiRd/Wr`.

## 2. Compiler Feature Inventory (What Exists)
- Graph optimizer has broad fusion/simplification passes:
  DCE, CSE, SinkCode, quantization optimization, reshape/transpose cleanup,
  MatMul/BatchMatMul/FC folds, LayerNorm/RMSNorm folds, concat/slice cleanup.
- IR optimizer includes memory and liveness cleanup:
  DSE, ShareBuffers, HoistDealloc, Insert/Extract optimization, peephole passes.
- AIC backend supports multiple scheduling/splitting modes:
  inter-device partitioning, inter-core splitting, intra-core size splitting,
  depth-first tiling/scheduling, and cluster-based partitioning.
- Quantization toolchain is mature:
  PGQ profile generation/merge, external quantization profiles, rowwise and
  channelwise modes, per-node precision overrides, MXFP6 and MXINT8 paths.
- Runtime/compile diagnostics are available:
  `-time-passes`, `-aic-perf-metrics`, `-aic-perf-warnings`, `-aic-op-stats`,
  `-aic-pmu-recipe`, `-aic-pmu-events`, `-ddr-stats`.

## 3. Strengths to Lean On
- Strong graph canonicalization can eliminate many redundant
  transpose/reshape/convert chains before backend codegen.
- Aggressive constant/weight transforms exist for HMX paths
  (including 32-lane-style packing/reordering and MXFP6 packing flows).
- Memory-aware scheduling/allocation is sophisticated:
  VTCM placement, spill/fill, multicast, and delayed deallocation controls.
- Retained-state and split-model IO support can preserve long-context decode
  throughput when model IO naming/layout is consistent.
- Compiler and runtime provide enough counters to quickly separate:
  compute underutilization vs DDR pressure vs scheduling imbalance.

## 4. Known Limitations and Perf Traps
- Automatic split coverage is not universal for large matmul-like workloads:
  input-channel split support has strict constraints in backend transforms.
- Unsupported/limited option combinations:
  - Single-device partitioning (SDP) + depth-first scheduling: not supported.
  - SDP + legacy stats levels 40-60: not supported.
  - SDP + MDP at the same time: not supported.
  - `directAPI` on AI100: not supported.
- MXINT8 custom IO limitations:
  - Layout conversion with MXINT8 custom IO is not supported.
  - MXINT8 axis handling is fixed to last dimension in current path.
  - Block sizing is tied to 32-element granularity.
- Batch-mode caveat:
  setting batch size for multiple model inputs is currently unsupported in
  `InferenceManager::processInputs`.
- Some scheduling experiments are disabled by default due regressions
  (e.g., TensorView hoisting), so do not assume latent flags improve perf.

## 5. PyTorch-Level Optimization Checklist
- Keep matmul/attention dimensions compiler-friendly:
  prefer stable static shapes and dimensions aligned to 32 where possible.
- Remove avoidable layout churn:
  collapse `permute/transpose/reshape/contiguous` chains around hot ops.
- Fuse cheap epilogues into larger kernels when semantically safe.
- Reduce intermediate tensor lifetimes:
  avoid materializing full-score/full-activation tensors when blockwise
  streaming is possible.
- Avoid tiny-kernel fragmentation in decode loops.
- Keep control flow static across iterations for export/compile stability.
- Maintain KV-cache layout/precision consistency across prefill/decode.
- Prefer formulations that map to known optimized patterns
  (LayerNorm/RMSNorm/GELU/FastGELU/MaskedSoftmax families where applicable).

## 6. QEfficient Attention Blocking Playbook (Auto + Manual)
Use this when optimizing `modeling*.py` attention paths and when extending the
same blocking logic to other large matmuls.

- Source implementation:
  - `efficient-transformers/QEfficient/blocking/attention_blocking.py`
  - `efficient-transformers/QEfficient/blocking/blocking_configurator.py`
  - `efficient-transformers/QEfficient/blocking/blocked_attention_forwards.py`
- Runtime strategy model:
  - Modes are explicit and dispatch-based: `NONE`, `KV`, `Q`, `H`, `QKV`, `HQKV`.
  - `generic_blocked_attention_interface` gates blocked execution and selects
    the forward kernel by mode.
  - KV-blocked decode requires cache support for `read_only_blockedKV`; otherwise
    fallback to non-blocked cache update path.
  - Effective blocking should be disabled when computed `num_q_blocks == 1` and
    `num_kv_blocks == 1` to avoid pointless loop overhead.
- Auto block-size/config selection (what to copy for new kernels):
  - Candidate block counts are generated with increasing step size, not full
    exhaustive search (`block_candidates_generator`), to keep tuning cheap.
  - Working-set estimate per NSP is modeled as:
    `q_size + kv_size + qk_size`, where:
    - `q_size ~ heads_per_iter * bs * q_tokens_per_block * head_dim * bytes`
    - `kv_size ~ heads_per_iter * bs * kv_tokens_per_block * head_dim * bytes`
    - `qk_size ~ heads_per_iter * bs * q_tokens_per_block * kv_tokens_per_block * bytes`
  - Head parallel factor is hardware-aware:
    `head_block_size = num_socs` in head-blocking mode and
    `heads_per_iter = ceil(head_block_size / num_socs)`.
  - Pick smallest total block count (`num_q_blocks * num_kv_blocks`) that keeps
    footprint under `VTCM_SIZE_THRESHOLD`; tie-break with better Q/KV balance
    (`q_kv_ratio` closer to 1.0).
- Kernel structure that improves accelerator performance:
  - Use streaming softmax accumulation (running max + running denominator) across
    KV blocks to avoid materializing full attention score matrices.
  - Split along Q/KV/head with static loop shapes, then concatenate once at end.
  - Reuse grouped-KV expansion helper (`repeat_kv`) only when needed by GQA/MQA.
  - Keep ONNX/tracing-safe control flow:
    avoid eager-only `break` in exported paths; use tensorized `torch.where`
    guards (`skip_future`) for graph stability.
  - Regenerate causal mask per block range and validate mask shape before apply.
- Trigger conditions:
  - low HMX utilization + high DDR traffic,
  - frequent VTCM spill/copy activity,
  - performance cliffs at longer sequence/context lengths.
- Guardrails:
  - Preserve numerics; validate max drift against baseline.
  - Ensure ONNX export stays legal and compilable after each blocking change.
  - Change one blocking hypothesis at a time.

## 7. Reuse Pattern for Non-Attention Big Matmuls
Apply the same principles from Q/KV/head blocking to FFN/projection matmuls.

- Use the same 3-term memory model style:
  - input tile footprint,
  - weight tile footprint,
  - partial/output tile footprint.
- Prefer minimal tile count satisfying VTCM threshold over aggressive
  over-splitting; excess tiles increase launch/sync overhead.
- Keep tile dimensions aligned to hardware-friendly granularity (typically 32)
  and stable across iterations for compiler consistency.
- Replace full intermediate materialization with streamed partial accumulation
  when reduction axis is large.
- Keep layout stable around matmul tiles; avoid per-tile transpose/contiguous.
- Preserve exportability: static loop bounds and tensor ops over Python control
  flow in hot paths.

## 8. Iteration Quality Gate (Mandatory)
For every optimization attempt:
- Export success.
- Compile success.
- Runtime measured under fixed benchmark settings (same BS/SL/CL/precision).
- Accuracy within tolerance.
- Keep/revert decision logged with rationale.

## 9. Plateau Escalation
If gains flatten:
- Enable richer counters (`opstats` + PMU recipe `KernelUtil`/`HmxMacs` + DDR).
- Rank top kernels by runtime share and memory traffic.
- Map hottest kernels back to `modeling*.py` code regions.
- Prioritize 1-2 highest-impact changes:
  1) block/retile big matmuls/attention,
  2) remove layout conversions and temporary tensors around those kernels.

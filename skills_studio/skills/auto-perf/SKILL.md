---
name: auto-perf
description: Optimize PyTorch model implementation for lower latency and higher throughput on a custom AI accelerator by iterating through an existing export-to-ONNX, compile, and hardware runtime script. Use when tasks involve modifying model code (especially modeling files), validating performance changes on device, and applying hardware-aware tactics based on cache, memory hierarchy, and compute architecture details.
---

# Auto Perf

## Objective
Improve end-to-end model runtime on target hardware by editing PyTorch model code, then re-running the existing export -> compile -> hardware execution pipeline.

This skill also supports analysis-only profiling turns where the goal is to identify bottlenecks and classify the workload without editing code.

## Required Inputs
Collect these before making changes:
- Hugging Face model card/id to pass as `--model-name`
- User-provided model config file path (`config.json`). Example:
  `<HF_HOME>/hub/models--<org>--<model>/snapshots/<rev>/config.json`
- Baseline command line and baseline runtime metrics (`ttft_seconds`, `decode_tokens_per_sec`)
- Accuracy acceptance criteria (max tolerated numerical drift)
- User-provided experiment log file path (CSV file path, absolute or repo-relative)
- ONNX graph signal from `run_inference.py` output (the printed ONNX export path)

For analysis-only profiling turns, use the existing ONNX, opstats summary files, and merged trace JSON files when the user provides them.

## Authorized Edit Scope (Strict)
Only these two files may be edited during this skill workflow:
- `efficient-transformers/QEfficient/transformers/models/llama/modeling_llama.py`
- `efficient-transformers/QEfficient/transformers/models/pytorch_transforms.py`

Do not edit any other file.

Implementation location details:
- QEfficient model classes are implemented in
  `efficient-transformers/QEfficient/transformers/models/llama/modeling_llama.py`.
- Mapping from upstream `transformers` code to hardware-specific classes (defined in `modeling_llama.py`) is implemented in
  `efficient-transformers/QEfficient/transformers/models/pytorch_transforms.py`.

Blocking reference implementation has moved to:
- `blocking/blocking_configurator.py`
- `blocking/attention_blocking.py`
- `blocking/blocked_attention_forwards.py`

Use these blocking files as references only unless explicitly instructed otherwise.

## Required Runtime Setup And Command
Use this exact flow for baseline and every optimization iteration:

1. Activate the Python environment in bash:
```bash
source <your-qeff-venv>/bin/activate
```

2. Run the export -> compile -> run pipeline:
```bash
python efficient-transformers/examples/autoperf/run_inference.py --model-name <huggingface-model-card>
```

Expected script behavior:
- It reports decode performance (tokens/sec) and either:
  - prefill TTFT directly, or
  - enough prefill metrics to derive TTFT.
- It prints an error when accuracy deviates beyond tolerance.
- It prints the exported ONNX file path in a line like:
  `model exported to ONNX format at: <path-to-model.onnx>`

Treat this script as the source of truth for both correctness gating and perf comparison.

## Execution Rules
- Establish a baseline first; never optimize without a measured starting point.
- Change one performance hypothesis at a time.
- Keep ONNX/export compatibility after every edit.
- Run the same benchmark settings each iteration (batch size, sequence length, precision, input shape).
- Run `run_inference.py` exactly once per optimization attempt.
- Capture and persist `run_inference.py` output each attempt; extract the ONNX path from that output.
- Parse the ONNX graph and use graph hotspots (operator types, shape/volume, layout churn) to drive the next hypothesis.
- After the experiment verdict is decided (keep/revert), delete the parent directory of the exported ONNX file if it exists. This cleanup is mandatory before the next experiment attempt starts.
- Continue the optimization loop until one of these stop conditions is met:
  - User explicitly stops the run.
  - 50 attempts have been completed.
- Reject changes that violate accuracy tolerance even if runtime improves.
- Ask the user once for the experiment log file location before baseline.
- Ask the user once for the model `config.json` path before baseline.
- Parse and use `config.json` hyperparameters as the source of truth for tensor/matmul shapes.
- If parent directories or the file do not exist, create them automatically.
- Use a single append-only pandas-backed CSV log file for baseline and every attempt.
- Log exactly these columns (and no extras): `timestamp`, `ttft_seconds`, `decode_tokens_per_sec`, `accuracy_match`, `summary_of_changes`.
- If accuracy does not match tolerance after a change, rollback that change immediately.
- If performance degrades after a change (TTFT increases or decode tokens/sec decreases vs baseline/current kept best), rollback that change.
- For every kept change, create a local git commit from the repo root with a one-line summary commit message.
- Before choosing a code-edit hypothesis, classify the run using available profiling data as one of:
  - `compute bound`
  - `memory / IO bound`
  - `layout / fragmentation bound`
  - `mixed`
- If the run is memory / IO bound, prioritize reducing gather/scatter, DDR->VTCM staging, dequant / format conversion, or cache / retained-state movement before pure GEMM-fusion hypotheses.

## Analysis-Only Mode

Use analysis-only mode when the user wants to:
- identify bottlenecks
- determine whether the workload is compute-bound vs IO-bound
- map opstats hotspots to attention / MLP / MoE / cache-update blocks
- inspect overlap vs serialization
- study repeated blocked subgraphs

In analysis-only mode:
- do not require code edits
- do not require the baseline/edit/re-run loop
- do not create commits
- use the existing ONNX, opstats summaries, and merged trace JSONs
- return:
  - bottleneck classification
  - top op kind families
  - top logical ONNX/model paths
  - repeated-block summary
  - suspected missed-overlap opportunities
  - suggested next code areas to inspect

## Opstats Interpretation Guide

When QAIC opstats summary files and merged trace JSON files are available, classify the bottleneck before proposing code edits.

### Read these summary fields first
For each core or slice, inspect:
- `DMAActivePct`
- `HVXActivePct`
- `HMXActivePct`
- DDR traffic count / bandwidth
- top `opdetails` rows by `ucycles`
- `sync HMX`, `sync HVX`, `barrier HMX`, `barrier HVX`, `sync DMAIssue`

Interpretation:
- If `DMAActivePct` is materially higher than both `HVXActivePct` and `HMXActivePct`, the run is likely memory / movement bound.
- If `HMXActivePct` is dominant and compute kernels clearly outrank movement kernels, the run is likely compute bound.
- If `HVXActivePct` is elevated and hot ops are mostly small elementwise, normalization, reshape, transpose, convert, reduce, or scatter/gather support kernels, the run may be layout- or fragmentation-bound.
- High DDR traffic with modest HMX/HVX utilization usually indicates staging, layout churn, gather/scatter, dequant, or weak locality.
- High `sync*` / `barrier*` rows indicate waiting, but they are symptoms. Trace back to the expensive preceding work instead of optimizing synchronization directly.

### Common low-level op kind heuristics
Use these generic interpretations:
- `aiccopytovtcm DMAIssue DDR`: DDR -> VTCM staging. Large values usually mean IO pressure.
- `aicgather DMAIssue DDR`: irregular gather from DDR. Often indicates expert fetch, cache lookup, indexing-heavy paths, or non-contiguous weight fetch.
- `aicscatternd`, `CtxScatter`: cache update or scatter-heavy path.
- `aicmulticastvtcm`: on-chip distribution / replication. Large values can indicate blocked attention fanout or repeated reuse of staged tensors.
- `blockdequantize_*`, `aicconvertweightstod32`: quantized weight unpack / format conversion overhead.
- `aicconvolutiond32`: commonly backs MatMul/GEMM-like regions in exported graphs.
- `aicconverttod32`, `aicconvertfromd32`, `aictranspose`, `aiccopysamevtcm2d`: layout conversion / format movement overhead around compute.
- `aicsoftmax`: attention probability step or similar normalized reduction.
- `aicnormalizationpre/post`: RMSNorm / LayerNorm support kernels.
- `aicmmap`, `aicmunmap`: mapping overhead, often around irregular gather-heavy regions.

### Recommended first-pass classification
Classify each run into one of:
- `compute bound`
- `memory / IO bound`
- `layout / fragmentation bound`
- `mixed`

Heuristic:
- `memory / IO bound`: DMA dominates, DDR traffic is high, and top rows are copy/gather/multicast/dequant.
- `compute bound`: HMX dominates and compute kernels clearly outrank movement kernels.
- `layout / fragmentation bound`: HVX dominates with many small convert/transpose/reshape/reduce/elementwise ops.
- `mixed`: none of the above is clearly dominant.

Record the classification in experiment notes or the final report.

## Relating Opstats To Model Architecture

Do not stop at low-level op kind summaries. Map hardware hotspots back to ONNX node names and then to model structure.

### Procedure
1. Use the merged trace JSON and collect events with:
   - `ph == "X"`
   - excluding `sync *`, `Core-*`, and `Device_*`
2. Normalize `opName` by stripping trailing generated ids such as `__1234`.
3. Sum event durations by normalized path to rank logical model paths.
4. Also compute wall-time span for each normalized path:
   - `start = min(ts)`
   - `end = max(ts + dur)`
5. Use ONNX node names to map hotspots back to architecture blocks.

### Architecture mapping heuristics
Typical mappings:
- `*/q_proj*`, `*/q_a_proj*`, `*/k_proj*`, `*/kv_a_proj*`, `*/v_proj*` -> attention projections
- repeated `MatMul_<n>` families with `Softmax`, `Where`, `Transpose`, `Slice`, `Concat` nearby -> blocked attention subgraphs
- `CtxScatter`, `CtxGather`, retained-state names -> cache movement / update
- `*/mlp/gate_proj*`, `*/mlp/up_proj*`, `*/mlp/down_proj*` -> dense MLP
- `*/mlp/gate/*`, `TopK`, `GatherElements`, `ScatterElements` -> MoE routing path
- `*/mlp/Gather_*` on expert tensors -> selected-expert weight or data fetch
- `*/mlp/shared_experts/*` -> shared-expert path
- `input_layernorm`, `post_attention_layernorm`, `q_*_layernorm`, `kv_*_layernorm` -> normalization blocks

### Repeated-block analysis
For blocked attention, grouped experts, or repeated heads:
- sum the full repeated family, not just one node
- identify whether the repeated family is dominated by:
  - score/value matmuls
  - transpose / reshape / concat / slice
  - gather / scatter
  - dequant / format conversion
- report both:
  - the hottest single repeated node
  - the hottest repeated family total

### Reporting expectations
Be able to produce conclusions such as:
- “attention is dominated by repeated score matmuls plus transpose/cache-update tax”
- “dense MLP is dominated by gate/up/down projection matmuls”
- “MoE is gather-bound, not router-TopK-bound”
- “cache writes are material while cache reads are minor”

## Parallelism And Serialization Analysis

When merged trace JSON is available, explicitly check whether independent DAG branches are actually overlapping.

### How to analyze
For each logical ONNX path:
- compute cumulative work: `sum(dur)`
- compute wall span: `min(ts)` to `max(ts + dur)`

Use:
- `sum(dur)` to rank hotspots
- wall-span overlap to detect scheduler behavior

### Important distinction
- A large `sum(dur)` with a much smaller wall span means the runtime is internally sharding or parallelizing that path.
- Lack of overlap between sibling paths can indicate:
  - graph-imposed serialization, or
  - scheduler / resource underlap

### How to separate the two
Use ONNX graph structure:
- If path B consumes path A output, it is graph-imposed serialization.
- If paths A and B are sibling consumers of the same producer but barely overlap, treat it as a missed-overlap candidate.

### What to report
For each suspected case, state:
- producer / sibling relationship from ONNX
- path start/end times from trace
- overlap or gap
- whether the serialization is graph-imposed or likely runtime-imposed

## Workflow
Loop behavior:
- This is a continuous optimization loop.
- Each try consists of exactly one edit hypothesis and one execution of:
  `python efficient-transformers/examples/autoperf/run_inference.py --model-name <huggingface-model-card>`.
- Do not stop the loop early unless the user asks to stop, or 50 tries are reached.

Model config pre-check (before Step 1):
- Ask user for the `config.json` path and validate it exists and is valid JSON.
- Treat the config values as mandatory constraints for all code edits.
- Read and use shape-driving hyperparameters, including:
  - `hidden_size`
  - `intermediate_size`
  - `num_hidden_layers`
  - `num_attention_heads`
  - `num_key_value_heads`
  - `head_dim` (if absent, derive from `hidden_size / num_attention_heads`)
  - `max_position_embeddings`
  - `vocab_size`

1. Initialize experiment log
- Ask the user for the log file path.
- Normalize and ensure parent directory exists.
- If the file does not exist, create it as an empty pandas DataFrame with this exact schema:
  `["timestamp", "ttft_seconds", "decode_tokens_per_sec", "accuracy_match", "summary_of_changes"]`.
- Always append new rows to this same file; never overwrite prior experiment rows.

2. Capture baseline
- Activate the environment: `source <your-qeff-venv>/bin/activate`.
- Run once while saving stdout/stderr:
  ```bash
  RUN_LOG=$(mktemp -t run_inference_baseline.XXXXXX.log)
  python efficient-transformers/examples/autoperf/run_inference.py --model-name <huggingface-model-card> 2>&1 | tee "$RUN_LOG"
  ```
- Record compile success/fail, accuracy status, TTFT (seconds), decode tokens/sec, and any memory stats (if available).
- Parse TTFT as:
  - preferred: direct TTFT metric from script output;
  - fallback: `ttft_seconds = prompt_len / prefill_tokens_per_sec` when direct TTFT is not printed.
- Extract ONNX path and parse graph summary:
  ```bash
  python skills_studio/skills/auto-perf/scripts/onnx_graph_analysis.py --run-output-log "$RUN_LOG"
  ```
- Confirm extracted ONNX path exists before proceeding.
- Append one baseline row to the log file with:
  - `timestamp`: current ISO-8601 time
  - `ttft_seconds`: parsed or derived from script output
  - `decode_tokens_per_sec`: parsed from script output
  - `accuracy_match`: `true` if within tolerance, else `false`
  - `summary_of_changes`: `baseline`

3. Select one optimization hypothesis
- Choose one bottleneck or hardware-alignment issue.
- Use both `config.json` and ONNX graph evidence to prioritize the most expensive operator shapes first.
- When opstats summary files or merged trace JSON files are available, use them to verify whether the bottleneck is compute, movement, or layout driven before selecting a code-edit hypothesis.
- Prefer hotspots from ONNX analysis in this order:
  1) large `MatMul`/`Gemm`/attention-related nodes,
  2) repeated layout-conversion chains (`Transpose`/`Reshape`/`Concat`/`Slice`),
  3) high-frequency small elementwise ops fragmenting kernels.
- If movement or irregular access dominates, prioritize:
  1) gather/scatter reduction,
  2) DDR->VTCM staging reduction,
  3) dequant / format conversion reduction,
  4) retained-state/cache movement reduction,
  before pure GEMM fusion ideas.
- Reference [hardware-optimization-checklist.md](references/hardware-optimization-checklist.md) to pick a targeted change.

4. Edit PyTorch modeling code
- Prefer local, reversible edits.
- Keep semantics unchanged unless the user explicitly allows approximation.
- Ensure all reshape/view/transpose logic remains compatible with `config.json` dimensions.
- Restrict edits to:
  `efficient-transformers/QEfficient/transformers/models/llama/modeling_llama.py`
  and
  `efficient-transformers/QEfficient/transformers/models/pytorch_transforms.py`.

5. Re-run pipeline
- Execute the same environment activation and run command, while saving output to a fresh run log:
  `source <your-qeff-venv>/bin/activate`
- Save output to a fresh run log, then extract ONNX path and parse graph again:
  ```bash
  RUN_LOG=$(mktemp -t run_inference_try.XXXXXX.log)
  python efficient-transformers/examples/autoperf/run_inference.py --model-name <huggingface-model-card> 2>&1 | tee "$RUN_LOG"
  python skills_studio/skills/auto-perf/scripts/onnx_graph_analysis.py --run-output-log "$RUN_LOG"
  ```
- Confirm export and compile still succeed.
- Collect correctness and performance metrics from script output (especially TTFT for prefill, decode tokens/sec, and any accuracy deviation error).

6. Compare and decide
- If improvement is real and correct, keep the change.
- If accuracy fails or performance regresses, rollback/revert and test the next hypothesis.
- If runtime/export/compile fails due to shape mismatch against `config.json`, rollback immediately.
- Mandatory cleanup after verdict and before next attempt:
  ```bash
  python skills_studio/skills/auto-perf/scripts/onnx_graph_analysis.py --run-output-log "$RUN_LOG" --cleanup-parent-dir --cleanup-only
  ```
- Append one row for the attempt to the same log file using the exact five columns:
  - `timestamp`
  - `ttft_seconds`
  - `decode_tokens_per_sec`
  - `accuracy_match`
  - `summary_of_changes` (short human summary of the code hypothesis tested)

7. Commit kept change locally
- Run git commands from the repo root.
- Commit only when the change is kept (accuracy passes and performance does not degrade).
- Use a one-line concise commit message summarizing the exact optimization.
- Use the helper script for commit+push of intended files:
  ```bash
  bash skills_studio/skills/auto-perf/scripts/commit_and_push_intended_files.sh "llama: fuse rotary ops and remove redundant transpose"
  ```

8. Finalize
- Report best variant, measured gain vs baseline, and key code diffs.
- Provide remaining bottlenecks and next-ranked hypotheses.

## Logging Implementation (Pandas CSV)
Use the shared implementation at:
- `scripts/experiment_logging.py`

Import and call:
- `ensure_log_file(log_path: str) -> Path`
- `append_experiment_log(log_csv: Path, ttft_seconds: float, decode_tokens_per_sec: float, accuracy_match: bool, summary_of_changes: str) -> None`

## ONNX Graph Parsing Implementation
Use:
- `scripts/onnx_graph_analysis.py`

Capabilities:
- Extract ONNX path from `run_inference.py` output log via:
  `extract_onnx_path_from_text(text: str) -> Path`
- Load ONNX and (when possible) run shape inference:
  `analyze_onnx_graph(onnx_path: Path) -> GraphSummary`
- Emit hotspot nodes and optimization hypotheses based on:
  - expensive ops (`MatMul`, `Gemm`, attention-like ops),
  - large static output volumes,
  - layout churn (`Transpose`/`Reshape`/`Concat`/`Slice`) around hot regions.
- When merged trace JSON is available, pair ONNX graph analysis with trace-path aggregation:
  - normalize `opName`
  - sum durations per logical path
  - compute wall spans per logical path
  - identify repeated families and sibling overlap

CLI examples:
```bash
python skills_studio/skills/auto-perf/scripts/onnx_graph_analysis.py --run-output-log /tmp/run.log
python skills_studio/skills/auto-perf/scripts/onnx_graph_analysis.py --onnx-path /abs/path/model.onnx --json
python skills_studio/skills/auto-perf/scripts/onnx_graph_analysis.py --run-output-log /tmp/run.log --cleanup-parent-dir --cleanup-only
```

## Commit/Push Helper
Use:
- `scripts/commit_and_push_intended_files.sh`

Behavior:
- Stages only:
  - `QEfficient/transformers/models/llama/modeling_llama.py`
  - `QEfficient/transformers/models/pytorch_transforms.py`
- Creates one local commit with the message you pass.
- Pushes current branch to `origin`.

## High-Value PyTorch Optimization Targets
Prioritize these in `modeling*.py` files:
- Reduce redundant `permute`, `transpose`, `contiguous`, and shape-conversion chains.
- Minimize small-kernel fragmentation by combining cheap elementwise ops where possible.
- Favor static, compiler-friendly control flow and stable tensor shapes.
- Reduce unnecessary intermediate tensors to lower memory traffic.
- Align tensor layouts and access patterns with accelerator-preferred formats.
- Avoid fallback-prone ops that degrade compilation quality.
- Replace Python-side loops with vectorized tensor operations when equivalent.

## Hardware-Aware Strategy
When hardware details are available, use them explicitly:
- Cache/memory hierarchy: prioritize data locality and tensor reuse.
- Limited on-chip memory: shorten tensor lifetimes and reduce activation footprint.
- Compute array geometry: align dimension ordering/tiling to maximize utilization.
- DMA or transfer overhead sensitivity: reduce host-device or memory movement steps.

Load [hardware-optimization-checklist.md](references/hardware-optimization-checklist.md) when choosing hypotheses or diagnosing plateaus.

## Output Format
Return a concise optimization report each cycle:
- Baseline TTFT (seconds) and decode tokens/sec
- Current TTFT (seconds) and decode tokens/sec
- Delta (absolute and percent)
- Accuracy status
- Compile/export status
- Code changes tested
- Keep/revert decision
- Next hypothesis

When opstats are used, also report:
- bottleneck class (`compute`, `memory / IO`, `layout / fragmentation`, `mixed`)
- top low-level op kinds by cost
- top logical model paths by cost
- repeated-block findings
- overlap / serialization findings

## Failure Handling
- If export fails: isolate last code edit and restore ONNX-friendly patterns.
- If compile fails: inspect unsupported ops and replace with supported equivalents.
- If runtime regresses: revert and test the next memory- or layout-focused hypothesis.
- If all hypotheses stall: request deeper hardware profiling counters and operator-level timing.

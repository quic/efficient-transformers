# Variable Speculative Decode Specializations

## Background

Speculative decoding on QAIC hardware requires **statically compiled shapes**. Each distinct
input shape must be registered as a "specialization" in `specializations.json` before
`qaic-compile` is invoked. The compiler produces a single QPC binary that dispatches to the
right kernel at runtime based on which input shape is presented.

Before this change, `QEFFAutoModelForCausalLM.compile()` accepted `num_speculative_tokens: int`
and compiled exactly **two** entries:

1. **Prefill** — `seq_len = prefill_seq_len`, `num_logits_to_keep = 1`
2. **TLM decode** — `seq_len = K+1`, `num_logits_to_keep = K+1`

This hard-wired a single proposal length. For PLD/n-gram methods, a second "fallback"
specialization (`seq_len=1`) was available via the now-removed `enable_fallback_decode_spec`
flag. For suffix decoding — where average proposal lengths of 1 or 2 are typical — there was
no way to compile for the right mix.

---

## What Was Implemented

### 1. `num_speculative_tokens: Optional[List[int]]` in `compile()` (`modeling_auto.py`)

The parameter type changed from `Optional[int]` to `Optional[List[int]]`. Each element `K`
compiles one decode specialization: `seq_len = K+1`, `num_logits_to_keep = K+1`. The list is
sorted and deduplicated before processing.

Passing a plain `int` still works — it is silently promoted to `[K]` for backward compatibility
with existing code.

`enable_fallback_decode_spec` is removed. Its old behavior is exactly
`num_speculative_tokens=[0, K]`.

### 2. `_build_decode_spec_for_k(k, ...)` private method (`modeling_auto.py`)

Replaces both `build_decode_specialization` (for TLM) and the removed
`build_fallback_decode_specialization`. Builds one decode specialization dict for a given K.
Returns `None` if the spec would duplicate the prefill spec (guards against `seq_len=1` when
`prefill_seq_len=1` and continuous batching is disabled).

### 3. `_select_k(actual_proposals, decode_ks)` helper (`prompt_lookup.py`)

Picks the **smallest K in `decode_ks` that covers the maximum actual proposal count** in the
current batch. This ensures the cheapest specialization is used per iteration while still
covering every batch item.

`actual_proposals` is an integer array of shape `[batch]` containing the number of non-fill
tokens in each item's proposal slots (`input_ids[:, 1:]`). Items with no valid proposals have
count 0; items with a full match have count `max_k`; items with a partial n-gram match
(continuation shorter than `max_k`, e.g. near the end of the prompt history) have a count
between 1 and `max_k − 1`.

### 4. Multi-spec runtime dispatch (`prompt_lookup.py`)

`pld_spec_decode_inference()` now accepts `num_speculative_tokens: Union[int, List[int]]`.
Per-K logit buffers are pre-allocated. Each decode iteration calls `_select_k`, dispatches to
the matching specialization, and pads the logit output back to `[batch, max_K+1, vocab]` so
the downstream token acceptance logic is unchanged.

---

## Files Changed

| File | Change |
|------|--------|
| `QEfficient/transformers/models/modeling_auto.py` | `num_speculative_tokens: Optional[List[int]]`, added `_build_decode_spec_for_k()`, replaced decode/fallback block with K-loop, removed `enable_fallback_decode_spec` |
| `QEfficient/base/modeling_qeff.py` | `_compile()`: write flat-format specializations.json for qaic-compile; strip `_graph_name` tag; convert values to strings |
| `examples/performance/speculative_decoding/prompt_lookup.py` | `_select_k()` helper, per-K buffer allocation, multi-spec dispatch, updated arg parser |
| `tests/transformers/spd/test_pld_inference.py` | `test_multi_spec_structure` (4 parametrized cases), `test_select_k` (5 parametrized cases) |
| `tests/unit_test/models/test_modeling_auto_cpu.py` | `TestTLMMultiSpecSpecializations` (8 tests) |

---

## Specializations.json — Examples

All examples use `prefill_seq_len=32`, `ctx_len=128`, `batch_size=1`, `full_batch_size=1`
(continuous batching enabled).

Each specialization entry contains:

| Field | Meaning |
|-------|---------|
| `seq_len` | Number of input tokens this kernel accepts |
| `num_logits_to_keep` | Number of output logits returned (always equals `seq_len` for TLM decode) |
| `ctx_len` | Static KV-cache allocation size |
| `batch_size` | Batch size for this phase |
| `full_batch_size` | Continuous-batching full-batch size |

---

### Case 1 — Baseline (plain int, backward compat)

```python
model.compile(num_speculative_tokens=4)   # treated internally as [4]
```

```json
{
  "specializations": [
    {
      "seq_len": "32",
      "num_logits_to_keep": "1",
      "ctx_len": "128",
      "batch_size": "1",
      "full_batch_size": "1"
    },
    {
      "seq_len": "5",
      "num_logits_to_keep": "5",
      "ctx_len": "128",
      "batch_size": "1",
      "full_batch_size": "1"
    }
  ]
}
```

| Entry | Role |
|-------|------|
| `seq_len=32` | **Prefill** — processes full prompt, returns 1 logit |
| `seq_len=5` | **TLM decode** — K=4 draft tokens + 1 anchor = 5 positions, returns 5 logits |

---

### Case 2 — PLD with fallback (replaces `enable_fallback_decode_spec=True`)

```python
model.compile(num_speculative_tokens=[0, 4])
```

```json
{
  "specializations": [
    {
      "seq_len": "32",
      "num_logits_to_keep": "1",
      "ctx_len": "128",
      "batch_size": "1",
      "full_batch_size": "1"
    },
    {
      "seq_len": "1",
      "num_logits_to_keep": "1",
      "ctx_len": "128",
      "batch_size": "1",
      "full_batch_size": "1"
    },
    {
      "seq_len": "5",
      "num_logits_to_keep": "5",
      "ctx_len": "128",
      "batch_size": "1",
      "full_batch_size": "1"
    }
  ]
}
```

| Entry | Role |
|-------|------|
| `seq_len=32` | **Prefill** |
| `seq_len=1` | **Fallback decode (K=0)** — used when no n-gram matches are found. Cheap single-token forward pass; avoids running the full K+1 kernel wastefully |
| `seq_len=5` | **Full TLM decode (K=4)** — used when n-gram proposals are available |

At runtime, `_select_k` dispatches to `seq_len=1` when all valid batch items have 0 proposals,
and to `seq_len=5` when any item has proposals (even a partial match). For `decode_ks=[0, 4]`
the dispatch is effectively binary because `find_candidate_pred_tokens` either produces a full
4-token continuation or 0; use `[0, 1, 2, 3, 4]` (Case 4) if you want fine-grained dispatch
for partial continuations near the end of the prompt history.

---

### Case 3 — Intermediate proposal lengths (PLD near end-of-history, or suffix decoding)

```python
model.compile(num_speculative_tokens=[1, 2])
```

```json
{
  "specializations": [
    {
      "seq_len": "32",
      "num_logits_to_keep": "1",
      "ctx_len": "128",
      "batch_size": "1",
      "full_batch_size": "1"
    },
    {
      "seq_len": "2",
      "num_logits_to_keep": "2",
      "ctx_len": "128",
      "batch_size": "1",
      "full_batch_size": "1"
    },
    {
      "seq_len": "3",
      "num_logits_to_keep": "3",
      "ctx_len": "128",
      "batch_size": "1",
      "full_batch_size": "1"
    }
  ]
}
```

| Entry | Role |
|-------|------|
| `seq_len=32` | **Prefill** |
| `seq_len=2` | **Short decode (K=1)** — used when max proposal in batch is 1 |
| `seq_len=3` | **Longer decode (K=2)** — used when max proposal in batch is 2 |

---

### Case 4 — Full range (maximum fine-grained dispatch)

```python
model.compile(num_speculative_tokens=[0, 1, 2, 3, 4])
```

```json
{
  "specializations": [
    {
      "seq_len": "32",
      "num_logits_to_keep": "1",
      "ctx_len": "128",
      "batch_size": "1",
      "full_batch_size": "1"
    },
    {
      "seq_len": "1",
      "num_logits_to_keep": "1",
      "ctx_len": "128",
      "batch_size": "1",
      "full_batch_size": "1"
    },
    {
      "seq_len": "2",
      "num_logits_to_keep": "2",
      "ctx_len": "128",
      "batch_size": "1",
      "full_batch_size": "1"
    },
    {
      "seq_len": "3",
      "num_logits_to_keep": "3",
      "ctx_len": "128",
      "batch_size": "1",
      "full_batch_size": "1"
    },
    {
      "seq_len": "4",
      "num_logits_to_keep": "4",
      "ctx_len": "128",
      "batch_size": "1",
      "full_batch_size": "1"
    },
    {
      "seq_len": "5",
      "num_logits_to_keep": "5",
      "ctx_len": "128",
      "batch_size": "1",
      "full_batch_size": "1"
    }
  ]
}
```

The hardware can dispatch to the cheapest kernel that covers the actual proposal distribution
each decode iteration. Trade-off: 6 specializations vs 2 means a larger QPC binary and longer
compile time, but maximum throughput efficiency at inference.

---

## Runtime Dispatch Logic

```
decode iteration
       │
       ├── for each batch item: count actual proposals
       │       0              → no n-gram match found
       │       1 .. max_k−1  → partial n-gram match (continuation shorter than max_k,
       │                        e.g. match is near the end of the prompt history)
       │       max_k          → full n-gram match
       │
       ├── selected_k = _select_k(max_actual_proposals, decode_ks)
       │       └── smallest K in decode_ks such that K >= max_actual_proposals
       │           (clamps to max(decode_ks) if need exceeds all)
       │
       ├── set logit buffer to logit_buffers[selected_k]  (shape [batch, K+1, vocab])
       ├── slice input_ids, position_ids to [:, :selected_k+1]
       ├── run TLM kernel for selected_k
       │
       └── if selected_k < max_k:
               pad output → [batch, max_k+1, vocab]  (zeros for unused positions)
           acceptance logic unchanged regardless of selected_k
```

---

## Unit Tests Written

**Total new tests: 17.** Combined with the 41 pre-existing unit tests, the full no-hardware
SPD suite is **58 tests, all passing**.

### `TestTLMMultiSpecSpecializations` — `test_modeling_auto_cpu.py` (8 tests, CPU-only)

| Test | What it verifies |
|------|-----------------|
| `test_build_decode_spec_for_k_seq_len` | `_build_decode_spec_for_k(k=K)` → `spec["seq_len"] == K+1` for K in {0,1,3,7} |
| `test_build_decode_spec_for_k_num_logits_to_keep` | Same method → `spec["num_logits_to_keep"] == K+1` |
| `test_build_decode_spec_for_k_returns_none_when_duplicate_prefill` | K=0, `prefill_seq_len=1`, no CB → returns `None` (would duplicate prefill) |
| `test_build_decode_spec_for_k_not_none_with_continuous_batching` | Same scenario but CB=True → spec returned (CB always needs decode) |
| `test_compile_list_produces_correct_spec_count` | `[0,3]` → 1 prefill + 2 decode entries |
| `test_compile_deduplication` | `[3,3,3]` → only 1 decode spec |
| `test_compile_sorting` | `[3,1,2]` → decode specs appear in ascending `seq_len` order |
| `test_compile_int_backward_compat` | `num_speculative_tokens=3` (plain int) → treated as `[3]`, 1 decode spec with `seq_len=4` |

### `test_multi_spec_structure` — `test_pld_inference.py` (4 parametrized, CPU-only)

Each `decode_ks` from `[[3], [0, 3], [1, 2, 3], [0, 1, 2, 3]]`:
- Every K produces `spec["seq_len"] == K+1` and `spec["num_logits_to_keep"] == K+1`
- All seq_lens are distinct (no duplicate specs generated)

### `test_select_k` — `test_pld_inference.py` (5 parametrized, CPU-only)

| Scenario | `actual_proposals` | `decode_ks` | Expected K |
|---|---|---|---|
| All zeros — all items missed n-gram | `[0, 0, 0]` | `[0, 3]` | `0` |
| Mix — some items have proposals | `[0, 3, 3]` | `[0, 3]` | `3` |
| Single spec — no choice | `[0, 0]` | `[3]` | `3` |
| Exact mid-point — picks smallest fitting K | `[1, 2]` | `[0, 2, 4]` | `2` |
| Need exceeds all — clamps to max | `[5, 5]` | `[0, 3]` | `3` |

---

## Errors Encountered

### 1. `TypeError: 'int' object is not iterable`

**Cause:** The initial implementation used `sorted(set(num_speculative_tokens))` unconditionally.
Existing tests in `test_spd_inference.py` and `test_pld_inference.py` pass a plain `int`
directly to `compile()`, which caused `set(4)` to raise.

**Fix:** Added a type guard before normalizing:

```python
_decode_ks = (
    sorted(set(num_speculative_tokens))
    if isinstance(num_speculative_tokens, (list, tuple))
    else ([num_speculative_tokens] if num_speculative_tokens is not None else None)
)
```

---

### 2. Duplicate `@pytest.mark.parametrize("model_id")` decorator

**Cause:** The old test function `test_fallback_decode_spec_structure` already had a
`@pytest.mark.parametrize("model_id", ...)` decorator. When replacing the function body, the
replacement text included the decorator again, resulting in it appearing twice and pytest
raising `duplicate parametrization of 'model_id'`.

**Fix:** Removed the duplicate, leaving exactly one `@pytest.mark.parametrize("model_id")`
and one `@pytest.mark.parametrize("decode_ks")`.

---

### 3. Syntax error — missing newline in `arg_parse()`

**Cause:** When removing the `--enable-fallback-decode-spec` argument block, the closing `)`
of the preceding `add_argument` call merged onto the same line as the next `add_argument`
call, producing a `SyntaxError: Simple statements must be separated by newlines`.

**Fix:** Restored the newline between the two statements.

---

### 4. Orphaned test method body (`assert "logits" in output_names`)

**Cause:** The string `assert "logits" in output_names` appeared in two different test
methods in `test_modeling_auto_cpu.py`. The Edit tool matched the first occurrence (in a
different class), replacing it — and leaving the CTC test's method body without its `def`
line, causing an `IndentationError`.

**Fix:** Restored the `def test_export_onnx_has_logits_output(...)` method header and renamed
the local variable to `output_names_ctc` to make it unique.

---

### 5. `qaic-compile: malloc.c: sysmalloc: Assertion` (SIGABRT) — pre-existing

**Status: Pre-existing SDK bug, not caused by this change.**

All hardware compile tests (`test_spd_inference` and `test_pld_inference` full/few/dummy
variants) fail with `Compiler exitcode: -6` — a C-level heap assertion inside the
`qaic-compile` binary itself:

```
qaic-compile: malloc.c:2617: sysmalloc: Assertion
  `(old_top == initial_top (av) && old_size == 0) || ...` failed.
```

The generated `specializations.json` is correct — verified by reading the file from disk
before the compiler crash:

```json
{
  "specializations": [
    { "seq_len": "32", "num_logits_to_keep": "1", "ctx_len": "128", ... },
    { "seq_len": "5",  "num_logits_to_keep": "5", "ctx_len": "128", ... }
  ]
}
```

The crash reproduces identically regardless of which specialization configuration is used and
is not related to this change.

> **Update (2026-05-08):** Resolved after SDK upgrade. See Hardware Test Results below.

---

### 6. Named-format specializations.json rejected by MDP firmware (`Failed to create ExecObj`)

**Discovered during:** vLLM integration testing on 4-device Llama-3.1-8B.

**Cause:** `_compile()` in `QEfficient/base/modeling_qeff.py` previously called
`to_named_specializations()` to wrap each specialization entry as:

```json
{"name": "Prefill_0", "symbols": {"batch_size": "4", "seq_len": "128", ...}}
```

This named format is **only** required by the QNN compiler path (which branches off early in
`_compile()` and passes specs directly to `qnn_compile()` without touching the JSON file). The
qaic-compile binary and its MDP (multi-device partition) firmware for 4-device
tensor-parallel only accept the legacy **flat format**:

```json
{"batch_size": "4", "seq_len": "128", "ctx_len": "2048", ...}
```

Loading a named-format MDP binary caused `RuntimeError: Failed to create ExecObj` at
`qaicrt.ExecObj(context, program)`. Single-device QPCs (llama-68m, E2E correctness tests)
tolerate the named format — only MDP QPCs are affected.

**Fix:** `_compile()` now strips the internal `_graph_name` tag and writes flat dicts
directly, bypassing `to_named_specializations()`:

```python
flat_specs = [{k: str(v) for k, v in spec.items() if k != "_graph_name"}
              for spec in specializations]
specializations_data = {"specializations": flat_specs}
create_json(str(specializations_json), specializations_data)
```

**Commit:** `5f1b6c5` — _Fix: write flat-format specializations.json for qaic-compile_

---

### 7. Integer values in specializations.json rejected by qaic-compile (exit status 255)

**Discovered during:** first attempt at running the fixed flat-format compile.

**Cause:** `_build_decode_spec_for_k()` and `build_prefill_specialization()` in
`modeling_auto.py` produce dictionaries with **integer** values (`seq_len=5`, not
`seq_len="5"`). When the previous fix bypassed `to_named_specializations()`, those integers
were written directly to JSON. qaic-compile rejects integer values in
`specializations.json`, exiting with code 255:

```
CalledProcessError: Command '['/opt/qti-aic/exec/qaic-compile', ...]'
returned non-zero exit status 255.
```

The integer form could be observed in the generated file:

```json
{"batch_size": 1, "seq_len": 128, ...}   ← rejected
```

vs. the correct string form:

```json
{"batch_size": "1", "seq_len": "128", ...}   ← accepted
```

`to_named_specializations()` had been converting ints to strings as a side-effect of the
`{k: str(v), ...}` construction in `_infer_specialization_name`. The flat-format path missed
this conversion.

**Fix:** Apply `str()` to every value while stripping `_graph_name`:

```python
flat_specs = [{k: str(v) for k, v in spec.items() if k != "_graph_name"}
              for spec in specializations]
```

**Commit:** `82f2d6c` — _Fix: convert specialization values to strings for qaic-compile_

---

## `modeling_qeff.py` — Flat-Format specializations.json

The multi-spec feature in `modeling_auto.py` exposed a latent issue in `_compile()` inside
`QEfficient/base/modeling_qeff.py`: the method was calling `to_named_specializations()` when
writing `specializations.json` for qaic-compile.  This had been harmless for 2-spec QPCs
(the converter produced valid output that qaic-compile accepted), but the 3-spec format
surfaced two problems: (a) the `{name, symbols}` wrapper was rejected by the MDP firmware,
and (b) integer values rather than strings caused qaic-compile to exit 255.

### What changed in `_compile()`

```python
# Before (called to_named_specializations — produced named format, int values):
specializations_data = {
    "specializations": to_named_specializations(specializations, ...)
}

# After (flat format, all values converted to strings):
flat_specs = [{k: str(v) for k, v in spec.items() if k != "_graph_name"}
              for spec in specializations]
specializations_data = {"specializations": flat_specs}
```

**What `_graph_name` is:** an internal routing tag set by `build_prefill_specialization()`,
`_build_decode_spec_for_k()`, and similar helpers so that `to_named_specializations()` can
assign human-readable names (`"Prefill"`, `"Decode"`, etc.).  It is not a qaic-compile
field and must be stripped before writing.

**Why `to_named_specializations()` is still needed:** the QNN compiler path (`enable_qnn=True`)
branches off at line 548 of `_compile()` and calls `qnn_compile()` directly, passing the
raw `specializations` list.  `qnn_compile()` internally calls `to_named_specializations()` on
that list.  The flat-format fix only affects the qaic-compile branch; QNN is untouched.

### Correct 3-spec flat-format output

```json
{
  "specializations": [
    { "batch_size": "1", "ctx_len": "2048", "full_batch_exec_size": "1",
      "full_batch_size": "1", "num_logits_to_keep": "1", "seq_len": "128" },
    { "batch_size": "1", "ctx_len": "2048",
      "full_batch_size": "1", "num_logits_to_keep": "1", "seq_len": "1"   },
    { "batch_size": "1", "ctx_len": "2048",
      "full_batch_size": "1", "num_logits_to_keep": "5", "seq_len": "5"   }
  ]
}
```

This matches the format used by all working 2-spec QPCs in the cache (`46f4c4d93f23c51c`
and `f51c4a3e527ccff9`).

### Commits

| Hash | Message |
|------|---------|
| `5f1b6c5` | Fix: write flat-format specializations.json for qaic-compile |
| `82f2d6c` | Fix: convert specialization values to strings for qaic-compile |

---

## Hardware Test Results (New SDK — 2026-05-08)
Tests were run on devices 5 and 6 using the `QAIC_TEST_DEVICE_ID` fixture.

| Test | Device | Result | Duration |
|---|---|---|---|
| `test_few_pld_inference[CB llama]` | 5 | ✅ PASSED | 43s |
| `test_few_spd_inference[CB llama]` | 6 | ✅ PASSED | ~2.5m |
| `test_few_spd_inference[CB qwen]`  | 6 | ✅ PASSED | ~2.5m |

**Observed inference metrics (CB llama, K=4):**
- Avg accepted tokens = **5.0** (= K+1, i.e., 100% acceptance rate when TLM = DLM ✓)
- Decode throughput = **~519 tokens/sec**

**Note on first re-run failure:** When CB llama and CB qwen compiled back-to-back on the
same device in the same pytest session, the Llama TLM compile (more complex: `num_cores=6`,
`num_logits_to_keep`) failed silently. Running each model sequentially on the same device
(the default behaviour of `pytest` without `-n`) is sufficient to avoid this.

---

## Test Results Summary

| Test suite | Count | Result |
|---|---|---|
| `test_speculative_decoding.py` (pre-existing unit tests) | 41 | ✅ All pass |
| `TestTLMMultiSpecSpecializations` (new) | 8 | ✅ All pass |
| `test_multi_spec_structure` (new) | 4 | ✅ All pass |
| `test_select_k` (new) | 5 | ✅ All pass |
| `test_tlm_multi_spec_logit_consistency` (new) | 4 | ✅ All pass |
| `test_multi_spec_qpc_logit_correctness` (new, hardware) | 4 | ✅ All pass |
| Hardware few-layers tests (PLD + SPD, new SDK) | 3 | ✅ All pass |

**Total Python-layer tests: 62 / 62 pass.**
**Total hardware tests: 7 / 7 pass.**

---

## Functional Correctness Guarantee

### The key property

For the multi-spec dispatch to be correct, dispatching to a *smaller* specialization (e.g.
`seq_len=1`, K=0) must produce the **same accepted token** as the full specialization
(`seq_len=K+1`) for the same anchor token and KV cache state.

This holds because Transformer self-attention is **causal**: the hidden state at position P
depends only on positions 0..P, never on positions P+1..K. Therefore:

```
tlm_forward(seq_len=1,   anchor_at_P, kv_cache)  →  logit_P
tlm_forward(seq_len=K+1, anchor_at_P + K_specs, kv_cache)  →  [logit_P, logit_P1, ..., logit_PK]
```

`logit_P` is identical in both cases. The accepted token (`argmax(logit_P)`) is therefore
the same regardless of which specialization is dispatched.

### What the test verifies (`test_tlm_multi_spec_logit_consistency`)

Added to `TestTLMForwardExecution` in `tests/unit_test/transforms/test_speculative_decoding.py`.
Runs entirely on CPU — no QAIC hardware required.

For K in {1, 2, 3, 5}:
1. Run `tlm_forward` with `seq_len=1` (single anchor token, `num_logits_to_keep=1`)
2. Run `tlm_forward` with `seq_len=K+1` (anchor + K speculative tokens, `num_logits_to_keep=K+1`)
   — same anchor, same KV cache, speculative tokens are random
3. Assert `logits_k0[0,0,:] ≈ logits_kK[0,0,:]` (anchor logit is identical, `atol=1e-5`)
4. Assert `argmax(logits_k0) == argmax(logits_kK)` (accepted token is identical)

### What is NOT yet tested (requires hardware)

- ~~That the KV cache is correctly updated after a `seq_len=1` dispatch (i.e., the next
  decode step uses consistent state)~~ **Covered by `test_multi_spec_qpc_logit_correctness`
  (KV state must be correct for the subsequent chunk's logits to match)**
- ~~End-to-end sequence equivalence: that a full generation run with dynamic dispatch produces
  the same token sequence as a run always using `seq_len=K+1`~~

Both properties are now directly verified by `test_multi_spec_qpc_logit_correctness`
(`tests/transformers/spd/test_spd_inference.py`), which runs on real QAIC hardware and checks:
- ALL K+1 logit vectors (not just argmax) match the vanilla DLM reference at every position
- Every accepted token matches
- This holds for `decode_ks` in `[0]`, `[3]`, `[0,3]`, and `[1,2,3]`

The end-to-end acceptance-rate guarantee is additionally covered by `test_few_spd_inference`
via the `mean_num_accepted_tokens == K+1` assertion.

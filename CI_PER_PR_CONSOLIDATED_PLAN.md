# Per-PR CI — Consolidated Design & Execution Plan

> **Single source of truth.** This document consolidates and supersedes the earlier scratch docs
> (`CI_PER_PR_DESIGN.md`, `CI_SCALABILITY_PLAN.md`, `TEST_COVERAGE_DESIGN_ANALYSIS.md`,
> `PER_PR_CAUSAL_TEXT_CI_REPRO.md`, `PER_PR_CAUSAL_TEXT_FAILURES.md`,
> `TWO_PHASE_RUN_FAILURE_REPORT.md`). Their verified facts, measured numbers, and failure
> taxonomies are folded in below.
>
> **The bargain that shapes this plan:**
> - **Parallel-run *structure* → follow PR 1075.** Split the device-independent work (export +
>   compile) from the device-bound work (`generate`) and run each at its own optimal width. This is
>   the proven shape, and it is what the implemented **two-phase (compile-warm / execute) split**
>   already does.
> - **Coverage & scalability *design* → follow the design docs.** Three-level CI (unit / fast
>   per-PR / nightly full), a data-driven **registry** as the coverage ledger, **source-salted**
>   shared QPC cache, task **adapters**, and a machine-checkable **coverage bargain**. These are
>   kept as the target architecture; the two-phase split is its first, minimal realization.

---

## 1. Executive Summary

QEfficient's lifecycle is **load → transform → export → compile → generate → compare**. Exactly one
step is device-bound:

> **Only `generate()` touches a QAIC card.** `load → transform → export → compile` are pure
> host/CPU work — `compile()` shells out to the AOT compiler `/opt/qti-aic/exec/qaic-compile` as a
> subprocess and never opens a card. Physical cards are leased only inside `QAICInferenceSession`
> at generate time. *(Verified empirically: `qaic-compile` runs with no card attached; confirmed in
> code at `QEfficient/base/modeling_qeff.py`, `QEfficient/compile/compile_helper.py`,
> `QEfficient/generation/cloud_infer.py`.)*

Therefore **cores and cards are two independent resource pools**, and the fast lane should schedule
each at its own width: **compile fans out across all cores** (`-n 32`); **generate fans out across
the cards** (`-n = #cards`, one worker per card, no contention). This is precisely the shape the
repo's own `tests/nightly_pipeline/` already proves works (export/compile parallel, generate
sequential), and the shape PR 1075 gestures at. This plan generalizes that proof into the per-PR
lane and makes it **modular and data-driven** so adding a model / knob / feature / whole model class
is a small, local edit.

The design has three layers and three lanes:

- **Layers:** a **registry** (data — what to test, doubling as the coverage ledger), **task
  adapters** (code — one file per model class, the only place class-specific lifecycle logic
  lives), and a **two-phase engine** (infra — written once: a compile phase that warms a shared QPC
  cache + reference-token manifest, and a generate phase that cache-hits and compares).
- **Lanes:** **L1 unit** (GitHub Actions, CPU), **L2 fast per-PR** (Jenkins, dummy models,
  end-to-end on hardware, first-token assertion), **L3 nightly full** (Jenkins cron, real weights,
  exact parity).

Two correctness items are **mandatory, not optional** (both code-verified):

1. **Salt the QPC cache key with a source fingerprint** of the exporter/transform/compiler source.
   The existing compile/export hashes omit source bodies, so without a salt a PR that edits the
   compiler reuses a stale QPC and goes green **without testing the change** — the single most
   dangerous failure mode for a repo whose purpose *is* the export/compile toolchain.
2. **Use a *shared* QPC cache with a per-hash lock**, replacing today's per-worker `QEFF_HOME`,
   which would otherwise break the compile→generate handoff.

---

## 2. Verified Foundations

Every decision below traces to one of these code facts. They were read directly (and F1, F5, F7
were additionally confirmed by running the implemented two-phase split).

| # | Fact | Design consequence |
|---|---|---|
| **F1** | `compile()` is host-only: runs the AOT compiler as a subprocess, no card. | Compile fans out across all cores, decoupled from cards. |
| **F2** | Device identity is chosen at **generate** time, programmatically (`generate(..., device_id=[...])` → `QAICInferenceSession` → `qaicrt.QIDList`). | Scheduler assigns `device_id` dynamically per test; single vs multi-device is a runtime choice. |
| **F3** | `num_devices` (cards) is fixed at **compile** time and is in the compile hash. | A row's device topology correctly busts the cache; single/multi-device QPCs never collide. |
| **F4** | `QAIC_VISIBLE_DEVICES` is **not read by QEfficient** — a process mask honored by `qaicrt`, set by conftest. | Card pinning composes *with* `device_id`; in-code selector is `device_id`, process mask is `QAIC_VISIBLE_DEVICES`. |
| **F5** | QPC cache-hit is a bare file check (`(qpc_path / "programqpc.bin").is_file()`) keyed on model+params only — **no source fingerprint**. Export hash includes transform **class names**, not bodies. | **Must salt the key** or exporter/compiler/transform PRs get false greens on a warm cache. |
| **F6** | Only whitelisted kwargs enter the model hash (`KWARGS_INCLUSION_LIST`). | A new graph-affecting compile/export param not whitelisted won't bust the cache — registry additions must check this. |
| **F7** | Today each xdist worker gets its **own** `QEFF_HOME/worker_N`. | Breaks a cross-phase shared cache; replace with shared store + per-hash lock. |
| **F8** | The lifecycle is fully **bundled** in one test body today (`check_causal_lm_pytorch_vs_kv_vs_ort_vs_ai100`). | Splitting requires exposing phases behind an adapter; that is the migration's real work. |
| **F9** | First-token relaxation already exists for the tiny lane. | Reuse as the `pr_fast` assertion rule; keep exact-24 for `nightly_full`. |
| **F10** | The nightly pipeline already implements the phase split + race-safe (fcntl) artifact merge. | Reuse its locking pattern for the shared QPC cache; reuse its phase discipline as the engine template. |

**Nesting landmine (found by direct experiment, not in the original design doc).** The QPC directory
nests **inside** the ONNX export directory (`compile_dir = onnx_path.parent`), and **all variants of
one model share the same content-addressed export dir** (the export hash excludes compile params).
Consequences that any shared-cache design must respect:
- `compile_only=True`'s existing cleanup (`manual_cleanup(onnx_path)` → `rmtree(dirname(onnx_path))`)
  **deletes the very QPC it just built.** A warm-up phase must not run that cleanup.
- One variant's cleanup **destroys sibling variants' QPCs.** Per-test cleanup must be suppressed in
  *both* phases of a shared-home run.
- Two workers compiling the *same* model's export dir concurrently **tear the `.onnx`.** Needs a
  per-model lock.

---

## 3. The Three Levels (coverage design — kept from the design docs)

| Level | Runs on | Models | Contract proven | Assertion | Trigger | Status today |
|---|---|---|---|---|---|---|
| **L1 — Unit** | GitHub Actions, CPU | none / tiny direct | QEff methods, transform registry, hash plumbing, cache semantics | strict CPU / ORT | every PR | **exists** — `pytest tests/unit_test -n auto` (~1,554 `test*` fns, 33 files) |
| **L2 — Fast per-PR** | Jenkins, cards + cores | **dummy only** | transform → export → compile → **generate matches HF/ORT** end-to-end | first-token (structural) | every PR | **this plan** — two-phase split, evolving to registry/engine |
| **L3 — Nightly full** | Jenkins cron | **real weights** | exact numerical parity, full matrix, multi-device breadth | exact-24 | scheduled | **must be wired** — `full_layers_model` is a parameter, not a schedule |

**Division of labor.** L1 = "did I break a QEff Python contract?" (no card, stays in Actions). L2 =
"does the changed code still export/compile/run and match the reference on real hardware?" on cheap
dummy models. L3 = "does it match exactly on real weights, at full breadth?" — the backstop L2's
dummy lane borrows against.

> **The coverage bargain (non-negotiable):** L2's dummy + first-token lane is only *sound* if L3 is
> an actually-scheduled job. Wiring the nightly cron is a prerequisite of this design, not a
> follow-up. Today `full_layers_model` is a selectable Jenkins parameter with **no `cron` trigger**
> in the checked-in CI — that gap must be closed before the fast lane is treated as
> coverage-preserving.

---

## 4. The Parallel-Run Structure — Two-Phase Split (PR 1075's shape)

This is the structural heart, taken from PR 1075's "split device-independent from device-bound work"
insight and already **implemented and measured** in this repo.

```
┌─ COMPILE PHASE ──── pytest -n <wide, e.g. 32> --dist worksteal · NO cards touched (F1) ─┐
│  for each registry row in the active lane:                                              │
│      m = adapter.load(row.model.id_for(profile), **row.from_pretrained)                 │
│      adapter.transform(m, num_devices=row.topology.num_devices, **row.transform)        │
│      adapter.export(m, **row.export)                                                    │
│      adapter.compile(m, num_devices=row.topology.num_devices, **row.compile)  # cache   │
│      manifest.put(row.id, adapter.reference(m, inputs))   # HF/ORT reference tokens     │
│  side effects: shared source-salted QPC cache warmed; ref-token manifest written        │
└─────────────────────────────────────────────────────────────────────────────────────-─┘
              handoff = shared QPC cache (F5 fix) + ref manifest (F10 lock)
┌─ GENERATE PHASE ─── pytest -n <#cards>  (one worker per card) · DEVICE-bound (F2) ───────┐
│  for each registry row in the active lane:                                              │
│      cards = scheduler.lease(worker_idx, row.topology.num_devices)   # dynamic device_id│
│      m = adapter.load(...); adapter.compile(...)   # -> CACHE HIT (instant)              │
│      got = adapter.generate(m, device_id=cards, **row.generate)                         │
│      adapter.compare(got, manifest.get(row.id), rule=row.assert[lane])                  │
└─────────────────────────────────────────────────────────────────────────────────────-─┘
```

**The generate phase is correct standalone.** Its `compile()` is a cache hit if the compile phase
pre-warmed, else an on-demand compile (F5). So the split is a pure *accelerator*, never a correctness
dependency: a developer runs only the generate phase locally; CI pre-warms at wide `-n` first. The
handoff is QEfficient's own content-addressed QPC store plus a tiny `id → [ref tokens]` manifest — no
fragile artifact database.

### 4.1 Resource model — cores vs cards, single vs multi-device

| Phase | Bound by | Parallelism | Rationale |
|---|---|---|---|
| Compile | CPU cores | `pytest -n <cores> --dist worksteal` | F1: no card needed; saturate cores. |
| Generate | Cards | `pytest -n <#cards>` (single-device rows) | F2: one worker per card, zero contention. |

**Card leasing** turns `(worker_idx, row.topology.num_devices)` into a collision-free `device_id`,
generalizing today's `offset + (idx % cards)`:
- **Single-device (default):** `device_id = [offset + worker_idx % cards]`; run generate at
  `-n #cards` → one worker per card. The common, fast path.
- **Multi-device (`num_devices = N > 1`):** compile with `num_devices=N` (F3 keys the cache), lease a
  disjoint contiguous slice (worker `w` → cards `[w·N, w·N+N)`), run at `-n ⌊#cards/N⌋`; the generate
  call passes `device_id=cards` with `len(cards) == N`. Per-PR scope = a *couple* of representative
  rows; exhaustive breadth → L3. `QEFF_QAIC_CARD_OFFSET`/`QEFF_NUM_QAIC_CARDS` let two Jenkins stages
  share cards on disjoint slices.

Because `num_devices` is in the compile hash (F3), single/multi-device QPCs are distinct entries and
never collide — topology is safe by construction.

### 4.2 Why split at all — honest cost/benefit (reconciled with measurements)

When the cache is **warm** (docs / model-data PRs), a single combined `-n #cards` phase is nearly as
fast, because compile is a cache hit. The split earns its keep on:
1. **Cold-cache PRs — the ones editing the exporter / compiler / transforms**, common in *this* repo.
   A combined `-n #cards` run throttles compile to card-width and leaves cards idle during every
   compile; the split runs those compiles at core-width and keeps every card saturated during
   generate.
2. **Matrix growth.** Phase A parallelism scales with cores (up to ~128 here); Phase B stays bounded
   by #cards. As the model set grows, the compile-warm phase amortizes across all workers while the
   device phase stays flat — so the split's relative benefit **increases with model count** (the
   user's key observation).

> **Measured caveat & the fix (see §7).** The *minimal* implemented split gave only ~10% on 34
> models because Phase B still re-ran the PyTorch-KV / HF / ORT **reference-token** passes and parity
> asserts that Phase A's `compile_only=True` skips — that CPU work, plus serialized on-device
> `generate()`, dominates Phase B, so compile is **not** ~80% of Phase B's wall time. **The engine
> design fixes exactly this:** move reference-token generation into the compile phase (the
> `manifest`), so the generate phase does *only* device `generate()` + a cheap compare against the
> manifest. This is the difference between the throwaway minimal split and the target engine, and it
> is where the projected multiplier actually lives.

---

## 5. Level-2 Architecture — Three Layers (extensibility design — kept)

```
tests/ci/
  registry/                LAYER 1 — DATA  (what to test; = coverage ledger)
    models.jsonl              one row per (model × feature × topology)
    profiles.json             dummy_layers / few_layers / full_layers definitions
    schema.py                 dataclass + validation (fail fast on malformed rows)
  adapters/                LAYER 2 — CODE, one per model class  (how to test it)
    base.py                   TaskAdapter protocol + shared helpers
    causal_lm.py  vlm.py  embedding.py  audio.py  seq_cls.py  reranker.py
    _registry.py              task-name -> adapter lookup
  engine/                  LAYER 3 — INFRA  (written once, task-agnostic)
    compile_phase.py          generic runner: -n <cores>, HOST, warms QPC cache + ref manifest
    generate_phase.py         generic runner: -n <#cards>, DEVICE, cache-hit + generate + compare
    scheduler.py              card leasing -> concrete device_id (single & multi)
    cache.py                  source-salted content-addressed QPC cache + per-hash lock
    manifest.py               reference-token store (race-safe, reuses nightly locking)
  conftest.py                 fixtures: profile, qaic_cards, qpc_cache, ref_manifest
```

### Layer 1 — The Registry (the extensibility crux, and the coverage ledger)

One JSONL row fully describes one test. **Per-phase kwargs are opaque data dicts** splatted into the
QEff API by the engine, so new `from_pretrained`/`export`/`compile`/`generate` params become one
added key — zero engine/adapter change.

```jsonc
{
  "id": "causal_lm.llama.cb.single",            // stable, unique; becomes the pytest param id
  "task": "causal_lm",                           // -> selects the adapter
  "model": {
    "real_id": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    "tiny_id": "hf-internal-testing/tiny-random-LlamaForCausalLM",  // null => see skip_lanes
    "model_type": "llama"
  },
  "topology":        { "num_devices": 1, "num_cores": 16 },
  "lanes":           ["pr_fast", "nightly_full"],
  "skip_lanes":      [],
  "from_pretrained": { "continuous_batching": true },
  "transform":       {},
  "export":          {},
  "compile":         { "prefill_seq_len": 1024, "ctx_len": 2048, "full_batch_size": 4, "mxfp6": false },
  "generate":        { "generation_len": 8 },
  "assert":          { "pr_fast": "first_token", "nightly_full": "exact_24" },
  "known_issues":    {},                         // e.g. {"pr_fast.ccl": "llama4 CCL export segfaults ..."}
  "owner":           "transformers-runtime"
}
```

Key properties:
- **`model.id_for(profile)`** returns `tiny_id` under dummy/few layers, `real_id` under full — making
  substitution an **explicit, per-row, reviewable choice** (the biggest cleanliness win over today's
  global `from_pretrained` monkey-patch).
- **`skip_lanes`** replaces the ad-hoc `skip_no_tiny` list with per-row, per-lane granularity + an
  `owner`, turning silent skips into managed coverage debt.
- **`assert`** encodes coverage *strength* per lane (F9), so the ledger can prove first-token in
  `pr_fast` is compensated by exact-24 in `nightly_full`.
- **`known_issues`** carries the existing `known_export_or_compile_issue` /
  `known_runtime_parity_issue` / `known_ccl_export_or_compile_issue` / `known_bf16_compile_issue`
  reason strings **as data** (see §8), replacing the scattered per-config keys with an owned,
  per-lane xfail ledger.
- **`schema.py`** validates every row at collection time (types, required fields, `tiny_id` XOR
  `skip_lanes` for `pr_fast`, every graph-affecting `compile`/`export` key present in
  `KWARGS_INCLUSION_LIST` — closing the F6 gap). Malformed rows fail fast with a clear message.

The registry **is** the coverage ledger: a one-line CI check asks *"is every `pr_fast` row also
covered by a `nightly_full` row (or owner-waived)?"* and fails the PR otherwise — realizing the
`ci_backend`/`trigger`/`strength` ledger the coverage analysis recommends as the same data that
drives execution.

> **Migration:** the initial `models.jsonl` is *generated* from the existing
> `tests/configs/causal_model_configs.json` (+ siblings) so the matrix is preserved exactly, not
> retyped. The generator is throwaway; the JSONL is the source of truth thereafter.

### Layer 2 — The Task Adapter (one file per model class)

Every model class implements one protocol — the **only** place class-specific logic lives. Each
method maps 1:1 to a step in today's bundled check-functions (F8), tagged by resource pool.

```python
class TaskAdapter(Protocol):
    task: str
    # ---- HOST (CPU) — compile phase, -n <cores> ----
    def load(self, model_id, **from_pretrained) -> QEFFBaseModel: ...
    def transform(self, m, *, num_devices, **transform) -> None: ...
    def export(self, m, **export) -> str: ...                       # -> onnx_path (host)
    def compile(self, m, *, num_devices, **compile) -> str: ...      # -> qpc_path (AOT, no card — F1)
    def reference(self, m, inputs) -> list[int]: ...                 # HF/ORT tokens for the manifest
    # ---- DEVICE — generate phase, -n <#cards> ----
    def generate(self, m, *, device_id, **generate) -> list[int]: ...
    # ---- HOST ----
    def compare(self, got, ref, *, rule) -> None: ...                # "first_token" | "exact_24"
```

- The engine composes `{load, transform, export, compile, reference}` into the compile phase and
  `{generate, compare}` into the generate phase — the exact host/device boundary of F1–F2.
- `compare` reuses the existing first-token logic (F9); `rule` comes from the row's `assert[lane]`.
- Adding a **new model class** = one `adapters/<task>.py` + one line in `_registry.py`. Engine,
  scheduler, cache, manifest never change.
- **Narrow shim, not global patch:** the adapter resolves the model id *before* calling QEff, so
  internal loads (tokenizer/processor for the same id) follow automatically. Genuinely separate ids
  (LoRA `adapter_model_id`, SPD draft/target) are explicit row fields. A scoped shim is retained
  **only** for unavoidable deep-internal loads, verified per-task during migration.

### Layer 3 — The Two-Phase Engine

Written once, task-agnostic (see the ASCII flow in §4). Reference-token generation lives in the
**compile phase** (the manifest); the generate phase does device `generate()` + compare only — the
fix for the §7 measured shortfall.

---

## 6. The QPC Cache — Content-Addressing, Source Salt, Shared Store

The load-bearing correctness core. Every claim here was read from code and confirmed by running the
split.

**What exists (F5).** A compiled QPC lands in `…-{compile_hash}/qpc/`; the cache-hit is
`if (qpc_path / "programqpc.bin").is_file(): return qpc_path`. Hash inputs = model config + params +
command + topology. **The compiler/exporter/transform *source* is not an input.**

**Problem.** A PR that edits a transform's `forward()` or exporter internals — without renaming a
class or changing a param — produces the **same hash** and, against a persistent warm cache, **reuses
a stale QPC**. The changed code is never compiled or run. **False green.** For a repo whose purpose
is the export/compile toolchain, this is the most dangerous failure mode of a naive warm cache.

**Fix — source-salted key.** Compute a `source_fingerprint` = git-tree hash of the graph-affecting
source (`QEfficient/{exporter, base/pytorch_transforms.py, transformers/**/pytorch_transforms.py,
compile/, base/modeling_qeff.py}`) and fold it into the cache key. Then:
- docs / test-only / model-data PRs → fingerprint unchanged → **warm cache → fast**;
- exporter / compiler / transform PRs → fingerprint changes → affected entries **auto-bust →
  recompiled at wide `-n`** → the changed code is actually exercised.

Implementation options, least-invasive first:
1. **Wrapper salt (no core change):** engine sets `compile_dir`/`QEFF_HOME` to a fingerprint-scoped
   subdir → changed fingerprint is a different path → miss. Zero change to QEff internals.
2. **Core change (cleaner, needs maintainer buy-in):** add `source_fingerprint` to
   `compile_hash_params` behind an env flag, so the benefit accrues to all QEff users. *(Decision D2.)*

**Shared store + per-hash lock (F7 fix).** The handoff requires both phases to see the *same* cache,
so the per-worker `QEFF_HOME/worker_N` must go for shared-home runs. Replace with a **single shared
content-addressed store** (persisted across runs, e.g. a Docker-mounted `qeff_qpcs`). Each QPC
already writes a *unique* `…-{hash}/` dir, so different `(model, params)` never race; the only race is
two workers compiling the *same* hash — guard it with a **per-hash (per-model) fcntl lock**, the
pattern the nightly pipeline already ships. This enables both cross-phase reuse *and* cross-PR reuse.

> **Implemented today (minimal form):** `tests/conftest.py` skips the per-worker remap and both
> session-level `qeff_models_clean_up()` calls when a two-phase flag is set
> (`_is_two_phase_shared_home_session()`); `check_causal_models.py` wraps export+compile in a
> per-model fcntl lock (`_model_export_compile_lock`); `test_causal_lm_models.py` suppresses the
> destructive per-test cleanup in both phases. Source-salt is **not yet implemented** — it is the
> first mandatory follow-up (§9 step 2).

---

## 7. Measured Results (implemented minimal two-phase split)

Env: pyenv `pr_review` (Python 3.10.19), **4 QAIC cards, 128 cores**, `HF_HUB_CACHE=/home/huggingface_hub`.
Scope: **only the per-PR causal-text suite** (`-k per_pr`) — 7 test functions parametrized to **183
tests** (34 models × 5 always-run variants + 10 blocking + 3 disagg). Not the full/few/dummy causal
suites, QNN, VLM, embedding, or audio.

**Single-phase baseline:** `pytest -k per_pr -n 4 --dist worksteal` → **12:20**, 110 passed / 73
xfailed / 0 genuine failures.

**Two-phase run (shared `QEFF_HOME`):**

| Phase | Invocation | Wall | Outcome |
|---|---|---|---|
| A (compile-warm, CPU) | `QEFF_PER_PR_COMPILE_WARM_ONLY=1 pytest -k per_pr -n 32 --dist worksteal` | **4:32** | 1 failed (gemma3 CCL flake), 125 passed, 57 xfailed; **125 warm QPCs survived** |
| B (execute, on-device) | `QEFF_PER_PR_SHARED_HOME=1 pytest -k per_pr -n 4 --dist worksteal` | **6:32** | 2 failed (gemma3 CCL flake), 108 passed, 73 xfailed; cache intact (125→157) |
| **Total** | | **~11:05** | vs 12:20 baseline = **~10%** |

(Phase-A vs B pass/xfail counts differ because Phase A forces `compile_only=True`, so
`known_runtime_parity_issue` xfails resolve as compile-only passes there — expected, not a defect.)

**Three defects found and fixed during implementation** (all rooted in the §2 nesting landmine):
1. **Session cleanup wiped the shared cache** (both `pytest_sessionstart` and `_sessionfinish`
   `rmtree` the whole `QEFF_HOME`). Fixed via `_is_two_phase_shared_home_session()` guards.
2. **Torn-ONNX races at `-n 32`** (concurrent writers to a shared export dir). Fixed with the
   per-model fcntl lock (Phase-A failures 2→1).
3. **Phase-B per-test cleanup destroyed sibling QPCs** (cache collapsed 125→57, first Phase B ran
   12:05 with 6 failures). Fixed by suppressing `manual_cleanup` for `QEFF_PER_PR_SHARED_HOME` too.

**Honest assessment.** The mechanism is correct (warm cache survives and is reused; no cross-variant
destruction), but the minimal split's speedup is **marginal (~10%)** because Phase B still runs the
reference-token CPU passes and the serialized on-device `generate()`, neither of which the compile
cache-hit removes. **The design-doc engine (§4.2, §5) closes this gap** by moving reference-token
generation into Phase A's manifest so Phase B does device-generate + compare only — that, plus
change-based selection (§9) and the cold-cache/scale wins (§4.2), is where the real multiplier is.
The minimal split as it stands is best kept as an **opt-in accelerator behind env flags** (default
single-phase behavior is untouched and unchanged) until the engine lands.

---

## 8. Failure & xfail Taxonomy (verified, from the reproduction runs)

**`failed`** = ran and assertions did not hold (red). **`xfailed`** = a documented known limitation
that failed as expected (neutral/green). **`xpassed`** = an xfail that unexpectedly passed (remove the
marker). None of the xfails below are new or caused by the two-phase split — they are pre-existing
model limitations carried as data.

### 8.1 Genuine known-issue xfails (compile/export or runtime-parity gaps)

| Reason key | Models | Signature (verbatim) |
|---|---|---|
| `known_bf16_compile_issue` (unconditional BF16 lane) | **all 34** | `MODEL_LOADER_UNSUPPORTED_DATATYPE`; `Non supported ONNX type COMPLEX128`; `Kernel lookup failed for: libjit_convert_f_to_f`. Qwen3.5/3.6 sub-signature: `FoldRMSNorm` / `getHandle<float>`. (`aic_hw_version=ai200`, `num_cores=4`) |
| `known_export_or_compile_issue` | `gemma4_dense_text`, `gemma4_moe_text` | `mat1 and mat2 shapes cannot be multiplied (32x1024 and 256x8)` at `o_proj(attn_output)` |
| `known_export_or_compile_issue` | `mixtral_moe_text` | `ReduceSum: Non-constant axes tensor not supported.` → `Compilation failed!` |
| `known_export_or_compile_issue` | `gptj_text` | `Range: 'limit' input must be a constant tensor` → `Compilation failed!` |
| `known_export_or_compile_issue` | `gpt_oss_moe_text` | `Clip operation` / `UNSUPPORTED_DATATYPE` → `Compilation failed!` |
| `known_ccl_export_or_compile_issue` | `llama4_text` (CCL lanes only) | `Fatal Python error: Segmentation fault` inside `torch.onnx.export` when `comp_ctx_lengths_decode` enabled |
| `known_runtime_parity_issue` (Phase B / `not compile_only` only) | `qwen3_5_dense_text`, `qwen3_6_dense_text`, `qwen3_5_moe_text`, `qwen3_6_moe_text` | `HF and QAIC decode tokens diverge after prefill.` (export+compile OK → compile-only coverage retained) |

### 8.2 Load-induced compile flake (NOT genuine, NOT a two-phase regression)

| Test | Phase(s) | Signature |
|---|---|---|
| `test_per_pr_causal_fp16_subfunction_cb_ccl[gemma3_text]` | A + B | `QAIC_ERROR: Cannot broadcast … Node: Where_195/` — `[1,1,1024,2016]` vs `[1,1,1024,2048]` (CCL slice `2016` vs pad `2048`) |
| `test_per_pr_causal_fp32_export_fp16_compile_subfunction_cb_ccl[gemma3_text]` | B | same |

**Classification = flake.** Passes deterministically in isolation (serial, single card, no flags:
`2 passed, 407 deselected in 44.50s`). Same error in Phase A (CPU-only, no device) *and* Phase B
(on-device) → not a shared-cache regression. gemma3's variants are already serialized by the
per-model lock → not a torn-file race; contention is CPU/compiler-global under many concurrent
`qaic-compile` subprocesses. It also surfaces intermittently in the single-phase `-n 4` baseline.
**Genuine failures across both phases: none.**

> **Mitigation (§9):** a bounded retry around `qaic-compile` on `Cannot broadcast` / `Compilation
> failed!`, or a proper fix of the `2016` vs `2048` slice-vs-pad mismatch at `Where_195` in gemma3's
> comp-ctx-length attention graph.

### 8.3 Per-PR suite dimensions (for reference)

Prompt len `1024`, ctx len `2048`, generation len `8`, continuous batching on, prefix-cache KV batch
`8`, CCL decode `[2048]`, ONNX subfunctions on. Dtype variants: FP16 (baseline/prefix/CCL/blocking/
disagg), FP32-export→FP16-compile (+CCL), BF16 export+compile (compile-only, all 34 xfail). Families
**not** covered by this causal harness (need a separate VLM-language harness): Qwen2.5-VL / Qwen3-VL /
Qwen3-VL-MoE text configs (not registered for `AutoModelForCausalLM`), Mllama, Molmo.

---

## 9. Migration Plan (non-disruptive, runs beside today's CI)

Each step is independently shippable and reversible; the current Jenkinsfile / single-phase run
remains the rollback target throughout.

1. **Confirm F1 empirically.** Run one `compile()` with cards masked and confirm success. *(Done —
   `qaic-compile` verified to run with no card.)*
2. **Land source-salt + shared-cache lock (§6) behind an env flag.** Safe to ship first; it only
   *tightens* correctness. *(Shared-cache lock + cleanup guards done; **source-salt still to do** —
   this is the mandatory guard against false-green compiler PRs.)*
3. **Build the engine + `causal_lm` adapter.** Generate `models.jsonl` from the existing causal
   configs. Move reference-token generation into the compile-phase manifest (fixes the §7 shortfall).
   Run the two phases beside the current suite; compare pass/xfail parity and timings.
4. **Prove the envelope.** Warm ≈ target on the causal core; cold ≤ ceiling; verify a compiler-edit
   PR busts and recompiles (a deliberate no-op-source-change canary must go red on stale reuse).
5. **Port adapters** vlm → embedding → audio → seq_cls → reranker, deleting the 3×
   `test_full_/test_few_/test_dummy_` duplicates as each family moves.
6. **Retire the global monkey-patch** once every row carries an explicit `tiny_id`, keeping only any
   narrow per-task shim proven necessary in step 5.
7. **Wire L3 cron** and flip the coverage-ledger check to *enforcing* (fail PRs that drop a `pr_fast`
   row without `nightly_full` coverage).
8. **Add change-based selection** on `pr_fast` (map each source path to the registry rows it can
   affect via `task` + the `source_fingerprint` dependency set; run only impacted rows + a small
   always-on smoke set) once the full-matrix warm time gets tight. *(Decision D1.)*
9. **Harden the gemma3 CCL flake** (§8.2) — bounded `qaic-compile` retry or the `Where_195` fix — so
   the wide compile-warm phase stays reliably green as concurrency and model count grow.

### Jenkins wiring (target)

```
Install QEfficient
Compute source_fingerprint      (git-tree hash of graph-affecting paths)  ── §6
Stage A — Compile phase         pytest tests/ci/engine/compile_phase.py -n <cores> --dist worksteal
                                (HOST; warms shared QPC cache + writes ref manifest)   ── F1
Stage B — Generate phase        pytest tests/ci/engine/generate_phase.py -n <#cards>
                                QEFF_NUM_QAIC_CARDS=<#cards> QEFF_QAIC_CARD_OFFSET=0
                                (DEVICE; cache-hit compile + generate + compare)        ── F2
Publish                         merged JUnit per phase; registry coverage-ledger check; durations
```

`QEFF_TEST_PROFILE=dummy_layers_model` selects `pr_fast` + `tiny_id`s. L3 reuses the *same* stages
with `full_layers_model` (→ `real_id`, exact-24) under `triggers { cron(...) }`. Undeclared markers
(`full_layers`/`few_layers`/`dummy_layers`/`feature`) are retired — lane selection is the registry's
`lanes` field, not a marker expression.

### Running the implemented minimal split today (reference)

```bash
PY=/home/rishinr/.pyenv/versions/3.10.19/envs/pr_review/bin/python
export HF_HUB_CACHE=/home/huggingface_hub HF_HUB_ENABLE_HF_TRANSFER=1
export QEFF_HOME=/path/to/shared_home        # scratch, must start clean; caller owns lifecycle
rm -rf "$QEFF_HOME"; mkdir -p "$QEFF_HOME"

# Phase A — compile-warm, CPU only, wide parallelism (forces compile_only + no-cleanup + shared home)
QEFF_PER_PR_COMPILE_WARM_ONLY=1 $PY -m pytest \
  tests/transformers/models/causal_lm_models/test_causal_lm_models.py -k per_pr -n 32 --dist worksteal

# Phase B — execute against the warm shared cache, one worker per card
QEFF_PER_PR_SHARED_HOME=1 $PY -m pytest \
  tests/transformers/models/causal_lm_models/test_causal_lm_models.py -k per_pr -n 4 --dist worksteal

rm -rf "$QEFF_HOME"
```

---

## 10. Risks & Mitigations

| Risk | Likelihood | Mitigation |
|---|---|---|
| **Stale-QPC false green** on compiler PRs (F5) | High without fix | Source-salted cache key (§6) — mandatory, landed first (step 2). |
| **Cross-phase cache miss** from per-worker `QEFF_HOME` (F7) | Certain if unaddressed | Shared store + per-hash fcntl lock (§6). **Done (minimal).** |
| **Sibling-QPC / self destruction** from bundled cleanup (nesting landmine) | Certain if unaddressed | Suppress destructive cleanup in both phases. **Done (minimal).** |
| **Graph-affecting param not in `KWARGS_INCLUSION_LIST`** (F6) → cache doesn't bust | Medium | `schema.py` rejects such rows at collection; audit list when adding params. |
| **Marginal speedup** if reference passes stay in Phase B | Observed (~10%) | Move reference-token generation into the compile-phase manifest (§4.2, §5). |
| **gemma3 CCL compile flake** under high `-n` | Medium | Bounded `qaic-compile` retry or `Where_195` fix (§8.2, step 9). |
| **3-min slips** as matrix grows | Medium (by design) | Change-based selection (§9 step 8); registry makes it a data query. |
| **Dummy lane treated as full coverage** | Medium (human) | Registry `assert`/`lanes` + enforcing ledger check; L3 cron a hard prerequisite. |
| **Compile-phase host OOM** at wide `-n` (VLM export is heavier) | Low | Cap concurrency per task class; VLM rows run at a lower `-n` slice. |

---

## 11. Decisions Needed

- **D1 — Per-PR scope:** full dummy matrix every PR, or change-based selection? *Rec: start
  full-matrix for correctness confidence; add selection the moment warm time gets tight.*
- **D2 — Source-salt implementation:** wrapper-level fingerprint-scoped cache dir (no core change) vs.
  `source_fingerprint` in `compile_hash_params`. *Rec: wrapper first; propose the core change once
  proven.*
- **D3 — Multi-device per-PR breadth:** one or two representative rows (rec) vs. broader now.
- **D4 — L3 schedule:** confirm the nightly cron + real-weight lane will be wired — the dummy lane's
  soundness depends on it. *Prerequisite, not an option.*
- **D5 — Cache persistence scope:** per-node local shared store vs. network-shared cache across nodes
  (enables horizontal compile fan-out).
- **D6 — Ship the minimal split now?** *Rec: keep it opt-in behind env flags as an accelerator; do
  not make it the default until the engine moves reference passes to Phase A (else only ~10%).*

---

## 12. Acceptance Criteria

- `pr_fast` completes within the agreed warm SLA on the causal core and within the cold ceiling, on
  the available cards / cores.
- A PR editing the exporter/compiler/transforms **provably recompiles** affected rows (no stale-QPC
  reuse) — tested by a deliberate no-op-source-change canary.
- Single-device generate runs `-n #cards` with **one worker per card, zero contention**; multi-device
  rows lease disjoint slices with `len(device_id) == num_devices`.
- Adding a model is a **one-row** edit; adding a model class is a **one-file** edit; neither touches
  the engine.
- The registry passes an **enforcing coverage-ledger check**: every `pr_fast` row is compensated by a
  `nightly_full` row or an owner-signed waiver.
- L3 nightly is an **actually-scheduled** job, not a manual parameter.
- Default (no-flag) single-phase behavior is **unchanged** at every step; a rollback path exists
  throughout.

---

## Appendix — Canonical Device API (verified)

```python
# COMPILE TIME — fixes the tensor-slicing partition, host-only (F1, F3)
qpc = model.compile(num_devices=N, num_cores=16, prefill_seq_len=…, ctx_len=…, **opts)

# RUN TIME — selects which physical cards run the QPC (F2)
model.generate(tokenizer, prompts, device_id=[c0, …, c_{N-1}])   # len(device_id) == N
#   -> QAICInferenceSession(qpc, device_ids) -> qaicrt.QIDList(device_ids)

# PROCESS MASK — set by conftest per worker; honored by qaicrt, NOT read by QEfficient (F4)
os.environ["QAIC_VISIBLE_DEVICES"] = str(card)
```

- `num_devices` = number of **cards** (MDP tensor-slicing when > 1); `num_cores` = cores **per card**.
- Device identity is **not** an env var inside QEfficient — the in-code selector is `device_id`;
  `QAIC_VISIBLE_DEVICES` is a complementary process-level mask.

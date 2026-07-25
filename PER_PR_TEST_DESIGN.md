# Per-PR Causal-LM Test Design (`CI_optimization_fork`)

> Authoritative capture of the per-PR causal-LM test architecture on branch
> `CI_optimization_fork`, including the newly added Speculative/TLM axis.
> All counts verified via `pytest --collect-only` in the `pr_review` env.

---

## 1. What this suite is

A **data-driven, registry-parametrized** validation matrix that exports →
compiles → runs every supported causal-LM architecture as a **tiny dummy model**
(1–2 layers) across a set of feature axes, on QAIC hardware, in continuous-batching
mode. It is the fast per-PR gate — distinct from the slower full/few-layer suites.

**Design goals:**

- One place to add a model (JSON), one place to add a feature axis (a test fn).
- Maximal xdist parallelism with **no cross-worker cache races**.
- A **two-phase** compile/execute split so the compile fan-out saturates cores
  and the execute fan-out saturates cards.
- Per-model, per-axis **escape hatches** (`known_*`) so a single broken model
  xfails one cell of the matrix without disabling the axis for everyone.

---

## 2. File map

```
tests/
├── configs/
│   └── causal_model_configs.json        ← the registry (data)
├── conftest.py                          ← xdist card-pinning, per-worker QEFF_HOME,
│                                          two-phase session lifecycle
└── transformers/models/
    ├── check_model_results.py           ← dump_and_compare_results (cosine/token compare)
    └── causal_lm_models/
        ├── test_causal_lm_models.py     ← THE per-PR test file (control plane)  ← only file edited
        └── check_causal_models.py       ← the engine: HF→QEff→ORT→AI100 parity  (data plane)
```

**Separation of concerns:**

- `test_causal_lm_models.py` = *control plane*. Declares axes, picks params, sets
  escape hatches. Thin.
- `check_causal_models.py` = *data plane*. One function,
  `check_causal_lm_pytorch_vs_kv_vs_ort_vs_ai100(...)`, does all export/compile/run/assert.
- `causal_model_configs.json` = *the data*. 34 model entries + capability flags.

---

## 3. The registry (`causal_model_configs.json`)

`per_pr_causal_text_models`: **34 entries**. Minimal entry:

```json
{
  "id": "gpt2_text",
  "model_name": "hf-internal-testing/tiny-random-GPT2LMHeadModel",
  "model_type": "gpt2",
  "is_moe": false,
  "supports_disagg": false,
  "supports_blocking": false,
  "num_hidden_layers": 1
}
```

### Field taxonomy

| Field                                         | Role                             | Present on                 |
| --------------------------------------------- | -------------------------------- | -------------------------- |
| `id`                                        | pytest param id (`_per_pr_id`) | all 34                     |
| `model_name`                                | HF card                          | all 34                     |
| `model_type`                                | architecture tag                 | all 34                     |
| `num_hidden_layers`                         | dummy-shrink depth               | all 34                     |
| **Capability flags**                    |                                  |                            |
| `is_moe`                                    | mixture-of-experts               | all 34 (8 true)            |
| `supports_blocking`                         | blocked-KV axis opt-in           | all 34 (**10 true**) |
| `supports_disagg`                           | MoE disagg axis opt-in           | all 34 (**3 true**)  |
| **Config shaping** (optional)           |                                  |                            |
| `config_overrides`                          | force config attrs               | few                        |
| `config_attr` / `use_text_config`         | dive into nested config          | few                        |
| `layer_types`                               | per-layer type list              | few                        |
| `num_cores`                                 | override compile cores           | few                        |
| `tokenizer_id`                              | tokenizer ≠ model               | few                        |
| **Escape hatches** (optional)           |                                  |                            |
| `known_export_or_compile_issue`             | xfail export/compile             | 5                          |
| `known_ccl_export_or_compile_issue`         | xfail CCL axis only              | 1                          |
| `known_runtime_parity_issue`                | xfail generate parity            | 4                          |
| `known_bf16_compile_issue`                  | xfail bf16 axis only             | 4                          |
| `known_speculative_export_or_compile_issue` | xfail speculative axis only      | 0 (available, unused)      |

The escape hatches are **axis-scoped**: a model can pass fp16 but xfail CCL, or
pass CCL but xfail bf16. Broken cells are annotated in data, not disabled in code.

---

## 4. Control plane — the axes (`test_causal_lm_models.py`)

Every per-PR test is `@parametrize("model_config", test_models_per_pr_causal, ids=_per_pr_id)`
and delegates to a single helper `_run_per_pr_qwen_causal_text_case(...)`.

### 4.1 The helper

```python
_run_per_pr_qwen_causal_text_case(
    model_config, manual_cleanup, *,
    torch_dtype=torch.float16, compile_only=False, retain_full_kv=False,
    qaic_config=None, comp_ctx_lengths_prefill=None, comp_ctx_lengths_decode=None,
    kv_cache_batch_size=None, num_cores=16, compile_options=None,
    num_speculative_tokens=None,          # ← added for the speculative axis
)
```

It (a) applies the **two-phase** env overrides, (b) fires per-model/axis
`known_*` xfails, (c) builds the dummy config via `_per_pr_dummy_config`, then
(d) calls the engine with fixed per-PR sizes:

```
PER_PR_PROMPT_LEN     = 1024
PER_PR_CTX_LEN        = 2048
PER_PR_GENERATION_LEN = 8
PER_PR_CCL_DECODE     = [2048]     (prefill CCL = None)
```

### 4.2 The axis matrix (verified collection counts)

| # | Test function                                                              | dtype      | Special knob                             | Models                | Count         |
| - | -------------------------------------------------------------------------- | ---------- | ---------------------------------------- | --------------------- | ------------- |
| 1 | `test_per_pr_causal_fp16_subfunction_cb`                                 | fp16       | baseline CB + onnx subfns                | all 34                | 34            |
| 2 | `test_per_pr_causal_fp16_subfunction_cb_prefix_caching`                  | fp16       | `kv_cache_batch_size=8`                | all 34                | 34            |
| 3 | `test_per_pr_causal_fp16_subfunction_cb_ccl`                             | fp16       | CCL decode`[2048]`                     | all 34                | 34            |
| 4 | `test_per_pr_causal_fp32_export_fp16_compile_subfunction_cb_ccl`         | fp32→fp16 | CCL                                      | all 34                | 34            |
| 5 | `test_per_pr_causal_bf16_subfunction_cb_ccl_compile_only`                | bf16       | CCL,**compile-only**, ai200/4-core | all 34                | 34            |
| 6 | `test_per_pr_causal_speculative_tlm_fp16_subfunction_cb` **(NEW)** | fp16       | `num_speculative_tokens=2` (TLM)       | all 34                | 34            |
| 7 | `test_per_pr_causal_fp16_subfunction_cb_blocking`                        | fp16       | `enable_blocking`, `num_kv_blocks=2` | `supports_blocking` | 10            |
| 8 | `test_per_pr_causal_moe_disagg_fp16_subfunction_cb_ccl`                  | fp16       | `retain_full_kv=True` + CCL            | `supports_disagg`   | 3             |
|   | **Total per-PR**                                                     |            |                                          |                       | **217** |

Plus QNN tests (`test_causal_lm_..._qnn`, `..._pl1_qnn`) that run on the
broader `test_models_causal` list, outside the per-PR matrix.

### 4.3 Notes on specific axes

- **bf16 (axis 5)** is `pytest.xfail`-ed unconditionally at the top
  (`PER_PR_BF16_COMPILER_ISSUE`) — a known repo/compiler gap on ai200+4-core; kept
  in the matrix as a live tripwire that will start passing when the compiler fixes it.
- **Speculative (axis 6, NEW)** compiles each dummy as a **Target Language Model**
  (`is_tlm=True`) with `Constants.NUM_SPECULATIVE_TOKENS` (=2). No engine change was
  needed — `check_causal_models.py` already threaded `num_speculative_tokens → is_tlm`
  through export, ORT (`run_kv_model_on_ort(is_tlm=...)`), and `compile(...)`.
  Escape hatch `known_speculative_export_or_compile_issue` is wired but currently
  unused (all 34 collect live).

---

## 5. Data plane — the engine (`check_causal_models.py`)

`check_causal_lm_pytorch_vs_kv_vs_ort_vs_ai100(...)` runs the parity ladder:

```
                   load HF causal-LM (dummy, n_layer shrunk, torch_dtype)
                                    │
              ┌─────────────────────┼──────────────────────┐
              ▼                     ▼                       ▼
   HF PyTorch reference     QEff PyTorch (KV)        (is_tlm from
   run_hf_model_on_pytorch  transform + KV run        num_speculative_tokens)
   (_CB variant if CB)      [skipped if compile_only
              │              or continuous_batching]        │
              └─────────────┬───────────────────────────────┘
                            ▼
              qeff_model.export(use_onnx_subfunctions)
                            │      [under _model_export_compile_lock]
                            ▼
                run_kv_model_on_ort(is_tlm)  ──►  assert HF == ORT
                            │                      assert KV == ORT
                            ▼
              qeff_model.compile(... num_speculative_tokens,
                                  comp_ctx_lengths_*, mdp_*, kv_cache_batch_size,
                                  num_cores, retain_full_kv, prefill_only, **opts)
                            │
                assert qconfig.json exists
                            │
            ┌───────────────┴────────────────┐
     compile_only?                        else
            │                                │
   manual_cleanup(onnx); return      qeff_model.generate() on AI100
                                             │
                              CB:  assert ORT[:gen]  == AI100[:gen]
                                   assert HF[:gen]   == AI100[:gen]
                              else: assert ORT == AI100 (exact tokens)
                                             │
                              compare_results? dump_and_compare_results(...)
```

**Parity is real, not compile-only-success.** The ladder asserts
HF-PyTorch == ONNXRuntime == AI100 token-for-token (per-PR sizes use exact-token
compare; `compare_results` adds the cosine/results-json path for full runs).

**Export/compile lock** (`_model_export_compile_lock`): a per-model `fcntl.flock`,
active **only** in the two-phase shared-home mode, serializes writers to the
content-addressed ONNX dir that a model's axis-variants share — so two workers
compiling `qwen3_moe`'s fp16 and ccl variants don't tear the same `.onnx`. No-op
in default single-phase runs.

---

## 6. Parallelism & the two-phase split (`conftest.py`)

### 6.1 Single-phase (default) parallelism

| Mechanism                      | Fixture                           | Effect                                                                                                                                                                                                     |
| ------------------------------ | --------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Card pinning**         | `_qaic_device_for_xdist_worker` | `gwN → QAIC_VISIBLE_DEVICES = offset + N % cards`. Under `-n 4` on a 4-card host, each worker owns one card; cross-worker compile/generate run in parallel, same-worker calls serialize on that card. |
| **Per-worker QEFF_HOME** | `_qeff_home_per_xdist_worker`   | Each worker gets`QEFF_HOME/worker_N`, patched into `cache`/`export_utils` module constants (they bind at import). Prevents compile-cache write races.                                                |
| **Session cleanup**      | `pytest_sessionstart/finish`    | rmtree QEFF_HOME clean at start/finish (skipped for nightly + two-phase).                                                                                                                                  |

Tunables: `QEFF_NUM_QAIC_CARDS` (default 4), `QEFF_QAIC_CARD_OFFSET` (run two
stages on disjoint card slices simultaneously).

### 6.2 Two-phase compile/execute split

Two env flags reshape the whole session (`_is_two_phase_shared_home_session`):

```
Phase A — COMPILE-WARM            Phase B — EXECUTE
QEFF_PER_PR_COMPILE_WARM_ONLY=1   QEFF_PER_PR_SHARED_HOME=1
  • compile_only forced             • real generate() + assertions
  • manual_cleanup → _no_cleanup    • manual_cleanup → _no_cleanup
  • no device touched               • hits QPCs Phase A warmed
  • saturates CPU cores             • saturates QAIC cards
        │                                 ▲
        └──── shared QEFF_HOME ───────────┘
        (per-worker remap + session wipe BOTH skipped;
         caller owns the shared-home lifecycle)
```

Why cleanup is suppressed in **both** phases: axis-variants of one model share a
content-addressed export dir with QPCs nested inside. A finishing variant's normal
cleanup would `rmtree` the shared dir and destroy sibling variants' warm QPCs /
in-flight compiles. The two-phase caller starts clean and cleans up once at the end.

---

## 7. What changed for the Speculative/TLM axis

Only **one file** touched — `test_causal_lm_models.py`:

1. **Import** — `from QEfficient.utils.constants import Constants, QnnConstants`
   (added `Constants`).
2. **Helper signature** — added `num_speculative_tokens=None` kwarg.
3. **Helper → engine** — passed `num_speculative_tokens=num_speculative_tokens`
   into `check_causal_lm_pytorch_vs_kv_vs_ort_vs_ai100`.
4. **New test** — `test_per_pr_causal_speculative_tlm_fp16_subfunction_cb`, all 34
   models, with a `known_speculative_export_or_compile_issue` escape hatch.

**No changes needed** to `check_causal_models.py` (plumbing pre-existed),
`causal_model_configs.json` (runs across all models; per-model annotation optional),
`constants.py` (`NUM_SPECULATIVE_TOKENS = 2` already defined), or `conftest.py`.

Verification (env `pr_review`):

- `import QEfficient` → OK
- speculative axis collects **34/34** models
- full per-PR matrix collects **217** tests, `ruff` clean

---

## 8. Design assessment (short)

**Strong:**

- Add-a-model = one JSON row; add-a-feature = one test fn. Genuine data/control/data-plane split.
- Escape hatches are axis-scoped and live-in-matrix (xfail, not skip) → broken
  cells self-heal when fixed, and never silently vanish.
- Two-phase split is the right shape: compile is CPU-bound, execute is card-bound;
  running them as separate saturating fan-outs beats one mixed fan-out.
- Parallelism correctness is handled at the seams (card pinning, per-worker home,
  per-model export lock) rather than hoped for.

**Watch items:**

- The unconditional bf16 xfail (axis 5) means that axis currently validates nothing
  but collection — intentional tripwire, but easy to forget it's dormant.
- `supports_disagg` = 3 and `supports_blocking` = 10 are hand-maintained; a new MoE
  model must remember to set them or it silently skips those axes.
- Speculative runs across all 34 with no annotations yet — first real hardware run
  will likely surface a few models needing `known_speculative_export_or_compile_issue`.

```
```

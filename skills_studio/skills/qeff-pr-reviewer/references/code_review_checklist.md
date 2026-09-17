# Code review checklist

Open this when reading code in Phase 4. Quick checks for naming, structure, docstrings, and comments. Repo-specific design checks are in `repo_conventions.md`; slop greps are in `slop_catalogue.md`.

## PEP 8 / structural style (calibrated severity — most are Issue, some are Blocker)

The repo enforces PEP 8 via ruff (line-length 120, isort, py310 target). Ruff catches whitespace, quotes, unused imports, line length — those are mechanical and fixed by `ruff check --fix`. They are **Nits** unless they ship en masse (>10 lint hits = "pre-commit was bypassed", escalate to Issue).

The PEP 8 violations ruff **can't** catch are the structural ones that obscure logic — those are calibrated **Issue (medium)** by default and **Blocker (high)** when they make the change unreviewable or hide bugs. Apply the severity scale below; don't downgrade structural violations to Nits just because PEP 8 also covers them. The rules below apply to *new or rewritten* code in the diff — don't flag pre-existing code untouched by the PR.

### Method size and self-containment

- **≤ 50 lines is the default.** New methods should be ~50 lines or shorter, do one thing, and read top-to-bottom. Up to ~80 lines is acceptable when the method is genuinely one linear sequence with no nested branching. The 100-line Auto-class methods are the *upper bound* in legacy code, not a target.
- A method that crosses ~100 lines without internal helpers → **Issue**. Reviewer should propose the natural split points (preprocessing / core / postprocessing, or per-branch helpers).
- A method that crosses ~200 lines, OR mixes ≥3 distinct responsibilities (e.g. argument validation + tensor reshape + cache update + ONNX export prep in one body) → **Blocker** — unreviewable, and almost always hiding a bug because nobody can hold the whole control flow in their head. Require decomposition before re-review.
- **Self-contained methods.** A method that reads or writes ≥3 attributes on `self` it does not own (reaches into a sibling/parent's state), or whose correctness depends on an undocumented mutation order with another method, → **Issue**. The method's contract should be visible from its signature + body, not from the order it's called in.

### Cognitive complexity

Cognitive complexity = how hard the method is to follow, not just branch count. Use this rough scale:

- Nesting depth ≤ 2 (one `if`/`for` deep) → fine.
- Nesting depth 3 (`if` inside `for` inside `try`, or three nested `if`s) → **Issue**; ask for an early return, a guard clause, or an extracted helper.
- Nesting depth ≥ 4, OR a body with > 5 boolean operators in one condition, OR a `for`/`while` whose body is > 30 lines → **Blocker**; the method is doing too much.
- A `forward()` whose flow can't be summarized in one sentence ("project, attend, project, residual") → **Issue**; usually means the attention/MoE/normalization stages need to be their own methods.
- Mutating shared state inside a deeply nested branch (e.g. `self.x = ...` four levels in) → **Blocker**; mutation should happen at a single, named site near the method's top or end.

### Naming (variables, methods, classes)

Names are PEP 8 mechanical *and* semantic. A wrong-cased class is a Nit; a name that misleads the reader is an Issue/Blocker.

- **Mechanical (Nit unless widespread):** `snake_case` for functions/vars/methods, `PascalCase` for classes, `UPPER_SNAKE` for module-level constants, `_leading_underscore` for internal, no single-letter names except loop indices / math (`i`, `j`, `k`, `q`, `k`, `v`, `b`, `h`, `s`, `d` are fine in tensor code; `x` for a single Tensor is fine; `a` for an unrelated value is not).
- **Generic / placeholder names (Issue):** `result`, `data`, `tmp`, `temp`, `helper`, `obj`, `thing`, `value`, `item`, `info`, `output`, `output2`, `final`, `processed_data`, `do_stuff`, `handle`, `process` in non-trivial code — describe what's flowing through (`attn_scores`, `routed_logits`, `kv_block_indices`).
- **Misleading names (Blocker):** a variable named `mask` that's actually `position_ids`; a method named `get_*` that mutates; a method named `_apply_*` that returns the model unchanged; a flag named `enable_*` that disables. These cause real bugs at the call site. Require renaming.
- **Class naming for QEff wrappers:** new attention/layer/model wrappers must follow `QEff<HFClassName>[<Suffix>]` (e.g. `QEffLlama4Attention`, `QEffPrefillChunkedQwen3MoeAttention`). A `QeffLLamaAttn` / `Qeff_Llama_Attention` form → Issue (transform registration matches the conventional spelling).
- **Boolean names:** `is_*` / `has_*` / `should_*` / `use_*`. A boolean called `prefill_or_decode` or `cb` → Nit, ask for `is_prefill` / `use_continuous_batching`.

### Function signatures

- **8+ positional args → Issue.** Group into a config object or force kwargs (`def foo(self, *, a, b, c, ...)`). The Auto class methods use kwargs for exactly this reason.
- **Mixed positional/keyword without `*`** when ≥ 4 args → Issue; readers can't tell which are required.
- **Mutable default args (`def f(x=[])`) → Blocker.** Classic Python footgun.
- **Type hints expected on public methods** (per ruff target). Missing hints on a new public method → Nit; on a new public Auto-class method → Issue.

### Docstrings

- Multi-paragraph docstrings on trivial getters → AI fill; **Issue** (delete or trim to one line).
- Docstrings whose `Args:`/`Returns:` don't match the signature (renamed param still listed, added param missing) → **Issue** — a docstring that lies is worse than none.
- Empty docstrings (`""""""`) → **Nit** (delete).
- Docstrings that restate the function name (`"""Get the model hash. Returns the model hash."""`) → **Nit** (trim or delete).
- This repo doesn't require docstrings on every function. Bare type hints are fine. **A bad docstring is always a defect; a missing one rarely is.**

### Comments

- `# Added for ...`, `# Per review feedback`, `# Fix for PROJ-1234` — belong in the PR description, not in source → **Issue** (delete).
- Commented-out code blocks (`# x = old_thing()` left around) → **Issue** unless paired with a load-bearing one-line `# re-enable when …` comment.
- Comments restating the next line (`# Increment counter`) → **Nit**.
- Multi-line section dividers (`# ======= section =======`) inside a method → **Issue** — that's a sign the method should be split.
- Genuine "why" comments (hidden invariant, surprising workaround) — leave alone.

## Perf / redundancy smells (calibrated — each has a silent-unless guard)

These are the critiques human reviewers make most often and the review historically missed. They are also the easiest to over-fire into junk, so each ships with a guard: flag the *specific* wasteful shape, never the bare pattern.

- **Tensor materialized just to read metadata.** `x.detach().clone().shape`, `int(t.clone().shape[-1])` where the copy is used only for a shape/`.size()`/`.numel()`/scalar read → the copy is pure waste. **Issue.** The clean form is `int(self.rotary_emb.cos_cached.shape[-1])` (in-tree). **MUST FIRE when** the diff contains `.detach().clone().shape` or `.clone().shape[` and the enclosing expression is `int(...)`, an assignment consumed by shape/size/numel, or scalar arithmetic on such — e.g. `rotary_dim = int(self.rotary_emb.cos_cached.detach().clone().shape[-1])`. **Silent only if** the same expression is *also* assigned to a variable, returned, or wrapped in `nn.Parameter` (real-tensor use). The "direct and only receiver" test is a guard against false positives on real-tensor `.clone()`, not permission to stay silent when the metadata-read pattern is unambiguous.
- **`copy.deepcopy` of a live model / large module.** `self.model = copy.deepcopy(model)` right after `super().__init__(model)` (transforms then mutate a throwaway), or any deepcopy of a full model/pipeline in an init/hot path → **Issue → Blocker** if it doubles export RAM. **Guard:** target must be a materialized model/`nn.Module`/pipeline whose copy is later discarded/overwritten. Never flag `copy.deepcopy` of a dict/config/kwargs (`copy.deepcopy(kwargs)`, `copy.deepcopy(qeff_model.hash_params)` are safe idiom), a meta-device / `init_empty_weights` model (zero real memory), or a copy made to avoid mutating the caller's model before an in-place op like `tie_weights()` (`quantizer_utils.py:51`).
- **Redundant kwarg at its default.** Passing `dtype=torch.float32` or a flag equal to the callee's documented default. **Nit → Issue** only if verified redundant AND it obscures intent. **Guard:** confirm the param exists with that default in the pinned `transformers==5.5.4`/torch version, AND the call is NOT in a `forward()`/`get_dummy_inputs`/export-glue path. Explicit `use_cache=True`/`return_dict=True`/`dtype=` in QEff `forward` overrides deliberately pin graph behavior against HF default-drift — not waste. In 5.5.4, `LlamaForCausalLM.forward` has no `return_dict` and `use_cache` defaults to `None`, so unverified "this is the default" is junk.
- **Redundant loop / recomputation.** Iterating a collection N times where one pass suffices; recomputing a loop invariant. **Issue.** **Guard:** state the single-pass rewrite concretely.
- **Mass-allocation waste.** `np.random.rand(...)`/`np.ones(...)`/`np.zeros(...)` for a buffer provably overwritten before its values are read (use `np.empty`); a full copy where a view/slice works. **Issue.** **Guard:** the overwrite must be visible. An all-ones mask, all-ones mask row, or any array whose values are the intended data is NOT waste — don't flag.

## Code-shape / redundancy (dead or pointless structure)

Verifiable, low-false-positive — flag only what grep/read confirms.

- **Override that only calls super.** `def apply(cls, m): m, t = super().apply(m); return m, t` → redundant, delete it. **Issue.** (In-tree deletable example: `pytorch_transforms.py:924`.) **Guard:** redundant only when the signature is identical to the parent's, unconditional, and forwards every arg unchanged. Do NOT flag an override that narrows/renames/drops/injects a param (`get_seq_length` dropping `cache_position`, a `forward` swallowing `**kwargs`, one that sets `kwargs["lora_ids"]` first) or wraps super in an `if`/`try` — verify against the parent signature first.
- **Dead `__init__`.** A `QEff<X>.__init__` that only calls `super().__init__(...)`, or re-assigns attributes the class-swap transform already installs → dead in the swap flow. **Issue** (Blocker if it re-inits pretrained weights). Respect the `__init__`-override exceptions in `repo_conventions.md#qeff-class-rules`.
- **Nested `try`/`try-finally` that flattens.** A `try` whose only content is another `try`, or a `finally` doing no real cleanup → **Issue** only when nesting obscures flow. Don't flag a `finally` that does real cleanup, nor sequential (non-nested) optional-import `try` blocks.
- **Unused import / dead constant / commented-out block.** An import or module constant the diff adds with zero references anywhere, or a commented-out block with no re-enable note. **Grep the whole tree, not just the diff** — `__init__.py` re-exports are consumed by other files. Skip `# noqa: F401`, `__all__` entries, and `__init__.py` re-export files. ruff already auto-flags F401 → treat a stray unused import as **Nit** unless en masse; a commented-out block or truly-dead constant is an **Issue.**

## Logical correctness (high-recall list)

- Every `if x is None` / `is not None` flip — common AI-edit accidents.
- `assert` without a message in production code.
- Hand-rolled regex on file paths or layer names — walk through edge cases.
- Class-level mutable state that gets externally mutated.
- Loops that look "off-by-one" in cache stitching, layer-window slicing, or KV-layer indexing.
- `try: ... except Exception: pass` or bare `except:`.
- `torch.zeros(...)` without `dtype=` feeding a model — silent downcast.
- `self.<attr> = self.<attr> - X` inside `forward` — per-instance state mutates on every call; second forward double-applies.
- `str(some_dict)` for hashing — fragile across Python versions; use `json.dumps(d, sort_keys=True)`.

## Code structure / placement

- Any new top-level function in `QEfficient/utils/` that already has an obvious home elsewhere (`utils/_utils.py`, `generation/`, `exporter/`, `compile/`) — flag misplacement.
- Imports out of isort order in new files (`stdlib → third-party → first-party`, alphabetized within group).
- Function-level imports of names that are module-import safe — move to module level (legit lazy imports for optional deps are different).
- Wildcard imports in library code (`from QEfficient... import *`).

## What "good" looks like

- ≤ 50-line methods that read top-to-bottom; decomposed when longer.
- Nesting depth ≤ 2; early returns and guard clauses instead of deep `if`/`else`.
- Names that read in a code-review meeting without explanation; no `result`/`tmp`/`data`.
- Kwargs over positional args once you pass 4-5 args.
- Type hints on public methods.
- Terse or absent docstrings; one-line "why" comments only when non-obvious.
- Imports in isort order, no commented-out blocks, no debug prints.

When you find these patterns, say so explicitly in the review — calibrating signal both ways shapes future contributions.


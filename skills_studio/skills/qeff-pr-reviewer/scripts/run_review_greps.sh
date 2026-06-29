#!/usr/bin/env bash
# Run all PR-review greps against a diff file and print categorized hits.
# Usage: run_review_greps.sh <diff-file> [<repo-root>]
# Prints per-category hits; empty categories are silent.

set -u

DIFF="${1:-}"
REPO_ROOT="${2:-.}"

if [[ -z "$DIFF" || ! -f "$DIFF" ]]; then
  echo "usage: $0 <diff-file> [<repo-root>]" >&2
  exit 2
fi

section() {
  printf "\n=== %s ===\n" "$1"
}

# --- Surface artifacts ---

section "Committed debug artifacts [B]"
grep -nE "^diff --git a/(dbg\.log|.*\.pyc|.*\.log|.*__pycache__|.*\.swp|.*\.bak|core\.[0-9]+) " "$DIFF" || true

section "Files literally named 'copy' [B]"
grep -nE "^\+\+\+ b/.*( copy| copy 2|_copy|\(copy\)|[Cc]opy_of_)\.py" "$DIFF" || true

section "Debug calls (pdb/breakpoint/debugger) [B]"
grep -nE "^\+.*\b(p[d]b\.set_trace|break[p]oint\(\)|i[p]db\.set_trace|IPython\.embed)" "$DIFF" || true

section "print() in library code [I]"
grep -nE "^\+\s*print\(" "$DIFF" | grep -vE "(examples/|scripts/|tests/)" || true

section "Unicode emoji / box-drawing chars in code [I]"
grep -nP "^\+.*[\x{2500}-\x{27BF}\x{1F300}-\x{1FAFF}]" "$DIFF" 2>/dev/null || true

# --- Imports ---

section "Silent fallback on import [I]"
grep -nE "^\+\s*except\s+\(?(ImportError|ModuleNotFoundError|RuntimeError)" "$DIFF" || true

section "Bare except [I]"
grep -nE "^\+\s*except\s*:" "$DIFF" || true

section "Wildcard import in library [I]"
grep -nE "^\+from QEfficient[\.\w]* import \*" "$DIFF" || true

section "Lint suppression added with feature [I]"
grep -nE "^\+.*\b(noqa|type:\s*ignore|pylint:|fmt: off)" "$DIFF" || true

# --- Feature toggling ---

section "os.environ-driven feature toggle [I]"
grep -nE "^\+.*os\.environ\.(get|setdefault)\(" "$DIFF" || true
grep -nE "^\+.*os\.environ\[" "$DIFF" || true

section "Toggle by commenting out [B]"
grep -nE "^\+\s*#.*(comment.*out|enable.*disable|toggle|uncomment)" "$DIFF" || true

section "Class-level mutable state [B]"
grep -nE "^\+\s+_(start|end|total_layers|cur_layer|state|idx|batch_size)\s*=\s*([0-9]|None)" "$DIFF" || true

# --- Comments / docstrings ---

section "Commented-out imports / code [I]"
grep -nE "^\+\s*#+\s*(import |from |class |def |return |@ )" "$DIFF" || true

section "TODO/FIXME/HACK added [I]"
grep -nE "^\+\s*#\s*(TODO|FIXME|HACK|XXX)" "$DIFF" || true

section "Comments referencing PR/review/ticket [I]"
grep -nE "^\+\s*#\s*(Added (for|per)|Per review|Per feedback|Fix for [A-Z]+-[0-9]+|For ticket|This was the|Used by)" "$DIFF" || true

# --- Tests ---

section "Test monkey-patches the SUT [B]"
grep -nE "monkeypatch\.setattr|@patch\(['\"]QEfficient" "$DIFF" || true

# --- Architecture ---

section "Changes in base/shared layer (verify EVERY model needs this — see layering contract) [audit→I/B]"
# Lines touched in files that are inherited by every model. Concentrated changes here are a yellow flag:
# model-specific logic belongs one layer down (models/<m>/, modeling_utils.py, generation/).
grep -nE "^\+\+\+ b/(QEfficient/base/(modeling_qeff|common|pytorch_transforms|onnx_transforms)\.py|QEfficient/transformers/models/modeling_auto\.py|QEfficient/transformers/cache_utils\.py)" "$DIFF" || true

section "Model-specific conditional / name inside generic code (layering leak) [B]"
# A concrete model_type string-compare, isinstance against a specific model, or a model/config
# class name referenced from generic code. The generic layer must not know which model it runs.
# Manually confirm the hit is inside a base/shared file before flagging (grep can't see file context alone).
grep -nE "^\+.*(model_type\s*==\s*[\"'][a-z0-9_]+[\"']|isinstance\([^,]+,\s*[A-Z][A-Za-z0-9_]*(Model|Config|ForCausalLM|ForConditionalGeneration)\))" "$DIFF" || true

section "Destructive self-mutation in forward (heuristic — verify it's inside forward()) [B]"
# Focus on numeric mutation patterns that almost never have a legitimate assignment shape.
# Skip plain `self.x = <other_thing>` which is too noisy (it covers normal field assignment).
grep -nE '^\+\s*self\.[a-zA-Z_][a-zA-Z0-9_]*\s*(\+=|-=|\*=|/=|=\s*self\.)' "$DIFF" || true

section "Hardcoded compute dtype (should be config.torch_dtype; QEff exports FP32/FP16/BF16) [I→B]"
# Exempt the legitimate softmax(..., dtype=torch.float32).to(q.dtype) up-cast idiom — verify manually.
grep -nE '^\+.*dtype\s*=\s*torch\.(float32|get_default_dtype\(\))' "$DIFF" || true

section "Hardcoded masked-attention fill (use MIN_MASKED_ATTENTION_VALUE) [B]"
grep -nE '^\+.*(masked_fill|torch\.where|torch\.full|torch\.tensor|fill_)\(.*-?(10000|1e4|1e9|50000|3\.0e4)' "$DIFF" || true
grep -nE '^\+.*torch\.finfo\([^)]*\)\.min' "$DIFF" || true

section "Hash plumbing references in this PR [audit]"
grep -nE "^\+.*(hash_params\[|create_export_hash|hash_dict_params|compile_hash_params|KWARGS_INCLUSION_LIST)" "$DIFF" || true

section "New public method signatures (export/compile/from_pretrained) — verify hash plumbing"
grep -nE "^\+.*def (export|compile|from_pretrained)\(" "$DIFF" || true

section "Hallucinated transformers imports — verify each name exists in transformers==5.5.4"
grep -hE "^\+from transformers[\.\w]* import" "$DIFF" || true

# --- Hygiene ---

section "Mutable default args (def f(x=[]) / x={}) [B]"
grep -nE '^\+\s*def [a-zA-Z_][a-zA-Z0-9_]*\([^)]*=\s*(\[\]|\{\}|set\(\))' "$DIFF" || true

section "Functions added (manually inspect for size > 100 lines / nesting > 3) [audit]"
# A unified diff can't measure nesting or full method length, so this lists every added def
# for Subagent C to inspect against PEP-8 structural rules in the checklist.
grep -nE '^\+\s*(async )?def [a-zA-Z_]' "$DIFF" || true

section "Trailing-whitespace adds (>5 = pre-commit bypassed)"
WS=$(grep -cE "^\+.*[[:space:]]$" "$DIFF" || true)
echo "trailing-WS lines added: $WS"

section "Missing trailing newline"
grep -E "^\\\\ No newline at end of file" "$DIFF" || true

section "Magic constants (top-of-file)"
grep -nE "^\+\s*[A-Z_][A-Z0-9_]{3,}\s*=\s*[0-9]+\s*$" "$DIFF" || true

section "assert without message [N]"
grep -nE "^\+\s*assert\s+[^,]+\s*$" "$DIFF" || true

section "Examples reaching for private names [I]"
grep -nE "^\+from QEfficient\.[^ ]+ import [^ ]*_[A-Za-z]" "$DIFF" | grep -B1 "^\+\+\+ b/examples/" || true

# --- Repo-specific automated checks ---

section "Unmapped QEff classes (added but absent from pytorch_transforms.py / modeling_auto.py)"
TF="$REPO_ROOT/QEfficient/transformers/models/pytorch_transforms.py"
MA="$REPO_ROOT/QEfficient/transformers/models/modeling_auto.py"
if [[ -f "$TF" && -f "$MA" ]]; then
  ADDED=$(mktemp)
  REGISTERED=$(mktemp)
  grep -hE "^\+class QEff[A-Za-z0-9_]+" "$DIFF" \
    | sed -E 's/^\+class (QEff[A-Za-z0-9_]+).*/\1/' | sort -u > "$ADDED"
  ( grep -hoE "QEff[A-Za-z0-9_]+" "$TF"; grep -hoE "QEff[A-Za-z0-9_]+" "$MA" ) | sort -u > "$REGISTERED"
  # Filter out classes that are intentionally not in _module_mapping:
  # *Wrapper / *DynamicCache / *RotaryEmbedding / *CustomRMSNorm* / *Layer (used internally)
  comm -23 "$ADDED" "$REGISTERED" \
    | grep -vE "(Wrapper$|DynamicCache$|RotaryEmbedding$|CustomRMSNorm[A-Za-z]*$|EncoderWrapper$|DecoderWrapper$|TextRotaryEmbedding$)" || true
  rm -f "$ADDED" "$REGISTERED"
else
  echo "(skipped: $TF or $MA missing)"
fi

section "PrefillOnly/Revert symmetry — flipped or missing inverse mapping"
# Flag any entry in RevertPrefill* mappings whose KEY is NOT a QEffPrefillOnly*/QEffPrefillChunked* class.
# A correct revert maps Prefill<X> -> regular_X; a flipped entry has regular_X as the key.
if [[ -f "$TF" ]]; then
  python3 - "$TF" <<'PY' || true
import re, sys
text = open(sys.argv[1]).read()
for cls in ("RevertPrefillKeepAttentionTransform", "RevertPrefillOnlyTransform"):
    m = re.search(rf"class {cls}\([^)]*\):\s*\n\s*_module_mapping\s*=\s*\{{([^}}]*)\}}", text)
    if not m:
        continue
    body = m.group(1)
    for line in body.splitlines():
        line = line.strip().rstrip(",")
        if not line or line.startswith("#") or "**" in line:
            continue
        if ":" not in line:
            continue
        key, _, val = line.partition(":")
        key, val = key.strip(), val.strip()
        if not key.startswith("QEff"):
            continue
        # Heuristic: in a Revert map, KEYs should be the chunked/prefill-only variant
        if not (key.startswith(("QEffPrefillOnly", "QEffPrefillChunked"))):
            print(f"{cls}: SUSPECT FLIPPED ENTRY  {key}: {val}  (key should be a Prefill/Chunked class)")
PY
fi


section "validate.md TBD / placeholder rows added"
grep -nE "^\+.*\|\s*(TBD|—|--|\?|N/A)\s*\|" "$DIFF" || true

# --- Header check (only meaningful when run inside a checkout against a base ref) ---

section "Hint: license header check"
echo "Run separately:"
echo "  for f in \$(git diff --name-only --diff-filter=A <base>..HEAD -- '*.py'); do"
echo "    grep -q 'SPDX-License-Identifier: BSD-3-Clause' \"\$f\" || echo MISSING: \$f"
echo "  done"
echo "Run also:"
echo "  for f in \$(git diff --name-only --diff-filter=A <base>..HEAD -- 'QEfficient/**/*.py'); do"
echo "    head -1 \"\$f\" | grep -q '^#!' && echo SHEBANG-ON-LIBRARY: \$f"
echo "  done"

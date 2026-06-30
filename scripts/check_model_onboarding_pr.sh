#!/usr/bin/env bash
set -euo pipefail

labels="${PR_LABELS:-}"
body="${PR_BODY:-}"
base_sha="${BASE_SHA:-}"
head_sha="${HEAD_SHA:-}"

check_labels=(
  "check:model-onboarding"
  "check:model-llm"
  "check:model-vlm"
  "check:model-moe"
  "check:model-embedding"
  "check:model-diffusion"
)

active_labels=()
for label in "${check_labels[@]}"; do
  if [[ "${labels}" == *"${label}"* ]]; then
    active_labels+=("${label}")
  fi
done

if [[ ${#active_labels[@]} -eq 0 ]]; then
  echo "No check:model-* label found; skipping model onboarding checklist."
  exit 0
fi

body_file="$(mktemp)"
trap 'rm -f "${body_file}"' EXIT
printf '%s\n' "${body}" > "${body_file}"

echo "Active model onboarding labels: ${active_labels[*]}"

changed_files=""
if [[ -n "${base_sha}" && -n "${head_sha}" ]] && git rev-parse --verify "${base_sha}^{commit}" >/dev/null 2>&1 && git rev-parse --verify "${head_sha}^{commit}" >/dev/null 2>&1; then
  changed_files="$(git diff --name-only "${base_sha}" "${head_sha}" || true)"
fi

if [[ -n "${changed_files}" ]]; then
  echo "Changed files detected:"
  printf '%s\n' "${changed_files}" | sed 's/^/  - /'
else
  echo "Changed files were not available in this context; validating PR body only."
fi

failures=()
notices=()

has_label() {
  local label="$1"
  [[ "${labels}" == *"${label}"* ]]
}

require_checked() {
  local description="$1"
  local pattern="$2"
  if ! grep -Eiq "^[[:space:]]*-[[:space:]]*\[[xX]\].*${pattern}" "${body_file}"; then
    failures+=("${description}")
  fi
}

if ! grep -Eiq "model onboarding checklist" "${body_file}"; then
  failures+=("PR body must include a 'Model Onboarding Checklist' section")
fi

require_checked "Architecture classified" "Architecture classified"
require_checked "Nearest QEff wrapper identified" "Nearest QEff wrapper"
require_checked "Wrapper/transform/modeling-auto wiring updated or marked N/A" "Wiring updated"
require_checked "Cache/runtime helper changes validated or marked N/A" "Cache/runtime helpers"
require_checked "Tests updated or marked N/A" "Tests updated"
require_checked "Docs/examples/configs updated or marked N/A" "Docs/examples/configs"
require_checked "Validation commands listed" "Validation commands"

if has_label "check:model-llm"; then
  require_checked "LLM HF/QEff/ORT parity checked" "LLM: HF/QEff/ORT parity"
  require_checked "LLM KV cache and prefill/decode checked" "LLM: KV cache/prefill/decode"
  require_checked "LLM continuous batching checked or marked N/A" "LLM: continuous batching"
  require_checked "LLM ONNX subfunctions checked or marked N/A" "LLM: ONNX subfunctions"
fi

if has_label "check:model-moe"; then
  require_checked "MoE router/top-k behavior checked" "MoE: router/top-k"
  require_checked "MoE expert packing checked" "MoE: expert packing"
  require_checked "MoE decode and prefill checked" "MoE: decode and prefill"
  require_checked "MoE subfunction/einsum export checked" "MoE: subfunction/einsum"
fi

if has_label "check:model-vlm"; then
  require_checked "VLM text-side parity checked" "VLM: text-side parity"
  require_checked "VLM full export checked" "VLM: full export"
  require_checked "VLM single/dual QPC checked or marked N/A" "VLM: single/dual QPC"
  require_checked "VLM image preprocessing/token insertion checked" "VLM: image preprocessing/token insertion"
fi

if has_label "check:model-embedding"; then
  require_checked "Embedding tensor parity checked" "Embedding: tensor parity"
  require_checked "Embedding pooling/attention-mask behavior checked" "Embedding: pooling/attention-mask"
  require_checked "Embedding multi-seq-len behavior checked or marked N/A" "Embedding: multi-seq-len"
  require_checked "Embedding ONNX transforms checked" "Embedding: ONNX transforms"
fi

if has_label "check:model-diffusion"; then
  require_checked "Diffusion component parity/MAD checked" "Diffusion: component parity/MAD"
  require_checked "Diffusion generation output checked" "Diffusion: generation output"
  require_checked "Diffusion ONNX/QPC artifacts checked" "Diffusion: ONNX/QPC artifacts"
  require_checked "Diffusion pipeline modes checked or marked N/A" "Diffusion: pipeline modes"
fi

if [[ -n "${changed_files}" ]]; then
  if printf '%s\n' "${changed_files}" | grep -Eq '^QEfficient/transformers/models/|^QEfficient/transformers/cache_utils.py|^QEfficient/utils/(generate_inputs|run_utils)\.py'; then
    if ! printf '%s\n' "${changed_files}" | grep -Eq '^tests/'; then
      notices+=("Model/runtime files changed but no tests/ files changed; make sure 'Tests updated or N/A' is defensible.")
    fi
    if ! printf '%s\n' "${changed_files}" | grep -Eq '^(docs/|examples/)'; then
      notices+=("Model/runtime files changed but no docs/ or examples/ files changed; make sure 'Docs/examples/configs or N/A' is defensible.")
    fi
  fi
fi

if [[ ${#notices[@]} -gt 0 ]]; then
  echo "Checklist notices:"
  for notice in "${notices[@]}"; do
    echo "::notice::${notice}"
  done
fi

if [[ ${#failures[@]} -gt 0 ]]; then
  echo "Model onboarding checklist is incomplete:"
  for failure in "${failures[@]}"; do
    echo "::error::${failure}"
  done
  echo ""
  echo "Add the optional model onboarding PR template content and check the applicable lines, or remove the check:model-* label."
  exit 1
fi

echo "Model onboarding checklist passed."

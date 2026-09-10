#!/usr/bin/env bash
# -----------------------------------------------------------------------------
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------

# Production validation matrix for examples/text_generation/basic_inference.py.
#
# The default "full" suite runs every supported row below. Use SUITE=smoke for
# a short baseline, CASE_FILTER='<bash-regex>' to select named cases, or
# DRY_RUN=1 to validate every command without loading models or compiling.

set -uo pipefail

REPO_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
EXAMPLE="${REPO_ROOT}/examples/text_generation/basic_inference.py"

if ! command -v pyenv >/dev/null 2>&1; then
    echo "ERROR: pyenv is required but was not found in PATH." >&2
    exit 2
fi

eval "$(pyenv init -)"
if pyenv commands | grep -qx "virtualenv-init"; then
    eval "$(pyenv virtualenv-init -)"
fi
if ! pyenv activate mainline; then
    echo "ERROR: unable to activate the 'mainline' pyenv." >&2
    exit 2
fi

export HF_HUB_CACHE=/home/huggingface_hub
export HF_HUB_ENABLE_HF_TRANSFER=1
export QEFF_HOME=/home/rishinr/tmpdir
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false

DENSE_MODEL=${DENSE_MODEL:-meta-llama/Llama-3.2-1B-Instruct}
MOE_MODEL=${MOE_MODEL:-openai/gpt-oss-20b}
SUITE=${SUITE:-full}
CASE_FILTER=${CASE_FILTER:-}

SINGLE_DEVICE_GROUP=${SINGLE_DEVICE_GROUP:-0}
TS2_DEVICE_GROUP=${TS2_DEVICE_GROUP:-0,1}
BLOCKING_DEVICE_GROUP=${BLOCKING_DEVICE_GROUP:-0,1,2,3}
DISAGG_PREFILL_DEVICE_GROUP=${DISAGG_PREFILL_DEVICE_GROUP:-0,1,2,3}
DISAGG_DECODE_DEVICE_GROUP=${DISAGG_DECODE_DEVICE_GROUP:-4,5}

NUM_CORES=${NUM_CORES:-16}
BF16_NUM_CORES=${BF16_NUM_CORES:-4}
PREFILL_SEQ_LEN=${PREFILL_SEQ_LEN:-32}
CTX_LEN=${CTX_LEN:-128}
GENERATION_LEN=${GENERATION_LEN:-16}
CCL_CTX_LEN=${CCL_CTX_LEN:-1024}
BLOCKING_CTX_LEN=${BLOCKING_CTX_LEN:-131072}
BLOCKING_NUM_DEVICES=${BLOCKING_NUM_DEVICES:-4}
DISAGG_PREFILL_SEQ_LEN=${DISAGG_PREFILL_SEQ_LEN:-128}
DISAGG_CTX_LEN=${DISAGG_CTX_LEN:-256}
DISAGG_GENERATION_LEN=${DISAGG_GENERATION_LEN:-16}
DISAGG_FULL_BATCH_SIZE=${DISAGG_FULL_BATCH_SIZE:-2}
CASE_TIMEOUT_SECONDS=${CASE_TIMEOUT_SECONDS:-0}
RUN_DISAGG=${RUN_DISAGG:-1}
DRY_RUN=${DRY_RUN:-0}

RUN_ID=${RUN_ID:-$(date +%Y%m%d_%H%M%S)}
RESULT_ROOT=${RESULT_ROOT:-${QEFF_HOME}/text_generation_matrix/${RUN_ID}}
LOG_DIR="${RESULT_ROOT}/logs"
ARTIFACT_DIR="${RESULT_ROOT}/artifacts"
SUMMARY_TSV="${RESULT_ROOT}/summary.tsv"
SUMMARY_MD="${RESULT_ROOT}/summary.md"

mkdir -p "${HF_HUB_CACHE}" "${QEFF_HOME}" "${LOG_DIR}" "${ARTIFACT_DIR}"
printf 'case\ttier\tstatus\texit_code\tduration_seconds\tlog\tdescription\n' >"${SUMMARY_TSV}"

declare -a PASSED_CASES=()
declare -a FAILED_CASES=()
declare -a SKIPPED_CASES=()

device_group() {
    printf '[%s]' "$1"
}

record_result() {
    local name=$1
    local tier=$2
    local status=$3
    local exit_code=$4
    local duration=$5
    local log_file=$6
    local description=$7
    printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
        "${name}" "${tier}" "${status}" "${exit_code}" "${duration}" "${log_file}" "${description}" \
        >>"${SUMMARY_TSV}"
}

skip_case() {
    local name=$1
    local tier=$2
    local description=$3
    local reason=$4
    SKIPPED_CASES+=("${name}: ${reason}")
    record_result "${name}" "${tier}" "SKIP" 0 0 "-" "${description} (${reason})"
    printf '\n[%s] SKIP: %s\n' "${name}" "${reason}"
}

run_case() {
    local name=$1
    local tier=$2
    local description=$3
    shift 3

    if [[ "${SUITE}" == "smoke" && "${tier}" != "smoke" ]]; then
        skip_case "${name}" "${tier}" "${description}" "excluded by SUITE=smoke"
        return
    fi
    if [[ -n "${CASE_FILTER}" && ! "${name}" =~ ${CASE_FILTER} ]]; then
        skip_case "${name}" "${tier}" "${description}" "excluded by CASE_FILTER"
        return
    fi

    local log_file="${LOG_DIR}/${name}.log"
    local case_artifacts="${ARTIFACT_DIR}/${name}"
    local start_time
    local end_time
    local duration
    local status
    local -a command=(python "${EXAMPLE}" "$@" --compile-dir "${case_artifacts}")
    if [[ "${DRY_RUN}" == "1" ]]; then
        command+=(--dry-run)
    fi

    mkdir -p "${case_artifacts}"
    printf '\n[%s] %s\n' "${name}" "${description}"
    printf '[%s] command:' "${name}"
    printf ' %q' "${command[@]}"
    printf '\n'

    start_time=$(date +%s)
    if ((CASE_TIMEOUT_SECONDS > 0)); then
        timeout --signal=INT --kill-after=60 "${CASE_TIMEOUT_SECONDS}" "${command[@]}" 2>&1 | tee "${log_file}"
        status=${PIPESTATUS[0]}
    else
        "${command[@]}" 2>&1 | tee "${log_file}"
        status=${PIPESTATUS[0]}
    fi
    end_time=$(date +%s)
    duration=$((end_time - start_time))

    if ((status == 0)); then
        PASSED_CASES+=("${name}")
        record_result "${name}" "${tier}" "PASS" "${status}" "${duration}" "${log_file}" "${description}"
        printf '[%s] PASS (%ss)\n' "${name}" "${duration}"
    else
        FAILED_CASES+=("${name}")
        record_result "${name}" "${tier}" "FAIL" "${status}" "${duration}" "${log_file}" "${description}"
        printf '[%s] FAIL exit=%s (%ss), log=%s\n' "${name}" "${status}" "${duration}" "${log_file}" >&2
    fi
}

run_basic_case() {
    local name=$1
    local tier=$2
    local description=$3
    shift 3
    run_case "${name}" "${tier}" "${description}" "$@"
}

run_dense() {
    local name=$1
    local tier=$2
    local description=$3
    shift 3
    run_basic_case \
        "${name}" "${tier}" "${description}" \
        --model-name "${DENSE_MODEL}" \
        --prompt "Explain why deterministic tests matter." \
        --prefill-seq-len "${PREFILL_SEQ_LEN}" \
        --ctx-len "${CTX_LEN}" \
        --generation-len "${GENERATION_LEN}" \
        --num-cores "${NUM_CORES}" \
        --num-devices 1 \
        --device-group "$(device_group "${SINGLE_DEVICE_GROUP}")" \
        "$@"
}

run_blocking_decode() {
    local mode=$1
    shift
    run_basic_case \
        "blocking_decode_${mode}" full "Llama decode-only compile with ${mode} blocking" \
        --model-name "${DENSE_MODEL}" \
        --dtype float16 \
        --stage decode \
        --compile-only \
        --prefill-seq-len 1 \
        --ctx-len "${BLOCKING_CTX_LEN}" \
        --num-cores "${NUM_CORES}" \
        --num-devices "${BLOCKING_NUM_DEVICES}" \
        --mxfp6-matmul \
        --mxint8-kv-cache \
        --use-onnx-subfunctions \
        --user-tiled \
        --enable-blocking \
        --blocking-mode "${mode}" \
        "$@"
}

run_blocking_prefill() {
    local mode=$1
    shift
    run_basic_case \
        "blocking_${mode}" full "Llama prefill-only compile with ${mode} blocking" \
        --model-name "${DENSE_MODEL}" \
        --dtype float16 \
        --stage prefill \
        --compile-only \
        --prefill-seq-len 128 \
        --ctx-len "${BLOCKING_CTX_LEN}" \
        --num-cores "${NUM_CORES}" \
        --num-devices "${BLOCKING_NUM_DEVICES}" \
        --mxfp6-matmul \
        --mxint8-kv-cache \
        --use-onnx-subfunctions \
        --user-tiled \
        --enable-chunking \
        --enable-blocking \
        --blocking-mode "${mode}" \
        "$@"
}

print_summary() {
    local total=$(( ${#PASSED_CASES[@]} + ${#FAILED_CASES[@]} + ${#SKIPPED_CASES[@]} ))
    {
        echo "# Text generation validation summary"
        echo
        echo "- Run ID: ${RUN_ID}"
        echo "- Dense model: ${DENSE_MODEL}"
        echo "- MoE/disaggregated model: ${MOE_MODEL}"
        echo "- Passed: ${#PASSED_CASES[@]}"
        echo "- Failed: ${#FAILED_CASES[@]}"
        echo "- Skipped: ${#SKIPPED_CASES[@]}"
        echo "- Total: ${total}"
        echo
        echo "Full machine-readable results: \`${SUMMARY_TSV}\`"
        if ((${#PASSED_CASES[@]})); then
            echo
            echo "## Passed cases"
            for item in "${PASSED_CASES[@]}"; do
                echo "- ${item}"
            done
        fi
        if ((${#FAILED_CASES[@]})); then
            echo
            echo "## Failures"
            for item in "${FAILED_CASES[@]}"; do
                echo "- ${item}: \`${LOG_DIR}/${item}.log\`"
            done
        fi
        if ((${#SKIPPED_CASES[@]})); then
            echo
            echo "## Skipped cases"
            for item in "${SKIPPED_CASES[@]}"; do
                echo "- ${item}"
            done
        fi
    } >"${SUMMARY_MD}"

    printf '\n======================================================================\n'
    printf 'TEXT GENERATION MATRIX SUMMARY\n'
    printf 'PASS=%s FAIL=%s SKIP=%s TOTAL=%s\n' \
        "${#PASSED_CASES[@]}" "${#FAILED_CASES[@]}" "${#SKIPPED_CASES[@]}" "${total}"
    printf 'TSV: %s\nMarkdown: %s\nLogs: %s\nArtifacts: %s\n' \
        "${SUMMARY_TSV}" "${SUMMARY_MD}" "${LOG_DIR}" "${ARTIFACT_DIR}"
    if ((${#FAILED_CASES[@]})); then
        printf '\nFailed cases:\n'
        printf '  - %s\n' "${FAILED_CASES[@]}"
    fi
    if ((${#SKIPPED_CASES[@]})); then
        printf '\nSkipped cases:\n'
        printf '  - %s\n' "${SKIPPED_CASES[@]}"
    fi
    printf '======================================================================\n'
}

printf 'Repository: %s\n' "${REPO_ROOT}"
printf 'Python: %s\n' "$(python --version 2>&1)"
printf 'HF_HUB_CACHE: %s\n' "${HF_HUB_CACHE}"
printf 'QEFF_HOME: %s\n' "${QEFF_HOME}"
printf 'Result root: %s\n' "${RESULT_ROOT}"
printf 'Suite: %s\n' "${SUITE}"
printf 'Dry run: %s\n' "${DRY_RUN}"

# Fast contract check; no model loading or compilation.
run_basic_case \
    cli_contract smoke "Parse the complete advanced CLI surface" \
    --disaggregated --full-batch-size 2 \
    --prefill-num-devices 4 --decode-num-devices 2 \
    --mdp-num-partitions 2 \
    --prefill-blocking-mode prefill_online \
    --decode-blocking-mode kv_headpar \
    --num-kv-blocks 2 --num-q-blocks 2 \
    --dtype float16 --aic-hw-version ai100 \
    --dry-run --print-resolved

# Dense Llama runtime and graph-generation coverage.
run_dense dense_fp32_runtime smoke "Llama FP32 export, compile, and runtime" --dtype float32
run_dense dense_fp16_runtime full "Llama FP16 export, compile, and runtime" --dtype float16
run_dense dense_subfunctions_runtime full "Llama ONNX subfunctions export, compile, and runtime" --use-onnx-subfunctions
run_dense \
    dense_fp16_subfunctions_runtime full "Llama FP16 plus ONNX subfunctions" \
    --dtype float16 --use-onnx-subfunctions
run_dense \
    dense_mxfp6_mxint8_runtime full "Llama MXFP6 weights plus MXINT8 KV cache" \
    --mxfp6-matmul --mxint8-kv-cache
run_dense \
    dense_optimized_subfunctions_runtime full "Llama MXFP6/MXINT8 plus ONNX subfunctions" \
    --mxfp6-matmul --mxint8-kv-cache --use-onnx-subfunctions

# Compute Context Length, alone and combined with subfunctions.
run_basic_case \
    dense_ccl_runtime full "Llama CCL export, compile, and runtime" \
    --model-name "${DENSE_MODEL}" \
    --prompt "Summarize compute context length." \
    --dtype float32 \
    --prefill-seq-len 128 --ctx-len "${CCL_CTX_LEN}" --generation-len "${GENERATION_LEN}" \
    --ccl-prefill 128 256 512 --ccl-decode 640 768 "${CCL_CTX_LEN}" \
    --num-cores "${NUM_CORES}" --num-devices 1 --device-group "$(device_group "${SINGLE_DEVICE_GROUP}")"
run_basic_case \
    dense_ccl_subfunctions_runtime full "Llama CCL plus ONNX subfunctions" \
    --model-name "${DENSE_MODEL}" \
    --prompt "Summarize compute context length." \
    --dtype float16 \
    --prefill-seq-len 128 --ctx-len "${CCL_CTX_LEN}" --generation-len "${GENERATION_LEN}" \
    --ccl-prefill 128 256 512 --ccl-decode 640 768 "${CCL_CTX_LEN}" \
    --use-onnx-subfunctions \
    --num-cores "${NUM_CORES}" --num-devices 1 --device-group "$(device_group "${SINGLE_DEVICE_GROUP}")"

# Continuous batching and tensor slicing.
run_basic_case \
    dense_continuous_batching_runtime full "Llama continuous batching runtime" \
    --model-name "${DENSE_MODEL}" \
    --prompt "First request" "Second request" \
    --continuous-batching --full-batch-size 2 \
    --prefill-seq-len "${PREFILL_SEQ_LEN}" --ctx-len "${CTX_LEN}" --generation-len "${GENERATION_LEN}" \
    --num-cores "${NUM_CORES}" --num-devices 1 --device-group "$(device_group "${SINGLE_DEVICE_GROUP}")"
run_basic_case \
    dense_ts2_runtime full "Llama tensor slicing across two devices" \
    --model-name "${DENSE_MODEL}" \
    --prompt "Explain tensor slicing." \
    --dtype float16 \
    --prefill-seq-len "${PREFILL_SEQ_LEN}" --ctx-len "${CTX_LEN}" --generation-len "${GENERATION_LEN}" \
    --num-cores "${NUM_CORES}" --num-devices 2 --device-group "$(device_group "${TS2_DEVICE_GROUP}")"

# BF16 is intentionally compile-only and targets AI200.
run_basic_case \
    dense_bf16_ai200_compile full "Llama BF16 AI200 compile-only" \
    --model-name "${DENSE_MODEL}" \
    --dtype bfloat16 --aic-hw-version ai200 --compile-only \
    --prefill-seq-len "${PREFILL_SEQ_LEN}" --ctx-len "${CTX_LEN}" \
    --num-cores "${BF16_NUM_CORES}" --num-devices 1
run_basic_case \
    dense_bf16_ai200_subfunctions_compile full "Llama BF16 AI200 plus subfunctions compile-only" \
    --model-name "${DENSE_MODEL}" \
    --dtype bfloat16 --aic-hw-version ai200 --compile-only --use-onnx-subfunctions \
    --prefill-seq-len "${PREFILL_SEQ_LEN}" --ctx-len "${CTX_LEN}" \
    --num-cores "${BF16_NUM_CORES}" --num-devices 1

# Pipeline-parallel MDP generation with TS2 per partition: 4 total devices / PP2.
run_basic_case \
    dense_mdp_pp2_ts2_onnx_compile full "Llama MDP generation using ONNX cuts (PP2 x TS2)" \
    --model-name "${DENSE_MODEL}" \
    --dtype float16 --stage prefill --compile-only \
    --prefill-seq-len 128 --ctx-len 256 \
    --num-cores "${NUM_CORES}" --num-devices 4 \
    --mdp-num-partitions 2 --mdp-strategy onnx
run_basic_case \
    dense_mdp_pp2_ts2_intersection_compile full "Llama MDP generation using compiler/ONNX intersection" \
    --model-name "${DENSE_MODEL}" \
    --dtype float16 --stage prefill --compile-only \
    --prefill-seq-len 128 --ctx-len 256 \
    --num-cores "${NUM_CORES}" --num-devices 4 \
    --mdp-num-partitions 2 --mdp-strategy intersection

# Supported text blocking flavours. These are compile-only because each mode
# creates a distinct long-context QPC; the representative KV mode below runs.
run_blocking_decode h --head-block-size 8
run_blocking_decode q --num-q-blocks 2
run_blocking_decode kv --num-kv-blocks 2
run_blocking_decode qkv --num-q-blocks 2 --num-kv-blocks 2
run_blocking_decode hqkv --head-block-size 8 --num-q-blocks 2 --num-kv-blocks 2
run_blocking_decode kv_headpar --num-kv-blocks 2 --headpar-split "${NUM_CORES}"

run_basic_case \
    blocking_kv_runtime full "Llama blocked-KV end-to-end runtime" \
    --model-name "${DENSE_MODEL}" \
    --prompt "Explain blocked attention." \
    --dtype float16 \
    --prefill-seq-len 1 --ctx-len "${BLOCKING_CTX_LEN}" --generation-len "${GENERATION_LEN}" \
    --num-cores "${NUM_CORES}" --num-devices "${BLOCKING_NUM_DEVICES}" \
    --device-group "$(device_group "${BLOCKING_DEVICE_GROUP}")" \
    --mxfp6-matmul --mxint8-kv-cache --use-onnx-subfunctions --user-tiled \
    --enable-blocking --blocking-mode kv --num-kv-blocks 2

# Prefill blocking variants used by disaggregated serving.
run_blocking_prefill prefill_q --num-q-blocks 2
run_blocking_prefill prefill_kv --num-kv-blocks 2
run_blocking_prefill prefill_qkv --num-q-blocks 2 --num-kv-blocks 2
run_blocking_prefill prefill_online --num-q-blocks 2 --num-kv-blocks 2

if [[ "${RUN_DISAGG}" == "1" ]]; then
    DISAGG_COMMON=(
        --model-name "${MOE_MODEL}"
        --prompt "Explain disaggregated serving." "Why use pipeline parallelism?"
        --dtype float16
        --disaggregated --full-batch-size "${DISAGG_FULL_BATCH_SIZE}"
        --prefill-seq-len "${DISAGG_PREFILL_SEQ_LEN}"
        --ctx-len "${DISAGG_CTX_LEN}"
        --generation-len "${DISAGG_GENERATION_LEN}"
        --num-cores "${NUM_CORES}"
        --prefill-num-devices 4 --decode-num-devices 2
        --prefill-device-group "$(device_group "${DISAGG_PREFILL_DEVICE_GROUP}")"
        --decode-device-group "$(device_group "${DISAGG_DECODE_DEVICE_GROUP}")"
        --mdp-num-partitions 2 --mdp-strategy onnx
        --moe-expert-parallel-chunk-size "${DISAGG_PREFILL_SEQ_LEN}"
        --mxfp6-matmul --mxint8-kv-cache
        --decode-aic-enable-depth-first --prefill-aic-enable-depth-first
    )

    run_basic_case \
        gpt_oss_disagg_pp2_ts2_runtime full "GPT-OSS DMA disaggregated runtime (PP2 x TS2 prefill, TS2 decode)" \
        "${DISAGG_COMMON[@]}"
    run_basic_case \
        gpt_oss_disagg_pp2_ts2_subfunctions_runtime full "GPT-OSS disaggregated runtime with ONNX subfunctions" \
        "${DISAGG_COMMON[@]}" --use-onnx-subfunctions
    run_basic_case \
        gpt_oss_disagg_pp2_ts2_ccl_runtime full "GPT-OSS disaggregated runtime with CCL" \
        "${DISAGG_COMMON[@]}" \
        --ccl-prefill "${DISAGG_CTX_LEN}" \
        --ccl-decode "${DISAGG_CTX_LEN}"
    run_basic_case \
        gpt_oss_disagg_pp2_ts2_blocked_runtime full "GPT-OSS disaggregated runtime with stage-specific blocking" \
        "${DISAGG_COMMON[@]}" --use-onnx-subfunctions \
        --prefill-blocking-mode prefill_online --decode-blocking-mode kv_headpar \
        --num-kv-blocks 2 --num-q-blocks 2 --headpar-split "${NUM_CORES}" \
        --prefill-user-tiled
else
    skip_case gpt_oss_disagg_pp2_ts2_runtime full "GPT-OSS DMA disaggregated runtime" "RUN_DISAGG=${RUN_DISAGG}"
    skip_case \
        gpt_oss_disagg_pp2_ts2_subfunctions_runtime full \
        "GPT-OSS disaggregated runtime with ONNX subfunctions" "RUN_DISAGG=${RUN_DISAGG}"
    skip_case gpt_oss_disagg_pp2_ts2_ccl_runtime full "GPT-OSS disaggregated runtime with CCL" "RUN_DISAGG=${RUN_DISAGG}"
    skip_case \
        gpt_oss_disagg_pp2_ts2_blocked_runtime full \
        "GPT-OSS disaggregated runtime with stage-specific blocking" "RUN_DISAGG=${RUN_DISAGG}"
fi

print_summary
if ((${#FAILED_CASES[@]})); then
    exit 1
fi

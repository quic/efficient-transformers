#!/usr/bin/env bash
# run_spd_all_datasets.sh — Run SPD benchmark across all three datasets
#
# Usage:
#   bash run_spd_all_datasets.sh
#
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
HF_TOKEN="${HF_TOKEN:?Set HF_TOKEN environment variable}"



# ── LLAMA QPC paths ─────────────────────────────────────────────────────────────────
# MODEL_NAME="meta-llama/Llama-3.1-8B-Instruct"
# HF_TOKEN="<set via environment variable>"
# TLM_QPC="/local/mnt/workspace/fannanya/aisyssol/cache/testing_fc_norm/LlamaForCausalLM/LlamaForCausalLM-dc1bb6ab4d8ca588/qpc-344c53fb488a0115/qpc"
# DLM_QPC="/local/mnt/workspace/fannanya/aisyssol/cache/testing/DFlashDraftModel/Qwen3ForCausalLM-31715e599db49bf1/qpc-e7b71407e263cffb/qpc"
# NOISE_EMBED="/local/mnt/workspace/fannanya/dflash/noise_embedding/llama8.1b_mask_id.npy"
# BLOCK_SIZE=10
# HIDDEN_SIZE=4096

# ## ── GPT_OSS QPC paths ─────────────────────────────────────────────────────────────────
# MODEL_NAME="openai/gpt-oss-20b"
# TLM_QPC="/local/mnt/workspace/fannanya/aisyssol/cache/testing_tlm/GptOssForCausalLM_disag/GptOssForCausalLM-7622afd7d6a9ace3/qpc-a0c1dbf3f669b244/qpc"
# DLM_QPC="/local/mnt/workspace/fannanya/aisyssol/cache/testing/DFlashDraftModel/Qwen3ForCausalLM-ce0589b7397caf3c/qpc-8451377d4e685fec/qpc"
# # 16 DLM_QPC="/local/mnt/workspace/fannanya/aisyssol/cache/testing_tlm/DFlashDraftModel/Qwen3ForCausalLM-ce0589b7397caf3c/qpc-c6edb48a49e8141e/qpc"
# NOISE_EMBED="/local/mnt/workspace/fannanya/dflash/noise_embedding/GPT-OSS-20B-noise-embeds.npy"
# BLOCK_SIZE=8
# HIDDEN_SIZE=2880
# OUTPUT_DIR="$SCRIPT_DIR/results-gptoss-20b-all-datasets"
# ── Inference hyper-parameters ────────────────────────────────────────────────
# ── Qwen 3-4b QPC paths ─────────────────────────────────────────────────────────────────
# MODEL_NAME="Qwen/Qwen3-4B"

# TLM_QPC="/local/mnt/workspace/fannanya/aisyssol/cache/testing/Qwen3ForCausalLM/Qwen3ForCausalLM-2cc6517527b5b929/qpc-29f6f83d70409485/qpc"
# DLM_QPC="/local/mnt/workspace/fannanya/aisyssol/cache/testing/DFlashDraftModel/Qwen3ForCausalLM-3b6e3ffa6fce521c/qpc-0d89fb537a0365ca/qpc"
# NOISE_EMBED="/local/mnt/workspace/fannanya/dflash/noise_embedding/Qwen3-4B-noise-embeds.npy"
# BLOCK_SIZE=16
# HIDDEN_SIZE=2560


# ── Sample limit (set to 0 to run all samples) ────────────────────────────────
NUM_SAMPLES=0

# ── Output directory for CSV results ─────────────────────────────────────────

# ── Device assignment ─────────────────────────────────────────────────────────
TLM_DEVICES="12 13 14 15"
# DLM_DEVICES="12 13 14 15"
# TLM_DEVICES="4 5 6 7"
DLM_DEVICES="4 5 6 7"
ITERATION=300
CTX_LEN=4096
# ── Datasets to run ───────────────────────────────────────────────────────────
#"gsm8k" "humaneval" "math500"
DATASETS=( "humaneval" "gsm8k" "math500")


# ── Qwen 3-4b QPC paths ─────────────────────────────────────────────────────────────────
MODEL_NAME="Qwen/Qwen3-4B"
#With MXFP6 and MXINT8
TLM_QPC="/local/mnt/workspace/fannanya/aisyssol/cache/qefficient_dflash/Qwen3ForCausalLM/Qwen3ForCausalLM-5076535a53cb6d9f/qpc-5ee761a30d1aea4e/qpc"
#NO MXFP6
# TLM_QPC="/local/mnt/workspace/vjanfaza/aisyssol_scratch/cache/testing_tlm/Qwen3ForCausalLM/Qwen3ForCausalLM-25725621f6becab1/qpc-bb8602b9b6453b51/qpc"
#KV Proj Scaling
# DLM_QPC="/local/mnt/workspace/vjanfaza/aisyssol_scratch/cache/testing_dlm/DFlashDraftModel/Qwen3ForCausalLM-f6abd1372a38cb7a/qpc-b776fce0518a3dd0/qpc"
#No KV Scaling
# DLM_QPC="/local/mnt/workspace/vjanfaza/aisyssol_scratch/cache/testing_dlm/DFlashDraftModel/Qwen3ForCausalLM-f6abd1372a38cb7a/qpc-b776fce0518a3dd0/qpc"
#No KV Scaling with MXFP6 and MXINT8 --> :))
DLM_QPC="/local/mnt/workspace/fannanya/aisyssol/cache/qefficient_dflash/DFlashDraftModel/Qwen3ForCausalLM-d6705de816daa752/qpc-9e63a7af52cb2dee/qpc"

NOISE_EMBED="/local/mnt/workspace/vjanfaza/dflash/noise_embedding/Qwen3-4B_noise_embeds.npy"
BLOCK_SIZE=16
HIDDEN_SIZE=2560
OUTPUT_DIR="$SCRIPT_DIR/results-qwen3-4b-all-datasets-final-4X8kq_proj"
# ─────────────────────────────────────────────────────────────────────────────
NUM_SAMPLES_ARG=()
if [ "${NUM_SAMPLES:-0}" -gt 0 ]; then
    NUM_SAMPLES_ARG=(--num_samples "$NUM_SAMPLES")
fi

echo "======================================================"
echo "  SPD Benchmark — All Datasets"
echo "  Model      : $MODEL_NAME"
echo "  Block Size : $BLOCK_SIZE   Hidden: $HIDDEN_SIZE   Iters: $ITERATION"
echo "  Samples    : ${NUM_SAMPLES:-all} per dataset"
echo "  Output dir : $OUTPUT_DIR"
echo "======================================================"

for DATASET in "${DATASETS[@]}"; do
    echo ""
    echo "------------------------------------------------------"
    echo "  Running dataset: $DATASET"
    echo "------------------------------------------------------"

    python3 "/local/mnt/workspace/fannanya/dflash-final-qefficient/efficient-transformers-main/examples/dflash/dflash_spd_dataset_run_dataset.py" \
        --dataset          "$DATASET"     \
        --tlm_qpc          "$TLM_QPC"     \
        --dlm_qpc          "$DLM_QPC"     \
        --model_name       "$MODEL_NAME"  \
        --noise_embed_path "$NOISE_EMBED" \
        --block_size       "$BLOCK_SIZE"  \
        --hidden_size      "$HIDDEN_SIZE" \
        --iteration        "$ITERATION"   \
        --ctx_len          "$CTX_LEN"     \
        --tlm_devices      $TLM_DEVICES   \
        --dlm_devices      $DLM_DEVICES   \
        --hf_token         "$HF_TOKEN"    \
        --output_dir       "$OUTPUT_DIR"  \
        "${NUM_SAMPLES_ARG[@]}"

    echo ""
    echo "  Finished: $DATASET"
done

echo ""
echo "======================================================"
echo "  All datasets complete."
echo "======================================================"



# ── Qwen 3-8b QPC paths ─────────────────────────────────────────────────────────────────
MODEL_NAME="Qwen/Qwen3-8B"

# TLM_QPC="/local/mnt/workspace/fannanya/aisyssol/cache/testing/Qwen3ForCausalLM/Qwen3ForCausalLM-340faa1ae75d2bc0/qpc-5be1ea933b876f36/qpc"
# DLM_QPC="/local/mnt/workspace/fannanya/aisyssol/cache/testing_dlm/Qwen3ForCausalLM-c900135532ffb90e/qpc-f1ef508bf6281381/qpc"
#With MXFP6 and MXINT8
TLM_QPC="/local/mnt/workspace/vjanfaza/aisyssol_scratch/cache/testing_tlm/Qwen3ForCausalLM/Qwen3ForCausalLM-aaaf3cd22ea1544d/qpc-bd6f251a67805969/qpc"
#NO MXFP6
# DLM_QPC="/local/mnt/workspace/fannanya/aisyssol/cache/testing_dlm_kv_proj_downscale/DFlashDraftModel/Qwen3ForCausalLM-c900135532ffb90e/qpc-bc77fda37c0ddaa7/qpc"
#No KV Scaling with MXFP6 and MXINT8 --> :))
DLM_QPC="/local/mnt/workspace/vjanfaza/aisyssol_scratch/cache/testing_dlm/DFlashDraftModel/Qwen3ForCausalLM-c900135532ffb90e/qpc-b87cc2652e714d6f/qpc"

NOISE_EMBED="/local/mnt/workspace/vjanfaza/dflash/noise_embedding/Qwen3-8B_noise_embeds.npy"
BLOCK_SIZE=16
HIDDEN_SIZE=4096
OUTPUT_DIR="$SCRIPT_DIR/results-qwen3-8b-all-datasets-final-4x8-kq_proj"
# ─────────────────────────────────────────────────────────────────────────────
NUM_SAMPLES_ARG=()
if [ "${NUM_SAMPLES:-0}" -gt 0 ]; then
    NUM_SAMPLES_ARG=(--num_samples "$NUM_SAMPLES")
fi

echo "======================================================"
echo "  SPD Benchmark — All Datasets"
echo "  Model      : $MODEL_NAME"
echo "  Block Size : $BLOCK_SIZE   Hidden: $HIDDEN_SIZE   Iters: $ITERATION"
echo "  Samples    : ${NUM_SAMPLES:-all} per dataset"
echo "  Output dir : $OUTPUT_DIR"
echo "======================================================"

for DATASET in "${DATASETS[@]}"; do
    echo ""
    echo "------------------------------------------------------"
    echo "  Running dataset: $DATASET"
    echo "------------------------------------------------------"

    python3 "$SCRIPT_DIR/dflash_spd_dataset_run_final.py" \
        --dataset          "$DATASET"     \
        --tlm_qpc          "$TLM_QPC"     \
        --dlm_qpc          "$DLM_QPC"     \
        --model_name       "$MODEL_NAME"  \
        --noise_embed_path "$NOISE_EMBED" \
        --block_size       "$BLOCK_SIZE"  \
        --hidden_size      "$HIDDEN_SIZE" \
        --iteration        "$ITERATION"   \
        --ctx_len          "$CTX_LEN"     \
        --tlm_devices      $TLM_DEVICES   \
        --dlm_devices      $DLM_DEVICES   \
        --hf_token         "$HF_TOKEN"    \
        --output_dir       "$OUTPUT_DIR"  \
        "${NUM_SAMPLES_ARG[@]}"

    echo ""
    echo "  Finished: $DATASET"
done

echo ""
echo "======================================================"
echo "  All datasets complete."
echo "======================================================"

# # # ── LLAMA QPC paths ─────────────────────────────────────────────────────────────────
MODEL_NAME="meta-llama/Llama-3.1-8B-Instruct"
HF_TOKEN="${HF_TOKEN:?Set HF_TOKEN environment variable}"
#With MXFP6 and MXINT8
TLM_QPC="/local/mnt/workspace/vjanfaza/aisyssol_scratch/cache/testing_tlm/LlamaForCausalLM/LlamaForCausalLM-dc1bb6ab4d8ca588/qpc-01885152b8551784/qpc"
#NO MXFP6
# DLM_QPC="/local/mnt/workspace/fannanya/aisyssol/cache/testing_dlm_kv_proj_downscale/DFlashDraftModel/Qwen3ForCausalLM-c4d14aa32295de8d/qpc-5cf0129a6db3cbfa/qpc"
#No KV Scaling with MXFP6 and MXINT8 --> :))
DLM_QPC="/local/mnt/workspace/vjanfaza/aisyssol_scratch/cache/testing_dlm/DFlashDraftModel/Qwen3ForCausalLM-c4d14aa32295de8d/qpc-d5751eb932313163/qpc"

NOISE_EMBED="/local/mnt/workspace/vjanfaza/dflash/noise_embedding/Llama-3.1-8B-Instruct_noise_embeds.npy"
BLOCK_SIZE=10
HIDDEN_SIZE=4096
OUTPUT_DIR="$SCRIPT_DIR/results-llama-8b-all-datasets-final-4x8kq_proj"
# ─────────────────────────────────────────────────────────────────────────────
NUM_SAMPLES_ARG=()
if [ "${NUM_SAMPLES:-0}" -gt 0 ]; then
    NUM_SAMPLES_ARG=(--num_samples "$NUM_SAMPLES")
fi

echo "======================================================"
echo "  SPD Benchmark — All Datasets"
echo "  Model      : $MODEL_NAME"
echo "  Block Size : $BLOCK_SIZE   Hidden: $HIDDEN_SIZE   Iters: $ITERATION"
echo "  Samples    : ${NUM_SAMPLES:-all} per dataset"
echo "  Output dir : $OUTPUT_DIR"
echo "======================================================"

for DATASET in "${DATASETS[@]}"; do
    echo ""
    echo "------------------------------------------------------"
    echo "  Running dataset: $DATASET"
    echo "------------------------------------------------------"

    python3 "$SCRIPT_DIR/dflash_spd_dataset_run_final.py" \
        --dataset          "$DATASET"     \
        --tlm_qpc          "$TLM_QPC"     \
        --dlm_qpc          "$DLM_QPC"     \
        --model_name       "$MODEL_NAME"  \
        --noise_embed_path "$NOISE_EMBED" \
        --block_size       "$BLOCK_SIZE"  \
        --hidden_size      "$HIDDEN_SIZE" \
        --iteration        "$ITERATION"   \
        --ctx_len          "$CTX_LEN"     \
        --tlm_devices      $TLM_DEVICES   \
        --dlm_devices      $DLM_DEVICES   \
        --hf_token         "$HF_TOKEN"    \
        --output_dir       "$OUTPUT_DIR"  \
        "${NUM_SAMPLES_ARG[@]}"

    echo ""
    echo "  Finished: $DATASET"
done

echo ""
echo "======================================================"
echo "  All datasets complete."
echo "======================================================"





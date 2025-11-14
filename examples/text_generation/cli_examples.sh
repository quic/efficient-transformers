#!/bin/bash

# -----------------------------------------------------------------------------
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------

# QEfficient CLI Examples for Text Generation
# This script provides a simplified workflow for running text generation on Cloud AI 100

echo "QEfficient CLI Workflow for Text Generation"
echo "==========================================="
echo ""
echo "This example demonstrates the complete workflow using Llama-3.1-8B"
echo ""

# ============================================================================
# STEP 1: EXPORT MODEL TO ONNX
# ============================================================================

echo "Step 1: Export Model to ONNX"
echo "-----------------------------"
echo "Export the HuggingFace model to ONNX format optimized for Cloud AI 100"
echo ""
cat << 'EOF'
python -m QEfficient.cloud.export \
    --model_name meta-llama/Llama-3.1-8B \
    --cache_dir ~/.cache/qeff_cache
EOF
echo ""
echo "This will download the model and convert it to ONNX format."
echo "The ONNX model will be saved in the QEfficient cache directory."
echo ""

# ============================================================================
# STEP 2: COMPILE MODEL TO QPC
# ============================================================================

echo "Step 2: Compile Model to QPC"
echo "-----------------------------"
echo "Compile the ONNX model to Qualcomm Program Container (QPC) format"
echo ""
cat << 'EOF'
python -m QEfficient.cloud.compile \
    --onnx_path ~/.cache/qeff_cache/meta-llama/Llama-3.1-8B/onnx/model.onnx \
    --qpc_path ./qpc_output \
    --batch_size 1 \
    --prompt_len 128 \
    --ctx_len 512 \
    --num_cores 16 \
    --device_group [0] \
    --mxfp6 \
    --mos 1 \
    --aic_enable_depth_first
EOF
echo ""
echo "Compilation parameters:"
echo "  --batch_size: Number of prompts to process simultaneously"
echo "  --prompt_len: Maximum input prompt length"
echo "  --ctx_len: Maximum context length (prompt + generation)"
echo "  --num_cores: Number of AI 100 cores to use (typically 14 or 16)"
echo "  --device_group: Device IDs to use (e.g., [0] for single device, [0,1,2,3] for multi-device)"
echo "  --mxfp6: Enable MXFP6 quantization for better performance"
echo "  --mos: Memory optimization strategy"
echo "  --aic_enable_depth_first: Enable depth-first execution"
echo ""

# ============================================================================
# STEP 3: EXECUTE WITH COMPILED QPC
# ============================================================================

echo "Step 3: Execute Inference with Compiled QPC"
echo "--------------------------------------------"
echo "Run inference using the pre-compiled QPC"
echo ""
cat << 'EOF'
python -m QEfficient.cloud.execute \
    --model_name meta-llama/Llama-3.1-8B \
    --qpc_path ./qpc_output/qpcs \
    --prompt "Write a short story about AI" \
    --device_group [0]
EOF
echo ""
echo "This uses the pre-compiled QPC for fast inference."
echo "You can run this multiple times with different prompts without recompiling."
echo ""

# ============================================================================
# STEP 4: END-TO-END INFERENCE (ALL-IN-ONE)
# ============================================================================

echo "Step 4: End-to-End Inference (Recommended)"
echo "-------------------------------------------"
echo "The 'infer' command handles export, compile, and execute in one step"
echo ""
cat << 'EOF'
python -m QEfficient.cloud.infer \
    --model_name meta-llama/Llama-3.1-8B \
    --batch_size 1 \
    --prompt_len 128 \
    --ctx_len 512 \
    --num_cores 16 \
    --device_group [0] \
    --prompt "Write a short story about AI" \
    --mxfp6 \
    --mxint8_kv_cache \
    --mos 1 \
    --aic_enable_depth_first
EOF
echo ""
echo "This is the recommended approach for most use cases."
echo "It automatically:"
echo "  1. Downloads and exports the model to ONNX (if not cached)"
echo "  2. Compiles to QPC (if not already compiled with these settings)"
echo "  3. Executes inference with your prompt"
echo ""

# ============================================================================
# ADDITIONAL EXAMPLES
# ============================================================================

echo ""
echo "Additional Examples"
echo "==================="
echo ""

echo "Multi-Device Inference (Multi-Qranium)"
echo "---------------------------------------"
cat << 'EOF'
python -m QEfficient.cloud.infer \
    --model_name meta-llama/Llama-3.1-8B \
    --batch_size 1 \
    --prompt_len 128 \
    --ctx_len 512 \
    --num_cores 16 \
    --device_group [0,1,2,3] \
    --prompt "Explain quantum computing" \
    --mxfp6 \
    --mxint8_kv_cache \
    --aic_enable_depth_first
EOF
echo ""

echo "Continuous Batching (Multiple Prompts)"
echo "---------------------------------------"
cat << 'EOF'
python -m QEfficient.cloud.infer \
    --model_name meta-llama/Llama-3.1-8B \
    --full_batch_size 4 \
    --prompt_len 128 \
    --ctx_len 512 \
    --num_cores 16 \
    --device_group [0] \
    --prompt "Hello|Hi there|Good morning|How are you" \
    --mxfp6 \
    --mxint8_kv_cache
EOF
echo ""
echo "Note: Use pipe (|) to separate multiple prompts for continuous batching"
echo ""

echo "Batch Processing from File"
echo "---------------------------"
cat << 'EOF'
python -m QEfficient.cloud.infer \
    --model_name meta-llama/Llama-3.1-8B \
    --full_batch_size 8 \
    --prompt_len 128 \
    --ctx_len 512 \
    --num_cores 16 \
    --device_group [0] \
    --prompts_txt_file_path examples/sample_prompts/prompts.txt \
    --mxfp6 \
    --mxint8_kv_cache
EOF
echo ""

# ============================================================================
# NOTES AND DOCUMENTATION
# ============================================================================

echo ""
echo "Important Notes"
echo "==============="
echo ""
echo "Terminal Compatibility:"
echo "  - Use bash terminal for best compatibility"
echo "  - If using ZSH, wrap device_group in single quotes: '--device_group [0]'"
echo ""
echo "Common Parameters:"
echo "  --model_name: HuggingFace model ID (e.g., meta-llama/Llama-3.1-8B)"
echo "  --prompt: Input text prompt"
echo "  --prompt_len: Maximum input sequence length"
echo "  --ctx_len: Maximum context length (input + output)"
echo "  --num_cores: AI 100 cores (typically 14 or 16)"
echo "  --device_group: Device IDs [0] for single, [0,1,2,3] for multi-device"
echo "  --mxfp6: Enable MXFP6 quantization (recommended)"
echo "  --mxint8_kv_cache: Enable MXINT8 KV cache (recommended)"
echo "  --aic_enable_depth_first: Enable depth-first execution"
echo ""
echo "For More Information:"
echo "  - Full CLI API Reference: https://quic.github.io/efficient-transformers/cli_api.html"
echo "  - Quick Start Guide: https://quic.github.io/efficient-transformers/quick_start.html"
echo "  - Features Guide: https://quic.github.io/efficient-transformers/features_enablement.html"
echo "  - Supported Models: https://quic.github.io/efficient-transformers/validate.html"
echo "  - Examples README: examples/text_generation/README.md"
echo ""

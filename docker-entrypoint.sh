#!/bin/bash
set -e

# ------------------------------------------------------------------
# Workspace
# ------------------------------------------------------------------

export HOME=${HOME:-/workspace}

mkdir -p "$HOME"

# ------------------------------------------------------------------
# Cache locations
# ------------------------------------------------------------------

export XDG_CACHE_HOME="$HOME/.cache"
export XDG_CONFIG_HOME="$HOME/.config"

export HF_HOME="$HOME/.cache/huggingface"
export HF_HUB_CACHE="$HOME/.cache/huggingface/hub"
export TRANSFORMERS_CACHE="$HOME/.cache/huggingface/transformers"

export TMPDIR="$HOME/tmp"
export TORCH_HOME="$HOME/.cache/torch"
export PIP_CACHE_DIR="$HOME/.cache/pip"

export QEFF_HOME="/app/qefficient-library"

# ------------------------------------------------------------------
# Create directories
# ------------------------------------------------------------------

mkdir -p \
    "$XDG_CACHE_HOME" \
    "$XDG_CONFIG_HOME" \
    "$HF_HOME" \
    "$HF_HUB_CACHE" \
    "$TRANSFORMERS_CACHE" \
    "$TMPDIR" \
    "$TORCH_HOME" \
    "$PIP_CACHE_DIR"

# ------------------------------------------------------------------
# Helpful startup info
# ------------------------------------------------------------------

echo "===================================="
echo "HOME=$HOME"
echo "HF_HOME=$HF_HOME"
echo "QEFF_HOME=$QEFF_HOME"
echo "===================================="

exec "$@"

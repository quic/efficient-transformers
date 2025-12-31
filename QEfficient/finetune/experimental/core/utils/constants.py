# -----------------------------------------------------------------------------
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------

"""
Constants used across test files in the experimental finetuning pipeline.
"""

from enum import Enum

# ============================================================================
# Enums
# ============================================================================


class TaskType(str, Enum):
    """Task types for model training."""

    CAUSAL_LM = "CAUSAL_LM"
    SEQ_CLS = "SEQ_CLS"
    SEQ_2_SEQ_LM = "SEQ_2_SEQ_LM"


class DatasetType(str, Enum):
    """Dataset types for training."""

    SEQ_COMPLETION = "seq_completion"
    SEQ_CLASSIFICATION = "seq_classification"


class AutoClassName(str, Enum):
    """Auto class names for model loading."""

    CAUSAL_LM = "AutoModelForCausalLM"
    SEQ_CLS = "AutoModelForSequenceClassification"
    SEQ_2_SEQ_LM = "AutoModelForSeq2SeqLM"


# ============================================================================
# Test Seeds and Ratios
# ============================================================================

TEST_SEED = 42
TEST_SPLIT_RATIO = 0.8

# ============================================================================
# PEFT/LoRA Configuration
# ============================================================================

TEST_LORA_R = 8
TEST_LORA_ALPHA = 16
TEST_LORA_DROPOUT = 0.1
TEST_LORA_TARGET_MODULES_LLAMA = ["q_proj", "v_proj"]
TEST_LORA_TARGET_MODULES_BERT = ["query", "value"]
TEST_LORA_BIAS = "none"

# ============================================================================
# Training Parameters
# ============================================================================

TEST_LEARNING_RATE = 5e-5
TEST_WEIGHT_DECAY = 0.01
TEST_WARMUP_STEPS = 5
TEST_NUM_TRAIN_EPOCHS = 1
TEST_MAX_STEPS = 5
TEST_LOGGING_STEPS = 1
TEST_PER_DEVICE_BATCH_SIZE = 1
TEST_MAX_SEQ_LENGTH_CAUSAL = 256
TEST_MAX_SEQ_LENGTH_SEQ_CLS = 128
TEST_MAX_LENGTH = 128
TEST_NUM_HIDDEN_LAYERS = 2

# ============================================================================
# Dataset Paths and Names
# ============================================================================

# HuggingFace Dataset Names
HF_DATASET_ALPACA = "tatsu-lab/alpaca"
HF_DATASET_GSM8K = "openai/gsm8k"
HF_DATASET_GSM8K_CONFIG = "main"
HF_DATASET_IMDB = "stanfordnlp/imdb"

# Dataset subset size for testing
TEST_DATASET_SUBSET_SIZE = 10

# ============================================================================
# Model Names
# ============================================================================

TEST_MODEL_LLAMA = "meta-llama/Llama-3.2-1B"
TEST_MODEL_SMOLLM = "HuggingFaceTB/SmolLM-135M"

# ============================================================================
# Optimizer Parameters
# ============================================================================

OPT_LEARNING_RATE = 1e-4
OPT_ADAM_BETAS = (0.9, 0.999)
OPT_ADAM_EPS = 1e-8
OPT_SGD_MOMENTUM = 0.9

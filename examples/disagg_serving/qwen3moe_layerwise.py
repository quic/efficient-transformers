# -----------------------------------------------------------------------------
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------

"""Layerwise prefill compile example for Qwen3-MoE (disaggregated serving).

The orchestration loop that previously lived in this script has been moved
behind the ``layerwise=True`` flag on ``.compile()`` / ``.export()``.

Note: ``layerwise=True`` is a provisional API and is scheduled for deprecation
once first-class multi-window export lands. Supported model types:
``qwen3_vl_moe``, ``qwen3_5_moe``, ``qwen3_moe``.
"""

from transformers import AutoConfig, AutoTokenizer

from QEfficient import QEFFAutoModelForCausalLM

model_id = "Qwen/Qwen3-235B-A22B-Instruct-2507"  # weights are not required to convert to fp32
PREFILL_SEQ_LEN = 4
CTX_LEN = 128

config = AutoConfig.from_pretrained(model_id)
tokenizer = AutoTokenizer.from_pretrained(model_id)


qeff_model = QEFFAutoModelForCausalLM.from_pretrained(model_id, config=config, layerwise=True)

qpc_path = qeff_model.compile(
    prefill_seq_len=PREFILL_SEQ_LEN,
    ctx_len=CTX_LEN,
    num_cores=16,
    mxfp6_matmul=True,
    mxint8_kv_cache=True,
    num_devices=1,
    split_retained_state_io=True,
    mos=1,
    aic_enable_depth_first=True,
    num_speculative_tokens=None,
    prefill_only=True,
    enable_chunking=True,
    use_onnx_subfunctions=True,
    layerwise=True,
    layerwise_window_size=1,
)

print(f"QPC path: {qpc_path}")

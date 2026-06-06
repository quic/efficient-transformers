# -----------------------------------------------------------------------------
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------

"""Layer-wise ONNX export + compile for a large Qwen3-MoE causal LM.

The layer-wise flow loads only one window of decoder layers at a time, exports
that window, then splits/prefixes/merges all windows into a single ONNX graph
that is equivalent to a full-model export. This keeps peak host memory bounded
by a single window's weights, enabling export of models that do not fit in
memory all at once.

Usage is identical to a normal QEFFAutoModelForCausalLM export, except:
  * pass ``layerwise=True`` to ``from_pretrained`` (the model is built on the
    ``meta`` device; weights are streamed per window during export), and
  * optionally pass ``layerwise_window_size`` to ``compile`` (defaults to 1).
"""

import time

from transformers import AutoConfig, AutoTokenizer

from QEfficient import QEFFAutoModelForCausalLM

model_id = "yujiepan/qwen3-moe-tiny-random"
# model_id = "Qwen/Qwen3-235B-A22B-Instruct-2507"  # weights are not required to convert to fp32

config = AutoConfig.from_pretrained(model_id)
tokenizer = AutoTokenizer.from_pretrained(model_id)

PREFILL_SEQ_LEN = 4
CTX_LEN = 128

# Build the model on `meta`; per-window weights are streamed during export.
qeff_model = QEFFAutoModelForCausalLM.from_pretrained(model_id, config=config, layerwise=True)

export_start = time.perf_counter()

# A single compile call drives the full layer-wise loop internally:
#   per window -> load weights -> apply transforms -> export window
#   then split -> add prefix -> merge into one final ONNX, then compile.
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
    layerwise_window_size=1,
)

print(f"Layer-wise export + compile completed in {time.perf_counter() - export_start:.2f}s")
print(f"QPC path: {qpc_path}")

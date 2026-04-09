# -----------------------------------------------------------------------------
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
#
# ----------------------------------------------------------------------------


from QEfficient import QEFFAutoModelForCausalLM

model_ref = "tiny-random/gpt-oss-bf16"
# model_ref can also be a local directory containing an edited config.json.
# In benchmark mode this uses the config-only path, so it does not need full model weights.
qeff_model = QEFFAutoModelForCausalLM.from_pretrained(
    model_ref,
    enable_benchmark=True,
)

# Example 1: prefill MoE with FFN blocking
# os.environ["NUM_FFN_BLOCKS"] = "4"
# os.environ["NUM_Q_BLOCKS"] = "4"
# os.environ["ENABLE_OPT_SWA"] = "0"  # baseline blocked path

manifest_path = qeff_model.compile(
    prefill_only=None,
    prefill_seq_len=1,
    ctx_len=16384,
)
report_path = qeff_model.generate(tokenizer=None, prompts=[])

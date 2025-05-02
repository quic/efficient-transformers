# -----------------------------------------------------------------------------
#
# Copyright (c) 2025 Qualcomm Innovation Center, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------

import torch
from transformers import Llama4ForCausalLM

from QEfficient import QEFFAutoModelForCausalLM
from QEfficient.utils._utils import load_hf_tokenizer
from QEfficient.utils.constants import Constants
from QEfficient.utils.run_utils import ApiRunner

torch.manual_seed(42)


# ──────────────────────────────────────────────────────────────────────────────
#  helper : split fused Gate+Up weights and copy into the model
# ──────────────────────────────────────────────────────────────────────────────
def load_all_split_gate_up_weights(
    model: torch.nn.Module,
    sd,
    delete_fused_key: bool = True,
) -> None:
    """
    For every transformer layer inside `model`:
      • expects   <PREFIX>.experts.gate_up_proj   in the *source* `sd`
      • copies halves into
            <PREFIX>.experts.gate_proj     <-- Gate   [E,H,I]
            <PREFIX>.experts.up_proj       <-- Up     [E,H,I]
    """
    num_layers = len(model.model.layers)

    for layer_idx in range(num_layers):
        # ---- build the textual prefix once per layer ----------
        prefix = f"model.layers.{layer_idx}.feed_forward.experts."

        fused_key = prefix + "gate_up_proj"
        gate_key = prefix + "gate_proj"
        up_key = prefix + "up_proj"

        # ---- split  [E,H,2I] → two  [E,H,I]  tensors ----------------------
        fused = sd[fused_key]  # [E, H, 2I]  (no .weight here)
        E, H, two_I = fused.shape
        ffn_dim = two_I // 2
        gate, up = fused.split(ffn_dim, dim=-1)  # views – no copy

        experts = model.model.layers[layer_idx].feed_forward.experts
        experts.gate_proj.data.copy_(gate)
        experts.up_proj.data.copy_(up)

        # ---- update the state-dict so load_state_dict sees the right keys -
        sd[gate_key] = gate
        sd[up_key] = up

        if delete_fused_key:
            del sd[fused_key]  # prevents extra / unexpected key msg

        print(f"[layer {layer_idx:02d}] loaded gate_proj & up_proj from fused tensor  (shape {fused.shape})")


model_id = "meta-llama/Llama-4-Scout-17B-16E-Instruct"
model = Llama4ForCausalLM.from_pretrained(
    model_id, torch_dtype=torch.float32, use_cache=True, attn_implementation="eager"
)
model.eval()

original_sd = model.state_dict()

tokenizer = load_hf_tokenizer(pretrained_model_name_or_path=model_id)
config = model.config
batch_size = len(Constants.INPUT_STR)
api_runner = ApiRunner(
    batch_size,
    tokenizer,
    config,
    Constants.INPUT_STR,
    Constants.PROMPT_LEN,
    Constants.CTX_LEN,
)

qeff_model = QEFFAutoModelForCausalLM(model)
load_all_split_gate_up_weights(qeff_model.model, original_sd)

onnx_model_path = qeff_model.export()
qpc_path = qeff_model.compile(
    prefill_seq_len=128,
    ctx_len=2048,
    num_cores=16,
    mxfp6_matmul=True,
    mxint8_kv_cache=True,
    num_devices=8,
    mos=1,
    aic_enable_depth_first=True,
    num_speculative_tokens=None,
)
print(f"qpc path is {qpc_path}")
exec_info = qeff_model.generate(
    tokenizer, prompts=Constants.INPUT_STR, generation_len=32, device_ids=[0, 1, 2, 3, 4, 5, 6, 7]
)

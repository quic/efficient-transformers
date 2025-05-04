# -----------------------------------------------------------------------------
#
# Copyright (c) 2025 Qualcomm Innovation Center, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------

import torch
import transformers
from transformers import AutoConfig, AutoModelForImageTextToText, AutoProcessor, TextStreamer

from QEfficient import QEFFAutoModelForImageTextToText


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

        # ---- update the state-dict so load_state_dict sees the right keys
        sd[gate_key] = gate
        sd[up_key] = up

        if delete_fused_key:
            del sd[fused_key]

        print(f"[layer {layer_idx:02d}] loaded gate_proj & up_proj from fused tensor  (shape {fused.shape})")


model_id = "meta-llama/Llama-4-Scout-17B-16E-Instruct"
config = AutoConfig.from_pretrained(model_id)
# For Testing Purpose Only
# config.text_config.num_hidden_layers = 1
# config.vision_config.num_hidden_layers = 2

model = AutoModelForImageTextToText.from_pretrained(model_id, attn_implementation="eager", config=config)
model.eval()

qeff_model = QEFFAutoModelForImageTextToText(model, kv_offload=True)
load_all_split_gate_up_weights(qeff_model.model.language_model, model.language_model.state_dict())

# TODO: Map the Vision Encoder to FP16 Only and Disable MXFP6 For Better Accuracy.
qeff_model.compile(
    prefill_seq_len=128,
    ctx_len=3072,
    img_size=336,
    num_cores=16,
    num_devices=8,
    batch_size_times_num_tiles=17,
    mxfp6_matmul=True,
    mxint8_kv_cache=True,
    aic_enable_depth_first=True,
    mos=1,
)

image_url = (
    "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/datasets/cat_style_layout.png"
)

messages = [
    {
        "role": "user",
        "content": [
            {"type": "image", "url": image_url},
            {"type": "text", "text": "Can you describe the image in detail."},
        ],
    },
]

tokenizer = transformers.AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
processor = AutoProcessor.from_pretrained(model_id)
inputs = processor.apply_chat_template(
    messages,
    add_generation_prompt=True,
    tokenize=True,
    return_dict=True,
    return_tensors="pt",
)
inputs["pixel_values"] = inputs["pixel_values"].to(torch.float32)
streamer = TextStreamer(tokenizer)
output = qeff_model.generate(inputs=inputs, device_ids=[0, 1, 2, 3, 4, 5, 6, 7], generation_len=100)
print(output.generated_ids)
print(tokenizer.batch_decode(output.generated_ids))
print(output)

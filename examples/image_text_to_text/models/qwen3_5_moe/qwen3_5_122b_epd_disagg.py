# -----------------------------------------------------------------------------
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------

"""
End-to-end E-P-D disaggregated inference example for Qwen3.5-122B-A10B.

Three QPCs are compiled separately:
  - Vision encoder  (skip_lang=True)
  - Lang prefill    (prefill_only=True, skip_vision=True)
  - Lang decode     (skip_vision=True)

KV-cache buffers are tagged with KV_CACHE_PREFIX so vLLM can regex-select
them for device-to-device transfer between prefill and decode workers.
Both prefill and decode use mxint8_kv_cache for a consistent wire dtype.
"""

from time import perf_counter

import numpy as np
import requests
import torch
import transformers
from PIL import Image
from qwen_vl_utils import process_vision_info
from transformers import AutoConfig, AutoProcessor

from QEfficient import QEFFAutoModelForImageTextToText
from QEfficient.generation.cloud_infer import QAICInferenceSession

# ---------------------------------------------------------------------------
# Global configuration
# ---------------------------------------------------------------------------
MODEL_ID = "Qwen/Qwen3.5-122B-A10B"
KV_CACHE_PREFIX = "vllmKvCache"

BATCH_SIZE = 1
PREFILL_SEQ_LEN = 32
CTX_LEN = 1024
GENERATION_LEN = 256

IMG_HEIGHT = 354
IMG_WIDTH = 536

NUM_CORES = 16
NUM_DEVICES_LANG = 4
NUM_DEVICES_VISION = 1
MOS = 1

# ---------------------------------------------------------------------------
# Truncated config for faster testing (reduce num_hidden_layers for full run)
# ---------------------------------------------------------------------------
config = AutoConfig.from_pretrained(MODEL_ID)
config.text_config.num_hidden_layers = 4
config.torch_dtype = "float16"
layer_types = list(getattr(config.text_config, "layer_types", []))
if len(layer_types) < config.text_config.num_hidden_layers:
    layer_types.extend(["full_attention"] * (config.text_config.num_hidden_layers - len(layer_types)))
config.text_config.layer_types = layer_types[: config.text_config.num_hidden_layers]

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
kv_infix = f"_{KV_CACHE_PREFIX}" if KV_CACHE_PREFIX else ""


def _update_retained_states(target_inputs, source_outputs):
    for layer_idx, layer_type in enumerate(config.text_config.layer_types):
        if layer_type == "full_attention":
            target_inputs[f"past_key.{layer_idx}{kv_infix}"] = source_outputs[
                f"past_key.{layer_idx}{kv_infix}_RetainedState"
            ]
            target_inputs[f"past_value.{layer_idx}{kv_infix}"] = source_outputs[
                f"past_value.{layer_idx}{kv_infix}_RetainedState"
            ]
        else:
            target_inputs[f"conv_state.{layer_idx}{kv_infix}"] = source_outputs[
                f"conv_state.{layer_idx}{kv_infix}_RetainedState"
            ]
            target_inputs[f"recurrent_state.{layer_idx}{kv_infix}"] = source_outputs[
                f"recurrent_state.{layer_idx}{kv_infix}_RetainedState"
            ]


# ---------------------------------------------------------------------------
# Model + processor
# ---------------------------------------------------------------------------
qeff_model = QEFFAutoModelForImageTextToText.from_pretrained(
    MODEL_ID, attn_implementation="eager", kv_offload=True, config=config
)
tokenizer = transformers.AutoTokenizer.from_pretrained(MODEL_ID)
processor = AutoProcessor.from_pretrained(MODEL_ID)

# ---------------------------------------------------------------------------
# Compile: vision encoder
# ---------------------------------------------------------------------------
vision_qpc_path = qeff_model.compile(
    batch_size=BATCH_SIZE,
    prefill_seq_len=PREFILL_SEQ_LEN,
    ctx_len=CTX_LEN,
    height=IMG_HEIGHT,
    width=IMG_WIDTH,
    num_cores=NUM_CORES,
    num_devices=NUM_DEVICES_VISION,
    mos=MOS,
    mxfp6_matmul=True,
    aic_enable_depth_first=True,
    skip_vision=False,
    split_model_io=True,
    skip_lang=True,
    use_onnx_subfunctions=True,
    layerwise=False,
)
print("vision_qpc_path:", str(vision_qpc_path.get("vision_qpc_path")))

# ---------------------------------------------------------------------------
# Compile: lang prefill
# ---------------------------------------------------------------------------
prefill_qpc_path = qeff_model.compile(
    batch_size=BATCH_SIZE,
    prefill_seq_len=PREFILL_SEQ_LEN,
    ctx_len=CTX_LEN,
    height=IMG_HEIGHT,
    width=IMG_WIDTH,
    num_cores=NUM_CORES,
    num_devices=NUM_DEVICES_LANG,
    mos=MOS,
    mxfp6_matmul=True,
    mxint8_kv_cache=True,
    aic_enable_depth_first=True,
    skip_vision=True,
    split_model_io=True,
    use_onnx_subfunctions=True,
    prefill_only=True,
    layerwise=True,
    layerwise_window_size=1,
    kv_cache_prefix=KV_CACHE_PREFIX,
)
print("prefill_qpc_path:", str(prefill_qpc_path.get("lang_prefill_qpc_path")))

# ---------------------------------------------------------------------------
# Compile: lang decode
# ---------------------------------------------------------------------------
decode_qpc_path = qeff_model.compile(
    batch_size=BATCH_SIZE,
    prefill_seq_len=1,
    ctx_len=CTX_LEN,
    height=IMG_HEIGHT,
    width=IMG_WIDTH,
    num_cores=NUM_CORES,
    num_devices=NUM_DEVICES_LANG,
    mos=MOS,
    mxfp6_matmul=True,
    mxint8_kv_cache=True,
    aic_enable_depth_first=True,
    skip_vision=True,
    split_model_io=True,
    use_onnx_subfunctions=False,
    layerwise=True,
    layerwise_window_size=1,
    kv_cache_prefix=KV_CACHE_PREFIX,
)
print("decode_qpc_path:", str(decode_qpc_path.get("lang_decode_qpc_path")))

# ---------------------------------------------------------------------------
# Sessions
# ---------------------------------------------------------------------------
vision_session = QAICInferenceSession(vision_qpc_path.get("vision_qpc_path"))
lang_prefill_session = QAICInferenceSession(prefill_qpc_path.get("lang_prefill_qpc_path"))
lang_decode_session = QAICInferenceSession(decode_qpc_path.get("lang_decode_qpc_path"))

# ---------------------------------------------------------------------------
# Inputs
# ---------------------------------------------------------------------------
image_url = "https://picsum.photos/id/237/536/354"
image = Image.open(requests.get(image_url, stream=True).raw)

messages = [
    [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": "Describe all the colors seen in the image."},
            ],
        }
    ]
] * BATCH_SIZE

texts = [processor.apply_chat_template(msg, tokenize=False, add_generation_prompt=True) for msg in messages]
image_inputs, video_inputs = process_vision_info(messages)
inputs = processor(text=texts, images=image_inputs, videos=video_inputs, padding=True, return_tensors="pt")
inputs = qeff_model.model.prepare_inputs_for_generation(
    inputs=inputs, prefill_seq_len=PREFILL_SEQ_LEN, batch_size=BATCH_SIZE
)

pad_token_id = 1
input_ids_length = inputs["input_ids"].shape[1]
num_chunks = -(input_ids_length // -PREFILL_SEQ_LEN)
padded_len = num_chunks * PREFILL_SEQ_LEN

inputs["input_ids"] = torch.nn.functional.pad(
    inputs["input_ids"], (0, padded_len - input_ids_length), "constant", pad_token_id
)
inputs["attention_mask"] = torch.nn.functional.pad(
    inputs["attention_mask"], (0, padded_len - input_ids_length), "constant", 0
)

for k, v in inputs.items():
    inputs[k] = np.array(v)

vision_inputs = {
    k: v
    for k, v in inputs.items()
    if k in {"pixel_values", "image_masks", "image_input_idx", "valid_idx", "aspect_ratio_ids", "aspect_ratio_mask"}
}
vision_inputs.update(
    {k: vision_inputs[k].astype("float16") for k in {"pixel_values", "image_masks"} if k in vision_inputs}
)

# ---------------------------------------------------------------------------
# Vision forward
# ---------------------------------------------------------------------------
vision_start = perf_counter()
vision_outputs = vision_session.run(vision_inputs)
vision_end = perf_counter()

# ---------------------------------------------------------------------------
# Prefill
# ---------------------------------------------------------------------------
lang_inputs = {k: v for k, v in inputs.items() if k not in vision_inputs}
if "position_ids" in inputs:
    lang_inputs["position_ids"] = inputs["position_ids"]
    lang_inputs.pop("attention_mask", None)
else:
    lang_inputs["position_ids"] = np.where(lang_inputs.pop("attention_mask"), np.arange(padded_len), -1)

lang_inputs["image_idx"] = np.array([[0]])
lang_inputs["vision_embeds"] = vision_outputs["vision_embeds"]

lang_prefill_session.set_buffers(vision_outputs)

lang_start = perf_counter()
chunk_inputs = lang_inputs.copy()
for i in range(num_chunks):
    chunk_inputs["input_ids"] = lang_inputs["input_ids"][:, i * PREFILL_SEQ_LEN : (i + 1) * PREFILL_SEQ_LEN]
    chunk_inputs["position_ids"] = lang_inputs["position_ids"][..., i * PREFILL_SEQ_LEN : (i + 1) * PREFILL_SEQ_LEN]
    outputs = lang_prefill_session.run(chunk_inputs)
    _update_retained_states(chunk_inputs, outputs)
    chunk_inputs["image_idx"] = outputs["image_idx_output"]

prefill_time = perf_counter() - lang_start + vision_end - vision_start
print(f"Prefill time : {prefill_time:.2f} secs")

# ---------------------------------------------------------------------------
# Decode
# ---------------------------------------------------------------------------
all_outputs = [np.argmax(outputs["logits"])]

decode_inputs = {
    "input_ids": np.argmax(outputs["logits"]).reshape(1, 1),
    "position_ids": np.max(lang_inputs["position_ids"], axis=-1, keepdims=True) + 1,
    "image_idx": outputs["image_idx_output"],
    "vision_embeds": outputs["vision_embeds_RetainedState"],
}
_update_retained_states(decode_inputs, outputs)

st = perf_counter()
decode_out = lang_decode_session.run(decode_inputs)
print(f"First decode step: {perf_counter() - st:.3f} secs")

all_outputs.append(np.argmax(decode_out["logits"]))
pos_id = np.max(decode_inputs["position_ids"], axis=-1, keepdims=True) + 1
loop_inputs = {
    "input_ids": np.argmax(decode_out["logits"]).reshape(1, 1),
    "position_ids": pos_id,
    "image_idx": decode_out["image_idx_output"],
    "vision_embeds": decode_out["vision_embeds_RetainedState"],
}
_update_retained_states(loop_inputs, decode_out)

st = perf_counter()
for _ in range(GENERATION_LEN - 2):
    decode_out = lang_decode_session.run(loop_inputs)
    all_outputs.append(np.argmax(decode_out["logits"]))
    pos_id += 1
    _update_retained_states(loop_inputs, decode_out)
    loop_inputs.update(
        {
            "input_ids": np.argmax(decode_out["logits"]).reshape(1, 1),
            "position_ids": pos_id,
        }
    )
ft = perf_counter()

print(f"Decode throughput: {(GENERATION_LEN - 2) / (ft - st):.2f} tok/sec")
print(f"\nOutput:\n{tokenizer.decode(all_outputs)}")

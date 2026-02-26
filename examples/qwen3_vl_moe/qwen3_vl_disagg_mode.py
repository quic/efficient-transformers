# -----------------------------------------------------------------------------
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------

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

model_id = "Qwen/Qwen3-VL-30B-A3B-Instruct"
config = AutoConfig.from_pretrained(model_id)

# TODO clean up this script
# For Testing Purpose Only
# config.vision_config.depth = 1
# config.text_config.num_hidden_layers = 1
num_devices = 4

qeff_model = QEFFAutoModelForImageTextToText.from_pretrained(
    model_id, attn_implementation="eager", kv_offload=True, config=config
)
tokenizer = transformers.AutoTokenizer.from_pretrained(model_id)
processor = AutoProcessor.from_pretrained(model_id)

PREFILL_SEQ_LEN = 128
CTX_LEN = 4096
BS = 1
torch.manual_seed(0)

skip_vision = True

prefill_qpc_path = qeff_model.compile(
    batch_size=BS,
    prefill_seq_len=PREFILL_SEQ_LEN,
    ctx_len=CTX_LEN,
    height=354,
    width=536,
    num_cores=16,
    num_devices=num_devices,
    mxfp6_matmul=True,
    mxint8_kv_cache=True,
    retain_full_kv=True,
    split_retained_state_io=True,
    retained_state=True,
    mos=1,
    aic_enable_depth_first=True,
    prefill_only=True,
    enable_chunking=True,
    skip_vision=True,
    use_onnx_subfunctions=False,
)


decode_qpc_path = qeff_model.compile(
    batch_size=BS,
    prefill_seq_len=1,
    ctx_len=CTX_LEN,
    height=354,
    width=536,
    num_cores=16,
    num_devices=num_devices,
    mxfp6_matmul=True,
    mxint8_kv_cache=True,
    retain_full_kv=True,
    split_retained_state_io=True,
    retained_state=True,
    mos=1,
    aic_enable_depth_first=True,
    prefill_only=True,
    enable_chunking=True,
    skip_vision=True,
    use_onnx_subfunctions=False,
)

if skip_vision:  # for only LLM with DA
    lang_prefill_session = QAICInferenceSession(prefill_qpc_path)
    lang_decode_session = QAICInferenceSession(decode_qpc_path)
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "Tell me about yourself."},
            ],
        },
    ]
else:
    vision_qpc_path = qeff_model.compile(
        batch_size=BS,
        prefill_seq_len=PREFILL_SEQ_LEN,
        ctx_len=CTX_LEN,
        height=354,
        width=536,
        num_cores=16,
        num_devices=num_devices,
        retained_state=True,
        mos=1,
        aic_enable_depth_first=True,
        # prefill_only=True,
        # enable_chunking=True,
        skip_vision=skip_vision,
        skip_lang=True,
        use_onnx_subfunctions=False,
    )
    vision_session = QAICInferenceSession(vision_qpc_path)
    lang_prefill_session = QAICInferenceSession(prefill_qpc_path)
    lang_decode_session = QAICInferenceSession(decode_qpc_path)
    ### IMAGE + TEXT ###
    image_url = "https://picsum.photos/id/237/536/354"
    image = Image.open(requests.get(image_url, stream=True).raw)

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": "Descibe all the colors seen in the image."},
            ],
        },
    ]


########################### example for inference

messages = [messages] * BS

texts = [processor.apply_chat_template(msg, tokenize=False, add_generation_prompt=True) for msg in messages]

image_inputs, video_inputs = process_vision_info(messages)
inputs = processor(
    text=texts,
    images=image_inputs,
    videos=video_inputs,
    padding=True,
    return_tensors="pt",
)
inputs = qeff_model.model.prepare_inputs_for_generation(inputs=inputs, prefill_seq_len=PREFILL_SEQ_LEN, batch_size=BS)

pad_token_id = 1
input_len = inputs["attention_mask"].sum(1, keepdims=True)
input_ids_length = inputs["input_ids"].shape[1]
num_chunks = -(input_ids_length // -PREFILL_SEQ_LEN)  # ceil divide without float
padded_len = num_chunks * PREFILL_SEQ_LEN  # Convert to a multiple of prompt_len
generation_len = CTX_LEN - input_len.max()
print(f"generation_len : {generation_len}")
generated_ids = np.full((BS, generation_len + 1), pad_token_id)


inputs["input_ids"] = torch.nn.functional.pad(
    inputs["input_ids"],
    (0, padded_len - input_ids_length),
    "constant",
    pad_token_id,
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

vision_inputs_fp16 = {"pixel_values", "image_masks"}
vision_inputs.update({k: vision_inputs[k].astype("float16") for k in vision_inputs_fp16 if k in vision_inputs})

vision_start = perf_counter()
vision_outputs = {}
if vision_inputs:
    vision_outputs = vision_session.run(vision_inputs)
vision_end = perf_counter()

# TODO : pass vision_embeds_RetainedState to prefill
# vision_outputs["vision_embeds_RetainedState"]
# *** KeyError: 'vision_embeds_RetainedState'

lang_inputs = {k: v for k, v in inputs.items() if k not in vision_inputs}
if "position_ids" in inputs:
    lang_inputs["position_ids"] = inputs["position_ids"]
    lang_inputs.pop("attention_mask")
else:
    lang_inputs["position_ids"] = np.where(
        lang_inputs.pop("attention_mask"), np.arange(padded_len), -1
    )  # Need to use -1 as position_ids for invalid tokens

lang_inputs["image_idx"] = np.array([[0]])


# RUN prefill
lang_start = perf_counter()

all_outputs = []
for i in range(num_chunks):
    chunk_inputs = lang_inputs.copy()
    chunk_inputs["input_ids"] = lang_inputs["input_ids"][:, i * PREFILL_SEQ_LEN : (i + 1) * PREFILL_SEQ_LEN]
    chunk_inputs["position_ids"] = lang_inputs["position_ids"][..., i * PREFILL_SEQ_LEN : (i + 1) * PREFILL_SEQ_LEN]
    outputs = lang_prefill_session.run(chunk_inputs)
    for i in range(config.text_config.num_hidden_layers):
        lang_inputs[f"past_key.{i}"] = outputs[f"past_key.{i}_RetainedState"]
        lang_inputs[f"past_value.{i}"] = outputs[f"past_value.{i}_RetainedState"]

    chunk_inputs["image_idx"] = outputs["image_idx_output"]
prefill_time = perf_counter() - lang_start + vision_end - vision_start
print(f"Prefill time  :{prefill_time:.2f} secs")


all_outputs.append(np.argmax(outputs["logits"]))
decode_inputs = {
    "input_ids": np.argmax(outputs["logits"]).reshape(1, 1),
    "position_ids": np.max(lang_inputs["position_ids"]).reshape(1, 1) + 1,
}

for i in range(config.text_config.num_hidden_layers):
    decode_inputs[f"past_key.{i}"] = outputs[f"past_key.{i}_RetainedState"]
    decode_inputs[f"past_value.{i}"] = outputs[f"past_value.{i}_RetainedState"]
    decode_inputs["vision_embeds_RetainedState"] = outputs["vision_embeds_RetainedState"]
    decode_inputs["image_idx_output"] = outputs["image_idx_output"]

st = perf_counter()
decode_out = lang_decode_session.run(decode_inputs)
print(f"time for first run of decode with KV as input = {perf_counter() - st} sec\n")

all_outputs.append(np.argmax(decode_out["logits"]))
pos_id = np.max(decode_inputs["position_ids"]).reshape(1, 1) + 1
loop_decode_inputs = {
    "input_ids": np.argmax(decode_out["logits"]).reshape(1, 1),
    "position_ids": pos_id,
}


for i in range(config.text_config.num_hidden_layers):
    loop_decode_inputs[f"past_key.{i}"] = decode_out[f"past_key.{i}_RetainedState"]
    loop_decode_inputs[f"past_value.{i}"] = decode_out[f"past_value.{i}_RetainedState"]
    loop_decode_inputs["vision_embeds_RetainedState"] = decode_out["vision_embeds_RetainedState"]
    loop_decode_inputs["image_idx_output"] = decode_out["image_idx_output"]


st = perf_counter()
for i in range(generation_len - 2):
    decode_out = lang_decode_session.run(loop_decode_inputs)
    all_outputs.append(np.argmax(decode_out["logits"]))
    pos_id += 1
    for j in range(config.text_config.num_hidden_layers):
        loop_decode_inputs[f"past_key.{j}"] = decode_out[f"past_key.{j}_RetainedState"]
        loop_decode_inputs[f"past_value.{j}"] = decode_out[f"past_value.{j}_RetainedState"]
        loop_decode_inputs["vision_embeds_RetainedState"] = decode_out["vision_embeds_RetainedState"]
        loop_decode_inputs["image_idx_output"] = decode_out["image_idx_output"]

    loop_decode_inputs.update(
        {
            "input_ids": np.argmax(decode_out["logits"]).reshape(1, 1),
            "position_ids": pos_id,
        }
    )
ft = perf_counter()

print(f"decode tok/sec={(generation_len - 2) / (ft - st)}")
print(f"\noutput\n{tokenizer.decode(all_outputs)}")

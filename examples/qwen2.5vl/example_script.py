# -----------------------------------------------------------------------------
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------

import torch
import torch.nn.functional as F
import transformers
from qwen_vl_utils import process_vision_info
from transformers import AutoConfig, AutoProcessor, TextStreamer

from QEfficient import QEFFAutoModelForImageTextToText

## For AWQ model update pytorch version to 2.8.*
model_id = "Qwen/Qwen2.5-VL-32B-Instruct"
config = AutoConfig.from_pretrained(model_id)

## Use complete model without changing num_hidden_layers as it will not work for TF version 4.55.0 for Qwen2.5VL model

qeff_model = QEFFAutoModelForImageTextToText.from_pretrained(
    model_id, attn_implementation="eager", kv_offload=True, config=config
)
tokenizer = transformers.AutoTokenizer.from_pretrained(model_id)
processor = AutoProcessor.from_pretrained(model_id)

batch_size = 1

## Define Num_frames ##
num_frames = 16

## Vision + Text ##
qeff_model.compile(
    batch_size=batch_size,
    prefill_seq_len=128,
    ctx_len=16384,
    num_cores=16,
    num_devices=8,
    height=910,
    width=512,
    num_frames=num_frames,
    mxfp6_matmul=True,
    mxint8_kv_cache=True,
    aic_enable_depth_first=True,
    mos=1,
)

content = []
for i in range(1, num_frames + 1):
    frame = {
        "type": "image",
        "image": f"./frame_{i}.jpg",
    }
    content.append(frame)

content.append({"type": "text", "text": "Describe the video"})

messages = [
    {
        "role": "user",
        "content": content,
    }
]

text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

image_inputs, video_inputs = process_vision_info(messages)

inputs = processor(
    text=[text],
    images=image_inputs,
    videos=video_inputs,
    padding=True,
    return_tensors="pt",
)

input_ids_length = inputs["input_ids"].shape[1]

inputs["position_ids"] = torch.arange(input_ids_length).view(1, 1, input_ids_length).expand(-1, batch_size, -1)

pos_ids, rope_deltas = qeff_model.model.model.get_rope_index(
    inputs["input_ids"],
    inputs["image_grid_thw"],
    video_grid_thw=None,
    second_per_grid_ts=None,
    attention_mask=inputs["attention_mask"],
)

inputs["position_ids"] = torch.cat((inputs["position_ids"], pos_ids), dim=0)

prefill_seq_len = 128
num_chunks = -(input_ids_length // -prefill_seq_len)  # ceil divide without float
padded_len = num_chunks * prefill_seq_len  # Convert to a multiple of prompt_len

inputs["position_ids"] = F.pad(
    inputs["position_ids"], pad=(0, padded_len - input_ids_length), mode="constant", value=-1
)

inputs.pop("image_grid_thw")
streamer = TextStreamer(tokenizer)
output = qeff_model.generate(inputs=inputs, generation_len=200, multi_frame_inference=True, num_franms=num_frames)
print(output.generated_ids)
print(tokenizer.batch_decode(output.generated_ids))
print(output)

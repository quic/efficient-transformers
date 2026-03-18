# -----------------------------------------------------------------------------
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------

import requests
import transformers
from PIL import Image
from qwen_vl_utils import process_vision_info
from transformers import AutoConfig, AutoProcessor, TextStreamer

from QEfficient import QEFFAutoModelForImageTextToText

# For AWQ model update pytorch version to 2.8.*
model_id = "Qwen/Qwen2.5-VL-3B-Instruct"
config = AutoConfig.from_pretrained(model_id)
config.text_config.num_hidden_layers = 2

qeff_model = QEFFAutoModelForImageTextToText.from_pretrained(
    model_id, attn_implementation="eager", kv_offload=True, config=config
)
tokenizer = transformers.AutoTokenizer.from_pretrained(model_id)
processor = AutoProcessor.from_pretrained(model_id)

# use skip_vision=True, if want to run only text
skip_vision = False

if skip_vision:  # Only Text
    batch_size = 1
    qeff_model.compile(
        batch_size=batch_size,
        prefill_seq_len=128,
        ctx_len=4096,
        num_cores=16,
        num_devices=8,
        height=354,
        width=536,
        mxfp6_matmul=False,
        aic_enable_depth_first=True,
        skip_vision=True,
        mos=1,
    )

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "Tell me about yourself."},
            ],
        },
    ]

    messages = [messages] * batch_size

    inputs = processor.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
    )

    import ipdb

    ipdb.set_trace()

    inputs = qeff_model.model.prepare_inputs_for_generation(inputs=inputs, prefill_seq_len=128, batch_size=batch_size)

    streamer = TextStreamer(tokenizer)
    output = qeff_model.generate(inputs=inputs, generation_len=100)
    print(output.generated_ids)
    print(tokenizer.batch_decode(output.generated_ids))
    print(output)

else:  # Vision + Text
    batch_size = 1
    ctx_len = 14336
    widths = [360, 320, 360, 454, 536, 640, 720, 910, 720, 1280, 1920]
    heights = [120, 180, 240, 256, 354, 360, 480, 512, 576, 720, 1080]
    num_frames = [177, 139, 78, 64, 37, 30, 20, 16, 16, 7, 7]

    qeff_model.compile(
        batch_size=batch_size,
        prefill_seq_len=128,
        ctx_len=ctx_len,
        num_cores=16,
        num_devices=2,
        height=heights,
        width=widths,
        num_frames=max(num_frames),
        mm_processor_kwargs={
            "min_pixels": 4 * 28 * 28,
            "max_pixels": 16384 * 28 * 28,
        },
        mxfp6_matmul=True,
        mxint8_kv_cache=True,
        aic_enable_depth_first=True,
        mos=1,
    )

    image_url = "https://picsum.photos/id/237/536/354"
    image = Image.open(requests.get(image_url, stream=True).raw)
    image = image.resize((360, 120))  # Resize to any dimension (width, height) present in specializations
    messages_1 = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": "Describe this image."},
            ],
        },
    ]
    messages_2 = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": "Describe about the color of the dog."},
            ],
        },
    ]
    messages = [messages_1] * batch_size
    texts = [processor.apply_chat_template(msg, tokenize=False, add_generation_prompt=True) for msg in messages]

    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=texts,
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )

    inputs = qeff_model.model.prepare_inputs_for_generation(inputs=inputs, prefill_seq_len=128, batch_size=batch_size)

    streamer = TextStreamer(tokenizer)
    output = qeff_model.generate(inputs=inputs, generation_len=100)
    print(output.generated_ids)
    print(tokenizer.batch_decode(output.generated_ids))
    print(output)

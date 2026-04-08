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
# config.text_config.num_hidden_layers = 2

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

    inputs = qeff_model.model.prepare_inputs_for_generation(inputs=inputs, prefill_seq_len=128, batch_size=batch_size)

    streamer = TextStreamer(tokenizer)
    output = qeff_model.generate(inputs=inputs, generation_len=100)
    print(output.generated_ids)
    print(tokenizer.batch_decode(output.generated_ids))
    print(output)

else:  # Vision + Text
    batch_size = 1
    ctx_len = 8192

    resolutions = [
        {"width": 360, "height": 120, "num_frames": 3},
        {"width": 320, "height": 180, "num_frames": 2},
        {"width": 360, "height": 240, "num_frames": 1},
        {"width": 454, "height": 256, "num_frames": 1},
    ]

    widths = [s["width"] for s in resolutions]
    heights = [s["height"] for s in resolutions]
    num_frames = [s["num_frames"] for s in resolutions]
    # vision_size = 4096  # vision_size is the maximum visual-token budget that limits the ViT

    # (Vision Transformer) embeddings passed to the language decoder, together
    # with the text prompt. Increasing this value preserves more visual detail,
    # but consumes more of the model’s context length.
    # This argument is **optional**; if not provided, vision_size is automatically
    # derived from the input image resolutions to support the largest resolution.

    qeff_model.compile(
        batch_size=batch_size,
        prefill_seq_len=128,
        ctx_len=ctx_len,
        num_cores=16,
        num_devices=2,
        height=heights,
        width=widths,
        num_frames=num_frames,
        mm_processor_kwargs={
            "min_pixels": 4 * 28 * 28,
            "max_pixels": 16384 * 28 * 28,
        },
        # vision_size=vision_size,
        mxfp6_matmul=True,
        mxint8_kv_cache=True,
        aic_enable_depth_first=True,
        mos=1,
        skip_lang=True,
    )

    image_url = "https://picsum.photos/id/237/536/354"
    image = Image.open(requests.get(image_url, stream=True).raw)
    image = image.resize((360, 120))  # Resize to any dimension (width, height) present in specializations
    frames = 3

    content = [{"type": "image", "image": image} for _ in range(frames)] + [
        {"type": "text", "text": "Describe the image"}
    ]

    messages = [{"role": "user", "content": content}]

    messages = [messages] * batch_size
    texts = [processor.apply_chat_template(msg, tokenize=False, add_generation_prompt=True) for msg in messages]

    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=texts,
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )

    image_grid_thw = inputs.get("image_grid_thw")
    inputs = qeff_model.model.prepare_inputs_for_generation(inputs=inputs, prefill_seq_len=128, batch_size=batch_size)

    streamer = TextStreamer(tokenizer)
    output = qeff_model.generate(
        inputs=inputs, tokenizer=tokenizer, generation_len=100, multi_specs=True, num_frames=frames
    )
    print(output.generated_ids)
    print(output.generated_texts)
    print(output)

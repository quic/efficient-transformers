# -----------------------------------------------------------------------------
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------

import requests
import transformers
from PIL import Image
from qwen_omni_utils import process_mm_info

# from qwen_vl_utils import process_vision_info
from transformers import AutoConfig, AutoProcessor, TextStreamer

from QEfficient import QEFFAutoModelForMultimodalLM

model_id = "Qwen/Qwen3-Omni-30B-A3B-Instruct"
config = AutoConfig.from_pretrained(model_id)

# config.talker_config.text_config.num_hidden_layers = 2
# config.thinker_config.text_config.num_hidden_layers = 2
# config.thinker_config.vision_config.deepstack_visual_indexes = [8]
# config.thinker_config.vision_config.depth = 9

config.enable_audio_output = False
config.torch_dtype = "float32"
qeff_model = QEFFAutoModelForMultimodalLM.from_pretrained(
    model_id, attn_implementation="eager", kv_offload=True, config=config
)
# import ipdb; ipdb.set_trace()
tokenizer = transformers.AutoTokenizer.from_pretrained(model_id)
processor = AutoProcessor.from_pretrained(model_id)
### use skip_vision=Ture, if want to run only text, or false ###
skip_vision = False

if skip_vision:
    ## Only Text ##
    ## Set Batch_Size ##
    batch_size = 1
    qeff_model.compile(
        batch_size=batch_size,
        prefill_seq_len=128,
        ctx_len=4096,
        num_cores=16,
        num_devices=4,
        height=354,
        width=536,
        mxfp6_matmul=True,
        aic_enable_depth_first=True,
        skip_vision=True,
        mos=1,
        # mdts_mos=1,
        # use_onnx_subfunctions=True,
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

else:
    batch_size = 1
    ## Vision + Text ##
    qeff_model.compile(
        batch_size=batch_size,
        prefill_seq_len=128,
        ctx_len=4096,
        num_cores=16,
        num_devices=4,
        height=354,
        width=536,
        mxfp6_matmul=True,
        mxint8_kv_cache=True,
        aic_enable_depth_first=True,
        mos=1,
        # mdts_mos=1,
        # use_onnx_subfunctions=True,
    )

    ### IMAGE + TEXT ###
    image_url = "https://picsum.photos/id/237/536/354"

    image = Image.open(requests.get(image_url, stream=True).raw)

    messages_1 = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": "Descibe all the colors seen in the image."},
            ],
        },
    ]

    messages = [messages_1] * batch_size

    conversation = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "audio", "audio": "/home/mohisoni/omni/cough.wav"},
                {"type": "text", "text": "What can you see and hear? Answer in one short sentence."},
            ],
        },
    ]

    # Set whether to use audio in video
    USE_AUDIO_IN_VIDEO = False

    # Preparation for inference
    text = processor.apply_chat_template(conversation, add_generation_prompt=True, tokenize=False)

    audios, images, videos = process_mm_info(conversation, use_audio_in_video=USE_AUDIO_IN_VIDEO)
    inputs = processor(
        text=text,
        audio=audios,
        images=images,
        videos=videos,
        return_tensors="pt",
        padding=True,
        use_audio_in_video=USE_AUDIO_IN_VIDEO,
    )
    # inputs = inputs.to(qeff_model.device).to(qeff_model.dtype)
    inputs = inputs.to("cpu")

    inputs = qeff_model.model.prepare_inputs_for_generation(inputs=inputs, prefill_seq_len=128, batch_size=batch_size)
    streamer = TextStreamer(tokenizer)
    output = qeff_model.generate(inputs=inputs, generation_len=100)
    print(output.generated_ids)
    print(tokenizer.batch_decode(output.generated_ids))
    print(output)

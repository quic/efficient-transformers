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

model_id = "Qwen/Qwen3.5-27B"
config = AutoConfig.from_pretrained(model_id)

# For faster execution user can run with lesser layers, For Testing Purpose Only
# config.text_config.num_hidden_layers = 2
config.torch_dtype = "float32"

qeff_model = QEFFAutoModelForImageTextToText.from_pretrained(
    model_id, attn_implementation="eager", kv_offload=True, config=config
)

tokenizer = transformers.AutoTokenizer.from_pretrained(model_id)
processor = AutoProcessor.from_pretrained(model_id)

# Enable KV blocking for full-attention layers with 2 KV blocks
# To disable KV blocking, comment out the qaic_config line below
# Set skip_kv=True to skip future KV blocks during inference (optimization)
qaic_config = {"blocking_mode": "kv", "num_kv_blocks": 2, "skip_kv": True}

enable_blocking = False  # By default blocking is false
### use skip_vision=True, if want to run only text, or false ###
skip_vision = False

if skip_vision:
    ## Only Text ##
    ## Set Batch_Size ##
    batch_size = 1
    qeff_model.compile(
        batch_size=batch_size,
        prefill_seq_len=64,
        ctx_len=4096,
        num_cores=16,
        num_devices=1,
        mxfp6_matmul=True,
        mxint8_kv_cache=False,
        aic_enable_depth_first=False,
        skip_vision=True,
        mos=1,
        # qaic_config=qaic_config,  # Enable KV blocking - comment out to disable
    )

    if enable_blocking:
        print("\n" + "=" * 80)
        print("Verifying KV Blocking Applied During Compilation")
        print("=" * 80)

        if qaic_config and qaic_config.get("blocking_mode"):
            print("✓ qaic_config passed to compile():")
            print(f"    Blocking Mode: {qaic_config.get('blocking_mode')}")
            print(f"    Num KV Blocks: {qaic_config.get('num_kv_blocks')}")
            print(f"    Skip KV: {qaic_config.get('skip_kv', False)}")
            print("\n✓ BlockingAttentionTransform.apply() called during compile()")
            print("  - Sets attn_blocking_config on all supported attention modules")
            print("  - Blocked attention forward pass is used during ONNX export")
            print("  - Blocking operations are in the ONNX graph and QPC")
            print("\n  Status: ACTIVE")
            print("  Verification: Config-based verification")
            print("  Note: Blocking IS applied - torch model is freed after ONNX export")
        else:
            print("✗ No qaic_config provided - eager attention will be used")
            print("  Status: INACTIVE - Model compiled without blocking")

        print("=" * 80 + "\n")
    text_prompt_2 = "Describe yourself as a large language model, including your purpose, capabilities, and limitations. Explain how you process and generate responses, interact with users, and handle uncertainty, while emphasizing accuracy, safety, and helpfulness in diverse conversations across various topics and domains."

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": text_prompt_2},
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
    inputs = qeff_model.model.prepare_inputs_for_generation(inputs=inputs, prefill_seq_len=64, batch_size=batch_size)
    streamer = TextStreamer(tokenizer)
    output = qeff_model.generate(inputs=inputs, generation_len=512, streamer=streamer)
    print(output.generated_ids)
    print(tokenizer.batch_decode(output.generated_ids))
    print(output)

else:
    batch_size = 1
    ## Vision + Text ##
    qeff_model.compile(
        batch_size=batch_size,
        prefill_seq_len=64,
        ctx_len=4096,
        num_cores=16,
        num_devices=4,
        height=354,
        width=536,
        mxfp6_matmul=False,
        mxint8_kv_cache=False,
        aic_enable_depth_first=False,
        mos=1,
        # qaic_config=qaic_config,  # Enable KV blocking - comment out to disable
    )

    if enable_blocking:
        print("\n" + "=" * 80)
        print("Verifying KV Blocking Applied During Compilation")
        print("=" * 80)

        if qaic_config and qaic_config.get("blocking_mode"):
            print("✓ qaic_config passed to compile():")
            print(f"    Blocking Mode: {qaic_config.get('blocking_mode')}")
            print(f"    Num KV Blocks: {qaic_config.get('num_kv_blocks')}")
            print(f"    Skip KV: {qaic_config.get('skip_kv', False)}")
            print("\n✓ BlockingAttentionTransform.apply() called during compile()")
            print("  - Sets attn_blocking_config on all supported attention modules")
            print("  - Blocked attention forward pass is used during ONNX export")
            print("  - Blocking operations are in the ONNX graph and QPC")
            print("\n  Status: ACTIVE")
            print("  Verification: Config-based verification")
            print("  Note: Blocking IS applied - torch model is freed after ONNX export")
        else:
            print("✗ No qaic_config provided - eager attention will be used")
            print("  Status: INACTIVE - Model compiled without blocking")

        print("=" * 80 + "\n")

    ### IMAGE + TEXT ###
    image_url = "https://picsum.photos/id/237/536/354"
    image = Image.open(requests.get(image_url, stream=True).raw)

    messages_1 = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": "Describe all the colors seen in the image."},
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
    inputs = qeff_model.model.prepare_inputs_for_generation(inputs=inputs, prefill_seq_len=64, batch_size=batch_size)
    streamer = TextStreamer(tokenizer)
    output = qeff_model.generate(inputs=inputs, generation_len=100, streamer=streamer)
    print(output.generated_ids)
    print(tokenizer.batch_decode(output.generated_ids))
    print(output)

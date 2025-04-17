# -----------------------------------------------------------------------------
#
# Copyright (c) 2025 Qualcomm Innovation Center, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------

from transformers import AutoProcessor, TextStreamer

from QEfficient import QEFFAutoModelForImageTextToText

# Add HuggingFace Token to access the model
HF_TOKEN = ""


def run_model(
    model_name,
    token,
    query,
    image_url,
    kv_offload=False,
    prefill_seq_len=5500,
    ctx_len=6000,
    generation_len=128,
    img_size=384,
    num_cores=16,
    num_devices=1,
):
    ## STEP - 1 Load the Processor and Model

    processor = AutoProcessor.from_pretrained(model_name, token=token)

    # `kv_offload` is used to compile the model in a Single QPC or 2 QPCs.
    # The Dual QPC approach splits the model to perform Image Encoding and Output generation in 2 different QPCs.
    # The outputs of the Vision Encoder are then passed to the Language model via host in this case.

    model = QEFFAutoModelForImageTextToText.from_pretrained(model_name, token=token, kv_offload=kv_offload)
    ## STEP - 2 Export & Compile the Model

    model.compile(
        prefill_seq_len=prefill_seq_len,
        ctx_len=ctx_len,
        num_patches=10,
        image_size_height=1109,
        image_size_width=1610,
        img_size=img_size,
        num_cores=num_cores,
        num_devices=num_devices,
        mxfp6_matmul=False,
    )

    ## STEP - 3 Load and process the inputs for Inference

    img_path = "https://huggingface.co/ibm-granite/granite-vision-3.2-2b/resolve/main/example.png"
    conversation = [
        {
            "role": "user",
            "content": [
                {"type": "image", "url": img_path},
                {"type": "text", "text": query},
            ],
        },
    ]
    inputs = processor.apply_chat_template(
        conversation, add_generation_prompt=True, tokenize=True, return_dict=True, return_tensors="pt"
    )

    ## STEP - 4 Run Inference on the compiled model

    streamer = TextStreamer(processor.tokenizer)
    model.generate(inputs=inputs, streamer=streamer, generation_len=128)

    # # QEff pytorch

    # # print(output)
    # # print(processor.tokenizer.batch_decode(output))
    # # print(output)
    # # breakpoint()

    # # AIC Output

    # print(output.generated_ids)
    # print(processor.tokenizer.batch_decode(output.generated_ids))
    # print(output)


if __name__ == "__main__":
    # Model name and Input parameters
    model_name = "ibm-granite/granite-vision-3.2-2b"
    query = "What is the highest scoring model on ChartQA and what is its score?"
    image_url = "https://huggingface.co/ibm-granite/granite-vision-3.2-2b/resolve/main/example.png"
    # Compilation parameters for the model
    kv_offload = True
    prefill_seq_len = 5500
    ctx_len = 6000
    generation_len = 10
    img_size = 384
    num_cores = 16
    num_devices = 4

    run_model(
        model_name=model_name,
        token=HF_TOKEN,
        query=query,
        kv_offload=kv_offload,
        image_url=image_url,
        prefill_seq_len=prefill_seq_len,
        ctx_len=ctx_len,
        generation_len=generation_len,
        img_size=img_size,
        num_cores=num_cores,
        num_devices=num_devices,
    )


"""
Expected Response:

The highest scoring model on ChartQA is Granite Vision with a score of 0.87.

"""

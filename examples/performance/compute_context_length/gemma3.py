# -----------------------------------------------------------------------------
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------

import torch
import transformers
from transformers import AutoConfig, AutoProcessor

from QEfficient import QEFFAutoModelForImageTextToText

# Change model_id to "google/gemma-3-27b-it" for 27B model
model_id = "google/gemma-3-4b-it"
config = AutoConfig.from_pretrained(model_id)
# For Testing Purpose Only
# config.text_config.num_hidden_layers = 1
# config.vision_config.num_hidden_layers = 2
tokenizer = transformers.AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
processor = AutoProcessor.from_pretrained(model_id)

## Activate Compute-Context-Length (CCL) feature by setting ccl_enabled=True when loading the model with from_pretrained().
## Use the optional comp_ctx_lengths argument to provide two lists of context lengths for the prefilling and decoding processes. If comp_ctx_lengths=None, the model will run with its default context length.
##   - The first list, comp_ctx_lengths_prefill, defines the compute-context-length values for the prefilling process.
##           -- The process starts with the first value in the list and gradually increases the context length based on the position_id of the current prompt chunk.
##   - The second list, comp_ctx_lengths_decode, defines the compute-context-length values for the decoding process.
##           -- During decoding, the model selects an appropriate context length from the list based on the input prompt length and cache index.
##           -- It starts from the correct value in the list and increases the context length dynamically when the cache index exceeds the current threshold.

ctx_len = 8192
comp_ctx_lengths_prefill = [3072]
comp_ctx_lengths_decode = [4096, ctx_len]

# pass HF_TOKEN if gated model
# For running the model in single QPC approach use kv_offload=False. For Dual QPC approach use kv_offload=True ###
qeff_model = QEFFAutoModelForImageTextToText.from_pretrained(
    model_id,
    config=config,
    attn_implementation="eager",
    kv_offload=True,
    qaic_config={
        "ccl_enabled": True,
    },
)

### use skip_vision=True, if want to run only text, or false ###
skip_vision = False

if skip_vision:
    ## Only Text ##
    qeff_model.compile(
        prefill_seq_len=128,
        ctx_len=ctx_len,
        img_size=896,
        num_cores=16,
        num_devices=4,
        mxfp6_matmul=False,
        mxint8_kv_cache=False,
        aic_enable_depth_first=True,
        skip_vision=True,
        mos=1,
        node_precision_info="examples/performance/compute_context_length/fp32_nodes_gemma3_4b.yaml",
        comp_ctx_lengths_prefill=comp_ctx_lengths_prefill,
        comp_ctx_lengths_decode=comp_ctx_lengths_decode,
    )

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "Describe the transformers architecture in LLMs."},
            ],
        },
    ]

    inputs = processor.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
    )

    output = qeff_model.generate(inputs=inputs, generation_len=100)
    print(tokenizer.batch_decode(output.generated_ids))
    print(output)

else:
    ## Vision + Text ##
    qeff_model.compile(
        prefill_seq_len=128,
        ctx_len=ctx_len,
        img_size=896,
        num_cores=16,
        num_devices=4,
        mxfp6_matmul=False,
        mxint8_kv_cache=False,
        aic_enable_depth_first=True,
        mos=1,
        node_precision_info="examples/performance/compute_context_length/fp32_nodes_gemma3_4b.yaml",
        comp_ctx_lengths_prefill=comp_ctx_lengths_prefill,
        comp_ctx_lengths_decode=comp_ctx_lengths_decode,
    )

    ### IMAGE + TEXT ###
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

    inputs = processor.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
    )
    inputs["pixel_values"] = inputs["pixel_values"].to(torch.float32)
    output = qeff_model.generate(inputs=inputs, generation_len=100)
    print(tokenizer.batch_decode(output.generated_ids))
    print(output)

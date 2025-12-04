# -----------------------------------------------------------------------------
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
#
# ----------------------------------------------------------------------------

from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

from QEfficient import QEFFAutoModelForCausalLM
from QEfficient.utils.test_utils import InternProcessor

model_id = "OpenGVLab/InternVL2_5-1B"
config = AutoConfig.from_pretrained(model_id, trust_remote_code=True)
# For Testing Purpose Only
config.llm_config.num_hidden_layers = 2
config.vision_config.num_hidden_layers = 2

# The original Intern-VL model, despite being multimodal, is loaded using `AutoModelForCausalLM` in Huggingface.
# To maintain compatibility, we load this model using `QEFFAutoModelForCausalLM`.
model_hf = AutoModelForCausalLM.from_pretrained(
    model_id,
    low_cpu_mem_usage=False,
    trust_remote_code=True,
    config=config,
)

tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True, use_fast=False)
processor = InternProcessor(model_hf, tokenizer)


continuous_batching = True
if continuous_batching:
    qeff_model = QEFFAutoModelForCausalLM.from_pretrained(
        model_id,
        attn_implementation="eager",
        kv_offload=True,
        config=config,
        continuous_batching=True,
        trust_remote_code=True,
    )

    qeff_model.compile(
        num_patches=13,  # Set num_patches according to image_height and image_width, default is 13 (747 x 1000)
        prefill_seq_len=128,
        ctx_len=4096,
        num_cores=16,
        num_devices=4,
        batch_size=1,
        full_batch_size=4,
        mxfp6_matmul=True,
        mxint8_kv_cache=True,
        aic_enable_depth_first=True,
        mos=1,
    )
else:
    qeff_model = QEFFAutoModelForCausalLM.from_pretrained(
        model_id, attn_implementation="eager", kv_offload=True, config=config, trust_remote_code=True
    )

    qeff_model.compile(
        num_patches=13,
        prefill_seq_len=128,
        ctx_len=4096,
        num_cores=16,
        num_devices=4,
        batch_size=1,
        mxfp6_matmul=True,
        mxint8_kv_cache=True,
        aic_enable_depth_first=True,
    )

image_urls = [
    "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/datasets/cat_style_layout.png",
    "https://huggingface.co/datasets/huggingface/documentation-images/resolve/0052a70beed5bf71b92610a43a52df6d286cd5f3/diffusers/rabbit.jpg",
    "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/datasets/cat_style_layout.png",
    "https://huggingface.co/datasets/huggingface/documentation-images/resolve/0052a70beed5bf71b92610a43a52df6d286cd5f3/diffusers/rabbit.jpg",
]

prompts = [
    "Can you describe the image in detail?",
    "What are the objects in the image?",
    "What is the main subject of the image?",
    "What colors are predominant in the image?",
]

exec_info = qeff_model.generate(
    tokenizer=tokenizer,
    prompts=prompts,
    processor=processor,
    images=image_urls,
    device_ids=[0, 1, 2, 3],
    generation_len=10,
    image_height=747,
    image_width=1000,
)

print("Generated texts:", exec_info.generated_texts)
print("Generated IDs:", exec_info.generated_ids)
print(exec_info)

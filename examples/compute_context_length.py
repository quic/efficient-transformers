# -----------------------------------------------------------------------------
#
# Copyright (c) 2025 Qualcomm Innovation Center, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------

## In this example, you can run a model for static and continuous batching with different Compute-Context-Length (CCL) inputs. ##

from transformers import AutoTokenizer

from QEfficient import QEFFAutoModelForCausalLM

## Using optional variable comp_ctx_lengths variable you can pass a list of context lengths. It will run the model with default context length if comp_ctx_lengths=None. ##
##       - The first comp_ctx_lengths_prefill list shows the compute-ctx-length list for prefilling process. ##
##       - The second comp_ctx_lengths_decode list will be used for decoding. During the decoding process, based on the position_id or cache index it will work with the specific compute-context-length in the list. It will start from a proper compute-context-length in the list based on input prompt length and will gradually increase the compute-context-length if the cache index passes the current compute-context-length. ##

ctx_len = 1024
comp_ctx_lengths_prefill = [256]
comp_ctx_lengths_decode = [512, ctx_len]

# model_name = "google/gemma-7b"
# model_name = "google/gemma-2-2b"
# model_name = "ibm-granite/granite-3.1-8b-instruct"
# model_name = "Snowflake/Llama-3.1-SwiftKV-8B-Instruct"
# model_name = "mistralai/Mistral-7B-v0.1"
# model_name = "microsoft/phi-1_5"
# model_name = "microsoft/Phi-3-mini-4k-instruct"
# model_name = "Qwen/Qwen2.5-7B-Instruct"
model_name = "meta-llama/Llama-3.2-1B"
# model_name = "Qwen/Qwen3-1.7B"
# model_name = "allenai/OLMo-2-0425-1B"
# model_name = "ibm-granite/granite-3.3-2b-base"
model = QEFFAutoModelForCausalLM.from_pretrained(
    model_name,
    continuous_batching=False,
    comp_ctx_lengths_prefill=comp_ctx_lengths_prefill,
    comp_ctx_lengths_decode=comp_ctx_lengths_decode,
    ctx_len=ctx_len,
)

# model compilation for either continuous or static batching. For continuous batching full_batch_size is needed.
model.compile(
    prefill_seq_len=128,
    ctx_len=ctx_len,
    num_cores=16,
    num_devices=1,
    batch_size=1,
    mxint8_kv_cache=True,
    mxfp6_matmul=True,
)

# Create tokenizer and run model.generate and passes the input prompts to it.
tokenizer = AutoTokenizer.from_pretrained(model_name)
model.generate(
    prompts=[
        "My name is ",
    ],
    tokenizer=tokenizer,
    generation_len=128,
)

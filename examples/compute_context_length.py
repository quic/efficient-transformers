# -----------------------------------------------------------------------------
#
# Copyright (c) 2025 Qualcomm Innovation Center, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------

## In this example, you can run a model for static and continuous batching with different Compute-Context-Length (CCL) inputs. ##

from transformers import AutoTokenizer

from QEfficient import QEFFAutoModelForCausalLM

## Activate Compute-Context-Length (CCL) feature by setting ccl_enabled=True when loading the model with from_pretrained().
## Use the optional comp_ctx_lengths argument to provide two lists of context lengths for the prefilling and decoding processes. If comp_ctx_lengths=None, the model will run with its default context length.
##   - The first list, comp_ctx_lengths_prefill, defines the compute-context-length values for the prefilling process.
##           -- The process starts with the first value in the list and gradually increases the context length based on the position_id of the current prompt chunk.
##   - The second list, comp_ctx_lengths_decode, defines the compute-context-length values for the decoding process.
##           -- During decoding, the model selects an appropriate context length from the list based on the input prompt length and cache index.
##           -- It starts from the correct value in the list and increases the context length dynamically when the cache index exceeds the current threshold.

ctx_len = 1024
comp_ctx_lengths_prefill = [256, 500]  # None #
comp_ctx_lengths_decode = [512, ctx_len]  # None #

model_name = "meta-llama/Llama-3.2-1B"

model = QEFFAutoModelForCausalLM.from_pretrained(
    model_name,
    continuous_batching=False,
    ccl_enabled=True,
)

# model compilation for either continuous or static batching. For continuous batching full_batch_size is needed.
model.compile(
    prefill_seq_len=128,
    ctx_len=ctx_len,
    num_cores=16,
    num_devices=1,
    mxint8_kv_cache=True,
    mxfp6_matmul=True,
    batch_size=1,
    comp_ctx_lengths_prefill=comp_ctx_lengths_prefill,
    comp_ctx_lengths_decode=comp_ctx_lengths_decode,
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

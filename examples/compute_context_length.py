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
##       - The first number in this list is the context length that will be used during prefilling. ##
##       - During the decoding process, based on the position_id or cache index it will work with the specific compute-context-length in the list. It will start from a proper compute-context-length in the list based on input prompt length and will gradually increase the compute-context-length if the cache index passes the current compute-context-length. ##
comp_ctx_lengths = [256, 512, 1024]  # None

model_name = "meta-llama/Llama-3.2-1B-Instruct"
model = QEFFAutoModelForCausalLM.from_pretrained(
    model_name, continuous_batching=True, comp_ctx_lengths=comp_ctx_lengths
)
# model = QEFFAutoModelForCausalLM.from_pretrained(model_name, comp_ctx_lengths=comp_ctx_lengths)

# model compilation for either continuous or static batching. For continuous batching full_batch_size is needed.
model.compile(
    prefill_seq_len=128,
    ctx_len=1024,
    num_cores=16,
    num_devices=1,
    full_batch_size=1,
    mxfp6_matmul=True,
    mxint8_kv_cache=True,
)
# model.compile(prefill_seq_len=128, ctx_len=1024, num_cores=16, num_devices=1,batch_size=4,mxfp6_matmul=True,mxint8_kv_cache=True)

# Create tokenizer and run model.generate and passes the input prompts to it. It also receives comp_ctx_lengths list which will be used during the decoding process to apply the best and most efficient compute context length.
tokenizer = AutoTokenizer.from_pretrained(model_name)
model.generate(
    prompts=[
        "What are some healthy foods to include in a balanced diet?",
        "What is a nutritious meal that can keep you energized throughout the day?",
        "What are some fun and relaxing activities to do over the weekend?",
        "What's your favorite hobby?",
    ],
    tokenizer=tokenizer,
)

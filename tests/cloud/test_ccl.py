# -----------------------------------------------------------------------------
#
# Copyright (c) 2025 Qualcomm Innovation Center, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------

## In this example, you can run a model for static and continuous batching with different Compute-Context-Length (CCL) inputs. ##

from transformers import AutoTokenizer

from QEfficient import QEFFAutoModelForCausalLM

# model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
model_name = "meta-llama/Llama-3.1-8B-Instruct"
# You can work in both static and continuous batching mode. For continuous batching you should set continuous_batching=True. The default value for this parameter is False.
# model = QEFFAutoModelForCausalLM.from_pretrained(model_name, continuous_batching=True)
model = QEFFAutoModelForCausalLM.from_pretrained(model_name)

## Using optional variable comp_ctx_lengths variable you can pass a list of context lengths. It will run the original model if comp_ctx_lengths=None. ##
##       - The first number in this list is the context length that will be used during prefilling. ##
##       - During the decoding process, based on the position_id or cache index it will work with the specific compute-context-length in the list. It will start from a proper compute-context-length in the list based on input prompt length and will gradually increase the compute-context-length if the cache index passes the current compute-context-length. ##

comp_ctx_lengths = None #[512, 1024, 2048, 4096, 8192]

# model compilation for either continuous or static batching. For continuous batching full_batch_size is needed.
# model.compile(
#     prefill_seq_len=128,
#     ctx_len=2048,
#     comp_ctx_lengths=comp_ctx_lengths,
#     num_cores=16,
#     num_devices=4,
#     full_batch_size=4,
# )
model.compile(prefill_seq_len=128, ctx_len=8192, comp_ctx_lengths=comp_ctx_lengths, num_cores=16, num_devices=4,batch_size=4)

# Create tokenizer and run model.generate and passs the input prompts to it. It also recieves comp_ctx_lengths list which will be used during the decoding process to apply the best and most efficient compute context length.
tokenizer = AutoTokenizer.from_pretrained(model_name)
model.generate(
    prompts=[
        "What are some healthy foods to include in a balanced diet?",
    ],
    tokenizer=tokenizer,
    comp_ctx_lengths=comp_ctx_lengths,
    generation_len=128,
    device_ids=[16,17,18,19],
)

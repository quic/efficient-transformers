# -----------------------------------------------------------------------------
#
# Copyright (c) 2025 Qualcomm Innovation Center, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------

## In this example, you can run a model for static and continuous batching with different Compute-Context-Length (CCL) inputs. ##

from QEfficient import QEFFAutoModelForCausalLM
from transformers import AutoTokenizer

model_name = "meta-llama/Llama-3.1-8B-Instruct"
# You can work in both static and continuous batching mode. For continuous batching you should set continuous_batching=True. The default value for this parameter is False. 
model = QEFFAutoModelForCausalLM.from_pretrained(model_name, num_hidden_layers=2, continuous_batching=True)
# model = QEFFAutoModelForCausalLM.from_pretrained(model_name, num_hidden_layers=2)

## Using CCL variable you can pass a list of context lengths. ##
##       - The first number in this list will be context length which will be used during prefilling. ##
##       - During the decoding process, based on the position_id or cache index it will work with the specific CCL in the list. It will start from a proper CCL in the list based on input prompt length and will gradually increase the CCL if the cache index passes the current CCL. ##

CCL = [512, 1024, 2048, 4096, 8192, 16384, 32768]

#model compilation for either continuous or static batching. For continuous batching full_batch_size is needed.
model.compile(prefill_seq_len=128, ctx_len=32768, CCL=CCL, num_cores=16, num_devices=1, full_batch_size=4)
# model.compile(prefill_seq_len=128, ctx_len=32768, CCL=CCL, num_cores=16, num_devices=1,batch_size=4)

#Create tokenizer and run model.generate and passs the input prompts to it. It also recieves CCL list which will be used during the decoding process to apply the best and most efficient compute context length.
tokenizer = AutoTokenizer.from_pretrained(model_name)
model.generate(
    prompts=["What are some healthy foods to include in a balanced diet?", "What is a nutritious meal that can keep you energized throughout the day?", "What are some fun and relaxing activities to do over the weekend?", "What's your favorite hobby?"], tokenizer=tokenizer, CCL=CCL, generation_len=128
)

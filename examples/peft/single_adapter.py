# -----------------------------------------------------------------------------
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------

## This example demonstrates single adapter usage with sequential adapter switching ##

from transformers import AutoTokenizer, TextStreamer

from QEfficient import QEffAutoPeftModelForCausalLM

base_model_name = "mistralai/Mistral-7B-v0.1"
tokenizer = AutoTokenizer.from_pretrained(base_model_name)
streamer = TextStreamer(tokenizer)
prefill_seq_len = 32
ctx_len = 1024
generation_len = 1024


## STEP 1 -- init base model
qeff_model = QEffAutoPeftModelForCausalLM.from_pretrained("predibase/magicoder", "magicoder")

## STEP 2 -- export & compile qeff model
qeff_model.compile(prefill_seq_len=prefill_seq_len, ctx_len=ctx_len)

## STEP 3 -- run inference with different adapters

# Magicoder adapter - code generation
qeff_model.set_adapter("magicoder")
inputs = tokenizer("def fibonacci", return_tensors="pt")
qeff_model.generate(**inputs, streamer=streamer, max_new_tokens=generation_len)

## STEP 3.1 -- load and use TLDR headline generator adapter
qeff_model.load_adapter("predibase/tldr_headline_gen", "tldr_headline_gen")
qeff_model.set_adapter("tldr_headline_gen")
inputs = tokenizer(
    """Summarize this passage in one sentence or less: Jeffrey Berns, CEO of Blockchains LLC, wants the Nevada government to allow companies like \
his to form local governments on land they own, granting them power over everything from \
schools to law enforcement. Berns envisions a city based on digital currencies and \
blockchain storage. His company is proposing to build a 15,000 home town 12 miles east of \
Reno. Nevada Lawmakers have responded with intrigue and skepticism. The proposed \
legislation has yet to be formally filed or discussed in public hearings.

Summary: """,
    return_tensors="pt",
)
qeff_model.generate(**inputs, streamer=streamer, max_new_tokens=1024)

## STEP 3.2 -- load and use GSM8K adapter for math problems
qeff_model.load_adapter("predibase/gsm8k", "gsm8k")
qeff_model.set_adapter("gsm8k")
inputs = tokenizer(
    "James decides to run 3 sprints 3 times a week. He runs 60 meters each sprint. \
How many total meters does he run a week?",
    return_tensors="pt",
)
qeff_model.generate(**inputs, streamer=streamer, max_new_tokens=1024)

## STEP 3.3 -- load and use AGNews adapter for news classification
qeff_model.load_adapter("predibase/agnews_explained", "agnews_explained")
qeff_model.set_adapter("agnews_explained")
inputs = tokenizer(
    """Below is a news article. Please classify it under one of the following \
classes (World, Business, Sports, Sci/Tech) and provide a reasonable coherent explanation for \
why the article is classified as such. Please format your response as a JSON payload.

### Article: US poverty rate climbs, along with number lacking health coverage (AFP) AFP - The \
number of Americans living in poverty or without health insurance grew last year, a government \
survey showed, adding potential dynamite in the battle for the White House.

### JSON Response

""",
    return_tensors="pt",
)
qeff_model.generate(**inputs, streamer=streamer, max_new_tokens=1024)

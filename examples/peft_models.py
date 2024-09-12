# -----------------------------------------------------------------------------
#
# Copyright (c) 2024 Qualcomm Innovation Center, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------

from transformers import AutoTokenizer, TextStreamer

from QEfficient import QEffAutoPeftModelForCausalLM

base_model_name = "mistralai/Mistral-7B-v0.1"
tokenizer = AutoTokenizer.from_pretrained(base_model_name)
streamer = TextStreamer(tokenizer)

m = QEffAutoPeftModelForCausalLM.from_pretrained("predibase/magicoder", "magicoder")
m.export()
m.compile(prefill_seq_len=32, ctx_len=1024)

# Magicoder adapter
m.set_adapter("magicoder")
inputs = tokenizer("def fibonacci", return_tensors="pt")
m.generate(**inputs, streamer=streamer, max_new_tokens=1024)

# TLDR, summary generator
m.load_adapter("predibase/tldr_headline_gen", "tldr_headline_gen")
m.set_adapter("tldr_headline_gen")
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
m.generate(**inputs, streamer=streamer, max_new_tokens=1024)

# Math problems
m.load_adapter("predibase/gsm8k", "gsm8k")
m.set_adapter("gsm8k")
inputs = tokenizer(
    "James decides to run 3 sprints 3 times a week. He runs 60 meters each sprint. \
How many total meters does he run a week?",
    return_tensors="pt",
)
m.generate(**inputs, streamer=streamer, max_new_tokens=1024)

# News explanation
m.load_adapter("predibase/agnews_explained", "agnews_explained")
m.set_adapter("agnews_explained")
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
m.generate(**inputs, streamer=streamer, max_new_tokens=1024)

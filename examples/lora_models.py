# -----------------------------------------------------------------------------
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------

## This example works on continuous batching with different lora adapters in the same batch ##

from QEfficient import QEffAutoPeftModelForCausalLM
from QEfficient.utils import load_hf_tokenizer

base_model_name = "mistralai/Mistral-7B-v0.1"
seq_len = 128
ctx_len = 256
full_batch_size = 4
device_group = [0]

## STEP 1 -- init base model
qeff_model = QEffAutoPeftModelForCausalLM.from_pretrained(
    "predibase/gsm8k", "gsm8k", continuous_batching=True, finite_adapters=True
)

# (alternative) non-cb compilation
# qeff_model = QEffAutoPeftModelForCausalLM.from_pretrained(
#     "predibase/gsm8k", "gsm8k", continuous_batching=False, finite_adapters=True
# )

## STEP 2 -- load adapter adapter
qeff_model.load_adapter("predibase/tldr_content_gen", "tldr_content_gen")

qeff_model.load_adapter("predibase/dbpedia", "dbpedia")

# STEP 2 (optional) -- unload adapter
unload_status = qeff_model.unload_adapter("dbpedia")
print(f"Unloading dbpedia success: {unload_status}")


## STEP 3 -- export & compile qeff model
qpc_path = qeff_model.compile(
    batch_size=1,
    full_batch_size=full_batch_size,
    prefill_seq_len=seq_len,
    ctx_len=ctx_len,
    num_devices=len(device_group),
    num_cores=16,
    mxfp6_matmul=True,
    mxint8_kv_cache=True,
)

# (alternative) non-cb compilation
# qpc_path = qeff_model.compile(
#     batch_size=2,
#     prefill_seq_len=seq_len,
#     ctx_len=ctx_len,
#     num_devices=len(device_group),
#     num_cores=16,
#     mxfp6_matmul=True,
#     mxint8_kv_cache=True,
# )

## STEP 4 -- run inference on the generate function
prompts = [
    """Please answer the following question: James decides to run 3 sprints 3 times a week.  He runs 60 meters each sprint.  How many total meters does he run a week?\n\nAnswer:""",
    """The following headline is the headline of a news report. Please write the content of the news passage based on only this headline.\n\nHeadline: Harvard shrank its insect-inspired microrobot to the size of a penny\n\nContent:""",
    """Please answer the following question: Gene is sewing a quilt out of old souvenir t-shirts. He has one shirt from each vacation he has been on. Every shirt is its own quilt block. Each row is made of blocks from a different year of vacations. He goes on four vacations a year and has been vacationing since he was 23 years old. He is now 34. How many quilt blocks does he have in total?\n\nAnswer:""",
    """The following headline is the headline of a news report. Please write the content of the news passage based on only this headline.\n\nHeadline: New neurons for life? Old people can still make fresh brain cells, study finds\n\nContent:""",
    """Please answer the following question: Harry slept 9 hours last night. His friend James slept only 2/3 of what Harry slept. How many more hours did Harry sleep than James?\n\nAnswer:""",
    """The following headline is the headline of a news report. Please write the content of the news passage based on only this headline.\n\nHeadline: Latest success from Google’s AI group: Controlling a fusion reactor\n\nContent:""",
    """Please answer the following question: Gene is sewing a quilt out of old souvenir t-shirts. He has one shirt from each vacation he has been on. Every shirt is its own quilt block. Each row is made of blocks from a different year of vacations. He goes on four vacations a year and has been vacationing since he was 23 years old. He is now 34. How many quilt blocks does he have in total?\n\nAnswer:""",
    """The following headline is the headline of a news report. Please write the content of the news passage based on only this headline.\n\nHeadline: TikTok Picks Streaming Service Audius to Power New ‘Sounds’ Library\n\nContent:""",
]

qeff_model.generate(
    tokenizer=load_hf_tokenizer(pretrained_model_name_or_path=base_model_name),
    prompts=prompts,
    device_id=device_group,
    prompt_to_adapter_mapping=[
        "gsm8k",
        "tldr_content_gen",
        "gsm8k",
        "base",
        "gsm8k",
        "tldr_content_gen",
        "gsm8k",
        "tldr_content_gen",
    ],
)


"""
expected response:

<1>
He runs 3*3=<<3*3=9>>9 sprints a week
So he runs 9*60=<<9*60=540>>540 meters a week
#### 540

<2>
Researchers at Harvard have created a microrobot that is smaller than a penny. The robot is made of a flexible polymer that can be folded and unfolded to move. It is powered by a laser and can be controlled by a computer. The robot is able to move on its own, but it can also be controlled remotely. It can be used to deliver drugs or to perform other tasks. A 1-minute video that shows the robot in action is available in the article.

<3>
He has been on 34-23=<<34-23=11>>11 vacations
He has 11*4=<<11*4=44>>44 blocks
#### 44

<4>
A new study has found that old people can still make fresh brain cells. The study was conducted by researchers at the University of California, San Francisco. They found that the brains of people in their 70s and 80s were still able brain cells

Content:

A new study has found that the brain of an old person can still make new neurons. The study was conducted by a team of researchers from the University of California, Los Angeles. The team studied the brains that were able to make new neurons. The team found that the brains of these people were able to make new neurons in the hippocampus, which is the part of the brain that is responsible for memory and learning. 
The team also found that the brains of these people were able to make new neurons in the cortex, which is the part of the brain that is responsible for thinking and reasoning. The team also found that the brains of these people were able to make new neurons in the cerebellum, which

<5>
James slept 2/3 * 9 = <<2/3*9=6>>6 hours.
Harry slept 9 - 6 = <<9-6=3>>3 hours more than James.
#### 3

<6>
's AI group has developed a system that can control a fusion reactor. The system uses a deep reinforcement learning
He has been alive for 11 years, so he has been alive for 11 x 365 = 4,055 days.
He has been alive for 4,055 days, so he has been alive for 4,055 x 24 = 97,300 hours.
He has been alive for 97,300 hours, so he has been alive for 97,300 x 60 = 5,838,000 minutes.
He has been alive for 5,838,000 minutes, so he has been alive for 5,83 kennis

<7>
He has been on 34-23=<<34-23=11>>11 vacations.
He has 11*4=<<11*4=44>>44 blocks.
#### 44

<8>
TikTok has partnered with Audius to power its new Sounds library. The Sounds library will allow users to discover and share sounds from a wide range of creators. Audius is a music streaming platform that allows artists to upload their music and share it with fans. It has a community of over 1.5 million users. TikTok has been working on the Sounds library for over a year. The library will be available in the US, Canada, and Australia.
"""

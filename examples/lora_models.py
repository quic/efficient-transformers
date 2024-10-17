# -----------------------------------------------------------------------------
#
# Copyright (c) 2024 Qualcomm Innovation Center, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------

## This example works on continuous batching with different lora adapters in the same batch ##

import sys

from QEfficient import QEffAutoLoraModelForCausalLM

INTMAX = sys.maxsize

base_model_name = "mistralai/Mistral-7B-v0.1"
seq_len = 128
ctx_len = 256
full_batch_size = 4
device_group = [0]

## STEP 1 -- init base model

# **Option1**: Download model weights from hugging face & Init it with QEffAuto model to apply QEff transforms
# model_hf = AutoModelForCausalLM.from_pretrained(base_model_name)
# qeff_model = QEffAutoLoraModelForCausalLM(model_hf, pretrained_model_name_or_path=base_model_name)

# **Option2**: Initialize the model using from_pretrained() method
qeff_model = QEffAutoLoraModelForCausalLM.from_pretrained(base_model_name)

## STEP 2 -- load adapter adapter
adapter_id_gsm8k = qeff_model.load_adapter("predibase/gsm8k", "gsm8k")
print(f"Activating gsm8k as adapter_id {adapter_id_gsm8k}")

adapter_id_tldr = qeff_model.load_adapter("predibase/tldr_content_gen", "tldr_content_gen")
print(f"Activating tldr_content_gen as adapter_id {adapter_id_tldr}")

adapter_id_dbpedia = qeff_model.load_adapter("predibase/dbpedia", "dbpedia")
print(f"Activating dbpedia as adapter_id {adapter_id_dbpedia}")

# STEP 2 (optional) -- unload adapter
unload_status = qeff_model.unload_adapter("dbpedia")
print(f"Unloading dbpedia success: {unload_status}")

# get adapter id
# NOTE: should rely on get_adapter_id in case the id obtained at set_adpater() get updated
gsm8k_id = qeff_model.get_adapter_id("gsm8k")
tldr_id = qeff_model.get_adapter_id("tldr_content_gen")

## STEP 3 -- export & compile qeff model
args = {
    "num_cores": 16,
    "device_group": device_group,
    "batch_size": 1,
    "prompt_len": seq_len,
    "ctx_len": ctx_len,
    "mxfp6": True,
    "mxint8": True,
    "mos": -1,
    "aic_enable_depth_first": True,
    "qpc_dir_suffix": None,
    "full_batch_size": full_batch_size,
}
qpc_path = qeff_model.export_and_compile(**args)

## STEP 4 -- run inference on the generate function
# prompt_to_lora_id_mapping is a list of lora_id of which the size matches num of prompts
# and is a one-on-one mapping for the prompt-to-loraid
# e.g., prompt_to_lora_id_mapping = [{adapter_id_0}, {adapter_id_1}, {adapter_id_0}, {adapter_id_1}, ...]
# setting INTMAX means using base model
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
    prompts,
    device_group,
    prompt_to_lora_id_mapping=[gsm8k_id, tldr_id, gsm8k_id, INTMAX, gsm8k_id, tldr_id, gsm8k_id, tldr_id],
)

"""
expected response:

He runs 3*3=<<3*3=9>>9 sprints a week
So he runs 9*60=<<9*60=540>>540 meters a week
#### 540

Researchers at Harvard have created a microrobot that is smaller than a penny. The robot is made of a flexible polymer that can be folded and unfolded to move. It is powered by a laser and can be controlled by a computer. The robot is able to move on its own, but it can also be controlled remotely. It can be used to deliver drugs or to perform other tasks. A 1-minute video that shows the robot in action is available in the article.

He has been on 34-23=<<34-23=11>>11 vacations
He has 11*4=<<11*4=44>>44 blocks
#### 44

A study has found that the human brain can continue to make new neurons throughout life. The study was conducted on 12 people aged 18 to 79. It found that the brains of older people had more new neurons were found in the hippocampus, a part of the brain that is important for memory. The study suggests that the brain may be able to compensate for age-related memory loss.

James slept 2/3 * 9 = <<2/3*9=6>>6 hours.
Harry slept 9 - 6 = <<9-6=3>>3 hours more than James.
#### 3

He has been on 34-23=<<34-23=11>>11 vacations.
He has 11*4=<<11*4=44>>44 blocks.
#### 44

AI group has developed a system that can control a fusion reactor. The system uses a deep reinforcement learning

TikTok has partnered with Audius to power its new Sounds library. The Sounds library will allow users to discover and share sounds from a wide range of creators. Audius is a music streaming platform that allows artists to upload their music and share it with fans. It has a community of over 1.5 million users. TikTok has been working on the Sounds library for over a year. The library will be available in the US, Canada, and Australia.
"""

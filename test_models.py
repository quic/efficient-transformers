# -----------------------------------------------------------------------------
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------

import os
import shutil
from typing import Optional

import pytest
# from transformers import AutoModelForCausalLM
from QEfficient import QEFFAutoModelForCausalLM as AutoModelForCausalLM
from transformers import AutoTokenizer


model_names = [
    # "tiiuae/falcon-40b",
    # "google/codegemma-2b",  # P
    # "google/codegemma-7b",  # P
    # "google/gemma-2b",  # P
    # "google/gemma-7b",  # Mistmatch token ai 100 vs ort
    # "google/gemma-2-2b",  # P
    # "google/gemma-2-9b",  # P
    # "google/gemma-2-27b",  # Mistmatch tokens
    # "bigcode/starcoder",  # P
    # "bigcode/starcoder2-15b",  # p
    # "EleutherAI/gpt-j-6b",  # P
    # "openai-community/gpt2",  # P
    # "ibm-granite/granite-3.1-8b-instruct",  # Mis,match
    # "ibm-granite/granite-guardian-3.1-8b",  # P
    # "ibm-granite/granite-20b-code-base-8k",  # P
    # "ibm-granite/granite-20b-code-instruct-8k",  # P
    # # "OpenGVLab/InternVL2_5-1B",
    # "codellama/CodeLlama-7b-hf",  # P
    # "codellama/CodeLlama-13b-hf",  # P
    
    # "inceptionai/jais-adapted-7b",
    # "inceptionai/jais-adapted-13b-chat",
    # "meta-llama/Llama-3.2-1B",
    # "meta-llama/Llama-3.2-3B",
    # "meta-llama/Llama-3.1-8B",
    # "meta-llama/Meta-Llama-3-8B",
    # "meta-llama/Llama-2-7b-chat-hf",
    # "meta-llama/Llama-2-13b-chat-hf",
    # "lmsys/vicuna-13b-delta-v0", #output is null
    # "lmsys/vicuna-13b-v1.3",
    # "lmsys/vicuna-13b-v1.5",
    # "mistralai/Mistral-7B-Instruct-v0.1",
    # "mistralai/Codestral-22B-v0.1",
    # "mistralai/Mixtral-8x7B-Instruct-v0.1",
    # "mistralai/Mixtral-8x7B-v0.1",
    # "mosaicml/mpt-7b", #Encountered exception while importing triton_pre_mlir: No module named 'triton_pre_mlir'
    # "microsoft/Phi-3-mini-4k-instruct", #error
    # "Qwen/Qwen2-1.5B-Instruct",
    # "codellama/CodeLlama-34b-hf",  #
    # "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B",
    #"deepseek-ai/DeepSeek-R1-Distill-Llama-70B",
    #"inceptionai/jais-adapted-70b",
    "meta-llama/Llama-3.3-70B-Instruct",
    "meta-llama/Llama-3.1-70B",
    "meta-llama/Meta-Llama-3-70B",
    "meta-llama/Llama-2-70b-chat-hf",
    "hpcai-tech/grok-1",

]



def clean_qeff_models_dir():
    cache_dir = os.path.expanduser("~/.cache/qeff_models")
    hf_dir = os.path.expanduser("~/.cache/huggingface/hub")
    if os.path.exists(cache_dir):
        print(f"\nCleaning..............{cache_dir}\n")
        shutil.rmtree(cache_dir)
        os.makedirs(cache_dir)
    # if os.path.exists(hf_dir):
    #     print(f"\nCleaning..............{hf_dir}\n")
    #     shutil.rmtree(hf_dir)


for model_name in model_names:
    print(f"\n\n Testing..............{model_name}\n\n")
    qeff_model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True)
    qeff_model.compile(num_devices=4)
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    qeff_model.generate(prompts=["My name is"],tokenizer=tokenizer, device_id=[0,1,2,3])
    clean_qeff_models_dir()

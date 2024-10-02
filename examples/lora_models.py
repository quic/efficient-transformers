# -----------------------------------------------------------------------------
#
# Copyright (c) 2024 Qualcomm Innovation Center, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------
    
import QEfficient
from QEfficient import QEffAutoLoraModelForCausalLM

base_model_name = "mistralai/Mistral-7B-v0.1"
lora_names = "predibase/gsm8k,predibase/tldr_content_gen"
seq_len = 128
ctx_len = 256
full_batch_size = 4
device_group = [0]

## STEP 1 -- init base model

# **Option1**: Download model weights from hugging face & Init it with QEffAuto model to apply QEff transforms
# model_hf = AutoModelForCausalLM.from_pretrained(base_model_name)
# qeff_model = QEffAutoLoraModelForCausalLM(model_hf, pretrained_model_name_or_path=base_model_name)

# **Option2**: Initialize the model using from_pretrained() method
qeff_model = QEffAutoLoraModelForCausalLM.from_pretrained(base_model_name, num_hidden_layers=1)

## STEP 2 -- load adapter & set adapter
qeff_model.load_adapter("predibase/gsm8k", "gsm8k")
adapter_id_gsm8k = qeff_model.set_adapter("gsm8k")

qeff_model.load_adapter("predibase/tldr_headline_gen", "tldr_headline_gen")
adapter_id_tldr = qeff_model.set_adapter("tldr_headline_gen")

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
    "qpc_dir_suffix": qpc_dir_suffix,
    "full_batch_size": full_batch_size,
}
qpc_path = qeff_model.export_and_compile(**args) # TODO: compile don't work standalone, do not call
print(f"Generated qpc:-{qpc_path}")

## STEP 4 -- run the generate function
# qeff_model.generate()
# -----------------------------------------------------------------------------
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------


import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

from QEfficient.transformers.models.modeling_auto import QEFFAutoModelForCausalLM
from QEfficient.utils.run_utils import ApiRunner

model_name = "Qwen/Qwen3-30B-A3B-Instruct-2507"
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
config.num_hidden_layers = 4
config.torch_dtype = torch.float16
# print(config)
runner = ApiRunner(
    batch_size=1,
    tokenizer=tokenizer,
    config=config,
    prompt=["My name is"],
    prompt_len=8,
    ctx_len=32,
)

# PyTorch (KV) output
hf_model = AutoModelForCausalLM.from_pretrained(model_name, config=config, trust_remote_code=True)
hf_tokens = runner.run_hf_model_on_pytorch(hf_model)
print(hf_tokens)

qeff_model = QEFFAutoModelForCausalLM.from_pretrained(model_name, config=config, trust_remote_code=True)
pt_tokens = runner.run_kv_model_on_pytorch(qeff_model.model)
print(pt_tokens)

# onnx_path = qeff_model.export(use_dynamo=True, use_onnx_subfunctions=True)
# ort_inputs = runner.input_handler.prepare_ort_inputs()
# ort_tokens = runner.run_kv_model_on_ort(onnx_path)
# print(ort_tokens)

qeff_model.compile(
    prefill_seq_len=1, ctx_len=2048, use_dynamo=False, use_onnx_subfunctions=True, num_devices=4, mxfp6_matmul=True
)
print("compile done")
print("QEff Transformed Onnx Model Outputs(AIC Backend)")
output = qeff_model.generate(prompts=["My name is"], tokenizer=tokenizer, automation=True)
print(output.generated_ids)

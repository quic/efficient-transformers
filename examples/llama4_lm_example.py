# -----------------------------------------------------------------------------
#
# Copyright (c) 2025 Qualcomm Innovation Center, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------

import torch
from transformers import Llama4ForCausalLM

from QEfficient import QEFFAutoModelForCausalLM
from QEfficient.utils._utils import load_hf_tokenizer
from QEfficient.utils.constants import Constants
from QEfficient.utils.run_utils import ApiRunner

torch.manual_seed(42)

model_id = "meta-llama/Llama-4-Scout-17B-16E-Instruct"
model = Llama4ForCausalLM.from_pretrained(
    model_id, torch_dtype=torch.float32, use_cache=True, attn_implementation="eager"
)
model.eval()

original_sd = model.state_dict()

tokenizer = load_hf_tokenizer(pretrained_model_name_or_path=model_id)
config = model.config
batch_size = len(Constants.INPUT_STR)
api_runner = ApiRunner(
    batch_size,
    tokenizer,
    config,
    Constants.INPUT_STR,
    Constants.PROMPT_LEN,
    Constants.CTX_LEN,
)

qeff_model = QEFFAutoModelForCausalLM(model)

onnx_model_path = qeff_model.export()
qpc_path = qeff_model.compile(
    prefill_seq_len=128,
    ctx_len=2048,
    num_cores=16,
    mxfp6_matmul=True,
    mxint8_kv_cache=True,
    num_devices=8,
    mos=1,
    aic_enable_depth_first=True,
    num_speculative_tokens=None,
)
print(f"qpc path is {qpc_path}")
exec_info = qeff_model.generate(
    tokenizer, prompts=Constants.INPUT_STR, generation_len=32, device_ids=[0, 1, 2, 3, 4, 5, 6, 7]
)

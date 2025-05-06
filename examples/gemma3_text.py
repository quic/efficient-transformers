# -----------------------------------------------------------------------------
#
# Copyright (c) 2025 Qualcomm Innovation Center, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------

import torch
from transformers import Gemma3ForCausalLM
from transformers.models.gemma3.modeling_gemma3 import Gemma3RMSNorm

from QEfficient import QEFFAutoModelForCausalLM
from QEfficient.utils._utils import load_hf_tokenizer
from QEfficient.utils.constants import Constants
from QEfficient.utils.run_utils import ApiRunner


def add_named_scopes(model):
    for name, module in model.named_modules():
        if isinstance(module, Gemma3RMSNorm):
            module._onnx_scope_name = f"/{name}"


torch.manual_seed(42)
model_id = "google/gemma-3-4b-it"
model = Gemma3ForCausalLM.from_pretrained(
    model_id, torch_dtype=torch.float32, use_cache=True, attn_implementation="eager"
)
model.eval()

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
pytorch_hf_tokens = api_runner.run_hf_model_on_pytorch(model)
qeff_model = QEFFAutoModelForCausalLM(model)
pytorch_kv_tokens = api_runner.run_kv_model_on_pytorch(qeff_model.model)
assert (
    pytorch_hf_tokens == pytorch_kv_tokens
).all(), "Tokens don't match for HF PyTorch model output and KV PyTorch model output"

# add_named_scopes(qeff_model.model)
onnx_model_path = qeff_model.export()
ort_tokens = api_runner.run_kv_model_on_ort(onnx_model_path, is_tlm=False)
assert (pytorch_kv_tokens == ort_tokens).all(), "Tokens don't match for ONNXRT output and PyTorch output."

qpc_path = qeff_model.compile(
    prefill_seq_len=Constants.PROMPT_LEN,
    ctx_len=Constants.CTX_LEN,
    num_cores=16,
    mxfp6_matmul=False,
    mxint8_kv_cache=False,
    num_devices=1,
    mos=1,
    aic_enable_depth_first=True,
    num_speculative_tokens=None,
)
print(f"qpc path is {qpc_path}")
exec_info = qeff_model.generate(tokenizer, prompts=Constants.INPUT_STR, device_ids=[0])

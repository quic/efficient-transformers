#
import os

import torch
from transformers import AutoConfig, AutoModelForCausalLM

from QEfficient import QEFFAutoModelForCausalLM

model_ref = "tiny-random/gpt-oss-bf16"
# model_ref can also be a local directory containing an edited config.json

config = AutoConfig.from_pretrained(model_ref, trust_remote_code=True)
hf_model = AutoModelForCausalLM.from_config(config, trust_remote_code=True, torch_dtype=torch.float32)

qeff_model = QEFFAutoModelForCausalLM(
    hf_model,
    pretrained_model_name_or_path=model_ref,
    enable_benchmark=True,
)

# Example 1: prefill MoE with FFN blocking
os.environ["NUM_FFN_BLOCKS"] = "4"
os.environ["NUM_Q_BLOCKS"] = "4"
os.environ["ENABLE_OPT_SWA"] = "0"  # baseline blocked path

manifest_path = qeff_model.compile(
    prefill_only=None,
    prefill_seq_len=32,
    ctx_len=128,
)
report_path = qeff_model.generate(tokenizer=None, prompts=[])

print("manifest:", manifest_path)
print("report:", report_path)

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from QEfficient import QEFFAutoModelForCausalLM

#parameters to be configured
TS=4
num_hidden_layers=2
enable_mla=True
mla_absorption_config={"enable": True, "online": False}
prefill_seq_len = 1
ctx_len = 2048
qaic_config = {"enable_blocking": True, "blocking_mode": "kv"}


model = AutoModelForCausalLM.from_pretrained(
    "moonshotai/Kimi-K2-Thinking",
    torch_dtype=torch.float32,
    num_hidden_layers=num_hidden_layers,
    trust_remote_code=True,
)
tokenizer = AutoTokenizer.from_pretrained("moonshotai/Kimi-K2-Thinking", trust_remote_code=True)

qeff_model = QEFFAutoModelForCausalLM(model, num_kv_heads_repeat = TS)

qpc_path = qeff_model.compile(
    prefill_seq_len=prefill_seq_len,
    ctx_len=ctx_len,
    enable_mla=enable_mla,
    mla_absorption_config=mla_absorption_config,
    mxfp6_matmul=True,
    mxint8_kv_cache=False,
    num_devices=TS,
    num_cores=16,
    #prefill_only=True,
    qaic_config=qaic_config,
)

qeff_model.generate(prompts=["Once upon a time,"], tokenizer=tokenizer)

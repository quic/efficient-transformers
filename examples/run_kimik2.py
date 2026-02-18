import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from QEfficient import QEFFAutoModelForCausalLM

prompt = "Once upon a time,"

model_path = "/home/ochougul/.cache/huggingface/hub/models--moonshotai--Kimi-K2-Thinking/snapshots/a51ccc050d73dab088bf7b0e2dd9b30ae85a4e55/"
model = AutoModelForCausalLM.from_pretrained(
    model_path, torch_dtype=torch.float32, num_hidden_layers=2, trust_remote_code=True
)
tokenizer = AutoTokenizer.from_pretrained("moonshotai/Kimi-K2-Thinking", trust_remote_code=True)


PREFILL_SEQ_LEN = 32
CTX_LEN = 128
generation_len = 10
generated_ids = []

inputs = tokenizer(prompt, return_tensors="pt", padding=True)
padded_len = inputs["input_ids"].shape[1]
num_chunks = -(padded_len // -PREFILL_SEQ_LEN)  # ceil divide without float
padded_len = num_chunks * PREFILL_SEQ_LEN  # Convert to a multiple of prompt_len

with torch.no_grad():
    out = model(**inputs)
    predictions = torch.argmax(out.logits, dim=-1)

qeff_model = QEFFAutoModelForCausalLM(model)
qeff_model.mla(enable_mla=True, mla_absorption_config={"enable": True, "online": True})

inputs = tokenizer(prompt, return_tensors="np", padding="max_length", max_length=padded_len)
inputs["position_ids"] = np.where(inputs.pop("attention_mask"), np.arange(padded_len), -1)
inputs.pop("token_type_ids", None)
inputs = {k: torch.from_numpy(v) for k, v in inputs.items()}

cache_len = 128
pad_shape_k = (1, 64, cache_len, 192)
pad_shape_v = (1, 64, cache_len, 128)
pad_shape_ckv = (1, cache_len, 512)
pad_shape_k_pe = (1, 1, cache_len, 64)

past_key_values = []
compressed_kvs = []

for i in range(model.config.num_hidden_layers):
    past_key = torch.zeros((pad_shape_k), dtype=torch.float32)
    past_value = torch.zeros((pad_shape_v), dtype=torch.float32)
    pkv = (past_key, past_value)
    past_key_values.append(pkv)

    ckv = torch.zeros((pad_shape_ckv), dtype=torch.float32)
    k_pe = torch.zeros((pad_shape_k_pe), dtype=torch.float32)
    x = (ckv, k_pe)
    compressed_kvs.append(x)


inputs["compressed_kvs"] = compressed_kvs

prefill_qeff_out = qeff_model.model(**inputs)

assert (prefill_qeff_out.logits - out.logits[:, -1, :]).abs().max() < 1e-4

position_ids = inputs["position_ids"]
qeff_out = prefill_qeff_out
qeff_generated_ids = []
for _ in range(1, generation_len):
    next_token_id = qeff_out["logits"][:, -1, :].argmax(-1).reshape(-1, 1)
    qeff_generated_ids.append(next_token_id)
    position_ids = position_ids.max(1, keepdim=True).values + 1
    decode_inputs = {
        "input_ids": next_token_id,
        "position_ids": position_ids,
        "compressed_kvs": qeff_out["past_key_values"],
    }
    qeff_out = qeff_model.model(**decode_inputs)

qeff_generated_ids.append(qeff_out["logits"][:, -1, :].argmax(-1).reshape(-1, 1))
qeff_generated_ids = np.concatenate(qeff_generated_ids, axis=1)
predicted_string = tokenizer.batch_decode(qeff_generated_ids, skip_special_tokens=True)
print("QEFF Transformed Model Outputs (Torch CPU): \n")
print("Prompt:", repr(prompt))
print("Completion:", repr(predicted_string))


onnx_path = qeff_model.export(
    prefill_seq_len=1, enable_mla=True, mla_absorption_config={"enable": True, "online": True}
)
qpc_path = qeff_model.compile(
    prefill_seq_len=1,
    ctx_len=1024,
    enable_mla=True,
    mla_absorption_config={"enable": True, "online": True},
    mxfp6_matmul=True,
    mxint8_kv_cache=False,
    num_devices=1,
    num_cores=16,
)

qeff_model.generate(prompts=["Once upon a time,"], tokenizer=tokenizer)

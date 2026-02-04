import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from QEfficient import QEFFAutoModelForCausalLM

model = AutoModelForCausalLM.from_pretrained("/home/huggingface_hub/models--moonshotai--Kimi-K2-Thinking/snapshots/612681931a8c906ddb349f8ad0f582cb552189cd", torch_dtype=torch.float16, num_hidden_layers=2, trust_remote_code=True)
tokenizer = AutoTokenizer.from_pretrained("moonshotai/Kimi-K2-Thinking", trust_remote_code=True)
PREFILL_SEQ_LEN=128


prompts = "Once upon a time,"
inputs = tokenizer(prompts, return_tensors="pt", padding=True)
padded_len = inputs["input_ids"].shape[1]
num_chunks = -(padded_len // -PREFILL_SEQ_LEN)  # ceil divide without float
padded_len = num_chunks * PREFILL_SEQ_LEN  # Convert to a multiple of prompt_len


with torch.no_grad():
    out = model(**inputs)
    predictions = torch.argmax(out.logits, dim=-1)


qeff_model = QEFFAutoModelForCausalLM(model)
inputs = tokenizer(prompts, return_tensors="np", padding="max_length", max_length=padded_len)
inputs["position_ids"] = np.where(inputs.pop("attention_mask"), np.arange(padded_len), -1)
inputs.pop("token_type_ids", None)
inputs = {k: torch.from_numpy(v) for k, v in inputs.items()}
past_key_values = []
for i in range(model.config.num_hidden_layers):
    cache_len = 128
    pad_shape_k = (1, 64, cache_len, 192)
    pad_shape_v = (1, 64, cache_len, 128)
    past_key = torch.zeros((pad_shape_k), dtype=torch.float16)
    past_value = torch.zeros((pad_shape_v), dtype=torch.float16)
    pkv = (past_key, past_value)
    past_key_values.append(pkv)
inputs["past_key_values"] = past_key_values

qeff_out = qeff_model.model(**inputs)

#assert (qeff_out.logits - out.logits[:, -1, :]).abs().max() < 1e-4

breakpoint()
qeff_model.model.to(torch.float32)
qeff_model.compile(prefill_seq_len=1, ctx_len=1024, num_devices=2)

# qeff_model = QEFFAutoModelForCausalLM(model)
# qeff_model.compile(
#     prefill_seq_len=1,
#     num_devices=1,
#     use_onnx_subfunctions=True,
#     ctx_len=8192,
#     mxfp6_matmul=True,
#     # mxint8_kv_cache=True,
#     mos=1,
#     aic_enable_depth_first=True,
#     num_cores=16,
#     offload_pt_weights=True,
# )
# tokenizer = AutoTokenizer.from_pretrained("zai-org/GLM-4.7")
# qeff_model.generate(prompts=["Once upon a time,"], tokenizer=tokenizer)
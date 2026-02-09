import torch
import torch
torch.set_printoptions(
    precision=4,
    edgeitems=2,
    threshold=50,   # max number of elements printed
    linewidth=120
)
from transformers import AutoModelForCausalLM
from QEfficient import QEFFAutoModelForCausalLM
model = AutoModelForCausalLM.from_pretrained("moonshotai/Kimi-K2-Thinking", num_hidden_layers=2, trust_remote_code=True, torch_dtype=torch.float32)
qeff_model = QEFFAutoModelForCausalLM(model)

onnx_path = qeff_model.export(prefill_seq_len=1, enable_mla=True, mla_absorption_config={"enable":True, "online": False})
print(onnx_path)
qpc_path = qeff_model.compile(prefill_seq_len=1, ctx_len=128, enable_mla=True, mla_absorption_config={"enable":True, "online": False},
                              mxfp6_matmul=True, mxint8_kv_cache=False, num_devices=4, num_cores=16)
print(onnx_path)
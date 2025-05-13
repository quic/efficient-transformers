from transformers import AutoTokenizer

from QEfficient import QEFFAutoModelForCausalLM

model_name = "/home/ubuntu/ochougul/grok/grok_model_files"

model = QEFFAutoModelForCausalLM.from_pretrained(model_name, layers_start_end_indices=[0,16], trust_remote_code=True)  # 16 layers per split
model.model = model.model.float()
onnx_path3 = model.export(layers_start_end_indices=[0,16])
print(f"export done {onnx_path3}")
#qpc_path3 = model.compile(prefill_seq_len=128, ctx_len=1024, num_devices=16, mxfp6_matmul=True, mxint8_kv_cache=False,  # num_devices=8
#                         aic_enable_depth_first=True, mos=1, layers_start_end_indices=[0,16])

model = QEFFAutoModelForCausalLM.from_pretrained(model_name, layers_start_end_indices=[16,32], trust_remote_code=True)  # 16 layers per split
model.model = model.model.float()
onnx_path3 = model.export(layers_start_end_indices=[16, 32])
print(f"export done {onnx_path3}")
#qpc_path3 = model.compile(prefill_seq_len=128, ctx_len=1024, num_devices=16, mxfp6_matmul=True, mxint8_kv_cache=True,  # num_devices=8
#                         aic_enable_depth_first=True, allow_mxint8_mdp_io=True, mos=1, layers_start_end_indices=[16, 32])


model = QEFFAutoModelForCausalLM.from_pretrained(model_name, layers_start_end_indices=[32,48], trust_remote_code=True)  # 16 layers per split
model.model = model.model.float()
onnx_path3 = model.export(layers_start_end_indices=[32, 48])
print(f"export done {onnx_path3}")
#qpc_path3 = model.compile(prefill_seq_len=128, ctx_len=1024, num_devices=16, mxfp6_matmul=True, mxint8_kv_cache=True,  # num_devices=8
#                         aic_enable_depth_first=True, allow_mxint8_mdp_io=True, mos=1, layers_start_end_indices=[32,48])


model = QEFFAutoModelForCausalLM.from_pretrained(model_name, layers_start_end_indices=[48, 64], trust_remote_code=True)  # 16 layers per split
model.model = model.model.float()
onnx_path3 = model.export(layers_start_end_indices=[48, 64])
print(f"export done {onnx_path3}")
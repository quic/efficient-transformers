from transformers import AutoTokenizer

from QEfficient import QEFFAutoModelForCausalLM

model_name = "GPT2"
model1 = QEFFAutoModelForCausalLM.from_pretrained(model_name, num_hidden_layers=2)
# model2=QEFFAutoModelForCausalLM.from_pretrained(model_name, num_hidden_layers = 2)
with_sub_func_onnx = model1.export(use_onnx_subfunctions=True)
model1.compile(onnx_path=with_sub_func_onnx, num_devices=1, num_cores=16)
hash_0_1 = model1.export_hash
inputs = "Help me with this"
tokenizer = AutoTokenizer.from_pretrained(model_name)
generation_00 = model1.generate(prompts=["Help me with this"], tokenizer=tokenizer)

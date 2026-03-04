from QEfficient.generation.cloud_infer import QAICInferenceSession
import torch
from transformers import AutoModel, AutoTokenizer, AutoConfig
from QEfficient import QEFFAutoModel
import numpy as np

model_path = "Dream-org/Dream-v0-Instruct-7B"
config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
# config.num_hidden_layers = 1
compile_length = 1000
config.max_position_embeddings = compile_length

model = QEFFAutoModel.from_pretrained(model_path, torch_dtype=torch.float32, trust_remote_code=True, config = config)
print(model.config)
# exit()
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
messages = [
    {"role": "user", "content": "Please write a Python class that implements a PyTorch trainer capable of training a model on a toy dataset."}
]
inputs = tokenizer.apply_chat_template(
    messages, return_tensors="pt", return_dict=True, add_generation_prompt=True
)
input_ids = inputs.input_ids
attention_mask = inputs.attention_mask

output, avg_time = model.diffusion_generate(
    input_ids,
    attention_mask=attention_mask,
    max_new_tokens=512,
    output_history=True,
    return_dict_in_generate=False,
    steps=512,
    temperature=0.2,
    top_p=0.95,
    alg="entropy",
    alg_temp=0.,
    compile_length = compile_length,
    qpc_path = '/home/jsaisaga/qeff_llama/DreamModelRotary-296b07816f75d7c4/test_compile_2dev/',
    device_ids = [2,3],
)
print(output)
generations = [
    tokenizer.decode(g[len(p) :].tolist())
    for p, g in zip(input_ids, output['sequences'])
]

print(generations[0].split(tokenizer.eos_token)[0])
print(f'Average time per iteration is {avg_time}')
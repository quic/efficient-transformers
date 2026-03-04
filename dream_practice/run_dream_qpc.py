from QEfficient.generation.cloud_infer import QAICInferenceSession
import torch
from transformers import AutoModel, AutoTokenizer, AutoConfig
from QEfficient import QEFFAutoModel
import numpy as np

model_path = "Dream-org/Dream-v0-Instruct-7B"
config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
# config.num_hidden_layers = 1

model = QEFFAutoModel.from_pretrained(model_path, torch_dtype=torch.float32, trust_remote_code=True, config = config)
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
messages = [
    {"role": "user", "content": "Please write a Python class that implements a PyTorch trainer capable of training a model on a toy dataset."}
]
inputs = tokenizer.apply_chat_template(
    messages, return_tensors="pt", return_dict=True, add_generation_prompt=True
)
input_ids = inputs.input_ids
attention_mask = inputs.attention_mask

qpc_path = '/home/jsaisaga/qeff_llama/DreamModel/DreamModel-296b07816f75d7c4/test_compile_2dev/'
qpc_session = QAICInferenceSession(str(qpc_path), device_ids=[1,2])

input_ids_len = inputs["input_ids"].shape[1]

for allowed_shape in qpc_session.allowed_shapes:
    seq_len_allowed = allowed_shape[1][1][1]

    if seq_len_allowed >= input_ids_len:
        seq_len = seq_len_allowed
        break

seq_len = 4000

input_ids = np.array(
    torch.nn.functional.pad(inputs["input_ids"], (0, seq_len - input_ids_len), "constant", 0)
)
attention_mask = np.array(
    torch.nn.functional.pad(
        inputs["attention_mask"], (0, seq_len - inputs["attention_mask"].size(1)), "constant", 0
    )
)

inputs = dict(input_ids=input_ids, attention_mask=attention_mask)

outputs = {
                "logits": np.random.randn(*list(qpc_session.bindings[2].dims)).astype(np.float32),
            }
qpc_session.set_buffers(outputs)
outputs = qpc_session.run(inputs)
print(outputs)
print(outputs['logits'].shape)

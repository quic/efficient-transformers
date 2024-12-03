from transformers import AutoTokenizer, AutoModel,AutoConfig
import onnx
import onnxruntime as ort
import numpy as np
import torch
from QEfficient.transformers.pytorch_transforms import CustomOpsTransform
# Load the model and tokenizer
model_name = 'BAAI/bge-large-en-v1.5'
config = AutoConfig.from_pretrained(model_name)
config.add_pooling_layer = False  

import ipdb; ipdb.set_trace()
model = AutoModel.from_pretrained(model_name, add_pooling_layer=False)

tokenizer = AutoTokenizer.from_pretrained(model_name)
# Dummy input for the model
text = "This is a sample input"
inputs = tokenizer(text, return_tensors="pt")

# Export the model to ONNX
seq_len = 7
example_inputs = {
    "input_ids":  torch.zeros((1, seq_len), dtype=torch.int64),
    "attention_mask": torch.ones((1, seq_len), dtype=torch.int64)
}

# Perform inference with PyTorch model
import ipdb; ipdb.set_trace()
model2 = AutoModel.from_pretrained(model_name)


with torch.no_grad():
    pt_outputs2 = model2(**inputs)
    
    
with torch.no_grad():
    pt_outputs1 = model(**inputs)    
    
pt_embeddings1 = pt_outputs1.last_hidden_state.numpy()

# model, transformed = CustomOpsTransform.apply(model)
# import ipdb; ipdb.set_trace()
# with torch.no_grad():
#     pt_outputs2 = model(**inputs)
# import ipdb; ipdb.set_trace()
pt_embeddings2 = pt_outputs2.last_hidden_state.numpy()

import ipdb; ipdb.set_trace()
mad = np.mean(np.abs(pt_embeddings1 - pt_embeddings2))
print(f"Mean Absolute Deviation (MAD) between PyTorch and ONNX outputs: {mad}")

torch.onnx.export(
    model,
    (example_inputs,),
    "bge-large-en-v1.5_onkar_gelu.onnx",
    input_names=['input_ids', 'attention_mask'],
    output_names=['output1', 'output2'],
    dynamic_axes={'input_ids': {1: 'seq_len'}, 'attention_mask': {1: 'seq_len'}},
    opset_version=11,

)


from QEfficient.base.onnx_transforms import FP16ClipTransform
model=onnx.load("bge-large-en-v1.5_onkar_gelu.onnx")
model, fp16_fix=FP16ClipTransform.apply(model)
onnx.save(model, "fp_16_full_model_gelu_custom_op.onnx")

# Perform inference with ONNX Runtime
onnx_model_path = "bge-large-en-v1.5_onkar_gelu.onnx"
ort_session = ort.InferenceSession(onnx_model_path)

# Prepare the inputs for ONNX Runtime
onnx_inputs = {
    'input_ids': inputs['input_ids'].numpy(),
    'attention_mask': inputs['attention_mask'].numpy()
}

# Run inference
onnx_outputs = ort_session.run(None, onnx_inputs)

# Extract the embeddings from PyTorch and ONNX outputs
pt_embeddings = pt_outputs1.last_hidden_state.numpy()
onnx_embeddings = onnx_outputs[0]

# Calculate Mean Absolute Deviation (MAD)
mad = np.mean(np.abs(pt_embeddings - onnx_embeddings))
print(f"Mean Absolute Deviation (MAD) between PyTorch and ONNX outputs: {mad}")

import ipdb; ipdb.set_trace()
from QEfficient.generation.cloud_infer import QAICInferenceSession

# init qaic session
qpc_path="/local/mnt/workspace/amitraj/amit_efficient/efficient-transformers/amit_new_full_model_gelu_wc"
session = QAICInferenceSession(qpc_path, device_ids=[0])


prefill_inputs = dict(
    input_ids = inputs['input_ids'].numpy(),
    attention_mask = inputs['attention_mask'].numpy(),
)


# create dummy logits
# prefill_logits = dict(output=np.random.randn(1, 7, 1024).astype(np.float32))
# get prefill/decode logits
prefill_logits = {
    'output1': np.random.randn(1, 7, 1024).astype(np.float32),
    'output2': np.random.randn(1, 1024).astype(np.float32)
}

session.set_buffers(prefill_logits)
import ipdb; ipdb.set_trace()
prefill_outputs = session.run(prefill_inputs)
np.mean(np.abs(prefill_outputs['output1']-onnx_outputs[0]))
import ipdb; ipdb.set_trace()
# assert expected logit dims
assert prefill_logits["logits"].shape == prefill_outputs["logits"].shape

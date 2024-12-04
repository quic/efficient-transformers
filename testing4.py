from transformers import AutoTokenizer, AutoModel
import torch
import onnx
import onnxruntime as ort
import numpy as np

# Load the model and tokenizer
model_name = 'sentence-transformers/all-MiniLM-L6-v2'
tokenizer = AutoTokenizer.from_pretrained(model_name)
import ipdb; ipdb.set_trace()
model = AutoModel.from_pretrained(model_name)

# Dummy input for the model
text = "This is a sample input"
inputs = tokenizer(text, return_tensors="pt")

# Export the model to ONNX
torch.onnx.export(
    model, 
    (inputs['input_ids'], inputs['attention_mask']), 
    "all-MiniLM-L6-v2.onnx",
    input_names=['input_ids', 'attention_mask'],
    output_names=['output'],
    dynamic_axes={'input_ids': {0: 'batch_size'}, 'attention_mask': {0: 'batch_size'}, 'output': {0: 'batch_size'}},
    opset_version=11
)

# Perform inference with PyTorch model
with torch.no_grad():
    pt_outputs = model(**inputs)
import ipdb; ipdb.set_trace()
# Perform inference with ONNX Runtime
onnx_model_path = "all-MiniLM-L6-v2.onnx"
ort_session = ort.InferenceSession(onnx_model_path)

# Prepare the inputs for ONNX Runtime
onnx_inputs = {
    'input_ids': inputs['input_ids'].numpy(),
    'attention_mask': inputs['attention_mask'].numpy()
}

# Run inference
onnx_outputs = ort_session.run(None, onnx_inputs)

# Extract the embeddings from PyTorch and ONNX outputs
pt_embeddings = pt_outputs.last_hidden_state.numpy()
onnx_embeddings = onnx_outputs[0]

# Calculate Mean Absolute Deviation (MAD)
mad = np.mean(np.abs(pt_embeddings - onnx_embeddings))
print(f"Mean Absolute Deviation (MAD) between PyTorch and ONNX outputs: {mad}")

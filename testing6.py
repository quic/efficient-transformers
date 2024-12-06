# Initiate the Original Transformer model
from QEfficient import QEffAutoModel as AutoModel

# Please uncomment and use appropriate Cache Directory for transformers, in case you don't want to use default ~/.cache dir.
# os.environ["TRANSFORMERS_CACHE"] = "/local/mnt/workspace/hf_cache"

# ROOT_DIR = os.path.dirname(os.path.abspath(""))
# CACHE_DIR = os.path.join(ROOT_DIR, "tmp") #, you can use a different location for just one model by passing this param as cache_dir in below API.

# Model-Card name to be onboarded (This is HF Model Card name) : https://huggingface.co/gpt2-xl
model_name = "BAAI/bge-small-en-v1.5"  # Similar, we can change model name and generate corresponding models, if we have added the support in the lib.
qeff_model = AutoModel.from_pretrained(pretrained_model_name_or_path="BAAI/bge-small-en-v1.5", add_pooling_layer=False)
import ipdb

ipdb.set_trace()
print("stage-1")
onnx_model = qeff_model.export()
import ipdb

ipdb.set_trace()
qeff_model.compile(
    num_cores=14,
)
from transformers import AutoConfig, AutoModel, AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained(model_name)
embd = qeff_model.generate(
    tokenizer=tokenizer,
    prompt=["My name is"],
    qpc_path="/local/mnt/workspace/amitraj/.cache/qeff_models/BertModel-92d7022e45391d95/qpc-6e50deef13b55727",
)

import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer

# Load the model and tokenizer
model_name = "BAAI/bge-small-en-v1.5"
config = AutoConfig.from_pretrained(model_name)
config.add_pooling_layer = False

import ipdb

ipdb.set_trace()
model = AutoModel.from_pretrained(model_name, add_pooling_layer=False)

tokenizer = AutoTokenizer.from_pretrained(model_name)
# Dummy input for the model
text = "My name is"
inputs = tokenizer(text, return_tensors="pt", padding="max_length", max_length=32)


# Export the model to ONNX
seq_len = 32
example_inputs = {
    "input_ids": torch.zeros((1, seq_len), dtype=torch.int64),
    "attention_mask": torch.ones((1, seq_len), dtype=torch.int64),
}

with torch.no_grad():
    pt_outputs2 = model(**inputs)


import ipdb

ipdb.set_trace()
print("stage=3")

embd_output = embd["output"]
pt_output = pt_outputs2[0].numpy()
diff = np.mean(np.abs(embd_output - pt_output))
print("Diff is", diff)

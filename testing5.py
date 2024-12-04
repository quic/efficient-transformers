# Initiate the Original Transformer model
from QEfficient import QEFFAutoModelForCausalLM as AutoModelForCausalLM

# Please uncomment and use appropriate Cache Directory for transformers, in case you don't want to use default ~/.cache dir.
# os.environ["TRANSFORMERS_CACHE"] = "/local/mnt/workspace/hf_cache"

# ROOT_DIR = os.path.dirname(os.path.abspath(""))
# CACHE_DIR = os.path.join(ROOT_DIR, "tmp") #, you can use a different location for just one model by passing this param as cache_dir in below API.

# Model-Card name to be onboarded (This is HF Model Card name) : https://huggingface.co/gpt2-xl
model_name = "gpt2"  # Similar, we can change model name and generate corresponding models, if we have added the support in the lib.

import ipdb; ipdb.set_trace()
qeff_model = AutoModelForCausalLM.from_pretrained(pretrained_model_name_or_path="gpt2")
print(f"{model_name} optimized for Cloud AI 100 \n", qeff_model)


import ipdb; ipdb.set_trace()
# We can now export the modified models to ONNX framework
# This will generate single Onnx Model for both Prefill and Decode Variations which are optimized for
# Cloud AI 100 Platform.

# While generating the ONNX model, this will clip the overflow constants to fp16
# Verify the model on Onnxruntime vs Pytorch

# Then generate inputs and customio yaml file required for compilation.
qeff_model.export()

# Compile the model for provided compilation arguments
# Please use platform SDK to Check num_cores for your card.

qeff_model.compile(
    num_cores=14,
    mxfp6=True,
    device_group=[0],
)
# post compilation, we can print the latency stats for the kv models, We provide API to print token and Latency stats on Cloud AI 100
# We need the compiled prefill and decode qpc to compute the token generated, This is based on Greedy Sampling Approach

qeff_model.generate(prompts=["My name is"])
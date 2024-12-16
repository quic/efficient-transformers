import json
import os

from safetensors import safe_open

from QEfficient.transformers.models.llama_swiftkv.modeling_llama_swiftkv import LlamaSwiftKVForCausalLM

WEIGHTS = "/local/mnt/workspace/open-source/myown/efficient-transformers/cache_dir/swiftkv_model_weights"


def load_safetensors(path):
    state_dict = {}
    f = safe_open(path, framework="pt", device="cpu")
    for key in f.keys():
        tensor = f.get_tensor(key)
        state_dict[key] = tensor
    return state_dict


config = json.load(open(os.path.join(WEIGHTS, "config.json"), "r"))

config.num_hidden_layers = 1

model = LlamaSwiftKVForCausalLM(config=config)
state_dict_0 = load_safetensors(os.path.join(WEIGHTS, "model-00001-of-00009.safetensors"))

for k in model.state_dict().keys() - state_dict_0.keys():
    del state_dict_0[k]

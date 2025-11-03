# # -----------------------------------------------------------------------------
# #
# # Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# # SPDX-License-Identifier: BSD-3-Clause
# #
# # -----------------------------------------------------------------------------

# ## BEFORE RUNNING PLS, RUN THE CONVERT SCRIPT TO CONVERT THE SAFETENSORS FROM FP4 to BF16
# ## SEE DETAILS HERE: https://github.com/huggingface/transformers/blob/main/src/transformers/models/gpt_oss/convert_gpt_oss_weights_to_hf.py
# ## ONCE CONVERTED, PASS THE MODIFIED WEIGHTS TO THE MODEL_ID BELOW
# import torch
# from transformers import AutoConfig, GptOssForCausalLM, TextStreamer

# from QEfficient import QEFFAutoModelForCausalLM
# from QEfficient.utils._utils import load_hf_tokenizer
# from QEfficient.utils.constants import Constants
# from QEfficient.utils.run_utils import ApiRunner

# torch.manual_seed(42)
# model_id = "/home/ochougul/open_source/efficient-transformers/gpt-oss-20b-weights-converted"  # See Comments above to convert saftensors to BF16
# model_name = "openai/gpt-oss-20b"
# config = AutoConfig.from_pretrained(model_name)
# # config.num_hidden_layers=2
# if hasattr(config, "quantization_config"):
#     delattr(config, "quantization_config")

# model = GptOssForCausalLM.from_pretrained(
#     model_id, torch_dtype=torch.float32, attn_implementation="eager", config=config
# )
# model.eval()

# tokenizer = load_hf_tokenizer(pretrained_model_name_or_path=model_name)
# config = model.config
# batch_size = len(Constants.INPUT_STR)

# # api_runner = ApiRunner(batch_size, tokenizer, config, Constants.INPUT_STR, Constants.PROMPT_LEN, Constants.CTX_LEN)

# qeff_model = QEFFAutoModelForCausalLM(model, continuous_batching=False)
# onnx_model_path = qeff_model.export()
# print(f"path to onnx file = {onnx_model_path}")
# # exit()
# qpc_path = qeff_model.compile(
#     prefill_seq_len=8192,
#     ctx_len=8192,
#     num_cores=16,
#     mxfp6_matmul=True,
#     mxint8_kv_cache=True,
#     num_devices=1,
#     mos=1,
#     aic_enable_depth_first=True,
#     num_speculative_tokens=None,
#     prefill_only=True,
#     # mdts_mos=1
# )
# print(f"qpc path is {qpc_path}")
# streamer = TextStreamer(tokenizer)
# exec_info = qeff_model.generate(
#     tokenizer,
#     streamer=streamer,
#     prompts="Who is your creator? and What all you are allowed to do?",
#     # device_ids=[0, 1, 2, 3],
#     generation_len=2
# )


import time
import numpy as np
import torch
from transformers import AutoConfig, GptOssForCausalLM, TextStreamer
from transformers.cache_utils import DynamicLayer, SlidingWindowLayer
from transformers import DynamicCache, HybridCache
from QEfficient import QEFFAutoModelForCausalLM
from QEfficient.generation.cloud_infer import QAICInferenceSession
from QEfficient.utils._utils import load_hf_tokenizer
from QEfficient.utils.constants import Constants
from QEfficient.utils.run_utils import ApiRunner
# prompt = "Who is your creator? and What all you are allowed to do?"
prompt = """
Billions of years ago, in the vast emptiness of the early universe, tiny fluctuations in the density of matter began to grow under the influence of gravity. Clouds of gas—mostly hydrogen and helium—started to collapse, forming the first stars. These stars grouped together, bound by gravity, creating the earliest galaxies.
Over time, these galaxies merged, collided, and evolved, shaping their spiral arms, elliptical forms, or irregular structures. Within their swirling depths, stars were born and died, enriching the galactic gas with heavier elements. These elements became the building blocks for planets, moons, and eventually life.
Thus, from the quiet whispers of cosmic dust, a galaxy emerged—an island of stars, nebulae, and mysteries, drifting through the infinite sea of space.
As the galaxy matured, its stars danced in intricate orbits, weaving patterns shaped by gravity and time. Supernovae exploded like cosmic fireworks, scattering elements across space and triggering new waves of star formation. Black holes formed at the hearts of galaxies, anchoring their structure and influencing their evolution. Over billions of years, the galaxy became a dynamic ecosystem—where stars are born, live, and die—each cycle adding to the richness of the cosmic tapestry.
"""
torch.manual_seed(42)
model_id = "/home/ochougul/open_source/efficient-transformers/gpt-oss-20b-weights-converted"  # See Comments above to convert saftensors to BF16
model_name = "openai/gpt-oss-20b"
config = AutoConfig.from_pretrained(model_name)
config.num_hidden_layers=2
if hasattr(config, "quantization_config"):
    delattr(config, "quantization_config")

# model = GptOssForCausalLM.from_pretrained(
#     model_id, torch_dtype=torch.float32, attn_implementation="eager", config=config
# )
tokenizer = load_hf_tokenizer(pretrained_model_name_or_path=model_name)
# model.eval()

# Run prefill
_prefill_seq_len = 8192
_ctx_len = 8192
inputs = tokenizer(prompt, return_tensors="np", padding=True)
position_ids = inputs["attention_mask"].sum(1, keepdims=True)
padded_len = inputs["input_ids"].shape[1]
num_chunks = -(padded_len // -_prefill_seq_len)  # ceil divide without float
padded_len = num_chunks * _prefill_seq_len  # Convert to a multiple of prompt_len

# Initialize variables specific to request
# Calculate the max generation length.
max_gen_len = _ctx_len - position_ids.max()
generation_len = 1



inputs = tokenizer(prompt, return_tensors="np", padding="max_length", max_length=padded_len)
inputs["position_ids"] = np.where(inputs.pop("attention_mask"), np.arange(padded_len), -1)
inputs.pop("token_type_ids", None)
# import ipdb; ipdb.set_trace()
# inputs = {k: torch.from_numpy(v).to(model.device) for k, v in inputs.items()}
# cache = HybridCache(config=config, batch_size=1, max_cache_len=8192)
# out = model(**tokenizer(prompt, return_tensors="pt"), past_key_values=cache)

# config = model.config
batch_size = len(Constants.INPUT_STR)

# api_runner = ApiRunner(batch_size, tokenizer, config, Constants.INPUT_STR, Constants.PROMPT_LEN, Constants.CTX_LEN)

# qeff_model = QEFFAutoModelForCausalLM(model, continuous_batching=False)
past_key_values = []
for i in range(config.num_hidden_layers):
    cache_len = 128 if i%2==0 else 8192
    pad_shape = (1, 8, cache_len, 64)
    past_key = torch.zeros((pad_shape), dtype=torch.float32)
    past_value = torch.zeros((pad_shape), dtype=torch.float32)
    pkv = (past_key, past_value)
    past_key_values.append(pkv)
inputs['past_key_values'] = past_key_values
# import ipdb; ipdb.set_trace()
# qeff_out = qeff_model.model(**inputs)
# # exit()
# onnx_model_path = qeff_model.export()
qpc_path = "/home/ochougul/open_source/efficient-transformers/SUBFUNC_CACHE/GptOssForCausalLM-e0c9bd83cb777717/qpc-98c289ed018d7ccc/qpc"
# qpc_path = "/home/ochougul/.cache/qeff_models/GptOssForCausalLM-869fb382739abc84/qpc-75f8677bd5235045/qpc"
# qpc_path = qeff_model.compile(
#     prefill_seq_len=8192,
#     ctx_len=8192,
#     num_cores=16,
#     # mxfp6_matmul=True,
#     # mxint8_kv_cache=True,
#     num_devices=1,
#     mos=1,
#     aic_enable_depth_first=True,
#     num_speculative_tokens=None,
#     prefill_only=True,
#     # mdts_mos=1
# )
st = time.time()
session = QAICInferenceSession(qpc_path)
print(f"Activation time = {time.time()-st}")
print(f"qpc path is {qpc_path}")
logits_out_placeholder = np.zeros((1, 1, 201088), dtype=np.float32)
session.set_buffers({"logits": logits_out_placeholder, 
                    #  "past_key.0_RetainedState": np.zeros((1, 8, 128, 64), dtype=np.float32),
                    #  "past_value.0_RetainedState": np.zeros((1, 8, 128, 64), dtype=np.float32),
                    #  "past_key.1_RetainedState": np.zeros((1, 8, 16384, 64), dtype=np.float32),
                    #  "past_value.1_RetainedState": np.zeros((1, 8, 16384, 64), dtype=np.float32)
                     }
                    )


inputs.pop("past_key_values")
# inputs = {k:v.detach().numpy() for k, v in inputs.items()}
# inputs["past_key.0"]= np.zeros((1, 8, 128, 64), dtype=np.float32)
# inputs["past_value.0"] =  np.zeros((1, 8, 128, 64), dtype=np.float32)
# inputs["past_key.1"] = np.zeros((1, 8, 16384, 64), dtype=np.float32)
# inputs["past_value.1"] =  np.zeros((1, 8, 16384, 64), dtype=np.float32)
st = time.time()
qpc_out = session.run(inputs)
print(f"qpc run time is {time.time()-st}")
# import ipdb; ipdb.set_trace()
from QEfficient.generation.cloud_infer import QAICInferenceSession
import torch
from transformers import AutoModel, AutoTokenizer, AutoConfig
from QEfficient import QEFFAutoModel
import numpy as np
from QEfficient.diffusers.models.blocking_configurator import build_transformer_blocking_config

model_path = "Dream-org/Dream-v0-Instruct-7B"
config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
# config.num_hidden_layers = 1
compile_length = 1000
config.max_position_embeddings = compile_length

model = QEFFAutoModel.from_pretrained(model_path, torch_dtype=torch.float32, trust_remote_code=True, config = config)
# print(model.config)

config = build_transformer_blocking_config(
    model_config=model.config,
    specializations={
        "batch_size": 1,
        "cl": 32000,
        "ctx_len": 32000
    },
    compile_config={
        "mdp_ts_num_devices": 16,
        "aic_num_cores": 16,
        "convert_to_fp16": True
    },
    blocking_mode="hqkv"
)
print(config)

# -----------------------------------------------------------------------------
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------

import json
import os
import types
from typing import Optional
from unittest import result
import torch
# from QEfficient.transformers.models.pytorch_transforms import Embedding_Transform
from QEfficient.utils import hf_download
from huggingface_hub import hf_hub_download

def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output  # First element of model_output contains all token embeddings
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

def average_pool(last_hidden_states: torch.Tensor,
                 attention_mask: torch.Tensor) -> torch.Tensor:
    last_hidden = last_hidden_states[0].masked_fill(~attention_mask[..., None].bool(), 0.0)
    return last_hidden.sum(dim=1) / attention_mask.sum(dim=1)[..., None]

def cls_pooling(token_embeddings,attention_mask):
    return token_embeddings[:, 0]

def max_pooling(token_embeddings,attention_mask):
    return torch.max(token_embeddings, 1)[0]

def min_pooling(token_embeddings,attention_mask):
    return torch.min(token_embeddings, 1)[0]

POOLING_MAP={
    "mean":mean_pooling,
    "avg":average_pool,
    "cls":cls_pooling,
    "max":max_pooling,
    "min":min_pooling,
}
# def define_pooling(modules_json_path):
#     dir_name=os.path.join(os.path.dirname(modules_json_path),"1_Pooling/config.json")
#     if os.path.exists(dir_name):
#         with open(dir_name) as fIn:
#             pooling_config = json.load(fIn)
#             pooling=[POOLING_MAP[k] for k,v in pooling_config.items() if v is True and k in POOLING_MAP]
#             return pooling    
#     else:
#         print("Pooling config not found")
#         return None

# def get_modules(repo_id):
#     modules_json_path=hf_download(repo_id,filename="modules.json")
#     with open(modules_json_path) as fIn:
#         modules_json = json.load(fIn)
#     for module in modules_json:
#         if module["type"] == "Pooling":
#             pooling=define_pooling(modules_json_path)
            
import torch.nn as nn

class PooledModel(nn.Module):
    def __init__(self, base_model, pooling_fn):
        super().__init__()
        self.config=base_model.config
        self.base_model = base_model
        self.pooling_fn = pooling_fn

    def forward(self, input_ids: Optional[torch.Tensor] = None, attention_mask: Optional[torch.Tensor] = None, **kwargs):
        output = self.base_model(input_ids, attention_mask, **kwargs)
        return self.pooling_fn(output, attention_mask)

 
def patch_model_with_pooling(model, pooling):
    def custom_forward(self, *args, **kwargs):
        output=self.base_forward(*args, **kwargs)
        return pooling[0](output, kwargs['attention_mask'])
    
    model.base_forward=model.forward
    model.forward=types.MethodType(custom_forward, model)

                
            
            
                                  
                
                
            
            
            
    # modules = {}
    # for module in modules_json:
    #     modules[module["name"]] = module["type"]
    # return modules




def get_modules_json_path(model_name_or_path):
    if os.path.isdir(model_name_or_path):
       # It's a local path
       local_json_path = os.path.join(model_name_or_path, "module.json")
       if os.path.isfile(local_json_path):
           return local_json_path   
    else:
       # It's a Hugging Face model ID
        try:
           json_path = hf_hub_download(repo_id=model_name_or_path, filename="modules.json")
           return json_path
        except Exception as e:
           print(f"Error: {e}")
           return None
    
# def embedding_transform_temp(func):
#     def wrapper(self,model, **kwargs):
#         model_name_or_path = kwargs['pretrained_model_name_or_path']
#         modules_json_path = get_modules_json_path(model_name_or_path)
#         if modules_json_path is not None:
#             with open(modules_json_path) as fIn:
#                 modules_json = json.load(fIn)
#                 for module in modules_json:
#                     if "Pooling" in module["type"]:
#                         pooling=average_pool
#                         model=PooledModel(args[1],pooling)
#         result = func(, **kwargs)
#         return result
#     return wrapper


def embedding_transform(func):
    def wrapper(self,model, **kwargs):
        if kwargs.get('pooling') is not None:
            pooling=kwargs['pooling']
            pooling_method=POOLING_MAP[pooling]
            model=PooledModel(model,pooling_method)
        result = func(self,model, **kwargs)
        return result
    return wrapper

# def embedding_transform_temp(func):
#     def wrapper(self,model, **kwargs):
#         model_name_or_path = kwargs['pretrained_model_name_or_path']
#         modules_json_path = get_modules_json_path(model_name_or_path)
#         if modules_json_path is not None:
#             with open(modules_json_path) as fIn:
#                 modules_json = json.load(fIn)
#                 for module in modules_json:
#                     if "Pooling" in module["type"]:
#                         pooling=average_pool
#                         model=PooledModel(args[1],pooling)
#         result = func(, **kwargs)
#         return result
#     return wrapper
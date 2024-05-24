import os
import pytest
import QEfficient
import json
from typing import List

from QEfficient.utils.constants import Constants, QEFF_MODELS_DIR

class model_setup:
    def __init__(self,model_name: str,
    num_cores: int,
    prompt: str,
    aic_enable_depth_first: bool = False,
    mos: int = -1,
    cache_dir: str = Constants.CACHE_DIR,
    hf_token: str = None,
    batch_size: int = 1,
    prompt_len: int = 32,
    ctx_len: int = 128,
    mxfp6: bool = False,
    device_group: List[int] = [
        0,
    ]):
        self.model_name = model_name
        self.num_cores = num_cores
        self.prompt = prompt
        self.aic_enable_depth_first = aic_enable_depth_first
        self.mos = mos
        self.cache_dir = cache_dir
        self.hf_token = hf_token
        self.batch_size = batch_size
        self.prompt_len = prompt_len
        self.ctx_len = ctx_len
        self.mxfp6 = mxfp6
        self.device_group = device_group

    def model_card_dir(self):
        return str(os.path.join(QEFF_MODELS_DIR, str(self.model_name)))
    
    def qpc_base_dir_name(self):
        return( f"qpc_{self.num_cores}cores_{self.batch_size}BS_{self.prompt_len}PL_{self.ctx_len}CL_"
        + f"{len(self.device_group)}"
        + "devices"
        + ("_mxfp6" if self.mxfp6 else "_fp16"))
    
    def qpc_dir_path(self):
        return str(os.path.join(self.model_card_dir(), self.qpc_base_dir_name(), "qpcs"))
    
    def onnx_dir_path(self):
        return str(os.path.join(self.model_card_dir(), "onnx"))
    
    def onnx_model_path(self):
        return str(os.path.join(self.onnx_dir_path(), self.model_name.replace("/", "_") + "_kv_clipped_fp16.onnx"))
    
    def model_hf_path(self):
        return str(os.path.join(self.cache_dir,self.model_name))
    
    def tokenizer(self):
        pass
    
    def base_path_and_generated_onnx_path(self):
        return str(self.onnx_dir_path()), str(os.path.join(self.onnx_dir_path(), self.model_name.replace("/", "_") + "_kv_clipped_fp16.onnx"))
    
    def generated_qpc_path(self):
        return self.qpc_dir_path()
    
    def specialization_json_path(self):
        return str(os.path.join(self.model_card_dir(), self.qpc_base_dir_name(), "specializations.json"))
    
    def custom_io_file_path(self):
        return str(os.path.join(self.onnx_dir_path(), "custom_io.yaml"))

@pytest.fixture
def setup(model_name,num_cores,prompt,aic_enable_depth_first,mos,cache_dir,hf_token,batch_size,prompt_len,ctx_len,mxfp6,device_group):
    return model_setup(model_name,num_cores,prompt,aic_enable_depth_first,mos,cache_dir,hf_token,batch_size,prompt_len,ctx_len,mxfp6,device_group)

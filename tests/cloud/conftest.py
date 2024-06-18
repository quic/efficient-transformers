# -----------------------------------------------------------------------------
#
# Copyright (c)  2023-2024 Qualcomm Innovation Center, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------
import os
import shutil
import pytest
import QEfficient
import json
from typing import List
from QEfficient.utils.constants import Constants, QEFF_MODELS_DIR, ROOT_DIR
from QEfficient.utils import get_qpc_dir_name_infer

class model_setup:
    """
    model_setup is a set up class for all the High Level testing script, 
    which provides all neccessary objects needed for checking the flow and creation 
    of the HL API code.
    """
    def __init__(self,model_name,num_cores,prompt,prompts_txt_file_path,aic_enable_depth_first,mos,cache_dir,hf_token,batch_size,prompt_len,ctx_len,mxfp6,mxint8,device_group):
        """
        Initialization set up
        ------
        param: model_name: str
        param: num_cores: int
        param: prompt: str
        param: prompts_txt_file_path: str
        param: aic_enable_depth_first: bool 
        param: mos: int
        param: cache_dir: str 
        param: hf_token: str 
        param: batch_size: int
        param: prompt_len: int 
        param: ctx_len: int 
        param: mxfp6: bool 
        param: mxint8: bool
        param: device_group: List[int] 
        """
        self.model_name = model_name
        self.num_cores = num_cores
        self.prompt = prompt
        self.prompts_txt_file_path = os.path.join(ROOT_DIR,prompts_txt_file_path) if prompts_txt_file_path is not None  else None
        self.aic_enable_depth_first = aic_enable_depth_first
        self.mos = mos
        self.cache_dir = cache_dir
        self.hf_token = hf_token
        self.batch_size = batch_size
        self.prompt_len = prompt_len
        self.ctx_len = ctx_len
        self.mxfp6 = mxfp6
        self.mxint8 = mxint8
        self.device_group = device_group

    def model_card_dir(self):
        return str(os.path.join(QEFF_MODELS_DIR, str(self.model_name)))
    
    def qpc_base_dir_name(self):
        return get_qpc_dir_name_infer(self.num_cores, self.mos, self.batch_size, self.prompt_len, self.ctx_len, self.mxfp6, self.mxint8, self.device_group)
    
    def qpc_dir_path(self):
        return str(os.path.join(self.model_card_dir(), self.qpc_base_dir_name(), "qpcs"))
    
    def onnx_dir_path(self):
        return str(os.path.join(self.model_card_dir(), "onnx"))
    
    def onnx_model_path(self):
        return str(os.path.join(self.onnx_dir_path(), self.model_name.replace("/", "_") + "_kv_clipped_fp16.onnx"))
    
    def model_hf_path(self):
        return str(os.path.join(self.cache_dir,self.model_name))
    
    def base_path_and_generated_onnx_path(self):
        return str(self.onnx_dir_path()), str(os.path.join(self.onnx_dir_path(), self.model_name.replace("/", "_") + "_kv_clipped_fp16.onnx"))
    
    def specialization_json_path(self):
        return str(os.path.join(self.model_card_dir(), self.qpc_base_dir_name(), "specializations.json"))
    
    def custom_io_file_path(self):
        if self.mxint8:
            return str(os.path.join(self.onnx_dir_path(), "custom_io_int8.yaml"))
        else:
            return str(os.path.join(self.onnx_dir_path(), "custom_io_fp16.yaml"))

@pytest.fixture
def setup(model_name,num_cores,prompt,prompts_txt_file_path,aic_enable_depth_first,mos,cache_dir,hf_token,batch_size,prompt_len,ctx_len,mxfp6,mxint8,device_group):
    """
    It is a fixture or shared object of all testing script within or inner folder,
    Args are coming from the dynamically generated tests method i.e, pytest_generate_tests via testing script or method
    --------
    Args: same as set up initialization
    Return: model_setup class object
    """
    yield model_setup(model_name,num_cores,prompt,prompts_txt_file_path,bool(aic_enable_depth_first),mos,cache_dir,hf_token,batch_size,prompt_len,ctx_len,bool(mxfp6),bool(mxint8),device_group)

def pytest_generate_tests(metafunc):  
    """
    pytest_generate_tests hook is used to create our own input parametrization,
    It generates all the test cases of different combination of input parameters which are read from the json file,
    and passed to each testing script module.
    -----------
    Ref: https://docs.pytest.org/en/7.3.x/how-to/parametrize.html
    """
    json_data = None
    json_file  = os.path.join(ROOT_DIR,"tests","cloud","HL_testing_input.json")
    with open(json_file,'r') as file:
        json_data =  json.load(file)
    print("\n**************JSON data***************\n\n",json_data)

    if ('model_name' in metafunc.fixturenames and 'num_cores' in metafunc.fixturenames and 'prompt' in metafunc.fixturenames
        and 'prompts_txt_file_path' in metafunc.fixturenames
        and 'aic_enable_depth_first' in metafunc.fixturenames and 'mos' in metafunc.fixturenames and 'cache_dir' in metafunc.fixturenames 
        and 'hf_token' in metafunc.fixturenames and 'batch_size' in metafunc.fixturenames and 'prompt_len' in metafunc.fixturenames 
        and 'ctx_len' in metafunc.fixturenames and 'mxfp6' in metafunc.fixturenames and 'mxint8' in metafunc.fixturenames
        and 'device_group' in metafunc.fixturenames):

        metafunc.parametrize("model_name", json_data['model_name'])
        metafunc.parametrize("num_cores", json_data['num_cores'])
        metafunc.parametrize("prompt",json_data['prompt'])
        metafunc.parametrize("prompts_txt_file_path",json_data['prompts_txt_file_path'])
        metafunc.parametrize("aic_enable_depth_first",json_data['aic_enable_depth_first'])
        metafunc.parametrize("mos",json_data['mos'])
        metafunc.parametrize("cache_dir",[Constants.CACHE_DIR])
        metafunc.parametrize("hf_token",json_data['hf_token'])
        metafunc.parametrize("batch_size",json_data['batch_size'])
        metafunc.parametrize("prompt_len",json_data['prompt_len'])
        metafunc.parametrize("ctx_len",json_data['ctx_len'])
        metafunc.parametrize("mxfp6",json_data['mxfp6'])
        metafunc.parametrize("mxint8",json_data['mxint8'])
        metafunc.parametrize("device_group",json_data['device_group'])
  
def pytest_collection_modifyitems(items):
    """
    pytest_collection_modifyitems is pytest a hook,
    which is used to re-order the execution order of the testing script/methods 
    with various combination of inputs. 
    called after collection has been performed, may filter or re-order the items in-place.
    Parameters:	
    items (List[_pytest.nodes.Item]) list of item objects
    ----------
    Ref: https://docs.pytest.org/en/4.6.x/reference.html#collection-hooks
    """
    print("\n*************Initial Test script/functions execution order****************\n\n",items)
    if len(items)>=4:
        run_first = ["test_export","test_compile","test_execute","test_infer"]
        num_tests = len(items)
        modules = {item: item.module.__name__ for item in items}
        items[:] = sorted(items, key=lambda x: run_first.index(modules[x]) if modules[x] in run_first else len(items))
        non_cloud_tests = []
        for itm in items:
            if modules[itm] not in run_first:
                non_cloud_tests.append(itm)
        
        num_cloud_tests = len(items) - len(non_cloud_tests)
        num_cloud_test_cases = num_cloud_tests//len(run_first)
        final_items = []
        for i in range(num_cloud_test_cases):
            for j in range(len(run_first)):
                final_items.append(items[i+j*num_cloud_test_cases])
        
        final_items.insert(0,final_items[3])
        final_items.extend(non_cloud_tests)
        items[:] = final_items
        print("\n*************Final Test script/functions execution order****************\n\n",items)

@pytest.fixture
def clean_up_after_test():
    yield
    if os.path.exists(QEFF_MODELS_DIR):
        shutil.rmtree(QEFF_MODELS_DIR)
        print(f'\n...............Cleaned up {QEFF_MODELS_DIR}')
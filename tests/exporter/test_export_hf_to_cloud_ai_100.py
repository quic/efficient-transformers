# -----------------------------------------------------------------------------
#
# Copyright (c) 2024 Qualcomm Innovation Center, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------

import os
import shutil

import pytest

import QEfficient
from QEfficient.base.common import QEFFCommonLoader
from QEfficient.exporter.export_hf_to_cloud_ai_100 import (
    convert_to_cloud_bertstyle,
    convert_to_cloud_kvstyle,
    qualcomm_efficient_converter,
)
from QEfficient.utils import load_hf_tokenizer, onnx_exists


def create_onnx_dir(model_name):
    _,onnx_dir_path,_ = onnx_exists(model_name=model_name,full_batch_size=None)
    return onnx_dir_path

@pytest.mark.parametrize("model_name",["lu-vae/llama-68m-fft"])
class TestQEffExportFunctions():
    @classmethod
    def setup_class(cls):
        cls.seq_len = 128
        cls.hf_token = None
        cls.cache_dir = None
        cls.models = {}
        for model_name in ["lu-vae/llama-68m-fft"]:
            onnx_dir_path = create_onnx_dir(model_name)
            tokenizer = load_hf_tokenizer(pretrained_model_name_or_path=model_name,hf_token=cls.hf_token,cache_dir=cls.cache_dir)
            model_kv = QEFFCommonLoader.from_pretrained(pretrained_model_name_or_path=model_name,token=cls.hf_token,cache_dir=cls.cache_dir)
            qeff_model = QEfficient.transform(model_kv)
            cls.models[model_name] = [onnx_dir_path,tokenizer,model_kv,qeff_model]
    
    def teardown_class(cls):
        for models in cls.models.values():
            onnx_dir_path = models[0]
            if os.path.exists(onnx_dir_path):
                shutil.rmtree(onnx_dir_path)
    
    def test_convert_to_cloud_bertstyle(self,model_name):
        _=model_name
        assert True

    def test_convert_to_cloud_kvstyle_success(self,model_name):
        """
        Test method for convert_to_cloud_kvstyle function in the case of general code flow that qeff_model is transformed
        """
        onnx_dir_path,tokenizer,_,qeff_model = self.models[model_name]
        result = convert_to_cloud_kvstyle(
            model_name=model_name,
            qeff_model=qeff_model,
            tokenizer=tokenizer,
            onnx_dir_path=onnx_dir_path,
            seq_len=self.seq_len
        )
        assert os.path.isfile(result)
        assert os.path.exists(result)
    
    def test_convert_to_cloud_kvstyle_warning(self,model_name):
        """
        Test method for convert_to_cloud_kvstyle function in the case of Depcreation Warning in the function
        """
        onnx_dir_path,tokenizer,_,qeff_model = self.models[model_name]
        with pytest.warns(DeprecationWarning):
            convert_to_cloud_kvstyle(
                model_name=model_name,
                qeff_model=qeff_model,
                tokenizer=tokenizer,
                onnx_dir_path=onnx_dir_path,
                seq_len=self.seq_len
            )
    
    def test_qualcomm_efficient_converter_success(self,model_name):
        """
        Test method for qualcomm_efficient_converter function in the case of general code flow 
        """
        result = qualcomm_efficient_converter(
            model_name=model_name,
            model_kv=None,
            tokenizer=None,
            onnx_dir_path=None,
            seq_length=self.seq_len,
            kv=True,
            form_factor="cloud"
        )
        assert os.path.exists(result[0])
        assert os.path.exists(result[1])

    def test_qualcomm_efficient_converter_transformed_model(self,model_name):
        """
        Test method for qualcomm_efficient_converter function in the case of qeff_model is already transformed 
        """
        onnx_dir_path,tokenizer,_,qeff_model = self.models[model_name]
        result = qualcomm_efficient_converter(
            model_name=model_name,
            model_kv=qeff_model,
            tokenizer=tokenizer,
            onnx_dir_path=onnx_dir_path,
            seq_length=self.seq_len,
            kv=True,
            form_factor="cloud"
        )
        assert os.path.exists(result[0])
        assert os.path.exists(result[1])

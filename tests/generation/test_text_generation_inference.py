# -----------------------------------------------------------------------------
#
# Copyright (c) 2024 Qualcomm Innovation Center, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------

import os
import shutil
from typing import Dict, List, Optional

import pytest

import QEfficient
from QEfficient.cloud.export import get_onnx_model_path
from QEfficient.generation.text_generation_inference import CloudAI100ExecInfo, cloud_ai_100_exec_kv
from QEfficient.utils import check_and_assign_cache_dir, get_qpc_dir_path, load_hf_tokenizer


def get_qpc(model_name, num_cores, aic_enable_depth_first,mos, batch_size,full_batch_size, prompt_len, ctx_len, mxfp6, mxint8, device_group,local_model_dir, cache_dir,hf_token):
    cache_dir = check_and_assign_cache_dir(local_model_dir, cache_dir)
    tokenizer = load_hf_tokenizer(
        pretrained_model_name_or_path=(local_model_dir if local_model_dir else model_name),
        cache_dir=cache_dir,
        hf_token=hf_token,
    )

    qpc_dir_path = get_qpc_dir_path(
        model_name, num_cores, mos, batch_size, prompt_len, ctx_len, mxfp6, mxint8, device_group, full_batch_size
    )

    onnx_model_path = get_onnx_model_path(
            model_name, cache_dir, tokenizer, hf_token, local_model_dir, full_batch_size
        )  # , base_dir_name)
    
    _ = QEfficient.compile(
        onnx_path=onnx_model_path,
        qpc_path=os.path.dirname(
            qpc_dir_path
        ),  # We need to pass parent directory of qpc_dir_path, as the compile function handles the qpcs directory creation
        num_cores=num_cores,
        batch_size=batch_size,
        prompt_len=prompt_len,
        ctx_len=ctx_len,
        mxfp6=mxfp6,
        mxint8=mxint8,
        aic_enable_depth_first=aic_enable_depth_first,
        mos=mos,
        device_group=device_group,
        full_batch_size=full_batch_size,
    )
    return tokenizer,onnx_model_path,qpc_dir_path

@pytest.mark.parametrize("model_name",["lu-vae/llama-68m-fft"])
class TestQEffInferenceFunctions():
    @classmethod
    def setup_method(cls):
        cls.num_cores: int = 14
        cls.device_group: Optional[List[int]] = None
        cls.prompt: Optional[str] = None  # type: ignore
        cls.prompts_txt_file_path: Optional[str] = "examples/prompts.txt"
        cls.aic_enable_depth_first: bool = False
        cls.mos: int = -1
        cls.batch_size: int = 1
        cls.full_batch_size: Optional[int] = None
        cls.prompt_len: int = 32
        cls.ctx_len: int = 128
        cls.generation_len: Optional[int] = None
        cls.mxfp6: bool = False
        cls.mxint8: bool = False
        cls.local_model_dir: Optional[str] = None
        cls.cache_dir: Optional[str] = None
        cls.hf_token: Optional[str] = None
        cls.models = []
    
    def teardown_class(cls):
        for dir in cls.models:
            if os.path.exists(os.path.dirname(dir)):
                shutil.rmtree(os.path.dirname(dir))
    
    def test_cloud_ai_100_exec_kv_batch_size_1(self,model_name):
        tokenizer,onnx_model_path,qpc_dir_path = get_qpc(model_name, self.num_cores, self.aic_enable_depth_first,self.mos, self.batch_size,self.full_batch_size, self.prompt_len, self.ctx_len, self.mxfp6, self.mxint8, self.device_group,self.local_model_dir, self.cache_dir,self.hf_token)
        result = cloud_ai_100_exec_kv(
                    tokenizer=tokenizer,
                    qpc_path=qpc_dir_path,
                    device_id=self.device_group,
                    prompt=self.prompt,
                    prompts_txt_file_path=self.prompts_txt_file_path,
                    generation_len=self.generation_len,
                    full_batch_size=self.full_batch_size,
                )
        self.models.extend([onnx_model_path,qpc_dir_path])
        assert isinstance(result, CloudAI100ExecInfo)

    def test_cloud_ai_100_exec_kv_full_batch_size(self,model_name):
        self.full_batch_size = 3
        tokenizer,onnx_model_path,qpc_dir_path = get_qpc(model_name, self.num_cores, self.aic_enable_depth_first,self.mos, self.batch_size,self.full_batch_size, self.prompt_len, self.ctx_len, self.mxfp6, self.mxint8, self.device_group,self.local_model_dir, self.cache_dir,self.hf_token)
        result = cloud_ai_100_exec_kv(
                    tokenizer=tokenizer,
                    qpc_path=qpc_dir_path,
                    device_id=self.device_group,
                    prompt=self.prompt,
                    prompts_txt_file_path=self.prompts_txt_file_path,
                    generation_len=self.generation_len,
                    full_batch_size=self.full_batch_size,
                )
        self.models.extend([onnx_model_path,qpc_dir_path])
        assert isinstance(result, CloudAI100ExecInfo)

    def test_latency_stats_bertstyle(self,model_name):
        # Will Implement
        _ = model_name
        assert True

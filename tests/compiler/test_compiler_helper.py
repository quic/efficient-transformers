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
from QEfficient.cloud.export import get_onnx_model_path
from QEfficient.compile.compile_helper import compile
from QEfficient.utils import check_and_assign_cache_dir, get_qpc_dir_path


@pytest.mark.parametrize("model_name",["lu-vae/llama-68m-fft"])
class TestQEffCompileFunctions():
    @classmethod
    def setup_method(cls):
        cls.num_cores: int = 16
        cls.device_group = [0] #  FIXME: use num_devices instead
        cls.aic_enable_depth_first: bool = False
        cls.mos: int = -1
        cls.batch_size: int = 1
        cls.prompt_len: int = 3
        cls.ctx_len: int = 128
        cls.mxfp6: bool = True
        cls.mxint8: bool = False
        cls.seq_len = 128
        cls.hf_token = None
        cls.local_model_dir = None
        cls.cache_dir = None
        cls.custom_io_file_path = None
        cls.full_batch_size = None
        cls.models = {}
        for model_name in ["lu-vae/llama-68m-fft"]:
            cache_dir = check_and_assign_cache_dir(cls.local_model_dir, cls.cache_dir)
            onnx_model_path = get_onnx_model_path(
                                model_name=model_name,
                                cache_dir=cache_dir,
                                hf_token=cls.hf_token,
                                local_model_dir=cls.local_model_dir,
                                full_batch_size=cls.full_batch_size,
                            )
            cls.models[model_name] = [str(onnx_model_path)]

    def teardown_class(cls):
        for model_dirs in cls.models.values():
            for dir in model_dirs:
                if os.path.exists(os.path.dirname(dir)):
                    shutil.rmtree(os.path.dirname(dir))

    def test_compile_for_mxint8(self,model_name):
        self.mxint8 = True
        qpc_dir_path = get_qpc_dir_path(model_name,self.num_cores,self.mos,self.batch_size,self.prompt_len,self.ctx_len,self.mxfp6,self.mxint8,self.device_group,self.full_batch_size)
        result = compile(onnx_path = self.models[model_name][0],
                            qpc_path = os.path.dirname(qpc_dir_path),
                            num_cores = self.num_cores,
                            device_group = self.device_group,  #  FIXME: use num_devices instead
                            aic_enable_depth_first = self.aic_enable_depth_first,
                            mos = self.mos,
                            batch_size = self.batch_size,
                            prompt_len = self.prompt_len,
                            ctx_len = self.ctx_len,
                            mxfp6 = self.mxfp6,
                            mxint8 = self.mxint8,
                            custom_io_file_path = self.custom_io_file_path,
                            full_batch_size = self.full_batch_size)
        assert result == qpc_dir_path
        self.models[model_name].append(qpc_dir_path)

    def test_compile_for_fp16(self,model_name):
        self.mxint8 = False
        qpc_dir_path = get_qpc_dir_path(model_name,self.num_cores,self.mos,self.batch_size,self.prompt_len,self.ctx_len,self.mxfp6,self.mxint8,self.device_group,self.full_batch_size)
        result = compile(onnx_path = self.models[model_name][0],
                            qpc_path = os.path.dirname(qpc_dir_path),
                            num_cores = self.num_cores,
                            device_group = self.device_group,  #  FIXME: use num_devices instead
                            aic_enable_depth_first = self.aic_enable_depth_first,
                            mos = self.mos,
                            batch_size = self.batch_size,
                            prompt_len = self.prompt_len,
                            ctx_len = self.ctx_len,
                            mxfp6 = self.mxfp6,
                            mxint8 = self.mxint8,
                            custom_io_file_path = self.custom_io_file_path,
                            full_batch_size = self.full_batch_size)
        assert result == qpc_dir_path
        self.models[model_name].append(qpc_dir_path)

    def test_compile_for_IO_notFound(self,model_name):
        qpc_dir_path = get_qpc_dir_path(model_name,self.num_cores,self.mos,self.batch_size,self.prompt_len,self.ctx_len,self.mxfp6,self.mxint8,self.device_group,self.full_batch_size)
        with pytest.raises(FileNotFoundError):
            compile(onnx_path = "",
                            qpc_path = os.path.dirname(qpc_dir_path),
                            num_cores = self.num_cores,
                            device_group = self.device_group,  #  FIXME: use num_devices instead
                            aic_enable_depth_first = self.aic_enable_depth_first,
                            mos = self.mos,
                            batch_size = self.batch_size,
                            prompt_len = self.prompt_len,
                            ctx_len = self.ctx_len,
                            mxfp6 = self.mxfp6,
                            mxint8 = self.mxint8,
                            custom_io_file_path = self.custom_io_file_path,
                            full_batch_size = self.full_batch_size)
        self.models[model_name].append(qpc_dir_path)

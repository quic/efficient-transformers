# -----------------------------------------------------------------------------
#
# Copyright (c) 2024 Qualcomm Innovation Center, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------
import os
import shutil

import pytest

from QEfficient.base.common import QEFFCommonLoader
from QEfficient.base.modeling_qeff import QEFFBaseModel
from QEfficient.transformers.transform import transform, transform_lm
from QEfficient.utils import load_hf_tokenizer, onnx_exists


def create_onnx_dir(model_name):
    _, onnx_dir_path, _ = onnx_exists(model_name=model_name, full_batch_size=None)
    return onnx_dir_path


@pytest.mark.parametrize("model_name", ["lu-vae/llama-68m-fft"])
class TestTransformFunctions:
    @classmethod
    def setup_method(cls):
        cls.seq_len = 128
        cls.hf_token = None
        cls.cache_dir = None
        cls.models = {}
        for model_name in ["lu-vae/llama-68m-fft"]:
            onnx_dir_path = create_onnx_dir(model_name)
            tokenizer = load_hf_tokenizer(
                pretrained_model_name_or_path=model_name, hf_token=cls.hf_token, cache_dir=cls.cache_dir
            )
            model_kv = QEFFCommonLoader.from_pretrained(
                pretrained_model_name_or_path=model_name, token=cls.hf_token, cache_dir=cls.cache_dir
            )
            cls.models[model_name] = [onnx_dir_path, tokenizer, model_kv]

    def teardown_class(cls):
        for models in cls.models.values():
            onnx_dir_path = models[0]
            if os.path.exists(onnx_dir_path):
                shutil.rmtree(onnx_dir_path)

    def test_transform_success(self, model_name):
        _, _, model = self.models[model_name]
        result = transform(model)
        assert isinstance(result, QEFFBaseModel)

    def test_transform_invalid_form_factor(self, model_name):
        _, _, model = self.models[model_name]
        with pytest.raises(AssertionError):
            transform(model, "edge")

    def test_transform_not_implemented(self, model_name):
        _, _, model = self.models[model_name]
        with pytest.raises(NotImplementedError):
            transform("not_edge")

    def test_transform_lm(self, model_name):
        _, _, model = self.models[model_name]
        result = transform_lm(model.model)
        assert getattr(result, "qeff_transformed", False)

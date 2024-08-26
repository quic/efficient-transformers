# -----------------------------------------------------------------------------
#
# Copyright (c) 2024 Qualcomm Innovation Center, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------

import pytest

from QEfficient.utils.device_utils import get_available_device_id
from tests.utils import get_cloud_ai_100_tokens, set_up

test_models = [
    "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    "gpt2",
    "Salesforce/codegen-350M-mono",
    "microsoft/phi-2",
    "microsoft/Phi-3-mini-4k-instruct",
    "tiiuae/falcon-7b",
    "Qwen/Qwen2-0.5B",
    "bigcode/starcoder2-3b",
    "Felladrin/Minueza-32M-Base",
    "wtang06/mpt-125m-c4",
    "hakurei/gpt-j-random-tinier",
    "mistralai/Mixtral-8x7B-Instruct-v0.1",
]


@pytest.mark.causal_lm
@pytest.mark.parametrize("model_name", test_models)
class TestQEfficientModels:
    def setup_class(cls):
        """
        Set up function to set up the test environment for TestQEfficientModels class
        :param cls
        """
        cls.setup_infos = {model_name: set_up({"model_name": model_name}) for model_name in test_models}

    @pytest.mark.xdist_group(name="causal_lm_group")
    def test_qefficient_model_torch(self, model_name):
        """
        Test function to validate the model before and after KV changes on Pytorch
        :param model_name: Name of model.
        """
        assert (
            self.setup_infos[model_name]["pytorch_hf_tokens"] == self.setup_infos[model_name]["pytorch_kv_tokens"]
        ).all(), "Tokens don't match for HF PyTorch model output and KV PyTorch model output"

    @pytest.mark.xdist_group(name="causal_lm_group")
    def test_qefficient_model_onnx(self, model_name):
        """
        Test function to validate the model before and after KV changes on ONNXRT
        :param model_name: Name of model.
        """
        assert (
            self.setup_infos[model_name]["pytorch_kv_tokens"] == self.setup_infos[model_name]["ort_tokens"]
        ).all(), "Tokens don't match for ONNXRT output and PyTorch output."

    @pytest.mark.xdist_group(name="causal_lm_group")
    @pytest.mark.skipif(not get_available_device_id, reason="No available devices to run model on Cloud AI 100")
    def test_qefficient_model_cloud_ai_100(self, model_name):
        """
        Test function to validate the model before and after KV changes on Cloud AI 100
        :param model_name: Name of model.
        """

        cloud_ai_100_tokens = get_cloud_ai_100_tokens(self.setup_infos[model_name])
        assert (
            self.setup_infos[model_name]["ort_tokens"] == cloud_ai_100_tokens
        ).all(), "Tokens don't match for ONNXRT output and Cloud AI 100 output."

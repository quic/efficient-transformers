# -----------------------------------------------------------------------------
#
# Copyright (c)  2023-2024 Qualcomm Innovation Center, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------

import unittest

import pytest
import transformers
from transformers.models.gptj.modeling_gptj import GPTJForCausalLM

from QEfficient.utils.constants import Constants
from QEfficient.utils.device_utils import get_available_device_id
from tests.utils import get_cloud_ai_100_tokens, set_up


def get_config():
    """
    Function to get config info from transformers.AutoConfig
    :param: None
    :return model_config - Dict
    """
    model_config = {}
    model_config["model_name"] = "hakurei/gpt-j-random-tinier"
    config = transformers.AutoConfig.from_pretrained(model_config["model_name"])
    n_heads = config.n_head
    d_head = config.n_embd // n_heads
    model_config["model_class"] = GPTJForCausalLM
    model_config["n_layer"] = config.n_layer
    model_config["padding_shape"] = [1, n_heads, Constants.CTX_LEN, d_head]
    return model_config


class TestQEfficientGPTJ(unittest.TestCase):
    @classmethod
    def setUpClass(self):
        """
        Set up function to set up the test environment for TestQEfficientGPTJ class
        :param None
        """
        self.model_config = get_config()
        self.setup_info = set_up(self.model_config)

    def test_qefficient_gptj_torch(self):
        """
        Test function to validate the gptj model before and after KV changes on Pytorch
        :param None
        """
        assert (
            self.setup_info["pytorch_hf_tokens"] == self.setup_info["pytorch_kv_tokens"]
        ).all(), "tokens aren't matching for hf pytorch model output and KV pytorch model output."

    def test_qefficient_gptj_onnx(self):
        """
        Test function to validate the gptj model before and after KV changes on ONNXRT
        :param None
        """
        assert (
            self.setup_info["pytorch_kv_tokens"] == self.setup_info["ort_tokens"]
        ).all(), "tokens aren't matching for onnxrt output and Pytorch output."

    @pytest.mark.skipif(not get_available_device_id, reason="No available devices to run model on Cloud AI 100")
    def test_qefficient_gptj_cloud_ai_100(self):
        """
        Test function to validate the gptj model before and after KV changes on Cloud AI 100
        :param None
        """
        cloud_ai_100_tokens = get_cloud_ai_100_tokens(self.setup_info)
        assert (
            self.setup_info["ort_tokens"] == cloud_ai_100_tokens
        ).all(), "tokens aren't matching for onnxrt output and Cloud AI 100 output."

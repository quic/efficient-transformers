# -----------------------------------------------------------------------------
#
# Copyright (c)  2023-2024 Qualcomm Innovation Center, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------


import unittest
import pytest

import transformers
from transformers.models.mixtral.modeling_mixtral import MixtralForCausalLM

from QEfficient.utils.constants import Constants
from QEfficient.utils.device_utils import get_available_device_id
from tests.utils import get_cloud_ai_100_tokens, set_up, skip_if_mq_not_enabled


def get_config():
    """
    Function to get config info from transformers.AutoConfig
    :param: None
    :return model_config - Dict
    """

    model_config = {}
    model_config["model_name"] = "Open-Orca/Mixtral-SlimOrca-8x7B"
    config = transformers.AutoConfig.from_pretrained(model_config["model_name"])
    config._attn_implementation = "eager"
    n_heads = config.num_attention_heads
    d_head = config.hidden_size // n_heads
    model_config["model_class"] = MixtralForCausalLM
    model_config["n_layer"] = config.num_hidden_layers
    model_config["padding_shape"] = [1, config.num_key_value_heads, Constants.CTX_LEN, d_head]

    return model_config



class TestQEfficientMixtral(unittest.TestCase):
    @classmethod
    def setUpClass(self):
        """
        Set up function to set up the test environment for TestQEfficientOPt class
        :param None
        """

        self.model_config = get_config()
        self.device_group = [0, 1, 2, 3]
        self.setup_info = set_up(self.model_config, self.device_group)

    # TODO: Check this test case @Ann: Working when single model runs
    @pytest.mark.skip(reason="Seeing issue in HF Model, Maybe due to tokenizer")
    def test_qefficient_mixtral_torch(self):
        """
        Test function to validate the mixtral model before and after KV changes on Pytorch
        :param None
        """
        assert (
            self.setup_info["pytorch_hf_tokens"] == self.setup_info["pytorch_kv_tokens"]
        ).all(), "tokens aren't matching for hf pytorch model output and KV pytorch model output."

    def test_qefficient_mixtral_onnx(self):
        """
        Test function to validate the mixtral model before and after KV changes on ONNXRT
        # :param None
        #"""
        assert (
            self.setup_info["pytorch_kv_tokens"] == self.setup_info["ort_tokens"]
        ).all(), "tokens aren't matching for onnxrt output and Pytorch output."

    @pytest.mark.skipif(not get_available_device_id(), reason="No available devices to run model on AIC")
    @skip_if_mq_not_enabled
    def test_qefficient_mixtral_cloud_ai_100(self):
        """
        Test function to validate the mixtral model before and after KV changes on Cloud AI 100
        :param None
        """
        cloud_ai_100_tokens = get_cloud_ai_100_tokens(self.setup_info)
        assert (
            self.setup_info["ort_tokens"] == cloud_ai_100_tokens
        ).all(), "tokens aren't matching for onnxrt output and Cloud AI 100 output."

# -----------------------------------------------------------------------------
#
# Copyright (c)  2023-2024 Qualcomm Innovation Center, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------

import unittest

import pytest
import transformers
from transformers.models.mpt.modeling_mpt import MptForCausalLM

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
    # Provide the base mpt model for the test verification
    model_config["model_name"] = "wtang06/mpt-125m-c4"
    # Above model output is garbage but since it's a small model it is being used for CI tests.
    # "team-lucid/mptk-1b" can be used for accurate tests. One more test can be added for this model and marked as nightly.
    config = transformers.AutoConfig.from_pretrained(model_config["model_name"], trust_remote_code=True)
    n_heads = config.n_heads
    d_head = config.d_model // config.n_heads
    model_config["model_class"] = MptForCausalLM
    model_config["n_layer"] = config.n_layers
    model_config["padding_shape"] = [1, n_heads, Constants.CTX_LEN, d_head]
    return model_config


class TestQEfficientMPT(unittest.TestCase):
    @classmethod
    def setUpClass(self):
        """
        Set up function to set up the test environment for TestQEfficientMPT class
        :param None
        """
        self.model_config = get_config()
        self.setup_info = set_up(self.model_config)

    def test_qefficient_mpt_torch(self):
        """
        Test function to validate the mpt model before and after KV changes on Pytorch
        :param None
        """
        assert (
            self.setup_info["pytorch_hf_tokens"] == self.setup_info["pytorch_kv_tokens"]
        ).all(), "tokens aren't matching for hf pytorch model output and KV pytorch model output."

    def test_qefficient_mpt_onnx(self):
        """
        Test function to validate the mpt model before and after KV changes on ONNXRT
        :param None
        """
        assert (
            self.setup_info["pytorch_kv_tokens"] == self.setup_info["ort_tokens"]
        ).all(), "tokens aren't matching for onnxrt output and Pytorch output."

    @pytest.mark.skipif(not get_available_device_id, reason="No available devices to run model on Cloud AI 100")
    def test_qefficient_mpt_cloud_ai_100(self):
        """
        Test function to validate the mpt model before and after KV changes on Cloud AI 100
        :param None
        """
        cloud_ai_100_tokens = get_cloud_ai_100_tokens(self.setup_info)
        assert (
            self.setup_info["ort_tokens"] == cloud_ai_100_tokens
        ).all(), "tokens aren't matching for onnxrt output and Cloud AI 100 output."

# -----------------------------------------------------------------------------
#
# Copyright (c)  2024 Qualcomm Innovation Center, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------

import json
import os
import unittest

import pytest
from transformers import AutoModelForCausalLM

from QEfficient.utils.constants import ROOT_DIR, Constants
from QEfficient.utils.device_utils import get_available_device_id
from tests.utils import get_cloud_ai_100_tokens, set_up


def get_config(model_config):
    """
    Function to get config info from model_config
    :param model_config: Dict containing model configuration
    :return model_config - Dict
    """
    n_heads = model_config.get("n_head")
    if n_heads is not None:  # Assuming n_head is a key in the config (GPTs/CodeGen)
        d_head = model_config["n_embd"] // n_heads
        model_config["model_class"] = AutoModelForCausalLM
        model_config["padding_shape"] = [1, n_heads, Constants.CTX_LEN, d_head]
    elif model_config.get("num_key_value_heads") is not None:  # Check for num_key_value_heads (Llama/Mistral)
        n_heads = model_config["num_key_value_heads"]
        d_head = model_config["hidden_size"] // model_config["num_attention_heads"]
        model_config["model_class"] = AutoModelForCausalLM
        model_config["padding_shape"] = [1, n_heads, Constants.CTX_LEN, d_head]
    elif model_config.get("n_heads") is not None:  # Check for n_heads and d_model in the config (MPT Model)
        n_heads = model_config["n_heads"]
        d_head = model_config["d_model"] // n_heads
        model_config["model_class"] = AutoModelForCausalLM
        model_config["padding_shape"] = [1, n_heads, Constants.CTX_LEN, d_head]
    else:
        raise ValueError("Invalid model configuration: n_head/n_heads or num_key_value_heads not found.")

    return model_config


class TestQEfficientModels(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """
        Set up function to set up the test environment for TestQEfficientModels class
        """
        cls.model_configs = []
        test_config_file_path = os.path.join(ROOT_DIR, "tests", "config.json")
        with open(test_config_file_path, "r") as f:
            configs = json.load(f)
            for model_config in configs["models"]:
                cls.model_configs.append(get_config(model_config))

        cls.setup_infos = [set_up(model_config) for model_config in cls.model_configs]

    def test_qefficient_model_torch(self):
        """
        Test function to validate the model before and after KV changes on Pytorch
        """
        for setup_info in self.setup_infos:
            assert (
                setup_info["pytorch_hf_tokens"] == setup_info["pytorch_kv_tokens"]
            ).all(), "Tokens don't match for HF PyTorch model output and KV PyTorch model output"

    def test_qefficient_model_onnx(self):
        """
        Test function to validate the model before and after KV changes on ONNXRT
        """
        for setup_info in self.setup_infos:
            assert (
                setup_info["pytorch_kv_tokens"] == setup_info["ort_tokens"]
            ).all(), "Tokens don't match for ONNXRT output and PyTorch output."

    @pytest.mark.skipif(not get_available_device_id, reason="No available devices to run model on Cloud AI 100")
    def test_qefficient_model_cloud_ai_100(self):
        """
        Test function to validate the model before and after KV changes on Cloud AI 100
        """
        cloud_ai_100_tokens_list = [get_cloud_ai_100_tokens(setup_info) for setup_info in self.setup_infos]
        for cloud_ai_100_tokens, setup_info in zip(cloud_ai_100_tokens_list, self.setup_infos):
            assert (
                setup_info["ort_tokens"] == cloud_ai_100_tokens
            ).all(), "Tokens don't match for ONNXRT output and Cloud AI 100 output."


if __name__ == "__main__":
    unittest.main()

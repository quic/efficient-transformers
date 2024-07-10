# -----------------------------------------------------------------------------
#
# Copyright (c) 2024 Qualcomm Innovation Center, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------

import json
import os

import pytest
from transformers import AutoConfig, AutoModelForCausalLM

from QEfficient.utils import get_config
from QEfficient.utils.constants import ROOT_DIR, Constants
from QEfficient.utils.device_utils import get_available_device_id
from tests.utils import get_cloud_ai_100_tokens, set_up

TEST_CONFIG_FILE_PATH = os.path.join(ROOT_DIR, "tests", "config.json")


def update_model_config(model_config):
    """
    Function to get number of layers, model class and padding shape from model_config
    :param model_config: Dict containing model configuration
    :return model_config - Dict
    """
    config = AutoConfig.from_pretrained(model_config["model_name"])
    n_heads, d_head, n_layer = get_config(config)

    model_config["n_layer"] = 2  # test only 2 layer models
    model_config["model_class"] = AutoModelForCausalLM
    model_config["padding_shape"] = [1, n_heads, Constants.CTX_LEN, d_head]

    return model_config


@pytest.mark.parametrize(
    "model_name",
    [conf["model_name"] for conf in json.load(open(TEST_CONFIG_FILE_PATH, "r"))["models"]],
    ids=lambda x: "model_name=" + str(x),
)
class TestQEfficientModels:
    def setup_class(cls):
        """
        Set up function to set up the test environment for TestQEfficientModels class
        :param cls
        """
        cls.model_configs = []
        with open(TEST_CONFIG_FILE_PATH, "r") as f:
            configs = json.load(f)
            for model_config in configs["models"]:
                cls.model_configs.append(update_model_config(model_config))

        cls.setup_infos = {model_config["model_name"]: set_up(model_config) for model_config in cls.model_configs}

    def test_qefficient_model_torch(self, model_name):
        """
        Test function to validate the model before and after KV changes on Pytorch
        :param model_name: Name of model.
        """
        (
            (
                self.setup_infos[model_name]["pytorch_hf_tokens"] == self.setup_infos[model_name]["pytorch_kv_tokens"]
            ).all(),
            "Tokens don't match for HF PyTorch model output and KV PyTorch model output",
        )

    def test_qefficient_model_onnx(self, model_name):
        """
        Test function to validate the model before and after KV changes on ONNXRT
        :param model_name: Name of model.
        """
        (
            (self.setup_infos[model_name]["pytorch_kv_tokens"] == self.setup_infos[model_name]["ort_tokens"]).all(),
            "Tokens don't match for ONNXRT output and PyTorch output.",
        )

    @pytest.mark.skipif(not get_available_device_id, reason="No available devices to run model on Cloud AI 100")
    def test_qefficient_model_cloud_ai_100(self, model_name):
        """
        Test function to validate the model before and after KV changes on Cloud AI 100
        :param model_name: Name of model.
        """

        cloud_ai_100_tokens = get_cloud_ai_100_tokens(self.setup_infos[model_name])
        (
            (self.setup_infos[model_name]["ort_tokens"] == cloud_ai_100_tokens).all(),
            "Tokens don't match for ONNXRT output and Cloud AI 100 output.",
        )

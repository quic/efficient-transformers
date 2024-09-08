# -----------------------------------------------------------------------------
#
# Copyright (c) 2024 Qualcomm Innovation Center, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------

import functools
import unittest

from transformers import AutoModelForCausalLM
from transformers.quantizers.auto import AUTO_QUANTIZATION_CONFIG_MAPPING, AUTO_QUANTIZER_MAPPING

from QEfficient.transformers.quantizers.quantizer_awq import QEffAwqConfig, QEffAwqQuantizer
from QEfficient.transformers.quantizers.quantizer_gptq import QEffGPTQConfig, QEffGPTQQuantizer
from QEfficient.utils import hf_download
from QEfficient.utils.device_utils import is_multi_qranium_setup_available


def skip_if_mq_not_enabled(test_method):
    """
    Wrapper function to skip test if MQ setup not enabled
    """

    @functools.wraps(test_method)
    def wrapper(self):
        if self.setup_info["qpc_gt_32gb"] and (not is_multi_qranium_setup_available()):
            raise unittest.SkipTest("Skip because MQ set up not available")

        return test_method(self)

    return wrapper


def load_pytorch_model(model_config):
    """
    Function to load model from huggingface and transform to KV model
    --------

    :model_config: Dict

    :return model_hf, params
    """
    model_path = hf_download(
        repo_id=model_config["model_name"],
        ignore_patterns=["*.onnx", "*.ot", "*.md", "*.tflite", "*.pdf", "*.h5", "*.msgpack"],
    )
    model_hf = AutoModelForCausalLM.from_pretrained(
        model_path,
        use_cache=True,
        num_hidden_layers=model_config["n_layer"],
        attn_implementation="eager",
        low_cpu_mem_usage=model_config["low_cpu_mem_usage"],
    )  # Run models for single layers only
    params = sum(p.numel() for p in model_hf.parameters())
    model_hf.eval()
    return model_hf, params


def replace_transformers_quantizers():
    AUTO_QUANTIZER_MAPPING.update({"awq": QEffAwqQuantizer})
    AUTO_QUANTIZATION_CONFIG_MAPPING.update({"awq": QEffAwqConfig})
    AUTO_QUANTIZER_MAPPING.update({"gptq": QEffGPTQQuantizer})
    AUTO_QUANTIZATION_CONFIG_MAPPING.update({"gptq": QEffGPTQConfig})

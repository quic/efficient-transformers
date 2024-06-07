# -----------------------------------------------------------------------------
#
# Copyright (c)  2023-24 Qualcomm Innovation Center, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------

import functools
import os
import shutil
import unittest

from QEfficient import QEFFAutoModelForCausalLM
from QEfficient.compile.compile_helper import compile_kv_model_on_cloud_ai_100
from QEfficient.exporter.export_hf_to_cloud_ai_100 import qualcomm_efficient_converter
from QEfficient.transformers.transform import transform_lm
from QEfficient.utils import hf_download, load_hf_tokenizer
from QEfficient.utils.constants import QEFF_MODELS_DIR, ROOT_DIR, Constants
from QEfficient.utils.device_utils import get_available_device_id, is_multi_qranium_setup_available, is_qpc_size_gt_32gb
from QEfficient.utils.run_utils import ApiRunner


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


def prepare_work_dir(work_dir):
    """
    Function to create the work directory location

    :param type(str): path to the workspace directory
    :return: folder is created successfully
    """
    temp_dir = os.path.join(work_dir)
    if os.path.exists(temp_dir):
        shutil.rmtree(temp_dir)
    # create empty temp dir
    os.makedirs(temp_dir)


def remove_temp_dir(work_dir):
    """
    Function to remove the temp work directory location

    :param type(str): path to the workspace directory
    """
    temp_dir = os.path.join(work_dir)
    if os.path.exists(temp_dir):
        shutil.rmtree(temp_dir)


def get_tokenizer(model_name):
    """
    Function to get tokenizer info from transformers.AutoTokenizer
    :param model_name: str
    :return tokenizer
    """
    tokenizer = load_hf_tokenizer(model_name=model_name)
    return tokenizer


def load_pytorch_model(model_config):
    """
    Function to load model from huggingface and transform to KV model
    :param model_config: json object
    :return model_hf
    """
    model_path = hf_download(
        repo_id=model_config["model_name"], ignore_patterns=["*.txt", "*.onnx", "*.ot", "*.md", "*.tflite", "*.pdf"]
    )
    model_hf = model_config["model_class"].from_pretrained(
        model_path, use_cache=True, num_hidden_layers=model_config["n_layer"], attn_implementation="eager"
    )  # Run models for single layers only
    params = sum(p.numel() for p in model_hf.parameters())
    model_hf.eval()
    return model_hf, params


def transform_pt_model_with_qeff(model_hf):
    """
    Function to take huggingface model and transform to KV model
    :param model_hf: pytorch model
    :return model_kv
    """
    model_kv = transform_lm(model_hf)
    model_kv.eval()
    return model_kv


def export_onnx(model_kv, tokenizer, model_name, model_class):
    """
    Function to export onnx model
    :param model_name: str
    :param model_class: type
    :return onnx_model_path : str
    """
    onnx_dir_path = os.path.join(QEFF_MODELS_DIR, model_name)
    base_path, onnx_model_path = qualcomm_efficient_converter(
        model_name=model_name,
        model_kv=QEFFAutoModelForCausalLM(model=model_kv), # type: ignore
        tokenizer=tokenizer,
        onnx_dir_path=onnx_dir_path,
        kv=True,
        return_path=True,
    )
    return base_path, onnx_model_path


def set_up(model_config, device_group=[0]):
    """
    Set up function to set up the test environment for TestQEfficientModel class
    :param None
    """
    tokenizer = get_tokenizer(model_config["model_name"])
    api_runner = ApiRunner(
        tokenizer,
        Constants.INPUT_STRING,
        Constants.PROMPT_LEN,
        Constants.CTX_LEN,
    )
    mxfp6 = False
    model_hf, params = load_pytorch_model(model_config)
    qpc_gt_32gb = is_qpc_size_gt_32gb(params, mxfp6)
    try:
        pytorch_hf_tokens = api_runner.run_hf_model_on_pytorch(model_hf)
    except Exception as e:
        print(f"Pytorch HuggingFace Pytorch Model run failed due to : {e}")

    model_kv = transform_pt_model_with_qeff(model_hf)
    try:
        pytorch_kv_tokens = api_runner.run_kv_model_on_pytorch(
            model_kv,
            model_config["n_layer"],
            model_config["padding_shape"],
        )
    except Exception as e:
        print(f"Pytorch KV Cache Model run failed due to : {e}")

    base_path, onnx_model_path = export_onnx(
        model_kv,
        tokenizer,
        model_config["model_name"],
        model_config["model_class"],
    )
    ort_tokens = api_runner.run_kv_model_on_ort(
        onnx_model_path,
        model_config["n_layer"],
        model_config["padding_shape"],
    )

    setup_info = {}
    setup_info["model_config"] = model_config
    setup_info["device_group"] = device_group
    setup_info["api_runner"] = api_runner
    setup_info["qpc_gt_32gb"] = qpc_gt_32gb
    setup_info["pytorch_hf_tokens"] = pytorch_hf_tokens
    setup_info["pytorch_kv_tokens"] = pytorch_kv_tokens
    setup_info["base_path"] = base_path
    setup_info["onnx_model_path"] = onnx_model_path
    setup_info["ort_tokens"] = ort_tokens
    return setup_info


def get_cloud_ai_100_tokens(setup_info):
    """
    Test function to validate the llama model before and after KV changes on Cloud AI 100
    :param None
    """
    device_id = get_available_device_id()
    tests_qpc_dir = os.path.join(setup_info["base_path"], "tests_qpc")
    os.makedirs(tests_qpc_dir, exist_ok=True)
    if device_id:
        _, test_qpcs_path = compile_kv_model_on_cloud_ai_100(
            onnx_path=setup_info["onnx_model_path"],
            specializations_json=f"{ROOT_DIR}/scripts/specializations.json",
            num_cores=14,
            base_path=tests_qpc_dir,
            mxfp6=False,
            custom_io_path=os.path.join(setup_info["base_path"], "custom_io_fp16.yaml"),
            aic_enable_depth_first=False,
            device_group=setup_info["device_group"],
        )
        from QEfficient.generation.cloud_infer import QAICInferenceSession

        session = QAICInferenceSession(test_qpcs_path, device_id, enable_debug_logs=False)
        try:
            cloud_ai_100_tokens = setup_info["api_runner"].run_kv_model_on_cloud_ai_100(
                session,
                setup_info["model_config"]["n_layer"],
                setup_info["model_config"]["padding_shape"],
            )
        except Exception as e:
            print(f"ONNX Model run on Cloud AI 100 failed due to : {e}")
        return cloud_ai_100_tokens

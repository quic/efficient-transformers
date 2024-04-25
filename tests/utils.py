# -----------------------------------------------------------------------------
#
# Copyright (c)  2023-24 Qualcomm Innovation Center, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------

import os
import shutil

import transformers

import QEfficient
from QEfficient.exporter.export_hf_to_cloud_ai_100 import qualcomm_efficient_converter
from QEfficient.exporter.export_utils import compile_kv_model_on_cloud_ai_100
from QEfficient.utils import hf_download
from QEfficient.utils.constants import Constants, QEFF_MODELS_DIR, ROOT_DIR
from QEfficient.utils.device_utils import get_available_device_id
from QEfficient.utils.run_utils import ApiRunner


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
    model_hf_path = hf_download(repo_id=model_name)
    tokenizer = transformers.AutoTokenizer.from_pretrained(model_hf_path, padding_side="left")
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    return tokenizer

def load_pytorch_model(model_name, model_class):
    """
    Function to load model from huggingface and transform to KV model
    :param model_name: str
    :param model_class: type
    :return model_hf
    """
    model_path = hf_download(repo_id=model_name)
    model_hf = model_class.from_pretrained(model_path, use_cache=True)
    model_hf.eval()
    return model_hf

def transform_pt_model_with_qeff(model_hf):
    """
    Function to take huggingface model and transform to KV model
    :param model_hf: pytorch model
    :return model_kv
    """
    model_kv = QEfficient.transform(model_hf, type="Transformers", form_factor="cloud")
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
        model_class=model_class,
        model_kv=model_kv,
        tokenizer=tokenizer,
        onnx_dir_path=onnx_dir_path,
        kv=True,
        return_path=True)
    return base_path, onnx_model_path

def set_up(model_config):
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

    model_hf = load_pytorch_model(model_config["model_name"], model_config["model_class"])
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
    try:
        ort_tokens = api_runner.run_kv_model_on_ort(
            onnx_model_path,
            model_config["n_layer"],
            model_config["padding_shape"],
        )
    except Exception as e:
        print(f"ONNX Model run on onnxrt failed due to : {e}")

    setup_info = {}
    setup_info["model_config"] = model_config
    setup_info["api_runner"] = api_runner
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
            custom_io_path=os.path.join(setup_info["base_path"], "custom_io.yaml"),
            aic_enable_depth_first=False,
            device_group=[0],
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

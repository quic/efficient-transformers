# -----------------------------------------------------------------------------
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------

import json
import os
import shutil

import pytest

from QEfficient.utils import get_onnx_dir_name
from QEfficient.utils.constants import QEFF_MODELS_DIR
from QEfficient.utils.logging_utils import logger


def pytest_addoption(parser):
    parser.addoption("--all", action="store_true", default=False, help="Run all test without skipping any test")


model_class_dict = {"gpt2": "GPT2LMHeadModel", "lu-vae/llama-68m-fft": "LlamaForCausalLM"}


class ModelSetup:
    """
    ModelSetup is a set up class for all the High Level testing script,
    which provides all neccessary objects needed for checking the flow and creation
    of the HL API code.
    """

    def __init__(
        self,
        model_name,
        num_cores,
        prompt,
        prompts_txt_file_path,
        aic_enable_depth_first,
        mos,
        cache_dir,
        hf_token,
        batch_size,
        prompt_len,
        ctx_len,
        mxfp6,
        mxint8,
        full_batch_size,
        device_group,
        enable_qnn,
        qnn_config,
        image_url,
    ):
        """
        Initialization set up
        ------
        param: model_name: str
        param: num_cores: int
        param: prompt: str
        param: prompts_txt_file_path: str
        param: aic_enable_depth_first: bool
        param: mos: int
        param: cache_dir: str
        param: hf_token: str
        param: batch_size: int
        param: prompt_len: int
        param: ctx_len: int
        param: mxfp6: bool
        param: mxint8: bool
        param: full_batch_size: int
        param: device_group: List[int]
        param: enable_qnn: bool
        param: qnn_config: str
        """
        self.model_name = model_name
        self.num_cores = num_cores
        self.prompt = prompt
        self.local_model_dir = None
        self.prompts_txt_file_path = prompts_txt_file_path if prompts_txt_file_path is not None else None
        self.aic_enable_depth_first = aic_enable_depth_first
        self.mos = mos
        self.cache_dir = cache_dir
        self.hf_token = hf_token
        self.batch_size = batch_size
        self.prompt_len = prompt_len
        self.ctx_len = ctx_len
        self.generation_len = None
        self.mxfp6 = mxfp6
        self.mxint8 = mxint8
        self.full_batch_size = full_batch_size
        self.device_group = device_group
        self.enable_qnn = enable_qnn
        self.qnn_config = qnn_config
        self.image_url = image_url

    def model_card_dir(self):
        return str(os.path.join(QEFF_MODELS_DIR, str(self.model_name)))

    def qpc_base_dir_path(self):
        base_dir_name = str(
            f"qpc{'_qnn_' if self.enable_qnn else '_'}{self.num_cores}cores_{self.batch_size}bs_{self.prompt_len}pl_{self.ctx_len}cl_{self.mos}mos"
            + f"{f'_{self.full_batch_size}fbs_' if self.full_batch_size is not None else '_'}"
            + f"{len(self.device_group) if self.device_group is not None else 1}"
            + "devices"
            + (
                "_mxfp6_mxint8"
                if (self.mxfp6 and self.mxint8)
                else "_mxfp6"
                if self.mxfp6
                else "_fp16_mxint8"
                if self.mxint8
                else "_fp16"
            )
        )
        return str(os.path.join(self.model_card_dir(), base_dir_name))

    def qpc_dir_path(self):
        return str(os.path.join(self.qpc_base_dir_path(), "qpcs"))

    def onnx_dir_name(self):
        return get_onnx_dir_name(self.model_name, self.full_batch_size is not None)

    def onnx_dir_path(self):
        return str(os.path.join(self.model_card_dir(), self.onnx_dir_name()))

    def onnx_model_path(self):
        return [
            str(os.path.join(self.onnx_dir_path(), self.model_name.replace("/", "_") + "_kv_clipped_fp16.onnx")),
            str(os.path.join(self.onnx_dir_path(), self.model_name.replace("/", "_") + "_kv.onnx")),
        ]

    def model_hf_path(self):
        return str(os.path.join(self.cache_dir, self.model_name))

    def base_path_and_generated_onnx_path(self):
        return str(self.onnx_dir_path()), str(
            os.path.join(self.onnx_dir_path(), self.model_name.replace("/", "_") + "_kv_clipped_fp16.onnx")
        )

    def specialization_json_path(self):
        return str(os.path.join(self.qpc_base_dir_path(), "specializations.json"))

    def custom_io_file_path(self):
        if self.mxint8:
            return str(os.path.join(self.onnx_dir_path(), "custom_io_int8.yaml"))
        else:
            return str(os.path.join(self.onnx_dir_path(), "custom_io_fp16.yaml"))


@pytest.fixture(scope="function")
def setup(
    model_name,
    num_cores,
    prompt,
    prompts_txt_file_path,
    aic_enable_depth_first,
    mos,
    cache_dir,
    hf_token,
    batch_size,
    prompt_len,
    ctx_len,
    mxfp6,
    mxint8,
    full_batch_size,
    device_group,
    enable_qnn,
    qnn_config,
    image_url,
):
    """
    It is a fixture or shared object of all testing script within or inner folder,
    Args are coming from the dynamically generated tests method i.e, pytest_generate_tests via testing script or method
    --------
    Args: same as set up initialization
    Return: model_setup class object
    """
    model_setup = ModelSetup(
        model_name,
        num_cores,
        prompt,
        prompts_txt_file_path,
        bool(aic_enable_depth_first),
        mos,
        cache_dir,
        hf_token,
        batch_size,
        prompt_len,
        ctx_len,
        bool(mxfp6),
        bool(mxint8),
        full_batch_size,
        device_group,
        enable_qnn,
        qnn_config,
        image_url,
    )

    yield model_setup
    del model_setup


def pytest_generate_tests(metafunc):
    """
    pytest_generate_tests hook is used to create our own input parametrization,
    It generates all the test cases of different combination of input parameters which are read from the json file,
    and passed to each testing script module.
    -----------
    Ref: https://docs.pytest.org/en/7.3.x/how-to/parametrize.html
    """
    json_file = "tests/cloud/CLI_testing_config.json"
    with open(json_file, "r") as file:
        json_data = json.load(file)

    metafunc.parametrize("model_name", json_data["model_name"], ids=lambda x: "model_name=" + str(x))
    metafunc.parametrize("num_cores", json_data["num_cores"], ids=lambda x: "num_cores=" + str(x))
    metafunc.parametrize("prompt", json_data["prompt"], ids=lambda x: "prompt=" + str(x))
    metafunc.parametrize(
        "prompts_txt_file_path", json_data["prompts_txt_file_path"], ids=lambda x: "prompts_txt_file_path=" + str(x)
    )
    metafunc.parametrize(
        "aic_enable_depth_first", json_data["aic_enable_depth_first"], ids=lambda x: "aic_enable_depth_first=" + str(x)
    )
    metafunc.parametrize("mos", json_data["mos"], ids=lambda x: "mos=" + str(x))
    metafunc.parametrize("cache_dir", [None], ids=lambda x: "cache_dir=" + str(x))
    metafunc.parametrize("hf_token", json_data["hf_token"], ids=lambda x: "hf_token=" + str(x))
    metafunc.parametrize("batch_size", json_data["batch_size"], ids=lambda x: "batch_size=" + str(x))
    metafunc.parametrize("prompt_len", json_data["prompt_len"], ids=lambda x: "prompt_len=" + str(x))
    metafunc.parametrize("ctx_len", json_data["ctx_len"], ids=lambda x: "ctx_len=" + str(x))
    metafunc.parametrize("mxfp6", json_data["mxfp6"], ids=lambda x: "mxfp6=" + str(x))
    metafunc.parametrize("mxint8", json_data["mxint8"], ids=lambda x: "mxint8=" + str(x))
    metafunc.parametrize("full_batch_size", json_data["full_batch_size"], ids=lambda x: "full_batch_size=" + str(x))
    metafunc.parametrize("device_group", json_data["device_group"], ids=lambda x: "device_group=" + str(x))
    metafunc.parametrize("enable_qnn", json_data["enable_qnn"], ids=lambda x: "enable_qnn=" + str(x))
    metafunc.parametrize("qnn_config", json_data["qnn_config"], ids=lambda x: "qnn_config=" + str(x))
    metafunc.parametrize("image_url", json_data["image_url"], ids=lambda x: "image_url=" + str(x))


def qeff_models_clean_up():
    if os.path.exists(QEFF_MODELS_DIR):
        shutil.rmtree(QEFF_MODELS_DIR)
        logger.info(f"\n.............Cleaned up {QEFF_MODELS_DIR}")


def pytest_sessionstart(session):
    logger.info("PYTEST Session Starting ...")
    qeff_models_clean_up()


def pytest_sessionfinish(session, exitstatus):
    inside_worker = getattr(session.config, "workerinput", None)
    if inside_worker is None:
        qeff_models_clean_up()
        logger.info("...PYTEST Session Ended.")

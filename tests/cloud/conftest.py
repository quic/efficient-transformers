# -----------------------------------------------------------------------------
#
# Copyright (c) 2024 Qualcomm Innovation Center, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------

import json
import os
import shutil

import pytest

from QEfficient.utils import get_qpc_dir_name_infer
from QEfficient.utils.constants import QEFF_MODELS_DIR, ROOT_DIR, Constants
from QEfficient.utils.logging_utils import logger


def pytest_addoption(parser):
    parser.addoption("--all", action="store_true", default=False, help="Run all test without skipping any test")


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
        device_group,
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
        param: device_group: List[int]
        """
        self.model_name = model_name
        self.num_cores = num_cores
        self.prompt = prompt
        self.prompts_txt_file_path = (
            os.path.join(ROOT_DIR, prompts_txt_file_path) if prompts_txt_file_path is not None else None
        )
        self.aic_enable_depth_first = aic_enable_depth_first
        self.mos = mos
        self.cache_dir = cache_dir
        self.hf_token = hf_token
        self.batch_size = batch_size
        self.prompt_len = prompt_len
        self.ctx_len = ctx_len
        self.mxfp6 = mxfp6
        self.mxint8 = mxint8
        self.device_group = device_group

    def model_card_dir(self):
        return str(os.path.join(QEFF_MODELS_DIR, str(self.model_name)))

    def qpc_base_dir_name(self):
        return get_qpc_dir_name_infer(
            self.num_cores,
            self.mos,
            self.batch_size,
            self.prompt_len,
            self.ctx_len,
            self.mxfp6,
            self.mxint8,
            self.device_group,
        )

    def qpc_dir_path(self):
        return str(os.path.join(self.model_card_dir(), self.qpc_base_dir_name(), "qpcs"))

    def onnx_dir_path(self):
        return str(os.path.join(self.model_card_dir(), "onnx"))

    def onnx_model_path(self):
        return str(os.path.join(self.onnx_dir_path(), self.model_name.replace("/", "_") + "_kv_clipped_fp16.onnx"))

    def model_hf_path(self):
        return str(os.path.join(self.cache_dir, self.model_name))

    def base_path_and_generated_onnx_path(self):
        return str(self.onnx_dir_path()), str(
            os.path.join(self.onnx_dir_path(), self.model_name.replace("/", "_") + "_kv_clipped_fp16.onnx")
        )

    def specialization_json_path(self):
        return str(os.path.join(self.model_card_dir(), self.qpc_base_dir_name(), "specializations.json"))

    def custom_io_file_path(self):
        if self.mxint8:
            return str(os.path.join(self.onnx_dir_path(), "custom_io_int8.yaml"))
        else:
            return str(os.path.join(self.onnx_dir_path(), "custom_io_fp16.yaml"))


@pytest.fixture
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
    device_group,
):
    """
    It is a fixture or shared object of all testing script within or inner folder,
    Args are coming from the dynamically generated tests method i.e, pytest_generate_tests via testing script or method
    --------
    Args: same as set up initialization
    Return: model_setup class object
    """
    yield ModelSetup(
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
        device_group,
    )


def pytest_generate_tests(metafunc):
    """
    pytest_generate_tests hook is used to create our own input parametrization,
    It generates all the test cases of different combination of input parameters which are read from the json file,
    and passed to each testing script module.
    -----------
    Ref: https://docs.pytest.org/en/7.3.x/how-to/parametrize.html
    """
    json_file = os.path.join(ROOT_DIR, "tests", "cloud", "high_level_testing.json")
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
    metafunc.parametrize("cache_dir", [Constants.CACHE_DIR], ids=lambda x: "cache_dir=" + str(x))
    metafunc.parametrize("hf_token", json_data["hf_token"], ids=lambda x: "hf_token=" + str(x))
    metafunc.parametrize("batch_size", json_data["batch_size"], ids=lambda x: "batch_size=" + str(x))
    metafunc.parametrize("prompt_len", json_data["prompt_len"], ids=lambda x: "prompt_len=" + str(x))
    metafunc.parametrize("ctx_len", json_data["ctx_len"], ids=lambda x: "ctx_len=" + str(x))
    metafunc.parametrize("mxfp6", json_data["mxfp6"], ids=lambda x: "mxfp6=" + str(x))
    metafunc.parametrize("mxint8", json_data["mxint8"], ids=lambda x: "mxint8=" + str(x))
    metafunc.parametrize("device_group", json_data["device_group"], ids=lambda x: "device_group=" + str(x))


def pytest_collection_modifyitems(config, items):
    """
    pytest_collection_modifyitems is pytest a hook,
    which is used to re-order the execution order of the testing script/methods
    with various combination of inputs.
    called after collection has been performed, may filter or re-order the items in-place.
    Parameters:
    items (List[_pytest.nodes.Item]) list of item objects
    ----------
    Ref: https://docs.pytest.org/en/4.6.x/reference.html#collection-hooks
    """
    run_first = ["test_export", "test_compile", "test_execute", "test_infer"]
    modules_name = {item.module.__name__ for item in items}
    cloud_modules = []
    non_cloud_modules = []
    for module in modules_name:
        if module in run_first:
            cloud_modules.append(module)
        else:
            non_cloud_modules.append(module)

    if len(cloud_modules) > 1:
        modules = {item: item.module.__name__ for item in items}
        items[:] = sorted(items, key=lambda x: run_first.index(modules[x]) if modules[x] in run_first else len(items))

        non_cloud_tests = []

        for itm in items:
            if modules[itm] not in cloud_modules:
                non_cloud_tests.append(itm)

        num_cloud_tests = len(items) - len(non_cloud_tests)
        num_cloud_test_cases = num_cloud_tests // len(cloud_modules)
        final_items = []

        for i in range(num_cloud_test_cases):
            for j in range(len(cloud_modules)):
                final_items.append(items[i + j * num_cloud_test_cases])

        final_items.extend(non_cloud_tests)
        items[:] = final_items

        if config.getoption("--all"):
            return

        first_model = items[0].callspec.params["model_name"] if hasattr(items[0], "callspec") else None

        for item in items:
            if item.module.__name__ in ["test_export", "test_compile", "test_execute"]:
                if hasattr(item, "callspec"):
                    params = item.callspec.params
                    if "model_name" in params and params["model_name"] != first_model:
                        item.add_marker(pytest.mark.skip(reason="Skipping because not needed now..."))
                    if "prompt_len" in params and params["prompt_len"] == 2:
                        item.add_marker(pytest.mark.skip(reason="Skipping because not needed now..."))

            if item.module.__name__ in ["test_infer"]:
                if hasattr(item, "callspec"):
                    params = item.callspec.params
                    if (
                        "prompt_len" in params
                        and params["prompt_len"] == 2
                        and "model_name" in params
                        and params["model_name"] != first_model
                    ):
                        item.add_marker(pytest.mark.skip(reason="Skipping because not needed now..."))


def cache_clean_up():
    if os.path.exists(Constants.CACHE_DIR):
        shutil.rmtree(Constants.CACHE_DIR)
        logger.info(f"\n.............Cleaned up {Constants.CACHE_DIR}")


def qeff_models_clean_up():
    if os.path.exists(QEFF_MODELS_DIR):
        shutil.rmtree(QEFF_MODELS_DIR)
        logger.info(f"\n.............Cleaned up {QEFF_MODELS_DIR}")


@pytest.fixture
def clean_up_after_test():
    yield
    qeff_models_clean_up()


def pytest_sessionstart(session):
    logger.info("PYTEST Session Starting ...")
    cache_clean_up()
    qeff_models_clean_up()


def pytest_sessionfinish(session, exitstatus):
    cache_clean_up()
    qeff_models_clean_up()
    logger.info("...PYTEST Session Ended.")

# -----------------------------------------------------------------------------
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------

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
    qeff_models_clean_up()


def pytest_sessionfinish(session, exitstatus):
    inside_worker = getattr(session.config, "workerinput", None)
    if inside_worker is None:
        qeff_models_clean_up()
        logger.info("...PYTEST Session Ended.")

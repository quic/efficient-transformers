# -----------------------------------------------------------------------------
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------

import pytest
from conftest import ModelSetup

from QEfficient.cloud.infer import main as infer

configs = [
    {
        "model_name": "gpt2",
        "num_cores": 16,
        "prompt": "My name is",
        "prompts_txt_file_path": "examples/prompts.txt",
        "aic_enable_depth_first": 1,
        "mos": 1,
        "cache_dir": None,
        "hf_token": None,
        "batch_size": 1,
        "prompt_len": 32,
        "ctx_len": 128,
        "mxfp6": 1,
        "mxint8": 1,
        "device_group": None,
        "full_batch_size": 3,
        "enable_qnn": True,
        "qnn_config": "QEfficient/compile/qnn_config.json",
        "image_url": "https://i.etsystatic.com/8155076/r/il/0825c2/1594869823/il_fullxfull.1594869823_5x0w.jpg",
    }
]


def check_infer(mocker, generation_len=32, **kwargs):
    ms = ModelSetup(
        kwargs["model_name"],
        kwargs["num_cores"],
        kwargs["prompt"],
        kwargs["prompts_txt_file_path"],
        bool(kwargs["aic_enable_depth_first"]),
        kwargs["mos"],
        kwargs["cache_dir"],
        kwargs["hf_token"],
        kwargs["batch_size"],
        kwargs["prompt_len"],
        kwargs["ctx_len"],
        bool(kwargs["mxfp6"]),
        bool(kwargs["mxint8"]),
        kwargs["full_batch_size"],
        kwargs["device_group"],
        kwargs["enable_qnn"],
        kwargs["qnn_config"],
    )

    infer(
        model_name=ms.model_name,
        num_cores=ms.num_cores,
        prompt=ms.prompt,
        local_model_dir=ms.local_model_dir,
        prompts_txt_file_path=ms.prompts_txt_file_path,
        aic_enable_depth_first=ms.aic_enable_depth_first,
        mos=ms.mos,
        hf_token=ms.hf_token,
        batch_size=ms.batch_size,
        prompt_len=ms.prompt_len,
        ctx_len=ms.ctx_len,
        generation_len=generation_len,
        mxfp6=ms.mxfp6,
        mxint8=ms.mxint8,
        full_batch_size=ms.full_batch_size,
        enable_qnn=ms.enable_qnn,
        qnn_config=ms.qnn_config,
        image_url=kwargs["image_url"],
    )


@pytest.mark.on_qaic
@pytest.mark.cli
@pytest.mark.usefixtures("clean_up_after_test")
@pytest.mark.parametrize("config", configs)
def test_infer(mocker, config):
    """
    test_infer is a HL infer api testing function,
    checks infer api code flow, object creations, internal api calls, internal returns.
    ---------
    Parameters:
    setup: is a fixture defined in conftest.py module.
    mocker: mocker is itself a pytest fixture, uses to mock or spy internal functions.
    ---------
    Ref: https://docs.pytest.org/en/7.1.x/how-to/fixtures.html
    Ref: https://pytest-mock.readthedocs.io/en/latest/usage.html
    """
    # testing infer without full_batch_size
    local_config = config.copy()
    local_config.update(full_batch_size=None, enable_qnn=False, qnn_config=None)
    check_infer(mocker=mocker, **local_config)


@pytest.mark.on_qaic
@pytest.mark.cli
@pytest.mark.usefixtures("clean_up_after_test")
@pytest.mark.parametrize("config", configs)
def test_infer_fb(mocker, config):
    # testing infer with full_batch_size
    local_config = config.copy()
    local_config.update(enable_qnn=False, qnn_config=None)
    check_infer(mocker=mocker, **local_config)


@pytest.mark.on_qaic
@pytest.mark.cli
@pytest.mark.qnn
@pytest.mark.usefixtures("clean_up_after_test")
@pytest.mark.parametrize("config", configs)
def test_infer_qnn(mocker, config):
    # testing infer without full_batch_size in QNN enviroment
    local_config = config.copy()
    local_config.update(
        full_batch_size=None,
        qnn_config=None,
    )
    check_infer(mocker=mocker, **local_config)


@pytest.mark.on_qaic
@pytest.mark.cli
@pytest.mark.qnn
@pytest.mark.usefixtures("clean_up_after_test")
@pytest.mark.parametrize("config", configs)
def test_infer_qnn_fb(mocker, config):
    # testing infer with full_batch_size in QNN enviroment
    local_config = config.copy()
    check_infer(mocker=mocker, **local_config)


@pytest.mark.on_qaic
@pytest.mark.cli
@pytest.mark.multimodal
@pytest.mark.usefixtures("clean_up_after_test")
@pytest.mark.parametrize("config", configs)
def test_infer_vlm(mocker, config):
    # testing infer for MM models
    local_config = config.copy()
    local_config.update(
        {
            "model_name": "llava-hf/llava-1.5-7b-hf",
            "prompt": "Describe the image.",
            "prompt_len": 1024,
            "ctx_len": 2048,
            "full_batch_size": None,
            "enable_qnn": False,
            "qnn_config": None,
        }
    )
    check_infer(mocker=mocker, generation_len=20, **local_config)

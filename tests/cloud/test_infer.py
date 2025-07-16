# -----------------------------------------------------------------------------
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------

import pytest

import QEfficient
from QEfficient.cloud.infer import main as infer


def check_infer(mocker, model_setup, generation_len=20):
    check_and_assign_cache_dir_spy = mocker.spy(QEfficient.cloud.infer, "check_and_assign_cache_dir")
    qeff_model_load_spy = mocker.spy(QEfficient.cloud.infer.QEFFCommonLoader, "from_pretrained")

    infer(
        model_name=model_setup.model_name,
        num_cores=model_setup.num_cores,
        prompt=model_setup.prompt,
        local_model_dir=model_setup.local_model_dir,
        prompts_txt_file_path=model_setup.prompts_txt_file_path,
        aic_enable_depth_first=model_setup.aic_enable_depth_first,
        mos=model_setup.mos,
        hf_token=model_setup.hf_token,
        batch_size=model_setup.batch_size,
        prompt_len=model_setup.prompt_len,
        ctx_len=model_setup.ctx_len,
        generation_len=generation_len,
        mxfp6=model_setup.mxfp6,
        mxint8=model_setup.mxint8,
        full_batch_size=model_setup.full_batch_size,
        enable_qnn=model_setup.enable_qnn,
        image_url=model_setup.image_url,
    )

    check_and_assign_cache_dir_spy.assert_called_once()
    qeff_model_load_spy.assert_called_once()


@pytest.mark.on_qaic
@pytest.mark.cli
def test_infer(mocker, setup):
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
    setup.model_name = "lu-vae/llama-68m-fft"
    setup.full_batch_size = None
    setup.enable_qnn = False
    setup.qnn_config = None
    check_infer(mocker, setup)


@pytest.mark.on_qaic
@pytest.mark.cli
def test_infer_fbs(mocker, setup):
    # testing infer with full_batch_size
    setup.model_name = "lu-vae/llama-68m-fft"
    setup.enable_qnn = False
    setup.qnn_config = None
    check_infer(mocker, setup)


@pytest.mark.on_qaic
@pytest.mark.cli
@pytest.mark.qnn
def test_infer_qnn(mocker, setup):
    # testing infer without full_batch_size in QNN environment
    setup.model_name = "lu-vae/llama-68m-fft"
    setup.full_batch_size = None
    setup.qnn_config = None
    check_infer(mocker, setup)


@pytest.mark.on_qaic
@pytest.mark.cli
@pytest.mark.qnn
def test_infer_qnn_fbs(mocker, setup):
    # testing infer with full_batch_size in QNN environment
    setup.model_name = "lu-vae/llama-68m-fft"
    setup.qnn_config = None
    check_infer(mocker, setup)


@pytest.mark.on_qaic
@pytest.mark.cli
@pytest.mark.multimodal
def test_infer_vlm(mocker, setup):
    # testing infer for MM models
    setup.model_name = "llava-hf/llava-1.5-7b-hf"
    setup.prompt = "Describe the image."
    setup.full_batch_size = None
    setup.enable_qnn = False
    setup.qnn_config = None

    check_infer(mocker, setup, generation_len=20)


@pytest.mark.on_qaic
@pytest.mark.cli
@pytest.mark.qnn
@pytest.mark.multimodal
def test_infer_vlm_qnn(mocker, setup):
    # testing infer for MM models in QNN environment
    setup.model_name = "llava-hf/llava-1.5-7b-hf"
    setup.prompt = "Describe the image."
    setup.full_batch_size = None
    setup.qnn_config = None

    check_infer(mocker, setup, generation_len=20)

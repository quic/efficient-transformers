# -----------------------------------------------------------------------------
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------

import pytest

import QEfficient
from QEfficient.cloud.infer import main as infer


def check_infer(
    mocker, model_name, prompt="My name is", full_batch_size=None, enable_qnn=False, image_url=None, generation_len=20
):
    check_and_assign_cache_dir_spy = mocker.spy(QEfficient.cloud.infer, "check_and_assign_cache_dir")
    qeff_model_load_spy = mocker.spy(QEfficient.cloud.infer.QEFFCommonLoader, "from_pretrained")
    load_hf_tokenizer_spy = mocker.spy(QEfficient.cloud.infer, "load_hf_tokenizer")
    execute_vlm_model_spy = mocker.spy(QEfficient.cloud.infer, "execute_vlm_model")

    infer(
        model_name=model_name,
        num_cores=16,
        prompt=prompt,
        local_model_dir=None,
        prompts_txt_file_path="examples/sample_prompts/prompts.txt",
        aic_enable_depth_first=True,
        mos=1,
        hf_token=None,
        batch_size=1,
        prompt_len=32,
        ctx_len=128,
        generation_len=generation_len,
        mxfp6=True,
        mxint8=True,
        full_batch_size=full_batch_size,
        enable_qnn=enable_qnn,
        image_url=image_url,
    )

    check_and_assign_cache_dir_spy.assert_called_once()
    qeff_model_load_spy.assert_called_once()
    if image_url is not None:
        execute_vlm_model_spy.assert_called_once()
    else:
        load_hf_tokenizer_spy.assert_called_once()


@pytest.mark.on_qaic
@pytest.mark.cli
def test_infer(mocker):
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
    check_infer(mocker, model_name="lu-vae/llama-68m-fft")


@pytest.mark.on_qaic
@pytest.mark.cli
def test_infer_fbs(mocker):
    # testing infer with full_batch_size
    check_infer(mocker, model_name="lu-vae/llama-68m-fft", full_batch_size=3)


@pytest.mark.on_qaic
@pytest.mark.cli
@pytest.mark.qnn
def test_infer_qnn(mocker):
    # testing infer without full_batch_size in QNN environment
    check_infer(mocker, model_name="lu-vae/llama-68m-fft", enable_qnn=True)


@pytest.mark.on_qaic
@pytest.mark.cli
@pytest.mark.qnn
def test_infer_qnn_fbs(mocker):
    # testing infer with full_batch_size in QNN environment
    check_infer(mocker, model_name="lu-vae/llama-68m-fft", full_batch_size=3, enable_qnn=True)


@pytest.mark.on_qaic
@pytest.mark.cli
@pytest.mark.multimodal
def test_infer_vlm(mocker):
    # testing infer for MM models
    check_infer(
        mocker,
        model_name="llava-hf/llava-1.5-7b-hf",
        prompt="Describe the image.",
        image_url="https://i.etsystatic.com/8155076/r/il/0825c2/1594869823/il_fullxfull.1594869823_5x0w.jpg",
    )

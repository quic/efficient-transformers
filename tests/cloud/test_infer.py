# -----------------------------------------------------------------------------
#
# Copyright (c) 2024 Qualcomm Innovation Center, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------


import pytest

from QEfficient.cloud.infer import main as infer


@pytest.mark.on_qaic
@pytest.mark.cli
@pytest.mark.usefixtures("clean_up_after_test")
def test_infer(setup, mocker):
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
    ms = setup

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
        generation_len=ms.generation_len,
        mxfp6=ms.mxfp6,
        mxint8=ms.mxint8,
        full_batch_size=ms.full_batch_size,
        enable_qnn=ms.enable_qnn,
    )

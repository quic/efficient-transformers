
# -----------------------------------------------------------------------------
#
# Copyright (c)  2024 Qualcomm Innovation Center, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------

import os

import pytest

import QEfficient
import QEfficient.cloud.infer
from QEfficient.cloud.infer import main as infer


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
    result = ms.check_batch_size_for_asserion_error()
    if result["error"] is not None:
        pytest.skip(f'...Skipping Because batch size is not compatible with the number of prompts: {result["error"]}')
    assert result['result'] is not None
    get_qpc_dir_name_infer_spy = mocker.spy(QEfficient.cloud.infer,"get_qpc_dir_name_infer")
    check_batch_size_and_num_prompts_spy = mocker.spy(QEfficient.cloud.infer,"check_batch_size_and_num_prompts")
    load_hf_tokenizer_spy = mocker.spy(QEfficient.cloud.infer,"load_hf_tokenizer")
    qpc_exists_spy = mocker.spy(QEfficient.cloud.infer,"qpc_exists")
    compile_spy = mocker.spy(QEfficient,"compile")
    cloud_ai_100_exec_kv_spy = mocker.spy(QEfficient.cloud.infer,"cloud_ai_100_exec_kv")
    infer(
            model_name = ms.model_name,
            num_cores = ms.num_cores,
            prompt = ms.prompt,
            prompts_txt_file_path = ms.prompts_txt_file_path,
            aic_enable_depth_first = ms.aic_enable_depth_first,
            mos = ms.mos,  
            hf_token = ms.hf_token,
            batch_size = ms.batch_size,
            prompt_len = ms.prompt_len,
            ctx_len = ms.ctx_len,
            mxfp6 = ms.mxfp6,
            mxint8 = ms.mxint8,
            device_group = ms.device_group,
            )  
    # prompt fucntion check
    get_qpc_dir_name_infer_spy.assert_called_once()
    check_batch_size_and_num_prompts_spy.assert_called_once()
    # tokenizer check
    load_hf_tokenizer_spy.assert_called_once()
    # qpc exist check
    qpc_exists_spy.assert_called_once()
    assert ms.qpc_dir_path() == qpc_exists_spy.spy_return[1]
    assert os.path.isdir(ms.qpc_dir_path())
    compile_spy.assert_not_called()
    cloud_ai_100_exec_kv_spy.assert_called_once()

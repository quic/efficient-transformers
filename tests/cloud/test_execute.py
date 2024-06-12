# -----------------------------------------------------------------------------
#
# Copyright (c)  2023-2024 Qualcomm Innovation Center, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------
import os
import warnings
import pytest
import QEfficient
import QEfficient.cloud.execute
from QEfficient.cloud.execute import main as execute

# @pytest.mark.run(after="test_compile") # Optional
def test_execute(setup, mocker):
    """
    test_execute is a HL execute api testing function,
    checks execute api code flow, object creations, internal api calls, internal returns.
    ---------
    Parameters:
    setup: is a fixture defined in conftest.py module.
    mocker: mocker is itself a pytest fixture, uses to mock or spy internal functions.
    """
    ms = setup
    
    load_hf_tokenizer_spy = mocker.spy(QEfficient.cloud.execute,"load_hf_tokenizer")
    get_compilation_batch_size_spy = mocker.spy(QEfficient.cloud.execute,"get_compilation_batch_size")
    check_batch_size_and_num_prompts_spy = mocker.spy(QEfficient.cloud.execute,"check_batch_size_and_num_prompts")
    cloud_ai_100_exec_kv_spy = mocker.spy(QEfficient.cloud.execute,"cloud_ai_100_exec_kv")
    except_flag = 0
    try:
        execute(model_name=ms.model_name,
                qpc_path=ms.qpc_dir_path(),
                device_group=ms.device_group,
                prompt=ms.prompt,
                prompts_txt_file_path=ms.prompts_txt_file_path,
                
                hf_token=ms.hf_token,)
    except AssertionError as e:
        warnings.warn(e.args[0] if e.args else "error")
        except_flag=1
    
    load_hf_tokenizer_spy.assert_called_once()
    get_compilation_batch_size_spy.assert_called_once()
    assert get_compilation_batch_size_spy.spy_return == ms.batch_size
    check_batch_size_and_num_prompts_spy.assert_called_once()
    if except_flag == 1:
        cloud_ai_100_exec_kv_spy.assert_not_called()
    else:
        lst = check_batch_size_and_num_prompts_spy.spy_return
        assert bool(lst) and isinstance(lst, list) and all(isinstance(elem, str) for elem in lst)
        cloud_ai_100_exec_kv_spy.assert_called_once()

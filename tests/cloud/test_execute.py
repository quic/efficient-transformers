# -----------------------------------------------------------------------------
#
# Copyright (c)  2024 Qualcomm Innovation Center, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------

import pytest

import QEfficient
import QEfficient.cloud.execute
from QEfficient.cloud.execute import main as execute


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
    result = ms.check_batch_size_for_asserion_error()
    if result["error"] is not None:
        pytest.skip(f'...Skipping Because batch size is not compatible with the number of prompts: {result["error"]}')
    assert result['result'] is not None
    load_hf_tokenizer_spy = mocker.spy(QEfficient.cloud.execute,"load_hf_tokenizer")
    get_compilation_dims_spy = mocker.spy(QEfficient.cloud.execute,"get_compilation_dims")
    check_batch_size_and_num_prompts_spy = mocker.spy(QEfficient.cloud.execute,"check_batch_size_and_num_prompts")
    cloud_ai_100_exec_kv_spy = mocker.spy(QEfficient.cloud.execute,"cloud_ai_100_exec_kv")
  
    execute(model_name=ms.model_name,
            qpc_path=ms.qpc_dir_path(),
            device_group=ms.device_group,
            prompt=ms.prompt,
            prompts_txt_file_path=ms.prompts_txt_file_path,
            generation_len=ms.generation_len,
            # cache_dir=ms.cache_dir,
            hf_token=ms.hf_token,
            full_batch_size=ms.full_batch_size,
            )
   
    load_hf_tokenizer_spy.assert_called_once()
    get_compilation_dims_spy.assert_called_once()
    assert get_compilation_dims_spy.spy_return == (ms.batch_size, ms.ctx_len)
    check_batch_size_and_num_prompts_spy.assert_called_once()
    cloud_ai_100_exec_kv_spy.assert_called_once()

# -----------------------------------------------------------------------------
#
# Copyright (c) 2024 Qualcomm Innovation Center, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------

import pytest

import QEfficient
import QEfficient.cloud.execute
from QEfficient.cloud.execute import main as execute


@pytest.mark.on_qaic
@pytest.mark.cli
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
    load_hf_tokenizer_spy = mocker.spy(QEfficient.cloud.execute, "load_hf_tokenizer")
    cloud_ai_100_exec_kv_spy = mocker.spy(QEfficient.cloud.execute, "cloud_ai_100_exec_kv")

    execute(
        model_name=ms.model_name,
        qpc_path=ms.qpc_dir_path(),
        prompt=ms.prompt,
        prompts_txt_file_path=ms.prompts_txt_file_path,
        generation_len=ms.generation_len,
        hf_token=ms.hf_token,
        full_batch_size=ms.full_batch_size,
    )

    load_hf_tokenizer_spy.assert_called_once()
    cloud_ai_100_exec_kv_spy.assert_called_once()

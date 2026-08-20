# -----------------------------------------------------------------------------
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------

import pytest

from QEfficient.cloud.execute import main as execute


@pytest.mark.cli
def test_execute_calls_tokenizer_and_runtime(mocker):
    tokenizer = object()
    load_hf_tokenizer = mocker.patch("QEfficient.cloud.execute.load_hf_tokenizer", return_value=tokenizer)
    cloud_ai_100_exec_kv = mocker.patch("QEfficient.cloud.execute.cloud_ai_100_exec_kv")

    execute(
        model_name="gpt2",
        qpc_path="/nonexistent/test-qpc",
        device_group=[0],
        local_model_dir=None,
        prompt=["My name is"],
        prompts_txt_file_path="examples/sample_prompts/prompts.txt",
        generation_len=20,
        cache_dir="/tmp/cache",
        hf_token="token",
        full_batch_size=3,
    )

    load_hf_tokenizer.assert_called_once_with(
        pretrained_model_name_or_path="gpt2",
        cache_dir="/tmp/cache",
        hf_token="token",
    )
    cloud_ai_100_exec_kv.assert_called_once_with(
        tokenizer=tokenizer,
        qpc_path="/nonexistent/test-qpc",
        device_id=[0],
        prompt=["My name is"],
        prompts_txt_file_path="examples/sample_prompts/prompts.txt",
        generation_len=20,
    )

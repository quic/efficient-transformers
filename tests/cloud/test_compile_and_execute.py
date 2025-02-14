# -----------------------------------------------------------------------------
#
# Copyright (c) 2024 Qualcomm Innovation Center, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------

import os

import pytest
import yaml

import QEfficient
import QEfficient.cloud.compile
from QEfficient.cloud.execute import main as execute
from QEfficient.cloud.export import get_onnx_model_path


@pytest.mark.on_qaic
@pytest.mark.cli
def test_compile(setup, mocker):
    """
    test_compile is a HL compile api testing function,
    checks compile api code flow, object creations, internal api calls, internal returns.
    ---------
    Parameters:
    setup: is a fixture defined in conftest.py module.
    mocker: mocker is itself a pytest fixture, uses to mock or spy internal functions.
    """
    ms = setup
    onnx_model_path = get_onnx_model_path(
        model_name=ms.model_name,
        cache_dir=ms.cache_dir,
        hf_token=ms.hf_token,
        full_batch_size=ms.full_batch_size,
        local_model_dir=ms.local_model_dir,
    )

    base_key = "past_key."
    base_value = "past_value."
    precision = "float16"

    data = []

    for i in range(12):
        data.append({"IOName": f"{base_key}{i}", "Precision": precision})
        data.append({"IOName": f"{base_value}{i}", "Precision": precision})

    for i in range(12):
        data.append({"IOName": f"{base_key}{i}_RetainedState", "Precision": precision})
        data.append({"IOName": f"{base_value}{i}_RetainedState", "Precision": precision})

    with open(((onnx_model_path.parent) / "custom_io.yaml"), "w") as file:
        yaml.dump(data, file)

    qpc_path = QEfficient.compile(
        onnx_path=onnx_model_path,
        qpc_path=os.path.dirname(ms.qpc_dir_path()),
        num_cores=ms.num_cores,
        device_group=ms.device_group,
        custom_io_file_path=(onnx_model_path.parent) / "custom_io.yaml",
        aic_enable_depth_first=ms.aic_enable_depth_first,
        mos=ms.mos,
        batch_size=ms.batch_size,
        prompt_len=ms.prompt_len,
        ctx_len=ms.ctx_len,
        mxfp6=ms.mxfp6,
        mxint8=ms.mxint8,
        full_batch_size=ms.full_batch_size,
        enable_qnn=ms.enable_qnn,
    )

    execute(
        model_name=ms.model_name,
        qpc_path=qpc_path,
        prompt=ms.prompt,
        prompts_txt_file_path=ms.prompts_txt_file_path,
        generation_len=ms.generation_len,
        hf_token=ms.hf_token,
        full_batch_size=ms.full_batch_size,
    )

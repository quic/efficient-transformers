# -----------------------------------------------------------------------------
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------


import os

import pytest
import yaml
from conftest import ModelSetup

import QEfficient
from QEfficient.cloud.execute import main as execute
from QEfficient.cloud.export import main as export

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


def check_export_compile_execute(
    mocker,
    **kwargs,
):
    # Setup model
    model_setup = ModelSetup(
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

    # Spy on internal functions
    mocker.spy(QEfficient.utils, "check_and_assign_cache_dir")
    mock_get_onnx = mocker.spy(QEfficient.cloud.export, "get_onnx_model_path")

    # Export model
    export(
        model_name=model_setup.model_name,
        hf_token=model_setup.hf_token,
        local_model_dir=model_setup.local_model_dir,
        full_batch_size=model_setup.full_batch_size,
    )

    onnx_model_path = mock_get_onnx.spy_return
    print(f"Captured ONNX path: {onnx_model_path}")

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

    # Compile model
    qpc_path = QEfficient.compile(
        onnx_path=onnx_model_path,
        qpc_path=os.path.dirname(model_setup.qpc_dir_path()),
        num_cores=model_setup.num_cores,
        device_group=model_setup.device_group,
        custom_io_file_path=(onnx_model_path.parent) / "custom_io.yaml",
        aic_enable_depth_first=model_setup.aic_enable_depth_first,
        mos=model_setup.mos,
        batch_size=model_setup.batch_size,
        prompt_len=model_setup.prompt_len,
        ctx_len=model_setup.ctx_len,
        mxfp6=model_setup.mxfp6,
        mxint8=model_setup.mxint8,
        full_batch_size=model_setup.full_batch_size,
        enable_qnn=model_setup.enable_qnn,
    )

    # Execute model
    execute(
        model_name=model_setup.model_name,
        qpc_path=qpc_path,
        prompt=model_setup.prompt,
        prompts_txt_file_path=model_setup.prompts_txt_file_path,
        generation_len=model_setup.generation_len,
        hf_token=model_setup.hf_token,
        full_batch_size=model_setup.full_batch_size,
    )


@pytest.mark.on_qaic
@pytest.mark.cli
@pytest.mark.parametrize("config", configs)
def test_export_compile_execute(mocker, config):
    # testing export -> compile -> infer without full_batch_size

    local_config = config.copy()
    local_config.update(full_batch_size=None, enable_qnn=False, qnn_config=None)
    check_export_compile_execute(mocker=mocker, **local_config)


@pytest.mark.on_qaic
@pytest.mark.cli
@pytest.mark.parametrize("config", configs)
def test_export_compile_execute_fb(mocker, config):
    # testing export -> compile -> infer with full_batch_size
    local_config = config.copy()
    local_config.update(enable_qnn=False, qnn_config=None)
    check_export_compile_execute(mocker=mocker, **local_config)


@pytest.mark.on_qaic
@pytest.mark.qnn
@pytest.mark.cli
@pytest.mark.parametrize("config", configs)
def test_export_compile_execute_qnn(mocker, config):
    # testing export -> compile -> infer without full_batch_size in QNN enviroment
    local_config = config.copy()
    local_config.update(full_batch_size=None, qnn_config=None)
    check_export_compile_execute(mocker=mocker, **local_config)


@pytest.mark.on_qaic
@pytest.mark.qnn
@pytest.mark.cli
@pytest.mark.parametrize("config", configs)
def test_export_compile_execute_qnn_fb(mocker, config):
    # testing export -> compile -> infer with full_batch_size in QNN enviroment
    local_config = config.copy()
    check_export_compile_execute(mocker=mocker, **local_config)

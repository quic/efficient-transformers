# -----------------------------------------------------------------------------
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------


import os

import pytest
import yaml

import QEfficient
from QEfficient.cloud.execute import main as execute
from QEfficient.cloud.export import main as export


def check_export_compile_execute(mocker, model_setup):
    # Spy on internal functions
    check_and_assign_cache_dir_spy = mocker.spy(QEfficient.cloud.export, "check_and_assign_cache_dir")
    get_onnx_model_path_spy = mocker.spy(QEfficient.cloud.export, "get_onnx_model_path")
    load_hf_tokenizer_spy = mocker.spy(QEfficient.cloud.execute, "load_hf_tokenizer")
    cloud_ai_100_exec_kv_spy = mocker.spy(QEfficient.cloud.execute, "cloud_ai_100_exec_kv")

    # Export model
    export(
        model_name=model_setup.model_name,
        hf_token=model_setup.hf_token,
        local_model_dir=model_setup.local_model_dir,
        full_batch_size=model_setup.full_batch_size,
    )

    check_and_assign_cache_dir_spy.assert_called_once()
    get_onnx_model_path_spy.assert_called_once()

    onnx_model_path = get_onnx_model_path_spy.spy_return
    assert os.path.isfile(onnx_model_path)

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

    assert os.path.isdir(qpc_path)

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

    load_hf_tokenizer_spy.assert_called_once()
    cloud_ai_100_exec_kv_spy.assert_called_once()


@pytest.mark.on_qaic
@pytest.mark.cli
def test_export_compile_execute(mocker, setup):
    # testing export -> compile -> infer without full_batch_size
    setup.full_batch_size = None
    setup.enable_qnn = False
    setup.qnn_config = None
    check_export_compile_execute(mocker, setup)


@pytest.mark.on_qaic
@pytest.mark.cli
def test_export_compile_execute_fbs(mocker, setup):
    # testing export -> compile -> infer with full_batch_size
    setup.enable_qnn = False
    setup.qnn_config = None
    check_export_compile_execute(mocker, setup)


@pytest.mark.on_qaic
@pytest.mark.qnn
@pytest.mark.cli
def test_export_compile_execute_qnn(mocker, setup):
    # testing export -> compile -> infer without full_batch_size in QNN environment
    setup.full_batch_size = None
    setup.qnn_config = None
    check_export_compile_execute(mocker, setup)


@pytest.mark.on_qaic
@pytest.mark.qnn
@pytest.mark.cli
def test_export_compile_execute_qnn_fbs(mocker, setup):
    # testing export -> compile -> infer with full_batch_size in QNN environment
    check_export_compile_execute(mocker, setup)

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


def check_export_compile_execute(mocker, model_name, full_batch_size=None, enable_qnn=False):
    check_and_assign_cache_dir_spy = mocker.spy(QEfficient.cloud.export, "check_and_assign_cache_dir")
    get_onnx_model_path_spy = mocker.spy(QEfficient.cloud.export, "get_onnx_model_path")
    load_hf_tokenizer_spy = mocker.spy(QEfficient.cloud.execute, "load_hf_tokenizer")
    cloud_ai_100_exec_kv_spy = mocker.spy(QEfficient.cloud.execute, "cloud_ai_100_exec_kv")

    # Export model
    export(
        model_name=model_name,
        full_batch_size=full_batch_size,
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
        qpc_path=os.path.dirname(onnx_model_path),
        num_cores=16,
        device_group=None,
        custom_io_file_path=(onnx_model_path.parent) / "custom_io.yaml",
        aic_enable_depth_first=True,
        mos=1,
        batch_size=1,
        prompt_len=32,
        ctx_len=128,
        mxfp6=True,
        mxint8=True,
        full_batch_size=full_batch_size,
        enable_qnn=enable_qnn,
    )

    assert os.path.isdir(qpc_path)

    # Execute model
    execute(
        model_name=model_name,
        qpc_path=qpc_path,
        prompt="My name is",
        prompts_txt_file_path="examples/prompts.txt",
        generation_len=20,
        full_batch_size=full_batch_size,
    )

    load_hf_tokenizer_spy.assert_called_once()
    cloud_ai_100_exec_kv_spy.assert_called_once()


@pytest.mark.on_qaic
@pytest.mark.cli
def test_export_compile_execute(mocker):
    # testing export -> compile -> infer without full_batch_size
    check_export_compile_execute(mocker, model_name="gpt2")


@pytest.mark.on_qaic
@pytest.mark.cli
def test_export_compile_execute_fbs(mocker):
    # testing export -> compile -> infer with full_batch_size
    check_export_compile_execute(mocker, model_name="gpt2", full_batch_size=3)


@pytest.mark.on_qaic
@pytest.mark.qnn
@pytest.mark.cli
def test_export_compile_execute_qnn(mocker):
    # testing export -> compile -> infer without full_batch_size in QNN environment
    check_export_compile_execute(mocker, model_name="gpt2", enable_qnn=True)


@pytest.mark.on_qaic
@pytest.mark.qnn
@pytest.mark.cli
def test_export_compile_execute_qnn_fbs(mocker):
    # testing export -> compile -> infer with full_batch_size in QNN environment
    check_export_compile_execute(mocker, model_name="gpt2", full_batch_size=3, enable_qnn=True)

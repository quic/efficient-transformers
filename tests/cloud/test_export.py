# -----------------------------------------------------------------------------
#
# Copyright (c) 2024 Qualcomm Innovation Center, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------

import os

import pytest

import QEfficient
import QEfficient.cloud.export
from QEfficient.cloud.export import main as export


@pytest.mark.cli
def test_export(setup, mocker):
    """
    test_export is a HL export api testing function,
    checks export api code flow, object creations, internal api calls, internal returns.
    ---------
    Parameters:
    setup: is a fixture defined in conftest.py module.
    mocker: mocker is itself a pytest fixture, uses to mock or spy internal functions.
    """
    ms = setup
    check_and_assign_cache_dir_spy = mocker.spy(QEfficient.cloud.export, "check_and_assign_cache_dir")
    get_onnx_model_path_spy = mocker.spy(QEfficient.cloud.export, "get_onnx_model_path")

    export(
        model_name=ms.model_name,
        hf_token=ms.hf_token,
        local_model_dir=ms.local_model_dir,
        full_batch_size=ms.full_batch_size,
    )

    check_and_assign_cache_dir_spy.assert_called_once()
    get_onnx_model_path_spy.assert_called_once()
    assert any(os.path.isfile(x) for x in ms.onnx_model_path())
    assert get_onnx_model_path_spy.spy_return in ms.onnx_model_path()

# -----------------------------------------------------------------------------
#
# Copyright (c)  2023-2024 Qualcomm Innovation Center, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------
import os

import QEfficient
import QEfficient.cloud.export
from QEfficient.cloud.export import main as export
from QEfficient.utils.constants import Constants


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
    get_onnx_model_path_spy = mocker.spy(QEfficient.cloud.export,"get_onnx_model_path")
    export(model_name=ms.model_name,cache_dir=Constants.CACHE_DIR,hf_token=ms.hf_token)
    get_onnx_model_path_spy.assert_called_once()
    assert os.path.isfile(ms.onnx_model_path())
    assert get_onnx_model_path_spy.spy_return == ms.onnx_model_path()

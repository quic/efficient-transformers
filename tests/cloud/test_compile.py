# -----------------------------------------------------------------------------
#
# Copyright (c) 2024 Qualcomm Innovation Center, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------

import os

import QEfficient
import QEfficient.cloud.compile


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
    for onnx_model_path in ms.onnx_model_path():
        if os.path.isfile(onnx_model_path):
            break
    QEfficient.compile(
        onnx_path=onnx_model_path,
        qpc_path=os.path.dirname(ms.qpc_dir_path()),
        num_cores=ms.num_cores,
        device_group=ms.device_group,
        aic_enable_depth_first=ms.aic_enable_depth_first,
        mos=ms.mos,
        batch_size=ms.batch_size,
        prompt_len=ms.prompt_len,
        ctx_len=ms.ctx_len,
        mxfp6=ms.mxfp6,
        mxint8=ms.mxint8,
    )

    assert os.path.isdir(ms.qpc_dir_path())
    assert os.path.isfile(ms.specialization_json_path())
    assert os.path.isfile(ms.custom_io_file_path())
    assert os.path.isdir(ms.qpc_dir_path())

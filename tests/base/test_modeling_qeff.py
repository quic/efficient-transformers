# -----------------------------------------------------------------------------
#
# Copyright (c) 2024 Qualcomm Innovation Center, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------

from types import SimpleNamespace

import onnx
import pytest

from QEfficient.base.modeling_qeff import QEFFBaseModel


def test_compiler_invalid_file(tmp_path):
    qeff_obj = SimpleNamespace()

    invalid_file = tmp_path / "invalid.onnx"
    with open(invalid_file, "wb") as fp:
        fp.write(chr(0).encode() * 100)

    with pytest.raises(RuntimeError):
        QEFFBaseModel._compile(qeff_obj, invalid_file, tmp_path)


def test_compiler_invalid_flag(tmp_path):
    qeff_obj = SimpleNamespace()

    onnx_model = onnx.parser.parse_model("""
    <
        ir_version: 8,
        opset_import: ["": 17]
    >
    test_compiler(float x) => (float y)
    {
        y = Identity(x)
    }
    """)
    valid_file = tmp_path / "valid.onnx"
    onnx.save(onnx_model, valid_file)

    with pytest.raises(RuntimeError):
        QEFFBaseModel._compile(
            qeff_obj, valid_file, tmp_path, convert_tofp16=True, compile_only=True, aic_binary_dir=tmp_path
        )

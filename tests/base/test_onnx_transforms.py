# -----------------------------------------------------------------------------
#
# Copyright (c) 2024 Qualcomm Innovation Center, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
#
# ----------------------------------------------------------------------------

import onnx

from QEfficient.base.onnx_transforms import FP16Clip


def test_fp16clip_transform():
    test_onnx = onnx.parser.parse_model("""
    <
        ir_version: 8,
        opset_import: ["" : 17]
    >
    test_fp16clip (float [n, 32] x) => (float [n, 32] y)
    <
        float val1 = {65505.0},
        int64[1] slice_ends = {2147483647},
        float zero = {0.0}
    >
    {
        mask = Greater(x, zero)
        val2 = Constant<value = float {-1e7}>()
        masked = Where(mask, val1, val2)
        slice_starts = Constant<value = int64[1] {0}>()
        y = Slice(masked, slice_starts, slice_ends)
    }
    """)
    onnx.checker.check_model(test_onnx, True, True, True)
    transformed_onnx, transformed = FP16Clip.apply(test_onnx)
    assert transformed
    assert onnx.numpy_helper.to_array(transformed_onnx.graph.initializer[0]) == 65504.0
    assert onnx.numpy_helper.to_array(transformed_onnx.graph.initializer[1]) == 2147483647
    assert onnx.numpy_helper.to_array(transformed_onnx.graph.node[1].attribute[0].t) == -65504.0

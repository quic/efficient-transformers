# -----------------------------------------------------------------------------
#
# Copyright (c) 2024 Qualcomm Innovation Center, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------

import ast
from types import FunctionType
from typing import Tuple, Union

from onnx import ValueInfoProto
from onnxscript import BOOL, FLOAT, INT64, script
from onnxscript import opset17 as op
from onnxscript.main import script_check
from onnxscript.values import OnnxFunction, Opset

custom_opset = Opset(domain="com.qualcomm.cloud", version=17)


def ConcatTraining(*inputs: Union[ValueInfoProto, str], axis=0, memo={}) -> Tuple[str, OnnxFunction]:
    num_inputs = len(inputs)
    key = str(num_inputs)
    if key in memo:
        return key, memo[key]

    fn_src = f"def ConcatTraining{key}("
    for i in range(num_inputs):
        fn_src += f"input{i}, "
    fn_src += "axis: int):\n"

    fn_src += "\treturn op.Concat("
    for i in range(num_inputs):
        fn_src += f"input{i}, "
    fn_src += "axis=axis), "

    fn_src += "op.Concat("
    for i in range(num_inputs):
        fn_src += f"op.GatherElements(op.Shape(input{i}), op.Unsqueeze(op.Constant(value_int=axis), 0)), "
    fn_src += "axis=0)"

    func_def = ast.parse(fn_src).body[0]
    fn = script_check(func_def, custom_opset, {"op": op}, fn_src, op)

    memo[key] = fn
    return key, fn


@script(custom_opset)
def Dropout(data: FLOAT, ratio: FLOAT, training_mode: BOOL) -> [FLOAT, BOOL]:
    "Dropout implementation with passthrough"
    return data, op.ConstantOfShape(op.Shape(data), [True])


@script(custom_opset)
def DropoutGrad(dy: FLOAT, mask: BOOL, ratio: FLOAT, training_mode: BOOL) -> FLOAT:
    "DropoutGrad implementation"
    return op.Where(mask, dy, op.ConstantOfShape(op.Shape(dy), value=[0.0]))


def FusedMatMul(
    A: ValueInfoProto,
    B: ValueInfoProto,
    transA: int = 0,
    transB: int = 0,
    transBatchA: int = 0,
    transBatchB: int = 0,
    alpha: float = 1.0,
    memo={},
) -> Tuple[str, OnnxFunction]:
    assert transBatchA == 0 and transBatchB == 0, "FusedMatMul with transBatch not supported"
    rankA = 0
    rankB = 0
    if transA:
        rankA = len(A.type.tensor_type.shape.dim)
        permA = list(range(rankA))
        permA[-2], permA[-1] = permA[-1], permA[-2]
    if transB:
        rankB = len(B.type.tensor_type.shape.dim)
        permB = list(range(rankB))
        permB[-2], permB[-1] = permB[-1], permB[-2]
    key = [transA, transB, rankA, rankB]
    if alpha != 1.0:
        key.append(alpha)

    key = "".join(map(str, key))
    if key in memo:
        return key, memo[key]

    fn_src = f"def FusedMatMul{key}(A, B):\n"
    if transA:
        fn_src += f"\tA = op.Transpose(A, perm={permA})\n"
    if transB:
        fn_src += f"\tB = op.Transpose(B, perm={permB})\n"
    if alpha != 1.0:
        fn_src += f"\treturn op.Constant(value_float={alpha}) * op.MatMul(A, B)\n"
    else:
        fn_src += "\treturn op.MatMul(A, B)\n"

    func_def = ast.parse(fn_src).body[0]
    fn = script_check(func_def, custom_opset, {"op": op}, fn_src, op)

    memo[key] = fn
    return key, fn


@script(custom_opset)
def LayerNormalizationGrad(dy: FLOAT, x: FLOAT, scale: FLOAT, axis: int = -1, eps: float = 1e-5):
    "LNG implementation"
    mean = op.ReduceMean(x, axes=[-1])
    centered = x - mean
    var = op.ReduceMean(op.Pow(centered, [2.0]), axes=[-1])
    inv_std_dev = op.Reciprocal(op.Sqrt(var + op.Constant(value_float=1e-5)))
    normalized = centered * inv_std_dev

    a = dy * normalized
    b = dy * scale * inv_std_dev
    c = b * normalized

    mean_b = op.ReduceMean(b, axes=[-1])
    mean_c = op.ReduceMean(c, axes=[-1])

    dx = b - mean_b - normalized * mean_c

    rank = op.Squeeze(op.Shape(op.Shape(dy)), [0])
    one = op.Constant(value_int=1)
    sum_axes = op.Range(op.Constant(value_int=0), rank - one, one)
    scale_grad = op.ReduceSum(a, axes=sum_axes, keepdims=0)
    bias_grad = op.ReduceSum(dy, axes=sum_axes, keepdims=0)

    return dx, scale_grad, bias_grad


@script(custom_opset)
def ReluGrad(dy: FLOAT, x: FLOAT) -> FLOAT:
    "ReluGrad implementation"
    return dy * op.Cast(x > op.Constant(value_float=0.0), to=1)


@script(custom_opset)
def SliceGrad(dy: FLOAT, shape: INT64, starts: INT64, ends: INT64, axes: INT64, steps: INT64):
    "SliceGrad implementation (only works when steps=1)"

    # **Quickfix**
    start_pads = starts
    end_pads = -ends
    zero = op.Constant(value_ints=[0])
    pads = op.Concat(zero, start_pads, zero, zero, end_pads, zero, axis=0)

    return op.Pad(dy, pads, constant_value=op.Constant(value_float=0.0))


@script(custom_opset)
def SoftmaxGrad_13(dy: FLOAT, y: FLOAT, axis: int = -1) -> FLOAT:
    "SoftmaxGrad implementation (only works when axis=-1)"
    # **Needs optimization**
    # (bs1, bs2, ... d)
    shape = op.Shape(y)

    # I (d, d) -> (bs1, bs2, ... d, d)
    zero = op.Constant(value_int=0)
    one = op.Constant(value_int=1)
    axis_len = shape[axis]
    arange = op.Range(zero, axis_len, one)
    eye = op.Unsqueeze(arange, 1) == op.Unsqueeze(arange, 0)
    eye = op.Cast(eye, to=1)

    # eye = op.OneHot(arange, axis_len, [0.0, 1.0])

    # **Quickfix**
    axes = op.Constant(value_ints=[-1])
    minus_2 = op.Constant(value_ints=[-2])

    # J = (bs1, bs2, ... d, d)
    J = op.Unsqueeze(y, axes) * (eye - op.Unsqueeze(y, minus_2))

    # dx = (bs1, bs2, ..., 1, d)
    dx = op.MatMul(op.Unsqueeze(dy, minus_2), J)
    return op.Reshape(dx, shape)


@script(custom_opset)
def SoftmaxCrossEntropyLoss(scores: FLOAT, labels: INT64, reduction: str = "mean") -> [FLOAT, FLOAT]:
    "SoftmaxCrossEntropyLoss implementation"
    log_prob = op.LogSoftmax(scores, axis=1)
    loss = -op.GatherElements(log_prob, op.Unsqueeze(labels, 1), axis=1)
    loss = op.ReduceMean(loss, keepdims=0)
    return loss, log_prob


@script(custom_opset)
def SoftmaxCrossEntropyLossGrad(dy: FLOAT, log_prob: FLOAT, labels: INT64, reduction: str = "mean"):
    "SoftmaxCrossEntropyLossGrad implementation"
    batch_size = op.Shape(log_prob)[0]
    n_cls = op.Shape(log_prob)[1]

    zero = op.Constant(value_int=0)
    one = op.Constant(value_int=1)
    col_indices = op.Range(zero, n_cls, one)
    labels = op.Unsqueeze(labels, 1)
    one_hot = op.Cast(labels == col_indices, to=1)

    # one_hot = op.OneHot(labels, n_cls, [0.0, 1.0])

    dx = (dy / op.Cast(batch_size, to=1)) * (op.Exp(log_prob) - one_hot)
    return dx


@script(custom_opset)
def InPlaceAccumulatorV2(buffer: FLOAT, value: FLOAT, overwrite: BOOL) -> FLOAT:
    "Accumulate the gradients (needs graph changes)"
    if overwrite:
        buffer = op.Add(buffer, value)
    else:
        buffer = value
    return buffer


@script(custom_opset, default_opset=op)
def SGD(weight: FLOAT, grad: FLOAT, lr: float = 0.001) -> FLOAT:
    "SGD implementation"
    # lr_value = op.Constant(value_float=lr)

    # **Quickfix**
    lr_value = op.Constant(value_float=0.001)
    return weight - lr_value * grad


name = fn = ""
exclude = {"script", "script_check"}
functions = {}
dynamic_functions = {}
for name, fn in globals().items():
    if isinstance(fn, OnnxFunction):
        functions[name] = fn
    elif isinstance(fn, FunctionType) and name not in exclude:
        dynamic_functions[name] = fn

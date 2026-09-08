# -----------------------------------------------------------------------------
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------

"""Dynamo export tests for KV-blocked attention represented as ONNX Loop."""

from __future__ import annotations

import collections
from pathlib import Path

import onnx
import pytest
import torch
from torch import nn

from QEfficient.base.onnx_transforms import StaticLoopInputsTransform
from QEfficient.blocking.attention_blocking import BlockingMode
from QEfficient.blocking.blocked_attention_forwards import blocked_kv_attention_forward_headpar_offline_loop
from QEfficient.blocking.blocking_configurator import build_transformer_blocking_config_for_transform
from QEfficient.utils.export_utils import build_dynamo_export_kwargs


class _TinyAttentionModule(nn.Module):
    num_key_value_groups = 1


class _CacheLayer:
    def __init__(self, keys: torch.Tensor, values: torch.Tensor):
        self.keys = keys
        self.values = values


class _LoopKVCache:
    def __init__(self, keys: torch.Tensor, values: torch.Tensor):
        self.layers = [_CacheLayer(keys, values)]


class _KVHeadparLoopWrapper(nn.Module):
    def __init__(self, keys: torch.Tensor, values: torch.Tensor):
        super().__init__()
        self.register_buffer("keys", keys)
        self.register_buffer("values", values)
        self.attn = _TinyAttentionModule()

    def forward(self, query: torch.Tensor, position_ids: torch.Tensor) -> torch.Tensor:
        cache = _LoopKVCache(self.keys, self.values)
        output, _ = blocked_kv_attention_forward_headpar_offline_loop(
            module=self.attn,
            query=query,
            key=query,
            value=query,
            attention_mask=None,
            scaling=0.5,
            num_kv_blocks=2,
            cache_kwargs={"position_ids": position_ids},
            layer_idx=0,
            past_key_value=cache,
            ctx_len=8,
            configured_split=2,
            skip_kv=True,
        )
        return output


def _collect_onnx_nodes(model: onnx.ModelProto) -> list[onnx.NodeProto]:
    nodes = []

    def visit(node_list):
        for node in node_list:
            nodes.append(node)
            for attr in node.attribute:
                if attr.type == onnx.AttributeProto.GRAPH:
                    visit(attr.g.node)
                elif attr.type == onnx.AttributeProto.GRAPHS:
                    for graph in attr.graphs:
                        visit(graph.node)

    visit(model.graph.node)
    for function in model.functions:
        visit(function.node)
    return nodes


@pytest.mark.dynamo
@pytest.mark.dynamo_export
def test_kv_headpar_loop_export_contains_onnx_loop(tmp_path: Path):
    torch.manual_seed(11)
    model = _KVHeadparLoopWrapper(
        keys=torch.randn(2, 2, 8, 4),
        values=torch.randn(2, 2, 8, 4),
    ).eval()
    query = torch.randn(2, 2, 1, 4)
    position_ids = torch.tensor([[0], [3]], dtype=torch.int64)
    onnx_path = tmp_path / "kv_headpar_loop.onnx"

    export_kwargs = build_dynamo_export_kwargs({"external_data": False})
    torch.onnx.export(
        model,
        (query, position_ids),
        onnx_path,
        input_names=["query", "position_ids"],
        output_names=["attn_output"],
        **export_kwargs,
    )

    exported = onnx.load(onnx_path, load_external_data=False)
    counts = collections.Counter(node.op_type for node in _collect_onnx_nodes(exported))
    assert counts["Loop"] == 1


@pytest.mark.dynamo
def test_kv_headpar_loop_config_is_dynamo_only():
    cfg = type(
        "Cfg",
        (),
        {
            "num_attention_heads": 2,
            "num_key_value_heads": 2,
            "hidden_size": 8,
        },
    )()

    dynamo_config = build_transformer_blocking_config_for_transform(
        cfg,
        ctx_len=8,
        seq_len=1,
        bs=1,
        num_devices=1,
        qaic_config={"blocking_mode": "kv_headpar", "num_kv_blocks": 2},
        aic_num_cores=2,
        dynamo=True,
    )
    legacy_config = build_transformer_blocking_config_for_transform(
        cfg,
        ctx_len=8,
        seq_len=1,
        bs=1,
        num_devices=1,
        qaic_config={"blocking_mode": "kv_headpar", "num_kv_blocks": 2},
        aic_num_cores=2,
        dynamo=False,
    )

    assert dynamo_config.mode == BlockingMode.KV_HEADPAR
    assert dynamo_config.use_kv_loop_op is True
    assert legacy_config.mode == BlockingMode.KV_HEADPAR
    assert legacy_config.use_kv_loop_op is False


def test_static_loop_inputs_transform_rewrites_loop_control_inputs():
    body = onnx.helper.make_graph(
        [
            onnx.helper.make_node(
                "Identity",
                ["cond_in"],
                ["cond_out"],
            ),
            onnx.helper.make_node(
                "Identity",
                ["state_in"],
                ["state_out"],
            ),
        ],
        "loop_body",
        [
            onnx.helper.make_tensor_value_info("iter", onnx.TensorProto.INT64, []),
            onnx.helper.make_tensor_value_info("cond_in", onnx.TensorProto.BOOL, []),
            onnx.helper.make_tensor_value_info("state_in", onnx.TensorProto.FLOAT, [1]),
        ],
        [
            onnx.helper.make_tensor_value_info("cond_out", onnx.TensorProto.BOOL, []),
            onnx.helper.make_tensor_value_info("state_out", onnx.TensorProto.FLOAT, [1]),
        ],
    )
    graph = onnx.helper.make_graph(
        [
            onnx.helper.make_node("Greater", ["state", "zero"], ["runtime_cond"]),
            onnx.helper.make_node("Loop", ["runtime_trip", "runtime_cond", "state"], ["loop_out"], body=body),
        ],
        "main",
        [onnx.helper.make_tensor_value_info("state", onnx.TensorProto.FLOAT, [1])],
        [onnx.helper.make_tensor_value_info("loop_out", onnx.TensorProto.FLOAT, [1])],
        [
            onnx.helper.make_tensor("zero", onnx.TensorProto.FLOAT, [1], [0.0]),
            onnx.helper.make_tensor("runtime_trip", onnx.TensorProto.INT64, [], [8]),
        ],
    )
    model = onnx.helper.make_model(graph)

    transformed = StaticLoopInputsTransform.apply(model, num_kv_blocks=2)

    assert transformed
    loop_node = next(node for node in model.graph.node if node.op_type == "Loop")
    initializers = {initializer.name: initializer for initializer in model.graph.initializer}
    assert loop_node.input[:2] == ["qeff_static_loop_trip_count_0", "qeff_static_loop_cond_true_0"]
    assert onnx.numpy_helper.to_array(initializers[loop_node.input[0]]).item() == 2
    assert onnx.numpy_helper.to_array(initializers[loop_node.input[1]]).item() is True


def test_static_loop_inputs_transform_rewrites_function_loop_control_inputs():
    body = onnx.helper.make_graph(
        [
            onnx.helper.make_node("Identity", ["cond_in"], ["cond_out"]),
            onnx.helper.make_node("Identity", ["state_in"], ["state_out"]),
        ],
        "loop_body",
        [
            onnx.helper.make_tensor_value_info("iter", onnx.TensorProto.INT64, []),
            onnx.helper.make_tensor_value_info("cond_in", onnx.TensorProto.BOOL, []),
            onnx.helper.make_tensor_value_info("state_in", onnx.TensorProto.FLOAT, [1]),
        ],
        [
            onnx.helper.make_tensor_value_info("cond_out", onnx.TensorProto.BOOL, []),
            onnx.helper.make_tensor_value_info("state_out", onnx.TensorProto.FLOAT, [1]),
        ],
    )
    function = onnx.helper.make_function(
        "qeff.test",
        "LoopFunction",
        ["runtime_trip", "runtime_cond", "state"],
        ["loop_out"],
        [onnx.helper.make_node("Loop", ["runtime_trip", "runtime_cond", "state"], ["loop_out"], body=body)],
        [onnx.helper.make_opsetid("", 18)],
    )
    graph = onnx.helper.make_graph(
        [],
        "main",
        [onnx.helper.make_tensor_value_info("state", onnx.TensorProto.FLOAT, [1])],
        [onnx.helper.make_tensor_value_info("state", onnx.TensorProto.FLOAT, [1])],
    )
    model = onnx.helper.make_model(graph, functions=[function])

    transformed = StaticLoopInputsTransform.apply(model, num_kv_blocks=4)

    assert transformed
    loop_node = next(node for node in model.functions[0].node if node.op_type == "Loop")
    assert loop_node.input[:2] == ["qeff_static_loop_trip_count_0", "qeff_static_loop_cond_true_0"]
    constant_outputs = {node.output[0] for node in model.functions[0].node if node.op_type == "Constant"}
    assert set(loop_node.input[:2]) <= constant_outputs

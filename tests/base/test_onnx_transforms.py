# -----------------------------------------------------------------------------
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
#
# ----------------------------------------------------------------------------

import numpy as np
import onnx

from QEfficient.base.onnx_transforms import (
    FP16ClipTransform,
    OnnxTransformPipeline,
    RenameWsubNodesTransform,
    SplitTensorsTransform,
)


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

    onnx_transforms = OnnxTransformPipeline(transforms=[FP16ClipTransform])
    transformed_onnx, transformed = onnx_transforms.apply(test_onnx, model_name="")
    assert transformed
    assert onnx.numpy_helper.to_array(transformed_onnx.graph.initializer[0]) == 65504.0
    assert onnx.numpy_helper.to_array(transformed_onnx.graph.initializer[1]) == 2147483647
    assert onnx.numpy_helper.to_array(transformed_onnx.graph.node[1].attribute[0].t) == -65504.0


def test_fp16clip_transform_external(tmp_path):
    external_tensors_file = "fp32_min.raw"
    test_onnx = onnx.parser.parse_model(f"""
    <
        ir_version: 8,
        opset_import: ["" : 17]
    >
    test_fp16clip (float [n, 32] x) => (float [n, 32] y)
    <
        float min_val = [ "location": "{external_tensors_file}" ],
        float zero = {{0.0}}
    >
    {{
        mask = Greater(x, zero)
        y = Where(mask, x, min_val)
    }}
    """)

    # Write onnx and external_data
    onnx_path = tmp_path / "test_fp16_clip_external.onnx"
    onnx.save(test_onnx, onnx_path)
    np.array(-1e10, dtype="float32").tofile(tmp_path / external_tensors_file)
    onnx.checker.check_model(onnx_path, True, True, True)

    onnx_transforms = OnnxTransformPipeline(transforms=[FP16ClipTransform])
    transformed_onnx, transformed = onnx_transforms.apply(test_onnx, model_name="", onnx_base_dir=str(tmp_path))
    assert transformed
    assert onnx.numpy_helper.to_array(transformed_onnx.graph.initializer[0]) == -65504.0


def test_rename_wsub_nodes_transform():
    weighted_ops = [
        ("self_attn.q_proj", "MatMul"),
        ("self_attn.k_proj", "MatMul"),
        ("self_attn.v_proj", "MatMul"),
        ("self_attn.o_proj", "MatMul"),
        ("mlp.gate_proj", "MatMul"),
        ("mlp.up_proj", "MatMul"),
        ("mlp.down_proj", "MatMul"),
        ("input_layernorm", "CustomRMSNorm"),
    ]
    weight_names = [f"model.layers.1.{role}.weight" for role, _ in weighted_ops]
    nodes = [
        onnx.helper.make_node(op_type, ["hidden", weight_name], [f"output_{index}"], name=f"{op_type}_{index}")
        for index, ((_, op_type), weight_name) in enumerate(zip(weighted_ops, weight_names))
    ]
    function = onnx.helper.make_function(
        "test",
        "DecoderLayer",
        ["hidden", *weight_names],
        [nodes[-1].output[0]],
        nodes,
        [onnx.helper.make_opsetid("", 17)],
    )
    model = onnx.helper.make_model(
        onnx.helper.make_graph([], "test", [], []),
        functions=[function],
        opset_imports=[onnx.helper.make_opsetid("", 17)],
    )

    assert RenameWsubNodesTransform.apply(model)
    transformed_function = model.functions[0]
    assert [node.name for node in transformed_function.node] == [
        role.replace(".", "/") + "/" + op_type for role, op_type in weighted_ops
    ]
    assert not RenameWsubNodesTransform.apply(model)


def test_split_tensors_transform(tmp_path):
    external_tensors_file = "tensors.raw"
    test_onnx = onnx.parser.parse_model(f"""
    <
        ir_version: 8,
        opset_import: ["": 17]
    >
    test_split () => ()
    <
        float[1, 32] tensor0 = [ "location": "{external_tensors_file}", "offset": "0", "length": "{32 * 4}" ],
        float[1, 32] tensor1 = [ "location": "{external_tensors_file}", "offset": "{32 * 4}", "length": "{32 * 4}" ],
        float[1, 16] tensor2 = [ "location": "{external_tensors_file}", "offset": "{64 * 4}", "length": "{16 * 4}" ]
    >
    {{
    }}
    """)

    # Write onnx and external_data
    onnx_path = tmp_path / "test_split_pre.onnx"
    onnx.save(test_onnx, onnx_path)
    tensors = np.random.rand(32 + 32 + 16).astype("float32")
    tensors.tofile(tmp_path / external_tensors_file)
    onnx.checker.check_model(onnx_path, True, True, True)

    onnx_transforms = OnnxTransformPipeline(transforms=[SplitTensorsTransform])
    trans_onnx, transformed = onnx_transforms.apply(
        test_onnx,
        model_name="test_split",
        onnx_base_dir=str(tmp_path),
        file_chunk_size=32 * 4,
        size_threshold=16 * 4,
    )

    tensor0_ext_data = onnx.external_data_helper.ExternalDataInfo(trans_onnx.graph.initializer[0])
    assert tensor0_ext_data.location == "test_split_0.onnx.data"

    tensor1_ext_data = onnx.external_data_helper.ExternalDataInfo(trans_onnx.graph.initializer[1])
    assert tensor1_ext_data.location == "test_split_1.onnx.data"

    tensor2 = trans_onnx.graph.initializer[2]
    assert tensor2.data_location == onnx.TensorProto.DataLocation.Value("DEFAULT")
    assert np.all(onnx.numpy_helper.to_array(tensor2) == tensors[-16:])

    # Save and test if all files are saved
    onnx_path = tmp_path / "test_split.onnx"
    onnx.save(trans_onnx, onnx_path)
    assert onnx_path.is_file()
    assert onnx_path.with_name(onnx_path.name.replace(".onnx", "_0.onnx.data")).is_file()
    assert onnx_path.with_name(onnx_path.name.replace(".onnx", "_1.onnx.data")).is_file()

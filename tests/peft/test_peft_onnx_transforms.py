# -----------------------------------------------------------------------------
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------

import textwrap

import onnx

from QEfficient.peft.onnx_transforms import AdapterWeightsToInputsTransform


def test_adapter_weights_to_inputs_transform():
    external_tensors_file = "weight.raw"
    adapter_name = "testAdapter1"
    test_onnx = onnx.parser.parse_model(f"""
    <
        ir_version: 8,
        opset_import: ["" : 17]
    >
    test_adapter_weights (float[n, 32] input) => (float[n, 32] output)
    <
        float[32, 32] layer1_{adapter_name}_weight = [ "location" : "{external_tensors_file}" ],
        float[32, 32] layer2_{adapter_name}_weight = [ "location" : "{external_tensors_file}" ]
    >
    {{
        layer1output = MatMul (input, layer1_{adapter_name}_weight)
        output = MatMul (layer1output, layer2_{adapter_name}_weight)
    }}
    """)

    out_onnx, transformed = AdapterWeightsToInputsTransform.apply(test_onnx, adapter_name=adapter_name)
    assert not transformed

    # Currently the onnx parser doesn't support using "." in identifier
    # Replace _ with .
    for init in test_onnx.graph.initializer:
        init.name = init.name.replace("_", ".")
    for node in test_onnx.graph.node:
        for i, inp in enumerate(node.input):
            node.input[i] = inp.replace("_", ".")
        for i, out in enumerate(node.output):
            node.output[i] = out.replace("_", ".")

    out_onnx, transformed = AdapterWeightsToInputsTransform.apply(test_onnx, adapter_name=adapter_name)
    assert transformed

    assert (
        onnx.printer.to_text(out_onnx)
        == textwrap.dedent("""
    <
       ir_version: 8,
       opset_import: ["" : 17]
    >
    test_adapter_weights (float[n,32] input, float[32,32] "layer1.weight", float[32,32] "layer2.weight") => (float[n,32] output, float[32,32] "layer1.weight_RetainedState", float[32,32] "layer2.weight_RetainedState") {
       layer1output = MatMul (input, "layer1.weight")
       output = MatMul (layer1output, "layer2.weight")
       ["layer1.weight_identity"] "layer1.weight_RetainedState" = Identity ("layer1.weight")
       ["layer2.weight_identity"] "layer2.weight_RetainedState" = Identity ("layer2.weight")
    }
    """).strip()
    )

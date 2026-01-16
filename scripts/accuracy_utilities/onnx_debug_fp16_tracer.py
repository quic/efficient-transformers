# -----------------------------------------------------------------------------
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
#
# ----------------------------------------------------------------------------

import argparse

import numpy as np
import onnx
import onnxruntime
from onnx import TensorProto, helper, shape_inference


def get_all_types(model):
    type_dict = {}
    for vi in model.graph.value_info:
        type_dict[vi.name] = vi.type.tensor_type.elem_type
    for vi in model.graph.output:
        type_dict[vi.name] = vi.type.tensor_type.elem_type
    for vi in model.graph.input:
        type_dict[vi.name] = vi.type.tensor_type.elem_type
    for init in model.graph.initializer:
        type_dict[init.name] = init.data_type
    return type_dict


def expose_all_intermediate_outputs_robust(onnx_path, save_path=None):
    model = onnx.load(onnx_path, load_external_data=False)
    model = shape_inference.infer_shapes(model)

    existing_output_names = {o.name for o in model.graph.output}
    type_dict = get_all_types(model)

    added_outputs = []

    for node in model.graph.node:
        for out_name in node.output:
            if out_name in existing_output_names:
                continue
            dtype = type_dict.get(out_name, TensorProto.FLOAT)  # Default fallback
            vi = helper.make_tensor_value_info(out_name, dtype, shape=None)
            model.graph.output.append(vi)
            added_outputs.append(out_name)

    print(f"Added {len(added_outputs)} outputs to model.")
    if not save_path:
        save_path = onnx_path.replace(".onnx", "_debug.onnx")
    onnx.save(model, save_path)
    print(f"Saved debug model to {save_path}")
    return save_path


def analyze_tensor(name, tensor):
    FP16_MAX = 65504.0
    issues = []
    if not np.issubdtype(tensor.dtype, np.floating):
        return issues
    if np.isnan(tensor).any():
        issues.append("NaNs")
    if np.isinf(tensor).any():
        issues.append("Infs")
    if np.abs(tensor).max() >= FP16_MAX:
        issues.append(f"Overflow > {FP16_MAX}")
    return issues


def analyze_onnx_fp16_overflow(onnx_path, inputs: dict, json_output=None, stop_on_first=False):
    node_path = "fp32_nodes" + (
        onnx_path.split("hunyuan_transformer")[-1].replace(".onnx", ".yaml")
    )  # this is specific to hunyuan
    onnx_path = onnx_path.replace(".onnx", "_debug.onnx")
    model = onnx.load(onnx_path, load_external_data=False)

    # Get fetchable outputs
    valid_outputs = set()
    for vi in list(model.graph.value_info) + list(model.graph.output):
        valid_outputs.add(vi.name)

    output_names = []
    # input_nodes = {}
    for node in model.graph.node:
        for out in node.output:
            if out in valid_outputs:
                output_names.append(out)

    # print(f'OUTPUT NAMES -> {output_names}')
    sess = onnxruntime.InferenceSession(onnx_path)
    outputs = sess.run(output_names, inputs)
    name_to_tensor = dict(zip(output_names, outputs))

    print("\n--- FP16 Overflow Summary ---")
    problem_nodes = []
    for name in output_names:
        tensor = name_to_tensor[name]
        issues = analyze_tensor(name, tensor)
        if issues:
            print(f"[FP16 ISSUE] {name} | shape={tensor.shape} | max={np.abs(tensor).max():.2f} | {issues}")
            problem_nodes.append((name, issues))
            if stop_on_first:
                break  # early exit

    with open(node_path, "w") as f:
        f.write("FP32NodeInstanceNames:\n")
        for node_name in problem_nodes:
            f.write(f" - {node_name[0]}\n")

        nodes_with_flagged_inputs = 0
        # nodes that take flagged nodes as input should also be set to fp32
        for node in model.graph.node:
            for inp in node.input:
                if inp in [name for name, _ in problem_nodes]:
                    for out in node.output:
                        if out not in [name for name, _ in problem_nodes]:
                            nodes_with_flagged_inputs += 1
                            print(f"Problematic Node {inp} is input to {out}")
                            f.write(f" - {out}\n")

    print(f"\nTotal FP16-unsafe outputs: {len(problem_nodes) + nodes_with_flagged_inputs}")
    return problem_nodes


# 4. Main Execution Flow
def main():
    parser = argparse.ArgumentParser(description="Debugger for ONNX models")
    parser.add_argument("--model_path", required=True, help="Path to ONNX model")
    parser.add_argument("--inputs", required=True, type=dict, help="Inputs to Onnx in Dict [str: value]")
    args = parser.parse_args()

    analyze_onnx_fp16_overflow(args.model_path, args.inputs)


if __name__ == "__main__":
    main()

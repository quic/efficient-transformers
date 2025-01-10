# -----------------------------------------------------------------------------
#
# Copyright (c) 2024 Qualcomm Innovation Center, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------

from typing import Optional

import onnx
import yaml
from onnx import helper

"""
    The network specilization file is generated by loading the onnx graph and fecthing the graph inputs and outputs.
"""


def fetch_nodes_info(
    onnx_graph_path: str,
    batch_size: int,
    sequence_length: int,
    context_length: int,
    file_path: str = "custom_io_config.yaml",
    full_batch_size: Optional[int] = None,
    decode_only: Optional[bool] = False,
) -> None:
    # Load the ONNX model
    onnx_model = onnx.load(onnx_graph_path)

    input_nodes = []
    input_nodes_info = []
    final_dict = {}
    output_nodes = []
    output_nodes_info = []
    for node in onnx_model.graph.input:
        input_nodes.append(node.name)
        input_info = {}
        input_info["DataType"] = str(helper.tensor_dtype_to_np_dtype(node.type.tensor_type.elem_type))
        if "past_key" in node.name or "past_value" in node.name:
            input_info["DataType"] = "float16"

        if "batch_index" in node.name:
            if full_batch_size:
                input_info["Shape"] = f"(1, 1), ({full_batch_size}, 1)"
            else:
                input_info["Shape"] = "(1, 1)"
        else:
            shapes = []
            for input_shape in node.type.tensor_type.shape.dim:
                if input_shape.HasField("dim_value"):
                    shape = input_shape.dim_value
                elif input_shape.HasField("dim_param"):
                    shape = input_shape.dim_param
                else:
                    shape = "shape_not_found"
                shapes.append(shape)

            if (
                ("batch_size" in shapes or "full_batch_size" in shapes)
                and ("ctx_len" in shapes or "max_context_len" in shapes)
                and len(shapes) >= 3
            ):
                shapeList = []
                for shape in shapes:
                    if isinstance(shape, str):
                        if "full_batch_size" in shape:
                            if full_batch_size:
                                shapeList.append(full_batch_size)
                            else:
                                print("ERROR: Full batch size is required to generate custom_io_config.yaml")
                                exit()
                        elif "batch_size" in shape:
                            shapeList.append(batch_size)
                        elif shape in ["ctx_len", "max_context_len"]:
                            shapeList.append(context_length)
                    else:
                        shapeList.append(shape)
                shape = str(shapeList).replace("[", "(").replace("]", ")")
            elif "batch_size" in shapes and ("seq_len" in shapes or "prompt_len" in shapes):
                shape_1 = (
                    str(
                        [
                            batch_size if isinstance(shape, str) and "batch_size" in shape else sequence_length
                            for shape in shapes
                        ]
                    )
                    .replace("[", "(")
                    .replace("]", ")")
                )
                if full_batch_size:
                    shape_2 = (
                        str(
                            [
                                full_batch_size if isinstance(shape, str) and "batch_size" in shape else 1
                                for shape in shapes
                            ]
                        )
                        .replace("[", "(")
                        .replace("]", ")")
                    )
                else:
                    shape_2 = (
                        str([batch_size if isinstance(shape, str) and "batch_size" in shape else 1 for shape in shapes])
                        .replace("[", "(")
                        .replace("]", ")")
                    )
                shape = shape_2 if decode_only else shape_1 + "," + shape_2
            elif ("batch_size" in shapes or "full_batch_size" in shapes) and (
                "ctx_len" in shapes or "max_context_len" in shapes
            ):
                shape = (
                    str(
                        [
                            batch_size if isinstance(shape, str) and "batch_size" in shape else context_length
                            for shape in shapes
                        ]
                    )
                    .replace("[", "(")
                    .replace("]", ")")
                )
            input_info["Shape"] = shape
        input_nodes_info.append({"Name": node.name, "Desired Model Parameters": input_info})

    # Prepare output tensor configuration
    for output in onnx_model.graph.output:
        output_nodes.append(output.name)
        output_info = {}
        output_info["DataType"] = str(helper.tensor_dtype_to_np_dtype(output.type.tensor_type.elem_type))
        if "past_key" in output.name or "past_value" in output.name:
            output_info["DataType"] = "float16"
        elif "logits" in output.name:
            output_info["DataType"] = "float32"
        output_nodes_info.append({"Name": output.name, "Desired Model Parameters": output_info})

    # Combine input and output configurations
    final_dict = {"Input Tensor Configuration": input_nodes_info, "Output Tensor Configuration": output_nodes_info}

    # Save the configuration to a YAML file
    try:
        with open(file_path, "w") as yaml_file:
            yaml.dump(final_dict, yaml_file, default_flow_style=False, sort_keys=False)
    except Exception as e:
        print(f"Failed to create YAML File for QNN Network Specialization Configuration{file_path}: {e}")
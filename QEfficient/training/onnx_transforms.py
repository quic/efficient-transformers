# -----------------------------------------------------------------------------
#
# Copyright (c) 2024 Qualcomm Innovation Center, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------

from typing import Set, Tuple

import onnx
from onnx import ModelProto

from QEfficient.base.onnx_transforms import OnnxTransform
from QEfficient.training.training_ops import aisw_opset, cloud_opset, dynamic_functions, functions


class RemoveAdapterNameTransform(OnnxTransform):
    """
    Remove the adapter names from parameters
    """

    @classmethod
    def apply(cls, model: ModelProto, *, adapter_name: str) -> Tuple[ModelProto, bool]:
        # Find nodes with lora weights as inputs
        weight_suffix = f".{adapter_name}.weight"
        lora_weight_nodes = {
            inp: node for node in model.graph.node for inp in node.input if inp.endswith(weight_suffix)
        }
        transformed = False
        for i, weight in enumerate(model.graph.initializer):
            if weight.name.endswith(weight_suffix):
                transformed = True

                # Rename weight input
                new_weight_name = weight.name[: -len(weight_suffix)] + ".weight"
                lora_weight_node = lora_weight_nodes[weight.name]
                for j, inp in enumerate(lora_weight_node.input):
                    if inp.endswith(weight_suffix):
                        lora_weight_node.input[j] = new_weight_name

                # Rename weight initializer
                model.graph.initializer[i].name = new_weight_name

        return model, transformed


class InputsToInitTransform(OnnxTransform):
    """
    Converts inputs into initializers, copying from a reference model.
    """

    @classmethod
    def apply(
        cls, model: ModelProto, *, reference_model_path: str, input_names: Set[str], **kwargs
    ) -> Tuple[ModelProto, bool]:
        reference_model = onnx.load(reference_model_path, load_external_data=False)
        initializers = {init.name: init for init in reference_model.graph.initializer}

        assert input_names.issubset(
            {x.name for x in model.graph.input}
        ), "Some input_names missing as inputs in backward model"
        assert input_names.issubset(initializers.keys()), "input_names missing as initializers in the reference model"

        inputs = []
        transformed = False
        for inp in model.graph.input:
            if inp.name in input_names:
                transformed = True
                init = onnx.TensorProto()
                init.CopyFrom(initializers[inp.name])
                model.graph.initializer.append(init)
            else:
                inputs.append(inp)

        if transformed:
            del model.graph.input[:]
            model.graph.input.extend(inputs)

        return model, transformed


class AddTrainingOpsTransform(OnnxTransform):
    """
    Add required training ops into the model
    """

    @classmethod
    def apply(cls, model: ModelProto, **kwargs) -> Tuple[ModelProto, bool]:
        model_function_names = set()
        model_functions = []

        # Graph modifications
        inputs = {inp.name: inp for inp in model.graph.input}
        outputs = {out.name: out for out in model.graph.output}
        value_info = {vi.name: vi for vi in model.graph.value_info}
        transformed = False
        for node in model.graph.node:
            if node.op_type in functions:
                transformed = True
                node.domain = cloud_opset.domain
                if node.op_type not in model_function_names:
                    if node.op_type == "SimplifiedLayerNormalization":
                        model_functions.append(functions["CustomRMSNorm"].to_function_proto())
                        model_function_names.add("CustomRMSNorm")
                    model_functions.append(functions[node.op_type].to_function_proto())
                    model_function_names.add(node.op_type)

                if node.op_type == "InPlaceAccumulatorV2":
                    # Replace bool outputs with float of same size as accumulation buffer
                    for out_name, out in outputs.items():
                        if out_name.endswith("accumulation.out"):
                            out.type.CopyFrom(inputs[out.name[: -len("out")] + "buffer"].type)

            elif node.op_type in dynamic_functions:
                transformed = True
                node.domain = cloud_opset.domain
                node_inputs = [value_info[x] if x in value_info else x for x in node.input]
                node_attributes = {a.name: onnx.helper.get_attribute_value(a) for a in node.attribute}
                fn_key, fn = dynamic_functions[node.op_type](*node_inputs, **node_attributes)
                node.op_type += fn_key
                if node.op_type not in model_function_names:
                    model_functions.append(fn.to_function_proto())
                    model_function_names.add(node.op_type)

            elif node.op_type == "SplitTraining":
                transformed = True
                node.op_type = "Split"
                node.domain = ""

        if transformed:
            # Set onnx base opset to 17 where LayerNormalization is present
            next(x for x in model.opset_import if x.domain == "").version = 17

            # Add custom domain
            model.opset_import.append(onnx.helper.make_opsetid(cloud_opset.domain, cloud_opset.version))
            model.opset_import.append(onnx.helper.make_opsetid(aisw_opset.domain, aisw_opset.version))

            # Add required functions
            model.functions.extend(model_functions)

            # Inline functions known to cause errors when run as functions
            model = onnx.inliner.inline_selected_functions(model, [("com.qualcomm.cloud", "SoftmaxCrossEntropyLoss")])

        return model, transformed


class AddOptimizerTransform(OnnxTransform):
    """
    Adds specified optimizer into the model
    """

    @classmethod
    def apply(cls, model: ModelProto, *, optimizer: str = "SGD", **kwargs) -> Tuple[ModelProto, bool]:
        transformed = False

        # Remove grad acc inputs
        for i, inp in reversed(list(enumerate(model.graph.input))):
            if inp.name.endswith("_grad.accumulation.buffer"):
                transformed = True
                model.graph.input.pop(i)

        # Replace grad acc outputs with retained-state weights
        for out in model.graph.output:
            if out.name.endswith("grad.accumulation.out"):
                transformed = True
                out.name = out.name.replace("grad.accumulation.out", "RetainedState")

        # Replace InPlaceAccumulator with optimizer
        for node in model.graph.node:
            if node.op_type == "InPlaceAccumulatorV2":
                transformed = True
                node.op_type = optimizer
                node.input.pop(-1)  # Remove lazy_reset_grad
                node.input.append("lr")
                node.input[0] = node.input[0][: -len("_grad.accumulation.buffer")]
                node.output[0] = node.output[0].replace("grad.accumulation.out", "RetainedState")

        if transformed:
            # Add lr input
            model.graph.input.append(onnx.helper.make_tensor_value_info("lr", 1, []))

            # Add optimizer to the model
            model.functions.remove(functions["InPlaceAccumulatorV2"].to_function_proto())
            model.functions.append(functions[optimizer].to_function_proto())

        return model, transformed
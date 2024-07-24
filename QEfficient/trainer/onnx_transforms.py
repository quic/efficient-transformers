from typing import Callable, Dict, List

import onnx
from onnx import helper

from .training_ops import custom_opset, dynamic_functions, functions


class OnnxTransform:
    @classmethod
    def apply(cls, model: onnx.ModelProto) -> onnx.ModelProto:
        raise NotImplementedError


class FP16Clip(OnnxTransform):
    @classmethod
    def apply(cls, model: onnx.ModelProto) -> onnx.ModelProto:
        # Implement FP16Clip transform
        pass


class SplitWeights(OnnxTransform):
    @classmethod
    def apply(cls, model: onnx.ModelProto) -> onnx.ModelProto:
        # Implement SplitWeights transform
        pass


class LoraAdapters(OnnxTransform):
    @classmethod
    def apply(cls, model: onnx.ModelProto) -> onnx.ModelProto:
        # Implement LoraAdapters transform
        pass


class FixTrainingOps(OnnxTransform):
    def __init__(self):
        self.model: onnx.ModelProto = None
        self.inputs: Dict[str, onnx.ValueInfoProto] = {}
        self.outputs: Dict[str, onnx.ValueInfoProto] = {}
        self.value_info: Dict[str, onnx.ValueInfoProto] = {}
        self.model_function_names: set = set()
        self.model_functions: List[onnx.FunctionProto] = []

    @classmethod
    def apply(cls, model: onnx.ModelProto) -> onnx.ModelProto:
        transformer = cls()
        return transformer.fix_training_ops(model)

    def fix_training_ops(self, model: onnx.ModelProto) -> onnx.ModelProto:
        self.model = model
        self._update_opset_version()
        self._initialize_graph_info()
        self._process_nodes()
        self._add_functions_and_domain()
        self._inline_specific_functions()
        return self.model

    def _update_opset_version(self):
        next(x for x in self.model.opset_import if x.domain == "").version = 17

    def _initialize_graph_info(self):
        self.inputs = {inp.name: inp for inp in self.model.graph.input}
        self.outputs = {out.name: out for out in self.model.graph.output}
        self.value_info = {vi.name: vi for vi in self.model.graph.value_info}

    def _process_nodes(self):
        for node in self.model.graph.node:
            if node.op_type in functions:
                self._process_function_node(node)
            elif node.op_type in dynamic_functions:
                self._process_dynamic_function_node(node)

    def _process_function_node(self, node: onnx.NodeProto):
        node.domain = custom_opset.domain
        self._add_function_to_model(node.op_type)
        self._handle_special_cases(node)

    def _process_dynamic_function_node(self, node: onnx.NodeProto):
        node.domain = custom_opset.domain
        node_inputs = [self.value_info[x] if x in self.value_info else x for x in node.input]
        node_attributes = {a.name: helper.get_attribute_value(a) for a in node.attribute}
        fn_key, fn = dynamic_functions[node.op_type](*node_inputs, **node_attributes)
        node.op_type += fn_key
        self._add_function_to_model(node.op_type, fn)

    def _add_function_to_model(self, op_type: str, fn=None):
        if op_type not in self.model_function_names:
            self.model_functions.append(fn.to_function_proto() if fn else functions[op_type].to_function_proto())
            self.model_function_names.add(op_type)

    def _handle_special_cases(self, node: onnx.NodeProto):
        special_cases: Dict[str, Callable] = {
            "InPlaceAccumulatorV2": self._handle_in_place_accumulator,
            "LayerNormalizationGrad": self._handle_layer_normalization_grad,
        }
        if node.op_type in special_cases:
            special_cases[node.op_type](node)

    def _handle_in_place_accumulator(self, node: onnx.NodeProto):
        for out_name, out in self.outputs.items():
            if out_name.endswith("accumulation.out"):
                out.type.CopyFrom(self.inputs[out.name[: -len("out")] + "buffer"].type)

    def _handle_layer_normalization_grad(self, node: onnx.NodeProto):
        # Remove the saved mean and variance inputs, until we have compiler fix verified.
        node.input.pop()
        node.input.pop()

    def _add_functions_and_domain(self):
        self.model.functions.extend(self.model_functions)
        self.model.opset_import.append(helper.make_opsetid(custom_opset.domain, custom_opset.version))

    def _inline_specific_functions(self):
        self.model = onnx.inliner.inline_selected_functions(
            self.model, [("com.qualcomm.cloud", "SoftmaxCrossEntropyLoss")]
        )

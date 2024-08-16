# -----------------------------------------------------------------------------
#
# Copyright (c) 2024 Qualcomm Innovation Center, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
#
# ----------------------------------------------------------------------------

from typing import Tuple

import onnx

from QEfficient.base.onnx_transforms import OnnxTransform


class AdaptersAsInputsTransform(OnnxTransform):
    @classmethod
    def apply(cls, model: onnx.ModelProto, *, adapter_name: str, **kwargs) -> Tuple[onnx.ModelProto, bool]:
        transformed = False
        removed_initializers = []
        for i, weight in enumerate(model.graph.initializer):
            if adapter_name in weight.name:
                transformed = True
                type_proto = onnx.helper.make_tensor_type_proto(weight.data_type, shape=list(weight.dims))
                inp = onnx.ValueInfoProto(name=weight.name, type=type_proto)
                out = onnx.ValueInfoProto(name=weight.name + "_RetainedState", type=type_proto)

                node = onnx.helper.make_node("Identity", [inp.name], [out.name], weight.name + "_identity")

                model.graph.input.append(inp)
                model.graph.output.append(out)
                model.graph.node.append(node)

                removed_initializers.append(i)

        if transformed:
            for i in sorted(removed_initializers, reverse=True):
                model.graph.initializer.pop(i)

            # Add adapter_name in metadata_props
            model.metadata_props.append(onnx.StringStringEntryProto(key="adapter_name", value=adapter_name))

        return model, transformed

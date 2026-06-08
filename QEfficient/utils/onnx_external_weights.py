# -----------------------------------------------------------------------------
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
#
# ----------------------------------------------------------------------------

import os
from pathlib import Path
from typing import Dict, Iterable, Tuple

import onnx
import torch
from onnx import TensorProto, external_data_helper

from QEfficient.utils.mem_profile import format_bytes

_TORCH_TO_ONNX_DTYPE = {
    torch.float16: TensorProto.FLOAT16,
    torch.float32: TensorProto.FLOAT,
    torch.float64: TensorProto.DOUBLE,
    torch.bfloat16: TensorProto.BFLOAT16,
    torch.int8: TensorProto.INT8,
    torch.uint8: TensorProto.UINT8,
    torch.int16: TensorProto.INT16,
    torch.int32: TensorProto.INT32,
    torch.int64: TensorProto.INT64,
    torch.bool: TensorProto.BOOL,
}


def low_memory_external_initializers_enabled() -> bool:
    return os.environ.get("QEFF_LOW_MEMORY_ONNX_EXPORT", "").lower() in {"1", "true", "yes", "on"}


def iter_named_tensors(module: torch.nn.Module) -> Iterable[Tuple[str, torch.Tensor]]:
    yield from module.named_parameters(remove_duplicate=False)
    yield from module.named_buffers(remove_duplicate=False)


def attach_external_initializers_from_torch(
    model: onnx.ModelProto,
    torch_model: torch.nn.Module,
    onnx_path: Path,
    data_file_name: str | None = None,
) -> Dict[str, object]:
    tensor_map = {name: tensor for name, tensor in iter_named_tensors(torch_model)}
    initializer_names = set()
    data_file_name = data_file_name or f"{onnx_path.name}.data"
    data_path = onnx_path.parent / data_file_name
    data_path.parent.mkdir(parents=True, exist_ok=True)

    total_bytes = 0
    attached_count = 0
    missing_names = []
    unsupported_names = []

    with open(data_path, "wb") as data_file:
        for graph_input in list(model.graph.input):
            name = graph_input.name
            tensor = tensor_map.get(name)
            if tensor is None:
                continue
            if tensor.dtype not in _TORCH_TO_ONNX_DTYPE:
                unsupported_names.append(f"{name}:{tensor.dtype}")
                continue

            detached = tensor.detach()
            if detached.device.type != "cpu":
                detached = detached.cpu()
            if not detached.is_contiguous():
                detached = detached.contiguous()

            offset = data_file.tell()
            if detached.dtype is torch.bfloat16:
                tensor_bytes = detached.view(torch.uint16).numpy()
            else:
                tensor_bytes = detached.numpy()
            data_file.write(memoryview(tensor_bytes))
            length = data_file.tell() - offset

            initializer = TensorProto()
            initializer.name = name
            initializer.data_type = _TORCH_TO_ONNX_DTYPE[detached.dtype]
            initializer.dims.extend(list(detached.shape))
            initializer.raw_data = b""
            external_data_helper.set_external_data(
                initializer,
                location=data_file_name,
                offset=offset,
                length=length,
            )
            model.graph.initializer.append(initializer)
            initializer_names.add(name)
            attached_count += 1
            total_bytes += length

    retained_inputs = [graph_input for graph_input in model.graph.input if graph_input.name not in initializer_names]
    del model.graph.input[:]
    model.graph.input.extend(retained_inputs)

    for graph_input in model.graph.input:
        name = graph_input.name
        if name in tensor_map and name not in initializer_names:
            missing_names.append(name)

    return {
        "external_initializer_count": attached_count,
        "external_initializer_bytes": format_bytes(total_bytes),
        "external_data_path": str(data_path),
        "external_data_file_size": format_bytes(data_path.stat().st_size if data_path.exists() else 0),
        "missing_initializer_inputs": missing_names[:10],
        "missing_initializer_input_count": len(missing_names),
        "unsupported_initializer_inputs": unsupported_names[:10],
        "unsupported_initializer_input_count": len(unsupported_names),
    }

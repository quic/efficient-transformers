# -----------------------------------------------------------------------------
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------

from pathlib import Path
from typing import Dict, List, Optional, Union
from warnings import warn
import importlib
import sys
import platform

import numpy as np

# try:
#     import qaicrt
# except ImportError:
#     import platform
#     import sys

#     sys.path.append(f"/opt/qti-aic/dev/lib/{platform.machine()}")
#     import qaicrt


class QAICInferenceSession:
    _qaicrt = None
    _aicapi = None

    @property
    def qaicrt(self):
        if QAICInferenceSession._qaicrt is None:
            try:
                QAICInferenceSession._qaicrt = importlib.import_module("qaicrt")
            except ImportError:
                sys.path.append(f"/opt/qti-aic/dev/lib/{platform.machine()}")
                QAICInferenceSession._qaicrt = importlib.import_module("qaicrt")
        return QAICInferenceSession._qaicrt

    @property
    def aicapi(self):
        if QAICInferenceSession._aicapi is None:
            try:
                QAICInferenceSession._aicapi = importlib.import_module("QAicApi_pb2")
            except ImportError:
                sys.path.append("/opt/qti-aic/dev/python")
                QAICInferenceSession._aicapi = importlib.import_module("QAicApi_pb2")
        return QAICInferenceSession._aicapi

    def __init__(
        self,
        qpc_path: Union[Path, str],
        device_ids: Optional[List[int]] = None,
        activate: bool = True,
        enable_debug_logs: bool = False,
    ):
        """
        Initialise for QAIC inference Session
        ---------

        :qpc_path: str. Path to the save generated binary file after compilation.
        :device_ids: List[int]. Device Ids to be used for compilation. if devices > 1, it enables multiple card setup.
        :activate: bool. If false, activation will be disabled. Default=True.
        :enable_debug_logs: bool. If True, It will enable debug logs. Default=False.
        """
        qaicrt = self.qaicrt
        aicapi = self.aicapi
        # Build the dtype map one time, not on every property access
        self.aic_to_np_dtype_mapping = {
            aicapi.FLOAT_TYPE: np.dtype(np.float32),
            aicapi.FLOAT_16_TYPE: np.dtype(np.float16),
            aicapi.INT8_Q_TYPE: np.dtype(np.int8),
            aicapi.UINT8_Q_TYPE: np.dtype(np.uint8),
            aicapi.INT16_Q_TYPE: np.dtype(np.int16),
            aicapi.INT32_Q_TYPE: np.dtype(np.int32),
            aicapi.INT32_I_TYPE: np.dtype(np.int32),
            aicapi.INT64_I_TYPE: np.dtype(np.int64),
            aicapi.INT8_TYPE: np.dtype(np.int8),
        }

        # Load QPC
        if device_ids is not None:
            devices = self.qaicrt.QIDList(device_ids)
            self.context = self.qaicrt.Context(devices)
            self.queue = self.qaicrt.Queue(self.context, device_ids[0])
        else:
            self.context = self.qaicrt.Context()
            self.queue = self.qaicrt.Queue(self.context, 0)  # Async API

        if enable_debug_logs:
            if self.context.setLogLevel(self.qaicrt.QLogLevel.QL_DEBUG) != self.qaicrt.QStatus.QS_SUCCESS:
                raise RuntimeError("Failed to setLogLevel")

        qpc = self.qaicrt.Qpc(str(qpc_path))

        # Load IO Descriptor
        iodesc = self.aicapi.IoDesc()
        status, iodesc_data = qpc.getIoDescriptor()
        if status != self.qaicrt.QStatus.QS_SUCCESS:
            raise RuntimeError("Failed to getIoDescriptor")
        iodesc.ParseFromString(bytes(iodesc_data))

        self.allowed_shapes = [
            [(self.aic_to_np_dtype_mapping[x.type].itemsize, list(x.dims)) for x in allowed_shape.shapes]
            for allowed_shape in iodesc.allowed_shapes
        ]
        self.bindings = iodesc.selected_set.bindings
        self.binding_index_map = {binding.name: binding.index for binding in self.bindings}

        # Create and load Program
        prog_properties = self.qaicrt.QAicProgramProperties()
        prog_properties.SubmitRetryTimeoutMs = 60_000
        if device_ids and len(device_ids) > 1:
            prog_properties.devMapping = ":".join(map(str, device_ids))

        self.program = self.qaicrt.Program(self.context, None, qpc, prog_properties)
        if self.program.load() != self.qaicrt.QStatus.QS_SUCCESS:
            raise RuntimeError("Failed to load program")

        if activate:
            self.activate()

        # Create input qbuffers and buf_dims
        self.qbuffers = [self.qaicrt.QBuffer(bytes(binding.size)) for binding in self.bindings]
        self.buf_dims = self.qaicrt.BufferDimensionsVecRef(
            [(self.aic_to_np_dtype_mapping[binding.type].itemsize, list(binding.dims)) for binding in self.bindings]
        )

    @property
    def input_names(self) -> List[str]:
        return [binding.name for binding in self.bindings if binding.dir == self.aicapi.BUFFER_IO_TYPE_INPUT]

    @property
    def output_names(self) -> List[str]:
        return [binding.name for binding in self.bindings if binding.dir == self.aicapi.BUFFER_IO_TYPE_OUTPUT]

    def activate(self):
        self.program.activate()
        self.execObj = self.qaicrt.ExecObj(self.context, self.program)

    def deactivate(self):
        del self.execObj
        self.program.deactivate()

    def set_buffers(self, buffers: Dict[str, np.ndarray]):
        for buffer_name, buffer in buffers.items():
            if buffer_name not in self.binding_index_map:
                warn(f'Buffer: "{buffer_name}" not found')
                continue
            buffer_index = self.binding_index_map[buffer_name]
            self.qbuffers[buffer_index] = self.qaicrt.QBuffer(buffer.tobytes())
            self.buf_dims[buffer_index] = (
                buffer.itemsize,
                buffer.shape if len(buffer.shape) > 0 else (1,),
            )

    def skip_buffers(self, skipped_buffer_names: List[str]):
        self.set_buffers({k: np.array([]) for k in skipped_buffer_names})

    def run(self, inputs: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        self.set_buffers(inputs)
        if self.execObj.setData(self.qbuffers, self.buf_dims) != self.qaicrt.QStatus.QS_SUCCESS:
            raise MemoryError("Failed to setData")

        if self.queue.enqueue(self.execObj) != self.qaicrt.QStatus.QS_SUCCESS:
            raise MemoryError("Failed to enqueue")

        if self.execObj.waitForCompletion() != self.qaicrt.QStatus.QS_SUCCESS:
            error_message = "Failed to run"

            if self.allowed_shapes:
                error_message += "\n\n(Only if 'No matching dimension found' error is present above)"
                error_message += "\nAllowed shapes:"
                for i, allowed_shape in enumerate(self.allowed_shapes):
                    error_message += f"\n{i}\n"
                    for binding, (elemsize, shape), (_, passed_shape) in zip(
                        self.bindings, allowed_shape, self.buf_dims
                    ):
                        if passed_shape == [0]:
                            if not binding.is_partial_buf_allowed:
                                warn(f"Partial buffer not allowed for: {binding.name}")
                            continue
                        error_message += f"{binding.name}:\t{elemsize}\t{shape}\n"
                error_message += "\n\nPassed shapes:\n"
                for binding, (elemsize, shape) in zip(self.bindings, self.buf_dims):
                    if shape == [0]:
                        continue
                    error_message += f"{binding.name}:\t{elemsize}\t{shape}\n"
            raise ValueError(error_message)

        status, output_qbuffers = self.execObj.getData()
        if status != self.qaicrt.QStatus.QS_SUCCESS:
            raise MemoryError("Failed to getData")

        outputs = {}
        for output_name in self.output_names:
            buffer_index = self.binding_index_map[output_name]
            if self.qbuffers[buffer_index].size == 0:
                continue
            outputs[output_name] = np.frombuffer(
                bytes(output_qbuffers[buffer_index]),
                self.aic_to_np_dtype_mapping[self.bindings[buffer_index].type],
            ).reshape(self.buf_dims[buffer_index][1])
        return outputs

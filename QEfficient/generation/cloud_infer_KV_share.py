# -----------------------------------------------------------------------------
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------
"""
=======================
optimised KV-cache handoff via shared DMA buffers and slice operations.

Design
------
KV handoff (zero-copy)
----------------------
On the last prefill chunk, before calling np_run_pipeline():

    session.set_data_for_kv_handoff(
        kv_cache_buffers,                          # shared numpy arrays
        [("batch_index", bidx), ("ctx_start", 0)], # slice offsets
        exec_obj_idx,
        session.kv_only_buff_map,                  # KV outputs only (no logits)
    )

"""

from __future__ import annotations

import json
import logging
import os
import platform
import sys
from pathlib import Path
from queue import Queue
from typing import Any

import numpy as np

# ── SDK imports ───────────────────────────────────────────────────────────────
try:
    import qaicrt
except ImportError:
    sys.path.append(f"/opt/qti-aic/dev/lib/{platform.machine()}")
    import qaicrt

try:
    import QAicApi_pb2 as aicapi
except ImportError:
    sys.path.append("/opt/qti-aic/dev/python")
    import QAicApi_pb2 as aicapi

logger = logging.getLogger(__name__)

# ── Env-var names (match vllm convention) ────────────────────────────────────
_PREFILL_QUEUE_LEN_ENV = "VLLM_QAIC_PREFILL_QUEUE_LEN"
_EXEC_TIMEOUT_ENV = "VLLM_QAIC_ASYNC_SCHEDULING_EXEC_TIMEOUT"

# ── dtype mapping ─────────────────────────────────────────────────────────────
AIC_TO_NP: dict[int, np.dtype] = {
    getattr(aicapi, "BFLOAT16_TYPE", 11): np.dtype(np.float16),
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


# ─────────────────────────────────────────────────────────────────────────────
# Helper: build a descriptive error message when waitForCompletion fails
# ─────────────────────────────────────────────────────────────────────────────
def _build_completion_error(bindings, allowed_shapes: list, buf_dims_for_idx) -> str:
    msg = "Failed to run"
    if not allowed_shapes:
        return msg
    msg += '\n\n(Only if "No matching dimension found" error is present above)'
    msg += "\nAllowed shapes:"
    for i, allowed_shape in enumerate(allowed_shapes):
        msg += f"\n{i}\n"
        for binding, (elemsize, shape), (_, passed_shape) in zip(bindings, allowed_shape, buf_dims_for_idx):
            if passed_shape[0] == 0:
                if not binding.is_partial_buf_allowed:
                    logger.warning("Partial buffer not allowed for: %s", binding.name)
                continue
            msg += f"{binding.name}:\t{elemsize}\t{shape}\n"
    msg += "\n\nPassed shapes:\n"
    for binding, (elemsize, shape) in zip(bindings, buf_dims_for_idx):
        if shape[0] == 0:
            continue
        msg += f"{binding.name}:\t{elemsize}\t{shape}\n"
    return msg


# ─────────────────────────────────────────────────────────────────────────────
class QAICInferenceSession:
    """
    Optimised QAIC inference session with KV-cache handoff via shared DMA
    buffers and slice operations.

    Parameters
    ----------
    qpc_path : str | Path
        Path to the compiled QPC binary directory.
    full_batch_size : int
        Maximum batch size the QPC was compiled for.  Used to compute the
        batch_index modulo when wiring KV slices.
    device_ids : list[int] | None
        Device IDs to use.  Single-device → non-MQ path; multi-device → MQ
        via devMapping.  None → default context (device 0).
    activate : bool
        Activate the program immediately on construction.
    enable_debug_logs : bool
        Enable QAIC runtime debug logging.
    stages : int
        Number of pipeline stages for pipelined prefill.
        prefill exec-obj pool size = stages + 1  (overridable via env-var).
    cluster_id : str | None
        "prefill"  → only prefill exec-objs, no decode slot.
        "decode"   → only one decode exec-obj, no prefill pool.
        None       → combined mode: one decode slot + prefill pool.
    """

    def __init__(
        self,
        qpc_path: str | Path,
        full_batch_size: int = 1,
        device_ids: list[int] | None = None,
        activate: bool = True,
        enable_debug_logs: bool = False,
        stages: int | None = 1,
        cluster_id: str | None = None,
    ) -> None:
        self.stages: int = stages if stages is not None else 1
        self.cluster_id = cluster_id
        self.full_batch_size = full_batch_size
        self.exec_timeout: int = int(os.getenv(_EXEC_TIMEOUT_ENV, 300))

        # ── Exec-obj pool layout ─────────────────────────────────────────────
        # Layout in self.execObj list:
        #   [0 .. decode_num)          → decode slot(s)
        #   [decode_num .. queue_len)  → prefill pool
        #
        # cluster_id="decode"  : 1 decode slot, 0 prefill
        # cluster_id="prefill" : 0 decode slots, stages+1 prefill
        # cluster_id=None      : 1 decode slot + 1 prefill slot (combined)
        if cluster_id == "decode":
            self.prefill_num_execObj: int = 0
            self.decode_num_execObj: int = 1
            self.decode_execObj_idx: int | None = 0
        elif cluster_id == "prefill":
            self.prefill_num_execObj = int(os.getenv(_PREFILL_QUEUE_LEN_ENV, self.stages + 1))
            self.decode_num_execObj = 0
            self.decode_execObj_idx = None
        else:
            self.prefill_num_execObj = int(os.getenv(_PREFILL_QUEUE_LEN_ENV, 1))
            self.decode_num_execObj = 1
            self.decode_execObj_idx = 0

        self.queue_len: int = self.prefill_num_execObj + self.decode_num_execObj

        # Thread-safe queue of available prefill exec-obj indices
        self.prefill_available_exec_objs: Queue[int] = Queue()
        _prefill_start = self.decode_num_execObj
        for i in range(_prefill_start, _prefill_start + self.prefill_num_execObj):
            self.prefill_available_exec_objs.put(i)

        logger.debug(
            "cluster_id=%s  prefill_exec_objs=%d  decode_exec_objs=%d",
            cluster_id,
            self.prefill_num_execObj,
            self.decode_num_execObj,
        )

        # ── Context + Queue ──────────────────────────────────────────────────
        if device_ids is not None:
            devices = qaicrt.QIDList(device_ids)
            self.context = qaicrt.Context(devices)
            self.queue = qaicrt.Queue(self.context, device_ids[0])
        else:
            self.context = qaicrt.Context()
            self.queue = qaicrt.Queue(self.context, 0)

        if enable_debug_logs:
            assert self.context.setLogLevel(qaicrt.QLogLevel.QL_DEBUG) == qaicrt.QStatus.QS_SUCCESS, (
                "Failed to setLogLevel"
            )
        _qprops = qaicrt.QAicQueueProperties()
        _qprops.numThreadsPerQueue = 1
        self.queue.initProperties(_qprops)

        # ── Load QPC + IO descriptor ─────────────────────────────────────────
        qpc = qaicrt.Qpc(str(qpc_path))
        iodesc = aicapi.IoDesc()
        status, iodesc_data = qpc.getIoDescriptor()
        assert status == qaicrt.QStatus.QS_SUCCESS, "Failed to getIoDescriptor"
        iodesc.ParseFromString(bytes(iodesc_data))

        self.allowed_shapes: list = [
            [(AIC_TO_NP[x.type].itemsize, list(x.dims)) for x in allowed_shape.shapes]
            for allowed_shape in iodesc.allowed_shapes
        ]
        self.bindings = iodesc.selected_set.bindings
        self.binding_index_map: dict[str, int] = {b.name: b.index for b in self.bindings}

        # ── Program ──────────────────────────────────────────────────────────
        prog_props = qaicrt.QAicProgramProperties()
        prog_props.dataPathTimeoutMs = 60_000

        _dev_id_non_mq = None
        if device_ids:
            if len(device_ids) == 1:
                _dev_id_non_mq = device_ids[0]
            else:
                prog_props.devMapping = ":".join(map(str, device_ids))

        self.program = qaicrt.Program(self.context, _dev_id_non_mq, qpc, prog_props)
        assert self.program.load() == qaicrt.QStatus.QS_SUCCESS, "Failed to load program"

        self.activate_done = False
        if activate:
            self.activate()

        # ── Per-exec-obj qbuffers and buf_dims ───────────────────────────────
        self.qbuffers: list[list[qaicrt.QBuffer]] = [
            [qaicrt.QBuffer(bytes(b.size)) for b in self.bindings] for _ in range(self.queue_len)
        ]
        self.buf_dims: list[qaicrt.BufferDimensionsVecRef] = [
            qaicrt.BufferDimensionsVecRef([(AIC_TO_NP[b.type].itemsize, list(b.dims)) for b in self.bindings])
            for _ in range(self.queue_len)
        ]

        # ── KV slicing spec (for zero-copy DMA handoff) ──────────────────────
        # Created once from the program; reused across all setDataWithSlices calls.
        self.kv_slicing_spec_handle = None
        self.repetition_penalty_spec_handle = None

        if "past_key.0_RetainedState" in self.binding_index_map:
            _rs_binding = self.bindings[self.binding_index_map["past_key.0_RetainedState"]]
            # kv_shape / kv_size used externally to allocate shared KV buffers
            self.kv_shape: list[int] = list(_rs_binding.dims)
            self.kv_shape[0] = 1  # per-batch-slot shape
            self.kv_size: np.dtype = AIC_TO_NP[_rs_binding.type]
            self.kv_slicing_spec_handle = self.get_slicing_spec_handle(self.get_json_for_kv_cache_slicing())

        if "past_repetition_penalty_buffer" in self.input_names:
            self.repetition_penalty_map: list[tuple[str, int]] = [
                (
                    "past_repetition_penalty_buffer",
                    self.binding_index_map["past_repetition_penalty_buffer"],
                )
            ]
            self.repetition_penalty_spec_handle = self.get_slicing_spec_handle(
                self.get_json_for_repetition_penalty_slicing()
            )

        # ── Buffer maps (sorted by layer then key/value) ─────────────────────
        # decode_buff_map  : input  KV buffers  (past_key.*, past_value.*)
        # prefill_buff_map : output KV retained states + logits (logits last)
        def _kv_sort_key(item: tuple[str, int]) -> tuple[int, int]:
            name = item[0]
            layer = int(name.split(".")[1]) if "." in name else 0
            kind = 0 if name.startswith("past_key") else 1
            return (layer, kind)

        self.decode_buff_map: list[tuple[str, int]] = sorted(
            [
                (name, self.binding_index_map[name])
                for name in self.input_names
                if name.startswith("past_key") or name.startswith("past_value")
            ],
            key=_kv_sort_key,
        )

        # decode_rs_buff_map : output RetainedState KV bindings (past_key.*_RetainedState)
        # Stores OUTPUT binding indices — used by set_data_for_kv_handoff on the
        # decode session to wire RetainedState outputs directly into kv_cache arrays.
        self.decode_rs_buff_map: list[tuple[str, int]] = sorted(
            [
                (name.replace("_RetainedState", ""), self.binding_index_map[name])
                for name in self.output_names
                if name.endswith("_RetainedState")
            ],
            key=_kv_sort_key,
        )

        # decode_rs_kv_only_buff_map : subset of decode_rs_buff_map containing only past_key.* / past_value.* RetainedState output entries.
        self.decode_rs_kv_only_buff_map: list[tuple[str, int]] = [
            entry
            for entry in self.decode_rs_buff_map
            if entry[0].startswith("past_key") or entry[0].startswith("past_value")
        ]

        self.prefill_buff_map: list[tuple[str, int]] = sorted(
            [
                (name.replace("_RetainedState", ""), self.binding_index_map[name])
                for name in self.output_names
                if name.endswith("_RetainedState")
            ],
            key=_kv_sort_key,
        )
        # Append logits at the end of prefill_buff_map
        for name in self.output_names:
            if name.startswith("log"):
                self.prefill_buff_map.append((name, self.binding_index_map[name]))

        # kv_only_buff_map : subset of prefill_buff_map containing only
        # past_key.* / past_value.* RetainedState output entries.
        # prefill_buff_map may also contain vision RetainedState outputs
        # (e.g. vision_embeds_RetainedState, deepstack_features_RetainedState)
        # for VLMs compiled with kv_offload=True.  Using prefill_buff_map[:-1]
        # directly for KV handoff would mismatch buffer sizes on those models.
        # kv_only_buff_map is always safe for both text-only and VLM QPCs.
        self.kv_only_buff_map: list[tuple[str, int]] = [
            entry
            for entry in self.prefill_buff_map
            if entry[0].startswith("past_key") or entry[0].startswith("past_value")
        ]

        # ── Skip KV buffers by default (retained-state managed via handoff) ──
        for slot in range(self.queue_len):
            self.skip_buffers([n for n in self.input_names if n.startswith("past_")], slot)
            self.skip_buffers([n for n in self.output_names if n.endswith("_RetainedState")], slot)

    # ── Properties ────────────────────────────────────────────────────────────

    @property
    def input_names(self) -> list[str]:
        return [b.name for b in self.bindings if b.dir == aicapi.BUFFER_IO_TYPE_INPUT]

    @property
    def output_names(self) -> list[str]:
        return [b.name for b in self.bindings if b.dir == aicapi.BUFFER_IO_TYPE_OUTPUT]

    # ── Lifecycle ─────────────────────────────────────────────────────────────

    def activate(self) -> None:
        """Activate the program and create one ExecObj per pool slot."""
        self.activate_done = True
        self.program.activate()
        self.execObj: list[qaicrt.ExecObj] = [qaicrt.ExecObj(self.context, self.program) for _ in range(self.queue_len)]

    def deactivate(self) -> None:
        """Deactivate the program and release ExecObjs."""
        if self.activate_done:
            del self.execObj
            self.program.deactivate()
            self.activate_done = False

    # ── Slicing spec helpers ──────────────────────────────────────────────────

    def get_json_for_kv_cache_slicing(self) -> str:
        """
        JSON spec for setDataWithSlices on past_key.* / past_value.*.
        Slice dimensions: [batch_index, :, ctx_start, :] — allows writing a
        single batch slot starting at a given context position without copying
        the full KV tensor.
        """
        elem_size = AIC_TO_NP[self.bindings[self.binding_index_map["past_key.0"]].type].itemsize
        spec = {
            "BufferSpecs": [
                {
                    "Name": "past_key.*",
                    "ElemSize": elem_size,
                    "DimSpecs": [
                        {"start": "batch_index"},  # dim 0: which batch slot to start at
                        {"start": 0},  # dim 1: all heads, start at 0
                        {"start": "ctx_start"},  # dim 2: which context position to start at
                        {"start": 0},
                    ],
                },
                {
                    "Name": "past_value.*",
                    "ElemSize": elem_size,
                    "DimSpecs": [
                        {"start": "batch_index"},
                        {"start": 0},
                        {"start": "ctx_start"},
                        {"start": 0},
                    ],
                },
            ]
        }
        return json.dumps(spec)

    def get_json_for_repetition_penalty_slicing(self) -> str:
        """JSON spec for setDataWithSlices on past_repetition_penalty_buffer."""
        elem_size = AIC_TO_NP[self.bindings[self.binding_index_map["past_repetition_penalty_buffer"]].type].itemsize
        spec = {
            "BufferSpecs": [
                {
                    "Name": "past_repetition_penalty_buffer",
                    "ElemSize": elem_size,
                    "DimSpecs": [
                        {"start": "batch_index"},
                        {"start": 0},
                    ],
                }
            ]
        }
        return json.dumps(spec)

    def get_slicing_spec_handle(self, buffer_spec_json: str):
        """Create and return a slicing spec handle from the program."""
        status, handle = self.program.createSlicingSpecHandle(buffer_spec_json)
        assert status == qaicrt.QStatus.QS_SUCCESS, "Failed to createSlicingSpecHandle"
        return handle

    # ── Buffer helpers ────────────────────────────────────────────────────────

    def get_bindings(self, binding_names: list[str]) -> list:
        return [b for b in self.bindings if b.name in binding_names]

    def get_bindings_shapes(self, binding_names: list[str]) -> dict[str, list[list[int]]]:
        """Return all allowed shapes for the requested buffer names."""
        result: dict[str, list[list[int]]] = {}
        for name in binding_names:
            if name not in self.binding_index_map:
                logger.warning("Unable to find binding: %s", name)
                continue
            idx = self.binding_index_map[name]
            result[name] = [shape[idx][1] for shape in self.allowed_shapes]
        return result

    def get_logits_ndim(self) -> int:
        """Return the number of dimensions of the logits output binding.

        Reads directly from the selected_set binding metadata rather than
        allowed_shapes, because allowed_shapes is empty for single-specialisation
        QPCs (the common case) and would always return the default 3.
        """
        if "logits" not in self.binding_index_map:
            logger.warning("logits binding not found, defaulting ndim to 3")
            return 3
        return len(self.bindings[self.binding_index_map["logits"]].dims)

    def set_buffers(self, buffers: dict[str, np.ndarray], index: int = 0) -> None:
        """
        Copy numpy arrays into the qbuffer/buf_dims for exec-obj slot `index`.
        Ensures contiguous memory layout before wrapping in QBuffer.
        """
        for name, buf in buffers.items():
            if name not in self.binding_index_map:
                logger.warning("Buffer: %s not found", name)
                continue
            buf_idx = self.binding_index_map[name]
            contiguous = np.ascontiguousarray(buf)
            if contiguous is not buf:
                logger.warning("Non-contiguous buffer for '%s'; copying to contiguous.", name)
                buffers[name] = contiguous
            buf = contiguous
            self.qbuffers[index][buf_idx] = qaicrt.QBuffer(buf)
            self.buf_dims[index][buf_idx] = (
                buf.itemsize,
                buf.shape if buf.ndim > 0 else (1,),
            )

    def skip_buffers(self, buffer_names: list[str], index: int = 0) -> None:
        """Mark buffers as skipped (empty) for exec-obj slot `index`."""
        self.set_buffers({name: np.array([]) for name in buffer_names}, index)

    def unskip_buffers(self, buffer_names: list[str], index: int = 0) -> None:
        """Restore skipped buffers to zero-filled arrays of their binding shape."""
        bufs: dict[str, np.ndarray] = {}
        for b in self.get_bindings(buffer_names):
            dtype = AIC_TO_NP[b.type]
            bufs[b.name] = np.zeros(list(b.dims), dtype=dtype)
        self.set_buffers(bufs, index)

    def get_tuple_list_from_dict(self, inputs: dict[str, Any]) -> list[tuple[int, np.ndarray]]:
        """
        Convert {name: array} → [(binding_index, array)] tuple list.
        This is the format expected by execObj.setData() and
        execObj.setDataWithSlices() — avoids a full qbuffer/buf_dims update
        and is the zero-copy path for inputs that are already contiguous.
        """
        result: list[tuple[int, np.ndarray]] = []
        for name, buf in inputs.items():
            if name not in self.binding_index_map:
                logger.warning("Buffer: %s not found", name)
                continue
            if buf is None:
                continue
            result.append((self.binding_index_map[name], buf))
        return result

    def _make_inputs_contiguous(self, inputs: dict[str, Any]) -> None:
        """Ensure all input arrays are C-contiguous in-place."""
        for k, v in inputs.items():
            inputs[k] = np.ascontiguousarray(v)

    # ── KV handoff (zero-copy DMA slice) ─────────────────────────────────────

    def _set_data_with_slices(
        self,
        buffers,
        slicing_parameters: list[tuple[str, int]],
        slicing_spec_handle,
        index: int = 0,
        buff_map: list[tuple[str, int]] | None = None,
    ):
        """
        Core zero-copy KV handoff primitive.

        If `buffers` is a list/ndarray, `buff_map` must be provided and the
        buffers are zipped with it to form the (binding_index, array) tuple list.
        If `buffers` is a dict, get_tuple_list_from_dict() is used instead.

        setDataWithSlices() tells the runtime to write the output of this
        exec-obj into a *slice* of the target buffer (identified by
        slicing_parameters) rather than the whole buffer — enabling multiple
        batch slots to share one large KV allocation without copying.
        """
        if isinstance(buffers, (list, np.ndarray)):
            assert buff_map is not None, "buff_map required when buffers is a list"
            assert len(buffers) == len(buff_map) or len(buffers) + 1 == len(buff_map), (
                "buffers length must match buff_map (or buff_map may include logits entry)"
            )
            slices_as_tuple_list = [(entry[1], buf) for entry, buf in zip(buff_map, buffers)]
        else:
            slices_as_tuple_list = self.get_tuple_list_from_dict(buffers)

        status, _ = self.execObj[index].setDataWithSlices(
            slices_as_tuple_list,  # [(binding_idx, numpy_array), ...]
            slicing_spec_handle,  # compiled address descriptor
            slicing_parameters,  # parametric values
        )
        assert status == qaicrt.QStatus.QS_SUCCESS, "Failed to setDataWithSlices"
        return buffers

    def set_data_for_kv_handoff(
        self,
        kv_cache_buffers,
        slicing_parameters: list[tuple[str, int]],
        index: int = 0,
        buff_map: list[tuple[str, int]] | None = None,
    ):
        """
        Wire prefill RetainedState outputs directly into the shared KV cache
        buffers via a sliced DMA descriptor.

        Call this BEFORE np_run_pipeline() on the last chunk so the runtime
        knows where to write the KV outputs without any numpy copy.

        Parameters
        ----------
        kv_cache_buffers : list[np.ndarray] | dict[str, np.ndarray]
            Shared KV numpy arrays (one per layer key/value).
        slicing_parameters : list[tuple[str, int]]
            e.g. [("batch_index", bidx), ("ctx_start", 0)]
        index : int
            Exec-obj slot index.
        buff_map : list[tuple[str, int]] | None
            Typically session.kv_only_buff_map (past_key.*/past_value.* only, safe for VLMs).
        """
        return self._set_data_with_slices(
            kv_cache_buffers,
            slicing_parameters,
            self.kv_slicing_spec_handle,
            index,
            buff_map,
        )

    def set_data_for_repetition_penalty(
        self,
        repetition_penalty_buffers,
        slicing_parameters: list[tuple[str, int]],
        index: int = 0,
    ):
        """Wire repetition-penalty buffer via sliced DMA."""
        return self._set_data_with_slices(
            repetition_penalty_buffers,
            slicing_parameters,
            self.repetition_penalty_spec_handle,
            index,
            self.repetition_penalty_map,
        )

    # ── Inference entry points ────────────────────────────────────────────────

    def np_run(
        self,
        inputs: dict[str, Any],
        slicing_parameters: list[tuple[str, int]] | None = None,
        is_prefill: bool = True,
    ) -> int:
        """
        Blocks until an exec-obj slot is available (prefill pool or fixed
        decode slot), then calls setData / setDataWithSlices and enqueues.
        Returns the exec-obj index so the caller can call complete_inf() later.

        Parameters
        ----------
        inputs : dict[str, np.ndarray]
            Model inputs.  Arrays are made contiguous in-place.
        slicing_parameters : list[tuple[str, int]] | None
            If provided, uses setDataWithSlices with kv_slicing_spec_handle.
        is_prefill : bool
            True  → draw from prefill_available_exec_objs pool.
            False → use fixed decode_execObj_idx slot.
        """
        if is_prefill:
            exec_idx = self.prefill_available_exec_objs.get(timeout=self.exec_timeout)
        else:
            assert self.decode_execObj_idx is not None, "decode_execObj_idx is None — session not configured for decode"
            exec_idx = self.decode_execObj_idx

        if slicing_parameters is not None:
            self._make_inputs_contiguous(inputs)
            slices = self.get_tuple_list_from_dict(inputs)
            status, _ = self.execObj[exec_idx].setDataWithSlices(
                slices, self.kv_slicing_spec_handle, slicing_parameters
            )
            assert status == qaicrt.QStatus.QS_SUCCESS, "setDataWithSlices failed"
        elif not is_prefill:
            self._make_inputs_contiguous(inputs)
            tuple_list = self.get_tuple_list_from_dict(inputs)
            status = self.execObj[exec_idx].setData(tuple_list)
            assert status == qaicrt.QStatus.QS_SUCCESS, "setData (decode tuple_list) failed"
        else:
            self.set_buffers(inputs, exec_idx)
            status = self.execObj[exec_idx].setData(self.qbuffers[exec_idx], self.buf_dims[exec_idx])
            assert status == qaicrt.QStatus.QS_SUCCESS, "setData failed"

        try:
            assert self.queue.enqueue(self.execObj[exec_idx]) == qaicrt.QStatus.QS_SUCCESS, "enqueue failed"
        except Exception as exc:
            logger.error("Error while enqueuing: %s", exc)
            return 0

        return exec_idx

    def np_run_pipeline(
        self,
        inputs: dict[str, np.ndarray],
        slicing_parameters: list[tuple[str, int]] | None = None,
        last_chunk: bool = False,
        kv_cache_buffers=None,
    ) -> int:
        """
        Pipelined prefill enqueue with KV-cache handoff on the last chunk.

        Draws an exec-obj from the prefill pool (blocks if none available),
        optionally wires the KV RetainedState outputs into the shared KV cache
        via set_data_for_kv_handoff() when last_chunk=True, then enqueues.

        The key difference from np_run:
          • On last_chunk=True, set_data_for_kv_handoff() is called BEFORE
            setData so the runtime wires the prefill output KV directly into
            the decode session's KV input slots — zero-copy, no numpy involved.
          • Non-last chunks do not need KV outputs (they are skipped by default).

        Parameters
        ----------
        inputs : dict[str, np.ndarray]
            Chunk inputs (input_ids, position_ids, batch_index, logits buffer).
        slicing_parameters : list[tuple[str, int]] | None
            Passed to setDataWithSlices if provided.
        last_chunk : bool
            True on the final chunk of a request.  Triggers KV handoff wiring.
        kv_cache_buffers : list[np.ndarray] | None
            Shared KV numpy arrays to wire into.  Must be non-None when
            last_chunk=True.
        """
        logger.debug("Waiting for prefill exec-obj (pipeline)")
        exec_idx = self.prefill_available_exec_objs.get(timeout=self.exec_timeout)
        logger.debug("Got prefill exec-obj %d", exec_idx)

        if last_chunk:
            assert kv_cache_buffers is not None, "kv_cache_buffers must be provided for last_chunk=True"
            batch_index = int(inputs["batch_index"].item())
            # Wire prefill RetainedState outputs → shared KV cache (zero-copy)
            self.set_data_for_kv_handoff(
                kv_cache_buffers,
                [
                    ("batch_index", batch_index % self.full_batch_size),
                    ("ctx_start", 0),
                ],
                exec_idx,
                self.kv_only_buff_map,
            )

        # Must use overload-4 setData(tuple_list) here — NOT setData(qbuffers, buf_dims).
        #
        # setDataWithSlices (called above in set_data_for_kv_handoff) wires
        # the KV RetainedState output slots (idx) to the shared kv_cache arrays.
        # setData(qbuffers, buf_dims) — overload 1 — overwrites ALL
        # slots, destroying the KV handoff wiring regardless of call order.
        #
        # setData(tuple_list) — overload 4 — only touches the indices explicitly listed.
        # get_tuple_list_from_dict(inputs) produces [(0, input_ids), (1, position_ids),
        # (10, logits_buf)] — indices in between  are absent, so the setDataWithSlices wiring
        # on those slots is never touched and survives intact.
        self._make_inputs_contiguous(inputs)
        tuple_list = self.get_tuple_list_from_dict(inputs)

        if slicing_parameters is None:
            status = self.execObj[exec_idx].setData(tuple_list)
            assert status == qaicrt.QStatus.QS_SUCCESS, "setData failed"
        else:
            status, _ = self.execObj[exec_idx].setDataWithSlices(
                tuple_list, self.kv_slicing_spec_handle, slicing_parameters
            )
            assert status == qaicrt.QStatus.QS_SUCCESS, "setDataWithSlices failed"

        assert self.queue.enqueue(self.execObj[exec_idx]) == qaicrt.QStatus.QS_SUCCESS, "enqueue failed"

        return exec_idx

    def complete_inf(self, index: int, is_prefill: bool = True) -> None:
        """
        Wait for exec-obj `index` to finish and release it back to the pool.

        Parameters
        ----------
        index : int
            Exec-obj index returned by np_run / np_run_pipeline.
        is_prefill : bool
            True  → return the slot to prefill_available_exec_objs.
            False → decode slot is fixed; nothing to return.
        """
        if self.execObj[index].waitForCompletion() != qaicrt.QStatus.QS_SUCCESS:
            raise ValueError(_build_completion_error(self.bindings, self.allowed_shapes, self.buf_dims[index]))
        logger.debug("Releasing exec-obj %d (is_prefill=%s)", index, is_prefill)
        if is_prefill:
            self.prefill_available_exec_objs.put(index)

    def get_outputs(self, index: int = 0) -> dict[str, np.ndarray]:
        """
        Read output buffers from exec-obj `index` after complete_inf().

        Returns a dict of {output_name: numpy_array} for all non-empty outputs.

        """
        status, out_qbuffers = self.execObj[index].getData()
        assert status == qaicrt.QStatus.QS_SUCCESS, "getData failed"

        outputs: dict[str, np.ndarray] = {}
        for binding in self.bindings:
            if binding.dir != aicapi.BUFFER_IO_TYPE_OUTPUT:
                continue
            raw = bytes(out_qbuffers[binding.index])
            if len(raw) == 0:
                continue
            dtype = AIC_TO_NP[binding.type]
            shape = list(binding.dims)
            outputs[binding.name] = np.frombuffer(raw, dtype=dtype).reshape(shape)
        return outputs

    def run(self, inputs: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        exec_idx = self.np_run(inputs, is_prefill=True)
        self.complete_inf(exec_idx, is_prefill=True)
        return self.get_outputs(exec_idx)

    def create_numpy_buffers(
        self,
        input_dict: dict,
        direction: str,
        shape: list[int],
        size: np.dtype,
    ) -> None:
        """Allocate zero KV buffers (input or output RetainedState) into input_dict."""
        if direction == "in":
            names = [n for n in self.input_names if n.startswith("past_key") or n.startswith("past_value")]
        elif direction == "out":
            names = [
                n
                for n in self.output_names
                if (n.startswith("past_key") or n.startswith("past_value")) and n.endswith("_RetainedState")
            ]
        else:
            raise ValueError(f"Invalid direction '{direction}'; expected 'in' or 'out'")
        for name in names:
            input_dict[name] = np.zeros(shape, dtype=size) if shape else np.array([])

    def create_output_buffers(
        self,
        input_dict: dict,
        shape: list[int],
        size: np.dtype,
        buffer_name: str = "logits",
    ) -> None:
        """Allocate an empty output buffer (e.g. logits) into input_dict."""
        if buffer_name not in self.binding_index_map:
            logger.warning("Buffer: %s not found", buffer_name)
            return
        input_dict[buffer_name] = np.empty(shape, dtype=size)

    def extract_outputs(self, input_dict: dict) -> dict[str, np.ndarray]:
        """Extract output-named keys from a combined input/output dict."""
        return {name: input_dict[name] for name in self.output_names if name in input_dict}

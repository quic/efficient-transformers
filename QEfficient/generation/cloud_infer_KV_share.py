# -----------------------------------------------------------------------------
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------
"""
Optimised KV-cache handoff via shared DMA buffers and slice operations.

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

import logging
from pathlib import Path
from typing import Any

import numpy as np

from QEfficient.generation._aic_types import AIC_TO_NP, aicapi, qaicrt
from QEfficient.generation._buff_map_registry import BuffMapRegistry
from QEfficient.generation._buffer_utils import (
    build_completion_error,
    inputs_to_tuple_list,
    make_inputs_contiguous,
)
from QEfficient.generation._exec_pool import ExecObjPool
from QEfficient.generation._kv_slicing import KVSlicingEngine

logger = logging.getLogger(__name__)


class QAICInferenceSession:
    """
    Optimised QAIC inference session with KV-cache handoff via shared DMA
    buffers and slice operations.

    Parameters
    ----------
    qpc_path : str | Path
        Path to the compiled QPC binary directory.
    full_batch_size : int
        Maximum batch size the QPC was compiled for.
    device_ids : list[int] | None
        Device IDs to use.
    activate : bool
        Activate the program immediately on construction.
    enable_debug_logs : bool
        Enable QAIC runtime debug logging.
    stages : int
        Number of pipeline stages for pipelined prefill.
    cluster_id : str | None
        "prefill"  -> only prefill exec-objs, no decode slot.
        "decode"   -> only one decode exec-obj, no prefill pool.
        None       -> combined mode: one decode slot + prefill pool.
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
        self.full_batch_size = full_batch_size

        # ── Exec-obj pool ─────────────────────────────────────────────────────
        self._pool = ExecObjPool(cluster_id=cluster_id, stages=stages if stages is not None else 1)

        # ── Context + Queue ───────────────────────────────────────────────────
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

        if device_ids and len(device_ids) > 1:
            prog_props.devMapping = ":".join(map(str, device_ids))

        self.program = qaicrt.Program(self.context, None, qpc, prog_props)
        assert self.program.load() == qaicrt.QStatus.QS_SUCCESS, "Failed to load program"

        self.activate_done = False
        if activate:
            self.activate()

        # ── Per-exec-obj qbuffers and buf_dims ───────────────────────────────
        self.qbuffers: list[list[qaicrt.QBuffer]] = [
            [qaicrt.QBuffer(bytes(b.size)) for b in self.bindings] for _ in range(self._pool.queue_len)
        ]
        self.buf_dims: list[qaicrt.BufferDimensionsVecRef] = [
            qaicrt.BufferDimensionsVecRef([(AIC_TO_NP[b.type].itemsize, list(b.dims)) for b in self.bindings])
            for _ in range(self._pool.queue_len)
        ]

        # ── KV slicing engine ────────────────────────────────────────────────
        self._kv_engine = KVSlicingEngine(
            program=self.program,
            bindings=self.bindings,
            binding_index_map=self.binding_index_map,
        )

        # ── Buffer map registry ──────────────────────────────────────────────
        self._buff_maps = BuffMapRegistry.build_from_bindings(
            input_names=self.input_names,
            output_names=self.output_names,
            binding_index_map=self.binding_index_map,
        )

        # ── Skip KV buffers by default (retained-state managed via handoff) ──
        for slot in range(self._pool.queue_len):
            self.skip_buffers([n for n in self.input_names if n.startswith("past_")], slot)
            self.skip_buffers([n for n in self.output_names if n.endswith("_RetainedState")], slot)

    # ── Public properties (backward-compatible) ───────────────────────────────

    @property
    def input_names(self) -> list[str]:
        return [b.name for b in self.bindings if b.dir == aicapi.BUFFER_IO_TYPE_INPUT]

    @property
    def output_names(self) -> list[str]:
        return [b.name for b in self.bindings if b.dir == aicapi.BUFFER_IO_TYPE_OUTPUT]

    @property
    def stages(self) -> int:
        return self._pool.stages

    @property
    def cluster_id(self) -> str | None:
        return self._pool.cluster_id

    @property
    def prefill_num_execObj(self) -> int:
        return self._pool.prefill_num

    @property
    def decode_num_execObj(self) -> int:
        return self._pool.decode_num

    @property
    def decode_execObj_idx(self) -> int | None:
        return self._pool.decode_idx

    @property
    def queue_len(self) -> int:
        return self._pool.queue_len

    @property
    def prefill_available_exec_objs(self):
        return self._pool._available_prefill

    @property
    def exec_timeout(self) -> int:
        return self._pool.exec_timeout

    # ── KV engine delegated properties ────────────────────────────────────────

    @property
    def kv_shape(self) -> list[int] | None:
        return self._kv_engine.kv_shape

    @property
    def kv_size(self) -> np.dtype | None:
        return self._kv_engine.kv_size

    @property
    def kv_slicing_spec_handle(self):
        return self._kv_engine.kv_slicing_spec_handle

    # ── Buff map delegated properties ─────────────────────────────────────────

    @property
    def decode_buff_map(self) -> list[tuple[str, int]]:
        return self._buff_maps.decode_buff_map

    @property
    def decode_rs_buff_map(self) -> list[tuple[str, int]]:
        return self._buff_maps.decode_rs_buff_map

    @property
    def decode_rs_kv_only_buff_map(self) -> list[tuple[str, int]]:
        return self._buff_maps.decode_rs_kv_only_buff_map

    @property
    def prefill_buff_map(self) -> list[tuple[str, int]]:
        return self._buff_maps.prefill_buff_map

    @property
    def kv_only_buff_map(self) -> list[tuple[str, int]]:
        return self._buff_maps.kv_only_buff_map

    # ── Lifecycle ─────────────────────────────────────────────────────────────

    def activate(self) -> None:
        """Activate the program and create one ExecObj per pool slot."""
        self.activate_done = True
        self.program.activate()
        self.execObj: list[qaicrt.ExecObj] = [
            qaicrt.ExecObj(self.context, self.program) for _ in range(self._pool.queue_len)
        ]

    def deactivate(self) -> None:
        """Deactivate the program and release ExecObjs."""
        if self.activate_done:
            del self.execObj
            self.program.deactivate()
            self.activate_done = False

    # ── Slicing spec helpers (delegated) ──────────────────────────────────────

    def get_json_for_kv_cache_slicing(self) -> str:
        return self._kv_engine._build_kv_cache_spec_json()

    def get_slicing_spec_handle(self, buffer_spec_json: str):
        return self._kv_engine._create_spec_handle(buffer_spec_json)

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
        """Return the number of dimensions of the logits output binding."""
        if "logits" not in self.binding_index_map:
            logger.warning("logits binding not found, defaulting ndim to 3")
            return 3
        return len(self.bindings[self.binding_index_map["logits"]].dims)

    def set_buffers(self, buffers: dict[str, np.ndarray], index: int = 0) -> None:
        """
        Copy numpy arrays into the qbuffer/buf_dims for exec-obj slot `index`.
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
        """Convert {name: array} -> [(binding_index, array)] tuple list."""
        return inputs_to_tuple_list(inputs, self.binding_index_map)

    # ── KV handoff (delegated to KVSlicingEngine) ─────────────────────────────

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
        """
        return self._kv_engine.set_data_for_kv_handoff(
            exec_obj=self.execObj[index],
            kv_cache_buffers=kv_cache_buffers,
            slicing_parameters=slicing_parameters,
            buff_map=buff_map,
        )

    # ── Inference entry points ────────────────────────────────────────────────

    def np_run(
        self,
        inputs: dict[str, Any],
        slicing_parameters: list[tuple[str, int]] | None = None,
        is_prefill: bool = True,
    ) -> int:
        """
        Blocks until an exec-obj slot is available, then calls setData /
        setDataWithSlices and enqueues. Returns the exec-obj index.
        """
        exec_idx = self._pool.acquire(is_prefill)

        if slicing_parameters is not None:
            make_inputs_contiguous(inputs)
            slices = inputs_to_tuple_list(inputs, self.binding_index_map)
            status, _ = self.execObj[exec_idx].setDataWithSlices(
                slices, self._kv_engine.kv_slicing_spec_handle, slicing_parameters
            )
            assert status == qaicrt.QStatus.QS_SUCCESS, "setDataWithSlices failed"
        elif not is_prefill:
            make_inputs_contiguous(inputs)
            tuple_list = inputs_to_tuple_list(inputs, self.binding_index_map)
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

        Draws an exec-obj from the prefill pool, optionally wires the KV
        RetainedState outputs into the shared KV cache via
        set_data_for_kv_handoff() when last_chunk=True, then enqueues.
        """
        logger.debug("Waiting for prefill exec-obj (pipeline)")
        exec_idx = self._pool.acquire(is_prefill=True)
        logger.debug("Got prefill exec-obj %d", exec_idx)

        if last_chunk:
            assert kv_cache_buffers is not None, "kv_cache_buffers must be provided for last_chunk=True"
            batch_index = int(inputs["batch_index"].item())
            self.set_data_for_kv_handoff(
                kv_cache_buffers,
                [
                    ("batch_index", batch_index % self.full_batch_size),
                    ("ctx_start", 0),
                ],
                exec_idx,
                self._buff_maps.kv_only_buff_map,
            )

        make_inputs_contiguous(inputs)
        tuple_list = inputs_to_tuple_list(inputs, self.binding_index_map)

        if slicing_parameters is None:
            status = self.execObj[exec_idx].setData(tuple_list)
            assert status == qaicrt.QStatus.QS_SUCCESS, "setData failed"
        else:
            status, _ = self.execObj[exec_idx].setDataWithSlices(
                tuple_list, self._kv_engine.kv_slicing_spec_handle, slicing_parameters
            )
            assert status == qaicrt.QStatus.QS_SUCCESS, "setDataWithSlices failed"

        assert self.queue.enqueue(self.execObj[exec_idx]) == qaicrt.QStatus.QS_SUCCESS, "enqueue failed"

        return exec_idx

    def complete_inf(self, index: int, is_prefill: bool = True) -> None:
        """
        Wait for exec-obj `index` to finish and release it back to the pool.
        """
        if self.execObj[index].waitForCompletion() != qaicrt.QStatus.QS_SUCCESS:
            raise ValueError(build_completion_error(self.bindings, self.allowed_shapes, self.buf_dims[index]))
        logger.debug("Releasing exec-obj %d (is_prefill=%s)", index, is_prefill)
        self._pool.release(index, is_prefill)

    def get_outputs(self, index: int = 0) -> dict[str, np.ndarray]:
        """
        Read output buffers from exec-obj `index` after complete_inf().
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

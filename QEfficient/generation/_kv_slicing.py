# -----------------------------------------------------------------------------
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------
"""KV-cache slicing engine for zero-copy DMA handoff on QAIC."""

from __future__ import annotations

import json
import logging

import numpy as np

from QEfficient.generation._aic_types import AIC_TO_NP, qaicrt

logger = logging.getLogger(__name__)


class KVSlicingEngine:
    """
    Manages KV-cache slicing spec handles and provides the core
    setDataWithSlices primitive for zero-copy DMA handoff.

    Parameters
    ----------
    program : qaicrt.Program
        The loaded QAIC program (needed for createSlicingSpecHandle).
    bindings : list
        IoDesc bindings (protobuf repeated field).
    binding_index_map : dict[str, int]
        Name -> binding index mapping.
    """

    def __init__(
        self,
        program,
        bindings,
        binding_index_map: dict[str, int],
    ) -> None:
        self.program = program
        self.bindings = bindings
        self.binding_index_map = binding_index_map

        self.kv_slicing_spec_handle = None
        self.kv_shape: list[int] | None = None
        self.kv_size: np.dtype | None = None

        first_kv_rs_name: str | None = next(
            (n for n in binding_index_map if n.startswith("past_key.") and n.endswith("_RetainedState")),
            None,
        )
        if first_kv_rs_name is not None:
            rs_binding = bindings[binding_index_map[first_kv_rs_name]]
            self.kv_shape = list(rs_binding.dims)
            self.kv_shape[0] = 1
            self.kv_size = AIC_TO_NP[rs_binding.type]
            self._first_kv_layer_name: str = first_kv_rs_name.replace("_RetainedState", "")
            self.kv_slicing_spec_handle = self._create_spec_handle(self._build_kv_cache_spec_json())

    def _create_spec_handle(self, buffer_spec_json: str):
        """Create and return a slicing spec handle from the program."""
        status, handle = self.program.createSlicingSpecHandle(buffer_spec_json)
        assert status == qaicrt.QStatus.QS_SUCCESS, "Failed to createSlicingSpecHandle"
        return handle

    def _build_kv_cache_spec_json(self) -> str:
        """
        JSON spec for setDataWithSlices on past_key.* / past_value.*.
        Slice dimensions: [batch_index, :, ctx_start, :].
        """
        elem_size = AIC_TO_NP[self.bindings[self.binding_index_map[self._first_kv_layer_name]].type].itemsize
        spec = {
            "BufferSpecs": [
                {
                    "Name": "past_key.*",
                    "ElemSize": elem_size,
                    "DimSpecs": [
                        {"start": "batch_index"},
                        {"start": 0},
                        {"start": "ctx_start"},
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

    def set_data_with_slices(
        self,
        exec_obj,
        buffers,
        slicing_parameters: list[tuple[str, int]],
        slicing_spec_handle,
        buff_map: list[tuple[str, int]] | None = None,
    ):
        """
        Core zero-copy KV handoff primitive.

        If buffers is a list/ndarray, buff_map must be provided and the
        buffers are zipped with it to form the (binding_index, array) tuple list.
        If buffers is a dict, inputs_to_tuple_list() is used instead.
        """
        from QEfficient.generation._buffer_utils import inputs_to_tuple_list

        if isinstance(buffers, (list, np.ndarray)):
            assert buff_map is not None, "buff_map required when buffers is a list"
            assert len(buffers) == len(buff_map) or len(buffers) + 1 == len(buff_map), (
                "buffers length must match buff_map (or buff_map may include logits entry)"
            )
            slices_as_tuple_list = [(entry[1], buf) for entry, buf in zip(buff_map, buffers)]
        else:
            slices_as_tuple_list = inputs_to_tuple_list(buffers, self.binding_index_map)

        status, _ = exec_obj.setDataWithSlices(
            slices_as_tuple_list,
            slicing_spec_handle,
            slicing_parameters,
        )
        assert status == qaicrt.QStatus.QS_SUCCESS, "Failed to setDataWithSlices"
        return buffers

    def set_data_for_kv_handoff(
        self,
        exec_obj,
        kv_cache_buffers,
        slicing_parameters: list[tuple[str, int]],
        buff_map: list[tuple[str, int]] | None = None,
    ):
        """
        Wire prefill RetainedState outputs directly into the shared KV cache
        buffers via a sliced DMA descriptor.

        Parameters
        ----------
        exec_obj : qaicrt.ExecObj
            The exec-obj to wire the handoff on.
        kv_cache_buffers : list[np.ndarray] | dict[str, np.ndarray]
            Shared KV numpy arrays (one per layer key/value).
        slicing_parameters : list[tuple[str, int]]
            e.g. [("batch_index", bidx), ("ctx_start", 0)]
        buff_map : list[tuple[str, int]] | None
            Typically kv_only_buff_map (past_key.*/past_value.* only).
        """
        return self.set_data_with_slices(
            exec_obj,
            kv_cache_buffers,
            slicing_parameters,
            self.kv_slicing_spec_handle,
            buff_map,
        )

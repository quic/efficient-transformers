# -----------------------------------------------------------------------------
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------

import json
import platform
import sys
from pathlib import Path
from queue import Queue
from typing import Dict, List, Optional, Tuple, Union
from warnings import warn

import numpy as np

# Slicing-spec DimSpecs templates, chosen per KV binding's attention type.
# The symbolic "batch_index" / "ctx_start" tokens are resolved at handoff time
# from the (name, offset) pairs passed as `slicing_parameters`.
FULL_ATTN_DIMSPEC = [
    {"start": "batch_index"},
    {"start": 0},
    {"start": "ctx_start"},
    {"start": 0},
]
LINEAR_ATTN_DIMSPEC = [
    {"start": "batch_index"},
    {"start": 0},
    {"start": 0},
]


def _is_linear_state_name(name: str) -> bool:
    """Return True for linear/recurrent (3-D) retained-state bindings."""
    return name.startswith(("conv_state.", "recurrent_state."))


def _kv_layer_sort_key(item: Tuple[str, int]) -> Tuple[int, str]:
    """Order KV bindings by layer index, then name.

    Standard attention sorts ``past_key`` before ``past_value``; MLA sorts
    ``compressed_kv`` before ``k_pe``. The layer index is parsed from the first
    dotted segment (``past_key.3`` -> 3); names without a dotted index sort as 0.
    """
    name = item[0]
    part = name.split(".")[1] if "." in name else "0"
    return int(part.split("_")[0]), name


def _public_retained_state_name(output_name: str) -> Optional[str]:
    """Map internal subfunction retained-state outputs to public runtime names."""
    suffix = "_InternalRetainedState"
    if output_name.endswith(suffix):
        return output_name[: -len(suffix)] + "_RetainedState"
    return None


def is_retained_state_name(name: str) -> bool:
    """Return True when an I/O binding participates in retained-state cache flow."""
    return name.startswith(("past_", "conv_state.", "recurrent_state.", "compressed_", "k_pe"))


def _add_basename_binding_aliases(binding_index_map: Dict[str, int], bindings) -> None:
    """Allow callers to use unprefixed I/O names for prefixed ONNX graphs."""
    for binding in bindings:
        binding_index_map.setdefault(binding.name.rsplit("/", 1)[-1], binding.index)


try:
    import qaicrt

    is_qaicrt_imported = True
except ImportError:
    try:
        sys.path.append(f"/opt/qti-aic/dev/lib/{platform.machine()}")
        import qaicrt

        is_qaicrt_imported = True
    except ImportError:
        is_qaicrt_imported = False

try:
    import QAicApi_pb2 as aicapi

    is_aicapi_imported = True
except ImportError:
    try:
        sys.path.append("/opt/qti-aic/dev/python")
        import QAicApi_pb2 as aicapi

        is_aicapi_imported = True
    except ImportError:
        is_qaicrt_imported = False


class QAICInferenceSession:
    def __init__(
        self,
        qpc_path: Union[Path, str],
        device_ids: Optional[List[int]] = None,
        activate: bool = True,
        enable_debug_logs: bool = False,
        kv_dma_share: bool = False,
        stages: Optional[int] = 1,
        cluster_id: Optional[str] = None,
        full_batch_size: int = 1,
    ):
        """
        Initialise for QAIC inference Session
        ---------

        :qpc_path: str. Path to the save generated binary file after compilation.
        :device_ids: List[int]. Device Ids to be used for compilation. if devices > 1, it enables multiple card setup.
        :activate: bool. If false, activation will be disabled. Default=True.
        :enable_debug_logs: bool. If True, It will enable debug logs. Default=False.
        :kv_dma_share: bool. If True, enable the DMA-based prefill->decode KV handoff
            path (`np_run` / `np_run_pipeline` / `set_data_for_kv_handoff`). When False
            (default) the session behaves exactly as before: the handoff members are
            inert and only the numpy-copy `run()` path is available.
        :stages: Optional[int]. Prefill pipeline depth; sizes the prefill execObj pool
            (`stages + 1`). Only used when `kv_dma_share=True`. Default=1.
        :cluster_id: Optional[str]. One of "prefill", "decode", or None (combined).
            Selects how the execObj pool is split. Only used when `kv_dma_share=True`.
        :full_batch_size: int. Number of decode slots; `batch_index` offsets wrap
            modulo this value at prefill handoff. Only used when `kv_dma_share=True`.
        """
        if not (is_qaicrt_imported and is_aicapi_imported):
            raise ImportError(
                "Unable to import `qaicrt` and/or `QAicApi_pb2` libraries required for executing QPC files on the CLOUD AI platform.\n"
                "Please ensure that the QAIC platform SDK and apps SDK are installed correctly."
            )

        # Build dtype mapping once (depends on aicapi constants)
        self.aic_to_np_dtype_mapping = {
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

        # KV-DMA-share configuration. When disabled, `queue_len == 1` and the
        # session keeps a single scalar execObj / qbuffers / buf_dims exactly as
        # before; all handoff members below are skipped.
        self.kv_dma_share = kv_dma_share
        self.stages = stages if stages is not None else 1
        self.cluster_id = cluster_id
        self.full_batch_size = full_batch_size
        self.decode_execObj_idx: Optional[int] = None
        if not kv_dma_share:
            self.prefill_num_execObj = 0
            self.decode_num_execObj = 0
            self.queue_len = 1
        elif cluster_id == "decode":
            self.prefill_num_execObj = 0
            self.decode_num_execObj = 1
            self.decode_execObj_idx = 0
            self.queue_len = 1
        elif cluster_id == "prefill":
            self.prefill_num_execObj = self.stages + 1
            self.decode_num_execObj = 0
            self.queue_len = self.prefill_num_execObj
        else:  # combined prefill+decode in one session
            self.prefill_num_execObj = self.stages + 1
            self.decode_num_execObj = 1
            self.decode_execObj_idx = 0
            self.queue_len = self.prefill_num_execObj + self.decode_num_execObj
        # Prefill exec slots follow the single decode slot (index 0) in the pool.
        self.prefill_available_exec_objs: Queue = Queue()
        if kv_dma_share:
            for i in range(self.decode_num_execObj, self.decode_num_execObj + self.prefill_num_execObj):
                self.prefill_available_exec_objs.put(i)

        # Load QPC
        if device_ids is not None:
            devices = qaicrt.QIDList(device_ids)
            self.context = qaicrt.Context(devices)
            self.queue = qaicrt.Queue(self.context, device_ids[0])
        else:
            self.context = qaicrt.Context()
            self.queue = qaicrt.Queue(self.context, 0)  # Async API
        if enable_debug_logs:
            if self.context.setLogLevel(qaicrt.QLogLevel.QL_DEBUG) != qaicrt.QStatus.QS_SUCCESS:
                raise RuntimeError("Failed to setLogLevel")
        qpc = qaicrt.Qpc(str(qpc_path))
        # Load IO Descriptor
        iodesc = aicapi.IoDesc()
        status, iodesc_data = qpc.getIoDescriptor()
        if status != qaicrt.QStatus.QS_SUCCESS:
            raise RuntimeError("Failed to getIoDescriptor")
        iodesc.ParseFromString(bytes(iodesc_data))
        self.allowed_shapes = [
            [(self.aic_to_np_dtype_mapping[x.type].itemsize, list(x.dims)) for x in allowed_shape.shapes]
            for allowed_shape in iodesc.allowed_shapes
        ]
        self.bindings = iodesc.selected_set.bindings
        self.binding_index_map = {binding.name: binding.index for binding in self.bindings}
        _add_basename_binding_aliases(self.binding_index_map, self.bindings)
        # Create and load Program
        prog_properties = qaicrt.QAicProgramProperties()
        prog_properties.dataPathTimeoutMs = 60_000
        dev_id_non_mq = None
        if device_ids:
            if len(device_ids) == 1:
                dev_id_non_mq = device_ids[0]
            elif len(device_ids) > 1:
                prog_properties.devMapping = ":".join(map(str, device_ids))
        self.program = qaicrt.Program(self.context, dev_id_non_mq, qpc, prog_properties)
        if self.program.load() != qaicrt.QStatus.QS_SUCCESS:
            raise RuntimeError("Failed to load program")
        self.is_active = False
        if activate:
            self.activate()
            self.is_active = True
        if not self.kv_dma_share:
            # Create input qbuffers and buf_dims (single-execObj `run()` path)
            self.qbuffers = [qaicrt.QBuffer(bytes(binding.size)) for binding in self.bindings]
            self.buf_dims = qaicrt.BufferDimensionsVecRef(
                [(self.aic_to_np_dtype_mapping[binding.type].itemsize, list(binding.dims)) for binding in self.bindings]
            )
        else:
            # Per-slot qbuffers / buf_dims for the pooled DMA-handoff path.
            self.qbuffers = [
                [qaicrt.QBuffer(bytes(binding.size)) for binding in self.bindings] for _ in range(self.queue_len)
            ]
            self.buf_dims = [
                qaicrt.BufferDimensionsVecRef(
                    [
                        (self.aic_to_np_dtype_mapping[binding.type].itemsize, list(binding.dims))
                        for binding in self.bindings
                    ]
                )
                for _ in range(self.queue_len)
            ]
            self._init_kv_handoff()

    def _init_kv_handoff(self):
        """Build ordered buffer maps and the compiled KV slicing spec.

        Only invoked when `kv_dma_share=True`. All maps are `list[(name, index)]`
        sorted by `(layer_index, name)` so a prefill write and a decode read that
        share the same family iterate the caller's arrays in the same order.
        """
        # Decode KV *input* bindings (VLM-safe: KV names only).
        self.decode_buff_map = [
            (name, self.binding_index_map[name]) for name in self.input_names if is_retained_state_name(name)
        ]
        self.decode_buff_map.sort(key=_kv_layer_sort_key)

        # Decode KV *RetainedState output* bindings, in the same family order.
        self.decode_rs_kv_only_buff_map = [
            (name, self.binding_index_map[name])
            for name in self.output_names
            if name.endswith("_RetainedState") and is_retained_state_name(name)
        ]
        self.decode_rs_kv_only_buff_map.sort(key=_kv_layer_sort_key)

        # Prefill RetainedState outputs: full-attention (4-D) families only for the
        # uniform path; the *_full map additionally carries linear/recurrent families.
        prefill_rs = [
            (name.replace("_RetainedState", ""), self.binding_index_map[name])
            for name in self.output_names
            if name.endswith("_RetainedState") and is_retained_state_name(name)
        ]
        prefill_rs.sort(key=_kv_layer_sort_key)
        self.decode_rs_full_buff_map = list(prefill_rs)
        self.kv_only_buff_map = [item for item in prefill_rs if not _is_linear_state_name(item[0])]

        # Per-slot KV geometry, in decode-input map order.
        self.kv_cache_info: List[Tuple[tuple, np.dtype]] = []
        for name, index in self.decode_buff_map:
            binding = self.bindings[index]
            self.kv_cache_info.append((tuple(binding.dims), self.aic_to_np_dtype_mapping[binding.type]))

        # Hybrid iff more than one distinct (shape, dtype) KV family exists.
        distinct = {(shape, dtype.str) for shape, dtype in self.kv_cache_info}
        self.is_hybrid_kv = len(distinct) > 1

        self.kv_slicing_spec_handle = None
        if self.kv_cache_info:
            spec_json = (
                self._build_full_kv_slicing_json() if self.is_hybrid_kv else self._build_uniform_kv_slicing_json()
            )
            self.kv_slicing_spec_handle = self._create_slicing_spec_handle(spec_json)

        # Readable (plain, non-RetainedState) outputs — e.g. `logits`. On the
        # pooled path `setData(tuple_list)` only wires the bindings we hand it, so
        # unlike the scalar `run()` path these outputs get no device buffer unless
        # we supply one. Pre-allocate a host array per slot and wire it into every
        # enqueue so the runtime DMA-writes the output in place (mirrors the vLLM
        # reference's `chunk_inputs["logits"] = logits[...]`); `get_outputs` then
        # reads straight from these arrays. Per-slot copies keep in-flight prefill
        # chunks from clobbering each other.
        self.readable_output_bindings = [
            (name, self.binding_index_map[name]) for name in self.output_names if not name.endswith("_RetainedState")
        ]
        self.output_buffers: List[Dict[str, np.ndarray]] = [
            {
                name: np.zeros(
                    tuple(self.bindings[index].dims),
                    dtype=self.aic_to_np_dtype_mapping[self.bindings[index].type],
                )
                for name, index in self.readable_output_bindings
            }
            for _ in range(self.queue_len)
        ]

        # Persistent input bindings (e.g. `vision_embeds`): constant across every
        # enqueue of a request. On the pooled path `setData(tuple_list)` wires only
        # the bindings handed to it per call, so — like `readable_output_bindings` —
        # these must be re-appended to every enqueue rather than living in `qbuffers`
        # (which the pooled path never reads). Registered once via
        # `set_persistent_inputs`; a per-call `inputs` entry of the same name wins.
        self.persistent_inputs: Dict[str, np.ndarray] = {}

    def _build_uniform_kv_slicing_json(self) -> str:
        """One BufferSpec per KV name family; every KV shares shape/dtype."""
        elem_size = self.kv_cache_info[0][1].itemsize
        names = sorted({name.split(".")[0] for name, _ in self.kv_only_buff_map})
        buffer_specs = [{"Name": f"{base}.*", "ElemSize": elem_size, "DimSpecs": FULL_ATTN_DIMSPEC} for base in names]
        return json.dumps({"BufferSpecs": buffer_specs})

    def _build_full_kv_slicing_json(self) -> str:
        """One BufferSpec per RetainedState binding; DimSpecs chosen by ndim."""
        buffer_specs = []
        for binding in self.bindings:
            name = binding.name
            if not (name.endswith("_RetainedState") and is_retained_state_name(name)):
                continue
            base_name = name.replace("_RetainedState", "")
            elem_size = self.aic_to_np_dtype_mapping[binding.type].itemsize
            dim_spec = FULL_ATTN_DIMSPEC if len(binding.dims) == 4 else LINEAR_ATTN_DIMSPEC
            buffer_specs.append({"Name": f"{base_name}.*", "ElemSize": elem_size, "DimSpecs": dim_spec})
        return json.dumps({"BufferSpecs": buffer_specs})

    def _create_slicing_spec_handle(self, buffer_spec_json: str):
        status, slicing_spec_handle = self.program.createSlicingSpecHandle(buffer_spec_json)
        if status != qaicrt.QStatus.QS_SUCCESS:
            raise RuntimeError("Failed to create SlicingSpecHandle")
        return slicing_spec_handle

    @property
    def input_names(self) -> List[str]:
        return [binding.name for binding in self.bindings if binding.dir == aicapi.BUFFER_IO_TYPE_INPUT]

    @property
    def output_names(self) -> List[str]:
        return [binding.name for binding in self.bindings if binding.dir == aicapi.BUFFER_IO_TYPE_OUTPUT]

    def activate(self):
        """Activate qpc"""
        if not self.is_active:
            self.program.activate()
            if self.kv_dma_share:
                self.execObj = [qaicrt.ExecObj(self.context, self.program) for _ in range(self.queue_len)]
            else:
                self.execObj = qaicrt.ExecObj(self.context, self.program)
            self.is_active = True

    def deactivate(self):
        """Deactivate qpc"""
        if self.is_active:
            del self.execObj
            self.program.deactivate()
            self.is_active = False

    def set_buffers(self, buffers: Dict[str, np.ndarray]):
        """
        Provide buffer mapping for input and output

        Args:
            :buffer (Dict[str, np.ndarray]): Parameter for buffer mapping.
        """

        for buffer_name, buffer in buffers.items():
            if buffer_name not in self.binding_index_map:
                warn(f'Buffer: "{buffer_name}" not found')
                continue
            buffer_index = self.binding_index_map[buffer_name]
            self.qbuffers[buffer_index] = qaicrt.QBuffer(buffer.tobytes())
            self.buf_dims[buffer_index] = (
                buffer.itemsize,
                buffer.shape if len(buffer.shape) > 0 else (1,),
            )

    def skip_buffers(self, skipped_buffer_names: List[str]):
        """
        skip buffer mapping for given list of buffer names

        Args:
            :skipped_buffer_name: List[str]. List of buffer name to be skipped.
        """

        self.set_buffers({k: np.array([]) for k in skipped_buffer_names})

    def run(self, inputs: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """
        Execute on cloud AI 100

        Args:
            :inputs (Dict[str, np.ndarray]): Processed numpy inputs for the model.

        Return:
            :Dict[str, np.ndarray]:
        """
        # Set inputs
        self.set_buffers(inputs)
        if self.execObj.setData(self.qbuffers, self.buf_dims) != qaicrt.QStatus.QS_SUCCESS:
            raise MemoryError("Failed to setData")
        # # Run with sync API
        # if self.execObj.run(self.qbuffers) != qaicrt.QStatus.QS_SUCCESS:
        # Run with async API
        if self.queue.enqueue(self.execObj) != qaicrt.QStatus.QS_SUCCESS:
            raise MemoryError("Failed to enqueue")
        if self.execObj.waitForCompletion() != qaicrt.QStatus.QS_SUCCESS:
            raise ValueError(self._shape_mismatch_message(self.buf_dims))
        # Get output buffers
        status, output_qbuffers = self.execObj.getData()
        if status != qaicrt.QStatus.QS_SUCCESS:
            raise MemoryError("Failed to getData")
        return self._build_outputs(output_qbuffers, self.qbuffers, self.buf_dims)

    def _shape_mismatch_message(self, buf_dims) -> str:
        """Build the "Failed to run" diagnostic listing allowed vs passed shapes."""
        error_message = "Failed to run"
        # Print additional error messages for unmatched dimension error
        if self.allowed_shapes:
            error_message += "\n\n"
            error_message += '(Only if "No matching dimension found" error is present above)'
            error_message += "\nAllowed shapes:"
            for i, allowed_shape in enumerate(self.allowed_shapes):
                error_message += f"\n{i}\n"
                for binding, (elemsize, shape), (_, passed_shape) in zip(self.bindings, allowed_shape, buf_dims):
                    if passed_shape == [0]:
                        if not binding.is_partial_buf_allowed:
                            warn(f"Partial buffer not allowed for: {binding.name}")
                        continue
                    error_message += f"{binding.name}:\t{elemsize}\t{shape}\n"
            error_message += "\n\nPassed shapes:\n"
            for binding, (elemsize, shape) in zip(self.bindings, buf_dims):
                if shape == [0]:
                    continue
                error_message += f"{binding.name}:\t{elemsize}\t{shape}\n"
        return error_message

    def _build_outputs(self, output_qbuffers, qbuffers, buf_dims) -> Dict[str, np.ndarray]:
        """Decode device output buffers into a name-keyed dict of numpy arrays."""
        outputs = {}
        for output_name in self.output_names:
            buffer_index = self.binding_index_map[output_name]
            # Skip unmapped outputs and DMA-wired RetainedState buffers, whose data
            # goes straight to the caller's host arrays so getData returns empty.
            if qbuffers[buffer_index].size == 0 or output_qbuffers[buffer_index].size == 0:
                continue
            output = np.frombuffer(
                bytes(output_qbuffers[buffer_index]),
                self.aic_to_np_dtype_mapping[self.bindings[buffer_index].type],
            ).reshape(buf_dims[buffer_index][1])
            outputs[output_name] = output
            output_basename = output_name.rsplit("/", 1)[-1]
            outputs.setdefault(output_basename, output)
            public_name = _public_retained_state_name(output_name)
            if public_name is not None:
                outputs[public_name] = output
                outputs.setdefault(public_name.rsplit("/", 1)[-1], output)
        return outputs

    # ------------------------------------------------------------------
    # DMA-based KV handoff path (enabled only when kv_dma_share=True)
    # ------------------------------------------------------------------

    def _tuple_list_from_dict(self, inputs: Dict[str, np.ndarray]) -> List[tuple]:
        """Map a name-keyed input dict to (binding_index, buffer) tuples."""
        tuple_list = []
        for name, buffer in inputs.items():
            if name not in self.binding_index_map:
                warn(f'Buffer: "{name}" not found')
                continue
            if buffer is None:
                continue
            tuple_list.append((self.binding_index_map[name], buffer))
        return tuple_list

    @staticmethod
    def _make_inputs_contiguous(inputs: Dict[str, np.ndarray]) -> None:
        for name, buffer in inputs.items():
            inputs[name] = np.ascontiguousarray(buffer)

    def set_persistent_inputs(self, buffers: Dict[str, np.ndarray]) -> None:
        """Register input buffers that stay constant across enqueues (pooled path).

        Use for large, unchanging inputs — e.g. a VLM's ``vision_embeds`` — so the
        caller supplies them once instead of in every ``np_run``/``np_run_pipeline``
        input dict. The buffers are appended to each enqueue's tuple list (mirrors
        ``readable_output_bindings``); a per-call ``inputs`` entry of the same name
        overrides the persistent one for that call. Unknown names are skipped.
        """
        for name, buffer in buffers.items():
            if name not in self.binding_index_map:
                warn(f'Buffer: "{name}" not found')
                continue
            self.persistent_inputs[name] = np.ascontiguousarray(buffer)

    def _tuple_list_with_outputs(self, inputs: Dict[str, np.ndarray], exec_obj_idx: int) -> List[tuple]:
        """Build the enqueue tuple list: caller inputs plus this slot's readable
        output buffers, so the runtime DMA-writes outputs like ``logits`` in place
        (``get_outputs`` reads them straight back from ``self.output_buffers``).
        Persistent inputs registered via ``set_persistent_inputs`` are appended too,
        unless overridden by a same-named entry in ``inputs``.
        """
        tuple_list = self._tuple_list_from_dict(inputs)
        for name, buffer in self.persistent_inputs.items():
            if name in inputs:
                continue
            tuple_list.append((self.binding_index_map[name], buffer))
        for name, index in self.readable_output_bindings:
            tuple_list.append((index, self.output_buffers[exec_obj_idx][name]))
        return tuple_list

    def set_data_for_kv_handoff(self, kv_cache_buffers, slicing_parameters, index=0, buff_map=None):
        """Wire a sliced DMA descriptor so the runtime writes RetainedState
        outputs directly into ``kv_cache_buffers`` at the ``slicing_parameters``
        offsets. ``buff_map`` is a list of ``(name, binding_index)`` whose order
        must match ``kv_cache_buffers``. Must be re-issued every decode step.
        """
        if buff_map is None:
            raise ValueError("set_data_for_kv_handoff requires a buff_map")
        if not (len(kv_cache_buffers) == len(buff_map) or len(kv_cache_buffers) + 1 == len(buff_map)):
            raise ValueError(
                f"KV buffer count mismatch: expected {len(buff_map)} (or {len(buff_map) - 1}), "
                f"got {len(kv_cache_buffers)}"
            )
        slices = [(binding_index, buf) for (_, binding_index), buf in zip(buff_map, kv_cache_buffers)]
        status, _ = self.execObj[index].setDataWithSlices(slices, self.kv_slicing_spec_handle, slicing_parameters)
        if status != qaicrt.QStatus.QS_SUCCESS:
            raise RuntimeError("Failed to setDataWithSlices")
        return kv_cache_buffers

    def np_run(self, inputs: Dict[str, np.ndarray], slicing_parameters=None, is_prefill: bool = True) -> int:
        """Enqueue one execution (non-blocking). Returns the execObj index; the
        caller must later ``complete_inf`` and ``get_outputs`` on that index.
        """
        if is_prefill:
            exec_obj_idx = self.prefill_available_exec_objs.get()
        else:
            if self.decode_execObj_idx is None:
                raise RuntimeError("No decode execObj configured for this session")
            exec_obj_idx = self.decode_execObj_idx
        self._make_inputs_contiguous(inputs)
        tuple_list = self._tuple_list_with_outputs(inputs, exec_obj_idx)
        if slicing_parameters is None:
            status = self.execObj[exec_obj_idx].setData(tuple_list)
        else:
            status, _ = self.execObj[exec_obj_idx].setDataWithSlices(
                tuple_list, self.kv_slicing_spec_handle, slicing_parameters
            )
        if status != qaicrt.QStatus.QS_SUCCESS:
            raise MemoryError("Failed to setData")
        if self.queue.enqueue(self.execObj[exec_obj_idx]) != qaicrt.QStatus.QS_SUCCESS:
            raise MemoryError("Failed to enqueue")
        return exec_obj_idx

    def np_run_pipeline(
        self, inputs: Dict[str, np.ndarray], slicing_parameters=None, last_chunk: bool = False, kv_cache_buffers=None
    ) -> int:
        """Enqueue one prefill chunk (non-blocking). On ``last_chunk`` the DMA
        handoff into ``kv_cache_buffers`` is wired before enqueue. Returns the
        prefill execObj index.
        """
        exec_obj_idx = self.prefill_available_exec_objs.get()
        if last_chunk:
            if kv_cache_buffers is None:
                raise ValueError("last_chunk requires kv_cache_buffers to wire the handoff")
            batch_index = int(inputs["batch_index"].item()) if "batch_index" in inputs else 0
            buff_map = self.decode_rs_full_buff_map if self.is_hybrid_kv else self.kv_only_buff_map
            self.set_data_for_kv_handoff(
                kv_cache_buffers,
                [("batch_index", batch_index % self.full_batch_size), ("ctx_start", 0)],
                exec_obj_idx,
                buff_map,
            )
        self._make_inputs_contiguous(inputs)
        tuple_list = self._tuple_list_with_outputs(inputs, exec_obj_idx)
        if slicing_parameters is None:
            status = self.execObj[exec_obj_idx].setData(tuple_list)
        else:
            status, _ = self.execObj[exec_obj_idx].setDataWithSlices(
                tuple_list, self.kv_slicing_spec_handle, slicing_parameters
            )
        if status != qaicrt.QStatus.QS_SUCCESS:
            raise MemoryError("Failed to setData")
        if self.queue.enqueue(self.execObj[exec_obj_idx]) != qaicrt.QStatus.QS_SUCCESS:
            raise MemoryError("Failed to enqueue")
        return exec_obj_idx

    def complete_inf(self, index: int, is_prefill: bool) -> None:
        """Block until execObj ``index`` finishes; release prefill slots back to
        the pool.
        """
        if self.execObj[index].waitForCompletion() != qaicrt.QStatus.QS_SUCCESS:
            raise ValueError(self._shape_mismatch_message(self.buf_dims[index]))
        if is_prefill:
            self.prefill_available_exec_objs.put(index)

    def get_outputs(self, index: int) -> Dict[str, np.ndarray]:
        """Return the readable (non-RetainedState) outputs of execObj ``index``.

        On the pooled path the runtime DMA-writes these outputs into the per-slot
        host arrays wired at enqueue, so we read straight from ``output_buffers``
        (``getData`` returns empty for tuple-list enqueues). RetainedState KV goes
        directly to the caller's shared arrays via the slicing spec and is not
        surfaced here.
        """
        outputs: Dict[str, np.ndarray] = {}
        for name, _ in self.readable_output_bindings:
            output = self.output_buffers[index][name]
            outputs[name] = output
            outputs.setdefault(name.rsplit("/", 1)[-1], output)
        return outputs

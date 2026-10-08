# -----------------------------------------------------------------------------
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------

"""Runtime input dumping utilities for qaic-runner replay.

The dumper is intentionally session-local: each QPC gets its own directory with
an ``aic_batch_io.json`` descriptor containing one entry per runtime invocation.
Inputs are serialized as raw host buffers, while output entries describe the
buffers qaic-runner should allocate for the same invocation sequence.
"""

from __future__ import annotations

import json
import os
import re
from itertools import count
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence, Union

import numpy as np

_DUMP_ENV_VAR = "QEFFICIENT_DUMP_INPUTS"
_DEFAULT_DUMP_DIR = "qeff_input_dumps"
_SESSION_COUNTER = count()


def _as_jsonable(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return {
            "shape": list(value.shape),
            "dtype": str(value.dtype),
            "min": value.min().item() if value.size else None,
            "max": value.max().item() if value.size else None,
        }
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): _as_jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_as_jsonable(item) for item in value]
    return value


def _safe_filename(name: str) -> str:
    safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", name).strip("_")
    return safe or "buffer"


def _resolve_dump_root(path: Optional[Union[str, Path, bool]]) -> Optional[Path]:
    if path is False:
        return None
    if path is None:
        env_path = os.environ.get(_DUMP_ENV_VAR)
        if not env_path:
            return None
        path = env_path
    if path is True:
        path = _DEFAULT_DUMP_DIR
    return Path(path).expanduser().resolve()


def resolve_runtime_dump_inputs_path(
    artifacts: Optional[Union[str, Path, bool]],
    dump_inputs_path: Optional[Union[str, Path, bool]] = None,
) -> Optional[Union[str, Path, bool]]:
    """Resolve runtime input dumping from path-valued artifacts mode.

    ``artifacts=True`` keeps the existing dry-run runner-bundle behavior in
    public ``generate`` APIs. A string or ``Path`` value enables executed runtime
    input dumps at that location. ``dump_inputs_path`` is retained for internal
    plumbing and backward compatibility.
    """
    if dump_inputs_path is not None:
        return dump_inputs_path
    if isinstance(artifacts, (str, Path)):
        return artifacts
    return None


def _make_unique_dir(root: Path, component_name: str) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    session_id = next(_SESSION_COUNTER)
    base = f"{session_id:02d}_{_safe_filename(component_name)}"
    session_dir = root / base
    suffix = 1
    while session_dir.exists():
        session_dir = root / f"{base}_{suffix}"
        suffix += 1
    session_dir.mkdir(parents=True)
    return session_dir


class QAICInputDumper:
    """Collect qaic-runner inputs for every invocation of one QPC session."""

    def __init__(self, *, session, qpc_path: Union[str, Path], root: Union[str, Path], component_name: str):
        self.session = session
        self.qpc_path = Path(qpc_path)
        self.component_name = component_name
        self.session_dir = _make_unique_dir(Path(root), component_name)
        self.data_dir = self.session_dir / "data"
        self.data_dir.mkdir()
        self.io_batches: list[list[dict[str, Any]]] = []
        self.invocations: list[dict[str, Any]] = []
        self.pending_slice_configs: dict[int, list[dict[str, Any]]] = {}
        self._write_manifest()

    @classmethod
    def from_config(
        cls,
        *,
        session,
        qpc_path: Union[str, Path],
        dump_inputs_path: Optional[Union[str, Path, bool]],
        component_name: Optional[str],
    ) -> Optional["QAICInputDumper"]:
        root = _resolve_dump_root(dump_inputs_path)
        if root is None:
            return None
        resolved_component = component_name or Path(qpc_path).name or "qpc"
        return cls(session=session, qpc_path=qpc_path, root=root, component_name=resolved_component)

    def _binding_item_size(self, name: str) -> int:
        binding = self.session.bindings[self.session.binding_index_map[name]]
        return int(self.session.aic_to_np_dtype_mapping[binding.type].itemsize)

    def _canonical_binding_name(self, name: str) -> str:
        return self.session.bindings[self.session.binding_index_map[name]].name

    def _binding_dims(self, name: str) -> list[int]:
        binding = self.session.bindings[self.session.binding_index_map[name]]
        return [int(dim) for dim in binding.dims]

    @staticmethod
    def _array_dims(value: np.ndarray) -> list[int]:
        return list(value.shape) if value.shape else [1]

    def _write_input(self, invocation_dir: Path, name: str, value: np.ndarray) -> dict[str, Any]:
        array = np.ascontiguousarray(np.asarray(value))
        relative_path = Path("data") / invocation_dir.name / f"{_safe_filename(name)}.raw"
        array.tofile(self.session_dir / relative_path)
        return {
            "path": relative_path.as_posix(),
            "io-direction": "in",
            "elem-size": int(array.itemsize),
            "map-to": name,
            "dims": self._array_dims(array),
        }

    def _output_entry(self, invocation_dir: Path, name: str, shape: Optional[Sequence[int]] = None) -> dict[str, Any]:
        relative_path = Path("data") / invocation_dir.name / f"{_safe_filename(name)}.raw"
        return {
            "path": relative_path.as_posix(),
            "io-direction": "out",
            "elem-size": self._binding_item_size(name),
            "map-to": name,
            "dims": [int(dim) for dim in (shape if shape is not None else self._binding_dims(name))],
        }

    def record_invocation(
        self,
        *,
        kind: str,
        inputs: Mapping[str, np.ndarray],
        output_names: Sequence[str],
        output_shapes: Optional[Mapping[str, Sequence[int]]] = None,
        metadata: Optional[Mapping[str, Any]] = None,
        exec_obj_index: Optional[int] = None,
    ) -> None:
        invocation_index = len(self.io_batches)
        invocation_dir = self.data_dir / f"{invocation_index:05d}"
        invocation_dir.mkdir()

        entries: list[dict[str, Any]] = []
        recorded_inputs = []
        for name, value in inputs.items():
            if value is None or name not in self.session.binding_index_map:
                continue
            name = self._canonical_binding_name(name)
            entry = self._write_input(invocation_dir, name, value)
            entries.append(entry)
            recorded_inputs.append(name)

        output_shapes = output_shapes or {}
        recorded_outputs = []
        for name in output_names:
            if name not in self.session.binding_index_map:
                continue
            canonical_name = self._canonical_binding_name(name)
            entries.append(
                self._output_entry(
                    invocation_dir,
                    canonical_name,
                    output_shapes.get(name, output_shapes.get(canonical_name)),
                )
            )
            name = canonical_name
            recorded_outputs.append(name)

        self.io_batches.append(entries)
        invocation_metadata = {
            "index": invocation_index,
            "kind": kind,
            "exec_obj_index": exec_obj_index,
            "inputs": recorded_inputs,
            "outputs": recorded_outputs,
        }
        if metadata:
            invocation_metadata.update(_as_jsonable(metadata))
        if exec_obj_index in self.pending_slice_configs:
            invocation_metadata["slice_configs"] = self.pending_slice_configs.pop(exec_obj_index)
        self.invocations.append(invocation_metadata)
        self._write_batch_io()
        self._write_manifest()

    def record_slice_config(
        self,
        *,
        index: int,
        kv_cache_buffers: Sequence[np.ndarray],
        slicing_parameters: Any,
        buff_map: Sequence[tuple[str, int]],
    ) -> None:
        slice_index = sum(len(items) for items in self.pending_slice_configs.values())
        slice_dir = self.data_dir / f"slice_{slice_index:05d}"
        slice_dir.mkdir()
        buffers = []
        for buffer_index, ((name, binding_index), value) in enumerate(zip(buff_map, kv_cache_buffers)):
            array = np.ascontiguousarray(np.asarray(value))
            relative_path = Path("data") / slice_dir.name / f"{buffer_index:05d}_{_safe_filename(name)}.raw"
            array.tofile(self.session_dir / relative_path)
            buffers.append(
                {
                    "name": name,
                    "binding-index": int(binding_index),
                    "path": relative_path.as_posix(),
                    "elem-size": int(array.itemsize),
                    "dims": self._array_dims(array),
                    "dtype": str(array.dtype),
                }
            )
        self.pending_slice_configs.setdefault(index, []).append(
            {
                "slicing_parameters": _as_jsonable(slicing_parameters),
                "buffers": buffers,
            }
        )
        self._write_manifest()

    def _write_batch_io(self) -> None:
        (self.session_dir / "aic_batch_io.json").write_text(json.dumps({"IO-files": self.io_batches}, indent=2))

    def _write_manifest(self) -> None:
        manifest = {
            "version": 1,
            "component": self.component_name,
            "qpc_path": str(self.qpc_path),
            "aic_batch_io": "aic_batch_io.json",
            "env_var": _DUMP_ENV_VAR,
            "invocations": self.invocations,
        }
        if self.pending_slice_configs:
            manifest["pending_slice_configs"] = _as_jsonable(self.pending_slice_configs)
        (self.session_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))

# -----------------------------------------------------------------------------
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
#
# ----------------------------------------------------------------------------

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Union

WEIGHT_SPEC_VERSION = 5


@dataclass
class TiedWeightAlias:
    """Alias entry for checkpoint weights shared by multiple model parameters."""

    alias: str
    canonical: str


@dataclass
class ExternalDataFile:
    """External checkpoint file referenced by a weight spec."""

    path: str
    format: str


CheckpointFile = ExternalDataFile


@dataclass
class WeightSpecLocation:
    """Location of one external weight tensor inside a checkpoint file."""

    file: Union[int, str]
    key: str


@dataclass
class WeightSpecInput:
    """Mapping from an ONNX graph value to its external checkpoint tensor.

    Entries use the same file/key location contract regardless of whether the
    checkpoint tensor is consumed directly or through an export-time transform.
    """

    name: str
    location: WeightSpecLocation  # required: every spec entry must point to a file
    role: str = "weight"
    weight_input: str | None = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class WeightSpec:
    """Serializable weight-free export metadata.

    The spec records the checkpoint files shipped beside the ONNX model and
    maps ONNX weight graph values back to tensor keys in those files.
    """

    model_name: str
    model_id: str
    files: List[ExternalDataFile] = field(default_factory=list)
    inputs: List[WeightSpecInput] = field(default_factory=list)
    version: int = WEIGHT_SPEC_VERSION

    def to_dict(self) -> Dict[str, Any]:
        """Return a JSON-serializable representation of the weight spec."""
        data = asdict(self)
        data["model_id"] = str(data["model_id"])
        for entry in data["inputs"]:
            if entry.get("role") == "weight":
                entry.pop("role", None)
            if entry.get("weight_input") is None:
                entry.pop("weight_input", None)
            if not entry.get("metadata"):
                entry.pop("metadata", None)
        return data


def save_weight_spec(path: Path, spec: WeightSpec) -> Path:
    """Write a weight spec JSON file.

    Parameters
    ----------
    path : Path
        Destination path for the JSON file.
    spec : WeightSpec
        Weight spec instance to serialize.

    Returns
    -------
    Path
        The destination path.
    """
    with path.open("w", encoding="utf-8") as handle:
        json.dump(spec.to_dict(), handle, indent=2, sort_keys=True)
    return path


def _load_files(raw: list) -> List[ExternalDataFile]:
    """Deserialize external file entries, including the legacy string format."""
    if not raw:
        return []
    # Backward compat: old format stored plain strings
    if isinstance(raw[0], str):
        return [ExternalDataFile(path=entry, format="safetensors") for entry in raw]
    return [ExternalDataFile(**entry) for entry in raw]


def _load_location(raw: dict) -> WeightSpecLocation:
    """Deserialize a weight location entry from the spec JSON payload."""
    # Backward compat: old format had a redundant "type" field on the location
    return WeightSpecLocation(file=raw["file"], key=raw["key"])


def load_weight_spec(path: Path) -> WeightSpec:
    """Load a weight spec JSON file.

    Parameters
    ----------
    path : Path
        Path to ``weight_spec.json``.

    Returns
    -------
    WeightSpec
        Parsed weight spec with backward-compatible fields normalized.
    """
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)

    return WeightSpec(
        model_name=data["model_name"],
        model_id=data["model_id"],
        files=_load_files(data.get("files", data.get("checkpoint_files", []))),
        inputs=[
            WeightSpecInput(
                name=entry["name"],
                location=_load_location(entry["location"]),
                role=entry.get("role", "weight"),
                weight_input=entry.get("weight_input"),
                metadata=entry.get("metadata", {}),
            )
            for entry in data["inputs"]
            if entry.get("location") is not None  # backward compat: skip old buffer-only entries
        ],
        version=data.get("version", WEIGHT_SPEC_VERSION),
    )


def resolve_weight_spec_path(onnx_path: Path) -> Path:
    """Return the sidecar weight spec path for an ONNX export path."""
    return onnx_path.with_name("weight_spec.json")

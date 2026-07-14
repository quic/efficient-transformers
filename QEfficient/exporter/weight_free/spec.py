#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
#

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Union

WEIGHT_SPEC_VERSION = 5


@dataclass
class TiedWeightAlias:
    alias: str
    canonical: str


@dataclass
class ExternalDataFile:
    path: str
    format: str


CheckpointFile = ExternalDataFile


@dataclass
class WeightSpecLocation:
    file: Union[int, str]
    key: str


@dataclass
class WeightSpecInput:
    name: str
    location: WeightSpecLocation  # required: every spec entry must point to a file


@dataclass
class WeightSpec:
    model_name: str
    model_id: str
    files: List[ExternalDataFile] = field(default_factory=list)
    inputs: List[WeightSpecInput] = field(default_factory=list)
    version: int = WEIGHT_SPEC_VERSION

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data["model_id"] = str(data["model_id"])
        return data


def save_weight_spec(path: Path, spec: WeightSpec) -> Path:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(spec.to_dict(), handle, indent=2, sort_keys=True)
    return path


def _load_files(raw: list) -> List[ExternalDataFile]:
    if not raw:
        return []
    # Backward compat: old format stored plain strings
    if isinstance(raw[0], str):
        return [ExternalDataFile(path=entry, format="safetensors") for entry in raw]
    return [ExternalDataFile(**entry) for entry in raw]


def _load_location(raw: dict) -> WeightSpecLocation:
    # Backward compat: old format had a redundant "type" field on the location
    return WeightSpecLocation(file=raw["file"], key=raw["key"])


def load_weight_spec(path: Path) -> WeightSpec:
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
            )
            for entry in data["inputs"]
            if entry.get("location") is not None  # backward compat: skip old buffer-only entries
        ],
        version=data.get("version", WEIGHT_SPEC_VERSION),
    )


def resolve_weight_spec_path(onnx_path: Path) -> Path:
    return onnx_path.with_name("weight_spec.json")

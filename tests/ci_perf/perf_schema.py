# -----------------------------------------------------------------------------
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

SCHEMA_VERSION = "1"

# Metrics that gate the PR (regression → PR blocked).
# Value = regression direction: "up" means going UP is bad, "down" means going DOWN is bad.
GATED_METRICS: dict[str, str] = {
    "prefill_time": "up",
    "decode_perf": "down",
    "onnx_size_bytes": "up",
    "qpc_size_bytes": "up",
}

# Metrics that are stored for visibility but never cause a failure.
INFO_METRICS: list[str] = ["total_perf", "total_time"]


@dataclass
class ModelPerfRecord:
    """Per-model metrics captured during a single test run."""

    prefill_time: Optional[float] = None
    decode_perf: Optional[float] = None
    total_perf: Optional[float] = None
    total_time: Optional[float] = None
    onnx_size_bytes: Optional[int] = None
    qpc_size_bytes: Optional[int] = None
    batch_size: int = 1
    test_nodeid: str = ""

    def to_dict(self) -> dict:
        return {
            "prefill_time": self.prefill_time,
            "decode_perf": self.decode_perf,
            "total_perf": self.total_perf,
            "total_time": self.total_time,
            "onnx_size_bytes": self.onnx_size_bytes,
            "qpc_size_bytes": self.qpc_size_bytes,
            "batch_size": self.batch_size,
            "test_nodeid": self.test_nodeid,
        }


@dataclass
class StagePerfReport:
    """Full perf_report.json written after each QAIC pytest stage."""

    schema_version: str = SCHEMA_VERSION
    stage_name: str = ""
    ci_build_tag: str = ""
    git_sha: str = ""
    hardware: dict = field(default_factory=dict)
    timestamp: str = ""
    models: dict[str, dict] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "schema_version": self.schema_version,
            "stage_name": self.stage_name,
            "ci_build_tag": self.ci_build_tag,
            "git_sha": self.git_sha,
            "hardware": self.hardware,
            "timestamp": self.timestamp,
            "models": self.models,
        }


@dataclass
class BaselineEntry:
    """Per-model entry stored in the persistent baseline DB."""

    prefill_time: Optional[float] = None
    decode_perf: Optional[float] = None
    total_perf: Optional[float] = None
    total_time: Optional[float] = None
    onnx_size_bytes: Optional[int] = None
    qpc_size_bytes: Optional[int] = None
    baseline_sha: str = ""
    baseline_timestamp: str = ""

    def to_dict(self) -> dict:
        return {
            "prefill_time": self.prefill_time,
            "decode_perf": self.decode_perf,
            "total_perf": self.total_perf,
            "total_time": self.total_time,
            "onnx_size_bytes": self.onnx_size_bytes,
            "qpc_size_bytes": self.qpc_size_bytes,
            "baseline_sha": self.baseline_sha,
            "baseline_timestamp": self.baseline_timestamp,
        }

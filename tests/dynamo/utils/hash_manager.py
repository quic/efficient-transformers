# -----------------------------------------------------------------------------
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------

"""
Collision-safe directory management for parallel dynamo export/compile.

Under ``pytest-xdist -n auto`` multiple workers can attack the same
architecture concurrently. Point every export/compile at a directory keyed by
a deterministic hash of ``(architecture, feature, precision, extras)`` so:

* Parallel workers never write into the same folder.
* Repeat runs of the same test reuse the same folder (compile cache friendly).
* One run's artifacts do not leak into another's.

``QEFF_DYNAMO_WORKDIR`` overrides the default root (``test-results/dynamo/work``).
"""

from __future__ import annotations

import hashlib
import os
from pathlib import Path
from typing import Any, Iterable, Optional

DEFAULT_WORK_ENV = "QEFF_DYNAMO_WORKDIR"
DEFAULT_WORK_SUBDIR = Path("test-results") / "dynamo" / "work"


def _stringify(value: Any) -> str:
    if isinstance(value, (str, int, float, bool)) or value is None:
        return str(value)
    if isinstance(value, (list, tuple)):
        return "[" + ",".join(_stringify(v) for v in value) + "]"
    if isinstance(value, dict):
        return "{" + ",".join(f"{k}:{_stringify(v)}" for k, v in sorted(value.items())) + "}"
    return repr(value)


def compute_key(
    *,
    architecture: str,
    feature: str,
    precision: str = "fp32",
    extras: Optional[Iterable[Any]] = None,
) -> str:
    """Deterministic 12-char hex key for this (arch, feature, precision, extras) tuple."""
    payload = "|".join(
        [
            architecture,
            feature,
            precision,
            _stringify(list(extras) if extras is not None else []),
        ]
    )
    return hashlib.blake2b(payload.encode("utf-8"), digest_size=6).hexdigest()


def work_root() -> Path:
    env_value = os.environ.get(DEFAULT_WORK_ENV)
    if env_value:
        return Path(env_value)
    return Path.cwd() / DEFAULT_WORK_SUBDIR


def workdir_for(
    *,
    architecture: str,
    feature: str,
    precision: str = "fp32",
    extras: Optional[Iterable[Any]] = None,
    root: Optional[Path] = None,
) -> Path:
    """Return a collision-safe workdir path (created if missing)."""
    key = compute_key(architecture=architecture, feature=feature, precision=precision, extras=extras)
    base = root or work_root()
    path = base / architecture / feature / f"{precision}-{key}"
    path.mkdir(parents=True, exist_ok=True)
    return path

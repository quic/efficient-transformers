# -----------------------------------------------------------------------------
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------

"""
Root conftest for tests/dynamo/.

All dynamo tests require torch >= 2.13 (the minimum version that supports
torch.compiler.nested_compile_region and the dynamo export path).
"""

from __future__ import annotations

import pytest
import torch


def _parse_torch_version() -> tuple:
    parts = torch.__version__.split(".")
    try:
        return (int(parts[0]), int(parts[1]))
    except (IndexError, ValueError):
        return (0, 0)


def pytest_configure(config):
    config.addinivalue_line("markers", "dynamo: mark a test as part of the dynamo test suite")
    config.addinivalue_line("markers", "dynamo_export: CPU-only dynamo export smoke and parity tests")
    config.addinivalue_line("markers", "dynamo_compile: on-QAIC dynamo compile tests")
    config.addinivalue_line("markers", "dynamo_on_qaic: on-QAIC dynamo compile/generate/parity tests")


def pytest_collection_modifyitems(config, items):
    torch_version = _parse_torch_version()
    if torch_version < (2, 13):
        skip = pytest.mark.skip(reason=f"Dynamo tests require torch >= 2.13; running {torch.__version__}")
        for item in items:
            if item.fspath.parts and "dynamo" in str(item.fspath):
                item.add_marker(skip)


@pytest.fixture(autouse=True)
def set_deterministic_seed():
    torch.manual_seed(42)


@pytest.fixture
def tmp_export_dir(tmp_path):
    export_dir = tmp_path / "qeff_dynamo_exports"
    export_dir.mkdir(parents=True, exist_ok=True)
    return export_dir

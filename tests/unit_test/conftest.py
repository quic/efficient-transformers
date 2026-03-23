# -----------------------------------------------------------------------------
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------
"""
Shared fixtures and configuration for QEfficient unit_test tests.

CPU-only tests that do NOT require QAIC hardware.
Run with: pytest tests/unit_test/ -n auto -v
"""

import pytest
import torch


def pytest_configure(config):
    """Register custom markers for unit_test tests."""
    config.addinivalue_line("markers", "cpu_only: CPU-only test (no QAIC hardware required)")
    config.addinivalue_line("markers", "slow: slow test (ONNX export, model loading)")
    config.addinivalue_line("markers", "accuracy: accuracy test (numerical comparison between stages)")
    config.addinivalue_line("markers", "causal_lm: CausalLM model test")
    config.addinivalue_line("markers", "seq_classification: SeqClassification model test")
    config.addinivalue_line("markers", "embedding: Embedding model test")
    config.addinivalue_line("markers", "speech: Speech Seq2Seq model test")
    config.addinivalue_line("markers", "transforms: PyTorch transform test")
    config.addinivalue_line("markers", "cache: Cache utility test")
    config.addinivalue_line("markers", "onnx: ONNX export/ORT test")
    config.addinivalue_line("markers", "input_handler: InputHandler utility test")
    config.addinivalue_line("markers", "diffusers: QEfficient diffusers module test")


def pytest_collection_modifyitems(items):
    """Auto-add cpu_only marker to all tests in this directory."""
    for item in items:
        if "tests/unit_test" in str(item.fspath):
            item.add_marker(pytest.mark.cpu_only)


@pytest.fixture(autouse=True)
def set_cpu_threads():
    """Limit CPU threads per worker to avoid contention in parallel runs."""
    original = torch.get_num_threads()
    torch.set_num_threads(min(4, original))
    yield
    torch.set_num_threads(original)


@pytest.fixture(autouse=True)
def set_deterministic_seed():
    """Set random seed for reproducibility across all tests."""
    torch.manual_seed(42)
    yield


@pytest.fixture
def tmp_export_dir(tmp_path):
    """Provide a temporary directory for ONNX exports (unique per test)."""
    export_dir = tmp_path / "qeff_exports"
    export_dir.mkdir(parents=True, exist_ok=True)
    yield export_dir

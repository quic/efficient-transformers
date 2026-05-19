# -----------------------------------------------------------------------------
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------
"""
Local conftest for SPD hardware tests.

Patches QAICInferenceSession so that calls without an explicit device_ids
argument default to [4] instead of device 0.  Set QAIC_TEST_DEVICE_ID in
the environment to override (e.g. QAIC_TEST_DEVICE_ID=5 pytest ...).
"""

import os

import pytest

_DEVICE_ID = int(os.environ.get("QAIC_TEST_DEVICE_ID", "4"))


@pytest.fixture(autouse=True)
def _use_test_device(monkeypatch):
    """Redirect all bare QAICInferenceSession() calls to _DEVICE_ID."""
    from QEfficient.generation.cloud_infer import QAICInferenceSession

    _orig_init = QAICInferenceSession.__init__

    def _patched_init(self, qpc_path, device_ids=None, **kwargs):
        if device_ids is None:
            device_ids = [_DEVICE_ID]
        _orig_init(self, qpc_path, device_ids=device_ids, **kwargs)

    monkeypatch.setattr(QAICInferenceSession, "__init__", _patched_init)

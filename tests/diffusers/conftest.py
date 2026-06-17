# -----------------------------------------------------------------------------
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------

"""Diffusers test conftest — remap gated model IDs to tiny public substitutes.

Under the tiny profile lane (QEFF_TEST_PROFILE=tiny_model), FLUX.1-schnell is
gated and cannot be accessed without HF auth. This conftest swaps the
module-level constant in test_flux.py to a public tiny substitute at collection
time so no auth token is needed.
"""

import os


def pytest_configure(config):
    profile = os.environ.get("QEFF_TEST_PROFILE", "").strip()
    if profile != "tiny_model":
        return

    try:
        from tests.utils.profile_test_config import resolve_model_id
    except Exception:
        return

    # Patch module-level constants in test_flux before any test runs.
    # test_flux sets `model_id = "black-forest-labs/FLUX.1-schnell"` at module
    # scope so it runs before fixtures; we rewrite it here instead.
    try:
        import tests.diffusers.test_flux as _flux_mod

        original = _flux_mod.INITIAL_TEST_CONFIG["model_setup"].get("model_id")
        if original is None:
            # model_id lives as a module-level var, not inside INITIAL_TEST_CONFIG
            if hasattr(_flux_mod, "model_id"):
                _flux_mod.model_id = resolve_model_id(_flux_mod.model_id)
    except Exception:
        pass

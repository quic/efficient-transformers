# -----------------------------------------------------------------------------
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------

"""
Package-local conftest for the CausalLM dynamo tests.

Marker registration and cross-cutting helpers live in the parent
``tests/dynamo/conftest.py`` so they apply to every subfolder. This file
exists so pytest treats ``tests/dynamo/causal_lm`` as a discoverable rootdir
and so we can attach lane-specific fixtures here in the future without
touching the parent conftest.
"""

from __future__ import annotations

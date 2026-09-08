# -----------------------------------------------------------------------------
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
#
# ----------------------------------------------------------------------------

from QEfficient.exporter.weight_free.ort_weight_injection import load_weight_free_ort_inputs
from QEfficient.exporter.weight_free.weight_spec import resolve_weight_spec_path

__all__ = ["load_weight_free_ort_inputs", "resolve_weight_spec_path"]

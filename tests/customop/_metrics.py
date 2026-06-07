# -----------------------------------------------------------------------------
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------

"""Shared measurement helpers so every paged-KV test step reports the three
quantities of interest: runtime, peak memory (RSS), and numerical precision."""

import contextlib
import resource
import sys
import time


def peak_rss_mb() -> float:
    ru = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    # macOS reports ru_maxrss in bytes; Linux in kilobytes.
    return ru / (1024 * 1024) if sys.platform == "darwin" else ru / 1024


@contextlib.contextmanager
def measure(step_name: str):
    """Time a block and print runtime + peak RSS for the step."""
    t0 = time.perf_counter()
    try:
        yield
    finally:
        dt_ms = (time.perf_counter() - t0) * 1e3
        print(f"[metrics] {step_name}: time={dt_ms:.1f}ms peak_rss={peak_rss_mb():.0f}MB")


def report_precision(step_name: str, a, b) -> float:
    """Print + return max/mean abs difference between two tensors (precision)."""
    import torch

    diff = (a.float() - b.float()).abs()
    mx, mean = diff.max().item(), diff.mean().item()
    print(f"[metrics] {step_name}: precision max_abs_diff={mx:.3e} mean_abs_diff={mean:.3e}")
    return mx

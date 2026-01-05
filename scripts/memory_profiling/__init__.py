# -----------------------------------------------------------------------------
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------

"""
QEfficient Memory Profiling

A production-ready memory profiling solution specifically designed for QEfficient workflows.
Provides manual operation marking, comprehensive metrics collection, and professional visualization.

Usage Example:

```python
from scripts.memory_profiling import QEffMemoryProfiler

profiler = QEffMemoryProfiler(verbose=True)
profiler.start_monitoring()
# ... your QEfficient code ...
profiler.stop_monitoring()
print(profiler.get_memory_report())
profiler.generate_memory_graph()
```
"""

__version__ = "2.0.0"
__author__ = "Qualcomm Technologies, Inc."

# Core profiler components
from .profiler import (
    MetricsCollector,
    ProfilerConfig,
    ProfileSample,
    QEffMemoryProfiler,
)

# Visualization component (imported on-demand)
try:
    from .visualizer import QEffMemoryVisualizer
except ImportError:
    # Handle case where matplotlib is not available
    QEffMemoryVisualizer = None

__all__ = [
    "QEffMemoryProfiler",
    "ProfilerConfig",
    "ProfileSample",
    "MetricsCollector",
    "QEffMemoryVisualizer",
    "__version__",
]

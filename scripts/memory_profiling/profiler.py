# -----------------------------------------------------------------------------
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------

"""
QEfficient Memory Profiler - Production-Ready Memory Monitoring

This module provides comprehensive memory profiling capabilities specifically
designed for QEfficient workflows.
"""

import os
import threading
import time
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import psutil

from QEfficient.utils.logging_utils import logger


@dataclass
class ProfilerConfig:
    """Configuration for memory profiler."""

    sampling_interval: float = 0.05
    output_file: Optional[str] = None
    verbose: bool = False
    enable_cpu_monitoring: bool = True
    enable_disk_monitoring: bool = True


@dataclass
class ProfileSample:
    """Single profiling sample containing all metrics."""

    timestamp: datetime
    rss_mb: float
    vms_mb: float
    cpu_percent: float = 0.0
    disk_read_mb: float = 0.0
    disk_write_mb: float = 0.0
    disk_read_rate: float = 0.0
    disk_write_rate: float = 0.0


class MetricsCollector:
    """Handles collection of system metrics."""

    def __init__(self, config: ProfilerConfig):
        self.config = config
        self.process = psutil.Process(os.getpid())
        self._last_disk_counters = None
        self._last_disk_time = None
        self._cpu_initialized = False
        self._last_cpu_ema = 0.0
        self._cpu_ema_alpha = 0.3

    def initialize_cpu_monitoring(self) -> None:
        """Initialize CPU monitoring."""
        try:
            self.process.cpu_percent()  # First call to establish baseline
            self._cpu_initialized = True
            if self.config.verbose:
                logger.info("CPU measurement initialized")
        except Exception as e:
            if self.config.verbose:
                logger.warning(f"CPU initialization warning: {e}")
            self._cpu_initialized = False

    def get_memory_usage(self) -> Tuple[float, float]:
        """Get current memory usage in MB."""
        try:
            mem_info = self.process.memory_info()
            rss_mb = mem_info.rss / 1024 / 1024
            vms_mb = mem_info.vms / 1024 / 1024
            return rss_mb, vms_mb
        except Exception:
            return 0.0, 0.0

    def get_cpu_usage(self) -> float:
        """Get CPU usage with smoothing and normalization."""
        if not self.config.enable_cpu_monitoring:
            return 0.0

        try:
            if self._cpu_initialized:
                cpu_percent = self.process.cpu_percent()
                if cpu_percent >= 0:
                    # Normalize CPU to prevent > 100% on multi-core systems
                    # psutil.Process.cpu_percent() can return > 100% on multi-core
                    normalized_cpu = min(cpu_percent, 100.0)

                    # Apply exponential moving average smoothing
                    smoothed_cpu = self._cpu_ema_alpha * normalized_cpu + (1 - self._cpu_ema_alpha) * self._last_cpu_ema
                    self._last_cpu_ema = smoothed_cpu
                    return smoothed_cpu
            return self._last_cpu_ema
        except Exception:
            return 0.0

    def get_disk_io_stats(self) -> Tuple[float, float, float, float]:
        """Get disk I/O statistics with rate calculation."""
        if not self.config.enable_disk_monitoring:
            return 0.0, 0.0, 0.0, 0.0

        try:
            current_time = time.time()
            io_counters = self.process.io_counters()

            # Use read_chars/write_chars if available, fallback to read_bytes/write_bytes
            if hasattr(io_counters, "read_chars") and hasattr(io_counters, "write_chars"):
                read_bytes = io_counters.read_chars / 1024 / 1024
                write_bytes = io_counters.write_chars / 1024 / 1024
                use_chars = True
            else:
                read_bytes = io_counters.read_bytes / 1024 / 1024
                write_bytes = io_counters.write_bytes / 1024 / 1024
                use_chars = False

            # Calculate rates
            read_rate = 0.0
            write_rate = 0.0

            if self._last_disk_counters is not None and self._last_disk_time is not None:
                time_delta = current_time - self._last_disk_time
                if time_delta > 0:
                    if use_chars:
                        read_delta = read_bytes - (self._last_disk_counters.read_chars / 1024 / 1024)
                        write_delta = write_bytes - (self._last_disk_counters.write_chars / 1024 / 1024)
                    else:
                        read_delta = read_bytes - (self._last_disk_counters.read_bytes / 1024 / 1024)
                        write_delta = write_bytes - (self._last_disk_counters.write_bytes / 1024 / 1024)

                    read_rate = read_delta / time_delta
                    write_rate = write_delta / time_delta

            # Update counters
            self._last_disk_counters = io_counters
            self._last_disk_time = current_time

            return read_bytes, write_bytes, read_rate, write_rate

        except Exception:
            return 0.0, 0.0, 0.0, 0.0

    def collect_sample(self) -> ProfileSample:
        """Collect a complete profiling sample."""
        timestamp = datetime.now()
        rss_mb, vms_mb = self.get_memory_usage()
        cpu_percent = self.get_cpu_usage()
        read_bytes, write_bytes, read_rate, write_rate = self.get_disk_io_stats()

        return ProfileSample(
            timestamp=timestamp,
            rss_mb=rss_mb,
            vms_mb=vms_mb,
            cpu_percent=cpu_percent,
            disk_read_mb=read_bytes,
            disk_write_mb=write_bytes,
            disk_read_rate=read_rate,
            disk_write_rate=write_rate,
        )


class QEffMemoryProfiler:
    """
    Production-ready memory profiler for QEfficient workflows.

    Features:
    - Manual operation marking for QEfficient workflows
    - Production-quality visualization with detailed segment analysis
    - Precise memory attribution and performance metrics
    - Professional-grade reporting suitable for debugging and optimization
    """

    # Segment colors for visualization
    SEGMENT_COLORS = {
        "Initialization": "#E8E8E8",
        "Model Loading": "#FF6B6B",
        "Export": "#FFEAA7",
        "Model Export": "#FFEAA7",
        "Compilation": "#98D8C8",
        "Model Compilation": "#98D8C8",
        "Generation": "#F7DC6F",
        "Text Generation": "#F7DC6F",
        "Cleanup": "#AED6F1",
        "Completion": "#D5DBDB",
    }

    def __init__(
        self, sampling_interval: float = 0.05, output_file: Optional[str] = None, verbose: bool = False, **kwargs
    ):
        """
        Initialize the QEfficient Memory Profiler.

        Args:
            sampling_interval: Time between memory samples in seconds
            output_file: Output file for memory profile graph
            verbose: Enable verbose output for monitoring operations
        """
        # Create configuration
        self.config = ProfilerConfig(
            sampling_interval=sampling_interval,
            output_file=output_file or "qeff_memory_profile.png",
            verbose=verbose,
            **kwargs,
        )

        # Initialize components
        self.metrics_collector = MetricsCollector(self.config)

        # Monitoring state
        self.monitoring = False
        self.monitor_thread = None

        # Data storage
        self.samples: List[ProfileSample] = []
        self.operations: List[Tuple[datetime, str]] = []

        # Peak tracking
        self.peak_rss = 0.0
        self.peak_vms = 0.0
        self.peak_rss_time: Optional[datetime] = None
        self.peak_vms_time: Optional[datetime] = None
        self.peak_operation: Optional[str] = None

        # Operation tracking
        self.current_operation = "Initialization"
        self.operation_start_time = datetime.now()
        self.operation_durations: Dict[str, float] = {}
        self.operation_memory_deltas: Dict[str, float] = {}

    # Legacy property accessors for backward compatibility
    @property
    def timestamps(self) -> List[datetime]:
        """Get timestamps from samples."""
        return [sample.timestamp for sample in self.samples]

    @property
    def rss_memory(self) -> List[float]:
        """Get RSS memory values from samples."""
        return [sample.rss_mb for sample in self.samples]

    @property
    def vms_memory(self) -> List[float]:
        """Get VMS memory values from samples."""
        return [sample.vms_mb for sample in self.samples]

    @property
    def cpu_usage(self) -> List[float]:
        """Get CPU usage values from samples."""
        return [sample.cpu_percent for sample in self.samples]

    @property
    def disk_read_bytes(self) -> List[float]:
        """Get disk read bytes from samples."""
        return [sample.disk_read_mb for sample in self.samples]

    @property
    def disk_write_bytes(self) -> List[float]:
        """Get disk write bytes from samples."""
        return [sample.disk_write_mb for sample in self.samples]

    @property
    def disk_read_rate(self) -> List[float]:
        """Get disk read rates from samples."""
        return [sample.disk_read_rate for sample in self.samples]

    @property
    def disk_write_rate(self) -> List[float]:
        """Get disk write rates from samples."""
        return [sample.disk_write_rate for sample in self.samples]

    @property
    def sampling_interval(self) -> float:
        """Get sampling interval."""
        return self.config.sampling_interval

    @property
    def output_file(self) -> str:
        """Get output file path."""
        return self.config.output_file

    @property
    def verbose(self) -> bool:
        """Get verbose flag."""
        return self.config.verbose

    def start_monitoring(self) -> None:
        """Start continuous memory monitoring in background thread."""
        if self.monitoring:
            return

        # Initialize CPU measurement
        self.metrics_collector.initialize_cpu_monitoring()

        self.monitoring = True
        self.monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self.monitor_thread.start()

        if self.config.verbose:
            logger.info(f"QEff Memory monitoring started (sampling every {self.config.sampling_interval}s)")

    def stop_monitoring(self) -> None:
        """Stop memory monitoring and generate reports."""
        if not self.monitoring:
            return

        self.monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=1.0)

        # Mark completion
        self.mark_operation("Completion")

        if self.config.verbose:
            logger.info("QEff Memory monitoring stopped")

    def _monitor_loop(self) -> None:
        """Background monitoring loop."""
        while self.monitoring:
            try:
                # Collect sample
                sample = self.metrics_collector.collect_sample()
                self.samples.append(sample)

                # Update peaks
                self._update_peaks(sample)

                time.sleep(self.config.sampling_interval)

            except Exception as e:
                if self.config.verbose:
                    logger.warning(f"Monitoring error: {e}")
                break

    def _update_peaks(self, sample: ProfileSample) -> None:
        """Update peak memory tracking."""
        if sample.rss_mb > self.peak_rss:
            self.peak_rss = sample.rss_mb
            self.peak_rss_time = sample.timestamp
            self.peak_operation = self.current_operation

        if sample.vms_mb > self.peak_vms:
            self.peak_vms = sample.vms_mb
            self.peak_vms_time = sample.timestamp

    def mark_operation(self, operation_name: str) -> None:
        """Mark the start of a new operation."""
        current_time = datetime.now()
        current_rss = self.samples[-1].rss_mb if self.samples else 0.0

        # Record previous operation duration and memory delta
        if self.current_operation != "Initialization" and self.samples:
            duration = (current_time - self.operation_start_time).total_seconds()
            self.operation_durations[self.current_operation] = duration

            # Calculate memory delta from start of operation
            start_idx = max(0, len(self.samples) - max(1, int(duration / self.config.sampling_interval)))
            start_rss = self.samples[start_idx].rss_mb if start_idx < len(self.samples) else current_rss
            memory_delta = current_rss - start_rss
            self.operation_memory_deltas[self.current_operation] = memory_delta

        # Start new operation
        self.current_operation = operation_name
        self.operation_start_time = current_time
        self.operations.append((current_time, operation_name))

        if self.config.verbose:
            logger.info(f"{operation_name} | Memory: {current_rss:.1f} MB RSS")

    def get_synchronized_data(self) -> Dict[str, List[float]]:
        """Get synchronized data arrays."""
        if not self.samples:
            return {}

        start_time = self.samples[0].timestamp
        return {
            "timestamps": [(s.timestamp - start_time).total_seconds() for s in self.samples],
            "rss_memory": [s.rss_mb for s in self.samples],
            "vms_memory": [s.vms_mb for s in self.samples],
            "cpu_usage": [s.cpu_percent for s in self.samples],
            "disk_read_bytes": [s.disk_read_mb for s in self.samples],
            "disk_write_bytes": [s.disk_write_mb for s in self.samples],
            "disk_read_rate": [s.disk_read_rate for s in self.samples],
            "disk_write_rate": [s.disk_write_rate for s in self.samples],
        }

    def mark_segment(self, segment_name: str) -> None:
        """Convenience method for manual segment marking (API mode)."""
        self.mark_operation(segment_name)

    def stop_and_save(self, filename: Optional[str] = None) -> str:
        """Stop monitoring and save results (API mode convenience)."""
        self.stop_monitoring()
        self.generate_memory_graph(filename)
        return self.get_memory_report()

    def get_memory_report(self) -> str:
        """Generate comprehensive memory usage report."""
        if not self.samples:
            return "No memory data collected"

        current_sample = self.samples[-1]
        initial_sample = self.samples[0]

        # Calculate statistics
        rss_values = [s.rss_mb for s in self.samples]
        avg_rss = sum(rss_values) / len(rss_values)
        max_rss = max(rss_values)
        min_rss = min(rss_values)

        # Auto-scale units
        rss_scale, rss_unit = (1024, "GB") if max_rss > 2048 else (1, "MB")

        # Calculate disk I/O statistics
        disk_io_stats = ""
        if self.samples and len(self.samples) > 1:
            total_read = current_sample.disk_read_mb - initial_sample.disk_read_mb
            total_write = current_sample.disk_write_mb - initial_sample.disk_write_mb
            max_read_rate = max(s.disk_read_rate for s in self.samples)
            max_write_rate = max(s.disk_write_rate for s in self.samples)
            avg_read_rate = sum(s.disk_read_rate for s in self.samples) / len(self.samples)
            avg_write_rate = sum(s.disk_write_rate for s in self.samples) / len(self.samples)

            disk_io_stats = f"""
Disk I/O Statistics:
   • Total Read:     {total_read:.2f} MB
   • Total Write:    {total_write:.2f} MB
   • Peak Read Rate: {max_read_rate:.2f} MB/s
   • Peak Write Rate:{max_write_rate:.2f} MB/s
   • Avg Read Rate:  {avg_read_rate:.2f} MB/s
   • Avg Write Rate: {avg_write_rate:.2f} MB/s"""

        report = f"""
QEFFICIENT PERFORMANCE MONITORING REPORT
{"=" * 60}
Peak Memory Usage:
   • RSS (Physical): {self.peak_rss / rss_scale:.2f} {rss_unit} at {self.peak_rss_time.strftime("%H:%M:%S") if self.peak_rss_time else "N/A"}
   • VMS (Virtual):  {self.peak_vms / rss_scale:.2f} {rss_unit} at {self.peak_vms_time.strftime("%H:%M:%S") if self.peak_vms_time else "N/A"}
   • Peak during:    {self.peak_operation}

Memory Statistics:
   • Current RSS:    {current_sample.rss_mb / rss_scale:.2f} {rss_unit} (Delta: {(current_sample.rss_mb - initial_sample.rss_mb) / rss_scale:+.2f} {rss_unit})
   • Current VMS:    {current_sample.vms_mb / rss_scale:.2f} {rss_unit} (Delta: {(current_sample.vms_mb - initial_sample.vms_mb) / rss_scale:+.2f} {rss_unit})
   • Average RSS:    {avg_rss / rss_scale:.2f} {rss_unit}
   • Min/Max RSS:    {min_rss / rss_scale:.2f} / {max_rss / rss_scale:.2f} {rss_unit}
   • Memory Range:   {(max_rss - min_rss) / rss_scale:.2f} {rss_unit}{disk_io_stats}

Monitoring Info:
   • Duration:       {(current_sample.timestamp - initial_sample.timestamp).total_seconds():.1f} seconds
   • Data Points:    {len(self.samples)}
   • Operations:     {len(self.operations)}
   • Sampling Rate:  {self.config.sampling_interval}s

QEfficient Operations Timeline:"""

        # Add operation timeline
        if self.operations:
            start_time = self.samples[0].timestamp
            for i, (op_time, op_name) in enumerate(self.operations):
                relative_time = (op_time - start_time).total_seconds()
                duration = self.operation_durations.get(op_name, 0)
                memory_delta = self.operation_memory_deltas.get(op_name, 0)

                duration_str = f"({duration:.1f}s)" if duration > 0 else ""
                memory_str = f"[{memory_delta / rss_scale:+.1f} {rss_unit}]" if abs(memory_delta) > 10 else ""

                report += f"\n   {i + 1:2d}. {relative_time:6.1f}s - {op_name} {duration_str} {memory_str}"

        return report

    def generate_memory_graph(self, filename: Optional[str] = None) -> None:
        """Generate professional memory usage graph with QEfficient operation segments."""
        if not self.samples:
            logger.warning("No data to plot")
            return

        output_file = filename or self.config.output_file

        # Import visualization module
        from .visualizer import QEffMemoryVisualizer

        visualizer = QEffMemoryVisualizer(self)
        visualizer.generate_professional_graph(output_file)

        if self.config.verbose:
            logger.info(f"QEfficient memory profile saved as: {output_file}")

    # Legacy methods for backward compatibility
    def get_memory_usage(self) -> Tuple[float, float]:
        """Get current memory usage in MB (legacy method)."""
        return self.metrics_collector.get_memory_usage()

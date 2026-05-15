# -----------------------------------------------------------------------------
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
#
# ----------------------------------------------------------------------------
"""Runtime MagCache helpers for WAN pipelines.

This module implements a pipeline-level (graph-agnostic) MagCache controller.
It does not modify ONNX/QPC graph signatures.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional, Sequence

import numpy as np
import torch

# Wan2.2 T2V-A14B mag ratios from MagCache4Wan2.2.
DEFAULT_WAN2_2_T2V_A14B_MAG_RATIOS = [
    1.00124,
    1.00155,
    0.99822,
    0.99851,
    0.99696,
    0.99687,
    0.99703,
    0.99732,
    0.9966,
    0.99679,
    0.99602,
    0.99658,
    0.99578,
    0.99664,
    0.99484,
    0.9949,
    0.99633,
    0.996,
    0.99659,
    0.99683,
    0.99534,
    0.99549,
    0.99584,
    0.99577,
    0.99681,
    0.99694,
    0.99563,
    0.99554,
    0.9944,
    0.99473,
    0.99594,
    0.9964,
    0.99466,
    0.99461,
    0.99453,
    0.99481,
    0.99389,
    0.99365,
    0.99391,
    0.99406,
    0.99354,
    0.99361,
    0.99283,
    0.99278,
    0.99268,
    0.99263,
    0.99057,
    0.99091,
    0.99125,
    0.99126,
    0.65523,
    0.65252,
    0.98808,
    0.98852,
    0.98765,
    0.98736,
    0.9851,
    0.98535,
    0.98311,
    0.98339,
    0.9805,
    0.9806,
    0.97776,
    0.97771,
    0.97278,
    0.97286,
    0.96731,
    0.96728,
    0.95857,
    0.95855,
    0.94385,
    0.94385,
    0.92118,
    0.921,
    0.88108,
    0.88076,
    0.80263,
    0.80181,
]


def nearest_interp(src_array: np.ndarray, target_length: int) -> np.ndarray:
    """Nearest-neighbor interpolation used by the upstream MagCache scripts."""
    src_length = len(src_array)
    if target_length == 1:
        return np.array([src_array[-1]], dtype=np.float32)

    scale = (src_length - 1) / (target_length - 1)
    mapped_indices = np.round(np.arange(target_length) * scale).astype(int)
    return src_array[mapped_indices].astype(np.float32)


@dataclass
class _StreamState:
    cached_residual: Optional[torch.Tensor] = None
    accumulated_ratio: float = 1.0
    accumulated_err: float = 0.0
    accumulated_steps: int = 0

    def reset_accumulators(self) -> None:
        self.accumulated_ratio = 1.0
        self.accumulated_err = 0.0
        self.accumulated_steps = 0

    def reset_all(self) -> None:
        self.cached_residual = None
        self.reset_accumulators()


@dataclass
class WanMagCacheRuntime:
    """Runtime state machine for WAN MagCache.

    This class tracks per-stream state (cond/uncond), applies stage-aware retention
    windows, and decides whether to skip a QAIC forward call.
    """

    num_inference_steps: int
    do_classifier_free_guidance: bool
    threshold: float
    max_skip_steps: int
    retention_ratio: float
    split_step: Optional[int] = None
    ratios: Optional[Sequence[float]] = None
    verbose: bool = False

    call_index: int = 0
    skipped_calls: int = 0
    executed_calls: int = 0
    stream_states: Dict[str, _StreamState] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.threshold < 0:
            raise ValueError(f"`magcache_thresh` must be >= 0, got {self.threshold}.")
        if self.max_skip_steps < 0:
            raise ValueError(f"`magcache_K` must be >= 0, got {self.max_skip_steps}.")
        if not 0.0 <= self.retention_ratio <= 1.0:
            raise ValueError(f"`magcache_retention_ratio` must be in [0, 1], got {self.retention_ratio}.")

        self.calls_per_step = 2 if self.do_classifier_free_guidance else 1
        self.total_calls = self.num_inference_steps * self.calls_per_step
        self._prepared_ratios = self._prepare_ratios(
            self.ratios,
            num_steps=self.num_inference_steps,
            calls_per_step=self.calls_per_step,
        )

        self.stream_states = {"cond": _StreamState()}
        if self.do_classifier_free_guidance:
            self.stream_states["uncond"] = _StreamState()

        if self.split_step is not None:
            # Convert timestep split to invocation split (cond/uncond aware).
            self.split_step = int(self.split_step) * self.calls_per_step

    @staticmethod
    def _prepare_ratios(
        ratios: Optional[Sequence[float]],
        num_steps: int,
        calls_per_step: int,
    ) -> np.ndarray:
        raw = np.asarray(
            DEFAULT_WAN2_2_T2V_A14B_MAG_RATIOS if ratios is None else list(ratios),
            dtype=np.float32,
        )

        if calls_per_step == 1:
            # If user provides interleaved cond/uncond ratios, use cond stream.
            if raw.size % 2 == 0 and raw.size > 0:
                raw = raw[0::2]
            prepared = np.concatenate([np.array([1.0], dtype=np.float32), raw])
            if len(prepared) != num_steps:
                prepared = nearest_interp(prepared, num_steps)
            return prepared

        prepared = np.concatenate([np.array([1.0, 1.0], dtype=np.float32), raw])
        if len(prepared) != num_steps * 2:
            mag_ratio_cond = nearest_interp(prepared[0::2], num_steps)
            mag_ratio_uncond = nearest_interp(prepared[1::2], num_steps)
            prepared = np.empty(num_steps * 2, dtype=np.float32)
            prepared[0::2] = mag_ratio_cond
            prepared[1::2] = mag_ratio_uncond
        return prepared

    def _cache_allowed_for_call(self, call_index: int) -> bool:
        # Single-stage mode (e.g., no high/low split): warmup-only retention window.
        if self.split_step is None:
            return call_index >= int(self.total_calls * self.retention_ratio)

        # Wan2.2 T2V/I2V-like stage-aware retention scheduling.
        retain_high = int(self.split_step * self.retention_ratio)
        retain_low_end = int((self.total_calls - self.split_step) * self.retention_ratio + self.split_step)

        if call_index < retain_high:
            return False
        if self.split_step <= call_index <= retain_low_end:
            return False
        return True

    def should_skip(self, stream_name: str) -> bool:
        state = self.stream_states[stream_name]

        if not self._cache_allowed_for_call(self.call_index):
            return False
        if state.cached_residual is None:
            return False

        ratio = float(self._prepared_ratios[self.call_index])
        state.accumulated_ratio *= ratio
        state.accumulated_steps += 1
        state.accumulated_err += abs(1.0 - state.accumulated_ratio)

        should_skip = state.accumulated_err < self.threshold and state.accumulated_steps <= self.max_skip_steps
        if should_skip:
            self.skipped_calls += 1
            return True

        state.reset_accumulators()
        return False

    def get_cached_residual(self, stream_name: str) -> torch.Tensor:
        cached = self.stream_states[stream_name].cached_residual
        if cached is None:
            raise RuntimeError(f"MagCache residual is empty for stream '{stream_name}'.")
        return cached

    def complete_call(self, stream_name: str, residual: torch.Tensor) -> None:
        state = self.stream_states[stream_name]
        state.cached_residual = residual.detach()
        self.executed_calls += 1

        self.call_index += 1
        if self.call_index >= self.total_calls:
            self._reset_for_next_video()

    def complete_skip(self, stream_name: str) -> None:
        if stream_name not in self.stream_states:
            raise KeyError(f"Unknown stream name '{stream_name}'.")
        self.call_index += 1
        if self.call_index >= self.total_calls:
            self._reset_for_next_video()

    def _reset_for_next_video(self) -> None:
        self.call_index = 0
        for state in self.stream_states.values():
            state.reset_all()

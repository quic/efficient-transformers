# -----------------------------------------------------------------------------
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
#
# ----------------------------------------------------------------------------
"""Runtime MagCache helpers for Qwen-Image pipelines.

This module implements a pipeline-level (graph-agnostic) MagCache controller.
It does not modify ONNX/QPC graph signatures.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional, Sequence

import numpy as np
import torch

# Qwen-Image mag ratios from MagCache4QwenImage.
DEFAULT_QWEN_IMAGE_MAG_RATIOS = [
    1.64062,
    1.64062,
    1.45312,
    1.45312,
    1.1875,
    1.1875,
    1.17188,
    1.17188,
    1.05469,
    1.05469,
    1.21094,
    1.21094,
    1.10938,
    1.10938,
    1.11719,
    1.11719,
    1.13281,
    1.13281,
    1.11719,
    1.11719,
    1.07812,
    1.07812,
    1.07031,
    1.07031,
    1.08594,
    1.08594,
    1.08594,
    1.08594,
    1.07812,
    1.07812,
    1.03906,
    1.03906,
    1.04688,
    1.04688,
    1.07812,
    1.07812,
    1.07031,
    1.07031,
    1.03125,
    1.03125,
    1.07812,
    1.07812,
    1.04688,
    1.04688,
    1.04688,
    1.04688,
    1.04688,
    1.04688,
    1.03906,
    1.03906,
    1.01562,
    1.01562,
    1.03125,
    1.03125,
    1.02344,
    1.02344,
    1.02344,
    1.02344,
    1.02344,
    1.02344,
    1.03906,
    1.03906,
    1.0,
    1.0,
    1.01562,
    1.01562,
    1.0,
    1.0,
    0.99219,
    0.99219,
    1.00781,
    1.00781,
    0.98047,
    0.98047,
    0.95703,
    0.95703,
    0.96875,
    0.96875,
    0.99219,
    0.99219,
    0.92578,
    0.92578,
    0.92578,
    0.92578,
    0.90625,
    0.90625,
    0.85938,
    0.85938,
    0.80469,
    0.80469,
    0.87891,
    0.87891,
    0.75,
    0.75,
    0.60938,
    0.60938,
    0.55078,
    0.55078,
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
class QwenImageMagCacheRuntime:
    """Runtime state machine for Qwen-Image MagCache."""

    num_inference_steps: int
    do_classifier_free_guidance: bool
    threshold: float
    max_skip_steps: int
    retention_ratio: float
    ratios: Optional[Sequence[float]] = None
    verbose: bool = False

    call_index: int = 0
    skipped_calls: int = 0
    executed_calls: int = 0
    stream_states: Dict[str, _StreamState] = field(default_factory=dict)

    def _debug_print(self, message: str) -> None:
        if self.verbose:
            print(message)

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

    @staticmethod
    def _prepare_ratios(
        ratios: Optional[Sequence[float]],
        num_steps: int,
        calls_per_step: int,
    ) -> np.ndarray:
        raw = np.asarray(
            DEFAULT_QWEN_IMAGE_MAG_RATIOS if ratios is None else list(ratios),
            dtype=np.float32,
        )

        if calls_per_step == 1:
            if raw.size % 2 == 0 and raw.size > 0:
                raw = raw[0::2]
            prepared = np.concatenate([np.array([1.0], dtype=np.float32), raw])
            if len(prepared) != num_steps:
                prepared = nearest_interp(prepared, num_steps)
            return prepared

        prepared = np.concatenate([np.array([1.0, 1.0], dtype=np.float32), raw])
        if len(prepared) != num_steps * 2:
            cond_ratios = nearest_interp(prepared[0::2], num_steps)
            uncond_ratios = nearest_interp(prepared[1::2], num_steps)
            prepared = np.empty(num_steps * 2, dtype=np.float32)
            prepared[0::2] = cond_ratios
            prepared[1::2] = uncond_ratios
        return prepared

    def _cache_allowed_for_call(self, call_index: int) -> bool:
        return call_index >= int(self.total_calls * self.retention_ratio)

    def should_skip(self, stream_name: str) -> bool:
        state = self.stream_states[stream_name]

        if not self._cache_allowed_for_call(self.call_index):
            self._debug_print(
                f"[MagCache] call={self.call_index} stream={stream_name} diff=N/A "
                f"thresh={self.threshold:.6f} decision=run (retention window)"
            )
            return False
        if state.cached_residual is None:
            self._debug_print(
                f"[MagCache] call={self.call_index} stream={stream_name} diff=N/A "
                f"thresh={self.threshold:.6f} decision=run (cache cold start)"
            )
            return False

        ratio = float(self._prepared_ratios[self.call_index])
        state.accumulated_ratio *= ratio
        state.accumulated_steps += 1
        state.accumulated_err += abs(1.0 - state.accumulated_ratio)

        should_skip = state.accumulated_err < self.threshold and state.accumulated_steps <= self.max_skip_steps
        self._debug_print(
            f"[MagCache] call={self.call_index} stream={stream_name} diff={state.accumulated_err:.6f} "
            f"thresh={self.threshold:.6f} k={state.accumulated_steps}/{self.max_skip_steps} "
            f"decision={'skip' if should_skip else 'run'}"
        )

        if should_skip:
            self.skipped_calls += 1
            self._debug_print(f"[MagCache] stream={stream_name} diff<{self.threshold:.6f}; skipping this step for now.")
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
            self._reset_for_next_image()

    def complete_skip(self, stream_name: str) -> None:
        if stream_name not in self.stream_states:
            raise KeyError(f"Unknown stream name '{stream_name}'.")

        self.call_index += 1
        if self.call_index >= self.total_calls:
            self._reset_for_next_image()

    def _reset_for_next_image(self) -> None:
        self.call_index = 0
        for state in self.stream_states.values():
            state.reset_all()

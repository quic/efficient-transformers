# -----------------------------------------------------------------------------
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------

"""Scoped per-window model loading for layer-wise export."""

from __future__ import annotations

import contextlib
import functools
from typing import Dict, Sequence, Tuple, Union

from QEfficient.utils.layerwise_utils import LayerwiseContext, attach_layerwise_context


class CustomLoader:
    """Build QEff models while restricting sharded checkpoints to one layer window."""

    def __init__(self, qeff_factory, layer_prefix: Union[str, Sequence[str]]):
        self.qeff_factory = qeff_factory
        self.layer_prefixes: Tuple[str, ...] = (layer_prefix,) if isinstance(layer_prefix, str) else tuple(layer_prefix)

    def _selected_layer_prefixes(self, start: int, end: int) -> Tuple[str, ...]:
        return tuple(f"{prefix}{idx}." for prefix in self.layer_prefixes for idx in range(start, end))

    @contextlib.contextmanager
    def _shard_filter(self, start: int, end: int):
        import transformers

        original_get_checkpoint_shard_files = transformers.modeling_utils.get_checkpoint_shard_files
        layer_prefixes = self.layer_prefixes
        selected_prefixes = self._selected_layer_prefixes(start, end)

        @functools.wraps(original_get_checkpoint_shard_files)
        def patched_get_checkpoint_shard_files(*args, **kwargs):
            shard_files, metadata = original_get_checkpoint_shard_files(*args, **kwargs)
            weight_map = metadata.get("weight_map") if isinstance(metadata, dict) else None
            if not weight_map:
                return shard_files, metadata

            filtered_weight_map: Dict[str, str] = {}
            for checkpoint_key, shard_name in weight_map.items():
                if checkpoint_key.startswith(layer_prefixes):
                    if checkpoint_key.startswith(selected_prefixes):
                        filtered_weight_map[checkpoint_key] = shard_name
                    continue
                filtered_weight_map[checkpoint_key] = shard_name

            if not filtered_weight_map:
                return shard_files, metadata

            shard_name_to_path = {path.split("/")[-1]: path for path in shard_files}
            filtered_shard_names = sorted(set(filtered_weight_map.values()))
            filtered_shard_files = [
                shard_name_to_path[name] for name in filtered_shard_names if name in shard_name_to_path
            ]
            if not filtered_shard_files:
                return shard_files, metadata

            filtered_metadata = dict(metadata)
            filtered_metadata["weight_map"] = filtered_weight_map
            filtered_metadata["all_checkpoint_keys"] = list(filtered_weight_map.keys())
            return filtered_shard_files, filtered_metadata

        transformers.modeling_utils.get_checkpoint_shard_files = patched_get_checkpoint_shard_files
        try:
            yield
        finally:
            transformers.modeling_utils.get_checkpoint_shard_files = original_get_checkpoint_shard_files

    def load_window(self, model_id: str, config, start: int, end: int, total_layers: int):
        context = LayerwiseContext(start=start, end=end, total_layers=total_layers)
        with self._shard_filter(start, end):
            qeff_model = self.qeff_factory(model_id, config)
        attach_layerwise_context(qeff_model, context)
        if hasattr(qeff_model, "model"):
            attach_layerwise_context(qeff_model.model, context)
        for attr_name in ("lang_model", "vision_model"):
            sub_wrapper = getattr(qeff_model, attr_name, None)
            if sub_wrapper is not None:
                attach_layerwise_context(sub_wrapper, context)
                if hasattr(sub_wrapper, "model"):
                    attach_layerwise_context(sub_wrapper.model, context)
        return qeff_model

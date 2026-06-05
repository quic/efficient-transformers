# -----------------------------------------------------------------------------
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------

"""Generic per-window model loader for layer-wise ONNX export.

:class:`CustomLoader` materializes a *real* PyTorch model that contains the
weights for only a requested window of decoder layers. It delegates the actual
checkpoint -> module weight conversion to HuggingFace ``from_pretrained`` (so
model-specific restructuring such as fused-MoE experts is handled correctly),
while transparently restricting the sharded checkpoint to the window's layers so
that arbitrarily large models can be exported one window at a time without ever
materializing the full set of weights.
"""

import contextlib
import functools
from typing import Optional, Sequence, Tuple, Union

import torch

from QEfficient.utils.logging_utils import logger


class CustomLoader:
    """Load a window of decoder layers as a real PyTorch model.

    Parameters
    ----------
    hf_auto_class : type
        The HuggingFace auto class used to load the model (e.g.
        ``AutoModelForCausalLM``).
    pretrained_model_name_or_path : str
        HuggingFace hub id or local path to the model directory.
    layer_prefix : str
        State-dict key prefix(es) used for the repeated decoder layers, e.g.
        ``"model.layers."`` (or a sequence such as
        ``("model.layers.", "model.language_model.layers.")`` for multimodal
        models). Keys matching ``f"{prefix}{i}."`` belong to decoder layer ``i``;
        all other keys are always loaded (vision encoder, projector, embeddings,
        final norm, lm_head, ...).
    total_layers : int
        Total number of decoder layers in the model.
    from_pretrained_kwargs : dict, optional
        Keyword arguments forwarded to ``hf_auto_class.from_pretrained`` (e.g.
        ``torch_dtype``, ``attn_implementation``, ``config``).
    """

    def __init__(
        self,
        hf_auto_class,
        pretrained_model_name_or_path: str,
        layer_prefix: Union[str, Sequence[str]],
        total_layers: int,
        from_pretrained_kwargs: Optional[dict] = None,
    ) -> None:
        self.hf_auto_class = hf_auto_class
        self.pretrained_model_name_or_path = pretrained_model_name_or_path
        self.layer_prefixes: Tuple[str, ...] = (
            (layer_prefix,) if isinstance(layer_prefix, str) else tuple(layer_prefix)
        )
        self.total_layers = int(total_layers)
        self.from_pretrained_kwargs = dict(from_pretrained_kwargs or {})

    # ------------------------------------------------------------------
    # Window weight-map filtering
    # ------------------------------------------------------------------
    def _selected_layer_prefixes(self, start: int, end: int) -> Tuple[str, ...]:
        return tuple(f"{prefix}{i}." for prefix in self.layer_prefixes for i in range(start, end))

    @contextlib.contextmanager
    def _shard_filter(self, start: int, end: int):
        """Restrict sharded checkpoints to the window's layers during load.

        Patches ``transformers.modeling_utils.get_checkpoint_shard_files`` so
        only the shards containing the window's decoder-layer weights (plus all
        non-layer / edge weights) are returned. This is a no-op for single-file
        (non-sharded) checkpoints, which are loaded in full.
        """
        import transformers

        original = transformers.modeling_utils.get_checkpoint_shard_files
        selected_prefixes = self._selected_layer_prefixes(start, end)
        layer_prefixes = self.layer_prefixes

        @functools.wraps(original)
        def patched(*args, **kwargs):
            shard_files, metadata = original(*args, **kwargs)
            weight_map = metadata.get("weight_map") if isinstance(metadata, dict) else None
            if not weight_map:
                return shard_files, metadata

            filtered_weight_map = {}
            for checkpoint_key, shard_name in weight_map.items():
                if checkpoint_key.startswith(layer_prefixes):
                    if checkpoint_key.startswith(selected_prefixes):
                        filtered_weight_map[checkpoint_key] = shard_name
                    continue
                # Non-layer / edge weight (embeddings, final norm, lm_head, ...).
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

            metadata = dict(metadata)
            metadata["weight_map"] = filtered_weight_map
            metadata["all_checkpoint_keys"] = list(filtered_weight_map.keys())
            return filtered_shard_files, metadata

        transformers.modeling_utils.get_checkpoint_shard_files = patched
        try:
            yield
        finally:
            transformers.modeling_utils.get_checkpoint_shard_files = original

    # ------------------------------------------------------------------
    # Loading
    # ------------------------------------------------------------------
    def load_window_model(self, start: int, end: int) -> torch.nn.Module:
        """Load and return a real PyTorch model for layers ``[start, end)``.

        Uses HuggingFace ``from_pretrained`` (which applies any checkpoint ->
        module weight conversion) while restricting sharded checkpoints to the
        window's layers. Decoder layers outside the window remain
        un-materialized (left on ``meta`` by the loader) and are skipped by the
        model ``forward`` via the ``_start/_end`` window contract.
        """
        if end <= start:
            raise ValueError(f"Invalid window: start={start}, end={end}")

        with self._shard_filter(start, end):
            model = self.hf_auto_class.from_pretrained(
                self.pretrained_model_name_or_path,
                **self.from_pretrained_kwargs,
            )
        logger.info(f"Loaded model weights for layer window [{start}, {end})")
        return model

# -----------------------------------------------------------------------------
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------

"""I/O name alias helpers for retained-state KV cache bindings."""

import re
from typing import Dict, Optional

_KV_BINDING_PREFIX_RE = re.compile(
    r"^(past_key\.|past_value\.|compressed_kv\.|k_pe\.|conv_state\.|recurrent_state\.)"
    r"(\d+)_([A-Za-z0-9]+)(_InternalRetainedState|_RetainedState)?$"
)


def public_retained_state_name(output_name: str) -> Optional[str]:
    """Map internal subfunction retained-state outputs to public runtime names."""
    suffix = "_InternalRetainedState"
    if output_name.endswith(suffix):
        return output_name[: -len(suffix)] + "_RetainedState"
    return None


def canonical_kv_binding_name(name: str) -> Optional[str]:
    """Return canonical KV/cache binding alias without optional kv_cache_prefix infix."""
    basename = name.rsplit("/", 1)[-1]
    match = _KV_BINDING_PREFIX_RE.match(basename)
    if not match:
        return None
    stem, layer_idx, _, suffix = match.groups()
    return f"{stem}{layer_idx}{suffix or ''}"


def is_retained_state_name(name: str) -> bool:
    """Return True when an I/O binding participates in retained-state cache flow."""
    return name.startswith(("past_", "conv_state.", "recurrent_state.", "compressed_", "k_pe"))


def add_basename_binding_aliases(binding_index_map: Dict[str, int], bindings) -> None:
    """Allow callers to use stable public aliases for prefixed ONNX graph bindings."""
    for binding in bindings:
        basename = binding.name.rsplit("/", 1)[-1]
        binding_index_map.setdefault(basename, binding.index)

        canonical = canonical_kv_binding_name(basename)
        if canonical is not None:
            binding_index_map.setdefault(canonical, binding.index)

        public_name = public_retained_state_name(basename)
        if public_name is not None:
            binding_index_map.setdefault(public_name, binding.index)
            canonical_public = canonical_kv_binding_name(public_name)
            if canonical_public is not None:
                binding_index_map.setdefault(canonical_public, binding.index)

# -----------------------------------------------------------------------------
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------
"""Buffer map registry for organizing KV cache and output binding indices."""

from __future__ import annotations


def _kv_sort_key(item: tuple[str, int]) -> tuple[int, int]:
    """Sort key: layer index first, then key=0 / value=1."""
    name = item[0]
    layer = int(name.split(".")[1]) if "." in name else 0
    kind = 0 if name.startswith("past_key") else 1
    return (layer, kind)


class BuffMapRegistry:
    """
    Organizes binding indices into structured buffer maps for KV-cache
    handoff between prefill and decode sessions.

    Attributes
    ----------
    decode_buff_map : list[tuple[str, int]]
        Input KV buffers (past_key.*, past_value.*) sorted by layer.
    decode_rs_buff_map : list[tuple[str, int]]
        Output RetainedState KV bindings (all retained states).
    decode_rs_kv_only_buff_map : list[tuple[str, int]]
        Output RetainedState restricted to past_key.*/past_value.* only.
    prefill_buff_map : list[tuple[str, int]]
        Output RetainedState + logits, sorted by layer with logits appended.
    kv_only_buff_map : list[tuple[str, int]]
        Subset of prefill_buff_map with only past_key.*/past_value.* entries.
    """

    def __init__(
        self,
        decode_buff_map: list[tuple[str, int]],
        decode_rs_buff_map: list[tuple[str, int]],
        decode_rs_kv_only_buff_map: list[tuple[str, int]],
        prefill_buff_map: list[tuple[str, int]],
        kv_only_buff_map: list[tuple[str, int]],
    ) -> None:
        self.decode_buff_map = decode_buff_map
        self.decode_rs_buff_map = decode_rs_buff_map
        self.decode_rs_kv_only_buff_map = decode_rs_kv_only_buff_map
        self.prefill_buff_map = prefill_buff_map
        self.kv_only_buff_map = kv_only_buff_map

    @classmethod
    def build_from_bindings(
        cls,
        input_names: list[str],
        output_names: list[str],
        binding_index_map: dict[str, int],
    ) -> "BuffMapRegistry":
        """
        Construct all buffer maps from binding metadata.

        Parameters
        ----------
        input_names : list[str]
            All input binding names.
        output_names : list[str]
            All output binding names.
        binding_index_map : dict[str, int]
            Name -> binding index mapping.
        """
        decode_buff_map: list[tuple[str, int]] = sorted(
            [
                (name, binding_index_map[name])
                for name in input_names
                if name.startswith("past_key") or name.startswith("past_value")
            ],
            key=_kv_sort_key,
        )

        decode_rs_buff_map: list[tuple[str, int]] = sorted(
            [
                (name.replace("_RetainedState", ""), binding_index_map[name])
                for name in output_names
                if name.endswith("_RetainedState")
            ],
            key=_kv_sort_key,
        )

        decode_rs_kv_only_buff_map: list[tuple[str, int]] = [
            entry
            for entry in decode_rs_buff_map
            if entry[0].startswith("past_key") or entry[0].startswith("past_value")
        ]

        prefill_buff_map: list[tuple[str, int]] = sorted(
            [
                (name.replace("_RetainedState", ""), binding_index_map[name])
                for name in output_names
                if name.endswith("_RetainedState")
            ],
            key=_kv_sort_key,
        )
        for name in output_names:
            if name == "logits":
                prefill_buff_map.append((name, binding_index_map[name]))

        kv_only_buff_map: list[tuple[str, int]] = [
            entry for entry in prefill_buff_map if entry[0].startswith("past_key") or entry[0].startswith("past_value")
        ]

        return cls(
            decode_buff_map=decode_buff_map,
            decode_rs_buff_map=decode_rs_buff_map,
            decode_rs_kv_only_buff_map=decode_rs_kv_only_buff_map,
            prefill_buff_map=prefill_buff_map,
            kv_only_buff_map=kv_only_buff_map,
        )

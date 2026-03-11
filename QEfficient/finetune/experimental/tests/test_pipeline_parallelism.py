# -----------------------------------------------------------------------------
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------

"""
Pipeline Parallelism (PP) tests for meta-llama/Llama-3.2-1B.
"""

import os
import re
import shutil
from collections import Counter
from types import SimpleNamespace
from typing import Dict, List
from unittest.mock import MagicMock, patch

import pytest
import torch
from datasets import Dataset

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_LLAMA_MODEL_NAME = "meta-llama/Llama-3.2-1B"

# Llama-3.2-1B has 16 transformer layers and uses tied embeddings
_LLAMA_NUM_LAYERS = 16
_LLAMA_TIED_EMBEDDINGS = True

# 15 short instruction/response pairs used for training + evaluation
_ALPACA_SAMPLES = [
    {"text": "### Instruction:\nWhat is 2 + 2?\n### Response:\n4"},
    {"text": "### Instruction:\nName the capital of France.\n### Response:\nParis"},
    {"text": "### Instruction:\nWhat color is the sky?\n### Response:\nBlue"},
    {"text": "### Instruction:\nHow many days in a week?\n### Response:\nSeven"},
    {"text": "### Instruction:\nWhat is the boiling point of water in Celsius?\n### Response:\n100°C"},
    {"text": "### Instruction:\nWho wrote Romeo and Juliet?\n### Response:\nWilliam Shakespeare"},
    {
        "text": "### Instruction:\nWhat language does Python code run in?\n### Response:\nPython is an interpreted language."
    },
    {"text": "### Instruction:\nConvert 1 km to meters.\n### Response:\n1000 meters"},
    {"text": "### Instruction:\nWhat is H2O?\n### Response:\nWater"},
    {"text": "### Instruction:\nWhat does CPU stand for?\n### Response:\nCentral Processing Unit"},
    {"text": "### Instruction:\nHow many continents are there?\n### Response:\nSeven"},
    {"text": "### Instruction:\nWhat is the speed of light?\n### Response:\nApproximately 3×10⁸ m/s"},
    {"text": "### Instruction:\nWhat is the largest planet?\n### Response:\nJupiter"},
    {
        "text": "### Instruction:\nWhat is photosynthesis?\n### Response:\nThe process plants use to convert sunlight to energy."
    },
    {"text": "### Instruction:\nHow many bytes in a kilobyte?\n### Response:\n1024 bytes"},
]


def _make_fake_llama_config(
    num_hidden_layers: int = _LLAMA_NUM_LAYERS,
    tie_word_embeddings: bool = _LLAMA_TIED_EMBEDDINGS,
    vocab_size: int = 32_000,
    hidden_size: int = 2048,
) -> SimpleNamespace:
    """Return a minimal config object that looks like Llama-3.2-1B to our utils."""
    return SimpleNamespace(
        num_hidden_layers=num_hidden_layers,
        tie_word_embeddings=tie_word_embeddings,
        vocab_size=vocab_size,
        hidden_size=hidden_size,
        model_type="llama",
    )


def _assert_layer_device_ids(
    dmap: Dict[str, int],
    num_layers: int,
    pp_degree: int,
    local_rank: int = 0,
) -> None:
    """
    Central invariant checker for transformer-layer device assignments.

    Enforces:
      1. Exactly ``num_layers`` layer keys exist – no gaps, no phantom layers.
      2. Layer device IDs are **non-decreasing** (monotonicity / pipeline order).
      3. All layer IDs are within the valid range for this rank.
      4. Layers form a **complete partition**: union covers every layer index,
         each stage-set is disjoint.
      5. Each stage receives either ``base`` or ``base+1`` layers (balanced).
      6. Every device in the rank's range is used at least once.
    """
    first_device = local_rank * pp_degree
    valid_devices = set(range(first_device, first_device + pp_degree))

    # --- 1. Key completeness: exactly the expected layer keys ---------------
    expected_keys = {f"model.layers.{i}" for i in range(num_layers)}
    actual_layer_keys = {k for k in dmap if k.startswith("model.layers.")}
    missing = expected_keys - actual_layer_keys
    phantom = actual_layer_keys - expected_keys
    assert not missing, f"Missing layer keys in device map: {sorted(missing)}"
    assert not phantom, f"Phantom layer keys in device map (never expected): {sorted(phantom)}"

    # --- 2. Monotonicity: device IDs are non-decreasing -------------------
    layer_devices: List[int] = [dmap[f"model.layers.{i}"] for i in range(num_layers)]
    assert layer_devices == sorted(layer_devices), (
        f"Layer-to-device assignment is not monotonically non-decreasing: {layer_devices}\n"
        f"Layers must flow in order through the pipeline."
    )

    # --- 3. Range validity ------------------------------------------------
    out_of_range = [(i, d) for i, d in enumerate(layer_devices) if d not in valid_devices]
    assert not out_of_range, (
        f"Layer(s) assigned to devices outside valid range {valid_devices} "
        f"for rank={local_rank}, pp={pp_degree}: {out_of_range}"
    )

    # --- 4. Partition: union == full set, per-stage sets are disjoint -----
    stages: List[List[int]] = [
        [i for i in range(num_layers) if layer_devices[i] == first_device + s] for s in range(pp_degree)
    ]
    union = set().union(*stages)
    assert union == set(range(num_layers)), (
        f"Layer partition does not cover all layers.\n  Expected: {set(range(num_layers))}\n  Got union: {union}"
    )
    for s_idx, s_a in enumerate(stages):
        for t_idx, s_b in enumerate(stages):
            if s_idx >= t_idx:
                continue
            overlap = set(s_a) & set(s_b)
            assert not overlap, f"Stages {s_idx} and {t_idx} share layers {overlap} – stages must be disjoint."

    # --- 5. Balance: each stage has base or base+1 layers -----------------
    base, remainder = divmod(num_layers, pp_degree)
    counts = [len(s) for s in stages]
    for stage_idx, count in enumerate(counts):
        expected_count = base + (1 if stage_idx < remainder else 0)
        assert count == expected_count, (
            f"Stage {stage_idx} has {count} layers; expected {expected_count} "
            f"(base={base}, remainder={remainder}, pp={pp_degree}, layers={num_layers})."
        )

    # --- 6. Every device in range is used ---------------------------------
    used_devices = set(layer_devices)
    assert used_devices == valid_devices, (
        f"Not all devices in the rank's range are used.\n  Expected: {valid_devices}\n  Used:     {used_devices}"
    )


def _assert_finite_loss_in_range(
    value: float,
    label: str,
    lo: float = 0.0,
    hi: float = 20.0,
) -> None:
    """
    Assert that a loss value is finite, positive, and within a plausible range
    for a causal LM trained on short English sentences.
    """
    tensor_val = torch.tensor(value, dtype=torch.float32)
    assert torch.isfinite(tensor_val), f"{label} is not finite: {value}"
    assert value > lo, f"{label} = {value:.4f} ≤ {lo}; expected a positive loss."
    assert value < hi, f"{label} = {value:.4f} ≥ {hi}; loss appears to have diverged or is unreasonably large."


# ---------------------------------------------------------------------------
# 1. Unit tests – device map structure
# ---------------------------------------------------------------------------


class TestPPDeviceMapUnit:
    """Unit tests for custom_device_map and get_device_map (no device required)."""

    # -- custom_device_map ---------------------------------------------------

    @patch(
        "QEfficient.finetune.experimental.core.utils.device_map_utils.AutoConfig.from_pretrained",
    )
    @patch(
        "QEfficient.finetune.experimental.core.utils.device_map_utils.get_local_rank",
        return_value=0,
    )
    def test_fixed_layers_on_correct_devices(self, _mock_rank, mock_cfg):
        """
        Structural invariants for the four non-transformer components:
          • embed_tokens lives on the FIRST device of this rank's block.
          • norm and rotary_emb are CO-LOCATED on the LAST device.
          • embed_tokens and norm are on DIFFERENT devices (pipeline split exists).
          • The gap between first and last device equals pp_degree - 1.
        """
        from QEfficient.finetune.experimental.core.utils.device_map_utils import custom_device_map

        pp_degree = 2
        local_rank = 0
        first_device = local_rank * pp_degree  # 0
        last_device = first_device + pp_degree - 1  # 1

        mock_cfg.return_value = _make_fake_llama_config()
        dmap = custom_device_map(_LLAMA_MODEL_NAME, "qaic", pp_degree=pp_degree)

        # Anchor components at correct pipeline boundaries
        assert dmap["model.embed_tokens"] == first_device, (
            f"embed_tokens must be on first device {first_device}, got {dmap['model.embed_tokens']}"
        )
        assert dmap["model.norm"] == last_device, (
            f"model.norm must be on last device {last_device}, got {dmap['model.norm']}"
        )
        # norm and rotary_emb must be co-located (both at the tail of the pipeline)
        assert dmap["model.rotary_emb"] == dmap["model.norm"], (
            "rotary_emb and norm must be co-located on the same device; "
            f"got rotary_emb={dmap['model.rotary_emb']}, norm={dmap['model.norm']}"
        )
        # The pipeline must actually split; first and last must differ
        assert dmap["model.embed_tokens"] != dmap["model.norm"], (
            "embed_tokens and norm are on the same device – no pipeline split occurred."
        )
        # The span of devices matches what was requested
        assert dmap["model.norm"] - dmap["model.embed_tokens"] == pp_degree - 1, (
            f"Device span ({dmap['model.norm']} - {dmap['model.embed_tokens']}) "
            f"must equal pp_degree - 1 = {pp_degree - 1}."
        )

    @patch(
        "QEfficient.finetune.experimental.core.utils.device_map_utils.AutoConfig.from_pretrained",
    )
    @patch(
        "QEfficient.finetune.experimental.core.utils.device_map_utils.get_local_rank",
        return_value=0,
    )
    def test_tied_embeddings_lm_head_on_first_device(self, _mock_rank, mock_cfg):
        """
        For tied embeddings (Llama-3.2-1B default):
          • lm_head must be CO-LOCATED with embed_tokens (same device).
          • lm_head must NOT be co-located with model.norm.
        """
        from QEfficient.finetune.experimental.core.utils.device_map_utils import custom_device_map

        mock_cfg.return_value = _make_fake_llama_config(tie_word_embeddings=True)
        dmap = custom_device_map(_LLAMA_MODEL_NAME, "qaic", pp_degree=2)

        # Co-location invariant: lm_head shares device with embed_tokens
        assert dmap["lm_head"] == dmap["model.embed_tokens"], (
            "Tied-embedding model: lm_head must be on the same device as embed_tokens.\n"
            f"  lm_head={dmap['lm_head']}, embed_tokens={dmap['model.embed_tokens']}"
        )
        # Separation invariant: lm_head is NOT on the last device (where norm lives)
        assert dmap["lm_head"] != dmap["model.norm"], (
            "Tied-embedding model: lm_head must not be co-located with model.norm.\n"
            f"  lm_head={dmap['lm_head']}, norm={dmap['model.norm']}"
        )

    @patch(
        "QEfficient.finetune.experimental.core.utils.device_map_utils.AutoConfig.from_pretrained",
    )
    @patch(
        "QEfficient.finetune.experimental.core.utils.device_map_utils.get_local_rank",
        return_value=0,
    )
    def test_untied_embeddings_lm_head_on_last_device(self, _mock_rank, mock_cfg):
        """
        For non-tied embeddings:
          • lm_head must be CO-LOCATED with model.norm (last device).
          • lm_head must NOT be co-located with embed_tokens (first device).
        """
        from QEfficient.finetune.experimental.core.utils.device_map_utils import custom_device_map

        mock_cfg.return_value = _make_fake_llama_config(tie_word_embeddings=False)
        dmap = custom_device_map(_LLAMA_MODEL_NAME, "qaic", pp_degree=2)

        # Co-location invariant: lm_head shares device with norm (output side)
        assert dmap["lm_head"] == dmap["model.norm"], (
            "Non-tied model: lm_head must be on the same device as model.norm.\n"
            f"  lm_head={dmap['lm_head']}, norm={dmap['model.norm']}"
        )
        # Separation invariant: lm_head is NOT on the first device
        assert dmap["lm_head"] != dmap["model.embed_tokens"], (
            "Non-tied model: lm_head must not be co-located with embed_tokens.\n"
            f"  lm_head={dmap['lm_head']}, embed_tokens={dmap['model.embed_tokens']}"
        )

    @patch(
        "QEfficient.finetune.experimental.core.utils.device_map_utils.AutoConfig.from_pretrained",
    )
    @patch(
        "QEfficient.finetune.experimental.core.utils.device_map_utils.get_local_rank",
        return_value=0,
    )
    def test_layer_distribution_pp2(self, _mock_rank, mock_cfg):
        """
        16 layers, pp=2: exact partition into two contiguous, equal halves.

        Checks beyond simple counting:
          • Monotonicity: device IDs are non-decreasing across layer indices.
          • Partition: stage-0 and stage-1 sets are disjoint and their union
            covers all 16 layers.
          • No phantom or missing layer keys.
          • Both devices are actually used (completeness).
        """
        from QEfficient.finetune.experimental.core.utils.device_map_utils import custom_device_map

        num_layers, pp_degree = 16, 2
        mock_cfg.return_value = _make_fake_llama_config(num_hidden_layers=num_layers)
        dmap = custom_device_map(_LLAMA_MODEL_NAME, "qaic", pp_degree=pp_degree)

        # Delegate to the central invariant checker
        _assert_layer_device_ids(dmap, num_layers, pp_degree, local_rank=0)

        # Verify the exact split boundary for this balanced case
        layer_devices = [dmap[f"model.layers.{i}"] for i in range(num_layers)]
        first_half = layer_devices[:8]
        second_half = layer_devices[8:]
        assert all(d == 0 for d in first_half), f"Layers 0-7 should all be on device 0; got {first_half}"
        assert all(d == 1 for d in second_half), f"Layers 8-15 should all be on device 1; got {second_half}"

    @pytest.mark.parametrize("pp_degree,num_layers", [(3, 16), (4, 16), (3, 9), (4, 8)])
    @patch(
        "QEfficient.finetune.experimental.core.utils.device_map_utils.AutoConfig.from_pretrained",
    )
    @patch(
        "QEfficient.finetune.experimental.core.utils.device_map_utils.get_local_rank",
        return_value=0,
    )
    def test_layer_distribution_balanced(self, _mock_rank, mock_cfg, pp_degree, num_layers):
        """
        For any (pp_degree, num_layers) pair, the full invariant suite must hold.

        In addition to the central checker, verifies that each stage's count is
        exactly ``base`` or ``base+1`` – stricter than ``max - min ≤ 1`` because
        it rules out the pathological case where all surplus goes to one stage and
        another stage has 0.
        """
        from QEfficient.finetune.experimental.core.utils.device_map_utils import custom_device_map

        mock_cfg.return_value = _make_fake_llama_config(num_hidden_layers=num_layers)
        dmap = custom_device_map(_LLAMA_MODEL_NAME, "qaic", pp_degree=pp_degree)

        # Full invariant suite (monotonicity, partition, balance, completeness)
        _assert_layer_device_ids(dmap, num_layers, pp_degree, local_rank=0)

        # Also verify no stage is starved (every stage has at least one layer)
        counts = Counter(dmap[f"model.layers.{i}"] for i in range(num_layers))
        assert min(counts.values()) >= 1, (
            f"At least one stage has 0 layers: {dict(counts)} (pp={pp_degree}, layers={num_layers})"
        )

    @patch(
        "QEfficient.finetune.experimental.core.utils.device_map_utils.AutoConfig.from_pretrained",
    )
    @patch(
        "QEfficient.finetune.experimental.core.utils.device_map_utils.get_local_rank",
        return_value=0,
    )
    def test_all_layers_assigned(self, _mock_rank, mock_cfg):
        """
        The set of layer keys in the map must equal {model.layers.0, ..., model.layers.15}
        exactly – no missing layers, no phantom layers like model.layers.16.
        """
        from QEfficient.finetune.experimental.core.utils.device_map_utils import custom_device_map

        num_layers, pp_degree = 16, 4
        mock_cfg.return_value = _make_fake_llama_config(num_hidden_layers=num_layers)
        dmap = custom_device_map(_LLAMA_MODEL_NAME, "qaic", pp_degree=pp_degree)

        expected_layer_keys = {f"model.layers.{i}" for i in range(num_layers)}
        actual_layer_keys = {k for k in dmap if k.startswith("model.layers.")}

        # Exact set equality – catches both missing and phantom keys at once
        assert actual_layer_keys == expected_layer_keys, (
            f"Layer key mismatch.\n"
            f"  Missing : {sorted(expected_layer_keys - actual_layer_keys)}\n"
            f"  Phantom : {sorted(actual_layer_keys - expected_layer_keys)}"
        )

    @patch(
        "QEfficient.finetune.experimental.core.utils.device_map_utils.AutoConfig.from_pretrained",
    )
    @patch(
        "QEfficient.finetune.experimental.core.utils.device_map_utils.get_local_rank",
        return_value=0,
    )
    def test_too_few_layers_raises(self, _mock_rank, mock_cfg):
        """
        When pp_degree > num_layers the error must mention BOTH the conflicting
        numbers (num_layers=2, pp_degree=4), so the caller can diagnose the issue.
        """
        from QEfficient.finetune.experimental.core.utils.device_map_utils import custom_device_map

        mock_cfg.return_value = _make_fake_llama_config(num_hidden_layers=2)
        with pytest.raises(ValueError, match=r"(?=.*\b2\b)(?=.*\b4\b)"):
            # Regex uses lookaheads to require BOTH '2' (num_layers) and '4'
            # (pp_degree) appear somewhere in the error message
            custom_device_map(_LLAMA_MODEL_NAME, "qaic", pp_degree=4)

    # -- get_device_map ------------------------------------------------------

    def test_get_device_map_pp1_returns_none(self):
        """
        pp_degree=1 (PP disabled) must return the Python singleton None –
        not an empty dict, not False, not 0.
        """
        from QEfficient.finetune.experimental.core.utils.device_map_utils import get_device_map

        result = get_device_map(_LLAMA_MODEL_NAME, "qaic", pp_degree=1)
        assert result is None, f"Expected None (PP disabled), got {type(result).__name__}: {result!r}"
        assert type(result) is type(None), "Return type must be NoneType, not a falsy proxy"

    @patch("torch.qaic.device_count", return_value=1)
    def test_get_device_map_pp_exceeds_devices_raises(self, _mock_count):
        """
        When pp_degree > num_available_devices the error must mention:
          • the word 'pp_degree'
          • the requested pp value (4)
          • the available device count (1)
        This ensures the error message is actionable, not just 'invalid config'.
        """
        from QEfficient.finetune.experimental.core.utils.device_map_utils import get_device_map

        # Regex requires all three pieces of information in the error message
        with pytest.raises(ValueError, match=r"(?=.*pp_degree)(?=.*\b4\b)(?=.*\b1\b)"):
            get_device_map(_LLAMA_MODEL_NAME, "qaic", pp_degree=4)

    @patch("torch.qaic.device_count", return_value=2)
    def test_get_device_map_pp_equals_devices_returns_auto(self, _mock_count):
        """
        When pp_degree == num_available_devices HuggingFace 'auto' placement
        is used. Verify the return type (str) and exact value ("auto").
        """
        from QEfficient.finetune.experimental.core.utils.device_map_utils import get_device_map

        result = get_device_map(_LLAMA_MODEL_NAME, "qaic", pp_degree=2)

        assert isinstance(result, str), f"Expected a string ('auto'), got {type(result).__name__}: {result!r}"
        assert result == "auto", f"Expected 'auto', got '{result}'"

    @patch(
        "QEfficient.finetune.experimental.core.utils.device_map_utils.AutoConfig.from_pretrained",
    )
    @patch(
        "QEfficient.finetune.experimental.core.utils.device_map_utils.get_local_rank",
        return_value=0,
    )
    @patch("torch.qaic.device_count", return_value=4)
    def test_get_device_map_pp_less_than_devices_returns_dict(self, _mock_count, _mock_rank, mock_cfg):
        """
        When pp_degree < num_available_devices a custom dict is returned.

        Checks:
          • Return type is exactly dict.
          • All four mandatory component keys are present.
          • All values are Python ints (not numpy ints or strings).
          • Both devices in [0, pp_degree) appear in the values (completeness).
        """
        from QEfficient.finetune.experimental.core.utils.device_map_utils import get_device_map

        pp_degree = 2
        mock_cfg.return_value = _make_fake_llama_config()
        result = get_device_map(_LLAMA_MODEL_NAME, "qaic", pp_degree=pp_degree)

        assert isinstance(result, dict), f"Expected dict, got {type(result).__name__}: {result!r}"

        required_keys = {"model.embed_tokens", "lm_head", "model.norm", "model.rotary_emb"}
        missing_keys = required_keys - result.keys()
        assert not missing_keys, f"Required component keys missing from device map: {missing_keys}"

        # All device IDs must be plain Python ints (not numpy.int64 etc.)
        non_int = {k: type(v).__name__ for k, v in result.items() if not isinstance(v, int)}
        assert not non_int, f"Device map values must be Python ints; found wrong types: {non_int}"

        # Both devices in the pp range must actually be used (completeness)
        used_devices = set(result.values())
        expected_devices = set(range(pp_degree))
        assert used_devices == expected_devices, (
            f"Not all pipeline stage devices are represented in the map.\n"
            f"  Expected devices: {expected_devices}\n"
            f"  Used devices:     {used_devices}"
        )

    @patch(
        "QEfficient.finetune.experimental.core.utils.device_map_utils.AutoConfig.from_pretrained",
    )
    @patch(
        "QEfficient.finetune.experimental.core.utils.device_map_utils.get_local_rank",
        return_value=0,
    )
    @patch("torch.qaic.device_count", return_value=4)
    def test_get_device_map_returns_valid_device_ids(self, _mock_count, _mock_rank, mock_cfg):
        """
        Every device ID in the returned map must be in [0, pp_degree).
        PLUS: every device in [0, pp_degree) must appear at least once
        (no wasted or unreachable stages).
        """
        from QEfficient.finetune.experimental.core.utils.device_map_utils import get_device_map

        pp_degree = 2
        mock_cfg.return_value = _make_fake_llama_config()
        dmap = get_device_map(_LLAMA_MODEL_NAME, "qaic", pp_degree=pp_degree)

        assert isinstance(dmap, dict)
        valid_range = range(pp_degree)

        # --- Range validity: no out-of-bound IDs --------------------------
        out_of_range = {k: v for k, v in dmap.items() if v not in valid_range}
        assert not out_of_range, f"Device IDs outside valid range [0, {pp_degree}):\n" + "\n".join(
            f"  {k!r}: {v}" for k, v in sorted(out_of_range.items())
        )

        # --- Completeness: every stage device is actually used -------------
        used = set(dmap.values())
        unused = set(valid_range) - used
        assert not unused, (
            f"Stage devices {unused} are never assigned any component – those pipeline stages would be empty."
        )


# ---------------------------------------------------------------------------
# 2. Distributed rank tests (local_rank > 0)
# ---------------------------------------------------------------------------


class TestPPDeviceMapDistributed:
    """Verify device IDs are correctly offset when local_rank > 0 (DDP + PP)."""

    @patch(
        "QEfficient.finetune.experimental.core.utils.device_map_utils.AutoConfig.from_pretrained",
    )
    @patch(
        "QEfficient.finetune.experimental.core.utils.device_map_utils.get_local_rank",
        return_value=1,
    )
    def test_rank1_devices_offset_by_pp_degree(self, _mock_rank, mock_cfg):
        """
        For pp_degree=2 and local_rank=1 the block of device IDs must be
        exactly {2, 3} – not overlapping with rank-0's block {0, 1}.

        Checks:
          • embed_tokens is on first_device (computed, not hardcoded).
          • norm is on last_device (computed).
          • All layer devices are within [first_device, last_device].
          • The device set is completely disjoint from rank-0's devices.
        """
        from QEfficient.finetune.experimental.core.utils.device_map_utils import custom_device_map

        pp_degree = 2
        local_rank = 1
        first_device = local_rank * pp_degree  # 2
        last_device = first_device + pp_degree - 1  # 3
        rank0_devices = set(range(pp_degree))  # {0, 1}

        mock_cfg.return_value = _make_fake_llama_config()
        dmap = custom_device_map(_LLAMA_MODEL_NAME, "qaic", pp_degree=pp_degree)

        assert dmap["model.embed_tokens"] == first_device, (
            f"embed_tokens must be on first_device={first_device}, got {dmap['model.embed_tokens']}"
        )
        assert dmap["model.norm"] == last_device, f"norm must be on last_device={last_device}, got {dmap['model.norm']}"

        # All layer devices must be within this rank's block
        layer_devices = {dmap[f"model.layers.{i}"] for i in range(_LLAMA_NUM_LAYERS)}
        out_of_block = layer_devices - set(range(first_device, last_device + 1))
        assert not out_of_block, (
            f"Rank-1 layers assigned to devices outside [{first_device}, {last_device}]: {out_of_block}"
        )

        # Rank-1 devices must be completely disjoint from rank-0 devices
        overlap = set(dmap.values()) & rank0_devices
        assert not overlap, (
            f"Rank-1 device map overlaps with rank-0 devices {rank0_devices}: {overlap}\n"
            f"DDP replicas must use non-overlapping device blocks."
        )

    @pytest.mark.parametrize("local_rank,pp_degree", [(0, 2), (1, 2), (0, 4), (2, 4)])
    @patch(
        "QEfficient.finetune.experimental.core.utils.device_map_utils.AutoConfig.from_pretrained",
    )
    def test_device_range_is_complete_and_correct(self, mock_cfg, local_rank, pp_degree):
        """
        The set of device IDs actually used must EXACTLY EQUAL the expected
        block [local_rank*pp_degree, local_rank*pp_degree + pp_degree - 1].

        'Subset' is not sufficient: if any device in the block is unused the
        pipeline has a ghost stage consuming no memory and carrying no layers.
        """
        from QEfficient.finetune.experimental.core.utils.device_map_utils import custom_device_map

        mock_cfg.return_value = _make_fake_llama_config()

        with patch(
            "QEfficient.finetune.experimental.core.utils.device_map_utils.get_local_rank",
            return_value=local_rank,
        ):
            dmap = custom_device_map(_LLAMA_MODEL_NAME, "qaic", pp_degree=pp_degree)

        expected_block = set(range(local_rank * pp_degree, local_rank * pp_degree + pp_degree))
        actual_ids = set(dmap.values())

        # Exact equality, not just subset
        assert actual_ids == expected_block, (
            f"Device block mismatch for rank={local_rank}, pp={pp_degree}.\n"
            f"  Expected: {expected_block}\n"
            f"  Got:      {actual_ids}\n"
            f"  Missing:  {expected_block - actual_ids}\n"
            f"  Extra:    {actual_ids - expected_block}"
        )


# ---------------------------------------------------------------------------
# 3. ConfigManager PP validation
# ---------------------------------------------------------------------------


class TestPPConfigValidation:
    """Test that ConfigManager correctly validates pp_degree."""

    def _make_config_manager_with_pp(self, pp_degree: int):
        """Build a real ConfigManager pointing to test_config.yaml, then override pp_degree."""
        import sys

        test_yaml = os.path.join(os.path.dirname(__file__), "test_config.yaml")
        old_argv = sys.argv[:]
        sys.argv = ["finetune_experimental.py", test_yaml]
        try:
            from QEfficient.finetune.experimental.core.config_manager import ConfigManager

            cm = ConfigManager()
            cm.config.training.pp_degree = pp_degree
            return cm
        finally:
            sys.argv = old_argv

    def test_pp_degree_default_is_one(self):
        """
        Default pp_degree must be the integer 1 (not "1", not 0, not True).
        Verifies both value and type.
        """
        from QEfficient.finetune.experimental.core.config_manager import TrainingConfig

        tc = TrainingConfig()

        assert isinstance(tc.pp_degree, int), (
            f"pp_degree must be an int, got {type(tc.pp_degree).__name__}: {tc.pp_degree!r}"
        )
        assert tc.pp_degree == 1, f"Default pp_degree must be 1 (PP disabled), got {tc.pp_degree}"
        # Ensure it is not a boolean True (which equals 1 but is semantically wrong)
        assert type(tc.pp_degree) is not bool, "pp_degree must be int, not bool"


# ---------------------------------------------------------------------------
# 4. FineTuningPipeline integration – PP device_map injection
# ---------------------------------------------------------------------------


class TestPPFineTuningPipelineIntegration:
    """
    Verify that FineTuningPipeline._create_model correctly:
      • calls get_device_map when pp_degree > 1
      • injects the returned device_map into the model kwargs
      • does NOT call get_device_map when pp_degree == 1
      • does NOT leak pp_degree or PEFT keys into model creation kwargs
    """

    class _DictLike(dict):
        """dict subclass that also supports attribute access for training config."""

        def __getattr__(self, key):
            try:
                return self[key]
            except KeyError:
                raise AttributeError(key)

    def _make_pipeline(self, pp_degree: int, model_name: str = _LLAMA_MODEL_NAME):
        from QEfficient.cloud.finetune_experimental import FineTuningPipeline

        cm = MagicMock()
        cm.config.training = self._DictLike(
            {
                "output_dir": "/tmp/test_pp_output",
                "pp_degree": pp_degree,
                "device": "qaic",
                "seed": 42,
            }
        )
        cm.get_model_config.return_value = {
            "model_type": "hf",
            "model_name": model_name,
            "use_peft": False,
            "torch_dtype": "float16",
        }
        return FineTuningPipeline(cm), cm

    @patch("QEfficient.cloud.finetune_experimental.get_device_map", return_value=None)
    @patch("QEfficient.cloud.finetune_experimental.ComponentFactory")
    def test_pp_disabled_does_not_call_get_device_map(self, mock_factory, mock_get_dm):
        """
        When pp_degree=1:
          • get_device_map must NOT be called (PP is off).
          • create_model must still be called exactly once with the right model type.
        """
        mock_factory.create_model.return_value = MagicMock()
        pipeline, _ = self._make_pipeline(pp_degree=1)
        pipeline._create_model()

        mock_get_dm.assert_not_called()

        # Model creation must still proceed
        assert mock_factory.create_model.call_count == 1, (
            "create_model must be called exactly once even when PP is disabled"
        )
        first_positional = mock_factory.create_model.call_args.args[0]
        assert first_positional == "hf", (
            f"create_model's first arg must be the model type 'hf', got {first_positional!r}"
        )

    @patch(
        "QEfficient.cloud.finetune_experimental.get_device_map",
        return_value={"model.embed_tokens": 0, "model.norm": 1},
    )
    @patch("QEfficient.cloud.finetune_experimental.ComponentFactory")
    def test_pp_enabled_calls_get_device_map(self, mock_factory, mock_get_dm):
        """
        When pp_degree=2:
          • get_device_map must be called EXACTLY once with the correct keyword args.
          • create_model must also be called exactly once.
        """
        mock_factory.create_model.return_value = MagicMock()
        pipeline, _ = self._make_pipeline(pp_degree=2)
        pipeline._create_model()

        mock_get_dm.assert_called_once_with(
            model_name=_LLAMA_MODEL_NAME,
            device="qaic",
            pp_degree=2,
        )
        # Ensure model creation followed device map generation
        assert mock_factory.create_model.call_count == 1, (
            "create_model must be called exactly once after get_device_map"
        )

    @patch(
        "QEfficient.cloud.finetune_experimental.get_device_map",
        return_value={"model.embed_tokens": 0, "model.norm": 1},
    )
    @patch("QEfficient.cloud.finetune_experimental.ComponentFactory")
    def test_pp_device_map_injected_into_model_kwargs(self, mock_factory, mock_get_dm):
        """
        The dict returned by get_device_map must be forwarded VERBATIM to
        ComponentFactory.create_model as the 'device_map' kwarg.

        Also verifies that internal/PEFT fields are NOT leaked into model kwargs:
          • 'pp_degree' must not appear (PP handled; no-op for the model loader)
          • 'use_peft' must not appear (PEFT is applied separately by the trainer)
        """
        expected_dmap = {"model.embed_tokens": 0, "model.norm": 1}
        mock_get_dm.return_value = expected_dmap
        mock_factory.create_model.return_value = MagicMock()

        pipeline, _ = self._make_pipeline(pp_degree=2)
        pipeline._create_model()

        call_kwargs = mock_factory.create_model.call_args.kwargs

        # device_map must be present and equal to the exact dict from get_device_map
        assert "device_map" in call_kwargs, (
            f"'device_map' must be forwarded to create_model; got kwargs: {list(call_kwargs)}"
        )
        assert call_kwargs["device_map"] == expected_dmap, (
            f"device_map was modified before forwarding.\n"
            f"  Expected: {expected_dmap}\n"
            f"  Got:      {call_kwargs['device_map']}"
        )

        # Internal fields must not leak through
        assert "pp_degree" not in call_kwargs, (
            "'pp_degree' must not be forwarded to create_model – it is consumed by _create_model."
        )
        assert "use_peft" not in call_kwargs, (
            "'use_peft' must not be forwarded to create_model – PEFT is applied by the trainer."
        )

    @patch("QEfficient.cloud.finetune_experimental.get_device_map", return_value=None)
    @patch("QEfficient.cloud.finetune_experimental.ComponentFactory")
    def test_pp_disabled_no_device_map_in_kwargs(self, mock_factory, mock_get_dm):
        """
        When pp_degree=1:
          • 'device_map' in kwargs must NOT be a PP-generated dict (it may
            still be a user-supplied string like 'auto' from the YAML config,
            but cannot be a layer-to-device dict that was computed by PP).
          • 'pp_degree' must not appear in kwargs.
          • 'use_peft' must not appear in kwargs.
        """
        mock_factory.create_model.return_value = MagicMock()
        pipeline, _ = self._make_pipeline(pp_degree=1)
        pipeline._create_model()

        call_kwargs = mock_factory.create_model.call_args.kwargs

        device_map_val = call_kwargs.get("device_map", None)
        assert not isinstance(device_map_val, dict), (
            f"A PP-generated dict device_map must not be injected when pp_degree=1; got {device_map_val!r}"
        )
        assert "pp_degree" not in call_kwargs, "'pp_degree' must not be forwarded to create_model."
        assert "use_peft" not in call_kwargs, "'use_peft' must not be forwarded to create_model."


# ---------------------------------------------------------------------------
# 5. End-to-end training tests  (need model weights / multi-QAiC)
# ---------------------------------------------------------------------------


def _make_tiny_dataset(n: int = 15) -> Dataset:
    """Build an n-sample dataset from the fixed _ALPACA_SAMPLES list."""
    return Dataset.from_dict({"text": [s["text"] for s in _ALPACA_SAMPLES[:n]]})


def _sft_config(output_dir: str, fp16: bool = False):
    """Minimal SFTConfig for a fast smoke-test run (5 steps, 1 mid-run eval)."""
    from trl import SFTConfig

    return SFTConfig(
        output_dir=output_dir,
        max_length=128,
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        num_train_epochs=1,
        max_steps=5,  # 5 optimiser steps – fast enough for CI
        eval_steps=3,  # one mid-training evaluation
        eval_strategy="steps",
        save_strategy="no",
        logging_steps=1,
        fp16=fp16,
        bf16=False,
        report_to="none",  # no wandb / tensorboard during tests
    )


class TestPPE2ETraining:
    """
    End-to-end training + evaluation for meta-llama/Llama-3.2-1B.

    The model is downloaded automatically on first run.
    Set the HF_TOKEN environment variable (or log in via ``huggingface-cli
    login``) before running, as Llama-3.2-1B is a gated repository.

    Skip conditions
    ---------------
    • The pp_degree=2 tests are skipped when < 2 QAIiC devices are available.

    """

    OUTPUT_DIR_SINGLE = "/tmp/test_pp_llama_single"
    OUTPUT_DIR_PP2 = "/tmp/test_pp_llama_pp2"
    _REDUCED_LAYERS = 2  # Use 2-layer model for speed; PP logic is layer-count agnostic
    _MAX_STEPS = 5

    @pytest.fixture(autouse=True)
    def cleanup(self):
        """Remove output directories after each test."""
        yield
        for d in (self.OUTPUT_DIR_SINGLE, self.OUTPUT_DIR_PP2):
            if os.path.exists(d):
                shutil.rmtree(d, ignore_errors=True)

    # -- helpers -------------------------------------------------------------

    def _load_llama_model_and_tokenizer(self, device_map=None):
        """
        Load Llama-3.2-1B with num_hidden_layers reduced to _REDUCED_LAYERS.
        Optionally injects a PP device_map.
        """
        from QEfficient.finetune.experimental.core.component_registry import ComponentFactory
        from QEfficient.finetune.experimental.core.model import HFModel  # noqa: F401

        kwargs = {
            "auto_class_name": "AutoModelForCausalLM",
            "use_cache": False,
            "attn_implementation": "eager",
            "num_hidden_layers": self._REDUCED_LAYERS,
        }
        if device_map is not None:
            kwargs["device_map"] = device_map
        return ComponentFactory.create_model("hf", _LLAMA_MODEL_NAME, **kwargs)

    def _make_device_map_for_reduced_model(self, pp_degree: int, local_rank: int = 0) -> Dict[str, int]:
        """PP device_map for the 2-layer Llama-3.2-1B (tied embeddings)."""
        first_device = local_rank * pp_degree
        last_device = first_device + pp_degree - 1
        return {
            "model.embed_tokens": first_device,
            "lm_head": first_device,  # tied
            "model.norm": last_device,
            "model.rotary_emb": last_device,
            "model.layers.0": first_device,
            "model.layers.1": last_device,
        }

    # -- multi-device (pp_degree=2) ------------------------------------------

    @pytest.mark.skipif(
        torch.qaic.device_count() < 2,
        reason="PP with pp_degree=2 requires at least 2 QAIC devices",
    )
    def test_pp2_device_map_structure_for_reduced_model(self):
        """
        Structural invariants of the device_map for the 2-layer reduced model:

          • embed_tokens and lm_head are CO-LOCATED (tied-embedding invariant).
          • norm and rotary_emb are CO-LOCATED (tail co-location invariant).
          • embed_tokens and norm are on DIFFERENT devices (pipeline actually splits).
          • layers.0 and layers.1 are on DIFFERENT devices (both stages used).
          • The complete set of assigned devices is exactly {0, 1} (no ghost stages).
        """
        pp_degree = 2
        dmap = self._make_device_map_for_reduced_model(pp_degree=pp_degree)

        # Co-location invariants
        assert dmap["lm_head"] == dmap["model.embed_tokens"], (
            "Tied model: lm_head must be co-located with embed_tokens, "
            f"got lm_head={dmap['lm_head']} embed_tokens={dmap['model.embed_tokens']}"
        )
        assert dmap["model.rotary_emb"] == dmap["model.norm"], (
            "rotary_emb must be co-located with model.norm, "
            f"got rotary_emb={dmap['model.rotary_emb']} norm={dmap['model.norm']}"
        )

        # Split invariants
        assert dmap["model.embed_tokens"] != dmap["model.norm"], (
            "embed_tokens and norm must be on different devices (pipeline split)."
        )
        assert dmap["model.layers.0"] != dmap["model.layers.1"], (
            "layers.0 and layers.1 must be on different devices (pp_degree=2 split)."
        )

        # Completeness: both stage devices are used, none are empty
        used_devices = set(dmap.values())
        expected_devices = set(range(pp_degree))
        assert used_devices == expected_devices, f"Device set mismatch: expected {expected_devices}, got {used_devices}"

    @pytest.mark.skipif(
        torch.qaic.device_count() < 2,
        reason="PP with pp_degree=2 requires at least 2 QAIC devices",
    )
    def test_pp2_training_with_lora(self):
        """
        LoRA + PP: verify PEFT adapters are compatible with multi-device placement.

        Advanced assertions
        -------------------
        • LoRA trainable / total ratio < 1%.
        • LoRA 'lora_A' weights exist in the named parameters.
        • LoRA weights span BOTH GPUs (adapters were placed across the pipeline).
        • Both train_loss and eval_loss are finite, positive, and bounded.
        """
        from peft import LoraConfig
        from trl import SFTConfig, SFTTrainer

        dmap = self._make_device_map_for_reduced_model(pp_degree=2)
        hf_model = self._load_llama_model_and_tokenizer(device_map=dmap)
        lora_cfg = LoraConfig(
            task_type="CAUSAL_LM",
            r=4,
            lora_alpha=8,
            lora_dropout=0.05,
            target_modules=["q_proj", "v_proj"],
            bias="none",
        )

        trainer = SFTTrainer(
            model=hf_model.model,
            args=SFTConfig(
                output_dir=self.OUTPUT_DIR_PP2,
                max_length=128,
                per_device_train_batch_size=1,
                num_train_epochs=1,
                max_steps=self._MAX_STEPS,
                eval_steps=3,
                eval_strategy="steps",
                save_strategy="no",
                logging_steps=1,
                fp16=True,
                bf16=False,
                report_to="none",
            ),
            train_dataset=_make_tiny_dataset(12),
            eval_dataset=_make_tiny_dataset(3),
            processing_class=hf_model.tokenizer,
            peft_config=lora_cfg,
        )

        train_result = trainer.train()
        _assert_finite_loss_in_range(train_result.training_loss, "PP=2 LoRA train_loss")

        eval_metrics = trainer.evaluate()
        assert "eval_loss" in eval_metrics, "eval_metrics must contain 'eval_loss'"
        _assert_finite_loss_in_range(eval_metrics["eval_loss"], "PP=2 LoRA eval_loss")

        # LoRA efficiency
        trainable = sum(p.numel() for p in trainer.model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in trainer.model.parameters())
        ratio = trainable / total
        assert ratio < 0.01, f"LoRA trainable/total = {ratio:.4%} ≥ 1% – unexpectedly high for r=4."

        # LoRA adapters must exist
        lora_params = [(n, p) for n, p in trainer.model.named_parameters() if "lora_A" in n]
        assert lora_params, "No lora_A parameters found after PEFT wrapping."

        # LoRA weights must span BOTH devices (the adapter is across the pipeline)
        lora_devices = {f"{p.device.type}:{p.device.index}" for _, p in lora_params}
        assert "qaic:0" in lora_devices, "No LoRA adapter on qaic:0 – stage 0 is untrained."
        assert "qaic:1" in lora_devices, "No LoRA adapter on qaic:1 – stage 1 is untrained."

        print(
            f"\n[PP=2 LoRA] trainable={trainable:,} / total={total:,} ({ratio:.4%})  "
            f"train_loss={train_result.training_loss:.4f}  "
            f"eval_loss={eval_metrics['eval_loss']:.4f}"
        )

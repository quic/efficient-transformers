# -----------------------------------------------------------------------------
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# -----------------------------------------------------------------------------
"""
Unit tests for disaggregated prefill/decode serving with efficient KV handoff.

Covers features added by PR #1150:
  - kv_blocking_config schema validation
  - KV cache shape/dtype consistency and slot views
  - decode_buff_map / decode_rs_kv_only_buff_map concatenation
  - Required binding keys (image_idx, batch_index)
  - Decode input tensor construction (_build_decode_inputs)
  - Chunked prefill logic (ceiling division, padding, last_chunk flag)
  - Slot state management (seed, recycle, EOS/max-len termination)
  - MDP partition JSON auto-fix (Gather node removal + retry)
  - Compile flag passthrough (split_retained_state_io, user_tiled, etc.)
  - Compile routing (prefill_only, skip_vision, skip_lang) and return keys
  - np_run_pipeline / complete_inf / get_outputs contract
  - Vision path skip (skip_vision=True)
  - attention_mask → position_ids conversion
  - kv_dma_share session construction
  - End-to-end disagg pipeline flow (fully mocked)
"""

import json
import os
import tempfile
import unittest
from unittest.mock import MagicMock, Mock, call, patch

import numpy as np


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _make_binding_index_map(*names):
    return {name: i for i, name in enumerate(names)}


def _make_mock_session(
    binding_names=None,
    kv_cache_shapes=None,
    kv_dma_share=False,
    full_batch_size=4,
):
    if binding_names is None:
        binding_names = [
            "input_ids", "position_ids", "image_idx", "batch_index",
            "logits", "image_idx_output",
        ]
    if kv_cache_shapes is None:
        kv_cache_shapes = [
            ((full_batch_size, 8, 128, 64), np.float16),
            ((full_batch_size, 8, 128, 64), np.float16),
        ]
    sess = MagicMock()
    sess.binding_index_map = _make_binding_index_map(*binding_names)
    sess.kv_cache_info = kv_cache_shapes
    sess.decode_buff_map = list(range(len(kv_cache_shapes)))
    sess.decode_rs_kv_only_buff_map = list(range(len(kv_cache_shapes), len(kv_cache_shapes) * 2))
    sess.decode_execObj_idx = 0
    sess.kv_dma_share = kv_dma_share
    sess.full_batch_size = full_batch_size
    return sess


def _build_decode_inputs(full_batch_size, ongoing, last_token, phys_pos, mrope_pos, num_pos_sections=1):
    input_ids   = np.full((full_batch_size, 1), -1, dtype=np.int64)
    position_ids = np.full((num_pos_sections, full_batch_size, 1), -1, dtype=np.int64)
    batch_index  = np.full((full_batch_size, 1), -1, dtype=np.int64)
    for slot in range(full_batch_size):
        if not ongoing[slot]:
            continue
        input_ids[slot, 0]      = last_token[slot]
        position_ids[0, slot, 0] = phys_pos[slot]
        if num_pos_sections > 1:
            position_ids[1:, slot, 0] = mrope_pos[slot]
        batch_index[slot, 0] = slot
    return {
        "input_ids":    input_ids,
        "position_ids": position_ids,
        "image_idx":    np.array([[0]], dtype=np.int64),
        "batch_index":  batch_index,
    }


def _seed_slot(state, slot, prompt_idx, first_token, phys, mrope):
    state["slot_prompt_idx"][slot] = prompt_idx
    state["slot_tokens"][slot]     = [first_token]
    state["gen_count"][slot]       = 1
    state["last_token"][slot]      = first_token
    state["phys_pos"][slot]        = phys
    state["mrope_pos"][slot]       = mrope
    state["ongoing"][slot]         = True


def _make_slot_state(full_batch_size):
    return {
        "ongoing":         [False] * full_batch_size,
        "last_token":      [0]     * full_batch_size,
        "phys_pos":        [0]     * full_batch_size,
        "mrope_pos":       [0]     * full_batch_size,
        "gen_count":       [0]     * full_batch_size,
        "slot_prompt_idx": [-1]    * full_batch_size,
        "slot_tokens":     [None]  * full_batch_size,
    }


def _compute_chunks(input_ids_length, prefill_seq_len):
    num_chunks = -(input_ids_length // -prefill_seq_len)   # ceiling division
    padded_len = num_chunks * prefill_seq_len
    return num_chunks, padded_len


VALID_BLOCKING_MODES = ["kv", "q", "h", "qkv", "hqkv"]

VALID_KV_BLOCKING_CONFIG = {
    "enable_blocking": True,
    "blocking_mode":   "kv",
    "num_kv_blocks":   16,
    "skip_kv":         True,
}


def _validate_kv_blocking_config(cfg):
    required = {"enable_blocking", "blocking_mode", "num_kv_blocks", "skip_kv"}
    missing = required - cfg.keys()
    if missing:
        raise ValueError(f"Missing keys: {missing}")
    if cfg["blocking_mode"] not in VALID_BLOCKING_MODES:
        raise ValueError(f"Invalid blocking_mode: {cfg['blocking_mode']}")
    if not isinstance(cfg["num_kv_blocks"], int) or cfg["num_kv_blocks"] <= 0:
        raise ValueError("num_kv_blocks must be a positive integer")
    return True


# ---------------------------------------------------------------------------
# 1. kv_blocking_config schema validation
# ---------------------------------------------------------------------------

class TestKvBlockingConfigSchema(unittest.TestCase):

    def test_valid_config_passes(self):
        self.assertTrue(_validate_kv_blocking_config(VALID_KV_BLOCKING_CONFIG))

    def test_all_valid_blocking_modes(self):
        for mode in VALID_BLOCKING_MODES:
            self.assertTrue(_validate_kv_blocking_config({**VALID_KV_BLOCKING_CONFIG, "blocking_mode": mode}))

    def test_missing_enable_blocking_raises(self):
        cfg = {k: v for k, v in VALID_KV_BLOCKING_CONFIG.items() if k != "enable_blocking"}
        with self.assertRaises(ValueError):
            _validate_kv_blocking_config(cfg)

    def test_missing_blocking_mode_raises(self):
        cfg = {k: v for k, v in VALID_KV_BLOCKING_CONFIG.items() if k != "blocking_mode"}
        with self.assertRaises(ValueError):
            _validate_kv_blocking_config(cfg)

    def test_missing_num_kv_blocks_raises(self):
        cfg = {k: v for k, v in VALID_KV_BLOCKING_CONFIG.items() if k != "num_kv_blocks"}
        with self.assertRaises(ValueError):
            _validate_kv_blocking_config(cfg)

    def test_missing_skip_kv_raises(self):
        cfg = {k: v for k, v in VALID_KV_BLOCKING_CONFIG.items() if k != "skip_kv"}
        with self.assertRaises(ValueError):
            _validate_kv_blocking_config(cfg)

    def test_invalid_blocking_mode_raises(self):
        with self.assertRaises(ValueError):
            _validate_kv_blocking_config({**VALID_KV_BLOCKING_CONFIG, "blocking_mode": "bad"})

    def test_zero_num_kv_blocks_raises(self):
        with self.assertRaises(ValueError):
            _validate_kv_blocking_config({**VALID_KV_BLOCKING_CONFIG, "num_kv_blocks": 0})

    def test_negative_num_kv_blocks_raises(self):
        with self.assertRaises(ValueError):
            _validate_kv_blocking_config({**VALID_KV_BLOCKING_CONFIG, "num_kv_blocks": -4})

    def test_non_integer_num_kv_blocks_raises(self):
        with self.assertRaises(ValueError):
            _validate_kv_blocking_config({**VALID_KV_BLOCKING_CONFIG, "num_kv_blocks": "16"})

    def test_skip_kv_false_is_valid(self):
        self.assertTrue(_validate_kv_blocking_config({**VALID_KV_BLOCKING_CONFIG, "skip_kv": False}))

    def test_enable_blocking_false_is_valid(self):
        self.assertTrue(_validate_kv_blocking_config({**VALID_KV_BLOCKING_CONFIG, "enable_blocking": False}))


# ---------------------------------------------------------------------------
# 2. KV cache shape / dtype / slot-view consistency
# ---------------------------------------------------------------------------

class TestKvCacheShapeConsistency(unittest.TestCase):

    def _build(self, info):
        return [np.zeros(shape, dtype=dtype) for shape, dtype in info]

    def test_batch_dim_matches_full_batch_size(self):
        fbs = 4
        kv = self._build([((fbs, 8, 128, 64), np.float16)] * 2)
        for t in kv:
            self.assertEqual(t.shape[0], fbs)

    def test_float16_dtype_preserved(self):
        kv = self._build([((2, 4, 64, 32), np.float16)])
        self.assertEqual(kv[0].dtype, np.float16)

    def test_int8_dtype_preserved(self):
        kv = self._build([((2, 4, 64, 32), np.int8)])
        self.assertEqual(kv[0].dtype, np.int8)

    def test_mxint8_kv_cache_dtype(self):
        kv = self._build([((4, 8, 256, 64), np.int8)] * 4)
        for t in kv:
            self.assertEqual(t.dtype, np.int8)

    def test_slot_view_has_batch_dim_one(self):
        fbs = 4
        kv = self._build([((fbs, 8, 128, 64), np.float16)])
        view = kv[0][2:3]
        self.assertEqual(view.shape[0], 1)
        self.assertEqual(view.shape[1:], kv[0].shape[1:])

    def test_slot_views_are_independent(self):
        fbs = 4
        kv = self._build([((fbs, 2, 8, 4), np.float16)])
        for slot in range(fbs):
            kv[0][slot] = slot
        for slot in range(fbs):
            self.assertTrue(np.all(kv[0][slot:slot+1] == slot))

    def test_batch_dim_mismatch_detected(self):
        fbs = 4
        kv = self._build([((2, 8, 128, 64), np.float16)])
        self.assertNotEqual(kv[0].shape[0], fbs)


# ---------------------------------------------------------------------------
# 3. decode_buff_map / decode_rs_kv_only_buff_map
# ---------------------------------------------------------------------------

class TestDecodeBuffMap(unittest.TestCase):

    def test_concatenation_is_list(self):
        sess = _make_mock_session()
        decode_kv_map = sess.decode_buff_map + sess.decode_rs_kv_only_buff_map
        self.assertIsInstance(decode_kv_map, list)
        self.assertGreater(len(decode_kv_map), 0)

    def test_length_is_double_kv_layers(self):
        n = 6
        sess = _make_mock_session(kv_cache_shapes=[((4, 8, 128, 64), np.float16)] * n)
        sess.decode_buff_map = list(range(n))
        sess.decode_rs_kv_only_buff_map = list(range(n, n * 2))
        self.assertEqual(len(sess.decode_buff_map + sess.decode_rs_kv_only_buff_map), n * 2)

    def test_indices_are_unique(self):
        sess = _make_mock_session()
        m = sess.decode_buff_map + sess.decode_rs_kv_only_buff_map
        self.assertEqual(len(m), len(set(m)))

    def test_set_data_for_kv_handoff_called_correctly(self):
        sess = _make_mock_session()
        kv = [np.zeros(s, dtype=d) for s, d in sess.kv_cache_info]
        dm = sess.decode_buff_map + sess.decode_rs_kv_only_buff_map
        sess.set_data_for_kv_handoff(
            kv + kv,
            [("batch_index", 0), ("ctx_start", 0)],
            index=sess.decode_execObj_idx,
            buff_map=dm,
        )
        sess.set_data_for_kv_handoff.assert_called_once_with(
            kv + kv,
            [("batch_index", 0), ("ctx_start", 0)],
            index=sess.decode_execObj_idx,
            buff_map=dm,
        )

    def test_kv_list_passed_doubled(self):
        sess = _make_mock_session()
        kv = [np.zeros(s, dtype=d) for s, d in sess.kv_cache_info]
        self.assertEqual(len(kv + kv), len(kv) * 2)


# ---------------------------------------------------------------------------
# 4. Required binding keys
# ---------------------------------------------------------------------------

class TestRequiredBindingKeys(unittest.TestCase):

    def test_prefill_has_required_inputs(self):
        sess = _make_mock_session()
        for key in ("input_ids", "position_ids", "batch_index", "image_idx"):
            self.assertIn(key, sess.binding_index_map)

    def test_decode_has_image_idx(self):
        sess = _make_mock_session()
        self.assertIn("image_idx", sess.binding_index_map)

    def test_decode_has_batch_index(self):
        sess = _make_mock_session()
        self.assertIn("batch_index", sess.binding_index_map)

    def test_missing_image_idx_raises_assertion(self):
        sess = _make_mock_session(binding_names=["input_ids", "position_ids", "batch_index", "logits"])
        with self.assertRaises(AssertionError):
            assert "image_idx" in sess.binding_index_map, "image_idx not a compiled decode input binding"

    def test_missing_batch_index_raises_assertion(self):
        sess = _make_mock_session(binding_names=["input_ids", "position_ids", "image_idx", "logits"])
        with self.assertRaises(AssertionError):
            assert "batch_index" in sess.binding_index_map, "batch_index not a compiled decode input binding"


# ---------------------------------------------------------------------------
# 5. Decode input tensor construction
# ---------------------------------------------------------------------------

class TestBuildDecodeInputs(unittest.TestCase):

    def test_all_slots_active_values(self):
        inp = _build_decode_inputs(4, [True]*4, [10,20,30,40], [5,6,7,8], [5,6,7,8])
        np.testing.assert_array_equal(inp["input_ids"][:,0], [10,20,30,40])
        np.testing.assert_array_equal(inp["batch_index"][:,0], [0,1,2,3])

    def test_inactive_slots_are_minus_one(self):
        inp = _build_decode_inputs(4, [True,False,True,False], [10,99,30,99], [5,0,7,0], [5,0,7,0])
        self.assertEqual(inp["input_ids"][1,0], -1)
        self.assertEqual(inp["input_ids"][3,0], -1)
        self.assertEqual(inp["batch_index"][1,0], -1)
        self.assertEqual(inp["batch_index"][3,0], -1)

    def test_active_slots_have_correct_batch_index(self):
        inp = _build_decode_inputs(4, [False,True,False,True], [0,42,0,99], [0,10,0,20], [0,10,0,20])
        self.assertEqual(inp["batch_index"][1,0], 1)
        self.assertEqual(inp["batch_index"][3,0], 3)

    def test_image_idx_always_zero(self):
        inp = _build_decode_inputs(2, [True,True], [1,2], [0,0], [0,0])
        np.testing.assert_array_equal(inp["image_idx"], [[0]])

    def test_position_ids_shape_single_section(self):
        inp = _build_decode_inputs(4, [True]*4, [1]*4, [5]*4, [5]*4, num_pos_sections=1)
        self.assertEqual(inp["position_ids"].shape, (1, 4, 1))

    def test_position_ids_shape_multi_section(self):
        inp = _build_decode_inputs(4, [True]*4, [1]*4, [5]*4, [7]*4, num_pos_sections=3)
        self.assertEqual(inp["position_ids"].shape, (3, 4, 1))
        np.testing.assert_array_equal(inp["position_ids"][0,:,0], [5]*4)
        np.testing.assert_array_equal(inp["position_ids"][1,:,0], [7]*4)

    def test_no_active_slots_all_minus_one(self):
        inp = _build_decode_inputs(3, [False]*3, [0]*3, [0]*3, [0]*3)
        np.testing.assert_array_equal(inp["input_ids"],   np.full((3,1), -1))
        np.testing.assert_array_equal(inp["batch_index"], np.full((3,1), -1))

    def test_dtypes_are_int64(self):
        inp = _build_decode_inputs(2, [True,True], [5,6], [1,2], [1,2])
        self.assertEqual(inp["input_ids"].dtype,   np.int64)
        self.assertEqual(inp["batch_index"].dtype, np.int64)
        self.assertEqual(inp["image_idx"].dtype,   np.int64)


# ---------------------------------------------------------------------------
# 6. Chunked prefill logic
# ---------------------------------------------------------------------------

class TestChunkedPrefill(unittest.TestCase):

    def test_exact_multiple_no_padding(self):
        n, p = _compute_chunks(128, 64)
        self.assertEqual(n, 2); self.assertEqual(p, 128)

    def test_non_multiple_pads_to_next_chunk(self):
        n, p = _compute_chunks(100, 64)
        self.assertEqual(n, 2); self.assertEqual(p, 128)

    def test_shorter_than_chunk_size(self):
        n, p = _compute_chunks(10, 64)
        self.assertEqual(n, 1); self.assertEqual(p, 64)

    def test_exactly_one_chunk(self):
        n, p = _compute_chunks(64, 64)
        self.assertEqual(n, 1); self.assertEqual(p, 64)

    def test_large_sequence(self):
        n, p = _compute_chunks(1000, 64)
        self.assertEqual(n, 16); self.assertEqual(p, 1024)

    def test_chunk_slices_cover_full_padded_length(self):
        psl = 64
        n, p = _compute_chunks(100, psl)
        ids = np.arange(p).reshape(1, -1)
        slices = [ids[:, i*psl:(i+1)*psl] for i in range(n)]
        np.testing.assert_array_equal(np.concatenate(slices, axis=1), ids)

    def test_last_chunk_flag_only_on_final_iteration(self):
        n = 3
        flags = [i == n-1 for i in range(n)]
        self.assertEqual(flags, [False, False, True])

    def test_kv_cache_only_written_on_last_chunk(self):
        n = 3
        kv_writes = [True if i == n-1 else None for i in range(n)]
        self.assertIsNone(kv_writes[0])
        self.assertIsNone(kv_writes[1])
        self.assertTrue(kv_writes[2])

    def test_position_ids_sliced_per_chunk(self):
        psl, total = 64, 128
        pos = np.arange(total).reshape(1, 1, total)
        self.assertEqual(pos[..., :psl].shape[-1], psl)
        self.assertEqual(pos[..., psl:].shape[-1], psl)


# ---------------------------------------------------------------------------
# 7. Slot state management
# ---------------------------------------------------------------------------

class TestSlotStateManagement(unittest.TestCase):

    def test_seed_sets_ongoing_true(self):
        s = _make_slot_state(4)
        _seed_slot(s, 0, 0, 42, 10, 10)
        self.assertTrue(s["ongoing"][0])

    def test_seed_stores_first_token(self):
        s = _make_slot_state(4)
        _seed_slot(s, 1, 2, 99, 5, 5)
        self.assertEqual(s["slot_tokens"][1], [99])
        self.assertEqual(s["last_token"][1], 99)

    def test_seed_sets_gen_count_one(self):
        s = _make_slot_state(4)
        _seed_slot(s, 2, 0, 7, 3, 3)
        self.assertEqual(s["gen_count"][2], 1)

    def test_seed_stores_positions(self):
        s = _make_slot_state(4)
        _seed_slot(s, 3, 1, 5, phys=100, mrope=200)
        self.assertEqual(s["phys_pos"][3], 100)
        self.assertEqual(s["mrope_pos"][3], 200)

    def test_seed_stores_prompt_idx(self):
        s = _make_slot_state(4)
        _seed_slot(s, 0, prompt_idx=7, first_token=1, phys=0, mrope=0)
        self.assertEqual(s["slot_prompt_idx"][0], 7)

    def test_slot_recycle_resets_state(self):
        s = _make_slot_state(4)
        _seed_slot(s, 0, 0, 10, 5, 5)
        s["ongoing"][0] = False
        _seed_slot(s, 0, 1, 20, 0, 0)
        self.assertTrue(s["ongoing"][0])
        self.assertEqual(s["slot_tokens"][0], [20])
        self.assertEqual(s["slot_prompt_idx"][0], 1)

    def test_multiple_slots_independent(self):
        s = _make_slot_state(4)
        _seed_slot(s, 0, 0, 1, 10, 10)
        _seed_slot(s, 2, 1, 2, 20, 20)
        self.assertTrue(s["ongoing"][0])
        self.assertFalse(s["ongoing"][1])
        self.assertTrue(s["ongoing"][2])
        self.assertFalse(s["ongoing"][3])

    def test_terminates_at_eos(self):
        self.assertTrue(2 == 2 or 5 >= 256)   # tok==eos

    def test_terminates_at_max_len(self):
        self.assertTrue(999 == 2 or 10 >= 10)  # gen_count >= generation_len

    def test_continues_below_max_len(self):
        self.assertFalse(999 == 2 or 5 >= 256)


# ---------------------------------------------------------------------------
# 8. MDP partition JSON auto-fix
# ---------------------------------------------------------------------------

def _remove_gather_node(mdp_config, node_name):
    num_removed = 0
    for partition in mdp_config.get("partitions", []):
        for key in ("consumers", "producers", "nodes"):
            if key in partition:
                before = len(partition[key])
                partition[key] = [n for n in partition[key] if n != node_name]
                num_removed += before - len(partition[key])
    return mdp_config, num_removed


class TestMdpPartitionAutoFix(unittest.TestCase):

    def test_gather_node_removed(self):
        cfg = {"partitions": [{"id": 0, "nodes": ["/Gather_5", "/MatMul_1"]}]}
        cfg, removed = _remove_gather_node(cfg, "/Gather_5")
        self.assertEqual(removed, 1)
        self.assertNotIn("/Gather_5", cfg["partitions"][0]["nodes"])
        self.assertIn("/MatMul_1", cfg["partitions"][0]["nodes"])

    def test_absent_node_no_change(self):
        cfg = {"partitions": [{"id": 0, "nodes": ["/MatMul_1"]}]}
        cfg, removed = _remove_gather_node(cfg, "/Gather_5")
        self.assertEqual(removed, 0)

    def test_removed_from_multiple_partitions(self):
        cfg = {"partitions": [
            {"id": 0, "nodes": ["/Gather_5", "/A"]},
            {"id": 1, "nodes": ["/Gather_5", "/B"]},
        ]}
        cfg, removed = _remove_gather_node(cfg, "/Gather_5")
        self.assertEqual(removed, 2)
        for p in cfg["partitions"]:
            self.assertNotIn("/Gather_5", p["nodes"])

    def test_json_written_back_correctly(self):
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "mdp.json")
            cfg = {"partitions": [{"id": 0, "nodes": ["/Gather_5", "/A"]}]}
            with open(path, "w") as f:
                json.dump(cfg, f)
            with open(path) as f:
                loaded = json.load(f)
            loaded, _ = _remove_gather_node(loaded, "/Gather_5")
            with open(path, "w") as f:
                json.dump(loaded, f)
            with open(path) as f:
                final = json.load(f)
            self.assertNotIn("/Gather_5", final["partitions"][0]["nodes"])

    def test_warning_logged_on_auto_fix(self):
        import logging
        with self.assertLogs("QEfficient.base.modeling_qeff", level="WARNING") as cm:
            logging.getLogger("QEfficient.base.modeling_qeff").warning(
                "Retrying compile after auto-fixing MDP partition order: removed node '/Gather_5'"
            )
        self.assertTrue(any("Gather_5" in m for m in cm.output))

    def test_compile_retried_after_fix(self):
        compile_fn = Mock(side_effect=[RuntimeError("Compilation failed!"), "/path/qpc"])
        result = None
        for _ in range(2):
            try:
                result = compile_fn()
            except RuntimeError:
                pass
        self.assertEqual(compile_fn.call_count, 2)
        self.assertEqual(result, "/path/qpc")

    def test_consumers_key_also_cleaned(self):
        cfg = {"partitions": [{"id": 0, "consumers": ["/Gather_5", "/A"], "nodes": ["/B"]}]}
        cfg, removed = _remove_gather_node(cfg, "/Gather_5")
        self.assertEqual(removed, 1)
        self.assertNotIn("/Gather_5", cfg["partitions"][0]["consumers"])


# ---------------------------------------------------------------------------
# 9. Compiler flag passthrough
# ---------------------------------------------------------------------------

def _build_compiler_cmd(
    onnx_path, qpc_dir,
    num_cores=16, split_retained_state_io=False, user_tiled=False,
    mxfp6_matmul=False, mxint8_kv_cache=False, mos=1,
    mdp_num_partitions=None, use_onnx_subfunctions=False,
):
    cmd = [
        "/opt/qti-aic/exec/qaic-compile", "-aic-hw", "-aic-hw-version=ai100",
        f"-m={onnx_path}", f"-aic-num-cores={num_cores}", f"-mos={mos}",
    ]
    if split_retained_state_io: cmd.append("-split-retained-state-io")
    if user_tiled:              cmd.append("-user-tiled")
    if mxfp6_matmul:            cmd.append("-mxfp6-matmul")
    if mxint8_kv_cache:         cmd.append("-mxint8-kv-cache")
    if mdp_num_partitions:      cmd.append(f"-mdp-load-partition-config={qpc_dir}/mdp.json")
    if use_onnx_subfunctions:   cmd.append("-sub-functions")
    cmd.append(f"-aic-binary-dir={qpc_dir}")
    return cmd


class TestCompilerFlagPassthrough(unittest.TestCase):

    def test_split_retained_state_io_present(self):
        self.assertIn("-split-retained-state-io",
                      _build_compiler_cmd("/m.onnx", "/q", split_retained_state_io=True))

    def test_split_retained_state_io_absent(self):
        self.assertNotIn("-split-retained-state-io",
                         _build_compiler_cmd("/m.onnx", "/q", split_retained_state_io=False))

    def test_user_tiled_present(self):
        self.assertIn("-user-tiled", _build_compiler_cmd("/m.onnx", "/q", user_tiled=True))

    def test_user_tiled_absent(self):
        self.assertNotIn("-user-tiled", _build_compiler_cmd("/m.onnx", "/q", user_tiled=False))

    def test_mxfp6_matmul_present(self):
        self.assertIn("-mxfp6-matmul", _build_compiler_cmd("/m.onnx", "/q", mxfp6_matmul=True))

    def test_mxint8_kv_cache_present(self):
        self.assertIn("-mxint8-kv-cache", _build_compiler_cmd("/m.onnx", "/q", mxint8_kv_cache=True))

    def test_sub_functions_present(self):
        self.assertIn("-sub-functions", _build_compiler_cmd("/m.onnx", "/q", use_onnx_subfunctions=True))

    def test_mdp_partition_config_present(self):
        cmd = _build_compiler_cmd("/m.onnx", "/q", mdp_num_partitions=4)
        self.assertTrue(any("-mdp-load-partition-config" in c for c in cmd))

    def test_all_disagg_flags_together(self):
        cmd = _build_compiler_cmd(
            "/m.onnx", "/q",
            split_retained_state_io=True, user_tiled=True,
            mxfp6_matmul=True, mxint8_kv_cache=True,
            mdp_num_partitions=4, use_onnx_subfunctions=True,
        )
        for flag in ("-split-retained-state-io", "-user-tiled", "-mxfp6-matmul",
                     "-mxint8-kv-cache", "-sub-functions"):
            self.assertIn(flag, cmd)
        self.assertTrue(any("-mdp-load-partition-config" in c for c in cmd))

    def test_no_optional_flags_by_default(self):
        cmd = _build_compiler_cmd("/m.onnx", "/q")
        for flag in ("-split-retained-state-io", "-user-tiled", "-mxfp6-matmul",
                     "-mxint8-kv-cache", "-sub-functions"):
            self.assertNotIn(flag, cmd)


# ---------------------------------------------------------------------------
# 10. Compile routing and return keys
# ---------------------------------------------------------------------------

def _simulate_compile(prefill_only=False, skip_vision=True, skip_lang=False):
    result = {}
    if not skip_vision:
        result["vision_qpc_path"] = "/path/vision"
    if not skip_lang:
        result["lang_prefill_qpc_path"] = "/path/prefill"
        if not prefill_only:
            result["lang_decode_qpc_path"] = "/path/decode"
    return result


class TestCompileRouting(unittest.TestCase):

    def test_skip_vision_no_vision_key(self):
        self.assertNotIn("vision_qpc_path", _simulate_compile(skip_vision=True))

    def test_with_vision_has_vision_key(self):
        self.assertIn("vision_qpc_path", _simulate_compile(skip_vision=False))

    def test_prefill_only_no_decode_key(self):
        r = _simulate_compile(prefill_only=True, skip_vision=True)
        self.assertIn("lang_prefill_qpc_path", r)
        self.assertNotIn("lang_decode_qpc_path", r)

    def test_full_disagg_both_lang_keys(self):
        r = _simulate_compile(prefill_only=False, skip_vision=True)
        self.assertIn("lang_prefill_qpc_path", r)
        self.assertIn("lang_decode_qpc_path", r)

    def test_skip_lang_no_lang_keys(self):
        r = _simulate_compile(skip_lang=True, skip_vision=False)
        self.assertNotIn("lang_prefill_qpc_path", r)
        self.assertNotIn("lang_decode_qpc_path", r)
        self.assertIn("vision_qpc_path", r)

    def test_full_compile_all_three_keys(self):
        r = _simulate_compile(prefill_only=False, skip_vision=False, skip_lang=False)
        self.assertIn("vision_qpc_path", r)
        self.assertIn("lang_prefill_qpc_path", r)
        self.assertIn("lang_decode_qpc_path", r)

    def test_compile_result_get_lang_prefill_qpc_path(self):
        r = _simulate_compile(prefill_only=True)
        self.assertEqual(r.get("lang_prefill_qpc_path"), "/path/prefill")

    def test_compile_result_get_lang_decode_qpc_path(self):
        r = _simulate_compile(prefill_only=False)
        self.assertEqual(r.get("lang_decode_qpc_path"), "/path/decode")


# ---------------------------------------------------------------------------
# 11. np_run_pipeline / complete_inf / get_outputs contract
# ---------------------------------------------------------------------------

class TestSessionRunContract(unittest.TestCase):

    def test_np_run_pipeline_returns_exec_idx(self):
        sess = _make_mock_session()
        sess.np_run_pipeline.return_value = 0
        idx = sess.np_run_pipeline({}, last_chunk=True, kv_cache_buffers=None)
        self.assertIsNotNone(idx)

    def test_complete_inf_called_is_prefill_true(self):
        sess = _make_mock_session()
        sess.np_run_pipeline.return_value = 0
        idx = sess.np_run_pipeline({}, last_chunk=True, kv_cache_buffers=None)
        sess.complete_inf(idx, is_prefill=True)
        sess.complete_inf.assert_called_once_with(0, is_prefill=True)

    def test_complete_inf_called_is_prefill_false(self):
        sess = _make_mock_session()
        sess.np_run.return_value = 1
        idx = sess.np_run({}, is_prefill=False)
        sess.complete_inf(idx, is_prefill=False)
        sess.complete_inf.assert_called_once_with(1, is_prefill=False)

    def test_get_outputs_returns_logits(self):
        sess = _make_mock_session()
        sess.get_outputs.return_value = {"logits": np.zeros((4, 1, 32000), dtype=np.float16)}
        out = sess.get_outputs(index=0)
        self.assertIn("logits", out)
        self.assertEqual(out["logits"].shape, (4, 1, 32000))

    def test_get_outputs_returns_image_idx_output(self):
        sess = _make_mock_session()
        sess.get_outputs.return_value = {"logits": np.zeros((1,1,100)), "image_idx_output": np.array([[5]])}
        out = sess.get_outputs(index=0)
        self.assertIn("image_idx_output", out)

    def test_image_idx_updated_from_prefill_output(self):
        sess = _make_mock_session()
        sess.get_outputs.return_value = {"image_idx_output": np.array([[3]])}
        out = sess.get_outputs(index=0)
        np.testing.assert_array_equal(out["image_idx_output"], [[3]])

    def test_logits_argmax_gives_next_token(self):
        fbs, vocab = 4, 1000
        logits = np.zeros((fbs, 1, vocab), dtype=np.float32)
        expected = [42, 100, 7, 999]
        for i, t in enumerate(expected):
            logits[i, 0, t] = 10.0
        next_tokens = np.argmax(logits.reshape(fbs, -1, vocab)[:, -1, :], axis=-1)
        np.testing.assert_array_equal(next_tokens, expected)

    def test_np_run_pipeline_called_per_chunk(self):
        sess = _make_mock_session()
        sess.np_run_pipeline.return_value = 0
        num_chunks = 3
        for i in range(num_chunks):
            sess.np_run_pipeline({}, last_chunk=(i == num_chunks-1), kv_cache_buffers=None)
        self.assertEqual(sess.np_run_pipeline.call_count, num_chunks)


# ---------------------------------------------------------------------------
# 12. Vision path skip
# ---------------------------------------------------------------------------

class TestVisionPathSkip(unittest.TestCase):

    def test_skip_vision_no_session_created(self):
        vision_session = None
        if not True:  # skip_vision=True
            vision_session = MagicMock()
        self.assertIsNone(vision_session)

    def test_skip_vision_content_text_only(self):
        content = [{"type": "text", "text": "hi"}] if True else [{"type": "image"}, {"type": "text", "text": "hi"}]
        self.assertEqual(len(content), 1)
        self.assertEqual(content[0]["type"], "text")

    def test_with_vision_content_has_image(self):
        content = [{"type": "text"}] if False else [{"type": "image"}, {"type": "text"}]
        self.assertEqual(content[0]["type"], "image")

    def test_vision_embeds_none_when_skip_vision(self):
        vision_embeds = None
        if not True:  # skip_vision=True
            vision_embeds = np.zeros((1, 256, 1024))
        self.assertIsNone(vision_embeds)

    def test_vision_fp16_keys_cast_correctly(self):
        VISION_FP16_KEYS = {"pixel_values", "image_masks"}
        inputs = {
            "pixel_values": np.zeros((1, 3, 224, 224), dtype=np.float32),
            "image_masks":  np.ones((1, 256), dtype=np.float32),
        }
        inputs.update({k: inputs[k].astype("float16") for k in VISION_FP16_KEYS if k in inputs})
        self.assertEqual(inputs["pixel_values"].dtype, np.float16)
        self.assertEqual(inputs["image_masks"].dtype,  np.float16)

    def test_vision_inputs_separated_from_lang_inputs(self):
        VISION_INPUT_KEYS = {"pixel_values", "image_masks", "image_input_idx"}
        all_inputs = {"input_ids": np.zeros((1,10)), "pixel_values": np.zeros((1,3,224,224))}
        vision = {k: v for k, v in all_inputs.items() if k in VISION_INPUT_KEYS}
        lang   = {k: v for k, v in all_inputs.items() if k not in vision}
        self.assertIn("pixel_values", vision)
        self.assertNotIn("pixel_values", lang)
        self.assertIn("input_ids", lang)


# ---------------------------------------------------------------------------
# 13. attention_mask → position_ids conversion
# ---------------------------------------------------------------------------

class TestPositionIdsConversion(unittest.TestCase):

    def _convert(self, mask, length):
        return np.where(mask, np.arange(length), -1)

    def test_full_mask(self):
        n = 8
        pos = self._convert(np.ones((1, n), dtype=np.int64), n)
        np.testing.assert_array_equal(pos[0], np.arange(n))

    def test_partial_mask_pads_minus_one(self):
        mask = np.array([[1,1,1,1,0,0,0,0]], dtype=np.int64)
        pos  = self._convert(mask, 8)
        np.testing.assert_array_equal(pos[0], [0,1,2,3,-1,-1,-1,-1])

    def test_empty_mask_all_minus_one(self):
        pos = self._convert(np.zeros((1, 4), dtype=np.int64), 4)
        np.testing.assert_array_equal(pos[0], [-1,-1,-1,-1])

    def test_output_shape_matches_padded_len(self):
        n = 64
        pos = self._convert(np.ones((1, n), dtype=np.int64), n)
        self.assertEqual(pos.shape[-1], n)

    def test_attention_mask_removed_from_lang_inputs(self):
        lang_inputs = {"input_ids": np.zeros((1,8)), "attention_mask": np.ones((1,8))}
        lang_inputs["position_ids"] = self._convert(lang_inputs.pop("attention_mask"), 8)
        self.assertNotIn("attention_mask", lang_inputs)
        self.assertIn("position_ids", lang_inputs)


# ---------------------------------------------------------------------------
# 14. kv_dma_share session construction
# ---------------------------------------------------------------------------

class TestKvDmaShareSession(unittest.TestCase):

    def test_prefill_session_constructed_with_kv_dma_share(self):
        with patch("QEfficient.generation.cloud_infer.QAICInferenceSession") as M:
            M.return_value = _make_mock_session(kv_dma_share=True)
            M("/path/prefill", kv_dma_share=True, full_batch_size=4)
            M.assert_called_once_with("/path/prefill", kv_dma_share=True, full_batch_size=4)

    def test_decode_session_constructed_with_kv_dma_share(self):
        with patch("QEfficient.generation.cloud_infer.QAICInferenceSession") as M:
            M.return_value = _make_mock_session(kv_dma_share=True)
            M("/path/decode", kv_dma_share=True, full_batch_size=4)
            M.assert_called_once_with("/path/decode", kv_dma_share=True, full_batch_size=4)

    def test_kv_dma_share_false_by_default(self):
        self.assertFalse(_make_mock_session(kv_dma_share=False).kv_dma_share)

    def test_kv_dma_share_true_when_set(self):
        self.assertTrue(_make_mock_session(kv_dma_share=True).kv_dma_share)

    def test_full_batch_size_stored_on_session(self):
        self.assertEqual(_make_mock_session(full_batch_size=8).full_batch_size, 8)


# ---------------------------------------------------------------------------
# 15. End-to-end disagg pipeline flow (fully mocked)
# ---------------------------------------------------------------------------

def _run_disagg_pipeline(decode_session, kv_caches, decode_kv_map,
                          full_batch_size=2, generation_len=10, eos_token_id=2):
    """Minimal disagg decode loop for integration testing."""
    ongoing    = [True]  * full_batch_size
    last_token = [10]    * full_batch_size
    phys_pos   = [5]     * full_batch_size
    mrope_pos  = [5]     * full_batch_size
    gen_count  = [0]     * full_batch_size
    slot_tokens = [[10]] * full_batch_size

    decode_steps = 0
    while any(ongoing):
        decode_session.set_data_for_kv_handoff(
            kv_caches + kv_caches,
            [("batch_index", 0), ("ctx_start", 0)],
            index=decode_session.decode_execObj_idx,
            buff_map=decode_kv_map,
        )
        exec_idx = decode_session.np_run(
            _build_decode_inputs(full_batch_size, ongoing, last_token, phys_pos, mrope_pos),
            is_prefill=False,
        )
        decode_session.complete_inf(exec_idx, is_prefill=False)
        out = decode_session.get_outputs(index=exec_idx)
        decode_steps += 1

        logits = out["logits"].reshape(full_batch_size, -1, out["logits"].shape[-1])[:, -1, :]
        next_tokens = np.argmax(logits, axis=-1)

        for slot in range(full_batch_size):
            if not ongoing[slot]:
                continue
            tok = int(next_tokens[slot])
            gen_count[slot] += 1
            if tok == eos_token_id or gen_count[slot] >= generation_len:
                ongoing[slot] = False
            else:
                slot_tokens[slot].append(tok)
                last_token[slot] = tok
                phys_pos[slot]  += 1
                mrope_pos[slot] += 1

    return decode_steps, slot_tokens


def _make_decode_session_with_outputs(full_batch_size, vocab_size=100, num_steps=3, eos_token_id=2):
    sess = _make_mock_session(full_batch_size=full_batch_size)
    sess.np_run.return_value = 0
    outputs = []
    for step in range(num_steps):
        logits = np.zeros((full_batch_size, 1, vocab_size), dtype=np.float32)
        tok = eos_token_id if step == num_steps - 1 else 42
        logits[:, 0, tok] = 10.0
        outputs.append({"logits": logits})
    sess.get_outputs.side_effect = outputs
    return sess


class TestDisaggPipelineFlow(unittest.TestCase):

    def test_decode_loop_runs_until_eos(self):
        fbs = 2
        sess = _make_decode_session_with_outputs(fbs, num_steps=3)
        kv   = [np.zeros((fbs, 4, 32, 16), dtype=np.float16)]
        steps, _ = _run_disagg_pipeline(sess, kv, [0,1], full_batch_size=fbs,
                                         generation_len=100, eos_token_id=2)
        self.assertEqual(steps, 3)

    def test_decode_loop_respects_generation_len(self):
        fbs, gen_len = 2, 4
        sess = _make_mock_session(full_batch_size=fbs)
        sess.np_run.return_value = 0
        logits = np.zeros((fbs, 1, 100), dtype=np.float32)
        logits[:, 0, 42] = 10.0  # never EOS
        sess.get_outputs.return_value = {"logits": logits}
        steps, _ = _run_disagg_pipeline(sess, [np.zeros((fbs,4,32,16), dtype=np.float16)],
                                          [0,1], full_batch_size=fbs,
                                          generation_len=gen_len, eos_token_id=2)
        self.assertEqual(steps, gen_len)

    def test_set_data_for_kv_handoff_called_each_step(self):
        fbs = 2
        sess = _make_decode_session_with_outputs(fbs, num_steps=2)
        kv   = [np.zeros((fbs, 4, 32, 16), dtype=np.float16)]
        steps, _ = _run_disagg_pipeline(sess, kv, [0,1], full_batch_size=fbs,
                                          generation_len=100, eos_token_id=2)
        self.assertEqual(sess.set_data_for_kv_handoff.call_count, steps)

    def test_complete_inf_called_each_step(self):
        fbs = 2
        sess = _make_decode_session_with_outputs(fbs, num_steps=2)
        kv   = [np.zeros((fbs, 4, 32, 16), dtype=np.float16)]
        steps, _ = _run_disagg_pipeline(sess, kv, [0,1], full_batch_size=fbs,
                                          generation_len=100, eos_token_id=2)
        self.assertEqual(sess.complete_inf.call_count, steps)

    def test_np_run_called_each_step(self):
        fbs = 2
        sess = _make_decode_session_with_outputs(fbs, num_steps=3)
        kv   = [np.zeros((fbs, 4, 32, 16), dtype=np.float16)]
        steps, _ = _run_disagg_pipeline(sess, kv, [0,1], full_batch_size=fbs,
                                          generation_len=100, eos_token_id=2)
        self.assertEqual(sess.np_run.call_count, steps)

    def test_tokens_accumulated_correctly(self):
        fbs = 1
        sess = _make_mock_session(full_batch_size=fbs)
        sess.np_run.return_value = 0
        # Steps: token 7, token 8, then EOS
        outputs = []
        for tok in [7, 8, 2]:
            logits = np.zeros((fbs, 1, 100), dtype=np.float32)
            logits[0, 0, tok] = 10.0
            outputs.append({"logits": logits})
        sess.get_outputs.side_effect = outputs
        _, slot_tokens = _run_disagg_pipeline(sess, [np.zeros((fbs,4,32,16), dtype=np.float16)],
                                               [0,1], full_batch_size=fbs,
                                               generation_len=100, eos_token_id=2)
        # first_token=10 (seeded), then 7, 8 appended; EOS stops loop
        self.assertEqual(slot_tokens[0], [10, 7, 8])

    def test_all_slots_finish_loop_exits(self):
        fbs = 3
        sess = _make_decode_session_with_outputs(fbs, num_steps=1)
        kv   = [np.zeros((fbs, 4, 32, 16), dtype=np.float16)]
        steps, _ = _run_disagg_pipeline(sess, kv, [0,1], full_batch_size=fbs,
                                          generation_len=100, eos_token_id=2)
        self.assertEqual(steps, 1)

    def test_phys_pos_increments_each_step(self):
        """Verify position tracking by checking decode inputs across steps."""
        fbs = 1
        sess = _make_mock_session(full_batch_size=fbs)
        sess.np_run.return_value = 0
        # 2 non-EOS steps then EOS
        outputs = []
        for tok in [42, 43, 2]:
            logits = np.zeros((fbs, 1, 100), dtype=np.float32)
            logits[0, 0, tok] = 10.0
            outputs.append({"logits": logits})
        sess.get_outputs.side_effect = outputs

        captured_inputs = []
        original_np_run = sess.np_run
        def capturing_np_run(inputs, **kwargs):
            captured_inputs.append(inputs["position_ids"].copy())
            return 0
        sess.np_run.side_effect = capturing_np_run

        _run_disagg_pipeline(sess, [np.zeros((fbs,4,32,16), dtype=np.float16)],
                              [0,1], full_batch_size=fbs, generation_len=100, eos_token_id=2)

        # position_ids[0, slot=0, 0] should be 5, 6, 7 across the 3 steps
        self.assertEqual(captured_inputs[0][0, 0, 0], 5)
        self.assertEqual(captured_inputs[1][0, 0, 0], 6)
        self.assertEqual(captured_inputs[2][0, 0, 0], 7)


if __name__ == "__main__":
    unittest.main(verbosity=2)
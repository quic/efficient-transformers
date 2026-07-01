# -----------------------------------------------------------------------------
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------
"""Exec-obj pool management for pipelined QAIC inference."""

from __future__ import annotations

import logging
import os
from queue import Queue

logger = logging.getLogger(__name__)

_PREFILL_QUEUE_LEN_ENV = "VLLM_QAIC_PREFILL_QUEUE_LEN"
_EXEC_TIMEOUT_ENV = "VLLM_QAIC_ASYNC_SCHEDULING_EXEC_TIMEOUT"


class ExecObjPool:
    """
    Manages exec-obj slot allocation for prefill/decode pipelines.

    Layout in the execObj list:
      [0 .. decode_num)         -> decode slot(s)
      [decode_num .. queue_len) -> prefill pool

    Parameters
    ----------
    cluster_id : str | None
        "prefill"  -> only prefill exec-objs, no decode slot.
        "decode"   -> only one decode exec-obj, no prefill pool.
        None       -> combined mode: 1 decode slot + 1 prefill slot.
    stages : int
        Number of pipeline stages for pipelined prefill.
        prefill exec-obj pool size = stages + 1 (overridable via env-var).
    """

    def __init__(self, cluster_id: str | None = None, stages: int = 1) -> None:
        self.cluster_id = cluster_id
        self.stages = stages
        self.exec_timeout: int = int(os.getenv(_EXEC_TIMEOUT_ENV, 300))

        if cluster_id == "decode":
            self.prefill_num: int = 0
            self.decode_num: int = 1
            self.decode_idx: int | None = 0
        elif cluster_id == "prefill":
            self.prefill_num = int(os.getenv(_PREFILL_QUEUE_LEN_ENV, self.stages + 1))
            self.decode_num = 0
            self.decode_idx = None
        else:
            self.prefill_num = int(os.getenv(_PREFILL_QUEUE_LEN_ENV, 1))
            self.decode_num = 1
            self.decode_idx = 0

        self.queue_len: int = self.prefill_num + self.decode_num

        self._available_prefill: Queue[int] = Queue()
        prefill_start = self.decode_num
        for i in range(prefill_start, prefill_start + self.prefill_num):
            self._available_prefill.put(i)

        logger.debug(
            "ExecObjPool: cluster_id=%s  prefill_slots=%d  decode_slots=%d",
            cluster_id,
            self.prefill_num,
            self.decode_num,
        )

    def acquire(self, is_prefill: bool = True) -> int:
        """
        Acquire an exec-obj slot index.

        For prefill: blocks until a slot is available from the pool.
        For decode: returns the fixed decode slot index.
        """
        if is_prefill:
            return self._available_prefill.get(timeout=self.exec_timeout)
        assert self.decode_idx is not None, "decode_idx is None -- session not configured for decode"
        return self.decode_idx

    def release(self, index: int, is_prefill: bool = True) -> None:
        """
        Release an exec-obj slot back to the pool.

        For prefill: returns the slot to the available queue.
        For decode: no-op (fixed slot).
        """
        if is_prefill:
            self._available_prefill.put(index)
            logger.debug("Released prefill exec-obj %d", index)

# -----------------------------------------------------------------------------
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------

"""
Cross-process device pool for QAIC runtime tests.

Runtime lanes must not double-book physical devices when pytest-xdist is
running multiple workers, so this module implements a file-lock backed
semaphore over ``device_ids``. Locking is done via ``fcntl.flock`` on a
per-device pidfile, which is respected by every process on the same host.

Device count is inferred from ``QEFF_DYNAMO_DEVICE_COUNT`` (default 4) or an
explicit ``QEFF_DYNAMO_DEVICE_IDS`` comma-separated list.
"""

from __future__ import annotations

import contextlib
import errno
import fcntl
import os
import time
from pathlib import Path
from typing import Iterator, List, Optional

DEFAULT_DEVICE_COUNT = 4
DEVICE_COUNT_ENV = "QEFF_DYNAMO_DEVICE_COUNT"
DEVICE_IDS_ENV = "QEFF_DYNAMO_DEVICE_IDS"
LOCK_ROOT_ENV = "QEFF_DYNAMO_LOCK_DIR"
DEFAULT_LOCK_SUBDIR = Path("test-results") / "dynamo" / "locks"


def _default_lock_root() -> Path:
    env_value = os.environ.get(LOCK_ROOT_ENV)
    if env_value:
        return Path(env_value)
    return Path.cwd() / DEFAULT_LOCK_SUBDIR


def available_device_ids() -> List[int]:
    """Return the device ids the pool should manage."""
    explicit = os.environ.get(DEVICE_IDS_ENV)
    if explicit:
        return [int(part.strip()) for part in explicit.split(",") if part.strip()]
    count = int(os.environ.get(DEVICE_COUNT_ENV, DEFAULT_DEVICE_COUNT))
    return list(range(count))


class DevicePoolExhausted(RuntimeError):
    """Raised when the requested number of devices cannot be acquired."""


class DevicePool:
    """File-lock-backed semaphore over QAIC device ids.

    Acquisition is *all-or-nothing*: ``acquire(n)`` locks n distinct device
    ids atomically and releases every partial hold on failure. This keeps
    multi-device tests (MDP) safe under xdist.
    """

    def __init__(self, device_ids: Optional[List[int]] = None, lock_root: Optional[Path] = None):
        self._device_ids = device_ids if device_ids is not None else available_device_ids()
        self._lock_root = lock_root or _default_lock_root()
        self._lock_root.mkdir(parents=True, exist_ok=True)

    @property
    def device_ids(self) -> List[int]:
        return list(self._device_ids)

    def _lock_path(self, device_id: int) -> Path:
        return self._lock_root / f"device-{device_id}.lock"

    def _try_lock(self, device_id: int):
        path = self._lock_path(device_id)
        # 'a+' avoids clobbering. We hold the fd for the duration of the lock.
        fd = open(path, "a+")
        try:
            fcntl.flock(fd.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as exc:
            fd.close()
            if exc.errno in (errno.EWOULDBLOCK, errno.EAGAIN):
                return None
            raise
        try:
            fd.seek(0)
            fd.truncate()
            fd.write(f"{os.getpid()}\n")
            fd.flush()
        except OSError:
            # Non-fatal metadata write failure; the lock itself is what matters.
            pass
        return fd

    def _release(self, holds):
        for fd in holds:
            try:
                fcntl.flock(fd.fileno(), fcntl.LOCK_UN)
            except OSError:
                pass
            try:
                fd.close()
            except OSError:
                pass

    @contextlib.contextmanager
    def acquire(self, count: int = 1, *, timeout: float = 300.0, poll_interval: float = 0.5) -> Iterator[List[int]]:
        """Acquire ``count`` distinct devices atomically, releasing on exit."""
        if count < 1:
            raise ValueError("count must be >= 1")
        if count > len(self._device_ids):
            raise DevicePoolExhausted(f"Requested {count} devices but pool only manages {len(self._device_ids)}")

        deadline = time.monotonic() + timeout
        while True:
            holds = []
            acquired: List[int] = []
            for device_id in self._device_ids:
                if len(acquired) == count:
                    break
                fd = self._try_lock(device_id)
                if fd is not None:
                    holds.append(fd)
                    acquired.append(device_id)
            if len(acquired) == count:
                try:
                    yield acquired
                finally:
                    self._release(holds)
                return
            # Not enough at this instant; release partial holds and retry.
            self._release(holds)
            if time.monotonic() >= deadline:
                raise DevicePoolExhausted(
                    f"Could not acquire {count} device(s) within {timeout:.1f}s (pool={self._device_ids})"
                )
            time.sleep(poll_interval)

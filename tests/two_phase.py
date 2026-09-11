# -----------------------------------------------------------------------------
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------

"""Shared plumbing for the two-phase (compile-warm, then execute) per-PR CI split.

The on-device stages spend most of their wall clock in export+compile, which needs
no accelerator, and only a small tail in on-device generate. Splitting the two lets
the compile phase run far wider than the card count allows:

  * Phase A -- ``QEFF_PER_PR_COMPILE_WARM_ONLY``: export+compile only, no device
    touched, so the worker count is bounded by CPU instead of by cards.
  * Phase B -- ``QEFF_PER_PR_SHARED_HOME``: the same tests again, now hitting the
    QPCs Phase A left in the cache, narrow enough to fit the available cards.

Both phases must share one ``QEFF_HOME`` for Phase B to hit that cache, so the
per-worker ``QEFF_HOME`` remap and the session-level cache wipe are both skipped in
this mode (see ``tests/conftest.py``). The caller owns the shared directory's
lifecycle: it starts clean and is removed once the whole two-phase run is done.
"""

import os
import re
from contextlib import contextmanager
from pathlib import Path

try:
    import fcntl
except ImportError:  # Non-POSIX host: fall back to no locking.
    fcntl = None

from QEfficient.utils.cache import QEFF_HOME


def is_two_phase_session():
    """True when either phase of the compile/execute split is active."""
    return bool(os.environ.get("QEFF_PER_PR_SHARED_HOME") or os.environ.get("QEFF_PER_PR_COMPILE_WARM_ONLY"))


def is_compile_warm_phase():
    """True in Phase A only: export+compile, no device touched."""
    return bool(os.environ.get("QEFF_PER_PR_COMPILE_WARM_ONLY"))


def no_cleanup(*args, **kwargs):
    """No-op cleanup used by both phases to preserve the shared warm QPC cache."""


def resolve_two_phase_cleanup(manual_cleanup, compile_only=False):
    """Adapt a check helper's ``(manual_cleanup, compile_only)`` pair to the active phase.

    Per-test cleanup has to be suppressed in *both* phases: variants of one model share a
    content-addressed export dir (the QPCs nest inside it), so a finishing variant's rmtree
    would destroy sibling variants' warm QPCs or in-progress compiles. Phase A additionally
    forces compile-only so it never reaches a device.

    Returns the pair unchanged outside a two-phase run, so default single-phase runs keep
    cleaning up after every test exactly as before.
    """
    if is_compile_warm_phase():
        return no_cleanup, True
    if os.environ.get("QEFF_PER_PR_SHARED_HOME"):
        return no_cleanup, compile_only
    return manual_cleanup, compile_only


@contextmanager
def model_export_compile_lock(model_name):
    """Serialize export+compile of one model across concurrent xdist workers.

    Only active in the two-phase shared-QEFF_HOME run (many variants of one model share a
    content-addressed ONNX export dir, so concurrent writers would tear the .onnx file).
    A no-op otherwise, so default single-phase runs are untouched.
    """
    if not is_two_phase_session() or fcntl is None:
        yield
        return

    lock_dir = Path(os.environ.get("QEFF_HOME", QEFF_HOME)) / ".locks"
    lock_dir.mkdir(parents=True, exist_ok=True)
    lock_path = lock_dir / (re.sub(r"[^A-Za-z0-9_.-]", "_", model_name) + ".lock")
    with open(lock_path, "a+", encoding="utf-8") as lockfile:
        fcntl.flock(lockfile.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(lockfile.fileno(), fcntl.LOCK_UN)

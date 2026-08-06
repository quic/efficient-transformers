# -----------------------------------------------------------------------------
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
#
# ----------------------------------------------------------------------------
"""Portable single-file parallel ONNX external-data save (no POSIX-only APIs).

This is a cross-platform (Linux / macOS / Windows) alternative to
``onnx_save_utils.save_dynamo_onnx``, which relies on ``os.pwrite`` and
``os.posix_fallocate`` — neither of which exists on Windows, and the latter
of which is Linux-only (also absent on macOS).

Design
------
Same overall layout as the pwrite-based version — one small ``.onnx`` proto
plus one flat ``.onnx.data`` file with tensors at precomputed offsets — but
the write path uses only APIs available on every platform:

  1. Compute each tensor's byte offset upfront, same as the pwrite version.
  2. Pre-size the data file with ``f.truncate(total_bytes)`` instead of
     ``posix_fallocate``. This is portable but produces a *sparse* file on
     Linux/macOS (no contiguous extent reserved) — it does not carry the
     HDD-elevator benefit that pre-allocation gives the pwrite version.
  3. Each worker opens its own file handle to the same path (``r+b``,
     large ``buffering``), seeks to its offset, and writes. A separate
     handle per worker is required: two threads sharing one file object and
     racing ``seek`` + ``write`` would be able to interleave and each other's
     writes, since seek-then-write is not atomic. Non-overlapping regions
     written through independent handles to the same underlying file are
     safe.
  4. Tensors are read via ``memoryview(...).cast("B")`` instead of
     ``const_value.tobytes()`` where possible, avoiding a full duplicate
     tensor-sized allocation per write. Falls back to ``tobytes()`` if the
     tensor does not support the buffer protocol directly.
  5. After all writes, each ``const_value`` is replaced with an
     ``ExternalTensor`` (location / offset / length), identical to the
     pwrite version.

Usage
-----
Replace::

    onnx_program.save(str(onnx_path))

with::

    from QEfficient.utils.onnx_save_utils_portable import save_dynamo_onnx_portable
    save_dynamo_onnx_portable(onnx_program, onnx_path)

Add ``del onnx_program; gc.collect()`` after the call to release ExportedProgram
weakrefs so ``_offload_model_weights`` (``torch.utils.swap_tensors``) can succeed.
"""

from __future__ import annotations

import logging
import os
from concurrent.futures import Future, ThreadPoolExecutor, as_completed
from pathlib import Path

logger = logging.getLogger(__name__)

_DEFAULT_NUM_WORKERS = 4

# Large internal buffer batches many small writes into fewer syscalls —
# the portable stand-in for the syscall-count benefit posix_fallocate/pwrite
# give on Linux.
_WRITE_BUFFER_SIZE = 64 * 1024 * 1024


def _as_buffer(const_value):
    """Return a zero-copy ``memoryview`` of ``const_value`` if possible,
    otherwise fall back to ``tobytes()`` (which allocates a full copy)."""
    try:
        return memoryview(const_value).cast("B")
    except TypeError:
        return memoryview(const_value.tobytes())


def _write_tensor_at_offset(
    path: Path,
    name: str,
    const_value,
    offset: int,
) -> tuple[str, int]:
    """Write one tensor into the shared data file at ``offset``.

    Opens its own file handle so concurrent workers each have an independent
    file position — required because ``seek`` followed by ``write`` is not
    atomic, and two threads sharing one file object could interleave.

    Returns:
        ``(name, nbytes_written)`` so the caller can verify completeness.
    """
    buf = _as_buffer(const_value)
    nbytes = len(buf)
    with open(path, "r+b", buffering=_WRITE_BUFFER_SIZE) as f:
        f.seek(offset)
        f.write(buf)
    logger.debug("wrote %s: %d bytes @ offset %d", name, nbytes, offset)
    return name, nbytes


def _make_external_tensor(_ir_core, *, location, offset, length, dtype, shape, name, base_dir):
    """Construct an ``ExternalTensor``, tolerating ``onnx_ir`` versions where
    ``base_dir`` is not a constructor argument (set as an attribute instead)."""
    try:
        return _ir_core.ExternalTensor(
            location=location,
            offset=offset,
            length=length,
            dtype=dtype,
            shape=shape,
            name=name,
            base_dir=base_dir,
        )
    except TypeError:
        ext = _ir_core.ExternalTensor(
            location=location,
            offset=offset,
            length=length,
            dtype=dtype,
            shape=shape,
            name=name,
        )
        ext.base_dir = base_dir
        return ext


def save_dynamo_onnx_portable(
    onnx_program,
    onnx_path: Path,
    num_workers: int = _DEFAULT_NUM_WORKERS,
    sync: bool = False,
) -> None:
    """Save a dynamo ``ONNXProgram`` to a single external data file, using
    only APIs available on Linux, macOS, and Windows.

    All weight tensors are written in parallel — each worker opens its own
    file handle and writes to its own non-overlapping byte range — into one
    pre-sized ``.data`` file.

    This is a drop-in replacement for ``onnx_program.save(str(onnx_path))``.

    Args:
        onnx_program: ``torch.onnx.ONNXProgram`` returned by
            ``torch.onnx.export(..., dynamo=True, f=None)``.
        onnx_path: Destination ``.onnx`` file.  The data file is written
            alongside it as ``<onnx_path>.data``.
        num_workers: Parallel writer threads.
        sync: If ``True``, call ``os.fsync`` on the data file before
            closing it, guaranteeing durability before the function returns.
            Leave ``False`` for normal export workflows where a crash just
            means re-exporting.
    """
    import gc as _gc
    import shutil as _shutil

    import onnx as _onnx
    import onnx_ir._core as _ir_core
    import onnx_ir.serde as _serde

    onnx_path = Path(onnx_path)
    base_dir = onnx_path.parent
    base_dir.mkdir(parents=True, exist_ok=True)

    data_filename = onnx_path.name + ".data"
    data_path = base_dir / data_filename
    data_path_tmp = base_dir / (data_filename + ".tmp")

    # Collect initializers that carry real tensor data.
    # PruneFakeInitializersTransform must have run before this call.
    initializers = [
        (name, value) for name, value in onnx_program.model.graph.initializers.items() if value.const_value is not None
    ]

    if not initializers:
        logger.warning("No initializers found in onnx_program — saving proto only.")
        proto = _serde.serialize_model(onnx_program.model)
        _onnx.save(proto, str(onnx_path))
        return

    # Sort largest-first: same rationale as the pwrite version — gives the
    # longest possible sequential run to whichever OS-level flushing happens
    # first, even though we no longer control page-cache ordering directly.
    initializers.sort(key=lambda x: x[1].const_value.nbytes, reverse=True)

    # Assign contiguous byte offsets.
    offsets: dict[str, int] = {}
    cursor = 0
    for name, value in initializers:
        offsets[name] = cursor
        cursor += value.const_value.nbytes
    total_bytes = cursor

    # Fast-fail before touching disk if there isn't enough free space.
    free = _shutil.disk_usage(base_dir).free
    if free < total_bytes * 1.05:
        raise RuntimeError(f"Insufficient disk space at {base_dir}: need {total_bytes} bytes, only {free} available")

    # Warn if concurrent buffer copies could exhaust available RAM. Advisory
    # only — memoryview avoids a copy in the common case, but the tobytes()
    # fallback path does allocate one.
    try:
        import psutil as _psutil

        available_ram = _psutil.virtual_memory().available
        largest_tensor_bytes = initializers[0][1].const_value.nbytes
        effective_workers_for_ram = min(num_workers, len(initializers))
        peak_extra = largest_tensor_bytes * effective_workers_for_ram
        if peak_extra > available_ram * 0.9:
            logger.warning(
                "Peak RAM for tensor buffering (~%d GB) exceeds 90%% of available RAM (%d GB). "
                "Consider reducing num_workers.",
                peak_extra // (1024**3),
                available_ram // (1024**3),
            )
    except ImportError:
        pass

    effective_workers = min(num_workers, len(initializers))
    logger.info(
        "Saving %d initializers (%d bytes total) to %s using %d workers (portable path)",
        len(initializers),
        total_bytes,
        data_path,
        effective_workers,
    )

    try:
        # Pre-size the file. Portable equivalent of posix_fallocate, but does
        # NOT reserve a contiguous extent — this is a sparse file on
        # Linux/macOS and loses the HDD-elevator benefit the pwrite version
        # relies on. Acceptable tradeoff for portability; not a drop-in
        # performance replacement on spinning disks.
        with open(data_path_tmp, "wb") as f:
            f.truncate(total_bytes)

        errors: list[tuple[str, Exception]] = []
        nbytes_written: dict[str, int] = {}

        with ThreadPoolExecutor(max_workers=effective_workers) as pool:
            tensor_futures: dict[Future, str] = {
                pool.submit(
                    _write_tensor_at_offset,
                    data_path_tmp,
                    name,
                    value.const_value,
                    offsets[name],
                ): name
                for name, value in initializers
            }

            for future in as_completed(tensor_futures):
                tensor_name = tensor_futures[future]
                try:
                    name, n = future.result()
                    nbytes_written[name] = n
                except Exception as exc:  # noqa: BLE001
                    errors.append((tensor_name, exc))
                    logger.error("Failed to write tensor %r: %s", tensor_name, exc)

        if errors:
            raise RuntimeError(
                f"Parallel tensor write failed for {len(errors)} tensor(s): " + ", ".join(name for name, _ in errors)
            )

        # Verify every tensor landed at its full expected size before the
        # ExternalTensor descriptors (which record these lengths) are built.
        for name, value in initializers:
            expected = value.const_value.nbytes
            actual = nbytes_written.get(name)
            if actual != expected:
                raise RuntimeError(f"Tensor {name!r}: wrote {actual} bytes but expected {expected}")

        if sync:
            fd = os.open(str(data_path_tmp), os.O_RDWR)
            try:
                os.fsync(fd)
            finally:
                os.close(fd)

        # Only reached on full success — atomic on POSIX; on Windows,
        # os.replace (used implicitly by Path.rename on 3.3+ semantics for
        # same-drive renames) also replaces the destination atomically.
        data_path_tmp.replace(data_path)
    except Exception:
        data_path_tmp.unlink(missing_ok=True)
        raise

    # Swap const_value → ExternalTensor so serialize_model emits only
    # location/offset/length — no raw_data bytes in the proto.
    for name, value in initializers:
        nbytes = value.const_value.nbytes
        value.const_value = _make_external_tensor(
            _ir_core,
            location=data_filename,
            offset=offsets[name],
            length=nbytes,
            dtype=value.const_value.dtype,
            shape=value.const_value.shape,
            name=name,
            base_dir=str(base_dir),
        )

    # Serialize and write the small graph proto (no weight bytes).
    proto = _serde.serialize_model(onnx_program.model)
    _onnx.save(proto, str(onnx_path))
    logger.info("Saved ONNX proto to %s", onnx_path)

    # Release onnx_ir object graph ahead of the caller's del onnx_program.
    onnx_program.model.graph.initializers.clear()
    _gc.collect()

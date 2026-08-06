# -----------------------------------------------------------------------------
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
#
# ----------------------------------------------------------------------------
"""Single-file parallel ONNX external-data save optimised for spinning-disk / HDD-backed LVM.

Problem with the per-tensor-file approach on HDD
-------------------------------------------------
Writing one file per initializer (~725 files for a 70B model) forces the OS to
allocate 725 separate inodes and directory entries.  Even with pre-creation and
``posix_fallocate``, parallel threads writing to different files cause the HDD
elevator to handle competing request streams with zero opportunity to coalesce
across file boundaries.  The result is near-zero ``wrqm/s`` and 400–1300 ms
``w_await`` regardless of thread count.

Fix: single data file + ``pwrite`` at pre-calculated offsets
------------------------------------------------------------
  1. Compute each tensor's byte offset upfront (no locking needed — offsets are
     assigned before any I/O starts).
  2. Pre-allocate the entire data file with ``posix_fallocate`` so ext4 reserves
     one contiguous extent covering all tensors.
  3. Each worker calls ``os.pwrite(fd, data, offset)`` — a single syscall that
     lands the tensor at its pre-assigned position without seeking.
  4. The OS page cache sees one file's dirty pages and flushes them as one large
     sequential stream.  The HDD elevator coalesces adjacent dirty pages and
     issues large merged requests — ``wrqm/s`` goes up, ``w_await`` comes down.
  5. After all writes, each ``const_value`` is replaced with an ``ExternalTensor``
     (location / offset / length) so ``serde.serialize_model`` emits a small
     proto with no inline ``raw_data``.

Layout
------
  <onnx_path>            — small ONNX proto (graph structure only, no weights)
  <onnx_path>.data       — single flat file: all weight tensors concatenated

Tensor ordering inside the data file
-------------------------------------
Tensors are sorted largest-first and assigned contiguous offsets in that order.
This means the beginning of the file contains the largest (most expensive)
tensors, which are submitted to the thread pool first.  On HDD, the page cache
starts flushing from offset 0 immediately, giving the elevator the longest
possible sequential run before small tensors at the tail arrive.

Usage
-----
Replace::

    onnx_program.save(str(onnx_path))

with::

    from QEfficient.base.dynamo_onnx_save_singlefile import save_dynamo_onnx_singlefile
    save_dynamo_onnx_singlefile(onnx_program, onnx_path)

Add ``del onnx_program; gc.collect()`` after the call to release ExportedProgram
weakrefs so ``_offload_model_weights`` (``torch.utils.swap_tensors``) can succeed.
"""

from __future__ import annotations

import logging
import os
from concurrent.futures import Future, ThreadPoolExecutor, as_completed
from pathlib import Path

logger = logging.getLogger(__name__)

# 2 workers is optimal for a single HDD spindle.  Each worker calls pwrite()
# into a non-overlapping region of the same fd; the page cache coalesces their
# dirty pages into large sequential flushes.  More workers add concurrency but
# the HDD elevator is the bottleneck, so 2 keeps the queue depth low and
# w_await short.  Raise to 4–8 for NVMe-backed storage.
_DEFAULT_NUM_WORKERS = 1


def _pwrite_tensor(
    fd: int,
    name: str,
    const_value,
    offset: int,
) -> tuple[str, int]:
    """Serialize one tensor and write it at ``offset`` inside the open ``fd``.

    Uses ``os.pwrite`` — a single syscall, no seek, no lock needed.  Multiple
    threads can call this concurrently on the same fd as long as their
    ``(offset, length)`` regions do not overlap.

    Returns:
        ``(name, nbytes_written)`` so the caller can verify completeness.
    """
    data: bytes = const_value.tobytes()
    nbytes = len(data)
    mv = memoryview(data)
    written = 0
    while written < nbytes:
        written += os.pwrite(fd, mv[written:], offset + written)
    logger.debug("pwrite %s: %d bytes @ offset %d", name, nbytes, offset)
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


def save_dynamo_onnx(
    onnx_program,
    onnx_path: Path,
    num_workers: int = _DEFAULT_NUM_WORKERS,
    sync: bool = False,
) -> None:
    """Save a dynamo ``ONNXProgram`` to a single external data file.

    All weight tensors are written in parallel via ``pwrite`` into one
    pre-allocated ``.data`` file.  The OS page cache sees one file's dirty
    pages and coalesces them into large sequential flushes — optimal for
    spinning-disk / HDD-backed LVM where per-file parallel writes cause
    head thrash.

    This is a drop-in replacement for ``onnx_program.save(str(onnx_path))``.

    Args:
        onnx_program: ``torch.onnx.ONNXProgram`` returned by
            ``torch.onnx.export(..., dynamo=True, f=None)``.
        onnx_path: Destination ``.onnx`` file.  The data file is written
            alongside it as ``<onnx_path>.data``.
        num_workers: Parallel writer threads.  Default 2 for HDD; raise to
            4–8 for NVMe.
        sync: If ``True``, call ``os.fdatasync`` on the data file before
            closing it, guaranteeing durability before the function returns.
            Costs 60–120 s for a 131 GB file on HDD; leave ``False`` for
            normal export workflows where a crash just means re-exporting.
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

    # Sort largest-first: largest tensors get the lowest offsets and are
    # submitted to the thread pool first.  The page cache starts flushing from
    # offset 0 immediately, giving the HDD elevator the longest sequential run.
    initializers.sort(key=lambda x: x[1].const_value.nbytes, reverse=True)

    # Assign contiguous byte offsets.  No padding/alignment needed — the ONNX
    # ExternalTensor spec allows arbitrary offsets, and pwrite handles any offset.
    offsets: dict[str, int] = {}
    cursor = 0
    for name, value in initializers:
        offsets[name] = cursor
        cursor += value.const_value.nbytes
    total_bytes = cursor

    # Fast-fail before touching disk if there isn't enough free space for the
    # full pre-allocation (5% headroom for filesystem overhead).
    free = _shutil.disk_usage(base_dir).free
    if free < total_bytes * 1.05:
        raise RuntimeError(f"Insufficient disk space at {base_dir}: need {total_bytes} bytes, only {free} available")

    # Warn if concurrent tobytes() copies could exhaust available RAM. This is
    # advisory only — available RAM fluctuates and the model may partially
    # offload during the write phase.
    try:
        import psutil as _psutil

        available_ram = _psutil.virtual_memory().available
        largest_tensor_bytes = initializers[0][1].const_value.nbytes
        effective_workers_for_ram = min(num_workers, len(initializers))
        peak_extra = largest_tensor_bytes * effective_workers_for_ram
        if peak_extra > available_ram * 0.9:
            logger.warning(
                "Peak RAM for tobytes() (~%d GB) exceeds 90%% of available RAM (%d GB). Consider reducing num_workers.",
                peak_extra // (1024**3),
                available_ram // (1024**3),
            )
    except ImportError:
        pass

    effective_workers = min(num_workers, len(initializers))
    logger.info(
        "Saving %d initializers (%d bytes total) to %s using %d workers",
        len(initializers),
        total_bytes,
        data_path,
        effective_workers,
    )

    try:
        # Pre-allocate the entire data file in one shot.
        # posix_fallocate tells ext4 to reserve one contiguous extent covering all
        # tensors upfront.  Without this, block allocation happens per-pwrite and
        # mballoc may scatter data blocks across the platter.
        fd = os.open(str(data_path_tmp), os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o644)
        try:
            _fallocate = getattr(os, "posix_fallocate", None)
            if _fallocate is not None:
                try:
                    _fallocate(fd, 0, total_bytes)
                except OSError as e:
                    logger.warning(
                        "posix_fallocate failed (%s) — proceeding without pre-allocation; "
                        "blocks may be scattered across the platter",
                        e,
                    )

            # Parallel pwrite phase.
            errors: list[tuple[str, Exception]] = []
            nbytes_written: dict[str, int] = {}

            with ThreadPoolExecutor(max_workers=effective_workers) as pool:
                tensor_futures: dict[Future, str] = {
                    pool.submit(
                        _pwrite_tensor,
                        fd,
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
                    f"Parallel tensor write failed for {len(errors)} tensor(s): "
                    + ", ".join(name for name, _ in errors)
                )

            # Verify every tensor landed at its full expected size before the
            # ExternalTensor descriptors (which record these lengths) are built.
            for name, value in initializers:
                expected = value.const_value.nbytes
                actual = nbytes_written.get(name)
                if actual != expected:
                    raise RuntimeError(f"Tensor {name!r}: wrote {actual} bytes but expected {expected}")

            if sync:
                os.fdatasync(fd)

        finally:
            os.close(fd)

        # Only reached on full success — atomic on POSIX, same filesystem.
        data_path_tmp.rename(data_path)
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

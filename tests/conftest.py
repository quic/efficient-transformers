# -----------------------------------------------------------------------------
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------

import os
import shutil
import tempfile
import time
from contextlib import contextmanager
from pathlib import Path

import pytest
from transformers import logging

import QEfficient.utils.cache as qeff_cache
from QEfficient.utils.cache import QEFF_HOME
from QEfficient.utils.logging_utils import logger

try:
    import fcntl
except ImportError:  # pragma: no cover - CI runs on Linux.
    fcntl = None

_QUICKCHECK_FILE = "tests/unit_test/models/test_model_quickcheck.py"
_QUICKCHECK_SUMMARY = {}
_QUICKCHECK_META = {
    "test_causal_lm_cpu_runtime_parity_with_api_runner": (
        "Causal LM",
        "Full parity: HF PyTorch vs QEff PyTorch vs ORT tokens",
    ),
    "test_vlm_text_side_runtime_parity_and_full_export": (
        "VLM",
        "Text-side full parity + full VLM export smoke",
    ),
    "test_vlm_export_smoke_additional_models": (
        "VLM",
        "Export smoke with text-side fallback when needed",
    ),
    "test_text_embedding_cpu_parity_and_export": (
        "Text Embedding",
        "Tensor parity: HF vs QEff PyTorch vs ORT",
    ),
    "test_audio_embedding_ctc_cpu_parity_and_export": (
        "Audio CTC",
        "Logits parity: HF vs ORT + export",
    ),
    "test_seq_classification_cpu_parity_and_export": (
        "Sequence Classification",
        "Logits parity: HF vs QEff PyTorch vs ORT",
    ),
    "test_whisper_export_smoke": (
        "Whisper",
        "Export smoke + retained-state outputs check",
    ),
    "test_causal_subfunction_export_smoke": (
        "Causal LM",
        "Subfunction export check (with/without QEffGPT2Block)",
    ),
    "test_causal_subfunction_export_smoke_all_models": (
        "Causal LM",
        "Full parity: HF PyTorch vs QEff PyTorch vs ORT tokens (subfunctions)",
    ),
    "test_prefix_caching_continuous_batching_export_and_ort_smoke": (
        "Prefix Caching",
        "Continuous-batching export structural checks",
    ),
    "test_awq_export_smoke": (
        "AWQ",
        "Export smoke + MatMulNBits presence check",
    ),
}


def _qaic_device_pool():
    pool = os.environ.get("QEFF_QAIC_DEVICE_POOL", "0,1,2,3")
    return [int(device_id) for device_id in pool.split(",") if device_id.strip()]


def _qaic_device_lock_dir():
    return Path(os.environ.get("QEFF_QAIC_DEVICE_LOCK_DIR", tempfile.gettempdir())) / "qeff_qaic_device_locks"


@contextmanager
def _allocated_qaic_device():
    devices = _qaic_device_pool()
    if not devices:
        yield None
        return

    lock_dir = _qaic_device_lock_dir()
    lock_dir.mkdir(parents=True, exist_ok=True)
    locked_file = None
    try:
        while True:
            for device_id in devices:
                lock_file = open(lock_dir / f"device_{device_id}.lock", "a+", encoding="utf-8")
                if fcntl is None:
                    locked_file = lock_file
                    yield device_id
                    return
                try:
                    fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                except BlockingIOError:
                    lock_file.close()
                    continue
                locked_file = lock_file
                yield device_id
                return
            time.sleep(1)
    finally:
        if locked_file is not None:
            try:
                if fcntl is not None:
                    fcntl.flock(locked_file.fileno(), fcntl.LOCK_UN)
                locked_file.close()
            except OSError:
                pass


def _configure_worker_qeff_home():
    global QEFF_HOME

    worker_id = os.environ.get("PYTEST_XDIST_WORKER")
    if not worker_id:
        return

    base_qeff_home = Path(os.environ.get("QEFF_HOME", str(QEFF_HOME)))
    worker_qeff_home = base_qeff_home if base_qeff_home.name == worker_id else base_qeff_home / worker_id
    worker_qeff_home.mkdir(parents=True, exist_ok=True)
    os.environ["QEFF_HOME"] = str(worker_qeff_home)

    QEFF_HOME = worker_qeff_home
    qeff_cache.QEFF_HOME = worker_qeff_home


def _is_nightly_pipeline_session(session):
    """Check if this is a nightly_pipeline test session"""
    # Check invocation args
    if hasattr(session.config, "invocation_params"):
        args_str = " ".join(session.config.invocation_params.args)
        if "nightly_pipeline" in args_str:
            return True

    # Check if any collected items are from nightly_pipeline
    if hasattr(session, "items") and session.items:
        for item in session.items:
            if "nightly_pipeline" in item.nodeid:
                return True

    return False


def qeff_models_clean_up(qeff_dir=None):
    """
    Clean up QEFF models and cache.

    Args:
        qeff_dir: Can be a string (file/dir path), PosixPath, or list of strings/PosixPath objects
                 If a file path is provided, its parent directory will be deleted
    """
    if qeff_dir is None:
        qeff_dir = QEFF_HOME

    if isinstance(qeff_dir, (str, Path)):
        paths = [qeff_dir]
    else:
        paths = qeff_dir

    for path in paths:
        try:
            path_str = str(path)
            if os.path.isfile(path_str):
                dir_to_delete = os.path.dirname(path_str)
                if os.path.exists(dir_to_delete):
                    shutil.rmtree(dir_to_delete)
                    print(f"\n.............Cleaned up {dir_to_delete}")
            elif os.path.isdir(path_str):
                if os.path.exists(path_str):
                    shutil.rmtree(path_str)
                    print(f"\n.............Cleaned up {path_str}")
        except Exception as e:
            print(f"\n.............Error cleaning up {path}: {e}")


@pytest.fixture
def manual_cleanup():
    """Fixture to manually trigger cleanup"""
    return qeff_models_clean_up


@pytest.fixture(autouse=True)
def qaic_device_allocator(request, monkeypatch):
    """Assign one QAIC device per on_qaic test when xdist is enabled in CI.

    The allocator is opt-in so full-layer or multi-device runs can stay on the
    default runtime behavior. For one-device tests it redirects the implicit
    default device 0 to the worker's locked device.
    """
    if "on_qaic" not in request.keywords or os.environ.get("QEFF_ENABLE_QAIC_DEVICE_ALLOCATOR") != "1":
        yield
        return

    with _allocated_qaic_device() as device_id:
        if device_id is None:
            yield
            return

        monkeypatch.setenv("QEFF_QAIC_DEVICE_ID", str(device_id))
        from QEfficient.generation.cloud_infer import QAICInferenceSession

        original_init = QAICInferenceSession.__init__

        def _init_with_allocated_device(self, qpc_path, device_ids=None, *args, **kwargs):
            if device_ids is None or device_ids == [0]:
                device_ids = [device_id]
            return original_init(self, qpc_path, device_ids, *args, **kwargs)

        monkeypatch.setattr(QAICInferenceSession, "__init__", _init_with_allocated_device)
        yield


def pytest_sessionstart(session):
    logger.info("PYTEST Session Starting ...")
    # Skip cleanup for nightly_pipeline tests
    if _is_nightly_pipeline_session(session):
        logger.info("Skipping cleanup for nightly_pipeline tests")
        return
    # Suppress transformers warnings about unused weights when loading models with fewer layers
    logging.set_verbosity_error()

    qeff_models_clean_up()


def pytest_configure(config):
    """Register custom markers for test categorization."""
    _configure_worker_qeff_home()
    config.addinivalue_line("markers", "llm_model: mark test as a pure LLM model inference test")
    config.addinivalue_line(
        "markers", "feature: mark test as a feature-specific test (SPD, sampler, prefix caching, LoRA, etc.)"
    )


def pytest_sessionfinish(session, exitstatus):
    inside_worker = getattr(session.config, "workerinput", None)
    # Skip cleanup for nightly_pipeline tests
    if _is_nightly_pipeline_session(session):
        logger.info("Skipping cleanup for nightly_pipeline tests")
        return
    if inside_worker is None:
        qeff_models_clean_up()
        logger.info("...PYTEST Session Ended.")


def pytest_runtest_logreport(report):
    if _QUICKCHECK_FILE not in report.nodeid:
        return

    if report.when == "call":
        _QUICKCHECK_SUMMARY[report.nodeid] = report.outcome
        return

    if report.when == "setup" and report.outcome == "skipped":
        _QUICKCHECK_SUMMARY.setdefault(report.nodeid, report.outcome)


def pytest_terminal_summary(terminalreporter):
    if not _QUICKCHECK_SUMMARY:
        return

    terminalreporter.section("Quickcheck Coverage Summary", sep="=")
    header = f"{'Status':7}  {'Test Case':58}  {'Category':24}  Validation"
    terminalreporter.write_line(header)
    terminalreporter.write_line("-" * len(header))

    for nodeid in sorted(_QUICKCHECK_SUMMARY):
        test_case = nodeid.split("::", 1)[1]
        base_name = test_case.split("[", 1)[0]
        category, validation = _QUICKCHECK_META.get(base_name, ("Other", "N/A"))
        status = _QUICKCHECK_SUMMARY[nodeid].upper()
        terminalreporter.write_line(f"{status:7}  {test_case:58}  {category:24}  {validation}")

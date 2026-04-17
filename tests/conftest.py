# -----------------------------------------------------------------------------
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------

import os
import shutil
from pathlib import Path

import pytest
from transformers import logging

from QEfficient.utils.cache import QEFF_HOME
from QEfficient.utils.logging_utils import logger

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


def qeff_models_clean_up(qeff_dir=QEFF_HOME):
    """
    Clean up QEFF models and cache.

    Args:
        qeff_dir: Can be a string (file/dir path), PosixPath, or list of strings/PosixPath objects
                 If a file path is provided, its parent directory will be deleted
    """
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


def pytest_sessionstart(session):
    logger.info("PYTEST Session Starting ...")

    # Suppress transformers warnings about unused weights when loading models with fewer layers
    logging.set_verbosity_error()

    qeff_models_clean_up()


def pytest_configure(config):
    """Register custom markers for test categorization."""
    config.addinivalue_line("markers", "llm_model: mark test as a pure LLM model inference test")
    config.addinivalue_line(
        "markers", "feature: mark test as a feature-specific test (SPD, sampler, prefix caching, LoRA, etc.)"
    )


def pytest_sessionfinish(session, exitstatus):
    inside_worker = getattr(session.config, "workerinput", None)
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

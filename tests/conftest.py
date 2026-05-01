# -----------------------------------------------------------------------------
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------

import logging as py_logging
import os
import shutil
from collections import defaultdict
from pathlib import Path

import pytest
from transformers import logging as hf_logging

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
    "test_qwen_quickcheck_subfunction_registration": (
        "Qwen",
        "Tiny synthetic decoder subfunction registration",
    ),
    "test_qwen_quickcheck_subfunction_setup_toggle": (
        "Qwen",
        "Subfunction/non-subfunction setup without ONNX export",
    ),
    "test_qwen_moe_quickcheck_layerwise_mode": (
        "Qwen MoE",
        "Layerwise/non-layerwise decoder registration",
    ),
    "test_qwen_quickcheck_hf_qeff_ort_default_parity": (
        "Qwen",
        "Full logits parity: HF PyTorch vs QEff PyTorch vs ORT (default export)",
    ),
    "test_qwen_moe_quickcheck_hf_qeff_ort_prefill_only_parity": (
        "Qwen MoE",
        "Full logits parity: HF PyTorch vs QEff PyTorch vs ORT (prefill-only transform)",
    ),
    "test_qwen_moe_quickcheck_layerwise_hf_qeff_ort_parity": (
        "Qwen MoE",
        "Full logits parity: HF PyTorch vs QEff PyTorch vs ORT (layerwise)",
    ),
    "test_repeat_kv_quickcheck_hf_qeff_ort_parity": (
        "Causal LM",
        "RepeatKV parity: HF PyTorch vs QEff PyTorch vs ORT logits",
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

# Reduce noisy PyTorch C++ warning logs (e.g., torchvision op registration warnings)
os.environ.setdefault("TORCH_CPP_LOG_LEVEL", "ERROR")
os.environ.setdefault("GLOG_minloglevel", "2")


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
    # Skip cleanup for nightly_pipeline tests
    if _is_nightly_pipeline_session(session):
        logger.info("Skipping cleanup for nightly_pipeline tests")
        return
    # Suppress transformers warnings about unused weights when loading models with fewer layers
    hf_logging.set_verbosity_error()

    # Suppress noisy ONNX torchvision-missing warnings from torch exporter internals.
    py_logging.getLogger("torch.onnx._internal.exporter._registration").setLevel(py_logging.ERROR)
    py_logging.getLogger("torch.onnx").setLevel(py_logging.ERROR)

    qeff_models_clean_up()


def pytest_configure(config):
    """Register custom markers for test categorization."""
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


def pytest_terminal_summary(terminalreporter, exitstatus, config):
    group_stats = defaultdict(lambda: defaultdict(int))
    seen_status = set()
    seen_total = set()

    def _group_from_report(report):
        keywords = getattr(report, "keywords", {}) or {}
        if "llm_model" in keywords:
            return "llm_model"
        if "feature" in keywords:
            return "feature"
        return "unmarked"

    for status in ("passed", "failed", "skipped", "xfailed", "xpassed", "error"):
        for report in terminalreporter.stats.get(status, []):
            nodeid = getattr(report, "nodeid", None)
            when = getattr(report, "when", "call")
            if not nodeid or when != "call":
                continue
            group = _group_from_report(report)

            status_key = (group, nodeid, status)
            if status_key in seen_status:
                continue
            seen_status.add(status_key)
            group_stats[group][status] += 1

            total_key = (group, nodeid)
            if total_key not in seen_total:
                seen_total.add(total_key)
                group_stats[group]["total"] += 1

    headers = ["group", "total", "passed", "failed", "skipped", "xfailed", "xpassed", "error"]
    rows = []
    order = ["llm_model", "feature", "unmarked"]
    for group in order:
        if group not in group_stats:
            continue
        rows.append([group] + [str(group_stats[group][name]) for name in headers[1:]])

    if rows:
        widths = [max(len(headers[i]), *(len(row[i]) for row in rows)) for i in range(len(headers))]

        def fmt(row):
            return " | ".join(row[i].ljust(widths[i]) for i in range(len(headers)))

        terminalreporter.write_sep("-", "QEff Test Summary")
        terminalreporter.write_line(fmt(headers))
        terminalreporter.write_line("-+-".join("-" * w for w in widths))
        for row in rows:
            terminalreporter.write_line(fmt(row))

        xfailed_reports = [r for r in terminalreporter.stats.get("xfailed", []) if getattr(r, "when", "call") == "call"]
        failed_reports = [r for r in terminalreporter.stats.get("failed", []) if getattr(r, "when", "call") == "call"]

        if xfailed_reports:
            terminalreporter.write_sep("-", "Known Limitations (xfailed)")
            for report in xfailed_reports:
                reason = getattr(getattr(report, "longrepr", None), "reprcrash", None)
                reason_text = reason.message if reason and hasattr(reason, "message") else "expected failure"
                terminalreporter.write_line(f"- {report.nodeid}: {reason_text}")

        if failed_reports:
            terminalreporter.write_sep("-", "Failures")
            for report in failed_reports:
                terminalreporter.write_line(f"- {report.nodeid}")

    if _QUICKCHECK_SUMMARY:
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

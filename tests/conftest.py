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


# Number of QAic cards on the CI machine. Workers are sharded round-robin
# across these cards via QAIC_VISIBLE_DEVICES. Override with QEFF_NUM_QAIC_CARDS
# if a host has a different count.
_QAIC_CARDS_DEFAULT = 4


def _xdist_worker_index():
    worker = os.environ.get("PYTEST_XDIST_WORKER")
    if not worker or not worker.startswith("gw"):
        return None
    try:
        return int(worker[2:])
    except ValueError:
        return None


@pytest.fixture(scope="session", autouse=True)
def _qaic_device_for_xdist_worker():
    """Pin each pytest-xdist worker to one of the QAic cards.

    Serial runs (no xdist) and runs that already export QAIC_VISIBLE_DEVICES
    are left untouched. Under ``pytest -n 4`` on a 4-card host, gw0..gw3 each
    own one card — so .compile()/.generate() across workers run in parallel,
    while same-worker calls remain sequential on that card.

    QEFF_QAIC_CARD_OFFSET allows two stages to run simultaneously on non-
    overlapping card slices. E.g. stage A sets QEFF_NUM_QAIC_CARDS=2 + offset=0
    → cards 0,1; stage B sets QEFF_NUM_QAIC_CARDS=2 + offset=2 → cards 2,3.
    """
    if "QAIC_VISIBLE_DEVICES" in os.environ:
        return
    idx = _xdist_worker_index()
    if idx is None:
        return
    cards = max(1, int(os.environ.get("QEFF_NUM_QAIC_CARDS", _QAIC_CARDS_DEFAULT)))
    offset = int(os.environ.get("QEFF_QAIC_CARD_OFFSET", 0))
    os.environ["QAIC_VISIBLE_DEVICES"] = str(offset + (idx % cards))


@pytest.fixture(scope="session", autouse=True)
def _qeff_home_per_xdist_worker():
    """Give each xdist worker its own QEFF_HOME subdir so compile-cache writes
    don't race. Serial runs are untouched. The fixture only nudges the env var
    for the lifetime of the worker process; cleanup of the per-worker subtree
    happens via the existing pytest_sessionfinish hook."""
    idx = _xdist_worker_index()
    if idx is None:
        return
    base = os.environ.get("QEFF_HOME")
    if not base:
        return
    worker_home = os.path.join(base, f"worker_{idx}")
    os.makedirs(worker_home, exist_ok=True)
    os.environ["QEFF_HOME"] = worker_home


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
    config.addinivalue_line("markers", "llm_model: mark test as a pure LLM model inference test")
    config.addinivalue_line(
        "markers", "feature: mark test as a feature-specific test (SPD, sampler, prefix caching, LoRA, etc.)"
    )
    _install_tiny_model_remap_if_active()


def pytest_collection_modifyitems(config, items):
    """Under the per-PR tiny lane, auto-skip parametrize cases whose model_id
    appears in skip_no_tiny (no usable tiny on HF, no dummy-config path yet).
    Matches against the parametrize id token in the nodeid."""
    try:
        from tests.utils.tiny_overrides import _load_skip_set, _tiny_lane_active
    except Exception:
        return
    if not _tiny_lane_active():
        return
    skip_set = _load_skip_set()
    if not skip_set:
        return
    skip_marker = pytest.mark.skip(reason="No tiny variant available on HF; skipped under per-PR profile.")
    for item in items:
        nodeid = item.nodeid
        for skip_id in skip_set:
            # parametrize ids are bracketed in the nodeid as [foo/bar-...]
            if f"[{skip_id}" in nodeid or f"-{skip_id}]" in nodeid or f"-{skip_id}-" in nodeid:
                item.add_marker(skip_marker)
                break


def _install_tiny_model_remap_if_active() -> None:
    """When QEFF_TEST_PROFILE is dummy_layers_model or few_layers_model, wrap
    `from_pretrained` on every QEFFAuto* class and on the AutoModelFor* classes
    that tests use directly, so the first positional/`pretrained_model_name_or_path`
    arg is rewritten via tests/utils/tiny_overrides.resolve_model_id.

    full_layers_model (nightly) and unset profile leave class methods untouched.
    Idempotent: tagged with __qeff_tiny_remap__ so a re-call on the same class
    is a no-op.
    """
    try:
        from tests.utils.tiny_overrides import resolve_model_id, _tiny_lane_active  # noqa
    except Exception:
        return
    if not _tiny_lane_active():
        return

    import functools

    def _wrap(cls):
        # Pull the underlying function via __dict__ to avoid the descriptor protocol —
        # cls.from_pretrained returns a bound method, which trips functools.wraps.
        # We deliberately ignore inherited from_pretrained: wrapping a subclass that
        # inherits from an already-wrapped base would double-wrap. The base wrap
        # already handles the subclass via classmethod dispatch (cls -> child class).
        original_classmethod = cls.__dict__.get("from_pretrained")
        if original_classmethod is None:
            return
        original_func = getattr(original_classmethod, "__func__", original_classmethod)
        if getattr(original_func, "__qeff_tiny_remap__", False):
            return

        def remapped(klass, *args, **kwargs):
            if args:
                args = (resolve_model_id(args[0]),) + args[1:]
            elif "pretrained_model_name_or_path" in kwargs:
                kwargs["pretrained_model_name_or_path"] = resolve_model_id(kwargs["pretrained_model_name_or_path"])
            return original_func(klass, *args, **kwargs)

        functools.update_wrapper(remapped, original_func)
        remapped.__qeff_tiny_remap__ = True
        cls.from_pretrained = classmethod(remapped)

    def _wrap_load_config(cls):
        # Wrap load_config for diffusers/transformers pipeline classes so that
        # `ClassName.load_config("black-forest-labs/FLUX.1-schnell", ...)` is
        # intercepted and the repo id is remapped to its tiny substitute.
        original_classmethod = cls.__dict__.get("load_config")
        if original_classmethod is None:
            return
        original_func = getattr(original_classmethod, "__func__", original_classmethod)
        if getattr(original_func, "__qeff_tiny_remap__", False):
            return

        def remapped_cfg(klass, *args, **kwargs):
            if args:
                args = (resolve_model_id(args[0]),) + args[1:]
            elif "pretrained_model_name_or_path" in kwargs:
                kwargs["pretrained_model_name_or_path"] = resolve_model_id(kwargs["pretrained_model_name_or_path"])
            return original_func(klass, *args, **kwargs)

        functools.update_wrapper(remapped_cfg, original_func)
        remapped_cfg.__qeff_tiny_remap__ = True
        cls.load_config = classmethod(remapped_cfg)

    targets = []
    try:
        from QEfficient.transformers.models.modeling_auto import (
            QEFFAutoModel,
            QEFFAutoModelForCausalLM,
            QEFFAutoModelForCTC,
            QEFFAutoModelForImageTextToText,
            QEFFAutoModelForSequenceClassification,
            QEFFAutoModelForSpeechSeq2Seq,
        )

        targets.extend(
            [
                QEFFAutoModel,
                QEFFAutoModelForCausalLM,
                QEFFAutoModelForCTC,
                QEFFAutoModelForImageTextToText,
                QEFFAutoModelForSequenceClassification,
                QEFFAutoModelForSpeechSeq2Seq,
            ]
        )
    except Exception:
        pass

    try:
        from transformers import (
            AutoConfig,
            AutoModel,
            AutoModelForCausalLM,
            AutoModelForCTC,
            AutoModelForImageTextToText,
            AutoModelForSequenceClassification,
            AutoModelForSpeechSeq2Seq,
            AutoTokenizer,
        )

        targets.extend(
            [
                AutoConfig,
                AutoModel,
                AutoModelForCausalLM,
                AutoModelForCTC,
                AutoModelForImageTextToText,
                AutoModelForSequenceClassification,
                AutoModelForSpeechSeq2Seq,
                AutoTokenizer,
            ]
        )
    except Exception:
        pass

    for cls in targets:
        try:
            _wrap(cls)
        except Exception:
            continue

    # Wrap load_config on diffusers pipeline/model classes so that
    # ClassName.load_config("black-forest-labs/FLUX.1-schnell", ...) is remapped.
    # load_config lives on ConfigMixin, not on each subclass, so wrap it there.
    try:
        from diffusers.configuration_utils import ConfigMixin

        _orig_lc = ConfigMixin.__dict__.get("load_config")
        if _orig_lc is not None and not getattr(getattr(_orig_lc, "__func__", _orig_lc), "__qeff_tiny_remap__", False):
            _orig_func = getattr(_orig_lc, "__func__", _orig_lc)

            def _remapped_load_config(klass, pretrained_model_name_or_path, *args, **kwargs):
                return _orig_func(klass, resolve_model_id(pretrained_model_name_or_path), *args, **kwargs)

            import functools as _ft

            _ft.update_wrapper(_remapped_load_config, _orig_func)
            _remapped_load_config.__qeff_tiny_remap__ = True
            ConfigMixin.load_config = classmethod(_remapped_load_config)
    except Exception:
        pass

    # Also remap module-level helpers `load_hf_tokenizer` / `load_hf_processor`.
    # These call `hf_download(repo_id=...)` *before* AutoTokenizer.from_pretrained,
    # turning the model id into a local snapshot dir; once it's a path, our
    # AutoTokenizer wrap above can't tell which repo it came from. So we have to
    # rewrite the id at the entry point of these helpers.
    try:
        from QEfficient.utils import _utils as _qeff_utils

        for fn_name in ("load_hf_tokenizer", "load_hf_processor"):
            fn = getattr(_qeff_utils, fn_name, None)
            if fn is None or getattr(fn, "__qeff_tiny_remap__", False):
                continue
            original_fn = fn

            def make_wrapper(orig):
                def wrapper(pretrained_model_name_or_path, *args, **kwargs):
                    return orig(resolve_model_id(pretrained_model_name_or_path), *args, **kwargs)

                functools.update_wrapper(wrapper, orig)
                wrapper.__qeff_tiny_remap__ = True
                return wrapper

            setattr(_qeff_utils, fn_name, make_wrapper(original_fn))
    except Exception:
        pass

    # Remap `load_adapter(adapter_model_id, ...)` on QEffAutoLoraModelForCausalLM
    # so adapter IDs (hallisky/*, predibase/*) are swapped to tiny siblings that
    # match the remapped base model dimensions.
    try:
        from QEfficient.peft.lora.auto import QEffAutoLoraModelForCausalLM as _LoraModel

        _orig_load_adapter = _LoraModel.load_adapter
        if not getattr(_orig_load_adapter, "__qeff_tiny_remap__", False):

            def _wrapped_load_adapter(self, adapter_model_id, *args, **kwargs):
                return _orig_load_adapter(self, resolve_model_id(adapter_model_id), *args, **kwargs)

            _wrapped_load_adapter.__qeff_tiny_remap__ = True
            _LoraModel.load_adapter = _wrapped_load_adapter
    except Exception:
        pass


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

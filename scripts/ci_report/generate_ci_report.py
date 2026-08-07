# -----------------------------------------------------------------------------
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------

"""Generate a self-contained HTML report for a per-PR QEfficient CI run.

The Jenkins pipeline (``scripts/Jenkinsfile``) runs pytest across several stages,
each writing its own JUnit XML file under ``tests/`` (``tests_log1.xml``,
``tests_log2.xml``, ``tests_log2_feature.xml`` ...), which are then merged into
``tests/tests_log.xml``. This script reads the *per-stage* files and emits ONE
standalone ``ci_report.html`` with:

  * a top summary (verdict, KPI tiles, per-stage table) for reviewers/maintainers, and
  * full per-test drill-down (failures with traceback + captured log, collapsible
    per-stage tables, by-model rollup, slowest tests) for the PR owner.

It is intentionally dependency-free (Python standard library only) so it runs in
the CI container venv with nothing extra installed. Because ``pyproject.toml`` sets
``junit_logging = "all"``, each ``<testcase>`` already embeds captured stdout/log
and full failure tracebacks, so the XML is a self-sufficient source.

Usage::

    python3 scripts/ci_report/generate_ci_report.py \
        --xml-dir tests --output tests/ci_report.html \
        --pr 1216 --commit <sha> --branch <ref> --profile dummy_layers_model

All arguments are optional and fall back to Jenkins environment variables
(``CHANGE_ID``, ``GIT_COMMIT``, ``BRANCH_NAME``, ``TEST_PROFILE``, ``BUILD_URL``).
"""

import argparse
import enum
import glob
import html
import os
import re
import sys
import traceback
import xml.etree.ElementTree as ET
from collections import Counter, OrderedDict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

# ── Constants ────────────────────────────────────────────────────────────────

MERGED_XML = "tests_log.xml"  # aggregate of all per-stage files; parsed only as a fallback
MAX_LOG_CHARS = 20_000  # cap captured stdout/log per test to keep the HTML bounded
MAX_TB_CHARS = 40_000  # cap tracebacks (kept longer — they matter most for fixing)
ANSI_RE = re.compile(r"\x1b\[[0-9;]*[A-Za-z]|\x1b\][^\x07]*\x07")


class Outcome(enum.Enum):
    """Test outcome. The value doubles as a CSS-friendly status token."""

    PASSED = "passed"
    FAILED = "failed"
    ERROR = "error"
    SKIPPED = "skipped"
    XFAIL = "xfail"


# Short label + badge CSS class per outcome.
BADGE = {
    Outcome.PASSED: ("PASS", "b-pass"),
    Outcome.FAILED: ("FAIL", "b-fail"),
    Outcome.ERROR: ("ERROR", "b-err"),
    Outcome.SKIPPED: ("SKIP", "b-skip"),
    Outcome.XFAIL: ("XFAIL", "b-xfail"),
}


@dataclass
class StageSpec:
    """Static metadata for a CI stage (declared in ``STAGE_MAP``)."""

    display: str
    marker_expr: str  # verbatim pytest -m expression; <PROFILE> is expanded at render time
    gate: str  # the Jenkins boolean param that gates this stage
    order: int


# Full 10-stage roster in pipeline order, keyed by the per-stage JUnit XML basename.
# This is the source of truth for the stage list, so absent files render as "Not Run".
STAGE_MAP = OrderedDict(
    [
        (
            "tests_log1.xml",
            StageSpec("HL API: Export & Compile", "(not on_qaic) and (not finetune) and <PROFILE>", "RUN_HL_APIS", 1),
        ),
        ("tests_log2.xml", StageSpec("HL API: QAIC LLM", "(llm_model) and (not qnn) and <PROFILE>", "RUN_HL_APIS", 2)),
        (
            "tests_log2_feature.xml",
            StageSpec("QAIC Feature", "(on_qaic) and (feature) and (not qnn) and <PROFILE>", "RUN_QAIC_FEATURE", 3),
        ),
        (
            "tests_log_embedding_audio.xml",
            StageSpec(
                "QAIC Embedding & Audio",
                "(embedding_audio_model) and (not qnn) and <PROFILE>",
                "RUN_QAIC_EMBEDDING_AUDIO",
                4,
            ),
        ),
        ("tests_log6.xml", StageSpec("QAIC Multimodal", "(multimodal) and (not qnn) and <PROFILE>", "RUN_QAIC_MM", 5)),
        (
            "tests_log_reranker.xml",
            StageSpec("QAIC Reranker", "tests/transformers/models/reranker/test_reranker_mad.py", "RUN_QAIC_MM", 6),
        ),
        ("tests_log_diffusion.xml", StageSpec("QAIC Diffusion", "diffusion_models", "RUN_QAIC_DIFFUSION", 7)),
        ("tests_log3.xml", StageSpec("CLI", "(cli and not qnn) and (not finetune)", "RUN_CLI", 8)),
        (
            "tests_log_dynamo_qaic.xml",
            StageSpec("QAIC Dynamo", "(dynamo) and (on_qaic) and <PROFILE>", "RUN_DYNAMO_QAIC", 9),
        ),
        ("tests_log_finetune.xml", StageSpec("Finetune", "(finetune)", "RUN_FINETUNE", 10)),
    ]
)


@dataclass
class TestCase:
    """A single test execution parsed from a JUnit ``<testcase>`` element."""

    nodeid: str
    module_path: str  # readable file path, e.g. tests/.../test_x.py
    test_fn: str  # function name up to the first '['
    param_id: str  # text inside [...] (model / config); "" if unparametrized
    outcome: Outcome
    duration: float
    message: str  # short one-line failure/error/skip message
    detail: str  # traceback / failure body
    captured: str  # combined system-out + system-err
    stage: str  # owning stage display name
    # True when the row was synthesized from the pytest slowest-N block (no
    # verbose ``[gw#]`` line existed for it). Such rows carry a real duration
    # but their PASSED status is a placeholder — we don't know whether the
    # test actually passed, only that pytest measured its duration. Rollups
    # (By-model, Feature matrix) must skip these to avoid inflating pass counts;
    # per-stage detail + Slowest tables still show them.
    seeded: bool = False


@dataclass
class Stage:
    """A CI stage plus every test it ran."""

    spec: StageSpec
    ran: bool = False
    parse_error: str = ""
    cases: list = field(default_factory=list)
    suite_time: float = 0.0
    # Authoritative per-outcome totals when we can't enumerate every test (e.g. reading a
    # non-verbose console log, where only the pytest summary line lists exact counts). When
    # set, ``counts`` returns this Counter instead of re-tallying ``cases``.
    summary_counts: Counter = None
    # Optional caveat rendered under the stage-detail header (e.g. "only slowest-N tests
    # are attributable — full outcomes came from the pytest summary line").
    note: str = ""

    @property
    def counts(self):
        if self.summary_counts is not None:
            return self.summary_counts
        c = Counter()
        for tc in self.cases:
            c[tc.outcome] += 1
        return c

    @property
    def slug(self):
        return f"s-{self.spec.order}"

    @property
    def failed(self):
        c = self.counts
        return c[Outcome.FAILED] + c[Outcome.ERROR] > 0


@dataclass
class Report:
    """The full parsed run: every stage (ran or not) plus run metadata."""

    stages: list
    meta: dict
    merged_fallback: bool = False

    @property
    def all_cases(self):
        for st in self.stages:
            yield from st.cases

    @property
    def totals(self):
        c = Counter()
        for st in self.stages:
            if not st.ran:
                continue
            for outcome, n in st.counts.items():
                c[outcome] += n
        return c

    @property
    def ran_stages(self):
        return [st for st in self.stages if st.ran]

    @property
    def wall_time(self):
        return sum(st.suite_time for st in self.ran_stages)

    @property
    def verdict(self):
        totals = self.totals
        if sum(totals.values()) == 0:
            return "NO TESTS"
        if totals[Outcome.FAILED] + totals[Outcome.ERROR] > 0:
            return "FAIL"
        return "PASS"

    @property
    def pass_rate(self):
        totals = self.totals
        denom = totals[Outcome.PASSED] + totals[Outcome.FAILED] + totals[Outcome.ERROR]
        if denom == 0:
            return None
        return 100.0 * totals[Outcome.PASSED] / denom


# ── Helpers ──────────────────────────────────────────────────────────────────


def strip_ansi(text):
    """Remove ANSI colour / control escape sequences and normalise carriage returns."""
    return ANSI_RE.sub("", text or "").replace("\r\n", "\n").replace("\r", "\n")


def esc(text):
    """HTML-escape a string (quotes included)."""
    return html.escape(text or "", quote=True)


def truncate(text, limit):
    """Truncate ``text`` to ``limit`` chars with a marker noting how much was cut."""
    if text is None:
        return ""
    if len(text) <= limit:
        return text
    return text[:limit] + f"\n... (truncated {len(text) - limit} more chars)"


def fmt_duration(seconds):
    """Render a float seconds value as a compact human string (e.g. 2h 18m, 47s)."""
    seconds = int(round(seconds))
    if seconds >= 3600:
        return f"{seconds // 3600}h {(seconds % 3600) // 60:02d}m"
    if seconds >= 60:
        return f"{seconds // 60}m {seconds % 60:02d}s"
    return f"{seconds}s"


def expand_profile_filter(profile):
    """Mirror the Jenkinsfile ``testFilter()`` switch so displayed markers match the run."""
    return {
        "dummy_layers_model": "(not full_layers) and (not few_layers)",
        "few_layers_model": "(not full_layers) and (not dummy_layers)",
        "full_layers_model": "(not dummy_layers) and (not few_layers)",
    }.get(profile, "<profile filter>")


def split_nodeid(classname, name):
    """Derive (module_path, test_fn, param_id) from JUnit classname/name attributes.

    * ``classname`` is a dotted module path (e.g. ``tests.foo.test_x``); we render it
      as ``tests/foo/test_x.py``.
    * ``name`` is ``test_fn[param]``; the param (between the first ``[`` and last ``]``)
      is the model / config id and may itself contain slashes or ``-`` separators.
    """
    module_path = classname.replace(".", "/") + ".py" if classname else ""
    test_fn = name
    param_id = ""
    if "[" in name and name.endswith("]"):
        test_fn = name[: name.index("[")]
        param_id = name[name.index("[") + 1 : -1]
    nodeid = f"{module_path}::{name}" if module_path else name
    return nodeid, module_path, test_fn, param_id


_NO_MODEL = "(no model param)"

# Parametrize enums / flags that must never be mistaken for a model token.
# These are prefixes-of / exact-matches-on hyphen-separated segments — a segment
# either matches exactly (``cb``, ``nocb``) or starts with the given prefix
# followed by digits (``kwargs4``, ``decode_ks0``). Keep this list conservative:
# too eager and we hide a genuine model slug; too lax and we surface junk.
_NON_MODEL_EXACT = frozenset(
    {
        "cb",
        "nocb",
        "subfunc",
        "non-subfunc",
        "non_subfunc",
        "pref+decode",
        "prefill_only",
        "prefill-only",
        "blocking",
        "non-blocking",
        "non_blocking",
        "qv",
        "bs4",
        "nan",
        "inf",
        "-inf",
        "true",
        "false",
    }
)
_NON_MODEL_PREFIXES = ("kwargs", "decode_ks", "actual_proposals", "torch_dtype")


def _is_number_token(token):
    """Return True when ``token`` is a pure integer or float literal (any sign)."""
    if not token:
        return False
    try:
        float(token)
        return True
    except ValueError:
        return False


def _is_non_model_token(token):
    """Return True when ``token`` is a parametrize enum / flag — never a model slug."""
    t = token.lower()
    if t in _NON_MODEL_EXACT:
        return True
    if _is_number_token(t):
        return True
    for prefix in _NON_MODEL_PREFIXES:
        if t.startswith(prefix) and t[len(prefix) :].lstrip("-_").isdigit():
            return True
    return False


def normalize_model(param_id):
    """Reduce a parametrize id to its model / config token for the by-model rollup.

    Rules, in order (first match wins):

    * Empty param_id ⇒ ``(no model param)``.
    * A ``module=<Class>-…`` shape (RMSNorm / ops-transform tests) ⇒ the value of
      ``module=`` — keeps rows aligned on the *component under test*.
    * A ``/`` anywhere ⇒ the full param_id verbatim; HF card paths (and stacked
      LoRA/PEFT ids that pin multiple cards) are preserved as-is.
    * Otherwise, take the FIRST ``-`` segment that isn't a pytest parametrize
      enum (``cb``/``nocb``/``subfunc``/``kwargs<n>``/…) or a pure number. This
      is the family slug (``starcoder2`` in ``starcoder2-cb``, ``llama`` in
      ``llama-3-32-128-kwargs0-0.8``). Fall back to ``(no model param)`` when
      every segment is a flag.
    """
    if not param_id:
        return _NO_MODEL

    first_segment = param_id.split("-", 1)[0]
    if first_segment.startswith("module="):
        return first_segment[len("module=") :] or _NO_MODEL

    # Strip leading enum/flag segments (``True``/``False``/``cb``/``nocb``/…) so
    # boolean prefixes from ``@pytest.mark.parametrize("full_batch", [True, False])``
    # don't split one model across two rows.
    segments = param_id.split("-")
    while segments and _is_non_model_token(segments[0]):
        segments = segments[1:]
    # Trailing bool/flag suffixes (``…-True``/``…-False``) leak in the same way for
    # HF-card param ids (``llava-hf/llava-1.5-7b-hf-True``); strip them too so the
    # by-model rollup, Dense/MoE oracle, and coverage tiles see one row per card.
    while len(segments) > 1 and _is_non_model_token(segments[-1]):
        segments = segments[:-1]
    trimmed = "-".join(segments)

    if "/" in trimmed:
        return trimmed
    if not trimmed:
        return _NO_MODEL

    for token in segments:
        if not _is_non_model_token(token):
            return token
    return _NO_MODEL


# ── Model categories ─────────────────────────────────────────────────────────

# Fixed roster of the model categories a per-PR run exercises, in display order.
# Each entry is (path_substring, label). A testcase is assigned to the FIRST entry
# whose substring appears in its module_path, so this is the source of truth for
# the category list: a category with zero tests in a run still renders (as "Not
# Run" / "0 tests"), exactly like STAGE_MAP does for stages, so a silent coverage
# gap can never masquerade as a pass.
#
# module_path is the reliable key (it survives the XML, merged-XML and console-log
# load paths); marker/stage are NOT — reranker carries the ``multimodal`` marker
# yet runs as its own stage, and sequence-classification carries ``llm_model`` and
# runs in the QAIC-LLM stage, so grouping on either would misfile both. Order
# matters only where one path is a substring of another; the current set is
# disjoint. Anything unmatched (CLI, finetune, unit/infra tests) falls to OTHER.
CATEGORY_OTHER = "Other"
# Category labels as named constants so the per-category Dense/MoE oracle and the
# renderers reference them by name rather than by repeated string literals.
CAT_CAUSAL = "Causal LM"
CAT_VLM = "Vision-Language (VLM)"
CAT_EMBEDDING = "Embedding"
CAT_AUDIO = "Audio"
CAT_SEQ_RERANKER = "Sequence / Reranker"
CAT_DIFFUSION = "Diffusion"

CATEGORY_ROSTER = [
    ("models/causal_lm_models/", CAT_CAUSAL),
    ("models/image_text_to_text/", CAT_VLM),
    ("models/embedding_models/", CAT_EMBEDDING),
    ("models/audio_models/", CAT_AUDIO),
    ("models/sequence_models/", CAT_SEQ_RERANKER),
    ("models/reranker/", CAT_SEQ_RERANKER),
    ("/diffusers/", CAT_DIFFUSION),
]

# Display order for categories, de-duplicated from the roster (Sequence / Reranker
# maps two paths to one label). OTHER is appended last and rendered only when it
# actually holds rows: it is the reconciling bucket for tests that belong to no
# model category (CLI, finetune, unit/infra, Dynamo) so the per-tile totals in the
# Coverage section always sum to the KPI Total. Without it a reviewer reading the
# tiles would conclude those tests never ran.
CATEGORY_ORDER = list(OrderedDict((label, None) for _sub, label in CATEGORY_ROSTER))
COVERAGE_CATEGORY_ORDER = CATEGORY_ORDER + [CATEGORY_OTHER]


def category_of(module_path):
    """Return the model-category label for a test's ``module_path``.

    Matches the first :data:`CATEGORY_ROSTER` entry whose path substring occurs in
    ``module_path``; returns :data:`CATEGORY_OTHER` for anything unmatched (CLI,
    finetune, unit/infra tests) or an empty path (console rows collapsed to
    module-level). Never raises — an unknown path is a bucket, not an error.
    """
    path = module_path or ""
    for substring, label in CATEGORY_ROSTER:
        if substring in path:
            return label
    return CATEGORY_OTHER


# ── Feature coverage ───────────────────────────────────────────────────────────

# Canonical column order for the model × feature coverage matrix. Each entry is
# (key, display, predicate); the predicate receives (fn_tokens, param_tokens) —
# already lower-cased sets of ``-``-separated segments plus the raw joined strings
# — and returns True when that test exercises the feature. Features are read
# from the per-PR causal-LM test-function names *and* their parametrize ids
# (``test_causal_lm_init[starcoder2-cb]`` is a CB test even though its function
# name has no ``cb`` in it), so the matrix reflects the actual run — no curated
# data source.
FEATURE_COLUMNS = [
    ("fp32", "FP32", lambda fn, fn_tok, p, p_tok: "fp32" in fn),
    ("fp16", "FP16", lambda fn, fn_tok, p, p_tok: "fp16" in fn),
    ("bf16", "BF16", lambda fn, fn_tok, p, p_tok: "bf16" in fn),
    # CB is set when a ``cb`` segment is present in fn OR param, AND ``nocb`` is not.
    (
        "cb",
        "Cont. Batch",
        lambda fn, fn_tok, p, p_tok: ("cb" in fn_tok or "cb" in p_tok) and "nocb" not in fn_tok and "nocb" not in p_tok,
    ),
    ("ccl", "CCL", lambda fn, fn_tok, p, p_tok: "ccl" in fn_tok or "ccl" in p_tok),
    (
        "subfunction",
        "Subfunction",
        lambda fn, fn_tok, p, p_tok: (
            ("subfunction" in fn or "subfunc" in fn_tok or "subfunc" in p_tok)
            and "non-subfunc" not in p
            and "non_subfunc" not in p
        ),
    ),
    (
        "prefix_caching",
        "Prefix Cache",
        lambda fn, fn_tok, p, p_tok: "prefix_caching" in fn or "prefix_caching" in p,
    ),
    (
        "blocking",
        "Blocking",
        lambda fn, fn_tok, p, p_tok: (
            ("blocking" in fn or "blocking" in p) and "non-blocking" not in p and "non_blocking" not in p
        ),
    ),
    ("disagg", "Disagg", lambda fn, fn_tok, p, p_tok: "disagg" in fn or "disagg" in p),
    (
        "spd",
        "SPD/PLD",
        lambda fn, fn_tok, p, p_tok: "speculative" in fn or "_tlm" in fn or "tlm" in p_tok,
    ),
    (
        "compile_only",
        "Compile-only",
        lambda fn, fn_tok, p, p_tok: "compile_only" in fn or "compile_only" in p,
    ),
]

# Per-PR end-to-end scenario columns for the Scenario coverage matrix. Unlike
# FEATURE_COLUMNS (atomic capabilities), each column here is ONE per-PR test
# function that pins a whole dtype+subfunction+CB+feature combination, so a
# single cell answers "did this scenario pass on this model?". Keyed on the
# exact test-function name (stable; verified against test_causal_lm_models.py).
# Display labels state ONLY each scenario's delta from the shared base
# (causal · subfunction · CB · FP16 export), which the matrix caption spells out
# once. A leading "+" means "baseline plus this feature"; labels without "+" pin a
# different dtype/mode. The exact test-function name still rides in the cell tooltip.
SCENARIO_COLUMNS = [
    ("test_per_pr_causal_fp16_subfunction_cb", "Baseline"),
    ("test_per_pr_causal_fp16_subfunction_cb_prefix_caching", "+ Prefix cache"),
    ("test_per_pr_causal_fp16_subfunction_cb_ccl", "+ CCL"),
    ("test_per_pr_causal_fp16_subfunction_cb_blocking", "+ Blocking"),
    ("test_per_pr_causal_fp32_export_fp16_compile_subfunction_cb_ccl", "FP32 export · CCL"),
    ("test_per_pr_causal_bf16_subfunction_cb_ccl_compile_only", "BF16 · CCL · compile-only"),
    ("test_per_pr_causal_moe_disagg_fp16_subfunction_cb_ccl", "MoE disagg · CCL"),
    ("test_per_pr_causal_speculative_tlm_fp16_subfunction_cb", "+ Speculation (TLM)"),
]

# VLM per-PR scenario columns. VLM test functions are split across three per-PR profile
# variants (``test_full_*``, ``test_few_*``, ``test_dummy_*``) that pin the same scenario
# to different layer-count profiles. The report should show one column per scenario
# regardless of which profile the run picked, so cells key on the *profile-stripped*
# test_fn (see :func:`_vlm_scenario_key`). Columns cover baseline parity + continuous
# batching + prefix caching + CCL / blocking dual-QPC + bf16 / MoE-MDP compile-only.
VLM_SCENARIO_COLUMNS = [
    ("image_text_to_text_pytorch_vs_kv_vs_ort_vs_ai100", "Baseline"),
    ("image_text_to_text_pytorch_vs_ai100_continuous_batching", "+ CB"),
    ("image_text_to_text_prefix_caching_cb", "+ Prefix cache CB"),
    ("image_text_to_text_ccl_dual_qpc", "+ CCL dual-QPC"),
    ("image_text_to_text_blocking_dual_qpc", "+ Blocking dual-QPC"),
    ("image_text_to_text_bf16_compile_only", "BF16 · compile-only"),
    ("image_text_to_text_onnx_mdp_compile_only", "MoE MDP · compile-only"),
]

# Per-PR profile prefixes stripped from a VLM test-fn to derive its scenario key.
# ``test_full_image_text_to_text_ccl_dual_qpc`` and ``test_dummy_image_text_to_text_ccl_dual_qpc``
# both collapse to ``image_text_to_text_ccl_dual_qpc`` so they share one column.
_VLM_TESTFN_PROFILE_PREFIXES = ("test_full_", "test_few_", "test_dummy_")


def _vlm_scenario_key(test_fn):
    """Strip the per-PR profile prefix from a VLM test-fn to get its scenario key.

    Returns the empty string when ``test_fn`` doesn't carry one of the known VLM
    profile prefixes (so unrelated VLM tests — reference/qnn/custom — don't collide
    with a curated scenario column).
    """
    for prefix in _VLM_TESTFN_PROFILE_PREFIXES:
        if test_fn.startswith(prefix):
            return test_fn[len(prefix) :]
    return ""


# Scenario-column keys where a pass means "compiled" (no on-device output verification):
# BF16/FP32 export lanes, ONNX-MDP compile-only, etc. Rendered with a distinct
# ``compile-only`` badge so a green cell here is not mistaken for parity confirmation.
# Keys are the SCENARIO / VLM_SCENARIO column keys (test-fn or profile-stripped test-fn).
_COMPILE_ONLY_SCENARIOS = frozenset(
    {
        "test_per_pr_causal_bf16_subfunction_cb_ccl_compile_only",
        "test_per_pr_causal_fp32_export_fp16_compile_subfunction_cb_ccl",
        "image_text_to_text_bf16_compile_only",
        "image_text_to_text_onnx_mdp_compile_only",
    }
)

# Severity order for reducing many test outcomes in one (model, feature) cell to a single
# status: a red cell always wins over a green one for the same capability.
_CELL_SEVERITY = {
    Outcome.ERROR: 4,
    Outcome.FAILED: 4,
    Outcome.XFAIL: 3,
    Outcome.SKIPPED: 2,
    Outcome.PASSED: 1,
}


def classify_features(tc):
    """Return the set of feature keys a test exercises, inferred from its name.

    We split both ``test_fn`` and ``param_id`` on ``-`` and pass every
    predicate both the raw strings and the segment sets, so features
    encoded in either dimension are picked up (per-PR causal-LM tests
    encode features in ``test_fn``; unit tests encode them in ``param_id``).
    """
    fn = tc.test_fn.lower()
    p = tc.param_id.lower()
    fn_tokens = set(re.split(r"[-_]+", fn)) if fn else set()
    p_tokens = set(re.split(r"[-_]+", p)) if p else set()
    return {key for key, _display, pred in FEATURE_COLUMNS if pred(fn, fn_tokens, p, p_tokens)}


# ── Parsing ──────────────────────────────────────────────────────────────────


def classify(tc_elem):
    """Return (Outcome, message, detail) for a JUnit ``<testcase>`` element."""
    err = tc_elem.find("error")
    if err is not None:
        return Outcome.ERROR, err.get("message", ""), err.text or ""
    fail = tc_elem.find("failure")
    if fail is not None:
        return Outcome.FAILED, fail.get("message", ""), fail.text or ""
    skip = tc_elem.find("skipped")
    if skip is not None:
        typ = (skip.get("type") or "").lower()
        msg = skip.get("message") or ""
        if "xfail" in typ or "xfail" in msg.lower():
            return Outcome.XFAIL, msg, skip.text or ""
        return Outcome.SKIPPED, msg, skip.text or ""
    return Outcome.PASSED, "", ""


def parse_testcase(tc_elem, stage_display):
    """Build a :class:`TestCase` from a JUnit ``<testcase>`` element."""
    classname = tc_elem.get("classname", "")
    name = tc_elem.get("name", "")
    nodeid, module_path, test_fn, param_id = split_nodeid(classname, name)
    outcome, message, detail = classify(tc_elem)
    try:
        duration = float(tc_elem.get("time", "0") or "0")
    except ValueError:
        duration = 0.0
    captured_parts = []
    for tag in ("system-out", "system-err"):
        el = tc_elem.find(tag)
        if el is not None and el.text:
            captured_parts.append(strip_ansi(el.text))
    return TestCase(
        nodeid=nodeid,
        module_path=module_path,
        test_fn=test_fn,
        param_id=param_id,
        outcome=outcome,
        duration=duration,
        message=strip_ansi(message),
        detail=strip_ansi(detail),
        captured="\n".join(captured_parts),
        stage=stage_display,
    )


def parse_stage_file(path, stage):
    """Populate ``stage`` from a per-stage JUnit XML file, deduping by nodeid (last wins)."""
    try:
        root = ET.parse(path).getroot()
    except ET.ParseError as exc:
        stage.parse_error = str(exc)
        stage.ran = True
        return

    suites = root.iter("testsuite")
    by_nodeid = OrderedDict()
    suite_time = 0.0
    for suite in suites:
        try:
            suite_time += float(suite.get("time", "0") or "0")
        except ValueError:
            pass
        for tc_elem in suite.findall("testcase"):
            tc = parse_testcase(tc_elem, stage.spec.display)
            if tc.nodeid in by_nodeid:
                print(f"warning: duplicate nodeid in {os.path.basename(path)}: {tc.nodeid}", file=sys.stderr)
            by_nodeid[tc.nodeid] = tc
    stage.cases = list(by_nodeid.values())
    stage.suite_time = suite_time
    stage.ran = True


def load_report(xml_dir, pattern, meta):
    """Discover per-stage XML in ``xml_dir`` and build the full :class:`Report`."""
    found = {os.path.basename(p): p for p in glob.glob(os.path.join(xml_dir, pattern))}
    stages = [Stage(spec=spec) for spec in STAGE_MAP.values()]
    per_stage_present = any(name in found for name in STAGE_MAP)

    if per_stage_present:
        for basename, stage in zip(STAGE_MAP, stages):
            if basename in found:
                parse_stage_file(found[basename], stage)
        return Report(stages=stages, meta=meta)

    # Fallback: only the merged file exists → single synthetic stage, provenance lost.
    if MERGED_XML in found:
        merged = Stage(spec=StageSpec("All Tests (merged)", "merged tests_log.xml", "", 1))
        parse_stage_file(found[MERGED_XML], merged)
        return Report(stages=[merged], meta=meta, merged_fallback=True)

    return Report(stages=stages, meta=meta)


# ── Console-log fallback ────────────────────────────────────────────────────

# Reads the Jenkins console log directly when per-stage JUnit XML isn't available.
# We can only recover partial per-test attribution from console output — xdist-verbose
# stages emit `[gw#] STATUS tests/...::name[param]` lines we can parse, but non-verbose
# stages emit progress dots. For those, the pytest summary line is the source of truth
# for totals and the `slowest N durations` block seeds a small set of attributable
# TestCase rows.
#
# Segmentation is anchored on ``generated xml file: .../<basename>.xml`` markers rather
# than on ``test session starts`` banners because the Jenkinsfile can launch multiple
# pytest processes concurrently, so ``test session starts`` banners for different
# stages can interleave in the log. The xml-file marker is unambiguous: each pytest
# process prints exactly one, and pytest's own slowest-N + summary lines follow
# immediately after — always contiguous, never interleaved.

_GW_LINE_RE = re.compile(r"^\[gw\d+\] (PASSED|FAILED|ERROR|SKIPPED|XFAIL|XPASS) (\S+)$")
# Plain (non-xdist) `pytest -v` output on a non-TTY writes the nodeid first with the status
# still pending, lets the test emit its stdout, then prints the status alone on a later line.
# Short-circuiting outcomes (e.g. an xfail raised at collection) print `nodeid STATUS` inline.
# Both shapes are matched here; the trailing status group is optional.
#
# The nodeid is anchored at line start but the rest of the line is NOT required to be empty:
# a library warning emitted during collection (``\`torch_dtype\` is deprecated!``, an HF
# "new version downloaded" notice) lands on the same line, and an end-anchored pattern would
# drop that test entirely. ``_nodeid_from_line`` re-balances ``[...]`` before trimming so
# parametrize ids containing spaces (``test_disagg_mode_prefill[Once upon a time-...]``) are
# not truncated at the first space.
_NODEID_LINE_RE = re.compile(r"^(tests/\S+::\S+?)(?:\s+(PASSED|FAILED|SKIPPED|XFAIL|XPASS|ERROR))?(?:\s|$)")
_BARE_STATUS_RE = re.compile(r"^(PASSED|FAILED|SKIPPED|XFAIL|XPASS|ERROR)\s*$")
_SUMMARY_RE = re.compile(r"^=+ (?P<body>.*?) in (?P<time>[0-9.]+)s(?: \([0-9hms: ]+\))? =+$")
_COUNT_TOKEN_RE = re.compile(r"(\d+)\s+(passed|failed|error|errors|skipped|xfailed|xpassed|deselected)")
_SLOW_LINE_RE = re.compile(r"^([0-9.]+)s\s+(?:call|setup|teardown)\s+(\S+)$")
_XML_GEN_RE = re.compile(r"generated xml file:.*?/([A-Za-z0-9_]+\.xml)")
_SLOW_HEADER_RE = re.compile(r"^=+\s*slowest\s+\d+\s+durations\s*=+$")

_CONSOLE_STATUS_TO_OUTCOME = {
    "PASSED": Outcome.PASSED,
    "FAILED": Outcome.FAILED,
    "ERROR": Outcome.ERROR,
    "SKIPPED": Outcome.SKIPPED,
    "XFAIL": Outcome.XFAIL,
    "XPASS": Outcome.PASSED,  # unexpected pass — treat as pass; XFail totals are surfaced separately
}

_SUMMARY_TOKEN_TO_OUTCOME = {
    "passed": Outcome.PASSED,
    "failed": Outcome.FAILED,
    "error": Outcome.ERROR,
    "errors": Outcome.ERROR,
    "skipped": Outcome.SKIPPED,
    "xfailed": Outcome.XFAIL,
    # xpassed / deselected are intentionally not folded into any Outcome bucket.
}


def _testcase_from_nodeid(nodeid, outcome, duration, stage_display, message="", seeded=False):
    """Build a TestCase from a bare pytest nodeid ("tests/foo/test_x.py::name[param]")."""
    module_path, _, name = nodeid.partition("::")
    if not name:  # be defensive: some rows are collapsed to module-level
        name = module_path
        module_path = ""
    test_fn = name
    param_id = ""
    if "[" in name and name.endswith("]"):
        test_fn = name[: name.index("[")]
        param_id = name[name.index("[") + 1 : -1]
    return TestCase(
        nodeid=nodeid,
        module_path=module_path,
        test_fn=test_fn,
        param_id=param_id,
        outcome=outcome,
        duration=duration,
        message=message,
        detail="",
        captured="",
        stage=stage_display,
        seeded=seeded,
    )


def _find_xml_markers(lines):
    """Return an ordered list of ``(line_idx, xml_basename)`` for every xml-file marker."""
    markers = []
    for i, ln in enumerate(lines):
        m = _XML_GEN_RE.search(ln)
        if m:
            markers.append((i, m.group(1)))
    return markers


def _parse_summary_line(text):
    """Return (Counter of outcomes, wall seconds) from a pytest summary line, or (None, None)."""
    m = _SUMMARY_RE.match(text.strip())
    if not m:
        return None, None
    counts = Counter()
    for n, tok in _COUNT_TOKEN_RE.findall(m.group("body")):
        outcome = _SUMMARY_TOKEN_TO_OUTCOME.get(tok)
        if outcome is not None:
            counts[outcome] += int(n)
    try:
        wall = float(m.group("time"))
    except ValueError:
        wall = 0.0
    return counts, wall


def _parse_stage_trailer(lines, marker_idx):
    """Scan the ~40 lines after an xml marker for that stage's slowest-N block + summary.

    Pytest prints ``generated xml file: ... -- slowest N durations -- summary`` as one
    contiguous trailer, so we don't need to guess where it ends: the summary line is the
    signal we stop on.
    """
    slow_durations = OrderedDict()
    summary_counts = None
    wall_time = 0.0
    in_slow = False
    for i in range(marker_idx + 1, min(marker_idx + 200, len(lines))):
        raw = strip_ansi(lines[i]).rstrip()
        if _SLOW_HEADER_RE.match(raw):
            in_slow = True
            continue
        counts, wall = _parse_summary_line(raw)
        if counts is not None:
            summary_counts = counts
            wall_time = wall or 0.0
            break
        if in_slow:
            m = _SLOW_LINE_RE.match(raw)
            if m:
                dur = float(m.group(1))
                nodeid = m.group(2)
                if nodeid not in slow_durations or dur > slow_durations[nodeid]:
                    slow_durations[nodeid] = dur
                continue
            if raw and not raw.startswith("="):
                in_slow = False
    return slow_durations, summary_counts, wall_time


def _nodeid_from_line(raw):
    """Return ``(nodeid, inline_status)`` for a plain ``pytest -v`` line, or ``(None, None)``.

    ``_NODEID_LINE_RE`` stops the nodeid at the first whitespace, which truncates parametrize
    ids that contain spaces (``test_disagg_mode_prefill[Once upon a time-openai/gpt-oss-20b]``).
    When the captured nodeid has an unclosed ``[``, we extend it to the matching ``]`` on the
    same line so the full id is kept; anything after that bracket is trailing noise (a library
    warning printed on the nodeid's line) and is discarded unless it is a status token.
    """
    m = _NODEID_LINE_RE.match(raw)
    if not m or "::" not in m.group(1):
        return None, None
    nodeid, inline_status = m.group(1), m.group(2)
    if nodeid.count("[") > nodeid.count("]"):
        close = raw.find("]", len(nodeid))
        if close != -1:
            nodeid = raw[: close + 1]
            rest = raw[close + 1 :].strip()
            inline_status = rest if _BARE_STATUS_RE.match(rest) else None
    return nodeid, inline_status


def _scan_plain_verbose_cases(lines, start, end):
    """Return ``{nodeid: status}`` from plain (non-xdist) ``pytest -v`` output in a range.

    Non-TTY verbose pytest splits one test across two lines: the nodeid (status pending)
    and, after the test's own stdout, the status alone. We therefore hold the most recent
    unresolved nodeid and bind it to the next standalone status line. A nodeid carrying an
    inline status resolves immediately, and encountering a second nodeid abandons any still
    unresolved one — output that never produced a status line (killed/truncated run) is
    dropped rather than mis-bound to a later test's result.
    """
    statuses = OrderedDict()
    pending = None
    for i in range(start, end):
        raw = strip_ansi(lines[i]).rstrip()
        nodeid, inline_status = _nodeid_from_line(raw)
        if nodeid:
            if inline_status:
                statuses.setdefault(nodeid, inline_status)
                pending = None
            else:
                pending = nodeid
            continue
        if pending:
            sm = _BARE_STATUS_RE.match(raw)
            if sm:
                statuses.setdefault(pending, sm.group(1))
                pending = None
    return statuses


def _scan_verbose_cases(lines, start, end, stage_display):
    """Collect per-test cases from a line range, from either verbose pytest format.

    ``[gw#] STATUS nodeid`` (xdist) is authoritative; plain ``pytest -v`` nodeid/status
    pairs fill in stages that ran without xdist (the QAIC stages in the reference log emit
    only this shape, so their per-test detail was previously lost entirely). A nodeid seen
    in the xdist form is never re-added from the plain form, so the two cannot double-count.
    """
    cases = []
    seen = set()
    for i in range(start, end):
        raw = strip_ansi(lines[i]).rstrip()
        m = _GW_LINE_RE.match(raw)
        if not m:
            continue
        status, nodeid = m.group(1), m.group(2)
        outcome = _CONSOLE_STATUS_TO_OUTCOME.get(status)
        if outcome is None or nodeid in seen:
            continue
        seen.add(nodeid)
        cases.append(_testcase_from_nodeid(nodeid, outcome, 0.0, stage_display))

    for nodeid, status in _scan_plain_verbose_cases(lines, start, end).items():
        outcome = _CONSOLE_STATUS_TO_OUTCOME.get(status)
        if outcome is None or nodeid in seen:
            continue
        seen.add(nodeid)
        cases.append(_testcase_from_nodeid(nodeid, outcome, 0.0, stage_display))
    return cases


def load_report_from_console(log_path, meta):
    """Build a :class:`Report` by parsing a Jenkins console log directly.

    The log is expected to contain one or more pytest sessions, each terminated by a
    ``generated xml file: .../<basename>.xml`` marker whose basename matches a key in
    ``STAGE_MAP``. Stages without a matching marker render as "Not Run", same as the
    XML path. Per-test lines are attributed to whichever stage's xml marker they precede,
    and both verbose shapes are read: xdist ``[gw#] STATUS nodeid`` and plain ``pytest -v``
    nodeid/status pairs. The slowest-N + summary trailer sits contiguously after each
    marker, so stage totals stay exact even when two Jenkins-launched pytest processes
    interleave their output.
    """
    with open(log_path, encoding="utf-8", errors="replace") as fh:
        lines = fh.readlines()

    stage_by_basename = OrderedDict((name, Stage(spec=spec)) for name, spec in STAGE_MAP.items())
    markers = _find_xml_markers(lines)
    unknown_basenames = []
    prev_end = 0
    for marker_idx, basename in markers:
        if basename not in stage_by_basename:
            unknown_basenames.append(basename)
            continue
        stage = stage_by_basename[basename]
        stage.ran = True
        slow_durations, summary_counts, wall_time = _parse_stage_trailer(lines, marker_idx)
        stage.suite_time = wall_time

        verbose_cases = _scan_verbose_cases(lines, prev_end, marker_idx, stage.spec.display)
        # Merge slowest-N durations into any verbose case matching the same nodeid.
        for tc in verbose_cases:
            if tc.nodeid in slow_durations:
                tc.duration = slow_durations[tc.nodeid]

        # Also seed cases from slowest-N entries that didn't appear as per-test lines —
        # useful for non-verbose stages where slowest-N is the only per-test signal.
        seen_ids = {tc.nodeid for tc in verbose_cases}
        seeded = list(verbose_cases)
        for nodeid, dur in slow_durations.items():
            if nodeid in seen_ids:
                continue
            seeded.append(_testcase_from_nodeid(nodeid, Outcome.PASSED, dur, stage.spec.display, seeded=True))
            seen_ids.add(nodeid)
        stage.cases = seeded

        if summary_counts is not None:
            verbose_c = Counter(tc.outcome for tc in verbose_cases)
            matches = all(verbose_c[o] == summary_counts.get(o, 0) for o in Outcome)
            if not matches:
                stage.summary_counts = summary_counts
                total = sum(summary_counts.values())
                attributable = len(verbose_cases)
                seeded_extra = len(seeded) - attributable
                unattributed = max(0, total - attributable - seeded_extra)
                stage.note = (
                    f"Console log per-test attribution is partial: {total} tests in the summary but only "
                    f"{attributable} attributable from per-test verbose lines "
                    f"({seeded_extra} additional seeded from the slowest-10 block, "
                    f"{unattributed} un-attributable). Totals in the top summary come "
                    "from pytest's own summary line."
                )
        elif not verbose_cases and not slow_durations:
            stage.parse_error = "no pytest summary line found near xml-file marker"

        prev_end = marker_idx + 1

    stages = list(stage_by_basename.values())
    report = Report(stages=stages, meta=meta)
    if unknown_basenames:
        report.meta.setdefault("console_warnings", []).append(
            f"{len(unknown_basenames)} xml marker(s) in the console log did not match a "
            "known stage basename and were dropped: " + ", ".join(repr(b) for b in unknown_basenames)
        )
    if not markers:
        report.meta.setdefault("console_warnings", []).append(
            "No `generated xml file: ...` markers found; the log may have been truncated."
        )
    return report


# ── Rendering ────────────────────────────────────────────────────────────────

CSS = """
  *,*::before,*::after{box-sizing:border-box;margin:0;padding:0}
  body{font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,Oxygen,Ubuntu,Cantarell,'Helvetica Neue',sans-serif;line-height:1.55;color:#24292e;background:#f6f8fa;padding:24px}
  .container{max-width:1200px;margin:0 auto;background:#fff;padding:32px 36px;border-radius:10px;box-shadow:0 2px 8px rgba(0,0,0,.12)}
  h1{font-size:1.75em;color:#1a1a2e;border-bottom:3px solid #0366d6;padding-bottom:10px;margin-bottom:14px}
  .meta{display:flex;flex-wrap:wrap;gap:8px;margin-bottom:22px}
  .pill{display:inline-block;background:#f6f8fa;color:#57606a;border:1px solid #e1e4e8;border-radius:12px;padding:2px 10px;font-size:.74em;font-weight:600}
  .pill a{color:#0366d6;text-decoration:none}
  .verdict{font-size:1.05em;font-weight:800;letter-spacing:.03em;border-radius:8px;padding:8px 18px;align-self:center}
  .v-pass{color:#dafbe1;background:#1a7f37}
  .v-fail{color:#ffebe9;background:#cf222e}
  .v-none{color:#57606a;background:#eaeef2}
  h2{font-size:1.05em;color:#1a1a2e;margin:26px 0 12px;padding-bottom:6px;border-bottom:2px solid #e1e4e8}
  .strip{display:flex;flex-wrap:wrap;gap:12px;margin-bottom:8px;align-items:stretch}
  .card{background:#f6f8fa;border:1px solid #e1e4e8;border-radius:8px;padding:10px 16px;min-width:104px;display:flex;flex-direction:column}
  .card .lbl{font-size:.62em;font-weight:700;color:#586069;text-transform:uppercase;letter-spacing:.04em;margin-bottom:4px}
  .card .val{font-size:1.5em;font-weight:800;color:#1a1a2e;font-variant-numeric:tabular-nums}
  .card.fail{border-left:4px solid #cf222e}.card.fail .val{color:#cf222e}
  .card.warn{border-left:4px solid #9a6700}
  .card.ok .val{color:#1a7f37}
  table{width:100%;border-collapse:collapse;font-size:.86em}
  th{text-align:left;font-weight:600;color:#24292e;border-bottom:2px solid #d1d5da;padding:9px 10px;white-space:nowrap;background:#f6f8fa}
  td{padding:8px 10px;border-bottom:1px solid #e1e4e8;vertical-align:top}
  th.num,td.num{text-align:right;font-variant-numeric:tabular-nums}
  th.scol{white-space:normal;width:92px;min-width:92px;vertical-align:bottom;font-size:.8em;line-height:1.28;font-weight:600}
  tbody tr:hover{background:#f6f8fa}
  code{font-family:'SFMono-Regular',Consolas,Menlo,monospace;font-size:.92em}
  .marker{font-family:'SFMono-Regular',Consolas,Menlo,monospace;font-size:.82em;color:#57606a;word-break:break-word}
  .badge{display:inline-block;padding:1px 8px;border-radius:12px;font-size:.76em;font-weight:700;white-space:nowrap}
  .b-pass{color:#1a7f37;background:#dafbe1}.b-fail{color:#cf222e;background:#ffebe9}
  .b-err{color:#bc4c00;background:#fff1e5}.b-skip{color:#57606a;background:#eaeef2}
  .b-xfail{color:#9a6700;background:#fff8c5}.b-nr{color:#57606a;background:#eaeef2;font-style:italic}
  .row-nr td{color:#8b949e}
  tr.group-row td{background:#eef1f5;border-bottom:1px solid #d1d5da;padding:6px 10px;font-size:.7em;font-weight:700;text-transform:uppercase;letter-spacing:.06em;color:#3a4149}
  tr.group-row .group-count{color:#8b949e;font-weight:600}
  tr.group-row:hover td{background:#eef1f5}
  th.compile-only,td.compile-only{background:#f6f0ff}
  th.compile-only::after{content:' *';color:#8256d0}
  .badge.b-compile{color:#4c1d95;background:#ede4ff;border:1px dashed #b39ddb}
  details.cat{border:1px solid #e1e4e8;border-radius:8px;margin:10px 0;overflow:hidden}
  details.cat>summary{cursor:pointer;padding:10px 14px;font-weight:600;background:#f6f8fa;list-style:none;display:flex;flex-wrap:wrap;align-items:center;gap:10px}
  details.cat>summary::-webkit-details-marker{display:none}
  details.cat>summary::before{content:'\\25B8';color:#57606a}
  details.cat[open]>summary::before{content:'\\25BE'}
  details.cat>summary .cat-label{font-size:1em;color:#1a1a2e}
  details.cat>summary .cat-tally{color:#57606a;font-size:.82em;font-weight:500;font-variant-numeric:tabular-nums}
  details.cat>summary .cat-splits{color:#57606a;font-size:.78em;font-weight:500}
  details.cat>.cat-body{padding:12px 14px}
  details.stage{border:1px solid #e1e4e8;border-radius:8px;margin:10px 0;overflow:hidden}
  details.stage>summary{cursor:pointer;padding:10px 14px;font-weight:600;background:#f6f8fa;list-style:none}
  details.stage>summary::-webkit-details-marker{display:none}
  details.stage>summary::before{content:'\\25B8';margin-right:8px;color:#57606a}
  details.stage[open]>summary::before{content:'\\25BE'}
  .toolbar{display:flex;flex-wrap:wrap;gap:8px;padding:10px 14px;background:#fbfcfd;border-bottom:1px solid #e1e4e8;align-items:center}
  .toolbar input{flex:1 1 200px;padding:5px 9px;border:1px solid #d1d5da;border-radius:6px;font-size:.85em}
  .chip{cursor:pointer;border:1px solid #d1d5da;background:#fff;border-radius:12px;padding:2px 10px;font-size:.75em;font-weight:600;color:#57606a}
  .chip.on{background:#0366d6;color:#fff;border-color:#0366d6}
  .btn{cursor:pointer;border:1px solid #d1d5da;background:#fff;border-radius:6px;padding:3px 10px;font-size:.75em;font-weight:600;color:#24292e}
  details.fail{border:1px solid #ffd7d5;border-radius:8px;margin:8px 0;overflow:hidden}
  details.fail>summary{cursor:pointer;padding:9px 12px;background:#fff5f5;font-size:.85em;list-style:none}
  details.fail>summary .badge{margin-right:8px}
  pre.tb{background:#0d1117;color:#e6edf3;padding:12px 14px;font-family:'SFMono-Regular',Consolas,Menlo,monospace;font-size:.8em;overflow-x:auto;white-space:pre-wrap;margin:0}
  details.log>summary{cursor:pointer;padding:7px 12px;background:#f6f8fa;font-size:.78em;font-weight:600;color:#57606a}
  .callout-ok{background:#dafbe1;border:1px solid #b6e3c0;color:#1a7f37;border-radius:8px;padding:12px 16px;font-weight:600}
  .note{background:#fff8c5;border:1px solid #f0d98c;color:#7a5c00;border-radius:8px;padding:8px 14px;font-size:.82em;margin:10px 0}
  .footer{margin-top:26px;padding-top:14px;border-top:1px solid #e1e4e8;font-size:.78em;color:#8b949e;text-align:right}
"""

JS = """
  function ciFilter(s){
    var bar=document.querySelector('.toolbar[data-stage="'+s+'"]');
    var q=bar.querySelector('input').value.toLowerCase();
    var chip=bar.querySelector('.chip.on');
    var st=chip?chip.getAttribute('data-st'):'all';
    document.querySelectorAll('#tb-'+s+' tr').forEach(function(r){
      var okText=r.getAttribute('data-s').indexOf(q)>=0;
      var okSt=(st==='all')||(r.getAttribute('data-status')===st);
      r.style.display=(okText&&okSt)?'':'none';
    });
  }
  function ciChip(s,st,el){
    el.parentNode.querySelectorAll('.chip').forEach(function(c){c.classList.remove('on');c.removeAttribute('data-st');});
    el.classList.add('on');el.setAttribute('data-st',st);ciFilter(s);
  }
  function ciSort(s){
    var tb=document.getElementById('tb-'+s);
    var rows=[].slice.call(tb.querySelectorAll('tr'));
    var asc=tb.getAttribute('data-asc')==='1';
    rows.sort(function(a,b){var x=+a.getAttribute('data-dur'),y=+b.getAttribute('data-dur');return asc?x-y:y-x;});
    tb.setAttribute('data-asc',asc?'0':'1');rows.forEach(function(r){tb.appendChild(r);});
  }
"""


def badge(outcome):
    label, cls = BADGE[outcome]
    return f'<span class="badge {cls}">{label}</span>'


def status_pill(has_run, failed, empty):
    if not has_run:
        return '<span class="badge b-nr">Not Run</span>'
    if empty:
        return '<span class="badge b-skip">0 tests</span>'
    return '<span class="badge b-fail">FAIL</span>' if failed else '<span class="badge b-pass">PASS</span>'


def render_header(report):
    m = report.meta
    parts = ["  <h1>&#129513; " + esc(m["title"]) + "</h1>", '  <div class="meta">']
    if m.get("pr"):
        parts.append(
            f'    <span class="pill">PR <a href="{esc(m["repo_url"])}/pull/{esc(m["pr"])}">#{esc(m["pr"])}</a></span>'
        )
    if m.get("commit"):
        short = esc(m["commit"][:9])
        parts.append(
            f'    <span class="pill">commit <a href="{esc(m["repo_url"])}/commit/{esc(m["commit"])}">{short}</a></span>'
        )
    if m.get("branch"):
        parts.append(f'    <span class="pill">branch {esc(m["branch"])}</span>')
    if m.get("profile"):
        parts.append(
            f'    <span class="pill" title="{esc(expand_profile_filter(m["profile"]))}">profile {esc(m["profile"])}</span>'
        )
    parts.append(f'    <span class="pill">{esc(m["generated_at"])}</span>')
    parts.append(f'    <span class="pill">wall {esc(fmt_duration(report.wall_time))}</span>')
    parts.append("  </div>")
    return "\n".join(parts)


def render_kpis(report):
    t = report.totals
    total = sum(t.values())
    verdict = report.verdict
    vcls = {"PASS": "v-pass", "FAIL": "v-fail", "NO TESTS": "v-none"}[verdict]
    vicon = {"PASS": "&#10003; ", "FAIL": "&#10007; ", "NO TESTS": ""}[verdict]
    rate = report.pass_rate
    rate_str = "n/a" if rate is None else f"{rate:.1f}%"

    def card(label, value, cls=""):
        cls = f" {cls}" if cls else ""
        return f'      <div class="card{cls}"><span class="lbl">{label}</span><span class="val">{value}</span></div>'

    rows = [
        '    <div class="strip">',
        f'      <span class="verdict {vcls}">{vicon}{verdict}</span>',
        card("Total", total),
        card("Passed", t[Outcome.PASSED], "ok"),
        card("Failed", t[Outcome.FAILED], "fail" if t[Outcome.FAILED] else ""),
        card("Errors", t[Outcome.ERROR], "fail" if t[Outcome.ERROR] else ""),
        card("Skipped", t[Outcome.SKIPPED]),
        card("XFailed", t[Outcome.XFAIL], "warn" if t[Outcome.XFAIL] else ""),
        card("Pass rate", rate_str),
        card("Wall time", fmt_duration(report.wall_time)),
        "    </div>",
    ]
    return "\n".join(rows)


def render_stage_summary(report):
    prof = report.meta.get("profile", "")
    profile_filter = expand_profile_filter(prof) if prof else "<PROFILE>"
    rows = [
        "    <h2>Stage summary</h2>",
        "    <table><thead><tr>",
        "      <th>Stage</th><th>Marker</th><th class='num'>Total</th><th class='num'>Pass</th>",
        "      <th class='num'>Fail</th><th class='num'>Err</th><th class='num'>Skip</th>",
        "      <th class='num'>XFail</th><th>Status</th><th class='num'>Duration</th>",
        "    </tr></thead><tbody>",
    ]
    any_not_run = False
    for st in report.stages:
        marker = esc(st.spec.marker_expr.replace("<PROFILE>", profile_filter))
        if not st.ran:
            any_not_run = True
            rows.append(
                f'      <tr class="row-nr"><td>{esc(st.spec.display)}</td><td class="marker">{marker}</td>'
                '<td class="num" colspan="6">&mdash;</td>'
                f"<td>{status_pill(False, False, False)}</td><td class='num'>&mdash;</td></tr>"
            )
            continue
        if st.parse_error:
            rows.append(
                f'      <tr><td>{esc(st.spec.display)}</td><td class="marker">{marker}</td>'
                '<td class="num" colspan="6">&mdash;</td>'
                f'<td><span class="badge b-err" title="{esc(st.parse_error)}">Parse Error</span></td>'
                "<td class='num'>&mdash;</td></tr>"
            )
            continue
        c = st.counts
        total = sum(c.values())
        rows.append(
            f'      <tr><td><a href="#{st.slug}">{esc(st.spec.display)}</a></td>'
            f'<td class="marker">{marker}</td>'
            f'<td class="num">{total}</td><td class="num">{c[Outcome.PASSED]}</td>'
            f'<td class="num">{c[Outcome.FAILED]}</td><td class="num">{c[Outcome.ERROR]}</td>'
            f'<td class="num">{c[Outcome.SKIPPED]}</td><td class="num">{c[Outcome.XFAIL]}</td>'
            f"<td>{status_pill(True, st.failed, total == 0)}</td>"
            f'<td class="num">{fmt_duration(st.suite_time)}</td></tr>'
        )
    rows.append("    </tbody></table>")
    if any_not_run:
        rows.append(
            '    <div class="note">&#9888; Some stages did not run (skipped after an earlier stage '
            'failed, or disabled via the run parameters). "Not Run" &ne; passed.</div>'
        )
    if report.merged_fallback:
        rows.append(
            '    <div class="note">&#9888; Only the merged <code>tests_log.xml</code> was found; '
            "per-stage attribution is unavailable, so all tests are shown under a single stage.</div>"
        )
    for warning in report.meta.get("console_warnings", []):
        rows.append(f'    <div class="note">&#9888; {esc(warning)}</div>')
    return "\n".join(rows)


def render_by_model(report):
    agg = OrderedDict()
    for tc in report.all_cases:
        if tc.seeded:
            continue
        key = normalize_model(tc.param_id)
        agg.setdefault(key, Counter())[tc.outcome] += 1
    if not agg:
        return ""

    def sort_key(item):
        _, c = item
        return (-(c[Outcome.FAILED] + c[Outcome.ERROR]), item[0])

    rows = [
        "    <h2>By model / config</h2>",
        "    <table><thead><tr><th>Model / Config</th><th class='num'>Pass</th>"
        "<th class='num'>Fail</th><th class='num'>Skip</th><th class='num'>XFail</th><th>Status</th></tr></thead><tbody>",
    ]
    for model, c in sorted(agg.items(), key=sort_key):
        failed = c[Outcome.FAILED] + c[Outcome.ERROR] > 0
        rows.append(
            f"      <tr><td><code>{esc(model)}</code></td>"
            f'<td class="num">{c[Outcome.PASSED]}</td><td class="num">{c[Outcome.FAILED] + c[Outcome.ERROR]}</td>'
            f'<td class="num">{c[Outcome.SKIPPED]}</td><td class="num">{c[Outcome.XFAIL]}</td>'
            f"<td>{status_pill(True, failed, False)}</td></tr>"
        )
    rows.append("    </tbody></table>")
    return "\n".join(rows)


def render_coverage_matrix(report):
    """Render a model × feature grid, one aggregated status glyph per (model, feature) cell.

    Rows are only the models that have at least one feature-encoding test in this run
    (whisper / embedding / plain-parity tests carry no feature token and stay out — they
    remain in the flat "By model / config" rollup). Columns are the fixed FEATURE_COLUMNS
    list, always rendered so blanks make an untested capability obvious. Seeded rows
    (slowest-N placeholders whose PASSED status is a guess) and ``(no model param)``
    unit tests are excluded — this view is meant to answer "which models exercised
    which features," not "how many unit-test files ran."
    """
    # cells[model][feature] -> Counter of outcomes; models with no feature hits never appear.
    cells = OrderedDict()
    for tc in report.all_cases:
        if tc.seeded:
            continue
        if category_of(tc.module_path) != CAT_CAUSAL:
            continue
        model = normalize_model(tc.param_id)
        if model == _NO_MODEL:
            continue
        feats = classify_features(tc)
        if not feats:
            continue
        by_feature = cells.setdefault(model, {})
        for key in feats:
            by_feature.setdefault(key, Counter())[tc.outcome] += 1
    if not cells:
        return ""

    def _lit_columns(model_cells):
        return sum(1 for c in model_cells.values() if c)

    def _fail_count(model_cells):
        return sum(c[Outcome.FAILED] + c[Outcome.ERROR] for c in model_cells.values())

    # Failing models first (fastest reviewer signal); then most-covered models
    # (rows with data float above rows with a single lit cell); alphabetical last.
    ordered = sorted(cells.items(), key=lambda item: (-_fail_count(item[1]), -_lit_columns(item[1]), item[0]))

    header_cells = "".join(f"<th class='num'>{esc(display)}</th>" for _key, display, _pred in FEATURE_COLUMNS)
    rows = [
        "    <h2>Feature coverage matrix</h2>",
        '    <div class="note" style="background:#f6f8fa;border-color:#e1e4e8;color:#57606a">'
        "Causal-LM tests only. A model appears here when at least one CB / dtype / subfunction / "
        "CCL / SPD / blocking / disagg / prefix-cache test ran against it in this run. "
        "Cells reduce to the most-severe outcome across all matching tests; hover for the raw counts."
        "</div>",
        f"    <table><thead><tr><th>Model / Config</th>{header_cells}</tr></thead><tbody>",
    ]
    for model, by_feature in ordered:
        tds = [f"      <tr><td><code>{esc(model)}</code></td>"]
        for key, _display, _pred in FEATURE_COLUMNS:
            counter = by_feature.get(key)
            tds.append(_coverage_cell(counter))
        tds.append("</tr>")
        rows.append("".join(tds))
    rows.append("    </tbody></table>")
    return "\n".join(rows)


# Model-family tokens that denote a Mixture-of-Experts architecture even when the
# normalized slug does not literally contain "moe" (Mixtral and GPT-OSS are always MoE).
_MOE_FAMILY_TOKENS = ("mixtral", "gpt_oss", "gpt-oss")

# Known Mixture-of-Experts VLM cards. Causal per-PR ids embed "moe" in the slug so a
# substring test suffices there, but VLM parametrize ids are raw HF cards
# (``Qwen/Qwen3-VL-30B-A3B-Instruct``) with no "moe" token, so the slug rule would
# wrongly file them as Dense. This set is the report-side mirror of the test's own
# selection (``[m for m in image_text_models if "moe" in m["model_type"]]`` in
# tests/configs/image_text_model_configs.json); a new MoE VLM needs one line here.
_MOE_VLM_CARDS = frozenset(
    {
        "Qwen/Qwen3-VL-30B-A3B-Instruct",
        "Qwen/Qwen3.5-35B-A3B",
    }
)

# Categories whose models split meaningfully on Dense vs Mixture-of-Experts — the
# axis that drives QEff export/compile behaviour. Categories NOT listed here have
# no MoE variants (embedding / audio / sequence / diffusion), so they render as a
# single flat group with no empty "MoE (0)" bucket.
_MOE_SPLIT_CATEGORIES = frozenset({CAT_CAUSAL, CAT_VLM})


def _is_moe_slug(model):
    """True when a normalized model slug denotes MoE via a literal token or known family."""
    slug = model.lower()
    return "moe" in slug or any(tok in slug for tok in _MOE_FAMILY_TOKENS)


def moe_group_for(category, model):
    """Return ``"MoE"``/``"Dense"`` for a model in ``category``, or ``None`` to not group.

    The MoE oracle is per-category because the signal that survives into the report
    differs by category (see :data:`_MOE_VLM_CARDS`): causal slugs carry "moe", VLM
    cards don't. Categories with no MoE variants return ``None`` so the renderer skips
    grouping entirely rather than drawing an empty MoE group.
    """
    if category not in _MOE_SPLIT_CATEGORIES:
        return None
    if category == CAT_VLM:
        return "MoE" if model in _MOE_VLM_CARDS else "Dense"
    return "MoE" if _is_moe_slug(model) else "Dense"


def _render_scenario_matrix_body(cases, category, columns, empty_msg=""):
    """Render a model × scenario matrix for one category, or ``""`` when empty.

    Shared code path for both the causal Scenario matrix and the per-category
    matrices in the new Coverage-by-category section. Cells aggregate outcomes
    per (model, column-key); compile-only columns are visually distinguished
    via :data:`_COMPILE_ONLY_SCENARIOS`; rows are grouped Dense/MoE per
    :func:`moe_group_for` when the category has a real MoE axis, otherwise a
    flat list. Fail-first row / group ordering, ``seeded`` + ``_NO_MODEL``
    exclusions, and fixed-width ``.scol`` headers are enforced here.

    ``cases`` — pre-filtered iterable of ``TestCase`` (already scoped to this
    category by the caller). ``columns`` is the ``[(key, display)]`` list; for
    causal, ``key`` is ``tc.test_fn``. For VLM, callers pass a wrapper that
    translates ``tc.test_fn`` into the profile-stripped key before lookup.
    """
    col_keys = {key for key, _display in columns}
    cells = OrderedDict()
    for tc, matrix_key in cases:
        if tc.seeded or matrix_key not in col_keys:
            continue
        model = normalize_model(tc.param_id)
        if model == _NO_MODEL:
            continue
        cells.setdefault(model, {}).setdefault(matrix_key, Counter())[tc.outcome] += 1
    if not cells:
        return empty_msg

    def _lit_columns(model_cells):
        return sum(1 for c in model_cells.values() if c)

    def _fail_count(model_cells):
        return sum(c[Outcome.FAILED] + c[Outcome.ERROR] for c in model_cells.values())

    ordered = sorted(cells.items(), key=lambda item: (-_fail_count(item[1]), -_lit_columns(item[1]), item[0]))

    grouped = OrderedDict()
    any_grouped = False
    for model, by_key in ordered:
        group = moe_group_for(category, model)
        if group is None:
            grouped.setdefault("_flat", []).append((model, by_key))
        else:
            any_grouped = True
            grouped.setdefault(group, []).append((model, by_key))
    group_fail = {g: sum(_fail_count(by_k) for _m, by_k in members) for g, members in grouped.items()}
    _GROUP_RANK = {"Dense": 0, "MoE": 1, "_flat": 99}
    group_order = sorted(grouped, key=lambda g: (-group_fail[g], _GROUP_RANK.get(g, 50), g))

    header_cells = "".join(
        f"<th class='num scol{' compile-only' if key in _COMPILE_ONLY_SCENARIOS else ''}' title='{esc(key)}'>"
        f"{esc(display)}</th>"
        for key, display in columns
    )
    rows = [f"    <table><thead><tr><th>Model / Config</th>{header_cells}</tr></thead><tbody>"]
    group_colspan = 1 + len(columns)
    for group in group_order:
        members = grouped[group]
        if any_grouped and group != "_flat":
            rows.append(
                f'      <tr class="group-row"><td colspan="{group_colspan}">'
                f'{esc(group)} <span class="group-count">({len(members)})</span></td></tr>'
            )
        for model, by_key in members:
            tds = [f"      <tr><td><code>{esc(model)}</code></td>"]
            for key, _display in columns:
                tds.append(_coverage_cell(by_key.get(key), compile_only=(key in _COMPILE_ONLY_SCENARIOS)))
            tds.append("</tr>")
            rows.append("".join(tds))
    rows.append("    </tbody></table>")
    return "\n".join(rows)


def _render_rollup_body(cases):
    """Render a flat models × single-parity-column rollup for a category, or ``""``.

    Used for Embedding / Audio / Sequence-Reranker where a scenario split would
    fabricate columns. Each row is one model; the single cell reduces every
    hit for that model to its worst outcome via :data:`_CELL_SEVERITY`. Seeded
    rows and ``_NO_MODEL`` unit tests are excluded.
    """
    agg = OrderedDict()
    for tc in cases:
        if tc.seeded:
            continue
        model = normalize_model(tc.param_id)
        if model == _NO_MODEL:
            continue
        agg.setdefault(model, Counter())[tc.outcome] += 1
    if not agg:
        return ""

    def _fail_count(c):
        return c[Outcome.FAILED] + c[Outcome.ERROR]

    ordered = sorted(agg.items(), key=lambda item: (-_fail_count(item[1]), item[0]))
    rows = [
        "    <table><thead><tr><th>Model / Config</th>"
        "<th class='num'>Pass</th><th class='num'>Fail</th><th class='num'>Skip</th>"
        "<th class='num'>XFail</th><th>Parity</th></tr></thead><tbody>"
    ]
    for model, c in ordered:
        failed = _fail_count(c) > 0
        rows.append(
            f"      <tr><td><code>{esc(model)}</code></td>"
            f'<td class="num">{c[Outcome.PASSED]}</td>'
            f'<td class="num">{c[Outcome.FAILED] + c[Outcome.ERROR]}</td>'
            f'<td class="num">{c[Outcome.SKIPPED]}</td>'
            f'<td class="num">{c[Outcome.XFAIL]}</td>'
            f"<td>{status_pill(True, failed, False)}</td></tr>"
        )
    rows.append("    </tbody></table>")
    return "\n".join(rows)


def _render_pipeline_list_body(cases, header="Pipeline test", show_models=False):
    """Render a flat test-fn list, or ``""`` when empty.

    Used where a model × scenario matrix would always be empty:

    * Diffusion tests are fixture-based (``test_wan_pipeline(wan_pipeline)``) with
      no model param. xfail is surfaced honestly (``flux`` is xfail today for the
      BF16 export gap) so it is not misread as parity confirmation.
    * The ``Other`` reconciling bucket (CLI / finetune / unit / Dynamo) has no
      model-category axis worth a grid.

    Rows are keyed on ``module::test_fn`` rather than ``test_fn`` alone: the same
    function name can legitimately appear in two modules (Dynamo and causal both
    define export/compile tests), and collapsing them would merge unrelated
    results into one row.

    ``show_models=True`` adds a Models column listing each row's parametrized
    models. Use it where the tile header advertises a model count (``Other``):
    a count the reader cannot see in the body is unverifiable. Fixture-based
    categories (diffusion) have no model param, so they leave it off.
    """
    per_fn = OrderedDict()
    models_by_fn = OrderedDict()
    for tc in cases:
        if tc.seeded or not tc.test_fn:
            continue
        module = (tc.module_path or "").rsplit("/", 1)[-1]
        key = f"{module}::{tc.test_fn}" if module else tc.test_fn
        per_fn.setdefault(key, Counter())[tc.outcome] += 1
        model = normalize_model(tc.param_id)
        if model != _NO_MODEL:
            models_by_fn.setdefault(key, []).append(model)
    if not per_fn:
        return ""

    def _sort_key(item):
        _fn, c = item
        return (-(c[Outcome.FAILED] + c[Outcome.ERROR]), item[0])

    model_th = "<th>Models</th>" if show_models else ""
    rows = [
        f"    <table><thead><tr><th>{esc(header)}</th>{model_th}"
        "<th class='num'>Runs</th><th>Outcome</th></tr></thead><tbody>"
    ]
    for fn, c in sorted(per_fn.items(), key=_sort_key):
        outcome = max(c, key=lambda o: (_CELL_SEVERITY[o], c[o]))
        _label, cls = BADGE[outcome]
        tip = ", ".join(f"{c[o]} {o.value}" for o in Outcome if c[o])
        total = sum(c.values())
        model_td = ""
        if show_models:
            seen = list(OrderedDict.fromkeys(models_by_fn.get(fn, [])))
            cell = (
                ", ".join(f"<code>{esc(m)}</code>" for m in seen)
                if seen
                else '<span style="color:#8b949e">&mdash;</span>'
            )
            model_td = f"<td>{cell}</td>"
        rows.append(
            f"      <tr><td><code>{esc(fn)}</code></td>{model_td}"
            f'<td class="num">{total}</td>'
            f'<td><span class="badge {cls}" title="{esc(tip)}">{esc(BADGE[outcome][0])}</span></td></tr>'
        )
    rows.append("    </tbody></table>")
    return "\n".join(rows)


# Presentation-mode dispatch for :func:`render_category_section`.
# ``matrix`` renders a model × curated-scenario grid (causal, VLM);
# ``rollup`` renders a flat models × single-parity column list (embedding, audio,
# sequence-reranker) — a scenario split there would fabricate columns;
# ``pipeline`` renders a fixture-based test-fn list (diffusion — no model param).
_CATEGORY_MODE = {
    CAT_CAUSAL: ("matrix", SCENARIO_COLUMNS),
    CAT_VLM: ("matrix", VLM_SCENARIO_COLUMNS),
    CAT_EMBEDDING: ("rollup", None),
    CAT_AUDIO: ("rollup", None),
    CAT_SEQ_RERANKER: ("rollup", None),
    CAT_DIFFUSION: ("pipeline", None),
    # Non-model tests (CLI / finetune / unit / Dynamo) have no model or scenario
    # axis worth a grid; list them by test so the bucket is auditable, not opaque.
    CATEGORY_OTHER: ("pipeline", None),
}


def _category_cases(report):
    """Group non-seeded ``TestCase`` by category label, including :data:`CATEGORY_OTHER`.

    Returned dict is keyed by category label and always contains every entry in
    :data:`COVERAGE_CATEGORY_ORDER` (zero-row categories map to an empty list, so
    the caller can render them as "Not Run / 0 tests" instead of silently omitting).
    ``Other`` collects everything with no model category (CLI / finetune / unit /
    Dynamo) so the per-tile totals reconcile against the KPI Total.
    """
    by_cat = OrderedDict((label, []) for label in COVERAGE_CATEGORY_ORDER)
    for tc in report.all_cases:
        if tc.seeded:
            continue
        label = category_of(tc.module_path)
        if label in by_cat:
            by_cat[label].append(tc)
    return by_cat


def _category_stage_notes(report):
    """Collect distinct partial-attribution stage notes per category.

    Propagates a stage's console-log ``note`` (e.g. "only slowest-N tests are
    attributable") to every category it touched, so a "0 tests" category driven
    by console-log attribution loss is not misread as a coverage hole.
    """
    notes = OrderedDict((label, []) for label in COVERAGE_CATEGORY_ORDER)
    for st in report.ran_stages:
        if not st.note:
            continue
        seen_here = set()
        for tc in st.cases:
            label = category_of(tc.module_path)
            if label in notes and label not in seen_here:
                seen_here.add(label)
                if st.note not in notes[label]:
                    notes[label].append(f"{st.spec.display}: {st.note}")
    return notes


# Which categories each stage feeds, keyed by the per-stage JUnit XML basename.
# Needed only for stages that did NOT run: they have no cases, so the category
# they would have fed cannot be inferred from the data. Without this a category
# renders a confident PASS from its other stages while a whole feeding stage was
# skipped — the tile would overstate coverage. Only stages with a real category
# mapping are listed; CLI / Dynamo / Finetune feed the ``Other`` bucket.
_STAGE_FEEDS_CATEGORIES = {
    "tests_log1.xml": (CAT_CAUSAL,),
    "tests_log2.xml": (CAT_CAUSAL, CAT_SEQ_RERANKER),
    # (on_qaic) and (feature) — SPD/PLD, prefix-caching and sampler tests all run
    # against causal models, so a skipped Feature stage is a causal coverage gap.
    "tests_log2_feature.xml": (CAT_CAUSAL,),
    "tests_log_embedding_audio.xml": (CAT_EMBEDDING, CAT_AUDIO),
    "tests_log6.xml": (CAT_VLM,),
    "tests_log_reranker.xml": (CAT_SEQ_RERANKER,),
    "tests_log_diffusion.xml": (CAT_DIFFUSION,),
}


def _category_not_run_stages(report):
    """Map category label → display names of stages that feed it but did not run.

    Lets a tile say "PASS, but the Feature stage never ran" instead of implying
    the category was fully exercised. Stage state is authoritative here: a stage
    that produced no XML is a coverage gap, not a pass.
    """
    gaps = OrderedDict((label, []) for label in COVERAGE_CATEGORY_ORDER)
    for basename, spec in STAGE_MAP.items():
        categories = _STAGE_FEEDS_CATEGORIES.get(basename)
        if not categories:
            continue
        stage = next((st for st in report.stages if st.spec.display == spec.display), None)
        if stage is not None and stage.ran:
            continue
        for label in categories:
            if label in gaps:
                gaps[label].append(spec.display)
    return gaps


def _category_dense_moe_split(category, cases):
    """Return ``(dense_models, moe_models)`` counts for a category, or ``None``.

    ``None`` means the category has no MoE axis (embedding/audio/etc.) and the
    Dense/MoE presence indicator should be suppressed in the header entirely.
    """
    if category not in _MOE_SPLIT_CATEGORIES:
        return None
    dense, moe = set(), set()
    for tc in cases:
        model = normalize_model(tc.param_id)
        if model == _NO_MODEL:
            continue
        group = moe_group_for(category, model)
        if group == "MoE":
            moe.add(model)
        elif group == "Dense":
            dense.add(model)
    return len(dense), len(moe)


def render_category_section(report):
    """Render the top-level **Coverage by category** section.

    Iterates :data:`CATEGORY_ORDER` in fixed order (roster discipline). Each
    category renders as one ``<details class="cat">``; the summary line always
    shows run-state / total / pass / fail / skip / xfail / model count and
    Dense+MoE presence where applicable, so a reviewer can spot a coverage
    gap without expanding. Auto-opens iff the category has a failure or is
    Not Run. Empty categories still appear (never silently omitted).
    """
    by_cat = _category_cases(report)
    notes_by_cat = _category_stage_notes(report)
    gaps_by_cat = _category_not_run_stages(report)

    # A category "did not run" when none of the stages likely to feed it produced
    # any parseable rows. Rather than probe stage→category mapping (fragile), we
    # infer "not run" from "no test cases arrived for this category AND at least
    # one relevant stage failed to produce parseable results OR the whole run is
    # missing that stage". Simpler and safer: if the total-cases count for a
    # category is zero we still show it as "Not Run" — the reader gets the gap
    # signal either way.
    rows = ["    <h2>Coverage by category</h2>"]
    rows.append(
        '    <div class="note" style="background:#f6f8fa;border-color:#e1e4e8;color:#57606a">'
        "One row per model category running in QEfficient, split <strong>Dense</strong> vs "
        "<strong>Mixture-of-Experts</strong> where that axis is real. Compile-only cells (dashed "
        "purple) verify build success only &mdash; no on-device parity check. A category with "
        "zero rows still appears as <em>Not Run</em>. Tests belonging to no model category "
        "(CLI / finetune / unit / Dynamo) are collected under <em>Other</em>, so these tallies "
        "sum to the KPI <strong>Total</strong> above."
        "</div>"
    )

    for category in COVERAGE_CATEGORY_ORDER:
        cases = by_cat.get(category, [])
        counts = Counter(tc.outcome for tc in cases)
        total = sum(counts.values())
        # ``Other`` is a reconciling bucket, not part of the fixed roster: it appears
        # only when it holds rows. Every real category renders even at zero rows.
        if category == CATEGORY_OTHER and total == 0:
            continue
        failed = counts[Outcome.FAILED] + counts[Outcome.ERROR] > 0
        has_run = total > 0
        pill = status_pill(has_run, failed, empty=(has_run is False and total == 0))
        # Model tally (excludes _NO_MODEL and seeded, matching the matrix/rollup filters).
        models = {
            normalize_model(tc.param_id) for tc in cases if not tc.seeded and normalize_model(tc.param_id) != _NO_MODEL
        }
        split = _category_dense_moe_split(category, cases)
        tally_bits = []
        if total:
            tally_bits.append(f"{total} tests" if total != 1 else "1 test")
            for o in (Outcome.PASSED, Outcome.FAILED, Outcome.ERROR, Outcome.SKIPPED, Outcome.XFAIL):
                if counts[o]:
                    tally_bits.append(f"{counts[o]} {o.value}")
            # Fixture-driven categories (diffusion) parametrize no model, so a model
            # count of zero is a property of the test style, not a coverage hole —
            # omit it rather than printing a misleading "0 models".
            if models:
                tally_bits.append(f"{len(models)} models" if len(models) != 1 else "1 model")
        else:
            tally_bits.append("0 tests")
        tally_str = " &middot; ".join(esc(b) for b in tally_bits)
        splits_str = ""
        if split is not None:
            dense_n, moe_n = split
            splits_str = f'<span class="cat-splits">Dense {dense_n} &middot; MoE {moe_n}</span>'

        open_attr = " open" if failed or not has_run else ""
        rows.append(f'    <details class="cat"{open_attr}>')
        rows.append(
            f'      <summary>{pill}<span class="cat-label">{esc(category)}</span>'
            f'<span class="cat-tally">{tally_str}</span>{splits_str}</summary>'
        )
        rows.append('      <div class="cat-body">')

        for note in notes_by_cat.get(category, []):
            rows.append(f'        <div class="note">&#9432; {esc(note)}</div>')

        # A stage that feeds this category but produced no XML is a coverage gap the
        # tile's own pass/fail tally cannot express — surface it as a warning so a
        # green tile is never read as "fully exercised".
        gap_stages = gaps_by_cat.get(category, [])
        if gap_stages and has_run:
            listed = ", ".join(esc(name) for name in gap_stages)
            rows.append(
                f'        <div class="note">&#9888; Partial coverage &mdash; {listed} '
                f"did not run, so tests it would have contributed to this category are missing. "
                f"The status above reflects only the stages that ran.</div>"
            )

        if category == CATEGORY_OTHER:
            rows.append(
                '        <div class="note" style="background:#f6f8fa;border-color:#e1e4e8;color:#57606a">'
                "Not a model category &mdash; these are the CLI, finetune, unit/infra and Dynamo tests "
                "that carry no model-category path. Listed so the per-category totals above reconcile "
                "against the KPI <strong>Total</strong>; see the stage detail for the full breakdown."
                "</div>"
            )

        if not has_run:
            rows.append(
                '        <div class="note" style="background:#f6f8fa;border-color:#e1e4e8;color:#57606a">'
                "No test rows arrived for this category in the current run. "
                "The category is part of the fixed coverage roster, so it is surfaced here "
                "rather than silently omitted &mdash; check the stage summary for the owning stage's status."
                "</div>"
            )
            rows.append("      </div>")
            rows.append("    </details>")
            continue

        mode, columns = _CATEGORY_MODE.get(category, ("rollup", None))
        body = ""
        if mode == "matrix":
            if category == CAT_VLM:
                keyed = [(tc, _vlm_scenario_key(tc.test_fn)) for tc in cases]
            else:
                keyed = [(tc, tc.test_fn) for tc in cases]
            body = _render_scenario_matrix_body(keyed, category, columns)
            if not body:
                body = (
                    '        <div class="note" style="background:#f6f8fa;border-color:#e1e4e8;color:#57606a">'
                    f"{total} tests ran in this category, but none matched the curated per-PR scenario columns. "
                    "See the by-model rollup or the stage detail for the full run."
                    "</div>"
                )
        elif mode == "rollup":
            body = _render_rollup_body(cases)
        elif mode == "pipeline":
            is_other = category == CATEGORY_OTHER
            header = "Test" if is_other else "Pipeline test"
            body = _render_pipeline_list_body(cases, header=header, show_models=is_other)

        if body:
            rows.append(body)
        rows.append("      </div>")
        rows.append("    </details>")

    return "\n".join(rows)


def _coverage_cell(counter, compile_only=False):
    """Render one matrix cell: a status glyph reduced from all outcomes hitting the cell.

    ``compile_only=True`` marks the cell as verifying compile success only (no
    on-device output check). The pass badge switches to a distinct dashed style
    (``b-compile``) so a green cell here is visibly not parity-confirmed.
    """
    if not counter:
        return '<td class="num" style="color:#8b949e">&mdash;</td>'
    outcome = max(counter, key=lambda o: (_CELL_SEVERITY[o], counter[o]))
    _label, cls = BADGE[outcome]
    glyph = {
        Outcome.PASSED: "&#10003;",  # check
        Outcome.FAILED: "&#10007;",  # cross
        Outcome.ERROR: "&#10007;",
        Outcome.XFAIL: "xfail",
        Outcome.SKIPPED: "skip",
    }[outcome]
    if compile_only and outcome == Outcome.PASSED:
        cls = "b-compile"
        glyph = "compiled"
    tip = ", ".join(f"{counter[o]} {o.value}" for o in Outcome if counter[o])
    if compile_only:
        tip = (
            ("compile-only (no on-device parity check); " + tip) if tip else "compile-only (no on-device parity check)"
        )
    td_cls = "num compile-only" if compile_only else "num"
    return f'<td class="{td_cls}"><span class="badge {cls}" title="{esc(tip)}">{glyph}</span></td>'


def render_failures(report):
    failures = [tc for tc in report.all_cases if tc.outcome in (Outcome.FAILED, Outcome.ERROR)]
    if not failures:
        return '    <h2>Failures &amp; Errors</h2>\n    <div class="callout-ok">&#10003; No failures or errors.</div>'
    rows = [f"    <h2>Failures &amp; Errors ({len(failures)})</h2>"]
    for i, tc in enumerate(failures):
        open_attr = " open" if i < 10 else ""
        model = normalize_model(tc.param_id)
        meta = f"{esc(model)} &middot; {esc(tc.stage)} &middot; {tc.duration:.1f}s"
        body = tc.message + ("\n\n" + tc.detail if tc.detail else "")
        rows.append(f'    <details class="fail"{open_attr}>')
        rows.append(
            f"      <summary>{badge(tc.outcome)}<code>{esc(tc.nodeid)}</code> &nbsp;"
            f'<span class="marker">{meta}</span></summary>'
        )
        rows.append(f'      <pre class="tb">{esc(truncate(body, MAX_TB_CHARS))}</pre>')
        if tc.captured.strip():
            rows.append('      <details class="log"><summary>Captured stdout / log</summary>')
            rows.append(f'        <pre class="tb">{esc(truncate(tc.captured, MAX_LOG_CHARS))}</pre>')
            rows.append("      </details>")
        rows.append("    </details>")
    return "\n".join(rows)


def render_stage_detail(report):
    rendered = [st for st in report.ran_stages if not st.parse_error]
    if not rendered:
        return ""
    rows = ["    <h2>Test detail by stage</h2>"]
    for st in rendered:
        c = st.counts
        summary_bits = [f"{c[o]} {o.value}" for o in Outcome if c[o]]
        summary = ", ".join(summary_bits) if summary_bits else "0 tests"
        s = st.slug
        open_attr = " open" if st.failed else ""
        rows.append(f'    <details class="stage" id="{s}"{open_attr}>')
        rows.append(f"      <summary>{esc(st.spec.display)} &mdash; {esc(summary)}</summary>")
        if st.note:
            rows.append(f'      <div class="note">&#9432; {esc(st.note)}</div>')
        rows.append(f'      <div class="toolbar" data-stage="{s}">')
        rows.append(
            f'        <input type="text" placeholder="Filter tests in this stage&hellip;" oninput="ciFilter(\'{s}\')">'
        )
        rows.append(f'        <span class="chip on" data-st="all" onclick="ciChip(\'{s}\',\'all\',this)">All</span>')
        for st_key, lbl in (("failed", "Failed"), ("error", "Errors"), ("skipped", "Skipped"), ("xfail", "XFail")):
            rows.append(f"        <span class=\"chip\" onclick=\"ciChip('{s}','{st_key}',this)\">{lbl}</span>")
        rows.append(f'        <span class="btn" onclick="ciSort(\'{s}\')">Sort by duration &#8645;</span>')
        rows.append("      </div>")
        rows.append(
            "      <table><thead><tr><th>Status</th><th>Test</th><th>Model / Config</th><th class='num'>Duration</th></tr></thead>"
        )
        rows.append(f'      <tbody id="tb-{s}">')
        for tc in st.cases:
            search = esc(f"{tc.test_fn} {tc.param_id}".lower())
            test_disp = esc(f"{os.path.basename(tc.module_path)}::{tc.test_fn}" if tc.module_path else tc.test_fn)
            param_disp = f"<code>{esc(tc.param_id)}</code>" if tc.param_id else "&mdash;"
            rows.append(
                f'        <tr data-status="{tc.outcome.value}" data-dur="{tc.duration:.2f}" data-s="{search}">'
                f"<td>{badge(tc.outcome)}</td><td><code>{test_disp}</code></td>"
                f'<td>{param_disp}</td><td class="num">{tc.duration:.2f}</td></tr>'
            )
        rows.append("      </tbody></table>")
        rows.append("    </details>")
    return "\n".join(rows)


def render_slowest(report, top=15):
    cases = sorted(report.all_cases, key=lambda tc: tc.duration, reverse=True)[:top]
    cases = [tc for tc in cases if tc.duration > 0]
    if not cases:
        return ""
    rows = [
        "    <h2>Slowest tests</h2>",
        "    <table><thead><tr><th class='num'>#</th><th class='num'>Duration</th>"
        "<th>Stage</th><th>Test</th><th>Model / Config</th></tr></thead><tbody>",
    ]
    for i, tc in enumerate(cases, 1):
        test_disp = esc(f"{os.path.basename(tc.module_path)}::{tc.test_fn}" if tc.module_path else tc.test_fn)
        param_disp = f"<code>{esc(tc.param_id)}</code>" if tc.param_id else "&mdash;"
        rows.append(
            f'      <tr><td class="num">{i}</td><td class="num">{tc.duration:.1f}s</td>'
            f"<td>{esc(tc.stage)}</td><td><code>{test_disp}</code></td><td>{param_disp}</td></tr>"
        )
    rows.append("    </tbody></table>")
    return "\n".join(rows)


def render_footer(report):
    m = report.meta
    bits = ["Generated by scripts/ci_report/generate_ci_report.py"]
    if m.get("build_url"):
        bits.append(f'<a href="{esc(m["build_url"])}">build log</a>')
    bits.append("source: tests/tests_log*.xml")
    return f'    <div class="footer">{" &middot; ".join(bits)}</div>'


def build_document(report):
    """Assemble the full standalone HTML document string.

    Section order follows the consumer's strict priority (see the redesign plan):
    ``Verdict + KPI strip`` answers "did anything fail?" in one glance; the
    ``Failures & Errors`` section comes next so a failing run surfaces the
    actual defect immediately. The new ``Coverage by category`` section then
    answers "is all intended functionality covered?" across every model
    category (Causal / VLM / Embedding / Audio / Sequence-Reranker / Diffusion),
    Dense/MoE split where the axis is real. The ``Feature coverage matrix`` and
    ``By model / config`` tables provide the atomic drill-down; stage detail
    and slowest are the tail. The causal Scenario matrix is no longer rendered
    standalone — it lives inside ``Coverage by category`` under the ``Causal LM``
    tile so the two views cannot diverge.
    """
    sections = [
        render_kpis(report),
        render_failures(report),
        render_category_section(report),
        render_coverage_matrix(report),
        render_stage_summary(report),
        render_by_model(report),
        render_stage_detail(report),
        render_slowest(report),
        render_footer(report),
    ]
    body = "\n\n".join(s for s in sections if s)
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>{esc(report.meta["title"])}{" — PR #" + esc(report.meta["pr"]) if report.meta.get("pr") else ""}</title>
<style>{CSS}</style>
</head>
<body>
<div class="container">

{render_header(report)}

{body}

</div>
<script>{JS}</script>
</body>
</html>
"""


# ── CLI ──────────────────────────────────────────────────────────────────────


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Generate a standalone HTML report from CI JUnit XML.")
    p.add_argument("--xml-dir", default="tests", help="Directory containing per-stage JUnit XML (default: tests).")
    p.add_argument("--glob", default="tests_log*.xml", dest="pattern", help="Glob for XML files within --xml-dir.")
    p.add_argument(
        "--console-log",
        default=None,
        help=(
            "Parse a Jenkins console log (e.g. ci_logs.log) instead of per-stage JUnit XML. "
            "Totals come from pytest summary lines; per-test drill-down is populated where the "
            "log has `[gw#] STATUS ...` (verbose) lines or slowest-N durations."
        ),
    )
    p.add_argument("-o", "--output", default="tests/ci_report.html", help="Output HTML path.")
    p.add_argument("--pr", default=os.environ.get("CHANGE_ID", ""), help="PR number (default: $CHANGE_ID).")
    p.add_argument("--commit", default=os.environ.get("GIT_COMMIT", ""), help="Commit SHA (default: $GIT_COMMIT).")
    p.add_argument("--branch", default=os.environ.get("BRANCH_NAME", ""), help="Branch/ref (default: $BRANCH_NAME).")
    p.add_argument(
        "--profile", default=os.environ.get("TEST_PROFILE", ""), help="Test profile (default: $TEST_PROFILE)."
    )
    p.add_argument("--title", default="QEfficient CI Test Report", help="Report title.")
    p.add_argument(
        "--repo-url", default="https://github.com/quic/efficient-transformers", help="Base repo URL for links."
    )
    p.add_argument(
        "--build-url", default=os.environ.get("BUILD_URL", ""), help="Jenkins build URL (default: $BUILD_URL)."
    )
    return p.parse_args(argv)


def main(argv=None):
    try:
        args = parse_args(argv)
        meta = {
            "title": args.title,
            "pr": args.pr,
            "commit": args.commit,
            "branch": args.branch,
            "profile": args.profile,
            "repo_url": args.repo_url.rstrip("/"),
            "build_url": args.build_url,
            "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
        }
        report = (
            load_report_from_console(args.console_log, meta)
            if args.console_log
            else load_report(args.xml_dir, args.pattern, meta)
        )
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(build_document(report), encoding="utf-8")

        t = report.totals
        print(
            f"Wrote {output} — verdict={report.verdict} "
            f"(passed={t[Outcome.PASSED]}, failed={t[Outcome.FAILED]}, error={t[Outcome.ERROR]}, "
            f"skipped={t[Outcome.SKIPPED]}, xfail={t[Outcome.XFAIL]}) "
            f"across {len(report.ran_stages)}/{len(report.stages)} stages."
        )
    except SystemExit as exc:
        # argparse raises SystemExit on --help / bad flags; already printed usage to stderr.
        # Jenkins post{always{}} needs a zero exit regardless.
        if exc.code not in (0, None):
            print(f"generate_ci_report: argument parsing failed (code={exc.code})", file=sys.stderr)
    except BaseException as exc:
        # Never let a report-generation failure break Jenkins post{always{}}.
        # Print the traceback to stderr so it lands in the console log for triage.
        print(f"generate_ci_report: unhandled error while producing the report: {exc!r}", file=sys.stderr)
        traceback.print_exc()
    return 0


if __name__ == "__main__":
    sys.exit(main())

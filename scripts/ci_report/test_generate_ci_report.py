# -----------------------------------------------------------------------------
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------
"""Focused test for scripts/ci_report/generate_ci_report.py.

Runs the generator against the bundled ``sample/`` fixtures and asserts the
report keeps its structural contract: fixed category roster, correct Dense/MoE
splits, distinct compile-only tag, Dynamo stage recognised, exit code 0. Fast:
uses only the standard library, no torch/HF import.

Run: ``pytest scripts/ci_report/test_generate_ci_report.py -v``.
"""

from __future__ import annotations

import importlib.util
import re
from pathlib import Path

import pytest

HERE = Path(__file__).parent
GENERATOR = HERE / "generate_ci_report.py"
SAMPLE_DIR = HERE / "sample"


def _load_module():
    spec = importlib.util.spec_from_file_location("generate_ci_report", GENERATOR)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def gcr():
    return _load_module()


@pytest.fixture(scope="module")
def sample_html(tmp_path_factory, gcr):
    out_dir = tmp_path_factory.mktemp("ci_report_sample")
    out_path = out_dir / "ci_report.html"
    exit_code = gcr.main(
        [
            "--xml-dir",
            str(SAMPLE_DIR),
            "--output",
            str(out_path),
            "--pr",
            "1216",
            "--commit",
            "ff0f20b98abcdef1234",
            "--branch",
            "CI_optimization_fork",
            "--profile",
            "dummy_layers_model",
        ]
    )
    assert exit_code == 0, "main() must always return 0 (Jenkins post-always contract)"
    assert out_path.exists()
    return out_path.read_text()


# ── Structural invariants ────────────────────────────────────────────────────


def test_dynamo_stage_recognised(gcr):
    """The Dynamo XML basename must be in STAGE_MAP (regression: it was silently dropped)."""
    assert "tests_log_dynamo_qaic.xml" in gcr.STAGE_MAP
    dynamo = gcr.STAGE_MAP["tests_log_dynamo_qaic.xml"]
    assert dynamo.gate == "RUN_DYNAMO_QAIC"
    # Dynamo runs after CLI (order 8) and before Finetune (order 10).
    assert dynamo.order == 9
    finetune = gcr.STAGE_MAP["tests_log_finetune.xml"]
    assert finetune.order == 10


def test_category_roster_stable(gcr):
    """Six-category roster + OTHER; changes must be intentional."""
    assert gcr.CATEGORY_ORDER == [
        gcr.CAT_CAUSAL,
        gcr.CAT_VLM,
        gcr.CAT_EMBEDDING,
        gcr.CAT_AUDIO,
        gcr.CAT_SEQ_RERANKER,
        gcr.CAT_DIFFUSION,
    ]
    # CLI / finetune / unit-test paths fall through to OTHER without crashing.
    assert gcr.category_of("tests/cloud/test_infer.py") == gcr.CATEGORY_OTHER
    assert gcr.category_of("") == gcr.CATEGORY_OTHER
    # Reranker + sequence-classification both file under Sequence / Reranker.
    assert gcr.category_of("tests/transformers/models/reranker/test_reranker_mad.py") == gcr.CAT_SEQ_RERANKER
    assert gcr.category_of("tests/transformers/models/sequence_models/test_seq_cls.py") == gcr.CAT_SEQ_RERANKER


def test_moe_oracle_per_category(gcr):
    """Causal MoE keyed on slug; VLM MoE keyed on the curated card set; others suppress grouping."""
    assert gcr.moe_group_for(gcr.CAT_CAUSAL, "qwen3_moe_text") == "MoE"
    assert gcr.moe_group_for(gcr.CAT_CAUSAL, "mixtral_moe_text") == "MoE"
    assert gcr.moe_group_for(gcr.CAT_CAUSAL, "gpt_oss_moe_text") == "MoE"
    assert gcr.moe_group_for(gcr.CAT_CAUSAL, "llama_text") == "Dense"
    # VLM MoE cards lack a "moe" token in the HF card, so slug-sniff would misfile them.
    assert gcr.moe_group_for(gcr.CAT_VLM, "Qwen/Qwen3-VL-30B-A3B-Instruct") == "MoE"
    assert gcr.moe_group_for(gcr.CAT_VLM, "llava-hf/llava-1.5-7b-hf") == "Dense"
    # Categories without a real MoE axis: grouping suppressed.
    for label in (gcr.CAT_EMBEDDING, gcr.CAT_AUDIO, gcr.CAT_SEQ_RERANKER, gcr.CAT_DIFFUSION):
        assert gcr.moe_group_for(label, "any-model") is None


def test_vlm_scenario_key_strips_profile_prefix(gcr):
    """The VLM matrix collapses full_/few_/dummy_ profile variants to one scenario column."""
    for prefix in ("test_full_", "test_few_", "test_dummy_"):
        key = gcr._vlm_scenario_key(prefix + "image_text_to_text_ccl_dual_qpc")
        assert key == "image_text_to_text_ccl_dual_qpc"
    # Non-profile VLM tests (reference / qnn / custom) don't collide with a scenario column.
    assert gcr._vlm_scenario_key("test_image_text_to_text_pytorch_vs_kv_vs_ort_vs_ai100_qnn") == ""
    assert gcr._vlm_scenario_key("test_custom_replicate_kv_pytorch_vs_ai100") == ""


# ── End-to-end rendered-HTML invariants ──────────────────────────────────────


def test_sample_report_has_expected_top_level_sections(sample_html):
    """Section order enforces the consumer priority: verdict → failures → coverage → drill-down."""
    positions = {
        section: sample_html.find(f"<h2>{section}")
        for section in [
            "Failures &amp; Errors",
            "Coverage by category",
            "Feature coverage matrix",
            "Stage summary",
            "By model / config",
            "Test detail by stage",
            "Slowest",
        ]
    }
    for name, pos in positions.items():
        assert pos != -1, f"section {name!r} missing from rendered report"
    # Failures before Coverage; Coverage before drill-down tables.
    assert positions["Failures &amp; Errors"] < positions["Coverage by category"]
    assert positions["Coverage by category"] < positions["Feature coverage matrix"]
    assert positions["Feature coverage matrix"] < positions["Stage summary"]
    assert positions["Stage summary"] < positions["Test detail by stage"]


def test_all_categories_render_regardless_of_data(sample_html, gcr):
    """Every category tile appears — even ones that produced zero rows in this run."""
    for category in gcr.CATEGORY_ORDER:
        assert f">{category}<" in sample_html or f'cat-label">{category}<' in sample_html, (
            f"Category tile for {category!r} missing"
        )


def test_dense_moe_grouping_in_causal_tile(sample_html):
    """Causal tile must show Dense and MoE group rows and the MoE totals in the summary."""
    # Sample has llama (Dense) + qwen3_moe / mixtral_moe / gpt_oss_moe (MoE).
    assert 'Dense <span class="group-count">' in sample_html
    assert 'MoE <span class="group-count">' in sample_html
    # Summary line for Causal LM shows the Dense/MoE tally.
    assert "Dense 1" in sample_html
    assert "MoE 3" in sample_html


def test_vlm_moe_card_lands_in_moe_group(sample_html):
    """The MoE VLM (Qwen3-VL-30B-A3B-Instruct) must render inside the VLM MoE group."""
    # Locate the VLM tile block.
    tile_start = sample_html.find('cat-label">Vision-Language (VLM)')
    assert tile_start != -1
    # Grab the tile body (up to the next </details>).
    tile_end = sample_html.find("</details>", tile_start)
    tile = sample_html[tile_start:tile_end]
    # Split on the MoE group-row so anything after the marker belongs to the MoE bucket.
    marker = 'MoE <span class="group-count">'
    assert marker in tile, "VLM tile must show a MoE group-row"
    moe_section = tile.split(marker, 1)[1]
    assert "Qwen/Qwen3-VL-30B-A3B-Instruct" in moe_section
    # And the Dense card is in the Dense bucket, not MoE.
    dense_section = tile.split(marker, 1)[0]
    assert "llava-hf/llava-1.5-7b-hf" in dense_section


def test_compile_only_cells_are_visually_distinct(sample_html):
    """A pass in a compile-only column renders with the dashed b-compile badge, not b-pass."""
    assert "b-compile" in sample_html
    # Tooltip warns readers that compile-only ≠ parity confirmation.
    assert "compile-only (no on-device parity check)" in sample_html
    # BF16 · compile-only column header carries the compile-only class.
    assert "class='num scol compile-only'" in sample_html


def test_diffusion_uses_pipeline_list_not_matrix(sample_html):
    """Diffusion tile lists pipeline tests directly (no model rows, no fabricated columns)."""
    tile_start = sample_html.find('cat-label">Diffusion')
    tile_end = sample_html.find("</details>", tile_start)
    tile = sample_html[tile_start:tile_end]
    assert "<th>Pipeline test</th>" in tile
    assert "test_wan_pipeline" in tile
    assert "test_flux_pipeline" in tile
    # Flux is xfail in the fixtures — must surface as XFAIL, not PASS.
    flux_row = re.search(r"test_flux_pipeline.*?</tr>", tile, flags=re.S)
    assert flux_row is not None
    assert "XFAIL" in flux_row.group()


def test_dynamo_stage_row_present_in_stage_summary(sample_html):
    """Dynamo stage appears in the Stage summary table (regression: was silently dropped)."""
    assert "QAIC Dynamo" in sample_html


def test_failures_render_at_top_with_traceback(sample_html):
    """A failing sample test surfaces above coverage tiles with its message body inline."""
    fail_h2 = sample_html.find("<h2>Failures &amp; Errors")
    assert fail_h2 != -1
    # Match text between the Failures h2 and the next h2 (Coverage by category).
    next_h2 = sample_html.find("<h2>", fail_h2 + 1)
    fail_block = sample_html[fail_h2:next_h2]
    assert "test_per_pr_causal_fp16_subfunction_cb" in fail_block
    assert "AssertionError" in fail_block


def test_console_style_row_without_double_colon_does_not_crash(gcr, tmp_path):
    """A JUnit row whose ``name`` has no ``[param]`` (module-level, empty test_fn) must not crash."""
    xml = tmp_path / "tests_log2.xml"
    xml.write_text(
        """<?xml version='1.0'?>
<testsuites>
  <testsuite name='pytest' tests='1' failures='0' errors='0' skipped='0' time='1.0'>
    <testcase classname='tests.transformers.models.causal_lm_models.test_causal_lm_models' name='test_module_level' time='1.0'/>
  </testsuite>
</testsuites>
"""
    )
    out = tmp_path / "ci_report.html"
    assert gcr.main(["--xml-dir", str(tmp_path), "--output", str(out)]) == 0
    assert out.exists()


# ── Regression tests for correctness fixes (Aug 2026 review) ─────────────────


def test_feature_matrix_excludes_vlm_rows(sample_html):
    """Feature-coverage matrix is captioned 'Causal-LM tests only' — VLM cards must not appear.

    Regression: render_coverage_matrix previously iterated all cases, so VLM HF-card params
    (llava-hf/…, Qwen/Qwen3-VL-…) leaked into a table that claimed to be causal-only.
    """
    h2 = sample_html.find("<h2>Feature coverage matrix")
    assert h2 != -1
    next_h2 = sample_html.find("<h2>", h2 + 1)
    section = sample_html[h2:next_h2]
    assert "llava-hf/llava-1.5-7b-hf" not in section
    assert "Qwen/Qwen3-VL-30B-A3B-Instruct" not in section


def test_trailing_bool_suffix_is_stripped(gcr):
    """`llava-hf/llava-1.5-7b-hf-True` and `…-hf` collapse to one row.

    Regression: normalize_model stripped leading non-model tokens only, so trailing
    True/False from parametrize("full_batch", [True, False]) split one card into two rows.
    """
    assert gcr.normalize_model("llava-hf/llava-1.5-7b-hf-True") == "llava-hf/llava-1.5-7b-hf"
    assert gcr.normalize_model("llava-hf/llava-1.5-7b-hf-False") == "llava-hf/llava-1.5-7b-hf"
    assert gcr.normalize_model("llava-hf/llava-1.5-7b-hf") == "llava-hf/llava-1.5-7b-hf"
    # And an MoE VLM with the same suffix still lands in the MoE bucket via moe_group_for.
    normalized = gcr.normalize_model("Qwen/Qwen3-VL-30B-A3B-Instruct-True")
    assert normalized == "Qwen/Qwen3-VL-30B-A3B-Instruct"
    assert gcr.moe_group_for(gcr.CAT_VLM, normalized) == "MoE"


def test_main_returns_zero_on_missing_console_log(gcr, tmp_path):
    """Jenkins post{always{}} contract: main() returns 0 even when inputs are missing."""
    out = tmp_path / "ci_report.html"
    exit_code = gcr.main(
        [
            "--console-log",
            str(tmp_path / "does_not_exist.log"),
            "--output",
            str(out),
        ]
    )
    assert exit_code == 0


def test_main_returns_zero_on_unwritable_output(gcr, tmp_path):
    """Even an unwritable output path must not raise past main()."""
    unwritable = tmp_path / "not-a-dir-file"
    unwritable.write_text("busy")
    # Trying to mkdir a child of a plain file should fail — main() must still return 0.
    out = unwritable / "sub" / "ci_report.html"
    assert gcr.main(["--xml-dir", str(SAMPLE_DIR), "--output", str(out)]) == 0


def test_render_scenario_matrix_is_removed(gcr):
    """The standalone causal Scenario matrix helper was dead code after the reorder."""
    assert not hasattr(gcr, "render_scenario_matrix")


# ── Coverage-section accounting invariants ───────────────────────────────────


def _coverage_section(html):
    start = html.find("<h2>Coverage by category")
    assert start != -1
    return html[start : html.find("<h2>Feature coverage matrix")]


def test_coverage_tiles_reconcile_with_kpi_total(sample_html):
    """Per-tile test counts must sum to the KPI Total — no test may vanish from the section.

    Regression: the roster excluded ``Other``, so the CLI / unit / Dynamo tests
    (5 of 27 in the fixtures) were silently dropped and a reviewer reading the
    tiles would conclude Dynamo never ran.
    """
    section = _coverage_section(sample_html)
    # Matches both "1 test" and "N tests" — the tally is singular for a single test.
    per_tile = [int(n) for n in re.findall(r"(\d+) tests?\b", section)]
    kpi_total = int(re.search(r'Total</span><span class="val">(\d+)', sample_html).group(1))
    assert sum(per_tile) == kpi_total, f"tiles sum to {sum(per_tile)} but KPI Total is {kpi_total}"


def test_other_bucket_surfaces_dynamo_and_cli(sample_html, gcr):
    """The Other tile must name the non-model tests it accounts for."""
    section = _coverage_section(sample_html)
    assert f'cat-label">{gcr.CATEGORY_OTHER}<' in section
    assert "test_dynamo_export_compile" in section, "Dynamo tests must be visible in the coverage section"
    assert "test_cli_infer_causal" in section
    # And it is labelled as a reconciling bucket, not a seventh model category.
    assert "Not a model category" in section


def test_zero_row_category_renders_not_run(gcr, tmp_path):
    """A category with no rows renders a Not Run tile — the roster's load-bearing property.

    Uses a fixture containing ONLY causal tests, so the other five categories are
    genuinely empty. Asserts each one still appears with the Not Run badge; this
    fails if a zero-row category is omitted (which a label-presence check would miss).
    """
    xml = tmp_path / "tests_log2.xml"
    xml.write_text(
        """<?xml version='1.0'?>
<testsuites>
  <testsuite name='pytest' tests='1' failures='0' errors='0' skipped='0' time='1.0'>
    <testcase classname='tests.transformers.models.causal_lm_models.test_causal_lm_models'
              name='test_per_pr_causal_fp16_subfunction_cb[llama_text]' time='1.0'/>
  </testsuite>
</testsuites>
"""
    )
    out = tmp_path / "ci_report.html"
    assert gcr.main(["--xml-dir", str(tmp_path), "--output", str(out)]) == 0
    section = _coverage_section(out.read_text())

    # Causal ran; every other roster category must be present AND marked Not Run.
    assert section.count('<span class="badge b-nr">Not Run</span>') == len(gcr.CATEGORY_ORDER) - 1
    for category in gcr.CATEGORY_ORDER:
        assert f'cat-label">{category}<' in section, f"{category!r} tile omitted entirely"
    # The empty tiles explain themselves rather than reading as a silent pass.
    assert "No test rows arrived for this category" in section
    # Other holds nothing here, so it must not render an empty tile.
    assert f'cat-label">{gcr.CATEGORY_OTHER}<' not in section


def test_diffusion_tile_omits_misleading_zero_model_count(sample_html):
    """Fixture-based diffusion has no model param; the tally must not claim "0 models"."""
    section = _coverage_section(sample_html)
    start = section.find('cat-label">Diffusion')
    assert start != -1
    tally = re.search(r'cat-tally">([^<]+)', section[start:]).group(1)
    assert "0 models" not in tally
    assert "xfail" in tally, f"xfail should stay visible in the tally, got {tally!r}"


def test_pipeline_list_rows_are_module_qualified(gcr):
    """Same test_fn in two modules must not collapse into one row."""

    class _Case:
        seeded = False

        def __init__(self, module_path, test_fn, outcome, param_id=""):
            self.module_path = module_path
            self.test_fn = test_fn
            self.outcome = outcome
            self.param_id = param_id

    body = gcr._render_pipeline_list_body(
        [
            _Case("tests/dynamo/causal_lm/test_causal_lm_dynamo.py", "test_export", gcr.Outcome.PASSED),
            _Case("tests/cloud/test_export.py", "test_export", gcr.Outcome.FAILED),
        ]
    )
    assert "test_causal_lm_dynamo.py::test_export" in body
    assert "test_export.py::test_export" in body


def test_other_tile_shows_the_models_its_tally_counts(sample_html, gcr):
    """A tally advertising "N models" must be verifiable in the tile body.

    Regression: the Other tile claimed "3 models" while its table had only
    Test / Runs / Outcome columns, so the count was unverifiable.
    """
    section = _coverage_section(sample_html)
    start = section.find(f'cat-label">{gcr.CATEGORY_OTHER}<')
    assert start != -1
    tile = section[start : section.find("</details>", start)]
    tally = re.search(r'cat-tally">([^<]+)', tile).group(1)
    if "models" in tally or "model" in tally:
        assert "<th>Models</th>" in tile, "tally counts models but body has no Models column"
        # The Dynamo model_types and the CLI slug are the models being counted.
        for model in ("glm4_moe", "llama", "llama_text"):
            assert model in tile, f"{model!r} counted in the tally but absent from the body"


def test_not_run_feeding_stage_warns_on_the_category_tile(sample_html):
    """A category whose feeding stage was skipped must not read as fully exercised.

    The fixtures omit tests_log2_feature.xml (QAIC Feature), which runs SPD/PLD,
    prefix-caching and sampler tests against causal models — so the Causal LM tile
    must disclose the gap rather than implying its tally is the whole picture.
    """
    section = _coverage_section(sample_html)
    start = section.find('cat-label">Causal LM')
    tile = section[start : section.find("</details>", start)]
    assert "Partial coverage" in tile
    assert "QAIC Feature" in tile


def test_single_test_category_uses_singular_grammar(sample_html):
    """Audio has exactly one test in the fixtures — the tally must not say "1 tests"."""
    section = _coverage_section(sample_html)
    start = section.find('cat-label">Audio')
    tile = section[start : section.find("</details>", start)]
    tally = re.search(r'cat-tally">([^<]+)', tile).group(1)
    assert "1 tests" not in tally
    assert "1 test " in tally or tally.strip().startswith("1 test")

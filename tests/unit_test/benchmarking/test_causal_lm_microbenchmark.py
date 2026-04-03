# -----------------------------------------------------------------------------
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------

import json
from pathlib import Path

import onnx
import pytest

from QEfficient.benchmarking.causal_lm_microbenchmark import (
    BenchmarkSummary,
    RuntimeStats,
    resolve_model_id,
)
from QEfficient.transformers.models.modeling_auto import QEFFAutoModelForCausalLM


def test_resolve_model_id_alias():
    alias, model_id = resolve_model_id("llama")
    assert alias == "llama"
    assert model_id == "hf-internal-testing/tiny-random-LlamaForCausalLM"


def test_enable_benchmark_flag_is_required():
    qeff_model = QEFFAutoModelForCausalLM.from_pretrained("hf-internal-testing/tiny-random-LlamaForCausalLM")
    with pytest.raises(ValueError, match="enable_benchmark=True"):
        qeff_model.get_benchmark_module_specs(mode="decode")


def test_llama_benchmark_inventory():
    qeff_model = QEFFAutoModelForCausalLM.from_pretrained(
        "hf-internal-testing/tiny-random-LlamaForCausalLM",
        enable_benchmark=True,
    )
    specs = qeff_model.get_benchmark_module_specs(mode="decode", ctx_len=8)
    assert [spec.module_name for spec in specs] == ["attention", "decoder"]


def test_gpt_oss_benchmark_inventory_prefill_and_decode():
    qeff_model = QEFFAutoModelForCausalLM.from_pretrained("tiny-random/gpt-oss-bf16", enable_benchmark=True)

    decode_specs = qeff_model.get_benchmark_module_specs(mode="decode", ctx_len=16)
    assert [spec.module_name for spec in decode_specs] == [
        "swa_attention",
        "swa_decoder",
        "full_attention",
        "full_decoder",
        "moe",
    ]

    prefill_specs = qeff_model.get_benchmark_module_specs(mode="prefill", ctx_len=16, enable_chunking=True)
    assert [spec.module_name for spec in prefill_specs] == [
        "prefill_chunked_swa_attention",
        "prefill_chunked_swa_decoder",
        "prefill_chunked_full_attention",
        "prefill_chunked_full_decoder",
        "prefill_chunked_moe",
    ]


def test_llama_export_benchmark_modules_smoke(tmp_path: Path):
    qeff_model = QEFFAutoModelForCausalLM.from_pretrained(
        "hf-internal-testing/tiny-random-LlamaForCausalLM",
        enable_benchmark=True,
    )
    summaries = qeff_model.export_benchmark_modules(
        mode="decode", batch_size=1, seq_len=4, ctx_len=8, export_dir=tmp_path
    )

    assert [summary.module_name for summary in summaries] == ["attention", "decoder"]
    attention_summary = summaries[0]
    assert Path(attention_summary.onnx_path).is_file()

    onnx_model = onnx.load(attention_summary.onnx_path, load_external_data=False)
    input_names = [input_.name for input_ in onnx_model.graph.input]
    output_names = [output.name for output in onnx_model.graph.output]
    assert "attention_mask" in input_names
    assert "attention_output" in output_names
    assert "past_key_RetainedState" in output_names
    assert "past_value_RetainedState" in output_names


def test_gpt_oss_export_moe_smoke(tmp_path: Path):
    qeff_model = QEFFAutoModelForCausalLM.from_pretrained("tiny-random/gpt-oss-bf16", enable_benchmark=True)
    summaries = qeff_model.export_benchmark_modules(
        mode="decode",
        benchmark_type="moe",
        batch_size=1,
        seq_len=4,
        ctx_len=16,
        export_dir=tmp_path,
    )

    assert len(summaries) == 1
    assert summaries[0].module_name == "moe"
    assert Path(summaries[0].onnx_path).is_file()


def test_gpt_oss_export_attention_variants_surface_real_masks(tmp_path: Path):
    qeff_model = QEFFAutoModelForCausalLM.from_pretrained("tiny-random/gpt-oss-bf16", enable_benchmark=True)
    summaries = qeff_model.export_benchmark_modules(
        mode="decode",
        benchmark_type="attention",
        batch_size=1,
        seq_len=4,
        ctx_len=16,
        export_dir=tmp_path,
    )

    onnx_by_module = {
        summary.module_name: onnx.load(summary.onnx_path, load_external_data=False) for summary in summaries
    }

    swa_input_names = [input_.name for input_ in onnx_by_module["swa_attention"].graph.input]
    full_input_names = [input_.name for input_ in onnx_by_module["full_attention"].graph.input]

    assert "sliding_mask" in swa_input_names
    assert "attention_mask" not in swa_input_names
    assert "attention_mask" in full_input_names
    assert "sliding_mask" not in full_input_names


def test_compile_export_only_uses_benchmark_backend():
    qeff_model = QEFFAutoModelForCausalLM.from_pretrained(
        "hf-internal-testing/tiny-random-LlamaForCausalLM",
        enable_benchmark=True,
    )
    manifest_path = qeff_model.compile(prefill_only=False, prefill_seq_len=4, ctx_len=8, export_only=True)

    manifest = json.loads(Path(manifest_path).read_text())
    assert [summary["module_name"] for summary in manifest["summaries"]] == ["attention", "decoder"]
    assert all(summary["qpc_path"] is None for summary in manifest["summaries"])


def test_compile_decode_benchmark_smoke():
    qeff_model = QEFFAutoModelForCausalLM.from_pretrained(
        "hf-internal-testing/tiny-random-LlamaForCausalLM",
        enable_benchmark=True,
    )
    manifest_path = qeff_model.compile(prefill_seq_len=4, ctx_len=8)

    manifest = json.loads(Path(manifest_path).read_text())
    assert [summary["module_name"] for summary in manifest["summaries"]] == ["attention", "decoder"]
    assert [summary["mode"] for summary in manifest["summaries"]] == ["both", "both"]
    assert all(summary["qpc_path"] is not None for summary in manifest["summaries"])
    assert all(Path(summary["qpc_path"]).is_dir() for summary in manifest["summaries"])


def test_generate_benchmark_mode_writes_json_report(monkeypatch, tmp_path: Path, capsys):
    qeff_model = QEFFAutoModelForCausalLM.from_pretrained(
        "hf-internal-testing/tiny-random-LlamaForCausalLM",
        enable_benchmark=True,
    )
    qeff_model._benchmark_manifest = type(
        "DummyManifest",
        (),
        {
            "prefill_only": False,
            "enable_chunking": False,
            "batch_size": 1,
            "seq_len": 4,
            "ctx_len": 8,
            "num_cores": 16,
            "num_devices": 1,
            "warmup_runs": 2,
            "benchmark_runs": 3,
            "summaries": [],
        },
    )()
    summary = BenchmarkSummary(
        benchmark_type="attention",
        module_name="attention",
        mode="decode",
        model_name="llama",
        model_id="hf-internal-testing/tiny-random-LlamaForCausalLM",
        architecture="llama",
        layer_index=0,
        batch_size=1,
        seq_len=4,
        ctx_len=8,
        resolved_dims={"hidden_size": 16},
        input_shapes={"hidden_states": [1, 4, 16]},
        output_shapes={"attention_output": [1, 4, 16]},
        onnx_path=str(tmp_path / "attention.onnx"),
        qpc_path="/tmp/qpc",
        prefill_runtime=None,
        seed_prefill_ms=0.12,
        first_decode_ms=0.23,
        decode_runtime=RuntimeStats(
            iterations=3,
            mean_ms=1.5,
            min_ms=1.1,
            max_ms=1.9,
            total_ms=4.5,
            tokens_per_second=666.67,
        ),
    )

    def fake_generate_benchmark_report(model, *args, **kwargs):
        model._benchmark_report_path = str(tmp_path / "benchmark-report.json")
        Path(model._benchmark_report_path).write_text(
            json.dumps({"summaries": [{"module_name": "attention", "decode_runtime": {"mean_ms": 1.5}}]}),
            encoding="utf-8",
        )
        return [summary]

    monkeypatch.setattr(
        "QEfficient.benchmarking.causal_lm_microbenchmark.generate_benchmark_report",
        fake_generate_benchmark_report,
    )

    returned = qeff_model.generate(tokenizer=None, prompts=[])

    report = json.loads(Path(returned).read_text())
    assert report["summaries"][0]["module_name"] == "attention"
    assert report["summaries"][0]["decode_runtime"]["mean_ms"] == 1.5
    stdout = capsys.readouterr().out
    assert stdout == ""

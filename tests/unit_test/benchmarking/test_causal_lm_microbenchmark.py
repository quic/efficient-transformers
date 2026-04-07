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
    BenchmarkManifest,
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
    assert [spec.module_name for spec in specs] == ["attention", "mlp"]


def test_gpt_oss_benchmark_inventory_prefill_and_decode():
    qeff_model = QEFFAutoModelForCausalLM.from_pretrained("tiny-random/gpt-oss-bf16", enable_benchmark=True)

    decode_specs = qeff_model.get_benchmark_module_specs(mode="decode", ctx_len=16)
    assert [spec.module_name for spec in decode_specs] == ["swa_attention", "full_attention", "moe"]

    prefill_specs = qeff_model.get_benchmark_module_specs(mode="prefill", ctx_len=16, enable_chunking=True)
    assert [spec.module_name for spec in prefill_specs] == [
        "prefill_chunked_swa_attention",
        "prefill_chunked_full_attention",
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

    assert [summary.module_name for summary in summaries] == ["attention", "mlp"]
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


def test_gpt_oss_chunked_prefill_swa_specialization_keeps_config_sliding_window():
    qeff_model = QEFFAutoModelForCausalLM.from_pretrained("tiny-random/gpt-oss-bf16", enable_benchmark=True)
    specs = qeff_model.get_benchmark_module_specs(mode="prefill", ctx_len=128, enable_chunking=True)
    swa_spec = next(spec for spec in specs if spec.module_name == "prefill_chunked_swa_attention")

    specialization = swa_spec.wrapper.specialization_values(batch_size=1, seq_len=32, ctx_len=128, mode="prefill")
    dynamic_axes = swa_spec.wrapper.dynamic_axes("attention_output")

    assert specialization["ctx_len"] == 160
    assert specialization["sliding_window"] == 128
    assert dynamic_axes["sliding_mask"][3] == "ctx_len"
    assert dynamic_axes["past_key"][2] == "ctx_len"


def test_compile_export_only_uses_benchmark_backend():
    qeff_model = QEFFAutoModelForCausalLM.from_pretrained(
        "hf-internal-testing/tiny-random-LlamaForCausalLM",
        enable_benchmark=True,
    )
    manifest_path = qeff_model.compile(prefill_only=False, prefill_seq_len=4, ctx_len=8, export_only=True)

    assert Path(manifest_path).is_file()
    payload = json.loads(Path(manifest_path).read_text())
    assert [summary["module_name"] for summary in payload["summaries"]] == ["attention", "mlp"]
    assert all(summary["qpc_path"] is None for summary in payload["summaries"])


def test_compile_decode_benchmark_smoke():
    qeff_model = QEFFAutoModelForCausalLM.from_pretrained(
        "hf-internal-testing/tiny-random-LlamaForCausalLM",
        enable_benchmark=True,
    )
    manifest_path = qeff_model.compile(prefill_seq_len=4, ctx_len=8)

    assert Path(manifest_path).is_file()
    payload = json.loads(Path(manifest_path).read_text())
    assert [summary["module_name"] for summary in payload["summaries"]] == ["attention", "mlp"]
    assert [summary["mode"] for summary in payload["summaries"]] == ["both", "both"]
    assert all(summary["qpc_path"] is not None for summary in payload["summaries"])
    assert all(Path(summary["qpc_path"]).is_dir() for summary in payload["summaries"])


def test_generate_benchmark_mode_prints_table(monkeypatch, capsys):
    qeff_model = QEFFAutoModelForCausalLM.from_pretrained(
        "hf-internal-testing/tiny-random-LlamaForCausalLM",
        enable_benchmark=True,
    )
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
        onnx_path="/tmp/attention.onnx",
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

    def fake_generate_benchmark_report(*args, **kwargs):
        return [summary]

    monkeypatch.setattr(
        "QEfficient.benchmarking.causal_lm_microbenchmark.generate_benchmark_report",
        fake_generate_benchmark_report,
    )
    qeff_model._benchmark_manifest = BenchmarkManifest(
        prefill_only=False,
        enable_chunking=False,
        batch_size=1,
        seq_len=4,
        ctx_len=8,
        num_cores=16,
        num_devices=1,
        warmup_runs=2,
        benchmark_runs=3,
        summaries=[summary],
    )

    returned = qeff_model.generate(tokenizer=None, prompts=[])

    assert Path(returned).is_file()
    stdout = capsys.readouterr().out
    assert "Mode" in stdout
    assert "Module" in stdout
    assert "attention" in stdout
    assert "decode" in stdout
    assert "inputs:" in stdout
    assert "outputs:" in stdout
    assert "prefill_stats:" not in stdout


def test_benchmark_mode_uses_config_only_stub():
    qeff_model = QEFFAutoModelForCausalLM.from_pretrained(
        "hf-internal-testing/tiny-random-LlamaForCausalLM",
        enable_benchmark=True,
    )
    assert qeff_model.enable_benchmark is True
    assert qeff_model.model.__class__.__name__ == "LlamaForCausalLM"
    assert not hasattr(qeff_model.model, "model")

# -----------------------------------------------------------------------------
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------

"""Local QEff-owned MXFP6 weight-free export smoke test.

This script creates a tiny Llama checkpoint locally, exports it through the
weight-free Dynamo path with QEff-owned MXFP6 enabled, and prints the resulting
ONNX/WeightSpec structure. It does not download from Hugging Face.

Example:
    python examples/dynamo/causal_lm/local_mxfp6_weight_free_export.py \
        --work-dir /tmp/qeff_mxfp6_smoke
"""

from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path

import onnx
from transformers import LlamaConfig, LlamaForCausalLM

from QEfficient.exporter.weight_free.weight_spec import load_weight_spec, resolve_weight_spec_path
from QEfficient.transformers.models.modeling_auto import QEFFAutoModelForCausalLM


def _make_config(args: argparse.Namespace) -> LlamaConfig:
    return LlamaConfig(
        num_hidden_layers=args.num_hidden_layers,
        num_attention_heads=args.num_attention_heads,
        num_key_value_heads=args.num_key_value_heads,
        hidden_size=args.hidden_size,
        intermediate_size=args.intermediate_size,
        vocab_size=args.vocab_size,
        max_position_embeddings=args.max_position_embeddings,
    )


def _load_onnx(onnx_path: Path):
    return onnx.load(str(onnx_path), load_external_data=False)


def _count_nodes(model, op_type: str) -> int:
    return sum(1 for node in model.graph.node if node.op_type == op_type)


def _count_function_body_nodes(model, op_type: str) -> int:
    return sum(1 for function in model.functions for node in function.node if node.op_type == op_type)


def _count_functions(model, op_type: str) -> int:
    return sum(1 for function in model.functions if function.name == op_type)


def _decoder_function_signatures(model) -> list[dict]:
    return [
        {
            "domain": function.domain,
            "name": function.name,
            "inputs": list(function.input),
        }
        for function in model.functions
        if function.domain != "com.qti.aisw.onnx"
    ]


def _mxfp6_graph_input_summary(model) -> dict:
    graph_inputs = {value_info.name for value_info in model.graph.input}
    return {
        "packed_graph_inputs": sum(1 for name in graph_inputs if name.endswith(".mxfp6_packed")),
        "scale_graph_inputs": sum(1 for name in graph_inputs if name.endswith(".mxfp6_scale")),
    }


def _has_lm_head_component(name: str) -> bool:
    return "lm_head" in [component for component in name.replace("/", ".").split(".") if component]


def _lm_head_summary(model, spec) -> dict:
    graph_inputs = {value_info.name for value_info in model.graph.input}
    spec_inputs = [entry for entry in spec.inputs if _has_lm_head_component(entry.name)]
    return {
        "graph_inputs": sorted(name for name in graph_inputs if _has_lm_head_component(name)),
        "mxfp6_graph_inputs": sorted(
            name
            for name in graph_inputs
            if _has_lm_head_component(name) and name.endswith((".mxfp6_packed", ".mxfp6_scale"))
        ),
        "spec_inputs": [
            {
                "name": entry.name,
                "key": entry.location.key,
                "role": entry.role,
            }
            for entry in spec_inputs
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create and export a tiny local Llama checkpoint with QEff-owned MXFP6 weight-free export.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--work-dir", type=Path, default=None, help="Directory for local checkpoint/export artifacts")
    parser.add_argument("--scale-dtype", default="float16", help="MXFP6 scale dtype alias")
    parser.add_argument("--use-onnx-subfunctions", action="store_true", help="Enable ONNX subfunction export")
    parser.add_argument("--num-hidden-layers", type=int, default=1)
    parser.add_argument("--num-attention-heads", type=int, default=4)
    parser.add_argument("--num-key-value-heads", type=int, default=4)
    parser.add_argument("--hidden-size", type=int, default=32)
    parser.add_argument("--intermediate-size", type=int, default=64)
    parser.add_argument("--vocab-size", type=int, default=128)
    parser.add_argument("--max-position-embeddings", type=int, default=64)
    args = parser.parse_args()

    root = args.work_dir or Path(tempfile.mkdtemp(prefix="qeff_mxfp6_local_"))
    root = root.expanduser().resolve()
    model_dir = root / "model"
    export_dir = root / "export"
    model_dir.mkdir(parents=True, exist_ok=True)
    export_dir.mkdir(parents=True, exist_ok=True)

    model = LlamaForCausalLM(_make_config(args)).eval()
    model.save_pretrained(model_dir, safe_serialization=True)

    qeff_model = QEFFAutoModelForCausalLM.from_pretrained(
        str(model_dir),
        weight_free=True,
        mxfp6=True,
        mxfp6_scale_dtype=args.scale_dtype,
    )
    qeff_model.model.eval()
    export_result = qeff_model.export(
        export_dir,
        use_onnx_subfunctions=args.use_onnx_subfunctions,
        offload_pt_weights=False,
    )
    onnx_path = Path(export_result[-1] if isinstance(export_result, (list, tuple)) else export_result)
    weight_spec_path = resolve_weight_spec_path(onnx_path)
    spec = load_weight_spec(weight_spec_path)
    onnx_model = _load_onnx(onnx_path)

    role_counts = {}
    for spec_input in spec.inputs:
        role_counts[spec_input.role] = role_counts.get(spec_input.role, 0) + 1

    summary = {
        "work_dir": str(root),
        "onnx_path": str(onnx_path),
        "weight_spec_path": str(weight_spec_path),
        "weight_spec_version": spec.version,
        "weight_spec_role_counts": role_counts,
        "top_level_dequantize_linear_nodes": _count_nodes(onnx_model, "DequantizeLinear"),
        "top_level_unpack_mxfp6_nodes": _count_nodes(onnx_model, "UnpackMxfp6"),
        "function_body_dequantize_linear_nodes": _count_function_body_nodes(onnx_model, "DequantizeLinear"),
        "function_body_unpack_mxfp6_nodes": _count_function_body_nodes(onnx_model, "UnpackMxfp6"),
        "unpack_mxfp6_functions": _count_functions(onnx_model, "UnpackMxfp6"),
        "decoder_function_signatures": _decoder_function_signatures(onnx_model),
        "matmul_nodes": _count_nodes(onnx_model, "MatMul"),
        "mxfp6_graph_input_summary": _mxfp6_graph_input_summary(onnx_model),
        "lm_head_summary": _lm_head_summary(onnx_model, spec),
    }
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

# -----------------------------------------------------------------------------
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------

"""Verify QEFFAutoModelForCausalLM.export(paged_kv=True) produces a paged graph.

Drives the real export pipeline on a tiny Qwen2 and asserts the exported ONNX:
  * declares block_table (and attention_mask) as graph inputs,
  * shapes past_key.*/past_value.* as a block pool (dim2 == page_size),
  * contains the paged scatter/gather ops.

AIC compile of this graph is box-gated; this test covers the export-plumbing
wiring on CPU.
"""

import tempfile

import pytest
import torch

pytest.importorskip("QEfficient")

import onnx  # noqa: E402
from transformers.models.qwen2.modeling_qwen2 import Qwen2Config, Qwen2ForCausalLM  # noqa: E402

from QEfficient.transformers.models.modeling_auto import QEFFAutoModelForCausalLM  # noqa: E402


def test_export_paged_kv_graph():
    cfg = Qwen2Config(
        vocab_size=64, hidden_size=32, intermediate_size=64, num_hidden_layers=2,
        num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=128,
        rms_norm_eps=1e-6, tie_word_embeddings=False,
    )
    cfg.torch_dtype = torch.float32
    torch.manual_seed(0)
    hf = Qwen2ForCausalLM(cfg)
    qeff = QEFFAutoModelForCausalLM(hf, continuous_batching=True)

    # Force the classic TorchScript exporter (default on the QAIC host's torch 2.7;
    # this CPU venv has torch 2.12 whose dynamo-default exporter is incompatible with
    # the pinned onnxscript). Product code is unchanged.
    import torch.onnx as _tonnx

    _orig_export = _tonnx.export

    def _classic_export(*a, **k):
        k.setdefault("dynamo", False)
        return _orig_export(*a, **k)

    _tonnx.export = _classic_export
    try:
        page_size = 8
        with tempfile.TemporaryDirectory() as d:
            onnx_path = qeff.export(export_dir=d, paged_kv=True, paged_block_size=page_size)
            model = onnx.load(onnx_path, load_external_data=False)
    finally:
        _tonnx.export = _orig_export

    graph_inputs = {i.name: i for i in model.graph.input}
    assert "block_table" in graph_inputs, f"block_table not exported; inputs={list(graph_inputs)}"
    assert "attention_mask" in graph_inputs, "attention_mask not exported for paged mode"
    assert any(n.startswith("past_key.") for n in graph_inputs), "no past_key.* inputs"

    # past_key.0 must be a block pool: dim2 == page_size (the paged page dim).
    pk0 = graph_inputs["past_key.0"]
    dims = pk0.type.tensor_type.shape.dim
    page_dim = dims[2]
    # dynamic axis -> dim_param "page_size"; or concrete page_size value.
    assert page_dim.dim_param == "page_size" or page_dim.dim_value == page_size, (
        f"past_key.0 dim2 is not the page dim: {page_dim}"
    )

    ops = {n.op_type for n in model.graph.node}
    for f in model.functions:
        ops |= {n.op_type for n in f.node}
    assert any("Scatter" in o for o in ops) and any("Gather" in o for o in ops), f"paged ops missing: {sorted(ops)}"

    print(f"[metrics] export_paged_kv: block_table+attention_mask are graph inputs; "
          f"past_key.0 page dim={page_dim.dim_param or page_dim.dim_value}")


if __name__ == "__main__":
    from _metrics import measure

    with measure("test_paged_export_plumbing"):
        test_export_paged_kv_graph()
    print("PAGED EXPORT PLUMBING: PASS")

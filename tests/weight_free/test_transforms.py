# -----------------------------------------------------------------------------
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------

"""
Unit tests for the dynamo-specific transforms used by the weight-free export pipeline.

These are the same transforms tested in tests/dynamo/test_transforms.py.
They are duplicated here because PruneFakeInitializersTransform is specifically
relevant to weight-free export (it removes meta/fake tensor initializers that only
appear when the model was traced on meta device).

CPU-only. No QAIC hardware required.
"""

from __future__ import annotations

import json
from types import SimpleNamespace
from unittest.mock import MagicMock

import onnx_ir as ir
import torch
from onnx import TensorProto, helper
from safetensors import safe_open
from safetensors.torch import save_file
from transformers import LlamaConfig, LlamaForCausalLM

from QEfficient.base.checkpoint_transforms import CHECKPOINT_PREPARED_MANIFEST, CheckpointTransformPipeline
from QEfficient.base.onnx_transforms import (
    PreserveNestedCacheRetainedStateTransform,
    PruneFakeInitializersTransform,
    RenameRepeatedSubgraphTransform,
)
from QEfficient.exporter.weight_free import checkpoint_key_resolver
from QEfficient.exporter.weight_free.checkpoint_key_resolver import find_checkpoint_key
from QEfficient.exporter.weight_free.checkpoint_transforms import (
    DtypeConversionCheckpointTransform,
    GraniteMoeFusedExpertSplitCheckpointTransform,
    MoEExpertStackingCheckpointTransform,
    MoEFusedExpertSplitCheckpointTransform,
)
from QEfficient.transformers.models.llama.modeling_llama import QEffLlamaDecoderLayer
from QEfficient.transformers.models.modeling_auto import QEFFAutoModelForCausalLM
from QEfficient.utils.export_utils import _generate_export_hash
from QEfficient.utils.torch_patches import temporarily_enable_nested_compile_regions

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def make_tiny_llama():
    cfg = LlamaConfig(
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=2,
        hidden_size=64,
        intermediate_size=128,
        vocab_size=500,
        max_position_embeddings=32,
    )
    model = LlamaForCausalLM(cfg).eval()
    return model, cfg


def _make_minimal_onnx_with_repeated_subgraphs(num_layers: int = 2, scatter_count_per_fn: int = 2):
    """
    Build a minimal ONNX ModelProto that mimics dynamo's repeated-subgraph output:
      - graph has num_layers call nodes (one per layer), each referencing repeated_subgraphN
      - each function contains scatter_count_per_fn CtxScatter nodes
      - graph outputs include past_key/value _RetainedState placeholders (dangling)
    """
    functions = []
    call_nodes = []
    graph_outputs = []
    graph_inputs = []

    for i in range(num_layers):
        fn_name = f"repeated_subgraph{i}"

        scatter_nodes = []
        fn_outputs = []
        for j in range(scatter_count_per_fn):
            kind = "key" if j == 0 else "value"
            scatter_out = f"scatter_{kind}_{i}"
            scatter_node = helper.make_node(
                "CtxScatter",
                inputs=[f"past_{kind}.{i}", f"new_{kind}_{i}", "position_ids"],
                outputs=[scatter_out],
                domain="qti.aisw",
            )
            scatter_nodes.append(scatter_node)
            fn_outputs.append(scatter_out)

        fn = helper.make_function(
            domain="",
            fname=fn_name,
            inputs=[f"past_key.{i}", f"past_value.{i}", f"hidden_{i}", "position_ids"],
            outputs=fn_outputs,
            nodes=scatter_nodes,
            opset_imports=[helper.make_opsetid("", 17), helper.make_opsetid("qti.aisw", 1)],
        )
        functions.append(fn)

        retained_key = f"past_key.{i}_RetainedState"
        retained_val = f"past_value.{i}_RetainedState"
        call_node = helper.make_node(
            fn_name,
            inputs=[f"past_key.{i}", f"past_value.{i}", f"hidden_{i}", "position_ids"],
            outputs=[],
            domain="",
        )
        call_nodes.append(call_node)

        graph_outputs.append(helper.make_tensor_value_info(retained_key, TensorProto.FLOAT, None))
        graph_outputs.append(helper.make_tensor_value_info(retained_val, TensorProto.FLOAT, None))

        graph_inputs.append(helper.make_tensor_value_info(f"past_key.{i}", TensorProto.FLOAT, None))
        graph_inputs.append(helper.make_tensor_value_info(f"past_value.{i}", TensorProto.FLOAT, None))
        graph_inputs.append(helper.make_tensor_value_info(f"hidden_{i}", TensorProto.FLOAT, None))

    graph_inputs.append(helper.make_tensor_value_info("position_ids", TensorProto.INT64, None))

    graph = helper.make_graph(call_nodes, "test_graph", graph_inputs, graph_outputs)
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    for fn in functions:
        model.functions.append(fn)
    return model


def _write_safetensors_checkpoint(root, tensors):
    shard_name = "model.safetensors"
    save_file({key: tensor.contiguous() for key, tensor in tensors.items()}, str(root / shard_name))
    (root / "model.safetensors.index.json").write_text(
        json.dumps({"metadata": {}, "weight_map": {key: shard_name for key in tensors}}, indent=2)
    )


def _load_prepared_tensors(root):
    index = json.loads((root / "model.safetensors.index.json").read_text())["weight_map"]
    loaded = {}
    for key, shard_name in index.items():
        with safe_open(str(root / shard_name), framework="pt") as handle:
            loaded[key] = handle.get_tensor(key)
    return loaded


# ---------------------------------------------------------------------------
# Test checkpoint layout transforms
# ---------------------------------------------------------------------------


class TestWeightFreeCheckpointTransforms:
    def test_checkpoint_pipeline_rebuilds_when_source_changes(self, tmp_path):
        src = tmp_path / "src"
        out = tmp_path / "out"
        src.mkdir()
        _write_safetensors_checkpoint(src, {"weight": torch.ones(2, dtype=torch.float16)})

        pipeline = CheckpointTransformPipeline([DtypeConversionCheckpointTransform])
        prepared = pipeline.apply(src, out, target_dtype=torch.float32)

        assert prepared == out
        assert (out / CHECKPOINT_PREPARED_MANIFEST).is_file()
        torch.testing.assert_close(_load_prepared_tensors(out)["weight"], torch.ones(2, dtype=torch.float32))

        _write_safetensors_checkpoint(src, {"weight": torch.ones(3, dtype=torch.float16)})
        prepared = pipeline.apply(src, out, target_dtype=torch.float32)

        assert prepared == out
        torch.testing.assert_close(_load_prepared_tensors(out)["weight"], torch.ones(3, dtype=torch.float32))

    def test_stacks_per_expert_weights_to_moe_weights(self, tmp_path):
        src = tmp_path / "src"
        out = tmp_path / "out"
        src.mkdir()
        prefix = "model.layers.0.block_sparse_moe"

        gate_0 = torch.arange(8, dtype=torch.float32).reshape(2, 4)
        gate_1 = gate_0 + 100
        up_0 = gate_0 + 200
        up_1 = gate_0 + 300
        down_0 = torch.arange(8, dtype=torch.float32).reshape(4, 2) + 400
        down_1 = down_0 + 100
        _write_safetensors_checkpoint(
            src,
            {
                f"{prefix}.experts.0.w1.weight": gate_0,
                f"{prefix}.experts.0.w3.weight": up_0,
                f"{prefix}.experts.0.w2.weight": down_0,
                f"{prefix}.experts.1.w1.weight": gate_1,
                f"{prefix}.experts.1.w3.weight": up_1,
                f"{prefix}.experts.1.w2.weight": down_1,
                "model.embed_tokens.weight": torch.ones(2, 4),
            },
        )

        changed = MoEExpertStackingCheckpointTransform.apply(
            src,
            out,
            target_dtype=torch.float32,
            max_workers_scan=1,
            max_workers_layers=1,
            max_workers_base=1,
        )

        assert changed
        tensors = _load_prepared_tensors(out)
        torch.testing.assert_close(
            tensors[f"{prefix}.moe_weights.gate"],
            torch.stack([gate_0, gate_1]).transpose(1, 2),
        )
        torch.testing.assert_close(
            tensors[f"{prefix}.moe_weights.up"],
            torch.stack([up_0, up_1]).transpose(1, 2),
        )
        torch.testing.assert_close(
            tensors[f"{prefix}.moe_weights.down"],
            torch.stack([down_0, down_1]).transpose(1, 2),
        )
        assert f"{prefix}.experts.gate_proj" not in tensors
        assert f"{prefix}.experts.down_proj_t" not in tensors

    def test_splits_dim2_fused_experts_with_bias_to_moe_weights(self, tmp_path):
        src = tmp_path / "src"
        out = tmp_path / "out"
        src.mkdir()
        prefix = "model.layers.0.mlp.experts"
        moe_prefix = "model.layers.0.mlp.moe_weights"
        gate = torch.full((2, 3, 4), 1.0)
        up = torch.full((2, 3, 4), 2.0)
        gate_up = torch.empty(2, 3, 8)
        gate_up[..., 0::2] = gate
        gate_up[..., 1::2] = up
        down = torch.arange(24, dtype=torch.float32).reshape(2, 4, 3)
        gate_bias = torch.full((2, 4), 3.0)
        up_bias = torch.full((2, 4), 4.0)
        gate_up_bias = torch.empty(2, 8)
        gate_up_bias[..., 0::2] = gate_bias
        gate_up_bias[..., 1::2] = up_bias
        down_bias = torch.full((2, 3), 5.0)
        _write_safetensors_checkpoint(
            src,
            {
                f"{prefix}.gate_up_proj": gate_up,
                f"{prefix}.down_proj": down,
                f"{prefix}.gate_up_proj_bias": gate_up_bias,
                f"{prefix}.down_proj_bias": down_bias,
            },
        )

        changed = MoEFusedExpertSplitCheckpointTransform.apply(src, out, target_dtype=torch.float32)

        assert changed
        tensors = _load_prepared_tensors(out)
        torch.testing.assert_close(tensors[f"{moe_prefix}.gate"], gate)
        torch.testing.assert_close(tensors[f"{moe_prefix}.up"], up)
        torch.testing.assert_close(tensors[f"{moe_prefix}.down"], down)
        torch.testing.assert_close(tensors[f"{moe_prefix}.gate_bias"], gate_bias)
        torch.testing.assert_close(tensors[f"{moe_prefix}.up_bias"], up_bias)
        torch.testing.assert_close(tensors[f"{moe_prefix}.down_bias"], down_bias)

    def test_splits_granitemoe_fused_parallel_experts_to_moe_weights(self, tmp_path):
        src = tmp_path / "src"
        out = tmp_path / "out"
        src.mkdir()
        prefix = "model.layers.0.block_sparse_moe"
        gate = torch.arange(8, dtype=torch.float32).reshape(2, 4)
        up = gate + 100
        gate_up = torch.cat((gate, up), dim=0).reshape(1, 4, 4)
        down = torch.arange(8, dtype=torch.float32).reshape(1, 4, 2)
        _write_safetensors_checkpoint(
            src,
            {
                f"{prefix}.input_linear.weight": gate_up,
                f"{prefix}.output_linear.weight": down,
            },
        )

        changed = GraniteMoeFusedExpertSplitCheckpointTransform.apply(src, out, target_dtype=torch.float32)

        assert changed
        tensors = _load_prepared_tensors(out)
        torch.testing.assert_close(tensors[f"{prefix}.moe_weights.gate"], gate_up[:, :2, :].transpose(1, 2))
        torch.testing.assert_close(tensors[f"{prefix}.moe_weights.up"], gate_up[:, 2:, :].transpose(1, 2))
        torch.testing.assert_close(tensors[f"{prefix}.moe_weights.down"], down.transpose(1, 2))

    def test_resolver_accepts_moe_weight_aliases(self):
        checkpoint_index = {
            "model.layers.0.mlp.moe_weights.gate": "model.safetensors",
            "model.layers.1.mlp.experts.moe_weights.up": "model.safetensors",
            "model.layers.2.block_sparse_moe.experts.down_proj_t": "model.safetensors",
        }
        backbone = MagicMock()
        backbone.base_model_prefix = "model"

        assert (
            find_checkpoint_key("model.layers.0.mlp.experts.moe_weights.gate", checkpoint_index, backbone)
            == "model.layers.0.mlp.moe_weights.gate"
        )
        assert (
            find_checkpoint_key("model.layers.1.mlp.moe_weights.up", checkpoint_index, backbone)
            == "model.layers.1.mlp.experts.moe_weights.up"
        )
        assert (
            find_checkpoint_key("model.layers.2.block_sparse_moe.moe_weights.down", checkpoint_index, backbone)
            == "model.layers.2.block_sparse_moe.experts.down_proj_t"
        )

    def test_promotes_embed_tokens_for_tied_model(self, tmp_path, monkeypatch):
        """When tie_word_embeddings=True, torch.export deduplicates tied weights —
        only model.embed_tokens.weight appears as an ONNX initializer, never
        lm_head.weight. Verify the canonical name is promoted correctly."""
        src = tmp_path / "src"
        src.mkdir()
        tied_weight = torch.arange(12, dtype=torch.float32).reshape(4, 3)
        _write_safetensors_checkpoint(src, {"model.embed_tokens.weight": tied_weight})

        class TiedModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.model = torch.nn.Module()
                self.model.embed_tokens = torch.nn.Embedding(4, 3)
                self.lm_head = torch.nn.Linear(3, 4, bias=False)
                self.lm_head.weight = self.model.embed_tokens.weight

        initializer = SimpleNamespace(shape=tied_weight.shape, dtype=ir.DataType.FLOAT)
        # Realistic: torch.export deduplicates tied weights — only the canonical
        # name (model.embed_tokens.weight) appears as an ONNX initializer.
        graph = SimpleNamespace(initializers={"model.embed_tokens.weight": initializer}, inputs=[])
        onnx_program = SimpleNamespace(model=SimpleNamespace(graph=graph))
        monkeypatch.setattr(
            checkpoint_key_resolver.ir,
            "Value",
            lambda name, shape, type: SimpleNamespace(name=name, shape=shape, type=type),
        )

        spec = checkpoint_key_resolver.promote_initializers_and_build_spec(
            onnx_program=onnx_program,
            model_ref=str(src),
            model_name="tiny-tied",
            qeff_model=SimpleNamespace(model=TiedModel()),
        )

        assert "model.embed_tokens.weight" not in graph.initializers
        assert [v.name for v in graph.inputs] == ["model.embed_tokens.weight"]
        assert spec.inputs[0].name == "model.embed_tokens.weight"
        assert spec.inputs[0].location.key == "model.embed_tokens.weight"


def _fake_export(
    self,
    example_inputs,
    output_names,
    dynamic_axes,
    onnx_transform_kwargs=None,
    export_dir=None,
    dynamo=False,
    dynamic_shapes=None,
    **export_kwargs,
):
    pass


class TestWeightFreeExportHash:
    def test_weight_free_export_hash_differs_from_regular_dynamo(self):
        config = SimpleNamespace(to_diff_dict=lambda: {"model_type": "llama"})
        common_model = SimpleNamespace(
            model=SimpleNamespace(config=config),
            hash_params={"pretrained_model_name_or_path": "tiny"},
            _use_onnx_subfunctions=False,
            _weight_free=False,
        )
        weight_free_model = SimpleNamespace(
            model=SimpleNamespace(config=config),
            hash_params={"pretrained_model_name_or_path": "tiny"},
            _use_onnx_subfunctions=False,
            _weight_free=True,
        )
        common_kwargs = {
            "example_inputs": {"input_ids": torch.ones(1, 2, dtype=torch.int64)},
            "output_names": ["logits"],
            "dynamic_axes": {"input_ids": {0: "batch_size"}},
            "dynamo": True,
        }

        regular_hash, regular_params = _generate_export_hash(
            common_model,
            (),
            dict(common_kwargs),
            _fake_export,
        )
        weight_free_hash, weight_free_params = _generate_export_hash(
            weight_free_model,
            (),
            dict(common_kwargs),
            _fake_export,
        )

        assert regular_hash != weight_free_hash
        assert "weight_free" not in regular_params
        assert weight_free_params["weight_free"] is True


# ---------------------------------------------------------------------------
# TestTemporarilyEnableNestedCompileRegions
# ---------------------------------------------------------------------------


class TestTemporarilyEnableNestedCompileRegions:
    def test_patches_decoder_layers_and_restores(self):
        model_hf, _ = make_tiny_llama()
        qeff_model = QEFFAutoModelForCausalLM(model_hf)
        inner_model = qeff_model.model

        decoder_layers = [m for m in inner_model.modules() if isinstance(m, QEffLlamaDecoderLayer)]
        assert len(decoder_layers) > 0, "No QEffLlamaDecoderLayer found in wrapped model"

        original_qualnames = [getattr(m.forward, "__qualname__", "") for m in decoder_layers]

        with temporarily_enable_nested_compile_regions(inner_model, target_classes=[QEffLlamaDecoderLayer]):
            for m in decoder_layers:
                fwd = getattr(m, "forward", None)
                qualname = getattr(fwd, "__qualname__", "")
                assert (
                    "mark_compile_region" in qualname or "nested_compile_region" in qualname or "inner" in qualname
                ), (
                    f"Expected nested_compile_region wrapper on {m.__class__.__name__}.forward, "
                    f"got qualname: {qualname!r}"
                )

        for m, orig_qn in zip(decoder_layers, original_qualnames):
            fwd = getattr(m, "forward", None)
            qualname = getattr(fwd, "__qualname__", "")
            assert qualname == orig_qn, f"forward qualname not restored: expected {orig_qn!r}, got {qualname!r}"

    def test_noop_when_already_wrapped(self):
        model_hf, _ = make_tiny_llama()
        qeff_model = QEFFAutoModelForCausalLM(model_hf)
        inner_model = qeff_model.model

        decoder_layers = [m for m in inner_model.modules() if isinstance(m, QEffLlamaDecoderLayer)]

        with temporarily_enable_nested_compile_regions(inner_model, target_classes=[QEffLlamaDecoderLayer]):
            wrapped_forwards_first = [id(m.forward) for m in decoder_layers]

            with temporarily_enable_nested_compile_regions(inner_model, target_classes=[QEffLlamaDecoderLayer]):
                wrapped_forwards_second = [id(m.forward) for m in decoder_layers]

        assert len(wrapped_forwards_first) == len(wrapped_forwards_second)


# ---------------------------------------------------------------------------
# TestPreserveNestedCacheRetainedStateTransform
# ---------------------------------------------------------------------------


class TestPreserveNestedCacheRetainedStateTransform:
    def test_adds_retained_state_outputs_to_call_nodes(self):
        model = _make_minimal_onnx_with_repeated_subgraphs(num_layers=2, scatter_count_per_fn=2)
        changed = PreserveNestedCacheRetainedStateTransform.apply(model)
        assert changed, "Transform should have modified the model (dangling _RetainedState outputs)"

        for fn in model.functions:
            assert len(fn.output) >= 2, (
                f"Function '{fn.name}' should have at least 2 outputs after transform, got {list(fn.output)}"
            )

    def test_noop_when_no_dangling_retained_states(self):
        model = _make_minimal_onnx_with_repeated_subgraphs(num_layers=2, scatter_count_per_fn=2)

        for out in list(model.graph.output):
            if out.name.endswith("_RetainedState"):
                model.graph.output.remove(out)

        changed = PreserveNestedCacheRetainedStateTransform.apply(model)
        assert not changed, "Transform should be a no-op when there are no dangling _RetainedState outputs"

    def test_noop_when_scatter_count_not_two(self):
        model = _make_minimal_onnx_with_repeated_subgraphs(num_layers=1, scatter_count_per_fn=1)
        PreserveNestedCacheRetainedStateTransform.apply(model)
        fn = model.functions[0]
        assert len(fn.output) == 1, f"Function with 1 scatter should not have outputs added, got {list(fn.output)}"


# ---------------------------------------------------------------------------
# TestRenameRepeatedSubgraphTransform
# ---------------------------------------------------------------------------


class TestRenameRepeatedSubgraphTransform:
    def test_renames_repeated_subgraph_functions(self):
        model = _make_minimal_onnx_with_repeated_subgraphs(num_layers=2)
        changed = RenameRepeatedSubgraphTransform.apply(model, target_classnames=["QEffLlamaDecoderLayer"])
        assert changed

        fn_names = [fn.name for fn in model.functions]
        assert "QEffLlamaDecoderLayer" in fn_names, f"Expected 'QEffLlamaDecoderLayer' in {fn_names}"
        assert "QEffLlamaDecoderLayer_1" in fn_names, f"Expected 'QEffLlamaDecoderLayer_1' in {fn_names}"

        node_op_types = [n.op_type for n in model.graph.node]
        assert "QEffLlamaDecoderLayer" in node_op_types
        assert "QEffLlamaDecoderLayer_1" in node_op_types

    def test_noop_on_empty_classnames(self):
        model = _make_minimal_onnx_with_repeated_subgraphs(num_layers=2)
        changed = RenameRepeatedSubgraphTransform.apply(model, target_classnames=[])
        assert not changed

    def test_noop_when_no_repeated_subgraph_functions(self):
        fn = helper.make_function(
            domain="",
            fname="SomeOtherFunction",
            inputs=[],
            outputs=[],
            nodes=[],
            opset_imports=[helper.make_opsetid("", 17)],
        )
        graph = helper.make_graph([], "g", [], [])
        model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
        model.functions.append(fn)

        changed = RenameRepeatedSubgraphTransform.apply(model, target_classnames=["QEffLlamaDecoderLayer"])
        assert not changed

    def test_handles_alternative_subgraph_pattern(self):
        fn0 = helper.make_function(
            domain="",
            fname="subgraph_0",
            inputs=[],
            outputs=[],
            nodes=[],
            opset_imports=[helper.make_opsetid("", 17)],
        )
        fn1 = helper.make_function(
            domain="",
            fname="subgraph_1",
            inputs=[],
            outputs=[],
            nodes=[],
            opset_imports=[helper.make_opsetid("", 17)],
        )
        call0 = helper.make_node("subgraph_0", inputs=[], outputs=[])
        call1 = helper.make_node("subgraph_1", inputs=[], outputs=[])
        graph = helper.make_graph([call0, call1], "g", [], [])
        model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
        model.functions.extend([fn0, fn1])

        changed = RenameRepeatedSubgraphTransform.apply(model, target_classnames=["MyDecoderLayer"])
        assert changed
        fn_names = {fn.name for fn in model.functions}
        assert "MyDecoderLayer" in fn_names


# ---------------------------------------------------------------------------
# TestPruneFakeInitializersTransform
# ---------------------------------------------------------------------------


class TestPruneFakeInitializersTransform:
    def _make_mock_onnx_program(self, initializer_names, used_names, fake_initializers):
        from torch._subclasses.fake_tensor import FakeTensor

        initializers = {}
        for name in initializer_names:
            mock_init = MagicMock()
            if name in fake_initializers:
                fake_tensor = MagicMock(spec=FakeTensor)
                mock_init.const_value.raw = fake_tensor
            else:
                mock_init.const_value.raw = torch.zeros(2)
            initializers[name] = mock_init

        mock_graph = MagicMock()
        mock_graph.initializers = initializers

        mock_node = MagicMock()
        mock_node.inputs = list(used_names)
        mock_graph.__iter__ = lambda self: iter([mock_node])
        mock_graph.outputs = []

        mock_program = MagicMock()
        mock_program.model.graph = mock_graph
        return mock_program

    def test_prunes_fake_tensor_initializers(self):
        program = self._make_mock_onnx_program(
            initializer_names=["weight_a", "weight_b"],
            used_names=set(),
            fake_initializers={"weight_a"},
        )
        changed = PruneFakeInitializersTransform.apply(program)
        assert changed
        assert "weight_a" not in program.model.graph.initializers

    def test_preserves_used_fake_initializers(self):
        program = self._make_mock_onnx_program(
            initializer_names=["weight_a"],
            used_names={"weight_a"},
            fake_initializers={"weight_a"},
        )
        changed = PruneFakeInitializersTransform.apply(program)
        assert not changed
        assert "weight_a" in program.model.graph.initializers

    def test_preserves_non_fake_initializers(self):
        program = self._make_mock_onnx_program(
            initializer_names=["real_weight"],
            used_names=set(),
            fake_initializers=set(),
        )
        changed = PruneFakeInitializersTransform.apply(program)
        assert not changed
        assert "real_weight" in program.model.graph.initializers

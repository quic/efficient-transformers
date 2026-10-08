# -----------------------------------------------------------------------------
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------

"""
Unit tests for the dynamo-specific transforms and context managers introduced
in the enable_dynamo_for_causallm branch.

Covered:
  - qeff_nested_compile_region
  - invoke_subgraph_export_patches
  - temporarily_enable_nested_compile_regions
  - temporarily_disable_nested_compile_regions
  - PreserveNestedCacheRetainedStateTransform
  - RenameRepeatedSubgraphTransform
  - PruneFakeInitializersTransform

CPU-only. No QAIC hardware required.
"""

from __future__ import annotations

import importlib
from unittest.mock import MagicMock

import torch
from onnx import TensorProto, helper
from transformers import LlamaConfig, LlamaForCausalLM

from QEfficient.base.onnx_transforms import (
    PreserveNestedCacheRetainedStateTransform,
    PruneFakeInitializersTransform,
    RenameRepeatedSubgraphTransform,
)
from QEfficient.transformers.models.llama.modeling_llama import QEffLlamaDecoderLayer
from QEfficient.transformers.models.modeling_auto import QEFFAutoModelForCausalLM
from QEfficient.utils import export_utils
from QEfficient.utils.torch_patches import (
    invoke_subgraph_export_patches,
    preserve_subfunction_source_lines,
    qeff_nested_compile_region,
    temporarily_disable_nested_compile_regions,
    temporarily_enable_nested_compile_regions,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class _InterpreterSpy:
    def __init__(self, graph_module):
        self.graph_module = graph_module

    def run(self, *operands):
        return self.graph_module(*operands)


def test_preserve_subfunction_source_lines_interprets_graph_modules(monkeypatch):
    invoke_subgraph = importlib.import_module("torch._higher_order_ops.invoke_subgraph")
    original_reenter_make_fx = invoke_subgraph.reenter_make_fx
    graph_module = torch.fx.symbolic_trace(torch.nn.Identity())
    calls = []

    def fake_reenter_make_fx(fn, *args, **kwargs):
        calls.append((fn, args, kwargs))
        return fn(*args)

    monkeypatch.setattr(invoke_subgraph, "reenter_make_fx", fake_reenter_make_fx)
    monkeypatch.setattr(torch.fx, "Interpreter", _InterpreterSpy)

    with preserve_subfunction_source_lines():
        result = invoke_subgraph.reenter_make_fx(graph_module, torch.ones(2))
        assert torch.equal(result, torch.ones(2))
        assert calls and calls[0][0] is not graph_module

    assert invoke_subgraph.reenter_make_fx is fake_reenter_make_fx
    assert original_reenter_make_fx is not invoke_subgraph.reenter_make_fx


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

        # Scatter nodes inside the function
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

        # Call node in the main graph — outputs start EMPTY so that the
        # _RetainedState names in graph.output are dangling (not produced by any
        # node). PreserveNestedCacheRetainedStateTransform must wire them up.
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


# ---------------------------------------------------------------------------
# TestNestedCompileRegions
# ---------------------------------------------------------------------------


class TestNestedCompileRegions:
    def test_qeff_decoder_layers_are_decorated(self):
        model_hf, _ = make_tiny_llama()
        qeff_model = QEFFAutoModelForCausalLM(model_hf)
        decoder_layers = [m for m in qeff_model.model.modules() if isinstance(m, QEffLlamaDecoderLayer)]

        assert decoder_layers
        assert all(getattr(layer.forward, "__qeff_nested_compile_region__", False) for layer in decoder_layers)

    def test_decorator_factory_forwards_reuse_configuration(self, monkeypatch):
        calls = []

        def reuse_hash_fn(value):
            return 0

        def fake_nested_compile_region(**kwargs):
            calls.append(kwargs)
            return lambda fn: fn

        monkeypatch.setattr(torch.compiler, "nested_compile_region", fake_nested_compile_region)

        @qeff_nested_compile_region(options="options", max_reuse_entries=3, reuse_hash_fn=reuse_hash_fn)
        def identity(value):
            return value

        assert identity(1) == 1
        assert calls == [{"options": "options", "max_reuse_entries": 3, "reuse_hash_fn": reuse_hash_fn}]
        assert identity.__qeff_nested_compile_region__

    def test_decorator_is_noop_when_torch_feature_is_unavailable(self, monkeypatch):
        monkeypatch.delattr(torch.compiler, "nested_compile_region")

        def identity(value):
            return value

        decorated = qeff_nested_compile_region(identity)
        assert decorated is identity
        assert decorated.__qeff_nested_compile_region__

    def test_external_fallback_wraps_without_activating_pytorch_patch(self):
        class ExternalBlock(torch.nn.Module):
            def forward(self, value):
                return value

        invoke_vars = importlib.import_module("torch._dynamo.variables.invoke_subgraph")
        original_call_function = invoke_vars.InvokeSubgraphHigherOrderVariable._call_function
        model = ExternalBlock()

        with temporarily_enable_nested_compile_regions(model, target_classes=[ExternalBlock]):
            assert model.forward.__qeff_nested_compile_region__
            assert invoke_vars.InvokeSubgraphHigherOrderVariable._call_function is original_call_function

        assert "forward" not in model.__dict__

    def test_temporarily_disables_and_restores_qeff_regions(self):
        class Model(torch.nn.Module):
            @qeff_nested_compile_region
            def forward(self, value):
                return value + 1

        model = Model()
        wrapped_forward = model.forward.__func__

        with temporarily_disable_nested_compile_regions(model):
            assert not getattr(model.forward, "__qeff_nested_compile_region__", False)
            assert model(torch.tensor(1)).item() == 2

        assert model.forward.__func__ is wrapped_forward

    def test_reuse_hash_fn_partitions_export_repeated_subgraphs(self):
        class Dense(torch.nn.Module):
            reuse_group_key = 0

            def forward(self, x):
                return x.sin() + 1

        class Moe(torch.nn.Module):
            reuse_group_key = 1

            def forward(self, x):
                return x.cos() + 2

        def reuse_hash_fn(layer, x):
            return layer.reuse_group_key

        @qeff_nested_compile_region(reuse_hash_fn=reuse_hash_fn)
        def layer_fn(layer, x):
            return layer(x)

        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.layers = torch.nn.ModuleList([Dense(), Dense(), Moe(), Dense(), Moe()])

            def forward(self, x):
                for layer in self.layers:
                    x = layer_fn(layer, x)
                return x

        model = Model()
        inputs = (torch.randn(8),)

        with invoke_subgraph_export_patches():
            with torch.no_grad(), torch._dynamo.config.patch(inline_single_use_invoke_subgraph=False):
                exported_program = torch.export.export(model, inputs, strict=False)

        assert torch.equal(exported_program.module()(*inputs), model(*inputs))
        repeated_subgraphs = {
            name for name, _ in exported_program.graph_module.named_modules() if name.startswith("repeated_subgraph")
        }
        assert repeated_subgraphs == {"repeated_subgraph0", "repeated_subgraph1"}

    def test_glm4_moe_reuse_hash_separates_dense_and_moe_layers(self):
        from transformers import AutoConfig
        from transformers.models.glm4_moe.modeling_glm4_moe import Glm4MoeDecoderLayer

        from QEfficient.transformers.models.glm4_moe.modeling_glm4_moe import QEffGlm4MoeDecoderLayer

        config = AutoConfig.for_model(
            "glm4_moe",
            max_position_embeddings=128,
            num_hidden_layers=2,
            num_attention_heads=4,
            hidden_size=64,
            intermediate_size=128,
            moe_intermediate_size=32,
            vocab_size=127,
            num_key_value_heads=2,
            n_routed_experts=4,
            num_experts_per_tok=2,
            first_k_dense_replace=1,
            n_group=1,
            topk_group=1,
            head_dim=16,
        )
        dense_layer = Glm4MoeDecoderLayer(config, layer_idx=0)
        moe_layer = Glm4MoeDecoderLayer(config, layer_idx=1)
        reuse_hash_fn = QEffGlm4MoeDecoderLayer.forward.__qeff_reuse_hash_fn__

        assert reuse_hash_fn(dense_layer) == 0
        assert reuse_hash_fn(moe_layer) == 1

    def test_graph_module_comparison_ignores_storage_bytes(self):
        from torch._subclasses.fake_tensor import FakeTensorMode

        def make_graph(example_value):
            graph = torch.fx.Graph()
            x = graph.placeholder("x")
            x.meta["example_value"] = example_value
            sin = graph.call_function(torch.ops.aten.sin.default, (x,))
            graph.output((sin,))
            return torch.fx.GraphModule({}, graph)

        small_storage = torch.empty(2)
        large_storage = torch.as_strided(torch.empty(100), (2,), (1,), 0)

        with FakeTensorMode() as fake_mode:
            small_fake = fake_mode.from_tensor(small_storage)
            large_fake = fake_mode.from_tensor(large_storage)

        with invoke_subgraph_export_patches():
            from torch._dynamo.variables.higher_order_ops import are_same_graph_modules

            assert are_same_graph_modules(
                "storage_bytes",
                make_graph(small_fake),
                make_graph(large_fake),
                fake_mode,
            )


def test_dynamo_subfunction_export_runs_without_grad(monkeypatch, tmp_path):
    class Model:
        model = torch.nn.Identity()
        model_architecture = "test"
        model_name = "test"
        hash_params = {}
        _onnx_transforms = []

    state = {
        "decoder_layer_classes": [],
        "dynamo": True,
        "onnx_transforms": [],
        "use_onnx_subfunctions": False,
        "hash_use_subfunctions": False,
        "hash_subfunction_version": None,
    }
    invoke_vars = importlib.import_module("torch._dynamo.variables.invoke_subgraph")
    original_call_function = invoke_vars.InvokeSubgraphHigherOrderVariable._call_function
    monkeypatch.setattr(export_utils, "validate_dynamo_export_requirements", lambda _: None)
    monkeypatch.setattr(
        export_utils,
        "_setup_onnx_subfunctions",
        lambda self, args, kwargs, dynamo: (args, kwargs, state),
    )
    monkeypatch.setattr(export_utils, "_prepare_export_directory", lambda self, kwargs: tmp_path / "model")
    monkeypatch.setattr(export_utils, "_generate_export_hash", lambda self, args, kwargs, func: ("hash", {}))
    monkeypatch.setattr(export_utils, "_save_export_metadata", lambda *args: None)
    monkeypatch.setattr(export_utils, "_cleanup_onnx_subfunctions", lambda *args: None)

    @export_utils.export_wrapper
    def export(model, **kwargs):
        assert not torch.is_grad_enabled()
        assert invoke_vars.InvokeSubgraphHigherOrderVariable._call_function is not original_call_function
        return kwargs["export_dir"] / "model.onnx"

    export(Model(), dynamo=True, use_onnx_subfunctions=True)
    assert invoke_vars.InvokeSubgraphHigherOrderVariable._call_function is original_call_function


def test_flat_dynamo_export_temporarily_disables_nested_regions(monkeypatch, tmp_path):
    class Region(torch.nn.Module):
        @qeff_nested_compile_region
        def forward(self, value):
            return value

    class Model:
        model = Region()
        model_architecture = "test"
        model_name = "test"
        hash_params = {}
        _onnx_transforms = []

    monkeypatch.setattr(export_utils, "validate_dynamo_export_requirements", lambda _: None)
    monkeypatch.setattr(export_utils, "_prepare_export_directory", lambda self, kwargs: tmp_path / "model")
    monkeypatch.setattr(export_utils, "_generate_export_hash", lambda self, args, kwargs, func: ("hash", {}))
    monkeypatch.setattr(export_utils, "_save_export_metadata", lambda *args: None)

    @export_utils.export_wrapper
    def export(model, **kwargs):
        assert not getattr(model.model.forward, "__qeff_nested_compile_region__", False)
        return kwargs["export_dir"] / "model.onnx"

    wrapper = Model()
    export(wrapper, dynamo=True, use_onnx_subfunctions=False)
    assert wrapper.model.forward.__qeff_nested_compile_region__


def test_visual_block_mappings_use_qeff_owned_nested_regions():
    from QEfficient.diffusers.models.pytorch_transforms import AttentionTransform
    from QEfficient.transformers.models.pytorch_transforms import KVCacheTransform

    expected_mappings = {
        "CLIPEncoderLayer": "QEffCLIPEncoderLayer",
        "Gemma4VisionEncoderLayer": "QEffGemma4VisionEncoderLayer",
        "InternVLVisionLayer": "QEffInternVLVisionLayer",
        "Llama4VisionEncoderLayer": "QEffLlama4VisionEncoderLayer",
        "MllamaVisionEncoderLayer": "QEffMllamaVisionEncoderLayer",
        "PixtralAttentionLayer": "QEffPixtralAttentionLayer",
        "Qwen2_5_VLVisionBlock": "QEffQwen2_5_VLVisionBlock",
        "Qwen3VLVisionBlock": "QEffQwen3VLVisionBlock",
        "Qwen3VLMoeVisionBlock": "QEffQwen3VLMoeVisionBlock",
        "Qwen3_5VisionBlock": "QEffQwen3_5VisionBlock",
        "Qwen3_5MoeVisionBlock": "QEffQwen3_5MoeVisionBlock",
        "SiglipEncoderLayer": "QEffSiglipEncoderLayer",
        "WhisperEncoderLayer": "QEffWhisperEncoderLayer",
    }
    mappings = {source.__name__: (source, target) for source, target in KVCacheTransform._module_mapping.items()}

    for source_name, target_name in expected_mappings.items():
        source, target = mappings[source_name]
        assert target.__name__ == target_name
        assert issubclass(target, source)
        assert target.forward.__qeff_nested_compile_region__

        module = torch.nn.Module()
        module.__class__ = source
        module, transformed = KVCacheTransform.apply(module)
        assert transformed
        assert type(module) is target

    wan_source, wan_target = next(
        (source, target)
        for source, target in AttentionTransform._module_mapping.items()
        if source.__name__ == "WanTransformerBlock"
    )
    assert wan_target.__name__ == "QEffWanTransformerBlock"
    assert issubclass(wan_target, wan_source)
    assert wan_target.forward.__qeff_nested_compile_region__

    wan_block = torch.nn.Module()
    wan_block.__class__ = wan_source
    wan_block, transformed = AttentionTransform.apply(wan_block)
    assert transformed
    assert type(wan_block) is wan_target


def test_visual_block_mapping_preserves_forward_behavior():
    from transformers.models.clip.configuration_clip import CLIPVisionConfig
    from transformers.models.clip.modeling_clip import CLIPEncoderLayer

    from QEfficient.transformers.models.pytorch_transforms import KVCacheTransform

    config = CLIPVisionConfig(hidden_size=16, intermediate_size=32, num_attention_heads=4, num_hidden_layers=1)
    layer = CLIPEncoderLayer(config).eval()
    hidden_states = torch.randn(2, 5, config.hidden_size)
    expected = layer(hidden_states, attention_mask=None)

    layer, transformed = KVCacheTransform.apply(layer)
    actual = layer(hidden_states, attention_mask=None)

    assert transformed
    torch.testing.assert_close(actual, expected)


# ---------------------------------------------------------------------------
# TestPreserveNestedCacheRetainedStateTransform
# ---------------------------------------------------------------------------


class TestPreserveNestedCacheRetainedStateTransform:
    def test_adds_retained_state_outputs_to_call_nodes(self):
        model = _make_minimal_onnx_with_repeated_subgraphs(num_layers=2, scatter_count_per_fn=2)

        # Initially the call nodes have outputs but the functions don't expose scatter outputs
        changed = PreserveNestedCacheRetainedStateTransform.apply(model)
        assert changed, "Transform should have modified the model (dangling _RetainedState outputs)"

        # After transform: function outputs should include scatter node outputs
        for i, fn in enumerate(model.functions):
            assert len(fn.output) >= 2, (
                f"Function '{fn.name}' should have at least 2 outputs after transform, got {list(fn.output)}"
            )

    def test_noop_when_no_dangling_retained_states(self):
        model = _make_minimal_onnx_with_repeated_subgraphs(num_layers=2, scatter_count_per_fn=2)

        # Remove all _RetainedState outputs from the graph — nothing is dangling
        for out in list(model.graph.output):
            if out.name.endswith("_RetainedState"):
                model.graph.output.remove(out)

        changed = PreserveNestedCacheRetainedStateTransform.apply(model)
        assert not changed, "Transform should be a no-op when there are no dangling _RetainedState outputs"

    def test_noop_when_scatter_count_not_two(self):
        # Build model where function has only 1 scatter node
        model = _make_minimal_onnx_with_repeated_subgraphs(num_layers=1, scatter_count_per_fn=1)
        PreserveNestedCacheRetainedStateTransform.apply(model)
        # The key invariant: no crash; the function with only 1 scatter is skipped
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

        # Call-site op_type must also be updated
        node_op_types = [n.op_type for n in model.graph.node]
        assert "QEffLlamaDecoderLayer" in node_op_types
        assert "QEffLlamaDecoderLayer_1" in node_op_types

    def test_noop_on_empty_classnames(self):
        model = _make_minimal_onnx_with_repeated_subgraphs(num_layers=2)
        changed = RenameRepeatedSubgraphTransform.apply(model, target_classnames=[])
        assert not changed

    def test_noop_when_no_repeated_subgraph_functions(self):
        # Build a model with a non-repeated_subgraph function name
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
        # torch < 2.5 naming: subgraph_0, subgraph_1
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
        """Build a mock onnx_program object matching PruneFakeInitializersTransform's API."""
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

        # Simulate used_names via graph nodes + outputs
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
            used_names=set(),  # neither is used
            fake_initializers={"weight_a"},
        )
        changed = PruneFakeInitializersTransform.apply(program)
        assert changed
        assert "weight_a" not in program.model.graph.initializers

    def test_preserves_used_fake_initializers(self):
        program = self._make_mock_onnx_program(
            initializer_names=["weight_a"],
            used_names={"weight_a"},  # it is used
            fake_initializers={"weight_a"},
        )
        changed = PruneFakeInitializersTransform.apply(program)
        assert not changed
        assert "weight_a" in program.model.graph.initializers

    def test_preserves_non_fake_initializers(self):
        program = self._make_mock_onnx_program(
            initializer_names=["real_weight"],
            used_names=set(),
            fake_initializers=set(),  # not a FakeTensor
        )
        changed = PruneFakeInitializersTransform.apply(program)
        assert not changed
        assert "real_weight" in program.model.graph.initializers

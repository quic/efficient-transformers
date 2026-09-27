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

import onnx
import onnx_ir as ir
import pytest
import torch
from onnx import TensorProto, helper
from safetensors import safe_open
from safetensors.torch import save_file
from transformers import LlamaConfig, LlamaForCausalLM

from QEfficient.base.checkpoint_transforms import CHECKPOINT_PREPARED_MANIFEST, CheckpointTransformPipeline
from QEfficient.base.onnx_transforms import (
    CustomOpTransform,
    PreserveNestedCacheRetainedStateTransform,
    PruneFakeInitializersTransform,
    RenameRepeatedSubgraphTransform,
)
from QEfficient.customop.quantization_ops import UnpackMxfp6Func
from QEfficient.exporter.weight_free import checkpoint_key_resolver
from QEfficient.exporter.weight_free.checkpoint_key_resolver import find_checkpoint_key
from QEfficient.exporter.weight_free.checkpoint_transforms import (
    DtypeConversionCheckpointTransform,
    GraniteMoeFusedExpertSplitCheckpointTransform,
    MoEExpertStackingCheckpointTransform,
    MoEFusedExpertSplitCheckpointTransform,
)
from QEfficient.exporter.weight_free.mxfp6 import (
    MXFP6_BLOCK_SIZE,
    MXFP6_PACKED_SUFFIX,
    MXFP6_SCALE_SUFFIX,
    Mxfp6Config,
    _is_lm_head_weight_name,
    finalize_mxfp6_export,
    normalize_mxfp6_config,
    pack_fp6_codes,
    quantize_to_mxfp6,
    unpack_fp6_codes,
)
from QEfficient.exporter.weight_free.ort_weight_injection import load_weight_free_ort_inputs
from QEfficient.exporter.weight_free.weight_spec import (
    ExternalDataFile,
    WeightSpec,
    WeightSpecInput,
    WeightSpecLocation,
    load_weight_spec,
    save_weight_spec,
)
from QEfficient.transformers.models.llama.modeling_llama import QEffLlamaDecoderLayer
from QEfficient.transformers.models.modeling_auto import QEFFAutoModelForCausalLM
from QEfficient.utils import runtime_requirements
from QEfficient.utils.export_utils import _generate_export_hash
from QEfficient.utils.runtime_requirements import validate_runtime_requirements
from QEfficient.utils.torch_patches import temporarily_enable_nested_compile_regions

ONNX_FLOAT6E2M3 = int(TensorProto.FLOAT6E2M3)

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

    def test_resolver_rejects_ambiguous_moe_weight_aliases(self):
        checkpoint_index = {
            "model.layers.0.mlp.experts.moe_weights.gate": "model.safetensors",
            "model.layers.0.mlp.moe_weights.gate": "model.safetensors",
        }
        backbone = MagicMock()
        backbone.base_model_prefix = "model"

        with pytest.raises(ValueError, match="Ambiguous checkpoint key"):
            find_checkpoint_key("model.layers.0.mlp.experts.moe_weights.gate", checkpoint_index, backbone)

    def test_resolver_rejects_ambiguous_prefix_fallbacks(self):
        checkpoint_index = {
            "base_model.model.embed_tokens.weight": "model.safetensors",
            "model.embed_tokens.weight": "model.safetensors",
        }
        backbone = MagicMock()
        backbone.base_model_prefix = "model"

        with pytest.raises(ValueError, match="Ambiguous checkpoint key"):
            find_checkpoint_key("base_model.model.embed_tokens.weight", checkpoint_index, backbone)

    @pytest.mark.parametrize(
        "state_kind,state_name",
        [
            ("parameter", "weight"),
            ("buffer", "running_scale"),
        ],
    )
    def test_promote_initializers_rejects_unresolved_model_state(self, tmp_path, monkeypatch, state_kind, state_name):
        src = tmp_path / "src"
        src.mkdir()
        _write_safetensors_checkpoint(src, {"other.weight": torch.ones(2, dtype=torch.float32)})

        class MissingStateModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                if state_kind == "parameter":
                    self.register_parameter(state_name, torch.nn.Parameter(torch.ones(2)))
                else:
                    self.register_buffer(state_name, torch.ones(2))

        initializer = SimpleNamespace(shape=(2,), dtype=ir.DataType.FLOAT)
        graph = SimpleNamespace(initializers={state_name: initializer}, inputs=[])
        onnx_program = SimpleNamespace(model=SimpleNamespace(graph=graph))
        monkeypatch.setattr(
            checkpoint_key_resolver.ir,
            "Value",
            lambda name, shape, type: SimpleNamespace(name=name, shape=shape, type=type),
        )

        with pytest.raises(ValueError, match=f"Could not resolve model initializer '{state_name}'"):
            checkpoint_key_resolver.promote_initializers_and_build_spec(
                onnx_program=onnx_program,
                model_ref=str(src),
                model_name="tiny-missing-state",
                qeff_model=SimpleNamespace(model=MissingStateModel()),
            )

    @pytest.mark.parametrize(
        "state_kind,state_name",
        [
            ("buffer", "rotary_emb.inv_freq"),
            ("buffer", "transformer.h.0.attn.embed_positions"),
            ("buffer", "model.embed_tokens.embed_scale"),
            ("parameter", "model.sin_cached"),
            ("parameter", "model.cos_cached"),
        ],
    )
    def test_promote_initializers_keeps_computed_state_embedded(self, tmp_path, monkeypatch, state_kind, state_name):
        src = tmp_path / "src"
        src.mkdir()
        _write_safetensors_checkpoint(src, {"other.weight": torch.ones(2, dtype=torch.float32)})

        class ComputedStateModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                parent = self
                parts = state_name.split(".")
                for part in parts[:-1]:
                    child = torch.nn.Module()
                    setattr(parent, part, child)
                    parent = child
                if state_kind == "parameter":
                    parent.register_parameter(parts[-1], torch.nn.Parameter(torch.ones(2)))
                else:
                    parent.register_buffer(parts[-1], torch.ones(2))

        initializer = SimpleNamespace(shape=(2,), dtype=ir.DataType.FLOAT)
        graph = SimpleNamespace(initializers={state_name: initializer}, inputs=[])
        onnx_program = SimpleNamespace(model=SimpleNamespace(graph=graph))
        monkeypatch.setattr(
            checkpoint_key_resolver.ir,
            "Value",
            lambda name, shape, type: SimpleNamespace(name=name, shape=shape, type=type),
        )

        spec = checkpoint_key_resolver.promote_initializers_and_build_spec(
            onnx_program=onnx_program,
            model_ref=str(src),
            model_name="tiny-computed-state",
            qeff_model=SimpleNamespace(model=ComputedStateModel()),
        )

        assert state_name in graph.initializers
        assert graph.inputs == []
        assert spec.inputs == []

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


@pytest.mark.weight_free_unit
class TestWeightFreeMxfp6:
    @pytest.mark.parametrize(
        "alias,normalized",
        [
            ("float16", "float16"),
            ("fp16", "float16"),
            ("half", "float16"),
            ("float32", "float32"),
            ("fp32", "float32"),
            ("float", "float32"),
            ("bfloat16", "bfloat16"),
            ("bf16", "bfloat16"),
            ("fp8", "e8m0"),
            ("e8m0", "e8m0"),
            ("float8e8m0", "e8m0"),
            ("float8_e8m0fnu", "e8m0"),
        ],
    )
    def test_mxfp6_scale_dtype_aliases(self, alias, normalized):
        assert normalize_mxfp6_config(True, alias) == Mxfp6Config(enabled=True, scale_dtype=normalized)

    def test_mxfp6_scale_dtype_defaults_to_float16(self):
        assert normalize_mxfp6_config(True) == Mxfp6Config(enabled=True, scale_dtype="float16")

    def test_mxfp6_rejects_unknown_scale_dtype(self):
        with pytest.raises(ValueError, match="mxfp6_scale_dtype"):
            normalize_mxfp6_config(True, "int8")

    def test_mxfp6_requires_weight_free(self):
        with pytest.raises(ValueError, match="requires `weight_free=True`"):
            QEFFAutoModelForCausalLM.from_pretrained("dummy-model", mxfp6=True)

    def test_mxfp6_rejects_layerwise(self):
        with pytest.raises(ValueError, match="layerwise=True"):
            QEFFAutoModelForCausalLM.from_pretrained("dummy-model", layerwise=True, weight_free=True, mxfp6=True)

    def test_mxfp6_rejects_source_quantization_config(self, monkeypatch):
        monkeypatch.setattr(
            "QEfficient.transformers.models.modeling_auto.validate_mxfp6_capabilities", lambda config: None
        )
        with pytest.raises(ValueError, match="source HF quantization configs"):
            QEFFAutoModelForCausalLM.from_pretrained(
                "dummy-model",
                weight_free=True,
                mxfp6=True,
                quantization_config={"quant_method": "gptq"},
            )

    def test_mxfp6_config_stored_on_wrapper(self, monkeypatch):
        model_hf, _ = make_tiny_llama()
        monkeypatch.setattr(
            "QEfficient.transformers.models.modeling_auto.validate_mxfp6_capabilities", lambda config: None
        )
        monkeypatch.setattr(
            "QEfficient.transformers.models.modeling_auto.validate_dynamo_export_requirements", lambda name: None
        )
        monkeypatch.setattr(
            "QEfficient.transformers.models.modeling_auto._build_meta_model", lambda *args, **kwargs: model_hf
        )

        qeff_model = QEFFAutoModelForCausalLM.from_pretrained(
            "dummy-model",
            weight_free=True,
            mxfp6=True,
            mxfp6_scale_dtype="half",
        )

        assert qeff_model._mxfp6_config == Mxfp6Config(enabled=True, scale_dtype="float16")

    def test_mxfp6_rejects_misclassified_vlm_redirect(self, monkeypatch):
        class InternVLChatModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.config = SimpleNamespace()

        model_hf = InternVLChatModel()
        monkeypatch.setattr(
            "QEfficient.transformers.models.modeling_auto.validate_mxfp6_capabilities", lambda config: None
        )
        monkeypatch.setattr(
            "QEfficient.transformers.models.modeling_auto.validate_dynamo_export_requirements", lambda name: None
        )
        monkeypatch.setattr(
            "QEfficient.transformers.models.modeling_auto._build_meta_model", lambda *args, **kwargs: model_hf
        )

        with pytest.raises(ValueError, match="text CausalLM weight-free export"):
            QEFFAutoModelForCausalLM.from_pretrained("dummy-model", weight_free=True, mxfp6=True)

    def test_mxfp6_rejects_compiler_owned_mxfp6_matmul(self, tmp_path):
        model_hf, _ = make_tiny_llama()
        qeff_model = QEFFAutoModelForCausalLM(
            model_hf,
            weight_free=True,
            mxfp6_config=Mxfp6Config(enabled=True, scale_dtype="float16"),
        )
        onnx_path = tmp_path / "model.onnx"
        onnx_path.write_bytes(b"placeholder")

        for option_name in ("mxfp6_matmul", "mxfp6-matmul", "mxfp6"):
            with pytest.raises(ValueError, match="mxfp6"):
                qeff_model._compile(onnx_path=str(onnx_path), compile_dir=str(tmp_path), **{option_name: True})

        with pytest.raises(ValueError, match="QNN compilation"):
            qeff_model._compile(onnx_path=str(onnx_path), compile_dir=str(tmp_path), enable_qnn=True)

    def test_fp6_pack_unpack_roundtrip(self):
        codes = torch.arange(64, dtype=torch.uint8)
        packed = pack_fp6_codes(codes)
        assert packed.numel() == 48
        torch.testing.assert_close(unpack_fp6_codes(packed), codes)

    def test_fp6_pack_uses_onnx_lsb_first_layout(self):
        codes = torch.tensor([0x00, 0x01, 0x02, 0x03], dtype=torch.uint8)

        torch.testing.assert_close(
            pack_fp6_codes(codes),
            torch.tensor([0x40, 0x20, 0x0C], dtype=torch.uint8),
        )

    def test_mxfp6_quantization_uses_block_scales_and_rejects_nonfinite(self):
        weight = torch.zeros(2, MXFP6_BLOCK_SIZE * 2, dtype=torch.float32)
        weight[0, :MXFP6_BLOCK_SIZE] = 15.0
        packed, scales = quantize_to_mxfp6(weight, "float16")

        assert packed.dtype == torch.uint8
        assert packed.shape == (2, MXFP6_BLOCK_SIZE * 2 * 3 // 4)
        assert packed.numel() == weight.numel() * 6 // 8
        assert UnpackMxfp6Func.apply(packed).shape == weight.shape
        assert scales.dtype == torch.float16
        torch.testing.assert_close(scales[0, 0], torch.tensor(2.0, dtype=torch.float16))
        torch.testing.assert_close(scales[0, 1], torch.tensor(1.0, dtype=torch.float16))

        weight[0, 0] = float("nan")
        with pytest.raises(ValueError, match="NaN or Inf"):
            quantize_to_mxfp6(weight, "float16")

    @pytest.mark.parametrize(
        "name",
        [
            "lm_head.weight",
            "model.lm_head.weight",
            "base_model.lm_head.weight",
            "base_model/lm_head/weight",
        ],
    )
    def test_lm_head_weight_name_predicate_matches_path_component_suffix(self, name):
        assert _is_lm_head_weight_name(name)

    @pytest.mark.parametrize(
        "name",
        [
            "prefix_lm_head.weight",
            "lm_head_projection.weight",
            "model.lm_head.weight.extra",
            "model.layers.0.lm_head_adapter.weight",
        ],
    )
    def test_lm_head_weight_name_predicate_rejects_partial_components(self, name):
        assert not _is_lm_head_weight_name(name)

    def test_mxfp6_finalizer_writes_v7_location_metadata(self, tmp_path, monkeypatch):
        monkeypatch.setattr("QEfficient.exporter.weight_free.mxfp6.validate_mxfp6_capabilities", lambda config: None)
        monkeypatch.setattr("QEfficient.exporter.weight_free.mxfp6._dql_schema_supports_output_dtype", lambda: True)
        monkeypatch.setattr("QEfficient.exporter.weight_free.mxfp6.TensorProto.FLOAT6E2M3", 999, raising=False)

        prepared = tmp_path / "prepared"
        prepared.mkdir()
        weight = torch.arange(4 * MXFP6_BLOCK_SIZE, dtype=torch.float32).reshape(4, MXFP6_BLOCK_SIZE)
        _write_safetensors_checkpoint(prepared, {"linear.weight": weight})

        x = helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 4])
        w = helper.make_tensor_value_info("linear.weight", TensorProto.FLOAT, [4, MXFP6_BLOCK_SIZE])
        y = helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, MXFP6_BLOCK_SIZE])
        graph = helper.make_graph(
            [helper.make_node("MatMul", ["x", "linear.weight"], ["y"])],
            "mxfp6_test",
            [x, w],
            [y],
        )
        model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 18)])
        onnx_path = tmp_path / "model.onnx"
        onnx.save(model, str(onnx_path))
        spec_path = save_weight_spec(
            tmp_path / "weight_spec.json",
            WeightSpec(
                model_name="tiny",
                model_id=str(prepared),
                files=[ExternalDataFile(path="model.safetensors", format="safetensors")],
                inputs=[
                    WeightSpecInput(
                        name="linear.weight",
                        location=WeightSpecLocation(file=0, key="linear.weight"),
                    )
                ],
            ),
        )

        finalize_mxfp6_export(onnx_path, spec_path, str(prepared), Mxfp6Config(enabled=True, scale_dtype="float16"))

        spec = load_weight_spec(spec_path)
        raw_spec = json.loads(spec_path.read_text())
        assert spec.version == 7
        assert [entry.role for entry in spec.inputs] == ["mxfp6_weight", "mxfp6_scale"]
        assert spec.inputs[0].name == "linear.weight" + MXFP6_PACKED_SUFFIX
        assert spec.inputs[0].location.key == "linear.weight" + MXFP6_PACKED_SUFFIX
        assert spec.inputs[0].metadata["logical_dtype"] == "float6e2m3"
        assert spec.inputs[0].metadata["storage_dtype"] == "uint8"
        assert spec.inputs[0].metadata["packing"] == "onnx_lsb_first_6bit"
        assert spec.inputs[0].metadata["scale_input"] == "linear.weight" + MXFP6_SCALE_SUFFIX
        assert spec.inputs[0].metadata["unpack_output"] == "linear.weight.mxfp6_unpacked"
        assert spec.inputs[0].metadata["packed_shape"] == [4, MXFP6_BLOCK_SIZE * 3 // 4]
        assert spec.inputs[1].location.key == "linear.weight" + MXFP6_SCALE_SUFFIX
        assert raw_spec["files"] == [{"format": "safetensors", "path": "model.safetensors"}]
        assert {entry["location"]["key"] for entry in raw_spec["inputs"]} == {
            "linear.weight" + MXFP6_PACKED_SUFFIX,
            "linear.weight" + MXFP6_SCALE_SUFFIX,
        }
        assert all(set(entry["location"]) == {"file", "key"} for entry in raw_spec["inputs"])

        def json_keys(value):
            if isinstance(value, dict):
                yield from value
                for nested_value in value.values():
                    yield from json_keys(nested_value)
            elif isinstance(value, list):
                for nested_value in value:
                    yield from json_keys(nested_value)

        assert not {"base64", "packed_bytes", "payload", "scale_values", "values"} & set(json_keys(raw_spec))

        tensors = _load_prepared_tensors(prepared)
        assert tensors["linear.weight"].dtype == torch.float32
        assert tensors["linear.weight" + MXFP6_PACKED_SUFFIX].dtype == torch.uint8
        assert tensors["linear.weight" + MXFP6_SCALE_SUFFIX].dtype == torch.float16
        rewritten = onnx.load(str(onnx_path), load_external_data=False)
        assert rewritten.opset_import[0].version == 28
        graph_inputs_by_name = {value_info.name: value_info for value_info in rewritten.graph.input}
        graph_input_names = set(graph_inputs_by_name)
        initializer_by_name = {initializer.name: initializer for initializer in rewritten.graph.initializer}
        assert "linear.weight" not in graph_input_names
        assert "linear.weight" + MXFP6_PACKED_SUFFIX in graph_input_names
        assert "linear.weight" + MXFP6_SCALE_SUFFIX in graph_input_names
        assert "linear.weight" + MXFP6_PACKED_SUFFIX not in initializer_by_name
        assert "linear.weight" + MXFP6_SCALE_SUFFIX not in initializer_by_name

        def input_shape(value_info):
            return [dim.dim_value for dim in value_info.type.tensor_type.shape.dim]

        packed_input = graph_inputs_by_name["linear.weight" + MXFP6_PACKED_SUFFIX]
        scale_input = graph_inputs_by_name["linear.weight" + MXFP6_SCALE_SUFFIX]
        assert packed_input.type.tensor_type.elem_type == TensorProto.UINT8
        assert input_shape(packed_input) == [4, MXFP6_BLOCK_SIZE * 3 // 4]
        assert scale_input.type.tensor_type.elem_type == TensorProto.FLOAT16
        assert input_shape(scale_input) == [4, 1]

        value_info_by_name = {value_info.name: value_info for value_info in rewritten.graph.value_info}
        unpack_value_info = value_info_by_name["linear.weight.mxfp6_unpacked"]
        assert unpack_value_info.type.tensor_type.elem_type == TensorProto.FLOAT6E2M3
        assert input_shape(unpack_value_info) == [4, MXFP6_BLOCK_SIZE]
        assert CustomOpTransform.apply(rewritten, onnx_export_opset=28)
        unpack_functions = [
            fn for fn in rewritten.functions if fn.domain == "com.qti.aisw.onnx" and fn.name == "UnpackMxfp6"
        ]
        assert len(unpack_functions) == 1
        unpack_cast_nodes = [node for node in unpack_functions[0].node if node.op_type == "Cast"]
        assert any(
            any(attr.name == "to" and helper.get_attribute_value(attr) == ONNX_FLOAT6E2M3 for attr in node.attribute)
            for node in unpack_cast_nodes
        )
        unpack_nodes = [node for node in rewritten.graph.node if node.op_type == "UnpackMxfp6"]
        dq_nodes = [node for node in rewritten.graph.node if node.op_type == "DequantizeLinear"]
        assert len(unpack_nodes) == 1
        assert unpack_nodes[0].domain == "com.qti.aisw.onnx"
        assert unpack_nodes[0].input == ["linear.weight" + MXFP6_PACKED_SUFFIX]
        assert unpack_nodes[0].output == ["linear.weight.mxfp6_unpacked"]
        assert len(dq_nodes) == 1
        assert dq_nodes[0].domain == ""
        assert dq_nodes[0].input == ["linear.weight.mxfp6_unpacked", "linear.weight" + MXFP6_SCALE_SUFFIX]
        assert dq_nodes[0].input[0] != "linear.weight" + MXFP6_PACKED_SUFFIX
        assert [node.op_type for node in rewritten.graph.node[:2]] == ["UnpackMxfp6", "DequantizeLinear"]
        attrs = {attr.name: helper.get_attribute_value(attr) for attr in dq_nodes[0].attribute}
        assert attrs["axis"] == -1
        assert attrs["block_size"] == MXFP6_BLOCK_SIZE
        assert attrs["output_dtype"] == TensorProto.FLOAT

    def test_mxfp6_finalizer_keeps_lm_head_weight_dense(self, tmp_path, monkeypatch):
        monkeypatch.setattr("QEfficient.exporter.weight_free.mxfp6.validate_mxfp6_capabilities", lambda config: None)
        monkeypatch.setattr("QEfficient.exporter.weight_free.mxfp6.TensorProto.FLOAT6E2M3", 999, raising=False)

        prepared = tmp_path / "prepared"
        prepared.mkdir()
        hidden_size = 4
        out_features = MXFP6_BLOCK_SIZE
        linear_weight = torch.arange(hidden_size * out_features, dtype=torch.float32).reshape(hidden_size, out_features)
        lm_head_weight = linear_weight + 1000
        _write_safetensors_checkpoint(
            prepared,
            {
                "linear.weight": linear_weight,
                "base_model.lm_head.weight": lm_head_weight,
            },
        )

        x = helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, hidden_size])
        h = helper.make_tensor_value_info("h", TensorProto.FLOAT, [1, hidden_size])
        linear = helper.make_tensor_value_info("linear.weight", TensorProto.FLOAT, [hidden_size, out_features])
        lm_head = helper.make_tensor_value_info("model.lm_head.weight", TensorProto.FLOAT, [hidden_size, out_features])
        y = helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, out_features])
        logits = helper.make_tensor_value_info("logits", TensorProto.FLOAT, [1, out_features])
        graph = helper.make_graph(
            [
                helper.make_node("MatMul", ["x", "linear.weight"], ["y"]),
                helper.make_node("MatMul", ["h", "model.lm_head.weight"], ["logits"]),
            ],
            "mxfp6_lm_head_test",
            [x, h, linear, lm_head],
            [y, logits],
        )
        model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 18)])
        onnx_path = tmp_path / "model.onnx"
        onnx.save(model, str(onnx_path))
        spec_path = save_weight_spec(
            tmp_path / "weight_spec.json",
            WeightSpec(
                model_name="tiny",
                model_id=str(prepared),
                files=[ExternalDataFile(path="model.safetensors", format="safetensors")],
                inputs=[
                    WeightSpecInput(
                        name="linear.weight",
                        location=WeightSpecLocation(file=0, key="linear.weight"),
                    ),
                    WeightSpecInput(
                        name="model.lm_head.weight",
                        location=WeightSpecLocation(file=0, key="base_model.lm_head.weight"),
                    ),
                ],
            ),
        )

        finalize_mxfp6_export(onnx_path, spec_path, str(prepared), Mxfp6Config(enabled=True, scale_dtype="float16"))

        spec = load_weight_spec(spec_path)
        rewritten = onnx.load(str(onnx_path), load_external_data=False)
        graph_input_names = {value_info.name for value_info in rewritten.graph.input}
        spec_by_name = {entry.name: entry for entry in spec.inputs}
        tensors = _load_prepared_tensors(prepared)

        assert "linear.weight" not in graph_input_names
        assert "linear.weight" + MXFP6_PACKED_SUFFIX in graph_input_names
        assert "linear.weight" + MXFP6_SCALE_SUFFIX in graph_input_names
        assert spec_by_name["linear.weight" + MXFP6_PACKED_SUFFIX].role == "mxfp6_weight"
        assert spec_by_name["linear.weight" + MXFP6_SCALE_SUFFIX].role == "mxfp6_scale"

        assert "model.lm_head.weight" in graph_input_names
        assert spec_by_name["model.lm_head.weight"].role == "weight"
        assert spec_by_name["model.lm_head.weight"].location.key == "base_model.lm_head.weight"
        assert "model.lm_head.weight" + MXFP6_PACKED_SUFFIX not in graph_input_names
        assert "model.lm_head.weight" + MXFP6_SCALE_SUFFIX not in graph_input_names
        assert "model.lm_head.weight" + MXFP6_PACKED_SUFFIX not in spec_by_name
        assert "model.lm_head.weight" + MXFP6_SCALE_SUFFIX not in spec_by_name
        assert "base_model.lm_head.weight" + MXFP6_PACKED_SUFFIX not in tensors
        assert "base_model.lm_head.weight" + MXFP6_SCALE_SUFFIX not in tensors

        unpack_nodes = [node for node in rewritten.graph.node if node.op_type == "UnpackMxfp6"]
        dq_nodes = [node for node in rewritten.graph.node if node.op_type == "DequantizeLinear"]
        assert len(unpack_nodes) == 1
        assert len(dq_nodes) == 1
        assert "lm_head" not in unpack_nodes[0].name
        assert "lm_head" not in dq_nodes[0].name

    def test_mxfp6_finalizer_supports_final_axis_transpose_topology(self, tmp_path, monkeypatch):
        monkeypatch.setattr("QEfficient.exporter.weight_free.mxfp6.validate_mxfp6_capabilities", lambda config: None)
        monkeypatch.setattr("QEfficient.exporter.weight_free.mxfp6.TensorProto.FLOAT6E2M3", 999, raising=False)

        prepared = tmp_path / "prepared"
        prepared.mkdir()
        weight = torch.arange(4 * MXFP6_BLOCK_SIZE, dtype=torch.float32).reshape(4, MXFP6_BLOCK_SIZE)
        _write_safetensors_checkpoint(prepared, {"linear.weight": weight})

        x = helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, MXFP6_BLOCK_SIZE])
        w = helper.make_tensor_value_info("linear.weight", TensorProto.FLOAT, [4, MXFP6_BLOCK_SIZE])
        y = helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, 4])
        graph = helper.make_graph(
            [
                helper.make_node("Transpose", ["linear.weight"], ["linear.weight.t"], perm=[1, 0]),
                helper.make_node("MatMul", ["x", "linear.weight.t"], ["y"]),
            ],
            "mxfp6_transpose_test",
            [x, w],
            [y],
        )
        model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 18)])
        onnx_path = tmp_path / "model.onnx"
        onnx.save(model, str(onnx_path))
        spec_path = save_weight_spec(
            tmp_path / "weight_spec.json",
            WeightSpec(
                model_name="tiny",
                model_id=str(prepared),
                files=[ExternalDataFile(path="model.safetensors", format="safetensors")],
                inputs=[
                    WeightSpecInput(
                        name="linear.weight",
                        location=WeightSpecLocation(file=0, key="linear.weight"),
                    )
                ],
            ),
        )

        finalize_mxfp6_export(onnx_path, spec_path, str(prepared), Mxfp6Config(enabled=True, scale_dtype="float16"))

        rewritten = onnx.load(str(onnx_path), load_external_data=False)
        unpack_nodes = [node for node in rewritten.graph.node if node.op_type == "UnpackMxfp6"]
        dq_nodes = [node for node in rewritten.graph.node if node.op_type == "DequantizeLinear"]
        transpose_nodes = [node for node in rewritten.graph.node if node.op_type == "Transpose"]
        matmul_nodes = [node for node in rewritten.graph.node if node.op_type == "MatMul"]

        assert len(unpack_nodes) == 1
        assert unpack_nodes[0].output == ["linear.weight.mxfp6_unpacked"]
        assert len(dq_nodes) == 1
        assert dq_nodes[0].input[0] == "linear.weight.mxfp6_unpacked"
        assert dq_nodes[0].input[1] == "linear.weight" + MXFP6_SCALE_SUFFIX
        assert dq_nodes[0].output == ["linear.weight"]
        assert transpose_nodes[0].input == ["linear.weight"]
        assert matmul_nodes[0].input[1] == "linear.weight.t"

    def test_mxfp6_finalizer_supports_function_transpose_matmul_topology(self, tmp_path, monkeypatch):
        monkeypatch.setattr("QEfficient.exporter.weight_free.mxfp6.validate_mxfp6_capabilities", lambda config: None)
        monkeypatch.setattr("QEfficient.exporter.weight_free.mxfp6.TensorProto.FLOAT6E2M3", 999, raising=False)

        prepared = tmp_path / "prepared"
        prepared.mkdir()
        weight = torch.arange(4 * MXFP6_BLOCK_SIZE, dtype=torch.float32).reshape(4, MXFP6_BLOCK_SIZE)
        _write_safetensors_checkpoint(prepared, {"model.layers.0.linear.weight": weight})

        function_domain = "pkg.torch.__subgraph__"
        function = helper.make_function(
            domain=function_domain,
            fname="repeated_subgraph0",
            inputs=["hidden_states", "arg2_1"],
            outputs=["fn_y"],
            nodes=[
                helper.make_node("Transpose", ["arg2_1"], ["arg2_1_t"], perm=[1, 0]),
                helper.make_node("MatMul", ["hidden_states", "arg2_1_t"], ["fn_y"]),
            ],
            opset_imports=[helper.make_opsetid("", 18)],
        )

        x = helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, MXFP6_BLOCK_SIZE])
        w = helper.make_tensor_value_info("layer.weight", TensorProto.FLOAT, [4, MXFP6_BLOCK_SIZE])
        y = helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, 4])
        call_node = helper.make_node(
            "repeated_subgraph0",
            ["x", "layer.weight"],
            ["y"],
            domain=function_domain,
        )
        graph = helper.make_graph([call_node], "mxfp6_function_test", [x, w], [y])
        model = helper.make_model(
            graph,
            opset_imports=[helper.make_opsetid("", 18), helper.make_opsetid(function_domain, 1)],
        )
        model.functions.append(function)
        onnx_path = tmp_path / "model.onnx"
        onnx.save(model, str(onnx_path))
        spec_path = save_weight_spec(
            tmp_path / "weight_spec.json",
            WeightSpec(
                model_name="tiny",
                model_id=str(prepared),
                files=[ExternalDataFile(path="model.safetensors", format="safetensors")],
                inputs=[
                    WeightSpecInput(
                        name="layer.weight",
                        location=WeightSpecLocation(file=0, key="model.layers.0.linear.weight"),
                    )
                ],
            ),
        )

        finalize_mxfp6_export(onnx_path, spec_path, str(prepared), Mxfp6Config(enabled=True, scale_dtype="float16"))

        spec = load_weight_spec(spec_path)
        rewritten = onnx.load(str(onnx_path), load_external_data=False)
        graph_input_names = {value_info.name for value_info in rewritten.graph.input}
        initializer_names = {initializer.name for initializer in rewritten.graph.initializer}
        unpack_nodes = [node for node in rewritten.graph.node if node.op_type == "UnpackMxfp6"]
        dq_nodes = [node for node in rewritten.graph.node if node.op_type == "DequantizeLinear"]
        call_nodes = [node for node in rewritten.graph.node if node.op_type == "repeated_subgraph0"]

        assert [entry.role for entry in spec.inputs] == ["mxfp6_weight", "mxfp6_scale"]
        assert "layer.weight" not in graph_input_names
        assert "layer.weight" + MXFP6_PACKED_SUFFIX in graph_input_names
        assert "layer.weight" + MXFP6_SCALE_SUFFIX in graph_input_names
        assert "layer.weight" + MXFP6_PACKED_SUFFIX not in initializer_names
        assert "layer.weight" + MXFP6_SCALE_SUFFIX not in initializer_names
        assert len(unpack_nodes) == 0
        assert len(dq_nodes) == 0
        assert len(call_nodes) == 1
        assert call_nodes[0].input == ["x", "layer.weight" + MXFP6_PACKED_SUFFIX, "layer.weight" + MXFP6_SCALE_SUFFIX]

        rewritten_function = next(
            fn for fn in rewritten.functions if fn.domain == function_domain and fn.name == "repeated_subgraph0"
        )
        assert list(rewritten_function.input) == ["hidden_states", "arg2_1.mxfp6_packed", "arg2_1.mxfp6_scale"]
        assert [node.op_type for node in rewritten_function.node] == [
            "UnpackMxfp6",
            "DequantizeLinear",
            "Transpose",
            "MatMul",
        ]
        assert rewritten_function.node[0].input == ["arg2_1.mxfp6_packed"]
        assert rewritten_function.node[0].output == ["arg2_1.mxfp6_unpacked"]
        assert rewritten_function.node[1].input == ["arg2_1.mxfp6_unpacked", "arg2_1.mxfp6_scale"]
        assert rewritten_function.node[1].output == ["arg2_1"]
        assert rewritten_function.node[2].input == ["arg2_1"]
        assert rewritten_function.node[3].input[1] == "arg2_1_t"
        assert any(opset.domain == "" and opset.version == 28 for opset in rewritten_function.opset_import)
        assert any(
            opset.domain == "com.qti.aisw.onnx" and opset.version == 1 for opset in rewritten_function.opset_import
        )
        assert CustomOpTransform.apply(rewritten, onnx_export_opset=28)
        unpack_functions = [
            fn for fn in rewritten.functions if fn.domain == "com.qti.aisw.onnx" and fn.name == "UnpackMxfp6"
        ]
        assert len(unpack_functions) == 1

    def test_mxfp6_finalizer_supports_function_direct_matmul_topology(self, tmp_path, monkeypatch):
        monkeypatch.setattr("QEfficient.exporter.weight_free.mxfp6.validate_mxfp6_capabilities", lambda config: None)
        monkeypatch.setattr("QEfficient.exporter.weight_free.mxfp6.TensorProto.FLOAT6E2M3", 999, raising=False)

        prepared = tmp_path / "prepared"
        prepared.mkdir()
        weight = torch.arange(MXFP6_BLOCK_SIZE * MXFP6_BLOCK_SIZE, dtype=torch.float32).reshape(
            MXFP6_BLOCK_SIZE, MXFP6_BLOCK_SIZE
        )
        _write_safetensors_checkpoint(prepared, {"model.layers.0.linear.weight": weight})

        function_domain = "pkg.torch.__subgraph__"
        function = helper.make_function(
            domain=function_domain,
            fname="repeated_subgraph0",
            inputs=["hidden_states", "arg2_1"],
            outputs=["fn_y"],
            nodes=[helper.make_node("MatMul", ["hidden_states", "arg2_1"], ["fn_y"])],
            opset_imports=[helper.make_opsetid("", 18)],
        )

        x = helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, MXFP6_BLOCK_SIZE])
        w = helper.make_tensor_value_info("layer.weight", TensorProto.FLOAT, [MXFP6_BLOCK_SIZE, MXFP6_BLOCK_SIZE])
        y = helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, MXFP6_BLOCK_SIZE])
        graph = helper.make_graph(
            [helper.make_node("repeated_subgraph0", ["x", "layer.weight"], ["y"], domain=function_domain)],
            "mxfp6_function_direct_test",
            [x, w],
            [y],
        )
        model = helper.make_model(
            graph,
            opset_imports=[helper.make_opsetid("", 18), helper.make_opsetid(function_domain, 1)],
        )
        model.functions.append(function)
        onnx_path = tmp_path / "model.onnx"
        onnx.save(model, str(onnx_path))
        spec_path = save_weight_spec(
            tmp_path / "weight_spec.json",
            WeightSpec(
                model_name="tiny",
                model_id=str(prepared),
                files=[ExternalDataFile(path="model.safetensors", format="safetensors")],
                inputs=[
                    WeightSpecInput(
                        name="layer.weight",
                        location=WeightSpecLocation(file=0, key="model.layers.0.linear.weight"),
                    )
                ],
            ),
        )

        finalize_mxfp6_export(onnx_path, spec_path, str(prepared), Mxfp6Config(enabled=True, scale_dtype="float16"))

        rewritten = onnx.load(str(onnx_path), load_external_data=False)
        call_node = next(node for node in rewritten.graph.node if node.op_type == "repeated_subgraph0")
        rewritten_function = next(
            fn for fn in rewritten.functions if fn.domain == function_domain and fn.name == "repeated_subgraph0"
        )

        assert [
            node.op_type for node in rewritten.graph.node if node.op_type in {"UnpackMxfp6", "DequantizeLinear"}
        ] == []
        assert call_node.input == ["x", "layer.weight" + MXFP6_PACKED_SUFFIX, "layer.weight" + MXFP6_SCALE_SUFFIX]
        assert list(rewritten_function.input) == ["hidden_states", "arg2_1.mxfp6_packed", "arg2_1.mxfp6_scale"]
        assert [node.op_type for node in rewritten_function.node] == ["UnpackMxfp6", "DequantizeLinear", "MatMul"]
        assert rewritten_function.node[2].input == ["hidden_states", "arg2_1"]

    def test_mxfp6_finalizer_preserves_multiple_function_formal_ordering(self, tmp_path, monkeypatch):
        monkeypatch.setattr("QEfficient.exporter.weight_free.mxfp6.validate_mxfp6_capabilities", lambda config: None)
        monkeypatch.setattr("QEfficient.exporter.weight_free.mxfp6.TensorProto.FLOAT6E2M3", 999, raising=False)

        prepared = tmp_path / "prepared"
        prepared.mkdir()
        weight0 = torch.arange(MXFP6_BLOCK_SIZE * MXFP6_BLOCK_SIZE, dtype=torch.float32).reshape(
            MXFP6_BLOCK_SIZE, MXFP6_BLOCK_SIZE
        )
        weight1 = weight0 + 1000
        _write_safetensors_checkpoint(
            prepared,
            {
                "model.layers.0.linear0.weight": weight0,
                "model.layers.0.linear1.weight": weight1,
            },
        )

        function_domain = "pkg.torch.__subgraph__"
        function = helper.make_function(
            domain=function_domain,
            fname="repeated_subgraph0",
            inputs=["hidden_states", "arg1_1", "arg2_1"],
            outputs=["fn_y"],
            nodes=[
                helper.make_node("MatMul", ["hidden_states", "arg1_1"], ["hidden_1"]),
                helper.make_node("MatMul", ["hidden_1", "arg2_1"], ["fn_y"]),
            ],
            opset_imports=[helper.make_opsetid("", 18)],
        )

        x = helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, MXFP6_BLOCK_SIZE])
        w0 = helper.make_tensor_value_info("layer.weight0", TensorProto.FLOAT, [MXFP6_BLOCK_SIZE, MXFP6_BLOCK_SIZE])
        w1 = helper.make_tensor_value_info("layer.weight1", TensorProto.FLOAT, [MXFP6_BLOCK_SIZE, MXFP6_BLOCK_SIZE])
        y = helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, MXFP6_BLOCK_SIZE])
        graph = helper.make_graph(
            [
                helper.make_node(
                    "repeated_subgraph0", ["x", "layer.weight0", "layer.weight1"], ["y"], domain=function_domain
                )
            ],
            "mxfp6_function_order_test",
            [x, w0, w1],
            [y],
        )
        model = helper.make_model(
            graph,
            opset_imports=[helper.make_opsetid("", 18), helper.make_opsetid(function_domain, 1)],
        )
        model.functions.append(function)
        onnx_path = tmp_path / "model.onnx"
        onnx.save(model, str(onnx_path))
        spec_path = save_weight_spec(
            tmp_path / "weight_spec.json",
            WeightSpec(
                model_name="tiny",
                model_id=str(prepared),
                files=[ExternalDataFile(path="model.safetensors", format="safetensors")],
                inputs=[
                    WeightSpecInput(
                        name="layer.weight0",
                        location=WeightSpecLocation(file=0, key="model.layers.0.linear0.weight"),
                    ),
                    WeightSpecInput(
                        name="layer.weight1",
                        location=WeightSpecLocation(file=0, key="model.layers.0.linear1.weight"),
                    ),
                ],
            ),
        )

        finalize_mxfp6_export(onnx_path, spec_path, str(prepared), Mxfp6Config(enabled=True, scale_dtype="float16"))

        rewritten = onnx.load(str(onnx_path), load_external_data=False)
        call_node = next(node for node in rewritten.graph.node if node.op_type == "repeated_subgraph0")
        rewritten_function = next(
            fn for fn in rewritten.functions if fn.domain == function_domain and fn.name == "repeated_subgraph0"
        )

        assert call_node.input == [
            "x",
            "layer.weight0" + MXFP6_PACKED_SUFFIX,
            "layer.weight1" + MXFP6_PACKED_SUFFIX,
            "layer.weight0" + MXFP6_SCALE_SUFFIX,
            "layer.weight1" + MXFP6_SCALE_SUFFIX,
        ]
        assert list(rewritten_function.input) == [
            "hidden_states",
            "arg1_1.mxfp6_packed",
            "arg2_1.mxfp6_packed",
            "arg1_1.mxfp6_scale",
            "arg2_1.mxfp6_scale",
        ]
        assert [node.op_type for node in rewritten_function.node] == [
            "UnpackMxfp6",
            "DequantizeLinear",
            "MatMul",
            "UnpackMxfp6",
            "DequantizeLinear",
            "MatMul",
        ]

    def test_mxfp6_finalizer_rejects_incomplete_shared_function_calls(self, tmp_path, monkeypatch):
        monkeypatch.setattr("QEfficient.exporter.weight_free.mxfp6.validate_mxfp6_capabilities", lambda config: None)
        monkeypatch.setattr("QEfficient.exporter.weight_free.mxfp6.TensorProto.FLOAT6E2M3", 999, raising=False)

        prepared = tmp_path / "prepared"
        prepared.mkdir()
        weight = torch.arange(MXFP6_BLOCK_SIZE * MXFP6_BLOCK_SIZE, dtype=torch.float32).reshape(
            MXFP6_BLOCK_SIZE, MXFP6_BLOCK_SIZE
        )
        _write_safetensors_checkpoint(prepared, {"model.layers.0.linear.weight": weight})

        function_domain = "pkg.torch.__subgraph__"
        function = helper.make_function(
            domain=function_domain,
            fname="repeated_subgraph0",
            inputs=["hidden_states", "arg2_1"],
            outputs=["fn_y"],
            nodes=[helper.make_node("MatMul", ["hidden_states", "arg2_1"], ["fn_y"])],
            opset_imports=[helper.make_opsetid("", 18)],
        )

        x0 = helper.make_tensor_value_info("x0", TensorProto.FLOAT, [1, MXFP6_BLOCK_SIZE])
        x1 = helper.make_tensor_value_info("x1", TensorProto.FLOAT, [1, MXFP6_BLOCK_SIZE])
        w0 = helper.make_tensor_value_info("layer0.weight", TensorProto.FLOAT, [MXFP6_BLOCK_SIZE, MXFP6_BLOCK_SIZE])
        w1 = helper.make_tensor_value_info("layer1.weight", TensorProto.FLOAT, [MXFP6_BLOCK_SIZE, MXFP6_BLOCK_SIZE])
        y0 = helper.make_tensor_value_info("y0", TensorProto.FLOAT, [1, MXFP6_BLOCK_SIZE])
        y1 = helper.make_tensor_value_info("y1", TensorProto.FLOAT, [1, MXFP6_BLOCK_SIZE])
        graph = helper.make_graph(
            [
                helper.make_node("repeated_subgraph0", ["x0", "layer0.weight"], ["y0"], domain=function_domain),
                helper.make_node("repeated_subgraph0", ["x1", "layer1.weight"], ["y1"], domain=function_domain),
            ],
            "mxfp6_function_incomplete_test",
            [x0, x1, w0, w1],
            [y0, y1],
        )
        model = helper.make_model(
            graph,
            opset_imports=[helper.make_opsetid("", 18), helper.make_opsetid(function_domain, 1)],
        )
        model.functions.append(function)
        onnx_path = tmp_path / "model.onnx"
        onnx.save(model, str(onnx_path))
        spec_path = save_weight_spec(
            tmp_path / "weight_spec.json",
            WeightSpec(
                model_name="tiny",
                model_id=str(prepared),
                files=[ExternalDataFile(path="model.safetensors", format="safetensors")],
                inputs=[
                    WeightSpecInput(
                        name="layer0.weight",
                        location=WeightSpecLocation(file=0, key="model.layers.0.linear.weight"),
                    )
                ],
            ),
        )

        with pytest.raises(NotImplementedError, match="partially rewrite shared ONNX subfunctions"):
            finalize_mxfp6_export(onnx_path, spec_path, str(prepared), Mxfp6Config(enabled=True, scale_dtype="float16"))

    def test_mxfp6_finalizer_skips_unsupported_function_formal_arg(self, tmp_path, monkeypatch):
        monkeypatch.setattr("QEfficient.exporter.weight_free.mxfp6.validate_mxfp6_capabilities", lambda config: None)
        monkeypatch.setattr("QEfficient.exporter.weight_free.mxfp6.TensorProto.FLOAT6E2M3", 999, raising=False)

        prepared = tmp_path / "prepared"
        prepared.mkdir()
        linear_weight = torch.arange(4 * MXFP6_BLOCK_SIZE, dtype=torch.float32).reshape(4, MXFP6_BLOCK_SIZE)
        norm_weight = torch.arange(MXFP6_BLOCK_SIZE, dtype=torch.float32)
        _write_safetensors_checkpoint(
            prepared,
            {
                "model.layers.0.linear.weight": linear_weight,
                "model.layers.0.input_layernorm.weight": norm_weight,
            },
        )

        function_domain = "pkg.torch.__subgraph__"
        linear_function = helper.make_function(
            domain=function_domain,
            fname="repeated_subgraph0",
            inputs=["hidden_states", "arg2_1"],
            outputs=["linear_y"],
            nodes=[
                helper.make_node("Transpose", ["arg2_1"], ["arg2_1_t"], perm=[1, 0]),
                helper.make_node("MatMul", ["hidden_states", "arg2_1_t"], ["linear_y"]),
            ],
            opset_imports=[helper.make_opsetid("", 18)],
        )
        rmsnorm_function = helper.make_function(
            domain=function_domain,
            fname="repeated_subgraph1",
            inputs=["hidden_states", "arg5_1"],
            outputs=["norm_y"],
            nodes=[
                helper.make_node(
                    "CustomRMSNorm",
                    ["hidden_states", "arg5_1"],
                    ["norm_y"],
                    domain="com.qti.aisw.onnx",
                    epsilon_f=1e-5,
                )
            ],
            opset_imports=[helper.make_opsetid("", 18), helper.make_opsetid("com.qti.aisw.onnx", 1)],
        )

        x = helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, MXFP6_BLOCK_SIZE])
        linear_w = helper.make_tensor_value_info("linear.weight", TensorProto.FLOAT, [4, MXFP6_BLOCK_SIZE])
        norm_w = helper.make_tensor_value_info("norm.weight", TensorProto.FLOAT, [MXFP6_BLOCK_SIZE])
        linear_y = helper.make_tensor_value_info("linear_y", TensorProto.FLOAT, [1, 4])
        norm_y = helper.make_tensor_value_info("norm_y", TensorProto.FLOAT, [1, MXFP6_BLOCK_SIZE])
        graph = helper.make_graph(
            [
                helper.make_node("repeated_subgraph0", ["x", "linear.weight"], ["linear_y"], domain=function_domain),
                helper.make_node("repeated_subgraph1", ["x", "norm.weight"], ["norm_y"], domain=function_domain),
            ],
            "mxfp6_unsupported_function_test",
            [x, linear_w, norm_w],
            [linear_y, norm_y],
        )
        model = helper.make_model(
            graph,
            opset_imports=[
                helper.make_opsetid("", 18),
                helper.make_opsetid(function_domain, 1),
                helper.make_opsetid("com.qti.aisw.onnx", 1),
            ],
        )
        model.functions.extend([linear_function, rmsnorm_function])
        onnx_path = tmp_path / "model.onnx"
        onnx.save(model, str(onnx_path))
        spec_path = save_weight_spec(
            tmp_path / "weight_spec.json",
            WeightSpec(
                model_name="tiny",
                model_id=str(prepared),
                files=[ExternalDataFile(path="model.safetensors", format="safetensors")],
                inputs=[
                    WeightSpecInput(
                        name="linear.weight",
                        location=WeightSpecLocation(file=0, key="model.layers.0.linear.weight"),
                    ),
                    WeightSpecInput(
                        name="norm.weight",
                        location=WeightSpecLocation(file=0, key="model.layers.0.input_layernorm.weight"),
                    ),
                ],
            ),
        )

        finalize_mxfp6_export(onnx_path, spec_path, str(prepared), Mxfp6Config(enabled=True, scale_dtype="float16"))

        spec = load_weight_spec(spec_path)
        rewritten = onnx.load(str(onnx_path), load_external_data=False)
        graph_input_names = {value_info.name for value_info in rewritten.graph.input}
        spec_by_name = {entry.name: entry for entry in spec.inputs}

        assert "linear.weight" not in graph_input_names
        assert "linear.weight" + MXFP6_PACKED_SUFFIX in graph_input_names
        assert "linear.weight" + MXFP6_SCALE_SUFFIX in graph_input_names
        assert spec_by_name["linear.weight" + MXFP6_PACKED_SUFFIX].role == "mxfp6_weight"
        assert spec_by_name["linear.weight" + MXFP6_SCALE_SUFFIX].role == "mxfp6_scale"

        assert "norm.weight" in graph_input_names
        assert "norm.weight" + MXFP6_PACKED_SUFFIX not in graph_input_names
        assert "norm.weight" + MXFP6_SCALE_SUFFIX not in graph_input_names
        assert spec_by_name["norm.weight"].role == "weight"
        assert "norm.weight" + MXFP6_PACKED_SUFFIX not in spec_by_name
        assert "norm.weight" + MXFP6_SCALE_SUFFIX not in spec_by_name

    def test_ort_loader_rejects_mxfp6_weight_spec(self, tmp_path):
        spec_path = save_weight_spec(
            tmp_path / "weight_spec.json",
            WeightSpec(
                model_name="tiny",
                model_id="tiny",
                inputs=[
                    WeightSpecInput(
                        name="linear.weight.mxfp6_packed",
                        location=WeightSpecLocation(file=0, key="linear.weight"),
                        role="mxfp6_weight",
                    )
                ],
            ),
        )

        with pytest.raises(NotImplementedError, match="MXFP6"):
            load_weight_free_ort_inputs(spec_path, {})


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

    def test_mxfp6_export_hash_separates_scale_dtypes(self):
        config = SimpleNamespace(to_diff_dict=lambda: {"model_type": "llama"})
        common_model = SimpleNamespace(
            model=SimpleNamespace(config=config),
            hash_params={"pretrained_model_name_or_path": "tiny"},
            _use_onnx_subfunctions=False,
            _weight_free=True,
        )
        fp16_model = SimpleNamespace(
            **common_model.__dict__,
            _mxfp6_config=Mxfp6Config(enabled=True, scale_dtype="float16"),
        )
        bf16_model = SimpleNamespace(
            **common_model.__dict__,
            _mxfp6_config=Mxfp6Config(enabled=True, scale_dtype="bfloat16"),
        )
        common_kwargs = {
            "example_inputs": {"input_ids": torch.ones(1, 2, dtype=torch.int64)},
            "output_names": ["logits"],
            "dynamic_axes": {"input_ids": {0: "batch_size"}},
            "dynamo": True,
        }

        common_hash, _ = _generate_export_hash(common_model, (), dict(common_kwargs), _fake_export)
        fp16_hash, fp16_params = _generate_export_hash(fp16_model, (), dict(common_kwargs), _fake_export)
        bf16_hash, bf16_params = _generate_export_hash(bf16_model, (), dict(common_kwargs), _fake_export)

        assert len({common_hash, fp16_hash, bf16_hash}) == 3
        assert fp16_params["mxfp6"] is True
        assert fp16_params["mxfp6_scale_dtype"] == "float16"
        assert bf16_params["mxfp6_scale_dtype"] == "bfloat16"


class TestRuntimeRequirements:
    def test_validate_runtime_requirements_accepts_matching_requirements(self, monkeypatch):
        requirements = {"torch": "==2.13.0", "accelerate": "==1.9.0"}
        versions = {"torch": "2.13.0+cpu", "accelerate": "1.9.0"}

        monkeypatch.setattr(runtime_requirements.metadata, "version", lambda name: versions[name])

        validate_runtime_requirements(
            requirements,
            feature_name="weight_free=True",
            install_command="pip install -r requirements.txt",
        )

    def test_validate_runtime_requirements_reports_all_mismatches(self, monkeypatch):
        requirements = {"accelerate": "==1.9.0", "missing-package": "==1.0"}

        def version(name):
            if name == "missing-package":
                raise runtime_requirements.metadata.PackageNotFoundError
            return "1.14.0"

        monkeypatch.setattr(runtime_requirements.metadata, "version", version)

        with pytest.raises(AssertionError) as exc_info:
            validate_runtime_requirements(
                requirements,
                feature_name="dynamo=True",
                install_command="pip install -r requirements.txt",
            )

        message = str(exc_info.value)
        assert "dynamo=True requires the Dynamo export environment" in message
        assert "Mismatched packages:" in message
        assert "Install or repair the environment with:" in message
        assert "accelerate: installed 1.14.0, expected ==1.9.0" in message
        assert "missing-package: not installed, expected ==1.0" in message

    def test_weight_free_from_pretrained_validates_requirements_before_model_build(self, monkeypatch):
        calls = []

        def validate(feature_name):
            calls.append(feature_name)
            raise AssertionError("bad weight-free environment")

        monkeypatch.setattr(
            "QEfficient.transformers.models.modeling_auto.validate_dynamo_export_requirements",
            validate,
        )
        monkeypatch.setattr(
            "QEfficient.transformers.models.modeling_auto._build_meta_model",
            lambda *args, **kwargs: pytest.fail("_build_meta_model should not be called"),
        )

        with pytest.raises(AssertionError, match="bad weight-free environment"):
            QEFFAutoModelForCausalLM.from_pretrained("dummy-model", weight_free=True)

        assert calls == ["weight_free=True"]

    def test_export_wrapper_validates_dynamo_requirements_before_export(self, monkeypatch, tmp_path):
        from QEfficient.utils.export_utils import export_wrapper

        calls = []

        def validate(feature_name):
            calls.append(feature_name)
            raise AssertionError("bad dynamo environment")

        monkeypatch.setattr("QEfficient.utils.export_utils.validate_dynamo_export_requirements", validate)

        class DummyQEff:
            @export_wrapper
            def export(
                self,
                example_inputs,
                output_names,
                dynamic_axes,
                export_dir=None,
                dynamo=False,
                dynamic_shapes=None,
            ):
                pytest.fail("export should not run")

        with (
            pytest.warns(DeprecationWarning, match="Direct \\.export\\(\\) is deprecated"),
            pytest.raises(AssertionError, match="bad dynamo environment"),
        ):
            DummyQEff().export(
                example_inputs={},
                output_names=[],
                dynamic_axes={},
                export_dir=tmp_path,
                dynamo=True,
            )

        assert calls == ["dynamo=True"]


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

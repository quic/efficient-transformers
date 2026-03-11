# -----------------------------------------------------------------------------
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------
"""
Tests for PEFT/LoRA transforms in QEfficient.

Tests verify:
  - QEffPeftModelForCausalLM: importable, has correct class structure
  - LoRA pytorch transforms: importable, have apply method
  - LoRA ONNX transforms: importable, have apply method
  - Wrapping a tiny Llama model with LoRA adapter works without error
  - LoRA-wrapped model produces finite logits

All tests run on CPU only, no network downloads required.
"""

import pytest
import torch
from transformers import LlamaConfig, LlamaForCausalLM

VOCAB_SIZE = 500
SEQ_LEN = 8
CTX_LEN = 32


def make_tiny_llama():
    cfg = LlamaConfig(
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=2,
        hidden_size=64,
        intermediate_size=128,
        vocab_size=VOCAB_SIZE,
        max_position_embeddings=CTX_LEN,
    )
    return LlamaForCausalLM(cfg).eval(), cfg


# ---------------------------------------------------------------------------
# Tests: PEFT module importability
# ---------------------------------------------------------------------------


class TestPEFTModuleImportability:
    """PEFT modules must be importable and have correct structure."""

    def test_qeff_peft_model_for_causal_lm_importable(self):
        from QEfficient.peft.auto import QEffAutoPeftModelForCausalLM
        assert QEffAutoPeftModelForCausalLM is not None

    def test_peft_pytorch_transforms_importable(self):
        from QEfficient.peft.pytorch_transforms import PeftModelInputsTransform
        assert PeftModelInputsTransform is not None

    def test_peft_onnx_transforms_importable(self):
        from QEfficient.peft.onnx_transforms import AdapterWeightsToInputsTransform
        assert AdapterWeightsToInputsTransform is not None

    def test_qeff_peft_model_has_from_pretrained(self):
        from QEfficient.peft.auto import QEffAutoPeftModelForCausalLM
        assert hasattr(QEffAutoPeftModelForCausalLM, "from_pretrained")
        assert callable(QEffAutoPeftModelForCausalLM.from_pretrained)

    def test_qeff_peft_model_has_pytorch_transforms(self):
        from QEfficient.peft.auto import QEffAutoPeftModelForCausalLM
        assert hasattr(QEffAutoPeftModelForCausalLM, "_pytorch_transforms")
        assert isinstance(QEffAutoPeftModelForCausalLM._pytorch_transforms, list)

    def test_qeff_peft_model_has_onnx_transforms(self):
        from QEfficient.peft.auto import QEffAutoPeftModelForCausalLM
        assert hasattr(QEffAutoPeftModelForCausalLM, "_onnx_transforms")
        assert isinstance(QEffAutoPeftModelForCausalLM._onnx_transforms, list)

    def test_peft_inputs_transform_has_apply(self):
        from QEfficient.peft.pytorch_transforms import PeftModelInputsTransform
        assert hasattr(PeftModelInputsTransform, "apply")
        assert callable(PeftModelInputsTransform.apply)

    def test_adapter_weights_transform_has_apply(self):
        from QEfficient.peft.onnx_transforms import AdapterWeightsToInputsTransform
        assert hasattr(AdapterWeightsToInputsTransform, "apply")
        assert callable(AdapterWeightsToInputsTransform.apply)

    def test_peft_model_importable_from_qefficient(self):
        """QEffAutoPeftModelForCausalLM must be accessible from the QEfficient package."""
        import QEfficient
        assert hasattr(QEfficient, "QEffAutoPeftModelForCausalLM")


# ---------------------------------------------------------------------------
# Tests: LoRA transform structure
# ---------------------------------------------------------------------------


class TestLoRATransformStructure:
    """LoRA transforms must have correct structure."""

    def test_peft_inputs_transform_has_apply_classmethod(self):
        from QEfficient.peft.pytorch_transforms import PeftModelInputsTransform
        import inspect
        assert isinstance(
            inspect.getattr_static(PeftModelInputsTransform, "apply"),
            classmethod,
        ), "PeftModelInputsTransform.apply must be a classmethod"

    def test_adapter_weights_transform_has_apply_classmethod(self):
        from QEfficient.peft.onnx_transforms import AdapterWeightsToInputsTransform
        import inspect
        assert isinstance(
            inspect.getattr_static(AdapterWeightsToInputsTransform, "apply"),
            classmethod,
        ), "AdapterWeightsToInputsTransform.apply must be a classmethod"

    def test_peft_pytorch_transforms_include_peft_inputs_transform(self):
        from QEfficient.peft.auto import QEffAutoPeftModelForCausalLM
        from QEfficient.peft.pytorch_transforms import PeftModelInputsTransform
        assert PeftModelInputsTransform in QEffAutoPeftModelForCausalLM._pytorch_transforms, (
            "PeftModelInputsTransform not in QEffAutoPeftModelForCausalLM._pytorch_transforms"
        )



# ---------------------------------------------------------------------------
# Tests: LoRA wrapping with peft library
# ---------------------------------------------------------------------------


class TestLoRAWrapping:
    """LoRA adapter wrapping must work without error on a tiny model."""

    def _make_lora_model(self):
        """Create a tiny Llama model with a LoRA adapter using peft library."""
        try:
            from peft import LoraConfig, get_peft_model
        except ImportError:
            pytest.skip("peft library not installed")

        model, cfg = make_tiny_llama()
        lora_config = LoraConfig(
            r=4,
            lora_alpha=8,
            target_modules=["q_proj", "v_proj"],
            lora_dropout=0.0,
            bias="none",
            task_type="CAUSAL_LM",
        )
        lora_model = get_peft_model(model, lora_config)
        return lora_model, cfg

    def test_lora_model_wraps_without_error(self):
        lora_model, cfg = self._make_lora_model()
        assert lora_model is not None

    def test_lora_model_has_lora_parameters(self):
        lora_model, cfg = self._make_lora_model()
        lora_params = [n for n, _ in lora_model.named_parameters() if "lora_" in n]
        assert len(lora_params) > 0, "LoRA model must have lora_ parameters"

    def test_lora_model_forward_produces_finite_logits(self):
        lora_model, cfg = self._make_lora_model()
        input_ids = torch.randint(0, VOCAB_SIZE, (1, SEQ_LEN))
        with torch.no_grad():
            out = lora_model(input_ids=input_ids)
        assert torch.isfinite(out.logits).all(), "LoRA model must produce finite logits"

    def test_qeff_peft_model_wraps_lora_model(self):
        """QEffAutoPeftModelForCausalLM must wrap a LoRA model without error."""
        from QEfficient.peft.auto import QEffAutoPeftModelForCausalLM

        lora_model, cfg = self._make_lora_model()
        qeff_peft = QEffAutoPeftModelForCausalLM(lora_model)
        assert qeff_peft is not None
        assert hasattr(qeff_peft, "model")

    def test_qeff_peft_model_has_model_name(self):
        from QEfficient.peft.auto import QEffAutoPeftModelForCausalLM

        lora_model, cfg = self._make_lora_model()
        qeff_peft = QEffAutoPeftModelForCausalLM(lora_model)
        assert hasattr(qeff_peft, "model_name")
        assert isinstance(qeff_peft.model_name, str)
        assert len(qeff_peft.model_name) > 0

    def test_qeff_peft_model_forward_produces_finite_logits(self):
        """QEffAutoPeftModelForCausalLM forward must produce finite logits."""
        from QEfficient.peft.auto import QEffAutoPeftModelForCausalLM

        lora_model, cfg = self._make_lora_model()
        qeff_peft = QEffAutoPeftModelForCausalLM(lora_model)

        n_layers = cfg.num_hidden_layers
        n_kv = cfg.num_key_value_heads
        head_dim = cfg.hidden_size // cfg.num_attention_heads
        input_ids = torch.randint(0, VOCAB_SIZE, (1, SEQ_LEN))
        position_ids = torch.arange(SEQ_LEN).unsqueeze(0)
        past_key_values = tuple(
            (
                torch.zeros(1, n_kv, CTX_LEN, head_dim),
                torch.zeros(1, n_kv, CTX_LEN, head_dim),
            )
            for _ in range(n_layers)
        )
        with torch.no_grad():
            out = qeff_peft.model(
                input_ids=input_ids,
                position_ids=position_ids,
                past_key_values=past_key_values,
            )
        assert torch.isfinite(out.logits).all(), "QEffPeftModelForCausalLM must produce finite logits"


# ---------------------------------------------------------------------------
# Tests: LoRA accuracy vs base model (GAP G)
# ---------------------------------------------------------------------------


class TestLoRAAccuracyVsBase:
    """LoRA model must produce different logits than base model (LoRA changes outputs)."""

    def _make_lora_model_and_base(self):
        """Create a tiny Llama model and a LoRA-wrapped version."""
        try:
            from peft import LoraConfig, get_peft_model
        except ImportError:
            pytest.skip("peft library not installed")

        model, cfg = make_tiny_llama()
        # Save base model logits before LoRA wrapping
        base_model = model

        lora_config = LoraConfig(
            r=4,
            lora_alpha=8,
            target_modules=["q_proj", "v_proj"],
            lora_dropout=0.0,
            bias="none",
            task_type="CAUSAL_LM",
        )
        lora_model = get_peft_model(base_model, lora_config)
        return lora_model, base_model, cfg

    def test_lora_model_logits_are_finite(self):
        """LoRA model logits must be finite (no NaN/Inf)."""
        lora_model, base_model, cfg = self._make_lora_model_and_base()
        input_ids = torch.randint(0, VOCAB_SIZE, (1, SEQ_LEN))
        with torch.no_grad():
            out = lora_model(input_ids=input_ids)
        assert torch.isfinite(out.logits).all(), "LoRA model must produce finite logits"

    def test_lora_model_output_shape_matches_base(self):
        """LoRA model output shape must match base model output shape."""
        lora_model, base_model, cfg = self._make_lora_model_and_base()
        input_ids = torch.randint(0, VOCAB_SIZE, (1, SEQ_LEN))
        with torch.no_grad():
            lora_out = lora_model(input_ids=input_ids)
        assert lora_out.logits.shape == (1, SEQ_LEN, VOCAB_SIZE), (
            f"LoRA output shape mismatch: {lora_out.logits.shape}"
        )

    def test_lora_model_with_random_weights_differs_from_base(self):
        """LoRA model with random (non-zero) weights must produce different logits than base."""
        try:
            from peft import LoraConfig, get_peft_model
        except ImportError:
            pytest.skip("peft library not installed")

        model, cfg = make_tiny_llama()
        input_ids = torch.randint(0, VOCAB_SIZE, (1, SEQ_LEN))

        # Get base model logits
        with torch.no_grad():
            base_logits = model(input_ids=input_ids).logits

        # Wrap with LoRA and initialize with non-zero weights
        lora_config = LoraConfig(
            r=4,
            lora_alpha=8,
            target_modules=["q_proj", "v_proj"],
            lora_dropout=0.0,
            bias="none",
            task_type="CAUSAL_LM",
        )
        lora_model = get_peft_model(model, lora_config)

        # Initialize LoRA B matrices with non-zero values (default is zeros)
        for name, param in lora_model.named_parameters():
            if "lora_B" in name:
                torch.nn.init.normal_(param, mean=0.0, std=0.1)

        with torch.no_grad():
            lora_logits = lora_model(input_ids=input_ids).logits

        max_diff = (base_logits - lora_logits).abs().max().item()
        assert max_diff > 1e-6, (
            f"LoRA model with non-zero B weights must produce different logits than base. "
            f"max_diff={max_diff:.2e}"
        )

    def test_lora_model_with_zero_b_weights_matches_base(self):
        """LoRA model with zero B weights (default init) must produce same logits as base."""
        try:
            from peft import LoraConfig, get_peft_model
        except ImportError:
            pytest.skip("peft library not installed")

        model, cfg = make_tiny_llama()
        input_ids = torch.randint(0, VOCAB_SIZE, (1, SEQ_LEN))

        # Get base model logits
        with torch.no_grad():
            base_logits = model(input_ids=input_ids).logits

        # Wrap with LoRA (default: B=0, so output is same as base)
        lora_config = LoraConfig(
            r=4,
            lora_alpha=8,
            target_modules=["q_proj", "v_proj"],
            lora_dropout=0.0,
            bias="none",
            task_type="CAUSAL_LM",
        )
        lora_model = get_peft_model(model, lora_config)

        with torch.no_grad():
            lora_logits = lora_model(input_ids=input_ids).logits

        max_diff = (base_logits - lora_logits).abs().max().item()
        assert max_diff < 1e-5, (
            f"LoRA model with zero B weights must match base model. max_diff={max_diff:.2e}"
        )

    def test_lora_trainable_params_are_subset_of_all_params(self):
        """LoRA trainable parameters must be a subset of all parameters."""
        try:
            from peft import LoraConfig, get_peft_model
        except ImportError:
            pytest.skip("peft library not installed")

        model, cfg = make_tiny_llama()
        lora_config = LoraConfig(
            r=4,
            lora_alpha=8,
            target_modules=["q_proj", "v_proj"],
            lora_dropout=0.0,
            bias="none",
            task_type="CAUSAL_LM",
        )
        lora_model = get_peft_model(model, lora_config)

        trainable_params = sum(p.numel() for p in lora_model.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in lora_model.parameters())
        assert trainable_params < total_params, (
            f"LoRA trainable params ({trainable_params}) must be less than total ({total_params})"
        )


# ---------------------------------------------------------------------------
# Tests: AdapterWeightsToInputsTransform ONNX graph (GAP G)
# ---------------------------------------------------------------------------


class TestAdapterWeightsToInputsTransformStructure:
    """AdapterWeightsToInputsTransform must have correct structure."""

    def test_adapter_weights_transform_importable(self):
        from QEfficient.peft.onnx_transforms import AdapterWeightsToInputsTransform
        assert AdapterWeightsToInputsTransform is not None

    def test_adapter_weights_transform_has_apply_method(self):
        from QEfficient.peft.onnx_transforms import AdapterWeightsToInputsTransform
        assert hasattr(AdapterWeightsToInputsTransform, "apply")
        assert callable(AdapterWeightsToInputsTransform.apply)

    def test_adapter_weights_transform_apply_is_classmethod(self):
        import inspect
        from QEfficient.peft.onnx_transforms import AdapterWeightsToInputsTransform
        assert isinstance(
            inspect.getattr_static(AdapterWeightsToInputsTransform, "apply"),
            classmethod,
        ), "AdapterWeightsToInputsTransform.apply must be a classmethod"

    def test_adapter_weights_transform_in_peft_onnx_transforms(self):
        """AdapterWeightsToInputsTransform (from base or peft) must be in QEffAutoPeftModelForCausalLM._onnx_transforms."""
        from QEfficient.peft.auto import QEffAutoPeftModelForCausalLM
        # AdapterWeightsToInputsTransform may be in base.onnx_transforms or peft.onnx_transforms
        transform_names = [t.__name__ for t in QEffAutoPeftModelForCausalLM._onnx_transforms]
        assert "AdapterWeightsToInputsTransform" in transform_names, (
            f"AdapterWeightsToInputsTransform not in QEffAutoPeftModelForCausalLM._onnx_transforms. "
            f"Found: {transform_names}"
        )

    def test_peft_onnx_transforms_list_not_empty(self):
        """QEffAutoPeftModelForCausalLM._onnx_transforms must not be empty."""
        from QEfficient.peft.auto import QEffAutoPeftModelForCausalLM
        assert len(QEffAutoPeftModelForCausalLM._onnx_transforms) > 0

    def test_peft_pytorch_transforms_list_not_empty(self):
        """QEffAutoPeftModelForCausalLM._pytorch_transforms must not be empty."""
        from QEfficient.peft.auto import QEffAutoPeftModelForCausalLM
        assert len(QEffAutoPeftModelForCausalLM._pytorch_transforms) > 0

    def test_peft_model_has_export_method(self):
        """QEffAutoPeftModelForCausalLM must have an export() method."""
        from QEfficient.peft.auto import QEffAutoPeftModelForCausalLM
        assert hasattr(QEffAutoPeftModelForCausalLM, "export")
        assert callable(QEffAutoPeftModelForCausalLM.export)

    def test_peft_model_has_compile_method(self):
        """QEffAutoPeftModelForCausalLM must have a compile() method."""
        from QEfficient.peft.auto import QEffAutoPeftModelForCausalLM
        assert hasattr(QEffAutoPeftModelForCausalLM, "compile")
        assert callable(QEffAutoPeftModelForCausalLM.compile)

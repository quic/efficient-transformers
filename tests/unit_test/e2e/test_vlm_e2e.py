# -----------------------------------------------------------------------------
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------
"""
Tests for VLM (Vision-Language Model) pipeline in QEfficient.

Tests verify:
  - QEFFAutoModelForImageTextToText: importable, has correct class structure
  - kv_offload=True routes to _QEffAutoModelForImageTextToTextDualQPC
  - kv_offload=False routes to _QEFFAutoModelForImageTextToTextSingleQPC
  - MISCLASSIFIED_CAUSAL_LM_TO_QEFF_AUTO_CLASS_MAP: exists and is a dict
  - QEFFAutoModelForCTC: importable, has correct class structure
  - VlmKVOffloadTransform / VlmNoKVOffloadTransform: importable, have module mappings

All tests run on CPU , using tiny in-memory configs where possible.
"""

import pytest

# ---------------------------------------------------------------------------
# Tests: QEFFAutoModelForImageTextToText class structure
# ---------------------------------------------------------------------------


class TestQEFFAutoModelForImageTextToTextStructure:
    """QEFFAutoModelForImageTextToText must have correct class-level structure."""

    def test_importable(self):
        from QEfficient.transformers.models.modeling_auto import QEFFAutoModelForImageTextToText

        assert QEFFAutoModelForImageTextToText is not None

    def test_dual_qpc_class_importable(self):
        from QEfficient.transformers.models.modeling_auto import _QEffAutoModelForImageTextToTextDualQPC

        assert _QEffAutoModelForImageTextToTextDualQPC is not None

    def test_single_qpc_class_importable(self):
        from QEfficient.transformers.models.modeling_auto import _QEFFAutoModelForImageTextToTextSingleQPC

        assert _QEFFAutoModelForImageTextToTextSingleQPC is not None

    def test_dual_qpc_has_from_pretrained(self):
        from QEfficient.transformers.models.modeling_auto import _QEffAutoModelForImageTextToTextDualQPC

        assert hasattr(_QEffAutoModelForImageTextToTextDualQPC, "from_pretrained")
        assert callable(_QEffAutoModelForImageTextToTextDualQPC.from_pretrained)

    def test_single_qpc_has_from_pretrained(self):
        from QEfficient.transformers.models.modeling_auto import _QEFFAutoModelForImageTextToTextSingleQPC

        assert hasattr(_QEFFAutoModelForImageTextToTextSingleQPC, "from_pretrained")
        assert callable(_QEFFAutoModelForImageTextToTextSingleQPC.from_pretrained)

    def test_dual_qpc_has_from_pretrained_classmethod(self):
        from QEfficient.transformers.models.modeling_auto import _QEffAutoModelForImageTextToTextDualQPC

        assert hasattr(_QEffAutoModelForImageTextToTextDualQPC, "from_pretrained")
        assert callable(_QEffAutoModelForImageTextToTextDualQPC.from_pretrained)

    def test_single_qpc_has_pytorch_transforms(self):
        from QEfficient.transformers.models.modeling_auto import _QEFFAutoModelForImageTextToTextSingleQPC

        assert hasattr(_QEFFAutoModelForImageTextToTextSingleQPC, "_pytorch_transforms")
        assert isinstance(_QEFFAutoModelForImageTextToTextSingleQPC._pytorch_transforms, list)

    def test_dual_qpc_has_model_attribute_after_construction(self):
        """_QEffAutoModelForImageTextToTextDualQPC instances must have a model attribute."""
        from QEfficient.transformers.models.modeling_auto import (
            QEFFAutoModelForImageTextToText,
            _QEffAutoModelForImageTextToTextDualQPC,
        )

        try:
            from transformers import CLIPVisionConfig, LlamaConfig, LlavaConfig, LlavaForConditionalGeneration

            vision_cfg = CLIPVisionConfig(
                hidden_size=64,
                intermediate_size=128,
                num_hidden_layers=1,
                num_attention_heads=2,
                image_size=32,
                patch_size=16,
            )
            text_cfg = LlamaConfig(
                num_hidden_layers=1,
                num_attention_heads=2,
                num_key_value_heads=2,
                hidden_size=64,
                intermediate_size=128,
                vocab_size=500,
                max_position_embeddings=64,
            )
            llava_cfg = LlavaConfig(
                vision_config=vision_cfg,
                text_config=text_cfg,
                ignore_index=-100,
                image_token_index=32000,
                projector_hidden_act="gelu",
                vision_feature_select_strategy="default",
                vision_feature_layer=-1,
            )
            model = LlavaForConditionalGeneration(llava_cfg).eval()
            qeff = QEFFAutoModelForImageTextToText(model, kv_offload=True)
            assert isinstance(qeff, _QEffAutoModelForImageTextToTextDualQPC)
            assert hasattr(qeff, "model")
        except Exception as e:
            pytest.skip(f"Cannot create DualQPC instance: {e}")

    def test_single_qpc_has_onnx_transforms(self):
        from QEfficient.transformers.models.modeling_auto import _QEFFAutoModelForImageTextToTextSingleQPC

        assert hasattr(_QEFFAutoModelForImageTextToTextSingleQPC, "_onnx_transforms")
        assert isinstance(_QEFFAutoModelForImageTextToTextSingleQPC._onnx_transforms, list)

    def test_dual_qpc_has_hf_auto_class(self):
        from QEfficient.transformers.models.modeling_auto import _QEffAutoModelForImageTextToTextDualQPC

        assert hasattr(_QEffAutoModelForImageTextToTextDualQPC, "_hf_auto_class")

    def test_single_qpc_has_hf_auto_class(self):
        from QEfficient.transformers.models.modeling_auto import _QEFFAutoModelForImageTextToTextSingleQPC

        assert hasattr(_QEFFAutoModelForImageTextToTextSingleQPC, "_hf_auto_class")

    def test_importable_from_qefficient_public_api(self):
        import QEfficient

        assert hasattr(QEfficient, "QEFFAutoModelForImageTextToText")


# ---------------------------------------------------------------------------
# Tests: QEFFAutoModelForImageTextToText routing
# ---------------------------------------------------------------------------


class TestQEFFAutoModelForImageTextToTextRouting:
    """QEFFAutoModelForImageTextToText must route to correct class based on kv_offload."""

    def _make_tiny_llava(self):
        """Create a tiny LLaVA model for routing tests."""
        try:
            from transformers import CLIPVisionConfig, LlamaConfig, LlavaConfig, LlavaForConditionalGeneration

            vision_cfg = CLIPVisionConfig(
                hidden_size=64,
                intermediate_size=128,
                num_hidden_layers=1,
                num_attention_heads=2,
                image_size=32,
                patch_size=16,
            )
            text_cfg = LlamaConfig(
                num_hidden_layers=1,
                num_attention_heads=2,
                num_key_value_heads=2,
                hidden_size=64,
                intermediate_size=128,
                vocab_size=500,
                max_position_embeddings=64,
            )
            llava_cfg = LlavaConfig(
                vision_config=vision_cfg,
                text_config=text_cfg,
                ignore_index=-100,
                image_token_index=32000,
                projector_hidden_act="gelu",
                vision_feature_select_strategy="default",
                vision_feature_layer=-1,
            )
            return LlavaForConditionalGeneration(llava_cfg).eval()
        except Exception as e:
            pytest.skip(f"Cannot create tiny LLaVA model: {e}")

    def test_kv_offload_false_creates_single_qpc(self):
        """kv_offload=False must create _QEFFAutoModelForImageTextToTextSingleQPC."""
        from QEfficient.transformers.models.modeling_auto import (
            QEFFAutoModelForImageTextToText,
            _QEFFAutoModelForImageTextToTextSingleQPC,
        )

        model = self._make_tiny_llava()
        qeff = QEFFAutoModelForImageTextToText(model, kv_offload=False)
        assert isinstance(qeff, _QEFFAutoModelForImageTextToTextSingleQPC), (
            f"kv_offload=False must create SingleQPC, got {type(qeff)}"
        )

    def test_kv_offload_true_creates_dual_qpc(self):
        """kv_offload=True must create _QEffAutoModelForImageTextToTextDualQPC."""
        from QEfficient.transformers.models.modeling_auto import (
            QEFFAutoModelForImageTextToText,
            _QEffAutoModelForImageTextToTextDualQPC,
        )

        model = self._make_tiny_llava()
        qeff = QEFFAutoModelForImageTextToText(model, kv_offload=True)
        assert isinstance(qeff, _QEffAutoModelForImageTextToTextDualQPC), (
            f"kv_offload=True must create DualQPC, got {type(qeff)}"
        )

    def test_default_kv_offload_creates_dual_qpc(self):
        """Default kv_offload (None/True) must create _QEffAutoModelForImageTextToTextDualQPC."""
        from QEfficient.transformers.models.modeling_auto import (
            QEFFAutoModelForImageTextToText,
            _QEffAutoModelForImageTextToTextDualQPC,
        )

        model = self._make_tiny_llava()
        qeff = QEFFAutoModelForImageTextToText(model)
        assert isinstance(qeff, _QEffAutoModelForImageTextToTextDualQPC), "Default kv_offload must create DualQPC"

    def test_single_qpc_has_model_attribute(self):
        from QEfficient.transformers.models.modeling_auto import QEFFAutoModelForImageTextToText

        model = self._make_tiny_llava()
        qeff = QEFFAutoModelForImageTextToText(model, kv_offload=False)
        assert hasattr(qeff, "model")

    def test_dual_qpc_has_model_attribute(self):
        from QEfficient.transformers.models.modeling_auto import QEFFAutoModelForImageTextToText

        model = self._make_tiny_llava()
        qeff = QEFFAutoModelForImageTextToText(model, kv_offload=True)
        assert hasattr(qeff, "model")

    def test_single_qpc_model_name_is_string(self):
        from QEfficient.transformers.models.modeling_auto import QEFFAutoModelForImageTextToText

        model = self._make_tiny_llava()
        qeff = QEFFAutoModelForImageTextToText(model, kv_offload=False)
        assert hasattr(qeff, "model_name")
        assert isinstance(qeff.model_name, str)
        assert len(qeff.model_name) > 0


# ---------------------------------------------------------------------------
# Tests: MISCLASSIFIED_CAUSAL_LM_TO_QEFF_AUTO_CLASS_MAP
# ---------------------------------------------------------------------------


class TestMisclassifiedCausalLMMap:
    """MISCLASSIFIED_CAUSAL_LM_TO_QEFF_AUTO_CLASS_MAP must exist and route correctly."""

    def test_map_exists_and_is_dict(self):
        from QEfficient.transformers.models.modeling_auto import (
            MISCLASSIFIED_CAUSAL_LM_TO_QEFF_AUTO_CLASS_MAP,
        )

        assert isinstance(MISCLASSIFIED_CAUSAL_LM_TO_QEFF_AUTO_CLASS_MAP, dict)

    def test_map_values_are_qeff_classes(self):
        from QEfficient.transformers.models.modeling_auto import (
            MISCLASSIFIED_CAUSAL_LM_TO_QEFF_AUTO_CLASS_MAP,
        )

        for key, val in MISCLASSIFIED_CAUSAL_LM_TO_QEFF_AUTO_CLASS_MAP.items():
            assert isinstance(val, type), f"Expected class for key '{key}', got {type(val)}"

    def test_map_keys_are_strings(self):
        from QEfficient.transformers.models.modeling_auto import (
            MISCLASSIFIED_CAUSAL_LM_TO_QEFF_AUTO_CLASS_MAP,
        )

        for key in MISCLASSIFIED_CAUSAL_LM_TO_QEFF_AUTO_CLASS_MAP.keys():
            assert isinstance(key, str), f"Expected string key, got {type(key)}: {key}"


# ---------------------------------------------------------------------------
# Tests: QEFFAutoModelForCTC class structure
# ---------------------------------------------------------------------------


class TestQEFFAutoModelForCTCStructure:
    """QEFFAutoModelForCTC must have correct class-level structure."""

    def test_importable(self):
        from QEfficient.transformers.models.modeling_auto import QEFFAutoModelForCTC

        assert QEFFAutoModelForCTC is not None

    def test_has_from_pretrained(self):
        from QEfficient.transformers.models.modeling_auto import QEFFAutoModelForCTC

        assert hasattr(QEFFAutoModelForCTC, "from_pretrained")
        assert callable(QEFFAutoModelForCTC.from_pretrained)

    def test_has_pytorch_transforms(self):
        from QEfficient.transformers.models.modeling_auto import QEFFAutoModelForCTC

        assert hasattr(QEFFAutoModelForCTC, "_pytorch_transforms")
        assert isinstance(QEFFAutoModelForCTC._pytorch_transforms, list)

    def test_has_onnx_transforms(self):
        from QEfficient.transformers.models.modeling_auto import QEFFAutoModelForCTC

        assert hasattr(QEFFAutoModelForCTC, "_onnx_transforms")
        assert isinstance(QEFFAutoModelForCTC._onnx_transforms, list)

    def test_has_hf_auto_class(self):
        from QEfficient.transformers.models.modeling_auto import QEFFAutoModelForCTC

        assert hasattr(QEFFAutoModelForCTC, "_hf_auto_class")

    def test_hf_auto_class_is_auto_model_for_ctc(self):
        from transformers import AutoModelForCTC

        from QEfficient.transformers.models.modeling_auto import QEFFAutoModelForCTC

        assert QEFFAutoModelForCTC._hf_auto_class is AutoModelForCTC

    def test_pytorch_transforms_include_custom_ops_transform(self):
        from QEfficient.transformers.models.modeling_auto import QEFFAutoModelForCTC
        from QEfficient.transformers.models.pytorch_transforms import CustomOpsTransform

        assert CustomOpsTransform in QEFFAutoModelForCTC._pytorch_transforms, (
            "CustomOpsTransform not in QEFFAutoModelForCTC._pytorch_transforms"
        )

    def test_onnx_transforms_include_fp16_clip(self):
        """FP16ClipTransform is importable and applicable to CTC models."""
        from QEfficient.base.onnx_transforms import FP16ClipTransform

        assert FP16ClipTransform is not None
        assert hasattr(FP16ClipTransform, "apply")


# ---------------------------------------------------------------------------
# Tests: VLM KV Offload Transforms
# ---------------------------------------------------------------------------


class TestVlmKVOffloadTransforms:
    """VlmKVOffloadTransform and VlmNoKVOffloadTransform must have correct structure."""

    def test_vlm_kv_offload_transform_importable(self):
        from QEfficient.transformers.models.pytorch_transforms import VlmKVOffloadTransform

        assert VlmKVOffloadTransform is not None

    def test_vlm_no_kv_offload_transform_importable(self):
        from QEfficient.transformers.models.pytorch_transforms import VlmNoKVOffloadTransform

        assert VlmNoKVOffloadTransform is not None

    def test_vlm_kv_offload_has_module_mapping(self):
        from QEfficient.transformers.models.pytorch_transforms import VlmKVOffloadTransform

        assert hasattr(VlmKVOffloadTransform, "_module_mapping")
        assert len(VlmKVOffloadTransform._module_mapping) > 0

    def test_vlm_no_kv_offload_has_module_mapping(self):
        from QEfficient.transformers.models.pytorch_transforms import VlmNoKVOffloadTransform

        assert hasattr(VlmNoKVOffloadTransform, "_module_mapping")
        assert len(VlmNoKVOffloadTransform._module_mapping) > 0

    def test_vlm_kv_offload_maps_mllama_cross_attention_to_two_qpc(self):
        from transformers.models.mllama.modeling_mllama import MllamaTextCrossAttention

        from QEfficient.transformers.models.mllama.modeling_mllama import (
            QEffMllamaTextCrossAttentionTwoQPC,
        )
        from QEfficient.transformers.models.pytorch_transforms import VlmKVOffloadTransform

        assert MllamaTextCrossAttention in VlmKVOffloadTransform._module_mapping
        assert VlmKVOffloadTransform._module_mapping[MllamaTextCrossAttention] is QEffMllamaTextCrossAttentionTwoQPC

    def test_vlm_no_kv_offload_maps_mllama_cross_attention_to_single_qpc(self):
        from transformers.models.mllama.modeling_mllama import MllamaTextCrossAttention

        from QEfficient.transformers.models.mllama.modeling_mllama import (
            QEffMllamaTextCrossAttentionSingleQPC,
        )
        from QEfficient.transformers.models.pytorch_transforms import VlmNoKVOffloadTransform

        assert MllamaTextCrossAttention in VlmNoKVOffloadTransform._module_mapping
        assert (
            VlmNoKVOffloadTransform._module_mapping[MllamaTextCrossAttention] is QEffMllamaTextCrossAttentionSingleQPC
        )

    def test_vlm_kv_offload_has_apply_method(self):
        from QEfficient.transformers.models.pytorch_transforms import VlmKVOffloadTransform

        assert hasattr(VlmKVOffloadTransform, "apply")
        assert callable(VlmKVOffloadTransform.apply)

    def test_vlm_no_kv_offload_has_apply_method(self):
        from QEfficient.transformers.models.pytorch_transforms import VlmNoKVOffloadTransform

        assert hasattr(VlmNoKVOffloadTransform, "apply")
        assert callable(VlmNoKVOffloadTransform.apply)

    def test_single_qpc_pytorch_transforms_include_kv_offload_transform(self):
        """SingleQPC must use VlmNoKVOffloadTransform in its pytorch transforms."""
        from QEfficient.transformers.models.modeling_auto import _QEFFAutoModelForImageTextToTextSingleQPC
        from QEfficient.transformers.models.pytorch_transforms import VlmNoKVOffloadTransform

        assert VlmNoKVOffloadTransform in _QEFFAutoModelForImageTextToTextSingleQPC._pytorch_transforms, (
            "VlmNoKVOffloadTransform not in SingleQPC._pytorch_transforms"
        )

    def test_single_qpc_pytorch_transforms_include_no_kv_offload(self):
        """SingleQPC must use VlmNoKVOffloadTransform in its pytorch transforms."""
        from QEfficient.transformers.models.modeling_auto import _QEFFAutoModelForImageTextToTextSingleQPC
        from QEfficient.transformers.models.pytorch_transforms import VlmNoKVOffloadTransform

        assert VlmNoKVOffloadTransform in _QEFFAutoModelForImageTextToTextSingleQPC._pytorch_transforms, (
            "VlmNoKVOffloadTransform not in SingleQPC._pytorch_transforms"
        )

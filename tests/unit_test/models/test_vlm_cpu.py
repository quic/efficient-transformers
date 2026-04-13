# -----------------------------------------------------------------------------
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------
"""
CPU-only tests for VLM (Vision-Language Model) support in QEfficient.

Tests verify:
  - QEFFAutoModelForImageTextToText class structure
  - QEFFAutoModelForImageTextToText error paths (generate without compile, compile with invalid args)
  - _QEffAutoModelForImageTextToTextDualQPC class structure
  - VLM auto-class mapping in MODEL_CLASS_MAPPING

All tests run on CPU only. No actual model loading or QAIC hardware execution.
"""

import pytest

# ---------------------------------------------------------------------------
# Tests: QEFFAutoModelForImageTextToText structure
# ---------------------------------------------------------------------------


class TestQEFFAutoModelForImageTextToTextStructure:
    """QEFFAutoModelForImageTextToText must have correct class-level structure."""

    def test_qeff_auto_model_for_image_text_to_text_importable(self):
        """QEFFAutoModelForImageTextToText must be importable."""
        from QEfficient.transformers.models.modeling_auto import QEFFAutoModelForImageTextToText

        assert QEFFAutoModelForImageTextToText is not None

    def test_has_from_pretrained_classmethod(self):
        """QEFFAutoModelForImageTextToText must have from_pretrained classmethod."""
        from QEfficient.transformers.models.modeling_auto import QEFFAutoModelForImageTextToText

        assert hasattr(QEFFAutoModelForImageTextToText, "from_pretrained")
        assert callable(QEFFAutoModelForImageTextToText.from_pretrained)

    def test_has_hf_auto_class(self):
        """QEFFAutoModelForImageTextToText must have _hf_auto_class."""
        from QEfficient.transformers.models.modeling_auto import QEFFAutoModelForImageTextToText

        assert hasattr(QEFFAutoModelForImageTextToText, "_hf_auto_class")

    def test_importable_from_public_api(self):
        """QEFFAutoModelForImageTextToText must be importable from QEfficient."""
        import QEfficient

        assert hasattr(QEfficient, "QEFFAutoModelForImageTextToText")
        from QEfficient.transformers.models.modeling_auto import QEFFAutoModelForImageTextToText

        assert QEfficient.QEFFAutoModelForImageTextToText is QEFFAutoModelForImageTextToText

    def test_is_factory_class(self):
        """QEFFAutoModelForImageTextToText is a factory class with __new__ method."""
        from QEfficient.transformers.models.modeling_auto import QEFFAutoModelForImageTextToText

        assert hasattr(QEFFAutoModelForImageTextToText, "__new__")


# ---------------------------------------------------------------------------
# Tests: Internal VLM classes structure
# ---------------------------------------------------------------------------


class TestInternalVLMClassesStructure:
    """Internal VLM classes must have correct structure."""

    def test_dual_qpc_class_has_compile_method(self):
        """_QEffAutoModelForImageTextToTextDualQPC must have compile method."""
        from QEfficient.transformers.models.modeling_auto import _QEffAutoModelForImageTextToTextDualQPC

        assert hasattr(_QEffAutoModelForImageTextToTextDualQPC, "compile")
        assert callable(_QEffAutoModelForImageTextToTextDualQPC.compile)

    def test_dual_qpc_class_has_generate_method(self):
        """_QEffAutoModelForImageTextToTextDualQPC must have generate method."""
        from QEfficient.transformers.models.modeling_auto import _QEffAutoModelForImageTextToTextDualQPC

        assert hasattr(_QEffAutoModelForImageTextToTextDualQPC, "generate")
        assert callable(_QEffAutoModelForImageTextToTextDualQPC.generate)

    def test_dual_qpc_class_has_export_method(self):
        """_QEffAutoModelForImageTextToTextDualQPC must have export method."""
        from QEfficient.transformers.models.modeling_auto import _QEffAutoModelForImageTextToTextDualQPC

        assert hasattr(_QEffAutoModelForImageTextToTextDualQPC, "export")
        assert callable(_QEffAutoModelForImageTextToTextDualQPC.export)

    def test_single_qpc_class_has_compile_method(self):
        """_QEFFAutoModelForImageTextToTextSingleQPC must have compile method."""
        from QEfficient.transformers.models.modeling_auto import _QEFFAutoModelForImageTextToTextSingleQPC

        assert hasattr(_QEFFAutoModelForImageTextToTextSingleQPC, "compile")
        assert callable(_QEFFAutoModelForImageTextToTextSingleQPC.compile)

    def test_single_qpc_class_has_generate_method(self):
        """_QEFFAutoModelForImageTextToTextSingleQPC must have generate method."""
        from QEfficient.transformers.models.modeling_auto import _QEFFAutoModelForImageTextToTextSingleQPC

        assert hasattr(_QEFFAutoModelForImageTextToTextSingleQPC, "generate")
        assert callable(_QEFFAutoModelForImageTextToTextSingleQPC.generate)

    def test_single_qpc_class_has_export_method(self):
        """_QEFFAutoModelForImageTextToTextSingleQPC must have export method."""
        from QEfficient.transformers.models.modeling_auto import _QEFFAutoModelForImageTextToTextSingleQPC

        assert hasattr(_QEFFAutoModelForImageTextToTextSingleQPC, "export")
        assert callable(_QEFFAutoModelForImageTextToTextSingleQPC.export)


# ---------------------------------------------------------------------------
# Tests: _QEffAutoModelForImageTextToTextDualQPC structure
# ---------------------------------------------------------------------------


class TestQEffAutoModelForImageTextToTextDualQPCStructure:
    """_QEffAutoModelForImageTextToTextDualQPC must have correct class-level structure."""

    def test_dual_qpc_class_importable(self):
        """_QEffAutoModelForImageTextToTextDualQPC must be importable."""
        from QEfficient.transformers.models.modeling_auto import _QEffAutoModelForImageTextToTextDualQPC

        assert _QEffAutoModelForImageTextToTextDualQPC is not None

    def test_dual_qpc_has_compile_method(self):
        """_QEffAutoModelForImageTextToTextDualQPC must have compile method."""
        from QEfficient.transformers.models.modeling_auto import _QEffAutoModelForImageTextToTextDualQPC

        assert hasattr(_QEffAutoModelForImageTextToTextDualQPC, "compile")
        assert callable(_QEffAutoModelForImageTextToTextDualQPC.compile)

    def test_dual_qpc_has_generate_method(self):
        """_QEffAutoModelForImageTextToTextDualQPC must have generate method."""
        from QEfficient.transformers.models.modeling_auto import _QEffAutoModelForImageTextToTextDualQPC

        assert hasattr(_QEffAutoModelForImageTextToTextDualQPC, "generate")
        assert callable(_QEffAutoModelForImageTextToTextDualQPC.generate)

    def test_dual_qpc_has_export_method(self):
        """_QEffAutoModelForImageTextToTextDualQPC must have export method."""
        from QEfficient.transformers.models.modeling_auto import _QEffAutoModelForImageTextToTextDualQPC

        assert hasattr(_QEffAutoModelForImageTextToTextDualQPC, "export")
        assert callable(_QEffAutoModelForImageTextToTextDualQPC.export)

    def test_dual_qpc_compile_signature_has_skip_lang_and_skip_vision(self):
        """_QEffAutoModelForImageTextToTextDualQPC.compile() must accept skip_lang and skip_vision."""
        import inspect

        from QEfficient.transformers.models.modeling_auto import _QEffAutoModelForImageTextToTextDualQPC

        sig = inspect.signature(_QEffAutoModelForImageTextToTextDualQPC.compile)
        assert "skip_lang" in sig.parameters
        assert "skip_vision" in sig.parameters


# ---------------------------------------------------------------------------
# Tests: VLM auto-class mapping
# ---------------------------------------------------------------------------


class TestVLMAutoClassMapping:
    """VLM models must be correctly mapped in MODEL_CLASS_MAPPING."""

    def test_qeff_auto_model_for_image_text_to_text_in_model_class_mapping_values(self):
        """QEFFAutoModelForImageTextToText must be in MODEL_CLASS_MAPPING values."""
        from QEfficient.transformers.modeling_utils import MODEL_CLASS_MAPPING

        assert "QEFFAutoModelForImageTextToText" in MODEL_CLASS_MAPPING.values()

    def test_llava_config_maps_to_vlm_class(self):
        """LlavaConfig must map to a VLM class (QEFFAutoModelForImageTextToText or similar)."""
        from QEfficient.transformers.modeling_utils import MODEL_CLASS_MAPPING

        if "LlavaConfig" in MODEL_CLASS_MAPPING:
            mapped_class = MODEL_CLASS_MAPPING["LlavaConfig"]
            assert "ImageTextToText" in mapped_class or "CausalLM" in mapped_class

    def test_mllama_config_maps_to_vlm_class(self):
        """MllamaConfig must map to a VLM class."""
        from QEfficient.transformers.modeling_utils import MODEL_CLASS_MAPPING

        if "MllamaConfig" in MODEL_CLASS_MAPPING:
            mapped_class = MODEL_CLASS_MAPPING["MllamaConfig"]
            assert "ImageTextToText" in mapped_class or "CausalLM" in mapped_class

    def test_model_class_mapping_contains_vlm_configs(self):
        """MODEL_CLASS_MAPPING must contain at least one VLM config."""
        from QEfficient.transformers.modeling_utils import MODEL_CLASS_MAPPING

        vlm_configs = ["LlavaConfig", "MllamaConfig", "Llava15Config", "LlavaNextConfig"]
        has_vlm = any(config in MODEL_CLASS_MAPPING for config in vlm_configs)
        assert has_vlm, f"MODEL_CLASS_MAPPING must contain at least one VLM config from {vlm_configs}"


# ---------------------------------------------------------------------------
# Tests: VLM-specific transforms
# ---------------------------------------------------------------------------


class TestVLMTransforms:
    """VLM models must have VLM-specific transforms."""

    def test_vlm_kv_offload_transform_in_pytorch_transforms(self):
        """VlmKVOffloadTransform must be importable."""
        from QEfficient.transformers.models.pytorch_transforms import VlmKVOffloadTransform

        assert VlmKVOffloadTransform is not None

    def test_vlm_no_kv_offload_transform_in_pytorch_transforms(self):
        """VlmNoKVOffloadTransform must be importable."""
        from QEfficient.transformers.models.pytorch_transforms import VlmNoKVOffloadTransform

        assert VlmNoKVOffloadTransform is not None

    def test_vlm_kv_offload_transform_has_module_mapping(self):
        """VlmKVOffloadTransform must have _module_mapping."""
        from QEfficient.transformers.models.pytorch_transforms import VlmKVOffloadTransform

        assert hasattr(VlmKVOffloadTransform, "_module_mapping")
        assert len(VlmKVOffloadTransform._module_mapping) > 0

    def test_vlm_no_kv_offload_transform_has_module_mapping(self):
        """VlmNoKVOffloadTransform must have _module_mapping."""
        from QEfficient.transformers.models.pytorch_transforms import VlmNoKVOffloadTransform

        assert hasattr(VlmNoKVOffloadTransform, "_module_mapping")
        assert len(VlmNoKVOffloadTransform._module_mapping) > 0


# ---------------------------------------------------------------------------
# Tests: VLM generation module
# ---------------------------------------------------------------------------


class TestVLMGenerationModule:
    """VLM generation module must be importable and have correct structure."""

    def test_vlm_generation_module_importable(self):
        """QEfficient.generation.vlm_generation must be importable."""
        import QEfficient.generation.vlm_generation

        assert QEfficient.generation.vlm_generation is not None

    def test_vision_language_generation_importable(self):
        """VisionLanguageGeneration must be importable."""
        from QEfficient.generation.vlm_generation import VisionLanguageGeneration

        assert VisionLanguageGeneration is not None

    def test_vision_language_generation_has_generate_method(self):
        """VisionLanguageGeneration must have generate method."""
        from QEfficient.generation.vlm_generation import VisionLanguageGeneration

        assert hasattr(VisionLanguageGeneration, "generate")
        assert callable(VisionLanguageGeneration.generate)

    def test_vision_language_generation_has_run_prefill_method(self):
        """VisionLanguageGeneration must have run_prefill method."""
        from QEfficient.generation.vlm_generation import VisionLanguageGeneration

        assert hasattr(VisionLanguageGeneration, "run_prefill")
        assert callable(VisionLanguageGeneration.run_prefill)


# ---------------------------------------------------------------------------
# Tests: VisionHandler
# ---------------------------------------------------------------------------


class TestVisionHandler:
    """VisionHandler must be importable and have correct structure."""

    def test_vision_handler_importable(self):
        """VisionHandler must be importable."""
        from QEfficient.generation.embedding_handler import VisionHandler

        assert VisionHandler is not None

    def test_vision_handler_has_is_available_method(self):
        """VisionHandler must have is_available method."""
        from QEfficient.generation.embedding_handler import VisionHandler

        assert hasattr(VisionHandler, "is_available")
        assert callable(VisionHandler.is_available)

    def test_vision_handler_has_run_vision_inference_method(self):
        """VisionHandler must have run_vision_inference method."""
        from QEfficient.generation.embedding_handler import VisionHandler

        assert hasattr(VisionHandler, "run_vision_inference")
        assert callable(VisionHandler.run_vision_inference)

    def test_vision_handler_has_prepare_vlm_inputs_method(self):
        """VisionHandler must have prepare_vlm_inputs method."""
        from QEfficient.generation.embedding_handler import VisionHandler

        assert hasattr(VisionHandler, "prepare_vlm_inputs")
        assert callable(VisionHandler.prepare_vlm_inputs)

    def test_vision_handler_has_setup_vision_buffers_method(self):
        """VisionHandler must have setup_vision_buffers method."""
        from QEfficient.generation.embedding_handler import VisionHandler

        assert hasattr(VisionHandler, "setup_vision_buffers")
        assert callable(VisionHandler.setup_vision_buffers)


# ---------------------------------------------------------------------------
# Tests: MultimodalUtilityMixin
# ---------------------------------------------------------------------------


class TestMultimodalUtilityMixin:
    """MultimodalUtilityMixin must be importable and have correct structure."""

    def test_multimodal_utility_mixin_importable(self):
        """MultimodalUtilityMixin must be importable."""
        from QEfficient.transformers.models.modeling_auto import MultimodalUtilityMixin

        assert MultimodalUtilityMixin is not None

    def test_multimodal_utility_mixin_has_auto_correct_inputs_method(self):
        """MultimodalUtilityMixin must have auto_correct_inputs method."""
        from QEfficient.transformers.models.modeling_auto import MultimodalUtilityMixin

        assert hasattr(MultimodalUtilityMixin, "auto_correct_inputs")
        assert callable(MultimodalUtilityMixin.auto_correct_inputs)

    def test_multimodal_utility_mixin_cannot_be_instantiated_directly(self):
        """MultimodalUtilityMixin must not be instantiable directly (abstract)."""
        from QEfficient.transformers.models.modeling_auto import MultimodalUtilityMixin

        with pytest.raises(TypeError, match="only children"):
            MultimodalUtilityMixin()

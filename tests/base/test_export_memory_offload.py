# -----------------------------------------------------------------------------
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------
import os

import pytest
from transformers import AutoModelForCausalLM

from QEfficient.transformers.models.modeling_auto import QEFFAutoModelForCausalLM

test_models_dict = {"gpt2": "hf-internal-testing/tiny-random-GPT2LMHeadModel"}

test_models = list(test_models_dict.keys())
if os.environ.get("QEFF_TEST_PROFILE", "").strip().lower() == "tiny_model":
    test_models = list(test_models_dict.values())

model_kwargs = {"attn_implementation": "eager"}


@pytest.fixture
def tmp_cache(tmp_path, monkeypatch):
    monkeypatch.setattr("QEfficient.utils.export_utils.QEFF_HOME", tmp_path)
    yield tmp_path


@pytest.mark.non_qaic
@pytest.mark.parametrize("model_name", test_models)
def test_offload_weights_method(model_name):
    """Test the _offload_model_weights method with both True and False values."""
    model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)
    qeff_model = QEFFAutoModelForCausalLM(model, continuous_batching=False)

    # Initially weights should not be offloaded
    assert not qeff_model._is_weights_offloaded
    assert not any(param.is_meta for param in qeff_model.model.parameters())

    # Test with offload_pt_weights=True
    success = qeff_model._offload_model_weights(offload_pt_weights=True)
    assert success
    assert qeff_model._is_weights_offloaded
    assert all(param.is_meta for param in qeff_model.model.parameters())

    # Reset for next test
    model2 = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)
    qeff_model2 = QEFFAutoModelForCausalLM(model2, continuous_batching=False)

    # Test with offload_pt_weights=False
    success = qeff_model2._offload_model_weights(offload_pt_weights=False)
    assert not success
    assert not qeff_model2._is_weights_offloaded
    assert not any(param.is_meta for param in qeff_model2.model.parameters())


@pytest.mark.non_qaic
@pytest.mark.parametrize("model_name", test_models)
def test_re_export_behavior_with_offloaded_weights(model_name, tmp_cache):
    """Test that re-export fails when weights are offloaded."""
    model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)
    qeff_model = QEFFAutoModelForCausalLM(model, continuous_batching=False)

    # First export should succeed
    _ = qeff_model.export()
    assert qeff_model.onnx_path is not None

    # Manually offload weights
    qeff_model._offload_model_weights(offload_pt_weights=True)
    assert qeff_model._is_weights_offloaded

    # Force a new export by removing the file
    import os

    os.remove(qeff_model.onnx_path)
    qeff_model.onnx_path = None

    # Re-export should fail with RuntimeError due to offloaded weights
    with pytest.raises(RuntimeError, match="weights have been offloaded"):
        qeff_model.export()


@pytest.mark.non_qaic
def test_vlm_dual_qpc_memory_offload_behavior():
    """Test asymmetric memory offload behavior for VLM dual QPC models."""

    # Mock vision model (should NOT offload weights)
    class MockVisionModel:
        def __init__(self):
            self._is_weights_offloaded = False

        def export(self, inputs, output_names, dynamic_axes, export_dir=None, offload_pt_weights=True):
            if offload_pt_weights:
                self._is_weights_offloaded = True
            return "vision_export_path"

    # Mock language model (should offload weights)
    class MockLangModel:
        def __init__(self):
            self._is_weights_offloaded = False

        def export(self, inputs, output_names, dynamic_axes, export_dir=None, offload_pt_weights=True):
            if offload_pt_weights:
                self._is_weights_offloaded = True
            return "lang_export_path"

    # Test dual QPC behavior
    vision_model = MockVisionModel()
    lang_model = MockLangModel()

    # Simulate dual QPC export behavior
    vision_model.export({}, [], {}, offload_pt_weights=False)  # Vision model doesn't offload
    lang_model.export({}, [], {}, offload_pt_weights=True)  # Language model offloads

    # Verify asymmetric behavior
    assert not vision_model._is_weights_offloaded  # Vision model should NOT be offloaded
    assert lang_model._is_weights_offloaded  # Language model should be offloaded


@pytest.mark.non_qaic
def test_vlm_single_qpc_memory_offload_behavior():
    """Test memory offload behavior for VLM single QPC models with both True and False."""

    class MockParam:
        def __init__(self, is_meta=False):
            self.is_meta = is_meta

    class MockModel:
        def __init__(self):
            self._params = [MockParam(is_meta=False)]

        def parameters(self):
            return self._params

    class MockSingleQPCModel:
        def __init__(self):
            self._is_weights_offloaded = False
            self.model = MockModel()

        def _offload_model_weights(self):
            self._is_weights_offloaded = True
            for param in self.model.parameters():
                param.is_meta = True
            return True

        def export(self, export_dir=None, offload_pt_weights=True):
            if offload_pt_weights:
                self._offload_model_weights()
            return "single_qpc_export_path"

    # Test with offload_pt_weights=True
    qeff_model = MockSingleQPCModel()
    qeff_model.export(offload_pt_weights=True)
    assert qeff_model._is_weights_offloaded
    assert all(param.is_meta for param in qeff_model.model.parameters())

    # Test with offload_pt_weights=False
    qeff_model2 = MockSingleQPCModel()
    qeff_model2.export(offload_pt_weights=False)
    assert not qeff_model2._is_weights_offloaded
    assert not any(param.is_meta for param in qeff_model2.model.parameters())

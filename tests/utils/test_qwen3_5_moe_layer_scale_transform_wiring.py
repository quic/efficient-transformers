# -----------------------------------------------------------------------------
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------

from QEfficient.transformers.models.modeling_auto import (
    QEFFAutoModelForCausalLM,
    QEffCausalLMForTextImageToTextModel,
    _QEFFAutoModelForImageTextToTextSingleQPC,
)
from QEfficient.transformers.models.pytorch_transforms import (
    KVCacheTransform,
    Qwen3_5MoeLayerScaleMetadataTransform,
)


def _assert_transform_after_kvcache(pipeline):
    kv_idx = pipeline.index(KVCacheTransform)
    scale_idx = pipeline.index(Qwen3_5MoeLayerScaleMetadataTransform)
    assert scale_idx == kv_idx + 1


def test_qwen3_5_moe_scale_transform_wired_for_causal_lm():
    _assert_transform_after_kvcache(QEFFAutoModelForCausalLM._pytorch_transforms)


def test_qwen3_5_moe_scale_transform_wired_for_image_text_dual_lang_path():
    _assert_transform_after_kvcache(QEffCausalLMForTextImageToTextModel._pytorch_transforms)


def test_qwen3_5_moe_scale_transform_wired_for_image_text_single_path():
    _assert_transform_after_kvcache(_QEFFAutoModelForImageTextToTextSingleQPC._pytorch_transforms)

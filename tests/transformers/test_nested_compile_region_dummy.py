# -----------------------------------------------------------------------------
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------

from pathlib import Path
from tempfile import mkdtemp

import numpy as np
import onnxruntime as ort
import torch
from transformers import AutoConfig, AutoModel, AutoModelForImageTextToText, AutoModelForSpeechSeq2Seq

from QEfficient.transformers.models.gemma3.modeling_gemma3 import QEffGemma3EncoderWrapper
from QEfficient.transformers.models.internvl.modeling_internvl import QEffInternEncoderWrapper
from QEfficient.transformers.models.mllama.modeling_mllama import (
    QEffMllamaCrossAttentionDecoderLayer,
    QEffMllamaSelfAttentionDecoderLayer,
    QEffMllamaVisionEncoder,
)
from QEfficient.transformers.models.modeling_auto import (
    QEFFAutoModel,
    QEFFAutoModelForImageTextToText,
    QEFFAutoModelForSpeechSeq2Seq,
)
from QEfficient.transformers.models.molmo.modeling_molmo import QEffMolmoEncoderWrapper, QEffMolmoSequentialBlock
from QEfficient.transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import (
    QEffQwen2_5_VisionTransformerPretrainedModel,
    QEffQwen2_5_VLDecoderLayer,
    QEffQwen_2_5_vl_EncoderWrapper,
)
from QEfficient.transformers.models.whisper.modeling_whisper import QEffWhisperDecoderLayer


def _is_nested_compile_wrapped(bound_method) -> bool:
    wrapped_forward = getattr(bound_method, "__func__", bound_method)
    return getattr(wrapped_forward, "__qualname__", "") == "mark_compile_region.<locals>.wrap.<locals>.inner"


def _build_dummy_qwen2_5_vl():
    config = AutoConfig.for_model("qwen2_5_vl")
    text = config.text_config
    vision = config.vision_config

    text.vocab_size = 128
    text.hidden_size = 64
    text.intermediate_size = 128
    text.num_hidden_layers = 2
    text.num_attention_heads = 4
    text.num_key_value_heads = 2
    text.max_position_embeddings = 64

    vision.depth = 2
    vision.hidden_size = 64
    vision.intermediate_size = 128
    vision.num_heads = 4
    vision.out_hidden_size = 64

    # QEff dummy input helpers read these from root config.
    config.hidden_size = text.hidden_size
    config.num_hidden_layers = text.num_hidden_layers
    config.num_attention_heads = text.num_attention_heads
    config.num_key_value_heads = text.num_key_value_heads
    config.image_token_id = 3

    return AutoModelForImageTextToText.from_config(config).eval()


def _build_dummy_whisper():
    config = AutoConfig.for_model(
        "whisper",
        d_model=64,
        encoder_layers=2,
        decoder_layers=2,
        encoder_attention_heads=4,
        decoder_attention_heads=4,
        encoder_ffn_dim=128,
        decoder_ffn_dim=128,
        max_source_positions=64,
        max_target_positions=32,
        vocab_size=128,
        num_mel_bins=80,
        pad_token_id=0,
        bos_token_id=1,
        eos_token_id=2,
        decoder_start_token_id=1,
    )
    return AutoModelForSpeechSeq2Seq.from_config(config).eval()


def test_dummy_whisper_decoder_layer_has_nested_compile_region():
    model = _build_dummy_whisper()
    qeff_model = QEFFAutoModelForSpeechSeq2Seq(model, pretrained_model_name_or_path="dummy-whisper")
    layer = qeff_model.model.model.decoder.layers[0]
    assert _is_nested_compile_wrapped(layer.forward), "Expected Whisper decoder layer to be wrapped with nested compile"


def test_dummy_qwen2_5_vl_decoder_layer_has_nested_compile_region():
    model = _build_dummy_qwen2_5_vl()
    qeff_model = QEFFAutoModelForImageTextToText(
        model,
        kv_offload=False,
        pretrained_model_name_or_path="dummy-qwen2_5_vl",
    )
    layer = qeff_model.model.language_model.layers[0]
    assert _is_nested_compile_wrapped(layer.forward), "Expected VLM decoder layer to be wrapped with nested compile"


def test_vlm_class_methods_have_nested_compile_region():
    classes = [
        QEffWhisperDecoderLayer,
        QEffQwen2_5_VLDecoderLayer,
        QEffQwen2_5_VisionTransformerPretrainedModel,
        QEffQwen_2_5_vl_EncoderWrapper,
        QEffGemma3EncoderWrapper,
        QEffInternEncoderWrapper,
        QEffMolmoSequentialBlock,
        QEffMolmoEncoderWrapper,
        QEffMllamaSelfAttentionDecoderLayer,
        QEffMllamaCrossAttentionDecoderLayer,
        QEffMllamaVisionEncoder,
    ]

    for cls in classes:
        assert _is_nested_compile_wrapped(cls.forward), f"Expected nested compile wrapper on {cls.__name__}.forward"


def test_dummy_whisper_dynamo_export_smoke(tmp_path):
    model = _build_dummy_whisper()
    qeff_model = QEFFAutoModelForSpeechSeq2Seq(model, pretrained_model_name_or_path="dummy-whisper")
    onnx_path = qeff_model.export(
        export_dir=tmp_path,
        use_dynamo=True,
        use_onnx_subfunctions=False,
    )
    assert Path(onnx_path).is_file()


def test_dummy_embedding_hf_vs_qeff_pt_vs_ort():
    config = AutoConfig.for_model(
        "bert",
        vocab_size=128,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        max_position_embeddings=32,
    )

    hf_model = AutoModel.from_config(config).eval()
    qeff_model = QEFFAutoModel(hf_model, pretrained_model_name_or_path="dummy-bert")

    inputs = {
        "input_ids": torch.tensor([[5, 6, 7, 0]], dtype=torch.int64),
        "attention_mask": torch.tensor([[1, 1, 1, 0]], dtype=torch.int64),
    }

    hf_output = hf_model(**inputs).last_hidden_state.detach().cpu().numpy()
    qeff_pt_output = qeff_model.generate(inputs=inputs, runtime_ai100=False)[0].detach().cpu().numpy()

    onnx_path = qeff_model.export(
        export_dir=Path(mkdtemp(prefix="qeff_embed_dummy_")),
        use_dynamo=True,
    )
    ort_session = ort.InferenceSession(str(onnx_path))
    ort_output = ort_session.run(
        None,
        {"input_ids": inputs["input_ids"].numpy(), "attention_mask": inputs["attention_mask"].numpy()},
    )[0]

    assert np.allclose(hf_output, qeff_pt_output, atol=1e-6), "HF and QEff PyTorch outputs do not match"
    assert np.allclose(hf_output, ort_output, atol=1e-5), "HF PyTorch and ORT outputs do not match"

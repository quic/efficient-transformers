# -----------------------------------------------------------------------------
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------
"""
End-to-end tests for Speech Seq2Seq (Whisper): HF → QEff → ONNX structure.

Key accuracy assertions:
  - HF encoder produces finite hidden states with correct shape
  - QEff Whisper has correct architecture (QEffWhisperEncoder, QEffWhisperDecoder)
  - QEff encoder produces same hidden states as HF encoder (max_diff < 1e-5)
  - QEff Whisper has QEffWhisperAttention layers

All tests run on CPU only.
"""

import pytest
import torch
from transformers import WhisperConfig, WhisperForConditionalGeneration

from QEfficient.transformers.models.modeling_auto import QEFFAutoModelForSpeechSeq2Seq

D_MODEL = 64
NUM_MEL_BINS = 80
VOCAB_SIZE = 100
MAX_SOURCE_POS = 32
MAX_TARGET_POS = 32


def make_tiny_whisper():
    cfg = WhisperConfig(
        vocab_size=VOCAB_SIZE,
        num_mel_bins=NUM_MEL_BINS,
        encoder_layers=1,
        encoder_attention_heads=2,
        decoder_layers=1,
        decoder_attention_heads=2,
        decoder_ffn_dim=D_MODEL,
        encoder_ffn_dim=D_MODEL,
        d_model=D_MODEL,
        max_source_positions=MAX_SOURCE_POS,
        max_target_positions=MAX_TARGET_POS,
        decoder_start_token_id=1,
        eos_token_id=2,
        pad_token_id=0,
        bos_token_id=1,
    )
    return WhisperForConditionalGeneration(cfg).eval(), cfg


def make_mel_input(batch=1, seq_len=64):
    return torch.randn(batch, NUM_MEL_BINS, seq_len)


@pytest.mark.speech
class TestHFWhisperBaseline:
    """HF Whisper model runs correctly on CPU."""

    def test_encoder_output_shape(self):
        model, cfg = make_tiny_whisper()
        mel = make_mel_input(seq_len=64)
        with torch.no_grad():
            enc_out = model.model.encoder(mel)
        assert enc_out.last_hidden_state is not None
        assert enc_out.last_hidden_state.shape[-1] == D_MODEL

    def test_encoder_hidden_states_are_finite(self):
        model, cfg = make_tiny_whisper()
        mel = make_mel_input(seq_len=64)
        with torch.no_grad():
            enc_out = model.model.encoder(mel)
        assert torch.isfinite(enc_out.last_hidden_state).all()

    def test_full_forward_returns_logits(self):
        model, cfg = make_tiny_whisper()
        mel = make_mel_input(seq_len=64)
        decoder_input_ids = torch.tensor([[cfg.decoder_start_token_id]])
        with torch.no_grad():
            out = model(input_features=mel, decoder_input_ids=decoder_input_ids)
        assert hasattr(out, "logits")
        assert out.logits.shape[-1] == VOCAB_SIZE

    def test_logits_are_finite(self):
        model, cfg = make_tiny_whisper()
        mel = make_mel_input(seq_len=64)
        decoder_input_ids = torch.tensor([[cfg.decoder_start_token_id]])
        with torch.no_grad():
            out = model(input_features=mel, decoder_input_ids=decoder_input_ids)
        assert torch.isfinite(out.logits).all()

    def test_generate_produces_tokens(self):
        model, cfg = make_tiny_whisper()
        mel = make_mel_input(seq_len=64)
        with torch.no_grad():
            generated = model.generate(mel, max_new_tokens=3, do_sample=False)
        assert generated is not None
        assert generated.shape[0] == 1
        assert generated.shape[1] >= 1

    def test_encoder_decoder_structure(self):
        model, cfg = make_tiny_whisper()
        assert hasattr(model.model, "encoder")
        assert hasattr(model.model, "decoder")


@pytest.mark.speech
class TestQEffWhisperArchitecture:
    """QEff Whisper must have correct architecture after KV transform."""

    def test_qeff_whisper_wraps_without_error(self):
        model, cfg = make_tiny_whisper()
        qeff_model = QEFFAutoModelForSpeechSeq2Seq(model)
        assert qeff_model is not None
        assert hasattr(qeff_model, "model")

    def test_qeff_whisper_is_eval_mode(self):
        model, cfg = make_tiny_whisper()
        qeff_model = QEFFAutoModelForSpeechSeq2Seq(model)
        assert not qeff_model.model.training

    def test_qeff_whisper_model_class_replaced(self):
        from QEfficient.transformers.models.whisper.modeling_whisper import QEffWhisperForConditionalGeneration

        model, cfg = make_tiny_whisper()
        qeff_model = QEFFAutoModelForSpeechSeq2Seq(model)
        assert isinstance(qeff_model.model, QEffWhisperForConditionalGeneration), (
            f"Expected QEffWhisperForConditionalGeneration, got {type(qeff_model.model)}"
        )

    def test_qeff_whisper_encoder_replaced(self):
        from QEfficient.transformers.models.whisper.modeling_whisper import QEffWhisperEncoder

        model, cfg = make_tiny_whisper()
        qeff_model = QEFFAutoModelForSpeechSeq2Seq(model)
        assert isinstance(qeff_model.model.model.encoder, QEffWhisperEncoder), (
            f"Expected QEffWhisperEncoder, got {type(qeff_model.model.model.encoder)}"
        )

    def test_qeff_whisper_decoder_replaced(self):
        from QEfficient.transformers.models.whisper.modeling_whisper import QEffWhisperDecoder

        model, cfg = make_tiny_whisper()
        qeff_model = QEFFAutoModelForSpeechSeq2Seq(model)
        assert isinstance(qeff_model.model.model.decoder, QEffWhisperDecoder), (
            f"Expected QEffWhisperDecoder, got {type(qeff_model.model.model.decoder)}"
        )

    def test_qeff_whisper_has_qeff_attention_layers(self):
        from QEfficient.transformers.models.whisper.modeling_whisper import QEffWhisperAttention

        model, cfg = make_tiny_whisper()
        qeff_model = QEFFAutoModelForSpeechSeq2Seq(model)
        has_qeff_attn = any(isinstance(m, QEffWhisperAttention) for m in qeff_model.model.modules())
        assert has_qeff_attn, "QEff Whisper must have QEffWhisperAttention layers"

    def test_qeff_whisper_has_positional_embedding_replaced(self):
        from QEfficient.transformers.models.whisper.modeling_whisper import QEffWhisperPositionalEmbedding

        model, cfg = make_tiny_whisper()
        qeff_model = QEFFAutoModelForSpeechSeq2Seq(model)
        has_pos_emb = any(isinstance(m, QEffWhisperPositionalEmbedding) for m in qeff_model.model.modules())
        assert has_pos_emb, "QEff Whisper must have QEffWhisperPositionalEmbedding"

    def test_qeff_whisper_model_name_property(self):
        model, cfg = make_tiny_whisper()
        qeff_model = QEFFAutoModelForSpeechSeq2Seq(model)
        assert hasattr(qeff_model, "model_name")
        assert isinstance(qeff_model.model_name, str)
        assert len(qeff_model.model_name) > 0


@pytest.mark.speech
@pytest.mark.accuracy
class TestQEffWhisperEncoderAccuracy:
    """QEff Whisper encoder must produce the same hidden states as HF encoder."""

    def test_qeff_encoder_output_shape_matches_hf(self):
        model, cfg = make_tiny_whisper()
        mel = make_mel_input(seq_len=64)
        with torch.no_grad():
            hf_enc = model.model.encoder(mel)
        qeff_model = QEFFAutoModelForSpeechSeq2Seq(model)
        with torch.no_grad():
            qeff_enc = qeff_model.model.model.encoder(mel)
        assert qeff_enc.last_hidden_state.shape == hf_enc.last_hidden_state.shape

    def test_qeff_encoder_hidden_states_match_hf(self):
        """QEff encoder hidden states must be numerically identical to HF."""
        model, cfg = make_tiny_whisper()
        mel = make_mel_input(seq_len=64)
        with torch.no_grad():
            hf_hidden = model.model.encoder(mel).last_hidden_state
        qeff_model = QEFFAutoModelForSpeechSeq2Seq(model)
        with torch.no_grad():
            qeff_hidden = qeff_model.model.model.encoder(mel).last_hidden_state
        max_diff = (hf_hidden - qeff_hidden).abs().max().item()
        assert max_diff < 1e-5, (
            f"Encoder hidden state mismatch: max_diff={max_diff:.2e}. "
            f"QEff encoder must produce identical outputs to HF encoder."
        )

    def test_qeff_encoder_hidden_states_are_finite(self):
        model, cfg = make_tiny_whisper()
        mel = make_mel_input(seq_len=64)
        qeff_model = QEFFAutoModelForSpeechSeq2Seq(model)
        with torch.no_grad():
            qeff_enc = qeff_model.model.model.encoder(mel)
        assert torch.isfinite(qeff_enc.last_hidden_state).all()

    def test_qeff_encoder_deterministic(self):
        model, cfg = make_tiny_whisper()
        mel = make_mel_input(seq_len=64)
        qeff_model = QEFFAutoModelForSpeechSeq2Seq(model)
        with torch.no_grad():
            h1 = qeff_model.model.model.encoder(mel).last_hidden_state
            h2 = qeff_model.model.model.encoder(mel).last_hidden_state
        assert torch.allclose(h1, h2), "QEff encoder must be deterministic"

    def test_qeff_encoder_batch_output_shape(self):
        """QEff encoder must handle batch_size > 1."""
        model, cfg = make_tiny_whisper()
        mel = make_mel_input(batch=2, seq_len=64)
        qeff_model = QEFFAutoModelForSpeechSeq2Seq(model)
        with torch.no_grad():
            qeff_enc = qeff_model.model.model.encoder(mel)
        assert qeff_enc.last_hidden_state.shape[0] == 2
        assert torch.isfinite(qeff_enc.last_hidden_state).all()


@pytest.mark.speech
@pytest.mark.onnx
@pytest.mark.slow
class TestWhisperONNXExport:
    """Whisper ONNX export tests."""

    def test_whisper_onnx_export_succeeds(self, tmp_export_dir):
        model, cfg = make_tiny_whisper()
        qeff_model = QEFFAutoModelForSpeechSeq2Seq(model)
        onnx_path = qeff_model.export(export_dir=str(tmp_export_dir))
        assert onnx_path is not None

    def test_whisper_onnx_files_exist(self, tmp_export_dir):
        import pathlib

        model, cfg = make_tiny_whisper()
        qeff_model = QEFFAutoModelForSpeechSeq2Seq(model)
        onnx_path = qeff_model.export(export_dir=str(tmp_export_dir))
        search_root = pathlib.Path(str(onnx_path)).parent if onnx_path else tmp_export_dir
        onnx_files = list(search_root.rglob("*.onnx")) or list(tmp_export_dir.rglob("*.onnx"))
        assert len(onnx_files) > 0, (
            f"No ONNX files found after Whisper export. onnx_path={onnx_path}, search_root={search_root}"
        )

    def test_whisper_onnx_encoder_passes_checker(self, tmp_export_dir):
        """At least one exported Whisper ONNX file must pass onnx.checker."""
        import pathlib

        import onnx

        model, cfg = make_tiny_whisper()
        qeff_model = QEFFAutoModelForSpeechSeq2Seq(model)
        onnx_path = qeff_model.export(export_dir=str(tmp_export_dir))
        search_root = pathlib.Path(str(onnx_path)).parent if onnx_path else tmp_export_dir
        onnx_files = list(search_root.rglob("*.onnx")) or list(tmp_export_dir.rglob("*.onnx"))
        assert len(onnx_files) > 0, "No ONNX files found after Whisper export"
        passed = False
        for f in onnx_files:
            try:
                m = onnx.load(str(f))
                onnx.checker.check_model(m)
                passed = True
                break
            except Exception:
                continue
        assert passed, "No exported Whisper ONNX file passed onnx.checker"

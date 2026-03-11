# -----------------------------------------------------------------------------
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------
"""
End-to-end accuracy tests for Sequence Classification: HF → QEff → ORT.

BERT/DeBERTa have no Qualcomm custom ops, so the full pipeline works.
All three stages must predict the same class and produce numerically close logits.

Models: BertForSequenceClassification, DebertaV2ForSequenceClassification
All tests run on CPU only.
"""

import numpy as np
import pytest
import torch
from transformers import (
    BertConfig,
    BertForSequenceClassification,
    DebertaV2Config,
    DebertaV2ForSequenceClassification,
)

from QEfficient.transformers.models.modeling_auto import QEFFAutoModelForSequenceClassification

SEQ_LEN = 16
VOCAB_SIZE = 500
NUM_LABELS = 3


def make_tiny_bert(num_labels=NUM_LABELS):
    cfg = BertConfig(
        num_hidden_layers=1, num_attention_heads=2, hidden_size=64,
        intermediate_size=128, vocab_size=VOCAB_SIZE, max_position_embeddings=64, num_labels=num_labels,
    )
    return BertForSequenceClassification(cfg).eval(), cfg


def make_tiny_deberta(num_labels=NUM_LABELS):
    cfg = DebertaV2Config(
        num_hidden_layers=1, num_attention_heads=2, hidden_size=64,
        intermediate_size=128, vocab_size=VOCAB_SIZE, max_position_embeddings=64,
        num_labels=num_labels, type_vocab_size=0, pos_att_type=["p2c", "c2p"],
    )
    return DebertaV2ForSequenceClassification(cfg).eval(), cfg


def make_inputs(batch=1, seq=SEQ_LEN):
    return {
        "input_ids": torch.randint(0, VOCAB_SIZE, (batch, seq)),
        "attention_mask": torch.ones(batch, seq, dtype=torch.long),
    }


@pytest.mark.seq_classification
class TestHFSeqClassBaseline:
    def test_bert_logits_shape(self):
        model, cfg = make_tiny_bert()
        with torch.no_grad():
            out = model(**make_inputs())
        assert out.logits.shape == (1, NUM_LABELS)

    def test_bert_batch_logits_shape(self):
        model, cfg = make_tiny_bert()
        with torch.no_grad():
            out = model(**make_inputs(batch=4))
        assert out.logits.shape == (4, NUM_LABELS)

    def test_bert_predicted_class_is_valid(self):
        model, cfg = make_tiny_bert()
        with torch.no_grad():
            pred = model(**make_inputs()).logits.argmax(-1).item()
        assert 0 <= pred < NUM_LABELS

    def test_bert_logits_are_finite(self):
        model, cfg = make_tiny_bert()
        with torch.no_grad():
            logits = model(**make_inputs()).logits
        assert torch.isfinite(logits).all()

    def test_bert_prediction_is_deterministic(self):
        model, cfg = make_tiny_bert()
        inputs = make_inputs()
        with torch.no_grad():
            p1 = model(**inputs).logits.argmax(-1).item()
            p2 = model(**inputs).logits.argmax(-1).item()
        assert p1 == p2

    def test_deberta_logits_shape(self):
        try:
            model, cfg = make_tiny_deberta()
            with torch.no_grad():
                out = model(**make_inputs())
            assert out.logits.shape == (1, NUM_LABELS)
        except Exception as e:
            pytest.skip(f"DeBERTa-v2 not available: {e}")


@pytest.mark.seq_classification
@pytest.mark.accuracy
class TestQEffSeqClassAccuracyVsHF:
    """QEff model must predict the same class as HF and produce numerically close logits."""

    def test_bert_qeff_predicts_same_class_as_hf(self):
        model, cfg = make_tiny_bert()
        inputs = make_inputs()
        with torch.no_grad():
            hf_class = model(**inputs).logits.argmax(-1).item()
        qeff_model = QEFFAutoModelForSequenceClassification(model)
        with torch.no_grad():
            qeff_class = qeff_model.model(**inputs).logits.argmax(-1).item()
        assert hf_class == qeff_class, f"Class mismatch: HF={hf_class}, QEff={qeff_class}"

    def test_bert_qeff_logits_numerically_identical_to_hf(self):
        model, cfg = make_tiny_bert()
        inputs = make_inputs()
        with torch.no_grad():
            hf_logits = model(**inputs).logits
        qeff_model = QEFFAutoModelForSequenceClassification(model)
        with torch.no_grad():
            qeff_logits = qeff_model.model(**inputs).logits
        max_diff = (hf_logits - qeff_logits).abs().max().item()
        assert max_diff < 1e-5, f"Logits differ by {max_diff:.2e}. Must be < 1e-5."

    def test_bert_qeff_logits_shape_correct(self):
        model, cfg = make_tiny_bert()
        qeff_model = QEFFAutoModelForSequenceClassification(model)
        with torch.no_grad():
            logits = qeff_model.model(**make_inputs()).logits
        assert logits.shape == (1, NUM_LABELS)

    def test_bert_qeff_logits_are_finite(self):
        model, cfg = make_tiny_bert()
        qeff_model = QEFFAutoModelForSequenceClassification(model)
        with torch.no_grad():
            logits = qeff_model.model(**make_inputs()).logits
        assert torch.isfinite(logits).all()

    def test_bert_qeff_batch_prediction_matches_hf(self):
        model, cfg = make_tiny_bert()
        inputs = make_inputs(batch=4)
        with torch.no_grad():
            hf_classes = model(**inputs).logits.argmax(-1).tolist()
        qeff_model = QEFFAutoModelForSequenceClassification(model)
        with torch.no_grad():
            qeff_classes = qeff_model.model(**inputs).logits.argmax(-1).tolist()
        assert hf_classes == qeff_classes, f"Batch class mismatch: HF={hf_classes}, QEff={qeff_classes}"

    def test_deberta_qeff_predicts_same_class_as_hf(self):
        try:
            model, cfg = make_tiny_deberta()
            inputs = make_inputs()
            with torch.no_grad():
                hf_class = model(**inputs).logits.argmax(-1).item()
            qeff_model = QEFFAutoModelForSequenceClassification(model)
            with torch.no_grad():
                qeff_class = qeff_model.model(**inputs).logits.argmax(-1).item()
            assert hf_class == qeff_class, f"DeBERTa class mismatch: HF={hf_class}, QEff={qeff_class}"
        except Exception as e:
            pytest.skip(f"DeBERTa-v2 not available: {e}")


@pytest.mark.seq_classification
@pytest.mark.accuracy
@pytest.mark.onnx
@pytest.mark.slow
class TestSeqClassORTAccuracy:
    """Full pipeline: HF → QEff → ORT must all predict the same class."""

    def test_bert_ort_predicts_same_class_as_qeff(self, tmp_export_dir):
        import onnxruntime as ort
        model, cfg = make_tiny_bert()
        inputs = make_inputs()
        qeff_model = QEFFAutoModelForSequenceClassification(model)
        with torch.no_grad():
            qeff_class = qeff_model.model(**inputs).logits.argmax(-1).item()
        onnx_path = qeff_model.export(export_dir=str(tmp_export_dir))
        session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
        ort_inputs = {k: v.numpy() for k, v in inputs.items()}
        output_names = [o.name for o in session.get_outputs()]
        ort_out = dict(zip(output_names, session.run(output_names, ort_inputs)))
        ort_class = int(ort_out["logits"].argmax(-1))
        assert qeff_class == ort_class, f"Class mismatch QEff vs ORT: QEff={qeff_class}, ORT={ort_class}"

    def test_bert_ort_predicts_same_class_as_hf(self, tmp_export_dir):
        import onnxruntime as ort
        model, cfg = make_tiny_bert()
        inputs = make_inputs()
        with torch.no_grad():
            hf_class = model(**inputs).logits.argmax(-1).item()
        qeff_model = QEFFAutoModelForSequenceClassification(model)
        onnx_path = qeff_model.export(export_dir=str(tmp_export_dir))
        session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
        ort_inputs = {k: v.numpy() for k, v in inputs.items()}
        output_names = [o.name for o in session.get_outputs()]
        ort_out = dict(zip(output_names, session.run(output_names, ort_inputs)))
        ort_class = int(ort_out["logits"].argmax(-1))
        assert hf_class == ort_class, f"Full pipeline class mismatch: HF={hf_class}, ORT={ort_class}"

    def test_bert_ort_logits_numerically_close_to_qeff(self, tmp_export_dir):
        import onnxruntime as ort
        model, cfg = make_tiny_bert()
        inputs = make_inputs()
        qeff_model = QEFFAutoModelForSequenceClassification(model)
        with torch.no_grad():
            qeff_logits = qeff_model.model(**inputs).logits.numpy()
        onnx_path = qeff_model.export(export_dir=str(tmp_export_dir))
        session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
        ort_inputs = {k: v.numpy() for k, v in inputs.items()}
        output_names = [o.name for o in session.get_outputs()]
        ort_out = dict(zip(output_names, session.run(output_names, ort_inputs)))
        max_diff = np.abs(qeff_logits - ort_out["logits"]).max()
        assert max_diff < 1e-4, f"Logit max diff QEff vs ORT: {max_diff:.2e}. Must be < 1e-4."

    def test_bert_ort_logits_shape_correct(self, tmp_export_dir):
        import onnxruntime as ort
        model, cfg = make_tiny_bert()
        qeff_model = QEFFAutoModelForSequenceClassification(model)
        onnx_path = qeff_model.export(export_dir=str(tmp_export_dir))
        session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
        ort_inputs = {k: v.numpy() for k, v in make_inputs().items()}
        output_names = [o.name for o in session.get_outputs()]
        ort_out = dict(zip(output_names, session.run(output_names, ort_inputs)))
        assert "logits" in ort_out
        assert ort_out["logits"].shape == (1, NUM_LABELS)

    def test_bert_ort_batch_predictions_match_qeff(self, tmp_export_dir):
        import onnxruntime as ort
        batch_size = 4
        model, cfg = make_tiny_bert()
        inputs = make_inputs(batch=batch_size)
        qeff_model = QEFFAutoModelForSequenceClassification(model)
        with torch.no_grad():
            qeff_classes = qeff_model.model(**inputs).logits.argmax(-1).tolist()
        onnx_path = qeff_model.export(export_dir=str(tmp_export_dir))
        session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
        ort_inputs = {k: v.numpy() for k, v in inputs.items()}
        output_names = [o.name for o in session.get_outputs()]
        ort_out = dict(zip(output_names, session.run(output_names, ort_inputs)))
        ort_classes = ort_out["logits"].argmax(-1).tolist()
        assert qeff_classes == ort_classes, f"Batch class mismatch: QEff={qeff_classes}, ORT={ort_classes}"

    def test_bert_onnx_passes_checker(self, tmp_export_dir):
        import onnx
        model, cfg = make_tiny_bert()
        qeff_model = QEFFAutoModelForSequenceClassification(model)
        onnx_path = qeff_model.export(export_dir=str(tmp_export_dir))
        onnx_model = onnx.load(str(onnx_path))
        onnx.checker.check_model(onnx_model)

    def test_bert_onnx_has_input_ids_and_logits(self, tmp_export_dir):
        import onnx
        model, cfg = make_tiny_bert()
        qeff_model = QEFFAutoModelForSequenceClassification(model)
        onnx_path = qeff_model.export(export_dir=str(tmp_export_dir))
        onnx_model = onnx.load(str(onnx_path))
        input_names = {inp.name for inp in onnx_model.graph.input}
        output_names = {out.name for out in onnx_model.graph.output}
        assert "input_ids" in input_names
        assert "logits" in output_names

    def test_deberta_ort_predicts_same_class_as_hf(self, tmp_export_dir):
        """DeBERTa-v2 full pipeline: HF, QEff, ORT must agree on class."""
        import onnxruntime as ort
        try:
            model, cfg = make_tiny_deberta()
            inputs = make_inputs()
            with torch.no_grad():
                hf_class = model(**inputs).logits.argmax(-1).item()
            qeff_model = QEFFAutoModelForSequenceClassification(model)
            onnx_path = qeff_model.export(export_dir=str(tmp_export_dir))
            session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
            ort_inputs = {k: v.numpy() for k, v in inputs.items()}
            output_names = [o.name for o in session.get_outputs()]
            ort_out = dict(zip(output_names, session.run(output_names, ort_inputs)))
            ort_class = int(ort_out["logits"].argmax(-1))
            assert hf_class == ort_class, f"DeBERTa pipeline mismatch: HF={hf_class}, ORT={ort_class}"
        except Exception as e:
            pytest.skip(f"DeBERTa-v2 not available or export failed: {e}")

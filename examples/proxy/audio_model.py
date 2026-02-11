# -----------------------------------------------------------------------------
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
#
# ----------------------------------------------------------------------------

"""
Simple example: How to enable proxy models for audio processing and generate IO files.
Demonstrates two model types: Speech-to-Seq2Seq (Whisper) and CTC (Wav2Vec2).
"""

from datasets import load_dataset
from transformers import AutoProcessor

from QEfficient import QEFFAutoModelForCTC, QEFFAutoModelForSpeechSeq2Seq

# Load audio sample
print("Loading audio sample...")
dataset = load_dataset("hf-internal-testing/librispeech_asr_dummy", "clean", split="validation")
audio_data = dataset[0]["audio"]["array"]
sample_rate = dataset[0]["audio"]["sampling_rate"]

# ===================================================================
# ============ Model Type 1: Speech-to-Seq2Seq (Whisper) ============
# ===================================================================

print("\n" + "=" * 70)
print("MODEL 1: WHISPER (Speech-to-Seq2Seq)")
print("=" * 70)

model_name_seq2seq = "openai/whisper-tiny"
processor_seq2seq = AutoProcessor.from_pretrained(model_name_seq2seq)

# Load proxy model
model_seq2seq = QEFFAutoModelForSpeechSeq2Seq.from_pretrained(model_name_seq2seq, enable_proxy=True)
print(model_seq2seq)

# Compile model
model_seq2seq.compile(num_cores=16)

# Process audio and generate
inputs = processor_seq2seq(audio_data, sampling_rate=sample_rate, return_tensors="pt")
result = model_seq2seq.generate(inputs=inputs, generation_len=25, write_io=True)
transcription = processor_seq2seq.batch_decode(result.generated_ids)[0]
print(f"Transcription: {transcription}\n")


# ===================================================================
# ============ Model Type 2: CTC (Wav2Vec2) ============
# ===================================================================

print("=" * 70)
print("MODEL 2: WAV2VEC2 (CTC)")
print("=" * 70)

model_name_ctc = "facebook/wav2vec2-base"
processor_ctc = AutoProcessor.from_pretrained(model_name_ctc)

# Load proxy model
model_ctc = QEFFAutoModelForCTC.from_pretrained(model_name_ctc, enable_proxy=True)
print(model_ctc)
# Compile model
model_ctc.compile(num_cores=16)

# Generate with IO files
transcription = model_ctc.generate(processor_ctc, inputs=audio_data, write_io=True)
print(f"Transcription: {transcription}\n")

# -----------------------------------------------------------------------------
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------

from typing import List

from datasets import load_dataset
from transformers import AutoProcessor

from QEfficient import QEFFAutoModelForCTC, QEFFAutoModelForSpeechSeq2Seq


def load_audio_sample(
    dataset_id: str = "hf-internal-testing/librispeech_asr_dummy",
    config: str = "clean",
    split: str = "validation",
    sample_idx: int = 0,
):
    """Load audio sample from a HuggingFace dataset."""
    print(f"Loading audio sample from dataset: {dataset_id}")
    ds = load_dataset(dataset_id, config, split=split)
    audio_data = ds[sample_idx]["audio"]["array"]
    sample_rate = ds[sample_idx]["audio"]["sampling_rate"]

    # Reshape to batch size 1
    audio_data = audio_data.reshape(-1)

    return audio_data, sample_rate


def infer_speech_seq2seq(
    model, processor, audio_data: list, sample_rate: int, generation_len: int = 25, num_cores: int = 16
):
    """
    Inference for Speech-to-Seq2Seq models (e.g., Whisper).

    Returns:
        str: Transcribed text
    """
    print("Running Speech-to-Seq2Seq inference...")

    # Compile the model
    model.compile(num_cores=num_cores)

    # Process audio
    inputs = processor(audio_data, sampling_rate=sample_rate, return_tensors="pt")

    # Generate output
    exec_info = model.generate(inputs=inputs, generation_len=generation_len)

    # Decode output
    transcription = processor.batch_decode(exec_info.generated_ids)[0]

    return transcription


def infer_ctc(model, processor, audio_data: list, num_cores: int = 16):
    """
    Inference for CTC models (e.g., Wav2Vec2).

    Returns:
        str: Transcribed text
    """
    print("Running CTC inference...")

    # Compile the model
    model.compile(num_cores=num_cores)

    # Generate output
    transcription = model.generate(processor, inputs=audio_data)

    return transcription


def detect_model_type(model_name: str) -> str:
    """
    Auto-detect model type based on model name.

    Args:
        model_name: HuggingFace model ID

    Returns:
        str: 'seq2seq' for Whisper models or 'ctc' for Wav2Vec2 models
    """
    model_name_lower = model_name.lower()

    if "whisper" in model_name_lower:
        return "seq2seq"
    elif "wav2vec" in model_name_lower:
        return "ctc"
    else:
        raise ValueError(
            f"Cannot auto-detect model type for {model_name}. "
            "Please use models containing 'whisper' or 'wav2vec' in their names."
        )


def run_inference_on_models(
    models: List[str],
    dataset_id: str = "hf-internal-testing/librispeech_asr_dummy",
    dataset_config: str = "clean",
    dataset_split: str = "validation",
    sample_idx: int = 0,
    generation_len: int = 25,
    num_cores: int = 16,
):
    """
    Run inference on a list of audio models.

    Args:
        models: List of HuggingFace model IDs
        dataset_id: Dataset ID for audio samples
        dataset_config: Dataset configuration
        dataset_split: Dataset split
        sample_idx: Index of audio sample to use
        generation_len: Context length for generation (seq2seq models only)
        num_cores: Number of cores for compilation
    """
    # Load audio data once
    print("Loading audio sample...")
    audio_data, sample_rate = load_audio_sample(
        dataset_id=dataset_id,
        config=dataset_config,
        split=dataset_split,
        sample_idx=sample_idx,
    )

    results = {}

    for model_name in models:
        try:
            print(f"\n{'=' * 60}")
            print(f"Processing: {model_name}")
            print(f"{'=' * 60}")

            # Auto-detect model type
            model_type = detect_model_type(model_name)
            print(f"Detected model type: {model_type.upper()}")

            # Load processor
            processor = AutoProcessor.from_pretrained(model_name)

            # Load and run appropriate model
            if model_type == "seq2seq":
                model = QEFFAutoModelForSpeechSeq2Seq.from_pretrained(model_name)
                print("\n============Original Qeff Model===============\n")
                print(model)
                print("\n=====================================\n")
                model = QEFFAutoModelForSpeechSeq2Seq.from_pretrained(model_name, enable_proxy=True)
                print("\n============Proxy Qeff Model===============\n")
                print(model)
                print("\n=====================================\n")

                transcription = infer_speech_seq2seq(
                    model, processor, audio_data, sample_rate, generation_len=generation_len, num_cores=num_cores
                )
            else:  # ctc
                model = QEFFAutoModelForSpeechSeq2Seq.from_pretrained(model_name)
                print("\n============Original Qeff Model===============\n")
                print(model)
                print("\n=====================================\n")
                model = QEFFAutoModelForCTC.from_pretrained(model_name, enable_proxy=True)
                print("\n============Proxy Qeff Model===============\n")
                print(model)
                print("\n=====================================\n")
                transcription = infer_ctc(model, processor, audio_data, num_cores=num_cores)

            print(f"Transcription: {transcription}")
            results[model_name] = {"status": "success", "transcription": transcription}

        except Exception as e:
            print(f"Error processing {model_name}: {str(e)}")
            results[model_name] = {"status": "failed", "error": str(e)}

    return results


models = [
    "openai/whisper-tiny",
    # "openai/whisper-base",
    # "openai/whisper-small",
    # "openai/whisper-medium",
    # "openai/whisper-large",
    # "openai/whisper-large-v3-turbo",
    "facebook/wav2vec2-base",
    # "facebook/wav2vec2-large",
]

results = run_inference_on_models(models, num_cores=16)

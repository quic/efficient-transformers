# -----------------------------------------------------------------------------
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------

"""Run Cohere ASR QPC inference on selected dataset clips."""

import argparse
import re
import unicodedata
from pathlib import Path

from datasets import load_dataset
from transformers import AutoProcessor

from QEfficient import QEFFAutoModelForSpeechSeq2Seq

DATASET_NAME = "hf-internal-testing/librispeech_asr_dummy"
DATASET_CONFIG = "clean"
DATASET_SPLIT = "validation"


def parse_indices(value: str) -> list[int]:
    try:
        indices = [int(item.strip()) for item in value.split(",") if item.strip()]
    except ValueError as error:
        raise argparse.ArgumentTypeError("indices must be comma-separated integers") from error
    if not indices or any(index < 0 for index in indices):
        raise argparse.ArgumentTypeError("indices must contain non-negative integers")
    return indices


def reset_session(model: QEFFAutoModelForSpeechSeq2Seq) -> None:
    if model.qpc_session is not None:
        model.qpc_session.deactivate()
        model.qpc_session = None


def normalize_text(value: str) -> str:
    value = unicodedata.normalize("NFKC", value).casefold()
    value = "".join(character if character.isalnum() else " " for character in value)
    return re.sub(r"\s+", " ", value).strip()


def trim_at_eos(token_ids: list[int], eos_token_id: int) -> tuple[list[int], bool]:
    try:
        eos_index = token_ids.index(eos_token_id)
    except ValueError:
        return token_ids, False
    return token_ids[: eos_index + 1], True


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-name", required=True, help="Cohere ASR model ID or local snapshot")
    parser.add_argument("--qpc-path", type=Path, required=True, help="Existing batch-1 QPC directory")
    parser.add_argument("--language", required=True, help="Processor language prompt, for example en or ar")
    parser.add_argument("--dataset-name", default=DATASET_NAME)
    parser.add_argument("--dataset-config", default=DATASET_CONFIG)
    parser.add_argument("--dataset-split", default=DATASET_SPLIT)
    parser.add_argument("--audio-column", default="audio")
    parser.add_argument("--text-column", default="text")
    parser.add_argument("--indices", type=parse_indices, default=parse_indices("0,1,2"))
    parser.add_argument("--generation-len", type=int, default=500)
    parser.add_argument("--device-id", type=int, default=0)
    parser.add_argument("--local-files-only", action="store_true")
    args = parser.parse_args()

    if not args.qpc_path.is_dir():
        parser.error(f"QPC directory does not exist: {args.qpc_path}")
    if args.generation_len <= 0:
        parser.error("--generation-len must be positive")

    dataset = load_dataset(args.dataset_name, args.dataset_config, split=args.dataset_split)
    invalid = [index for index in args.indices if index >= len(dataset)]
    if invalid:
        parser.error(f"dataset indices out of range for {len(dataset)} clips: {invalid}")

    processor = AutoProcessor.from_pretrained(
        args.model_name,
        local_files_only=args.local_files_only,
        trust_remote_code=False,
    )
    model = QEFFAutoModelForSpeechSeq2Seq.from_pretrained(
        args.model_name,
        local_files_only=args.local_files_only,
        trust_remote_code=False,
        dtype="float32",
    )
    model.qpc_path = args.qpc_path

    print(f"model={args.model_name}")
    print(f"qpc={args.qpc_path}")
    print(f"dataset={args.dataset_name}/{args.dataset_config}:{args.dataset_split}")
    print(f"language={args.language} device={args.device_id}")

    try:
        for index in args.indices:
            sample = dataset[index]
            source_audio = sample[args.audio_column]
            audio_values = source_audio["array"]
            sample_rate = int(source_audio["sampling_rate"])
            reference = sample.get(args.text_column, "")
            inputs = processor(
                audio_values,
                sampling_rate=sample_rate,
                language=args.language,
                return_tensors="pt",
            )
            execution = model.generate(
                inputs=inputs,
                generation_len=args.generation_len,
                device_ids=[args.device_id],
            )
            transcription = processor.batch_decode(execution.generated_ids, skip_special_tokens=True)[0].strip()
            token_ids, eos_reached = trim_at_eos(execution.generated_ids[0].tolist(), model.model.config.eos_token_id)
            print(f"\nclip[{index}]")
            print(f"desired_text:          {reference}")
            print(f"actual_generated_text: {transcription}")
            print(f"exact_text_match:      {transcription == reference}")
            print(f"normalized_text_match: {normalize_text(transcription) == normalize_text(reference)}")
            print(f"eos_reached:           {eos_reached}")
            print(f"generated_ids_to_eos:  {token_ids}")
            reset_session(model)
    finally:
        reset_session(model)


if __name__ == "__main__":
    main()

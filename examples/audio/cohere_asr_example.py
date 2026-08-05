# -----------------------------------------------------------------------------
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------

"""CohereLabs/cohere-transcribe-03-2026 speech-to-text on Cloud AI 100."""

import argparse

from datasets import load_dataset
from transformers import AutoProcessor

from QEfficient import QEFFAutoModelForSpeechSeq2Seq

MODEL_ID = "CohereLabs/cohere-transcribe-03-2026"


def main():
    parser = argparse.ArgumentParser(description="cohere_asr speech-to-text on Cloud AI 100")
    parser.add_argument("--model-id", default=MODEL_ID)
    parser.add_argument("--num-devices", type=int, default=4)
    parser.add_argument("--generation-len", type=int, default=100)
    args = parser.parse_args()

    ds = load_dataset("hf-internal-testing/librispeech_asr_dummy", "clean", split="validation")
    data = ds[0]["audio"]["array"].reshape(-1)
    sample_rate = ds[0]["audio"]["sampling_rate"]
    processor = AutoProcessor.from_pretrained(args.model_id)

    model = QEFFAutoModelForSpeechSeq2Seq.from_pretrained(args.model_id)
    model.compile(num_cores=16, num_devices=args.num_devices)
    exec_info = model.generate(
        inputs=processor(data, sampling_rate=sample_rate, return_tensors="pt"),
        generation_len=args.generation_len,
    )
    print(processor.batch_decode(exec_info.generated_ids)[0])


if __name__ == "__main__":
    main()

# -----------------------------------------------------------------------------
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------

import argparse

from datasets import load_dataset
from transformers import AutoProcessor

from QEfficient import QEFFAutoModelForCTC


def main():
    parser = argparse.ArgumentParser(description="CTC speech recognition inference with Wav2Vec2")
    parser.add_argument(
        "--model-name",
        type=str,
        default="facebook/wav2vec2-base-960h",
        help="HuggingFace CTC model ID (e.g., Wav2Vec2)",
    )

    parser.add_argument("--num-cores", type=int, default=16, help="Number of cores")
    args = parser.parse_args()

    print(f"Loading CTC model: {args.model_name}")

    ## STEP 1 -- load audio sample
    # Using a standard english dataset
    print("Loading audio sample from dataset...")
    ds = load_dataset("hf-internal-testing/librispeech_asr_dummy", "clean", split="validation")
    data = ds[0]["audio"]["array"]

    # Reshape so shape corresponds to data with batch size 1
    data = data.reshape(-1)

    # Load processor
    processor = AutoProcessor.from_pretrained(args.model_name)

    ## STEP 2 -- Load the model
    model = QEFFAutoModelForCTC.from_pretrained(args.model_name)

    ## STEP 3 -- Compile the model
    model.compile(num_cores=args.num_cores)

    ## STEP 4 -- Run the model and generate the output
    model_output = model.generate(processor, inputs=data)
    print(f"\nTranscription: {model_output}")


if __name__ == "__main__":
    main()

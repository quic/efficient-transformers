# -----------------------------------------------------------------------------
#
# Copyright (c) 2024 Qualcomm Innovation Center, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------

from datasets import load_dataset
from transformers import AutoProcessor

from QEfficient import QEFFAutoModelForSpeechSeq2Seq

base_model_name = "openai/whisper-tiny"
ctx_len = 25

## STEP 1 -- load audio sample, using a standard english dataset, can load specific files if longer audio needs to be tested; also load initial processor
ds = load_dataset("hf-internal-testing/librispeech_asr_dummy", "clean", split="validation")
data = ds[0]["audio"]["array"]
# reshape to so shape corresponds to data with batch size 1
data = data.reshape(-1)
sample_rate = ds[0]["audio"]["sampling_rate"]
processor = AutoProcessor.from_pretrained(base_model_name)

## STEP 2 -- init base model
qeff_model = QEFFAutoModelForSpeechSeq2Seq.from_pretrained(base_model_name)

## STEP 3 -- export and compile model
qeff_model.compile()

## STEP 4 -- generate output for loaded input and processor
exec_info = qeff_model.generate(
    inputs=processor(data, sampling_rate=sample_rate, return_tensors="pt"), generation_len=ctx_len
)

## STEP 5 (optional) -- use processor to decode output
print(processor.batch_decode(exec_info.generated_ids)[0])

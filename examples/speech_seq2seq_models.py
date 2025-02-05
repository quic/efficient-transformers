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
ctx_len = 150

## STEP 1 -- load audio sample, using a standard english dataset, can load specific files if longer audio needs to be tested; also load initial processor
ds = load_dataset("hf-internal-testing/librispeech_asr_dummy", "clean", split="validation")
data = ds[0]["audio"]["array"]
# reshape to so shape corresponds to data with batch size 1
data = data.reshape(-1)
sample_rate = ds[0]["audio"]["sampling_rate"]
processor = AutoProcessor.from_pretrained(base_model_name)

## STEP 2 -- init base model
qeff_model = QEFFAutoModelForSpeechSeq2Seq.from_pretrained(base_model_name)

## STEP 3 -- export model
onnx_path = qeff_model.export()

## STEP 4 -- compile model
qeff_model.compile(onnx_path)

## STEP 5 -- generate output for loaded input and processor
exec_info = qeff_model.generate(processor, inputs=data, sample_rate=sample_rate, generation_len=ctx_len)
print(exec_info.generated_texts)

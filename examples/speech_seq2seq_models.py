# -----------------------------------------------------------------------------
#
# Copyright (c) 2024 Qualcomm Innovation Center, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------

import numpy as np
import torch
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

## STEP 3 -- export and compile model
qeff_model.compile()

## STEP 4 -- prepare generate inputs
bs = 1
seq_len = 1
input_features = (
    processor(data, sampling_rate=sample_rate, return_tensors="pt").input_features.numpy().astype(np.float32)
)
decoder_input_ids = (
    torch.ones((bs, seq_len), dtype=torch.int64) * qeff_model.model.config.decoder_start_token_id
).numpy()
decoder_position_ids = torch.arange(seq_len, dtype=torch.int64).view(1, seq_len).repeat(bs, 1).numpy()
inputs = dict(
    input_features=input_features,
    decoder_input_ids=decoder_input_ids,
    decoder_position_ids=decoder_position_ids,
)

## STEP 5 -- generate output for loaded input and processor
exec_info = qeff_model.generate(inputs=inputs, generation_len=ctx_len)

## STEP 6 (optional) -- use processor to decode output
print(processor.batch_decode(exec_info.generated_ids))

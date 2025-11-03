# -----------------------------------------------------------------------------
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------

from datasets import load_dataset
from transformers import AutoProcessor

from QEfficient import QEFFAutoModelForCTC

base_model_name = "facebook/wav2vec2-base-960h"

ds = load_dataset("hf-internal-testing/librispeech_asr_dummy", "clean", split="validation")
data = ds[0]["audio"]["array"]
# reshape to so shape corresponds to data with batch size 1
data = data.reshape(-1)
sample_rate = ds[0]["audio"]["sampling_rate"]
processor = AutoProcessor.from_pretrained(base_model_name)

model = QEFFAutoModelForCTC.from_pretrained(base_model_name)
model.compile(num_cores=16)
print(model.generate(processor, inputs=data))

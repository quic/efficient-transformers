# -----------------------------------------------------------------------------
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------

import requests
import torch
import transformers
from PIL import Image
from transformers import AutoConfig, AutoProcessor, TextStreamer

from QEfficient import QEFFAutoModelForCausalLM

model_id = "allenai/Molmo-7B-D-0924"
config = AutoConfig.from_pretrained(model_id, trust_remote_code=True)
# For Testing Purpose Only
# config.num_hidden_layers = 2

## Activate Compute-Context-Length (CCL) feature by setting ccl_enabled=True when loading the model with from_pretrained().
## Use the optional comp_ctx_lengths argument to provide two lists of context lengths for the prefilling and decoding processes. If comp_ctx_lengths=None, the model will run with its default context length.
##   - The first list, comp_ctx_lengths_prefill, defines the compute-context-length values for the prefilling process.
##           -- The process starts with the first value in the list and gradually increases the context length based on the position_id of the current prompt chunk.
##   - The second list, comp_ctx_lengths_decode, defines the compute-context-length values for the decoding process.
##           -- During decoding, the model selects an appropriate context length from the list based on the input prompt length and cache index.
##           -- It starts from the correct value in the list and increases the context length dynamically when the cache index exceeds the current threshold.

# load the model
ctx_len = 8192
comp_ctx_lengths_prefill = [3072]  # None #
comp_ctx_lengths_decode = [4096, 8192]  # None #

qeff_model = QEFFAutoModelForCausalLM.from_pretrained(
    model_id,
    kv_offload=True,
    trust_remote_code=True,
    config=config,
    qaic_config={
        "ccl_enabled": True,
    },
)
tokenizer = transformers.AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)

### use skip_vision=True, if want to run only text, or false ###
skip_vision = False

if skip_vision:
    ## Only Text ##
    qeff_model.compile(
        prefill_seq_len=128,
        ctx_len=ctx_len,
        num_cores=16,
        num_devices=4,
        mxint8_kv_cache=True,
        aic_enable_depth_first=True,
        skip_vision=True,
        mos=1,
        comp_ctx_lengths_prefill=comp_ctx_lengths_prefill,
        comp_ctx_lengths_decode=comp_ctx_lengths_decode,
    )

    inputs = processor.process(text="Tell me about yourself")
    inputs = {k: v.unsqueeze(0) for k, v in inputs.items()}
    inputs["input_ids"] = inputs["input_ids"].to(torch.int64)
    inputs["attention_mask"] = torch.ones((inputs["input_ids"].shape), dtype=torch.int64)

    streamer = TextStreamer(tokenizer)
    output = qeff_model.generate(inputs=inputs, device_ids=[0, 1, 2, 3], generation_len=100)
    print(output.generated_ids)
    print(tokenizer.batch_decode(output.generated_ids))
    print(output)

else:
    ## Vision + Text ##
    qeff_model.compile(
        prefill_seq_len=128,
        ctx_len=ctx_len,
        num_cores=16,
        num_devices=4,
        mxint8_kv_cache=True,
        aic_enable_depth_first=True,
        mos=1,
        comp_ctx_lengths_prefill=comp_ctx_lengths_prefill,
        comp_ctx_lengths_decode=comp_ctx_lengths_decode,
    )

    ### IMAGE + TEXT ###
    image_url = "https://picsum.photos/id/237/536/354"

    image = Image.open(requests.get(image_url, stream=True).raw)
    image = image.resize((536, 354))

    inputs = processor.process(images=[image], text="Can you describe the image in detail.")

    inputs = {k: v.unsqueeze(0) for k, v in inputs.items()}
    inputs["pixel_values"] = inputs.pop("images")
    inputs["attention_mask"] = torch.ones((inputs["input_ids"].shape), dtype=torch.int64)

    valid = inputs["image_input_idx"] > 0
    valid = valid.reshape(1, -1)
    inputs["valid_idx"] = torch.nonzero(valid)[:, 1].unsqueeze(0)

    streamer = TextStreamer(tokenizer)
    output = qeff_model.generate(inputs=inputs, device_ids=[0, 1, 2, 3], generation_len=100)
    print(output.generated_ids)
    print(tokenizer.batch_decode(output.generated_ids))
    print(output)
    print()

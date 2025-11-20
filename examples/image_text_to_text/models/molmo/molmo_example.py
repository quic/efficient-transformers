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

# For faster execution user can run on 2 layers, This is only for testing purpose
# config.num_hidden_layers = 2

# load the model
qeff_model = QEFFAutoModelForCausalLM.from_pretrained(model_id, kv_offload=True, trust_remote_code=True, config=config)
tokenizer = transformers.AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)

### use skip_vision=Ture, if want to run only text, ow false ###
skip_vision = False

if skip_vision:
    ## Only Text ##
    qeff_model.compile(
        prefill_seq_len=128,
        ctx_len=4096,
        num_cores=16,
        num_devices=4,
        mxint8_kv_cache=True,
        aic_enable_depth_first=True,
        skip_vision=True,
        mos=1,
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
        ctx_len=4096,
        num_cores=16,
        num_devices=4,
        mxint8_kv_cache=True,
        aic_enable_depth_first=True,
        mos=1,
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

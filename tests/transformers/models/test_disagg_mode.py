# -----------------------------------------------------------------------------
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------

import time

import numpy as np
import pytest
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, HybridCache

from QEfficient import QEFFAutoModelForCausalLM
from QEfficient.generation.cloud_infer import QAICInferenceSession
from QEfficient.transformers.quantizers import replace_transformers_quantizers, undo_transformers_quantizers

replace_transformers_quantizers()

model_id = "openai/gpt-oss-20b"  # weights are not required to convert to fp32
# prompt = """
# Billions of years ago, in the vast emptiness of the early universe, tiny fluctuations in the density of matter began to grow under the influence of gravity. Clouds of gas—mostly hydrogen and helium—started to collapse, forming the first stars. These stars grouped together, bound by gravity, creating the earliest galaxies.
# Over time, these galaxies merged, collided, and evolved, shaping their spiral arms, elliptical forms, or irregular structures. Within their swirling depths, stars were born and died, enriching the galactic gas with heavier elements. These elements became the building blocks for planets, moons, and eventually life.
# Thus, from the quiet whispers of cosmic dust, a galaxy emerged—an island of stars, nebulae, and mysteries, drifting through the infinite sea of space.
# As the galaxy matured, its stars danced in intricate orbits, weaving patterns shaped by gravity and time. Supernovae exploded like cosmic fireworks, scattering elements across space and triggering new waves of star formation. Black holes formed at the hearts of galaxies, anchoring their structure and influencing their evolution. Over billions of years, the galaxy became a dynamic ecosystem—where stars are born, live, and die—each cycle adding to the richness of the cosmic tapestry.
# """
prompt = """
Once upon a time, in a small town, there lived a young boy named Alex. Alex was a curious and adventurous child, always eager to explore the world around him. One day, while playing in the park, Alex stumbled upon a mysterious old book hidden beneath a pile of leaves. The book was filled with stories of distant lands, magical creatures, and extraordinary adventures.

As Alex flipped through the pages, he discovered a map that led to a hidden treasure. Excited by the prospect of a real-life treasure hunt, Alex decided to embark on a thrilling journey. He packed his backpack with snacks, a flashlight, and a compass, and set off into the unknown.

The path to the treasure was not an easy one. Alex had to navigate through dense forests, cross rickety bridges, and solve riddles that guarded the treasure's location.
"""


@pytest.mark.parametrize("model_id", [model_id])
def test_disagg_mode(model_id):
    all_outputs = []
    # Run prefill
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    PREFILL_SEQ_LEN = 256
    CTX_LEN = 256
    inputs = tokenizer(prompt, return_tensors="np", padding=True)
    position_ids = inputs["attention_mask"].sum(1, keepdims=True)
    padded_len = inputs["input_ids"].shape[1]
    num_chunks = -(padded_len // -PREFILL_SEQ_LEN)  # ceil divide without float
    padded_len = num_chunks * PREFILL_SEQ_LEN  # Convert to a multiple of prompt_len

    # Initialize variables specific to request
    # Calculate the max generation length.
    max_gen_len = CTX_LEN - position_ids.max()
    generation_len = 50

    # model = AutoModelForCausalLM.from_pretrained(model_id, num_hidden_layers=2)
    model = AutoModelForCausalLM.from_pretrained(model_id)
    config = model.config
    inputs = tokenizer(prompt, return_tensors="np", padding="max_length", max_length=padded_len)
    inputs["position_ids"] = np.where(inputs.pop("attention_mask"), np.arange(padded_len), -1)
    inputs.pop("token_type_ids", None)
    inputs = {k: torch.from_numpy(v).to(model.device) for k, v in inputs.items()}
    cache = HybridCache(config=config, batch_size=1, max_cache_len=CTX_LEN)
    ins = tokenizer(prompt, return_tensors="pt")
    out = model(**ins, past_key_values=cache)
    puts = {
        "input_ids": out.logits[:, -1, :].argmax().reshape(1, -1),
        "position_ids": ins["input_ids"].shape[-1].reshape(1, -1),
    }
    import ipdb

    ipdb.set_trace()
    new_out = model(**puts, past_key_values=cache)
    model.generation_config.do_sample = False
    orig_all_out = model.generate(
        **tokenizer(prompt, return_tensors="pt"), past_key_values=cache, max_new_tokens=max_gen_len
    )
    undo_transformers_quantizers()

    qeff_model = QEFFAutoModelForCausalLM.from_pretrained(model_id)
    # qeff_model = QEFFAutoModelForCausalLM.from_pretrained(model_id, num_hidden_layers=2)
    qeff_model.prefill(True)
    config = qeff_model.model.config
    inputs = tokenizer(prompt, return_tensors="np", padding="max_length", max_length=padded_len)
    inputs["position_ids"] = np.where(inputs.pop("attention_mask"), np.arange(padded_len), -1)
    inputs.pop("token_type_ids", None)
    inputs = {k: torch.from_numpy(v) for k, v in inputs.items()}
    past_key_values = []
    for i in range(config.num_hidden_layers):
        cache_len = 128 if i % 2 == 0 else PREFILL_SEQ_LEN
        pad_shape = (1, 8, cache_len, 64)
        past_key = torch.zeros((pad_shape), dtype=torch.float32)
        past_value = torch.zeros((pad_shape), dtype=torch.float32)
        pkv = (past_key, past_value)
        past_key_values.append(pkv)
    inputs["past_key_values"] = past_key_values

    qeff_out = qeff_model.model(**inputs)

    import ipdb

    ipdb.set_trace()

    decode_qpc_path = qeff_model.compile(
        prefill_seq_len=1,
        ctx_len=CTX_LEN,
        num_cores=16,
        mxfp6_matmul=True,
        mxint8_kv_cache=True,
        num_devices=1,
        mos=1,
        aic_enable_depth_first=True,
        num_speculative_tokens=None,
        offload_pt_weights=False,
    )
    prefill_qpc_path = qeff_model.compile(
        prefill_seq_len=PREFILL_SEQ_LEN,
        ctx_len=CTX_LEN,
        num_cores=16,
        mxfp6_matmul=True,
        mxint8_kv_cache=True,
        num_devices=1,
        mos=1,
        aic_enable_depth_first=True,
        num_speculative_tokens=None,
        prefill_only=True,
    )

    prefill_session = QAICInferenceSession(prefill_qpc_path)

    logits_out_placeholder = np.zeros((1, 1, 201088), dtype=np.float32)
    prefill_session.set_buffers({"logits": logits_out_placeholder})
    inputs.pop("past_key_values")
    inputs = {k: v.detach().numpy() for k, v in inputs.items()}
    st = time.time()
    qpc_out = prefill_session.run(inputs)
    print(f"time for prefill_run={time.time() - st} sec\n")
    import ipdb

    ipdb.set_trace()
    decode_session = QAICInferenceSession(decode_qpc_path)
    decode_session.set_buffers({"logits": logits_out_placeholder})
    decode_session.skip_buffers(
        [x for x in decode_session.input_names + decode_session.output_names if x.startswith("past_")]
    )

    decode_inputs = {
        "input_ids": np.argmax(qpc_out["logits"]).reshape(1, 1),
        "position_ids": np.max(inputs["position_ids"]).reshape(1, 1) + 1,
    }

    all_outputs.append(decode_inputs["input_ids"][0][0])
    for i in range(config.num_hidden_layers):
        decode_inputs[f"past_key.{i}"] = qpc_out[f"past_key.{i}_RetainedState"]
        decode_inputs[f"past_value.{i}"] = qpc_out[f"past_value.{i}_RetainedState"]

    st = time.time()
    decode_out = decode_session.run(decode_inputs)
    print(f"time for first run of decode with KV as input = {time.time() - st} sec\n")

    st = time.time()
    for i in range(generation_len - 2):
        loop_decode_inputs = {
            "input_ids": np.argmax(decode_out["logits"]).reshape(1, 1),
            "position_ids": np.max(inputs["position_ids"]).reshape(1, 1) + 1,
        }
        all_outputs.append(loop_decode_inputs["input_ids"][0][0])
        decode_out = decode_session.run(loop_decode_inputs)

    print(f"time for decode generation = {(time.time() - st) / (generation_len - 2)}")
    print(all_outputs)
    print(tokenizer.decode(all_outputs))

# -----------------------------------------------------------------------------
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------

import os

import numpy as np
import pytest
from transformers import AutoTokenizer

from QEfficient.generation.text_generation_inference import TextGeneration
from QEfficient.transformers.models.modeling_auto import QEFFAutoModelForCausalLM
from QEfficient.utils._utils import create_json
from QEfficient.utils.constants import QnnConstants

test_models = ["gpt2"]


# The test should first generate output with some prefix+suffix1 or batch_id and then confirm that we are still able to execute of prefix+suffix2 on same batch id and getting correct output.
@pytest.mark.on_qaic
@pytest.mark.parametrize("model_name", test_models)
def test_simple_prefix_caching(model_name):
    qeff_model = QEFFAutoModelForCausalLM.from_pretrained(model_name, continuous_batching=True)
    qeff_model.compile(
        prefill_seq_len=128,
        ctx_len=256,
        full_batch_size=2,
        kv_cache_batch_size=4,
        num_cores=14,
    )
    prefix_caching_inference(model_name=model_name, qpc_path=qeff_model.qpc_path)
    assert os.path.isfile(os.path.join(os.path.dirname(qeff_model.qpc_path), "qconfig.json"))


@pytest.mark.on_qaic
@pytest.mark.qnn
@pytest.mark.parametrize("model_name", test_models)
def test_simple_prefix_caching_qnn(model_name):
    qeff_model = QEFFAutoModelForCausalLM.from_pretrained(model_name, continuous_batching=True)
    qnn_config_json_path = os.path.join(os.getcwd(), "qnn_config.json")
    create_json(qnn_config_json_path, QnnConstants.QNN_SAMPLE_CONFIG)

    qeff_model.compile(
        prefill_seq_len=128,
        ctx_len=256,
        full_batch_size=2,
        kv_cache_batch_size=4,
        num_cores=14,
        enable_qnn=True,
        qnn_config=qnn_config_json_path,
    )
    prefix_caching_inference(model_name=model_name, qpc_path=qeff_model.qpc_path)
    assert os.path.isfile(os.path.join(os.path.dirname(qeff_model.qpc_path), "qconfig.json"))
    os.remove(qnn_config_json_path)


def prefix_caching_inference(model_name, qpc_path):
    prefixes = ["Once upon a time ", "Once upon a time "]
    suffixes1 = ["in a land far away", "there was a small village"]
    suffixes2 = ["a little girl", "in a bustling city"]

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    generator = TextGeneration(tokenizer=tokenizer, qpc_path=qpc_path, full_batch_size=2, ctx_len=256)

    prompts = [pref + suff for pref, suff in zip(prefixes, suffixes1)]

    # generation for batch_indices = 0, 1
    prompts_exec_info = generator.generate(prompts)
    ##############################
    # generation for batch_indices
    ##############################
    # Run prefill for indices 2, 3 with same prompts
    out2, pos2, gen_len2 = generator._qaic_model.run_prefill(
        prompts[0], generation_len=None, decode_batch_id=np.array(2, dtype=np.int64).reshape(1, 1)
    )
    out3, pos3, gen_len3 = generator._qaic_model.run_prefill(
        prompts[1], generation_len=None, decode_batch_id=np.array(3, dtype=np.int64).reshape(1, 1)
    )

    # Run decode for batch indices 2, 3
    decode_inputs = {
        "input_ids": np.array([[out2["logits"].argmax(2)[0][0]], [out3["logits"].argmax(2)[0][0]]]),
        "position_ids": np.array([[pos2[0][0]], [pos3[0][0]]]),
        "batch_index": np.array([[2], [3]], dtype=np.int64),
    }

    # Set logits placeholder for decode
    logits_out_placeholder = np.zeros(
        (
            generator._qaic_model.full_batch_size,
            generator._qaic_model._decode_seq_len,
            generator._qaic_model._vocab_size,
        ),
        dtype=np.float32,
    )
    generator._qaic_model._session.set_buffers({"logits": logits_out_placeholder})

    generation_outputs = []
    for i in range(gen_len2):
        generation_outputs.append(decode_inputs["input_ids"])
        outputs = generator._qaic_model._session.run(decode_inputs)
        logits = outputs["logits"]
        if len(logits.shape) == 2:
            logits = np.expand_dims(logits, 1)
        next_token_id = logits.argmax(2)

        decode_inputs["input_ids"] = next_token_id
        decode_inputs["position_ids"] += 1

    assert np.all(generator._qaic_model.generated_ids[0, :gen_len2] == [int(val[0]) for val in generation_outputs])
    assert np.all(generator._qaic_model.generated_ids[1, :gen_len2] == [int(val[1]) for val in generation_outputs])

    ##############################
    # Now rerun with cached prefix on 0th index with prompt3 and use -1 for 1st index
    ##############################

    nprompts = [pref + suff for pref, suff in zip(prefixes, suffixes2)]

    ## Prefill run on index 0
    prompt = nprompts[0]
    inputs = tokenizer(prompt, return_tensors="np", padding=True)
    position_ids = inputs["attention_mask"].sum(1, keepdims=True)
    padded_len = inputs["input_ids"].shape[1]
    num_chunks = -(padded_len // -generator._qaic_model._prefill_seq_len)
    padded_len = num_chunks * generator._qaic_model._prefill_seq_len  # Convert to a multiple of prompt_len

    # Initialize variables specific to request
    # Calculate the max generation length.
    max_gen_len = generator._qaic_model._ctx_len - position_ids.max()

    # Set the prefill logic buffer
    logits_out_placeholder = np.zeros((1, 1, generator._qaic_model._vocab_size), dtype=np.float32)
    generator._qaic_model._session.set_buffers({"logits": logits_out_placeholder})
    inputs = tokenizer(prompt, return_tensors="np", padding="max_length", max_length=padded_len)
    inputs["position_ids"] = np.where(inputs.pop("attention_mask"), np.arange(padded_len), -1)
    inputs.pop("token_type_ids", None)
    inputs["batch_index"] = np.array([[0]], dtype=np.int64)
    norm_outputs = generator._qaic_model._session.run(inputs)
    inputs["input_ids"][:, :3] = inputs["input_ids"][:, 4:7]
    inputs["input_ids"][:, 3:] = 50256
    inputs["position_ids"][:, :3] = inputs["position_ids"][:, 4:7]
    inputs["position_ids"][:, 3:] = -1
    mod_outputs = generator._qaic_model._session.run(inputs)
    assert (mod_outputs["logits"] == norm_outputs["logits"]).all()
    decode_inputs = {
        "input_ids": np.array([[mod_outputs["logits"].argmax(2)[0][0]], [0]]),
        "position_ids": np.array([[position_ids[0][0]], [-1]]),
        "batch_index": np.array([[0], [1]], dtype=np.int64),
    }

    # Set logits placeholder for decode
    logits_out_placeholder = np.zeros(
        (
            generator._qaic_model.full_batch_size,
            generator._qaic_model._decode_seq_len,
            generator._qaic_model._vocab_size,
        ),
        dtype=np.float32,
    )
    generator._qaic_model._session.set_buffers({"logits": logits_out_placeholder})

    generation_outputs = []
    for i in range(max_gen_len):
        generation_outputs.append(decode_inputs["input_ids"])
        outputs = generator._qaic_model._session.run(decode_inputs)
        logits = outputs["logits"]
        if len(logits.shape) == 2:
            logits = np.expand_dims(logits, 1)
        next_token_id = logits.argmax(2)

        decode_inputs["input_ids"] = next_token_id
        decode_inputs["position_ids"][0][0] += 1

    # TODO: add a check if this matches normal execution for same prompt
    ##############
    # Now run decode on 1st index again with mod_inputs and check if output is correct
    ##############
    decode_inputs = {
        "input_ids": np.array([[0], [prompts_exec_info.generated_ids[1][0]]]),
        "position_ids": np.array([[-1], [9]]),
        "batch_index": np.array([[0], [1]], dtype=np.int64),
    }

    # Set logits placeholder for decode
    logits_out_placeholder = np.zeros(
        (
            generator._qaic_model.full_batch_size,
            generator._qaic_model._decode_seq_len,
            generator._qaic_model._vocab_size,
        ),
        dtype=np.float32,
    )
    generator._qaic_model._session.set_buffers({"logits": logits_out_placeholder})

    generation_outputs_prefill_cached = []
    for i in range(max_gen_len):
        generation_outputs_prefill_cached.append(decode_inputs["input_ids"])
        outputs = generator._qaic_model._session.run(decode_inputs)
        logits = outputs["logits"]
        if len(logits.shape) == 2:
            logits = np.expand_dims(logits, 1)
        next_token_id = logits.argmax(2)

        decode_inputs["input_ids"] = next_token_id
        decode_inputs["position_ids"][1][0] += 1

    assert np.all(
        prompts_exec_info.generated_ids[1][:247] == [int(val[1]) for val in generation_outputs_prefill_cached][:247]
    )

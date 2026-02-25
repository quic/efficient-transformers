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

model_id = "openai/gpt-oss-120b"  # weights are not required to convert to fp32

prompt2 = """
Once upon a time, in a small town, there lived a young boy named Alex. Alex was a curious and adventurous child, always eager to explore the world around him. One day, while playing in the park, Alex stumbled upon a mysterious old book hidden beneath a pile of leaves. The book was filled with stories of distant lands, magical creatures, and extraordinary adventures.

As Alex flipped through the pages, he discovered a map that led to a hidden treasure. Excited by the prospect of a real-life treasure hunt, Alex decided to embark on a thrilling journey. He packed his backpack with snacks, a flashlight, and a compass, and set off into the unknown.

The path to the treasure was not an easy one. Alex had to navigate through dense forests, cross rickety bridges, and solve riddles that guarded the treasure's location.
"""
prompt1 = "Once upon a time"

prompts = [prompt1, prompt2]


@pytest.mark.on_qaic
@pytest.mark.llm_model
@pytest.mark.parametrize("model_id", [model_id])
@pytest.mark.parametrize("prompt", prompts)
def test_disagg_mode_prefill(model_id, prompt):
    # Run prefill
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    PREFILL_SEQ_LEN = 256
    CTX_LEN = 256
    inputs = tokenizer(prompt, return_tensors="np", padding=True)
    padded_len = inputs["input_ids"].shape[1]
    num_chunks = -(padded_len // -PREFILL_SEQ_LEN)  # ceil divide without float
    padded_len = num_chunks * PREFILL_SEQ_LEN  # Convert to a multiple of prompt_len

    replace_transformers_quantizers()
    model = AutoModelForCausalLM.from_pretrained(model_id, num_hidden_layers=2)
    config = model.config
    inputs = tokenizer(prompt, return_tensors="np", padding="max_length", max_length=padded_len)
    inputs["position_ids"] = np.where(inputs.pop("attention_mask"), np.arange(padded_len), -1)
    inputs.pop("token_type_ids", None)
    inputs = {k: torch.from_numpy(v).to(model.device) for k, v in inputs.items()}
    cache = HybridCache(config=config, batch_size=1, max_cache_len=CTX_LEN)
    ins = tokenizer(prompt, return_tensors="pt")
    out = model(**ins, past_key_values=cache)

    undo_transformers_quantizers()

    qeff_model = QEFFAutoModelForCausalLM.from_pretrained(model_id, num_hidden_layers=2)
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

    # Check our pytorch implementation
    assert (qeff_out.logits - out.logits[:, -1, :]).abs().max() < 1e-4

    prefill_qpc_path = qeff_model.compile(
        prefill_seq_len=PREFILL_SEQ_LEN,
        ctx_len=CTX_LEN,
        num_cores=16,
        mxfp6_matmul=False,
        mxint8_kv_cache=False,
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
    del prefill_session
    # Check QAIC output isclose with QEFF pytorch output
    assert (torch.from_numpy(qpc_out["logits"]) - qeff_out.logits).abs().max() < 5e-2


@pytest.mark.skip(reason="no way of currently testing this without the assert sdk")
@pytest.mark.on_qaic
@pytest.mark.llm_model
@pytest.mark.parametrize("model_id", [model_id])
@pytest.mark.parametrize("prompt", prompts)
def test_disagg_mode_prefill_chunked(model_id, prompt):
    # Run prefill
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    PREFILL_SEQ_LEN = 128
    CTX_LEN = 128 * 3
    inputs = tokenizer(prompt, return_tensors="np", padding=True)
    padded_len = inputs["input_ids"].shape[1]
    num_chunks = -(padded_len // -PREFILL_SEQ_LEN)  # ceil divide without float
    padded_len = num_chunks * PREFILL_SEQ_LEN  # Convert to a multiple of prompt_len

    replace_transformers_quantizers()
    model = AutoModelForCausalLM.from_pretrained(model_id, num_hidden_layers=2)
    config = model.config
    inputs = tokenizer(prompt, return_tensors="np", padding="max_length", max_length=padded_len)
    inputs["position_ids"] = np.where(inputs.pop("attention_mask"), np.arange(padded_len), -1)
    inputs.pop("token_type_ids", None)
    inputs = {k: torch.from_numpy(v).to(model.device) for k, v in inputs.items()}
    cache = HybridCache(config=config, batch_size=1, max_cache_len=CTX_LEN)
    ins = tokenizer(prompt, return_tensors="pt")
    out = model(**ins, past_key_values=cache)

    undo_transformers_quantizers()

    qeff_model = QEFFAutoModelForCausalLM.from_pretrained(model_id, num_hidden_layers=2)
    qeff_model.prefill(True, enable_chunking=True)
    config = qeff_model.model.config
    inputs = tokenizer(prompt, return_tensors="np", padding="max_length", max_length=padded_len)
    inputs["position_ids"] = np.where(inputs.pop("attention_mask"), np.arange(padded_len), -1)
    inputs.pop("token_type_ids", None)
    inputs = {k: torch.from_numpy(v) for k, v in inputs.items()}
    past_key_values = []
    for i in range(config.num_hidden_layers):
        cache_len = CTX_LEN
        pad_shape = (1, 8, cache_len, 64)
        past_key = torch.zeros((pad_shape), dtype=torch.float32)
        past_value = torch.zeros((pad_shape), dtype=torch.float32)
        pkv = (past_key, past_value)
        past_key_values.append(pkv)
    inputs["past_key_values"] = past_key_values

    for i in range(num_chunks):
        chunk_inputs = inputs.copy()
        chunk_inputs["input_ids"] = inputs["input_ids"][:, i * PREFILL_SEQ_LEN : (i + 1) * PREFILL_SEQ_LEN]
        chunk_inputs["position_ids"] = inputs["position_ids"][:, i * PREFILL_SEQ_LEN : (i + 1) * PREFILL_SEQ_LEN]

        qeff_out = qeff_model.model(**chunk_inputs)
        inputs["past_key_values"] = qeff_out["past_key_values"]

    # Check our pytorch implementation
    assert (qeff_out.logits - out.logits[:, -1, :]).abs().max() < 1e-4

    prefill_qpc_path = qeff_model.compile(
        prefill_seq_len=PREFILL_SEQ_LEN,
        ctx_len=CTX_LEN,
        num_cores=16,
        mxfp6_matmul=False,
        mxint8_kv_cache=False,
        num_devices=1,
        mos=1,
        aic_enable_depth_first=True,
        num_speculative_tokens=None,
        prefill_only=True,
        enable_chunking=True,
    )
    prefill_session = QAICInferenceSession(prefill_qpc_path)
    prefill_session.skip_buffers(
        [x for x in prefill_session.input_names + prefill_session.output_names if x.startswith("past_")]
    )
    logits_out_placeholder = np.zeros((1, 1, 201088), dtype=np.float32)
    prefill_session.set_buffers({"logits": logits_out_placeholder})
    inputs.pop("past_key_values")
    inputs = {k: v.detach().numpy() for k, v in inputs.items()}
    st = time.time()
    for i in range(num_chunks):
        chunk_inputs = inputs.copy()
        chunk_inputs["input_ids"] = inputs["input_ids"][:, i * PREFILL_SEQ_LEN : (i + 1) * PREFILL_SEQ_LEN]
        chunk_inputs["position_ids"] = inputs["position_ids"][:, i * PREFILL_SEQ_LEN : (i + 1) * PREFILL_SEQ_LEN]
        qpc_out = prefill_session.run(chunk_inputs)
    print(f"time for prefill_run={time.time() - st} sec\n")
    del prefill_session
    # Check QAIC output isclose with QEFF pytorch output
    assert (torch.from_numpy(qpc_out["logits"]) - qeff_out.logits).abs().max() < 8e-2

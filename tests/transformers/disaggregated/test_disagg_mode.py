# -----------------------------------------------------------------------------
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------

import json
import os
import time

import numpy as np
import pytest
import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer, HybridCache

from QEfficient import QEFFAutoModelForCausalLM
from QEfficient.generation.cloud_infer import QAICInferenceSession
from QEfficient.transformers.quantizers import replace_transformers_quantizers, undo_transformers_quantizers

# Dummy model configs — loaded from the shared config file.
_CONFIG_FILE = os.path.join(os.path.dirname(__file__), "..", "..", "configs", "causal_model_configs.json")
with open(_CONFIG_FILE) as _f:
    _raw = json.load(_f)

_DISAGG_DUMMY_CONFIGS = {
    entry["model_name"]: {
        "model_type": entry["model_type"],
        "tokenizer_id": entry.get("tokenizer_id", entry["model_name"]),
        **entry["additional_params"],
    }
    for entry in _raw["disaggregated_dummy_models"]
}

# Test parameters: model IDs to test (loaded from config)
# - model_id_blocking: models that use blocking/sliding window attention
# - model_id_chunking: models that use chunking
model_id_blocking = [name for name, cfg in _DISAGG_DUMMY_CONFIGS.items() if cfg["model_type"] == "gpt_oss"]
model_id_chunking = [name for name, cfg in _DISAGG_DUMMY_CONFIGS.items() if cfg["model_type"] == "qwen3_moe"]

prompt2 = """
Once upon a time, in a small town, there lived a young boy named Alex. Alex was a curious and adventurous child, always eager to explore the world around him. One day, while playing in the park, Alex stumbled upon a mysterious old book hidden beneath a pile of leaves. The book was filled with stories of distant lands, magical creatures, and extraordinary adventures.

As Alex flipped through the pages, he discovered a map that led to a hidden treasure. Excited by the prospect of a real-life treasure hunt, Alex decided to embark on a thrilling journey. He packed his backpack with snacks, a flashlight, and a compass, and set off into the unknown.

The path to the treasure was not an easy one. Alex had to navigate through dense forests, cross rickety bridges, and solve riddles that guarded the treasure's location.
"""
prompt1 = "Once upon a time"
prompts = [prompt1, prompt2]


def _make_dummy_model(model_id: str) -> AutoModelForCausalLM:
    """Create a tiny model from a dummy config — no weight download required.

    A fixed seed ensures the weights are reproducible across test runs so that
    the QAIC-compiled model (which may be cached on disk) always matches the
    in-process PyTorch model used for reference comparisons.

    Weights are scaled to std≈0.02 (matching real transformer init) so that
    intermediate activations stay small and float16 rounding errors on QAIC
    remain within the 5e-2 tolerance used for logit accuracy checks.
    """
    cfg = _DISAGG_DUMMY_CONFIGS[model_id]
    model_type = cfg["model_type"]
    params = {k: v for k, v in cfg.items() if k not in ("model_type", "tokenizer_id")}
    config = AutoConfig.for_model(model_type, **params)
    torch.manual_seed(42)
    model = AutoModelForCausalLM.from_config(config, attn_implementation="eager")
    with torch.no_grad():
        for param in model.parameters():
            param.mul_(0.02)
    return model


@pytest.mark.on_qaic
@pytest.mark.llm_model
@pytest.mark.parametrize("model_id", model_id_blocking)
@pytest.mark.parametrize("prompt", prompts)
def test_disagg_mode_prefill(model_id, prompt):
    # Run prefill
    tokenizer_id = _DISAGG_DUMMY_CONFIGS[model_id].get("tokenizer_id", model_id)
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    PREFILL_SEQ_LEN = 256
    CTX_LEN = 256

    # Tokenize once; reuse for both reference and qeff model
    raw_inputs = tokenizer(prompt, return_tensors="np", padding=True)
    padded_len = raw_inputs["input_ids"].shape[1]
    num_chunks = -(padded_len // -PREFILL_SEQ_LEN)  # ceil divide without float
    padded_len = num_chunks * PREFILL_SEQ_LEN  # Convert to a multiple of prompt_len

    replace_transformers_quantizers()
    model = _make_dummy_model(model_id)
    config = model.config

    raw_inputs = tokenizer(prompt, return_tensors="np", padding="max_length", max_length=padded_len)
    raw_inputs["position_ids"] = np.where(raw_inputs.pop("attention_mask"), np.arange(padded_len), -1)
    raw_inputs.pop("token_type_ids", None)

    inputs = {k: torch.from_numpy(v).to(model.device) for k, v in raw_inputs.items()}
    cache = HybridCache(config=config, batch_size=1, max_cache_len=CTX_LEN)
    ins = tokenizer(prompt, return_tensors="pt")
    out = model(**ins, past_key_values=cache)

    undo_transformers_quantizers()

    qeff_model = QEFFAutoModelForCausalLM(model)
    qeff_model.prefill(True)
    config = qeff_model.model.config

    inputs = {k: torch.from_numpy(v) for k, v in raw_inputs.items()}
    past_key_values = []
    for i in range(config.num_hidden_layers):
        cache_len = config.sliding_window if i % 2 == 0 else PREFILL_SEQ_LEN
        pad_shape = (1, config.num_key_value_heads, cache_len, config.head_dim)
        past_key = torch.zeros(pad_shape, dtype=torch.float32)
        past_value = torch.zeros(pad_shape, dtype=torch.float32)
        past_key_values.append((past_key, past_value))
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
    logits_out_placeholder = np.zeros((1, 1, config.vocab_size), dtype=np.float32)
    prefill_session.set_buffers({"logits": logits_out_placeholder})
    inputs.pop("past_key_values")
    inputs = {k: v.detach().numpy() for k, v in inputs.items()}
    st = time.time()
    qpc_out = prefill_session.run(inputs)
    print(f"time for prefill_run={time.time() - st} sec\n")
    del prefill_session
    assert (torch.from_numpy(qpc_out["logits"]) - qeff_out.logits).abs().max() < 5e-2


@pytest.mark.on_qaic
@pytest.mark.llm_model
@pytest.mark.parametrize("model_id", model_id_chunking)
@pytest.mark.parametrize("prompt", prompts)
def test_disagg_mode_prefill_chunked(model_id, prompt):
    # Run prefill
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    PREFILL_SEQ_LEN = 128
    CTX_LEN = 128 * 3

    # Tokenize once; reuse for both reference and qeff model
    raw_inputs = tokenizer(prompt, return_tensors="np", padding=True)
    padded_len = raw_inputs["input_ids"].shape[1]
    num_chunks = -(padded_len // -PREFILL_SEQ_LEN)  # ceil divide without float
    padded_len = num_chunks * PREFILL_SEQ_LEN  # Convert to a multiple of prompt_len

    replace_transformers_quantizers()
    model = _make_dummy_model(model_id)
    config = model.config

    raw_inputs = tokenizer(prompt, return_tensors="np", padding="max_length", max_length=padded_len)
    raw_inputs["position_ids"] = np.where(raw_inputs.pop("attention_mask"), np.arange(padded_len), -1)
    raw_inputs.pop("token_type_ids", None)

    inputs = {k: torch.from_numpy(v).to(model.device) for k, v in raw_inputs.items()}
    cache = HybridCache(config=config, batch_size=1, max_cache_len=CTX_LEN)
    ins = tokenizer(prompt, return_tensors="pt")
    out = model(**ins, past_key_values=cache)

    undo_transformers_quantizers()

    # Reuse the already-loaded model — avoids a second full model load
    qeff_model = QEFFAutoModelForCausalLM(model)
    qeff_model.prefill(True, enable_chunking=True)
    config = qeff_model.model.config

    # head_dim is explicit in gpt_oss but computed for qwen3_moe
    head_dim = getattr(config, "head_dim", config.hidden_size // config.num_attention_heads)

    inputs = {k: torch.from_numpy(v) for k, v in raw_inputs.items()}
    past_key_values = []
    for i in range(config.num_hidden_layers):
        pad_shape = (1, config.num_key_value_heads, CTX_LEN, head_dim)
        past_key = torch.zeros(pad_shape, dtype=torch.float32)
        past_value = torch.zeros(pad_shape, dtype=torch.float32)
        past_key_values.append((past_key, past_value))
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
    logits_out_placeholder = np.zeros((1, 1, config.vocab_size), dtype=np.float32)
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
    assert (torch.from_numpy(qpc_out["logits"]) - qeff_out.logits).abs().max() < 5e-2


@pytest.mark.on_qaic
@pytest.mark.llm_model
@pytest.mark.parametrize("model_id", model_id_blocking)
@pytest.mark.parametrize("prompt", [prompt1])
def test_disagg_mode_prefill_only_and_decode_only(model_id, prompt):
    # Run prefill for original pytorch model
    tokenizer_id = _DISAGG_DUMMY_CONFIGS[model_id].get("tokenizer_id", model_id)
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    PREFILL_SEQ_LEN = 256
    CTX_LEN = 256

    raw_inputs = tokenizer(prompt, return_tensors="np", padding=True)
    padded_len = raw_inputs["input_ids"].shape[1]
    num_chunks = -(padded_len // -PREFILL_SEQ_LEN)  # ceil divide without float
    padded_len = num_chunks * PREFILL_SEQ_LEN  # Convert to a multiple of prompt_len

    replace_transformers_quantizers()
    model = _make_dummy_model(model_id)
    config = model.config

    raw_inputs = tokenizer(prompt, return_tensors="np", padding="max_length", max_length=padded_len)
    raw_inputs["position_ids"] = np.where(raw_inputs.pop("attention_mask"), np.arange(padded_len), -1)
    raw_inputs.pop("token_type_ids", None)

    inputs = {k: torch.from_numpy(v).to(model.device) for k, v in raw_inputs.items()}
    cache = HybridCache(config=config, batch_size=1, max_cache_len=CTX_LEN)
    ins = tokenizer(prompt, return_tensors="pt")
    orig_out = model(**ins, past_key_values=cache)

    position_ids = inputs["position_ids"]
    generated_ids = []
    generation_len = 10
    out = orig_out
    for _ in range(1, generation_len):
        next_token_id = out["logits"][:, -1, :].argmax(-1).reshape(-1, 1)
        generated_ids.append(next_token_id)
        position_ids = position_ids.max(1, keepdim=True).values + 1
        decode_inputs = {
            "input_ids": next_token_id,
            "position_ids": position_ids,
            "past_key_values": out["past_key_values"],
        }
        out = model(**decode_inputs)

    generated_ids.append(out["logits"][:, -1, :].argmax(-1).reshape(-1, 1))
    generated_ids = np.concatenate(generated_ids, axis=1)
    predicted_string = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
    print("Original HF Model Outputs (Torch CPU): \n")
    print("Prompt:", repr(prompt))
    print("Completion:", repr(predicted_string))

    undo_transformers_quantizers()

    prefill_qeff_model = QEFFAutoModelForCausalLM(model)
    prefill_qeff_model.prefill(enable=True)
    config = prefill_qeff_model.model.config

    past_key_values = []
    for i in range(config.num_hidden_layers):
        cache_len = config.sliding_window if i % 2 == 0 else PREFILL_SEQ_LEN
        pad_shape = (1, config.num_key_value_heads, cache_len, config.head_dim)
        past_key = torch.zeros(pad_shape, dtype=torch.float32)
        past_value = torch.zeros(pad_shape, dtype=torch.float32)
        past_key_values.append((past_key, past_value))
    inputs["past_key_values"] = past_key_values

    prefill_qeff_out = prefill_qeff_model.model(**inputs)

    # Check our pytorch implementation
    assert (prefill_qeff_out.logits - orig_out.logits[:, -1, :]).abs().max() < 1e-4

    decode_qeff_model = QEFFAutoModelForCausalLM(model)
    decode_qeff_model.prefill(enable=False)
    qeff_out = prefill_qeff_out

    position_ids = inputs["position_ids"]
    qeff_generated_ids = []
    for _ in range(1, generation_len):
        next_token_id = qeff_out["logits"][:, -1, :].argmax(-1).reshape(-1, 1)
        qeff_generated_ids.append(next_token_id)
        position_ids = position_ids.max(1, keepdim=True).values + 1
        decode_inputs = {
            "input_ids": next_token_id,
            "position_ids": position_ids,
            "past_key_values": qeff_out["past_key_values"],
        }
        qeff_out = decode_qeff_model.model(**decode_inputs)

    qeff_generated_ids.append(out["logits"][:, -1, :].argmax(-1).reshape(-1, 1))
    qeff_generated_ids = np.concatenate(qeff_generated_ids, axis=1)
    predicted_string = tokenizer.batch_decode(qeff_generated_ids, skip_special_tokens=True)
    print("QEFF Transformed Model Outputs (Torch CPU): \n")
    print("Prompt:", repr(prompt))
    print("Completion:", repr(predicted_string))

    assert (qeff_generated_ids == generated_ids).all()

    prefill_qpc_path = prefill_qeff_model.compile(
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
    logits_out_placeholder = np.zeros((1, 1, config.vocab_size), dtype=np.float32)
    prefill_session.set_buffers({"logits": logits_out_placeholder})
    inputs.pop("past_key_values")
    inputs = {k: v.detach().numpy() for k, v in inputs.items()}
    qpc_out = prefill_session.run(inputs)
    del prefill_session
    assert (torch.from_numpy(qpc_out["logits"]) - prefill_qeff_out.logits).abs().max() < 5e-2

    decode_qpc_path = decode_qeff_model.compile(
        prefill_seq_len=1,
        ctx_len=CTX_LEN,
        num_cores=16,
        mxfp6_matmul=False,
        mxint8_kv_cache=False,
        num_devices=1,
        mos=1,
        aic_enable_depth_first=True,
        num_speculative_tokens=None,
        offload_pt_weights=False,  # Need the weights in memory for prefill-model export/compilation in the next step
    )

    qpc_outputs = []
    decode_session = QAICInferenceSession(decode_qpc_path)
    decode_session.set_buffers({"logits": logits_out_placeholder})

    decode_inputs = {
        "input_ids": np.argmax(qpc_out["logits"]).reshape(1, 1),
        "position_ids": np.max(inputs["position_ids"]).reshape(1, 1) + 1,
    }

    qpc_outputs.append(decode_inputs["input_ids"][0][0])
    for i in range(config.num_hidden_layers):
        if i % 2 == 0 and decode_inputs["position_ids"] >= config.sliding_window:
            k = qpc_out[f"past_key.{i}_RetainedState"]
            v = qpc_out[f"past_value.{i}_RetainedState"]
            mod_pos_id = config.sliding_window - decode_inputs["position_ids"][0][0] % config.sliding_window
            decode_inputs[f"past_key.{i}"] = np.concatenate((k[:, :, mod_pos_id:, :], k[:, :, :mod_pos_id, :]), axis=-2)
            decode_inputs[f"past_value.{i}"] = np.concatenate(
                (v[:, :, mod_pos_id:, :], v[:, :, :mod_pos_id, :]), axis=-2
            )
        else:
            decode_inputs[f"past_key.{i}"] = qpc_out[f"past_key.{i}_RetainedState"]
            decode_inputs[f"past_value.{i}"] = qpc_out[f"past_value.{i}_RetainedState"]

    decode_out = decode_session.run(decode_inputs)
    decode_session.skip_buffers(
        [x for x in decode_session.input_names + decode_session.output_names if x.startswith("past_")]
    )
    pos_id = np.max(decode_inputs["position_ids"]).reshape(1, 1) + 1
    for i in range(generation_len - 1):
        loop_decode_inputs = {
            "input_ids": np.argmax(decode_out["logits"]).reshape(1, 1),
            "position_ids": pos_id,
        }
        qpc_outputs.append(loop_decode_inputs["input_ids"][0][0])
        decode_out = decode_session.run(loop_decode_inputs)
        pos_id += 1

    print("QPC Outputs (AIC): \n")
    print("Prompt:", repr(prompt))
    print("Completion:", repr(tokenizer.decode(qpc_outputs)))


@pytest.mark.on_qaic
@pytest.mark.llm_model
@pytest.mark.parametrize("model_id", model_id_blocking)
@pytest.mark.parametrize("prompt", [prompt1])
def test_disagg_mode_prefix_caching(model_id, prompt):
    PREFILL_SEQ_LEN = 128
    CTX_LEN = 128 * 3
    config = AutoConfig.from_pretrained(model_id, num_hidden_layers=2)
    prefill_qeff_model = QEFFAutoModelForCausalLM.from_pretrained(
        model_id, num_hidden_layers=2, continuous_batching=True
    )
    prefill_qeff_model.prefill(enable=True, enable_chunking=True)
    prefill_qpc_path = prefill_qeff_model.compile(
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
        full_batch_size=1,
        kv_cache_batch_size=2,
    )

    decode_qeff_model = QEFFAutoModelForCausalLM.from_pretrained(
        model_id, num_hidden_layers=2, continuous_batching=True
    )
    decode_qeff_model.prefill(enable=False)
    decode_qpc_path = decode_qeff_model.compile(
        prefill_seq_len=1,
        ctx_len=CTX_LEN,
        num_cores=16,
        mxfp6_matmul=False,
        mxint8_kv_cache=False,
        num_devices=1,
        mos=1,
        aic_enable_depth_first=True,
        num_speculative_tokens=None,
        offload_pt_weights=False,  # Need the weights in memory for prefill-model export/compilation in the next step
        full_batch_size=1,
        kv_cache_batch_size=2,
        retain_full_kv=True,
    )

    out1, ids1 = prefix_caching_inference(model_id, prefill_qpc_path, decode_qpc_path, prompt, decode_batch_id=0)
    out2, ids2 = prefix_caching_inference(model_id, prefill_qpc_path, decode_qpc_path, prompt, decode_batch_id=1)

    for i in range(config.num_hidden_layers):
        assert (
            np.abs(
                out1[f"past_key.{i}_RetainedState"][0, :, :, :] - out2[f"past_key.{i}_RetainedState"][1, :, :, :]
            ).max()
            < 5e-2
        )
        assert (
            np.abs(
                out1[f"past_value.{i}_RetainedState"][0, :, :, :] - out2[f"past_value.{i}_RetainedState"][1, :, :, :]
            ).max()
            < 5e-2
        )


def prefix_caching_inference(model_id, prefill_qpc_path, decode_qpc_path, prompt, decode_batch_id):
    PREFILL_SEQ_LEN = 128
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    config = AutoConfig.from_pretrained(model_id, num_hidden_layers=2)
    inputs = tokenizer(prompt, return_tensors="np", padding=True)
    padded_len = inputs["input_ids"].shape[1]
    num_chunks = -(padded_len // -PREFILL_SEQ_LEN)  # ceil divide without float
    padded_len = num_chunks * PREFILL_SEQ_LEN  # Convert to a multiple of prompt_len

    inputs = tokenizer(prompt, return_tensors="np", padding="max_length", max_length=padded_len)
    inputs["position_ids"] = np.where(inputs.pop("attention_mask"), np.arange(padded_len), -1)
    inputs.pop("token_type_ids", None)
    inputs["batch_index"] = np.array([[decode_batch_id]], dtype=np.int64)

    prefill_session = QAICInferenceSession(prefill_qpc_path)
    logits_out_placeholder = np.zeros((1, 1, config.vocab_size), dtype=np.float32)
    prefill_session.set_buffers({"logits": logits_out_placeholder})
    for i in range(num_chunks):
        chunk_inputs = inputs.copy()
        chunk_inputs["input_ids"] = inputs["input_ids"][:, i * PREFILL_SEQ_LEN : (i + 1) * PREFILL_SEQ_LEN]
        chunk_inputs["position_ids"] = inputs["position_ids"][:, i * PREFILL_SEQ_LEN : (i + 1) * PREFILL_SEQ_LEN]
        qpc_out = prefill_session.run(chunk_inputs)
    del prefill_session

    qpc_outputs = []
    decode_inputs = {
        "input_ids": np.argmax(qpc_out["logits"]).reshape(1, 1),
        "position_ids": np.max(inputs["position_ids"]).reshape(1, 1) + 1,
        "batch_index": inputs["batch_index"],
    }
    qpc_outputs.append(decode_inputs["input_ids"][0][0])

    decode_session = QAICInferenceSession(decode_qpc_path)
    decode_session.set_buffers({"logits": logits_out_placeholder})
    generation_len = 5

    for i in range(config.num_hidden_layers):
        if i % 2 == 0 and decode_inputs["position_ids"] >= config.sliding_window:
            k = qpc_out[f"past_key.{i}_RetainedState"]
            v = qpc_out[f"past_value.{i}_RetainedState"]
            mod_pos_id = config.sliding_window - decode_inputs["position_ids"][0][0] % config.sliding_window
            decode_inputs[f"past_key.{i}"] = np.concatenate((k[:, :, mod_pos_id:, :], k[:, :, :mod_pos_id, :]), axis=-2)
            decode_inputs[f"past_value.{i}"] = np.concatenate(
                (v[:, :, mod_pos_id:, :], v[:, :, :mod_pos_id, :]), axis=-2
            )
        else:
            decode_inputs[f"past_key.{i}"] = qpc_out[f"past_key.{i}_RetainedState"]
            decode_inputs[f"past_value.{i}"] = qpc_out[f"past_value.{i}_RetainedState"]

    decode_out = decode_session.run(decode_inputs)
    pos_id = np.max(decode_inputs["position_ids"]).reshape(1, 1) + 1
    for i in range(generation_len - 1):
        loop_decode_inputs = {
            "input_ids": np.argmax(decode_out["logits"]).reshape(1, 1),
            "position_ids": pos_id,
            "batch_index": inputs["batch_index"],
        }
        for i in range(config.num_hidden_layers):
            loop_decode_inputs[f"past_key.{i}"] = decode_out[f"past_key.{i}_RetainedState"]
            loop_decode_inputs[f"past_value.{i}"] = decode_out[f"past_value.{i}_RetainedState"]
        qpc_outputs.append(loop_decode_inputs["input_ids"][0][0])
        decode_out = decode_session.run(loop_decode_inputs)
        pos_id += 1

    print("QPC Outputs (AIC): \n")
    print("Prompt:", repr(prompt))
    print("Completion:", repr(tokenizer.decode(qpc_outputs)))
    return qpc_out, qpc_outputs

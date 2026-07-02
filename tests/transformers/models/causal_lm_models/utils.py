# -----------------------------------------------------------------------------
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------

import time

import numpy as np
import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
from transformers.cache_utils import DynamicCache

from QEfficient import QEFFAutoModelForCausalLM
from QEfficient.generation.cloud_infer import QAICInferenceSession
from QEfficient.transformers.quantizers import replace_transformers_quantizers, undo_transformers_quantizers


def _make_model(model_id: str) -> AutoModelForCausalLM:
    """Create a tiny model from a dummy config — no weight download required.

    A fixed seed ensures the weights are reproducible across test runs so that
    the QAIC-compiled model (which may be cached on disk) always matches the
    in-process PyTorch model used for reference comparisons.

    Weights are scaled to std≈0.02 (matching real transformer init) so that
    intermediate activations stay small and float16 rounding errors on QAIC
    remain within the 5e-2 tolerance used for logit accuracy checks.
    """
    torch.manual_seed(42)
    model = AutoModelForCausalLM.from_pretrained(model_id, attn_implementation="eager", torch_dtype=torch.float32)
    with torch.no_grad():
        for param in model.parameters():
            param.mul_(0.02)
    return model


def _prepare_inputs(model_id, prompt, prefill_seq_len):
    """Tokenize prompt for both HF reference and QAIC-formatted inputs."""
    tokenizer = AutoTokenizer.from_pretrained(model_id)

    _raw_padded = tokenizer(prompt, return_tensors="np", padding=True)
    padded_len = _raw_padded["input_ids"].shape[1]
    num_chunks = -(padded_len // -prefill_seq_len)  # ceil divide
    padded_len = num_chunks * prefill_seq_len

    raw_inputs = tokenizer(prompt, return_tensors="np", padding="max_length", max_length=padded_len)
    raw_inputs["position_ids"] = np.where(raw_inputs.pop("attention_mask"), np.arange(padded_len), -1)
    raw_inputs.pop("token_type_ids", None)

    hf_inputs = tokenizer(prompt, return_tensors="pt")
    qeff_inputs = {k: torch.from_numpy(v) for k, v in raw_inputs.items()}
    return tokenizer, hf_inputs, qeff_inputs, raw_inputs, num_chunks


def _make_kv_cache(config, prefill_seq_len, ctx_len, sliding_window=False):
    """Build zeroed past_key_values list for QEff model forward pass."""
    head_dim = getattr(config, "head_dim", config.hidden_size // config.num_attention_heads)
    past_key_values = []
    for i in range(config.num_hidden_layers):
        cache_len = (config.sliding_window if i % 2 == 0 else prefill_seq_len) if sliding_window else ctx_len
        shape = (1, config.num_key_value_heads, cache_len, head_dim)
        past_key_values.append((torch.zeros(shape, dtype=torch.float32), torch.zeros(shape, dtype=torch.float32)))
    return past_key_values


def _default_compile_kwargs(prefill_seq_len, ctx_len, num_cores, **overrides):
    """Return compile kwargs with defaults, allowing per-call overrides."""
    kwargs = dict(
        prefill_seq_len=prefill_seq_len,
        ctx_len=ctx_len,
        num_cores=num_cores,
        mxfp6_matmul=False,
        mxint8_kv_cache=False,
        num_devices=1,
        mos=1,
        aic_enable_depth_first=True,
        num_speculative_tokens=None,
    )
    kwargs.update(overrides)
    return kwargs


def _run_hf_prefill(model_id, hf_inputs):
    """Run a single forward pass through the HF model and return (model, config, out)."""
    replace_transformers_quantizers()
    model = _make_model(model_id)
    config = model.config
    out = model(**hf_inputs, past_key_values=DynamicCache(config=config))
    undo_transformers_quantizers()
    return model, config, out


def _run_qeff_prefill(
    model, qeff_inputs, prefill_seq_len, ctx_len,
    *,
    sliding_window,                  # use alternating sliding-window cache lengths (gpt_oss)
    enable_chunking=False,           # feed the sequence in prefill_seq_len-sized chunks sequentially
    num_chunks=None,
):
    """Apply disaggregated-prefill transforms and return (qeff_model, config, qeff_out)."""

    qeff_model = QEFFAutoModelForCausalLM(model)
    qeff_model.prefill(enable=True, enable_chunking=enable_chunking)
    config = qeff_model.model.config
    qeff_inputs["past_key_values"] = _make_kv_cache(config, prefill_seq_len, ctx_len, sliding_window=sliding_window)

    if enable_chunking:
        num_chunks = 1 if num_chunks is None else num_chunks
        for i in range(num_chunks):
            chunk_inputs = qeff_inputs.copy()
            chunk_inputs["input_ids"] = qeff_inputs["input_ids"][:, i * prefill_seq_len : (i + 1) * prefill_seq_len]
            chunk_inputs["position_ids"] = qeff_inputs["position_ids"][:, i * prefill_seq_len : (i + 1) * prefill_seq_len]
            qeff_out = qeff_model.model(**chunk_inputs)
            qeff_inputs["past_key_values"] = qeff_out["past_key_values"]
    else:
        qeff_out = qeff_model.model(**qeff_inputs)
    return qeff_model, config, qeff_out


def _run_qpc_prefill(
    qeff_inputs, prefill_qpc_path, config, prefill_seq_len,
    enable_chunking=False,           # feed the sequence in prefill_seq_len-sized chunks sequentially
    num_chunks=None,
    skip_past_buffers=False,         # skip host-side past_ I/O so the device retains KV state across chunks
):
    """Run a compiled prefill QPC on QAIC and return (qpc_out, qeff_inputs_np)."""    

    if isinstance(next(iter(qeff_inputs.values())), torch.Tensor):
        qeff_inputs.pop("past_key_values", None)
        qeff_inputs_np = {k: v.detach().numpy() for k, v in qeff_inputs.items()}
    else:
        qeff_inputs_np = {k: v for k, v in qeff_inputs.items()}
    
    session = QAICInferenceSession(prefill_qpc_path)
    logits_out_placeholder = np.zeros((1, 1, config.vocab_size), dtype=np.float32)
    session.set_buffers({"logits": logits_out_placeholder})

    if enable_chunking:
        if skip_past_buffers:
            session.skip_buffers(
                [x for x in session.input_names + session.output_names if x.startswith("past_")]
            )
    session.set_buffers({"logits": np.zeros((1, 1, config.vocab_size), dtype=np.float32)})

    t0 = time.time()
    if enable_chunking:
        num_chunks = 1 if num_chunks is None else num_chunks
        for i in range(num_chunks):
            chunk_inputs = qeff_inputs_np.copy()
            chunk_inputs["input_ids"] = qeff_inputs_np["input_ids"][:, i * prefill_seq_len : (i + 1) * prefill_seq_len]
            chunk_inputs["position_ids"] = qeff_inputs_np["position_ids"][:, i * prefill_seq_len : (i + 1) * prefill_seq_len]
            qpc_out = session.run(chunk_inputs)
    else:
        qpc_out = session.run(qeff_inputs_np)
    print(f"time for prefill_run={time.time() - t0} sec\n")
    del session
    return qpc_out, qeff_inputs_np


def _rotate_sliding_kv(qpc_out, decode_inputs, config):
    """Copy prefill KV outputs into decode_inputs, rotating sliding-window layers.

    Even-indexed layers use sliding-window attention; their cache must be rotated
    so that the oldest entries are at the front when the position exceeds the window.
    """
    for i in range(config.num_hidden_layers):
        if i % 2 == 0 and decode_inputs["position_ids"] >= config.sliding_window:
            k = qpc_out[f"past_key.{i}_RetainedState"]
            v = qpc_out[f"past_value.{i}_RetainedState"]
            rot_offset = config.sliding_window - decode_inputs["position_ids"][0][0] % config.sliding_window
            decode_inputs[f"past_key.{i}"] = np.concatenate((k[:, :, rot_offset:, :], k[:, :, :rot_offset, :]), axis=-2)
            decode_inputs[f"past_value.{i}"] = np.concatenate((v[:, :, rot_offset:, :], v[:, :, :rot_offset, :]), axis=-2)
        else:
            decode_inputs[f"past_key.{i}"] = qpc_out[f"past_key.{i}_RetainedState"]
            decode_inputs[f"past_value.{i}"] = qpc_out[f"past_value.{i}_RetainedState"]


def _prefix_caching_inference(
    model_id, config, prefill_qpc_path, decode_qpc_path, prompt,
    decode_batch_id,   # which slot in the decode KV cache batch to write into
    prefill_seq_len,
):
    generation_len = 5
    tokenizer, _, _, inputs, num_chunks = _prepare_inputs(model_id, prompt, prefill_seq_len)
    inputs["batch_index"] = np.array([[decode_batch_id]], dtype=np.int64)

    # QPC prefill run on QAIC
    qpc_out, inputs = _run_qpc_prefill(inputs, prefill_qpc_path, config, prefill_seq_len, enable_chunking=True, num_chunks=num_chunks)

    decode_inputs = {
        "input_ids": np.argmax(qpc_out["logits"]).reshape(1, 1),
        "position_ids": np.max(inputs["position_ids"]).reshape(1, 1) + 1,
        "batch_index": inputs["batch_index"],
    }
    qpc_outputs = [decode_inputs["input_ids"][0][0]]
    _rotate_sliding_kv(qpc_out, decode_inputs, config)

    # QPC decode run on QAIC
    decode_session = QAICInferenceSession(decode_qpc_path)
    decode_session.set_buffers({"logits": np.zeros((1, 1, config.vocab_size), dtype=np.float32)})
    decode_out = decode_session.run(decode_inputs)
    pos_id = np.max(decode_inputs["position_ids"]).reshape(1, 1) + 1

    for _ in range(generation_len - 1):
        loop_inputs = {
            "input_ids": np.argmax(decode_out["logits"]).reshape(1, 1),
            "position_ids": pos_id,
            "batch_index": inputs["batch_index"],
        }
        for j in range(config.num_hidden_layers):
            loop_inputs[f"past_key.{j}"] = decode_out[f"past_key.{j}_RetainedState"]
            loop_inputs[f"past_value.{j}"] = decode_out[f"past_value.{j}_RetainedState"]
        qpc_outputs.append(loop_inputs["input_ids"][0][0])
        decode_out = decode_session.run(loop_inputs)
        pos_id += 1

    print("QPC Outputs (AIC): \n")
    print("Prompt:", repr(prompt))
    print("Completion:", repr(tokenizer.decode(qpc_outputs)))
    return qpc_out, qpc_outputs

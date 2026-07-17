# -----------------------------------------------------------------------------
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------


import numpy as np
import pytest
import torch
from transformers import AutoConfig
from transformers.cache_utils import DynamicCache

from QEfficient import QEFFAutoModelForCausalLM
from QEfficient.generation.cloud_infer import QAICInferenceSession
from QEfficient.transformers.quantizers import replace_transformers_quantizers, undo_transformers_quantizers

from .utils import (
    _default_compile_kwargs,
    _make_model,
    _prefix_caching_inference,
    _prepare_inputs,
    _rotate_sliding_kv,
    _run_hf_prefill,
    _run_qeff_prefill,
    _run_qpc_prefill,
)

test_models_blocking_dict = {"openai/gpt-oss-20b": "tiny-random/gpt-oss-bf16"}
test_models_chunking_dict = {"Qwen/Qwen3-30B-A3B-Instruct-2507": "hf-internal-testing/tiny-random-Qwen3MoeForCausalLM"}


# if os.environ.get("QEFF_TEST_PROFILE", "").strip().lower() == "tiny_model":
model_id_blocking = list(test_models_blocking_dict.values())
model_id_chunking = list(test_models_chunking_dict.values())
# else:
#     model_id_blocking = list(test_models_blocking_dict.keys())
#     model_id_chunking = list(test_models_chunking_dict.keys())


prompt2 = """
Once upon a time, in a small town, there lived a young boy named Alex. Alex was a curious and adventurous child, always eager to explore the world around him. One day, while playing in the park, Alex stumbled upon a mysterious old book hidden beneath a pile of leaves. The book was filled with stories of distant lands, magical creatures, and extraordinary adventures.

As Alex flipped through the pages, he discovered a map that led to a hidden treasure. Excited by the prospect of a real-life treasure hunt, Alex decided to embark on a thrilling journey. He packed his backpack with snacks, a flashlight, and a compass, and set off into the unknown.

The path to the treasure was not an easy one. Alex had to navigate through dense forests, cross rickety bridges, and solve riddles that guarded the treasure's location.
"""
prompt1 = "Once upon a time"
prompts = [prompt1, prompt2]

TORCH_ATOL = 1e-4
QAIC_ATOL = 5e-2


@pytest.mark.qaic
@pytest.mark.llm  # FIXME split into llm and vllm later
@pytest.mark.parametrize("model_id", model_id_blocking)
@pytest.mark.parametrize("prompt", prompts)
def test_disagg_mode_prefill(model_id, prompt):
    PREFILL_SEQ_LEN = 256
    CTX_LEN = 256
    _, hf_inputs, qeff_inputs, _, _ = _prepare_inputs(model_id, prompt, PREFILL_SEQ_LEN)

    # HF run
    model, config, out = _run_hf_prefill(model_id, hf_inputs)

    # QEff PyTorch run
    qeff_model, config, qeff_out = _run_qeff_prefill(model, qeff_inputs, PREFILL_SEQ_LEN, CTX_LEN, sliding_window=True)
    assert (qeff_out.logits - out.logits[:, -1, :]).abs().max() < TORCH_ATOL

    # QPC run on QAIC
    prefill_qpc_path = qeff_model.compile(**_default_compile_kwargs(PREFILL_SEQ_LEN, CTX_LEN, 16, prefill_only=True))
    qpc_out, _ = _run_qpc_prefill(qeff_inputs, prefill_qpc_path, config, PREFILL_SEQ_LEN)
    assert (torch.from_numpy(qpc_out["logits"]) - qeff_out.logits).abs().max() < QAIC_ATOL


@pytest.mark.qaic
@pytest.mark.llm  # FIXME split into llm and vllm later
@pytest.mark.parametrize("model_id", model_id_chunking)
@pytest.mark.parametrize("prompt", prompts)
def test_disagg_mode_prefill_chunked(model_id, prompt):
    PREFILL_SEQ_LEN = 128
    CTX_LEN = 128 * 3
    _, hf_inputs, qeff_inputs, _, num_chunks = _prepare_inputs(model_id, prompt, PREFILL_SEQ_LEN)

    # HF run
    model, config, out = _run_hf_prefill(model_id, hf_inputs)

    # QEff PyTorch run (chunked)
    qeff_model, config, qeff_out = _run_qeff_prefill(
        model, qeff_inputs, PREFILL_SEQ_LEN, CTX_LEN, sliding_window=False, enable_chunking=True, num_chunks=num_chunks
    )
    assert (qeff_out.logits - out.logits[:, -1, :]).abs().max() < TORCH_ATOL

    # QPC run on QAIC (chunked)
    prefill_qpc_path = qeff_model.compile(
        **_default_compile_kwargs(
            PREFILL_SEQ_LEN,
            CTX_LEN,
            config.num_experts,
            prefill_only=True,
            enable_chunking=True,
        )
    )
    qpc_out, _ = _run_qpc_prefill(
        qeff_inputs,
        prefill_qpc_path,
        config,
        PREFILL_SEQ_LEN,
        enable_chunking=True,
        num_chunks=num_chunks,
        skip_past_buffers=True,
    )
    assert (torch.from_numpy(qpc_out["logits"]) - qeff_out.logits).abs().max() < QAIC_ATOL


@pytest.mark.qaic
@pytest.mark.llm  # FIXME split into llm and vllm later
@pytest.mark.parametrize("model_id", model_id_blocking)
@pytest.mark.parametrize("prompt", [prompt1])
def test_disagg_mode_prefill_only_and_decode_only(model_id, prompt):
    PREFILL_SEQ_LEN = 256
    CTX_LEN = 256
    generation_len = 10
    tokenizer, hf_inputs, qeff_inputs, _, _ = _prepare_inputs(model_id, prompt, PREFILL_SEQ_LEN)

    # HF prefill run
    replace_transformers_quantizers()
    model = _make_model(model_id)
    config = model.config
    orig_out = model(**hf_inputs, past_key_values=DynamicCache(config=config))

    # HF decode loop
    position_ids = qeff_inputs["position_ids"]
    generated_ids = []
    out = orig_out
    for _ in range(1, generation_len):
        next_token_id = out["logits"][:, -1, :].argmax(-1).reshape(-1, 1)
        generated_ids.append(next_token_id)
        position_ids = position_ids.max(1, keepdim=True).values + 1
        out = model(input_ids=next_token_id, position_ids=position_ids, past_key_values=out["past_key_values"])
    generated_ids.append(out["logits"][:, -1, :].argmax(-1).reshape(-1, 1))
    generated_ids = np.concatenate(generated_ids, axis=1)
    print("Original HF Model Outputs (Torch CPU): \n")
    print("Prompt:", repr(prompt))
    print("Completion:", repr(tokenizer.batch_decode(generated_ids, skip_special_tokens=True)))
    undo_transformers_quantizers()

    # QEff PyTorch prefill
    prefill_qeff_model, config, prefill_qeff_out = _run_qeff_prefill(
        model, qeff_inputs, PREFILL_SEQ_LEN, CTX_LEN, sliding_window=True
    )
    assert (prefill_qeff_out.logits - orig_out.logits[:, -1, :]).abs().max() < TORCH_ATOL

    # QEff PyTorch decode loop
    decode_qeff_model = QEFFAutoModelForCausalLM(model)
    decode_qeff_model.prefill(enable=False)
    qeff_out = prefill_qeff_out
    position_ids = qeff_inputs["position_ids"]
    qeff_generated_ids = []
    for _ in range(1, generation_len):
        next_token_id = qeff_out["logits"][:, -1, :].argmax(-1).reshape(-1, 1)
        qeff_generated_ids.append(next_token_id)
        position_ids = position_ids.max(1, keepdim=True).values + 1
        qeff_out = decode_qeff_model.model(
            input_ids=next_token_id, position_ids=position_ids, past_key_values=qeff_out["past_key_values"]
        )
    qeff_generated_ids.append(qeff_out["logits"][:, -1, :].argmax(-1).reshape(-1, 1))
    qeff_generated_ids = np.concatenate(qeff_generated_ids, axis=1)
    print("QEFF Transformed Model Outputs (Torch CPU): \n")
    print("Prompt:", repr(prompt))
    print("Completion:", repr(tokenizer.batch_decode(qeff_generated_ids, skip_special_tokens=True)))
    assert (qeff_generated_ids == generated_ids).all()

    # QPC Compilation
    decode_qpc_path = decode_qeff_model.compile(
        **_default_compile_kwargs(
            1,
            CTX_LEN,
            16,
            offload_pt_weights=False,  # weights must stay in memory for prefill compile
        )
    )
    prefill_qpc_path = prefill_qeff_model.compile(
        **_default_compile_kwargs(PREFILL_SEQ_LEN, CTX_LEN, 16, prefill_only=True)
    )

    # QPC prefill run on QAIC
    qpc_out, qeff_inputs_np = _run_qpc_prefill(qeff_inputs, prefill_qpc_path, config, PREFILL_SEQ_LEN)
    assert (torch.from_numpy(qpc_out["logits"]) - prefill_qeff_out.logits).abs().max() < QAIC_ATOL

    # QPC decode run on QAIC
    decode_session = QAICInferenceSession(decode_qpc_path)
    decode_session.set_buffers({"logits": np.zeros((1, 1, config.vocab_size), dtype=np.float32)})
    decode_inputs = {
        "input_ids": np.argmax(qpc_out["logits"]).reshape(1, 1),
        "position_ids": np.max(qeff_inputs_np["position_ids"]).reshape(1, 1) + 1,
    }
    qpc_outputs = [decode_inputs["input_ids"][0][0]]
    _rotate_sliding_kv(qpc_out, decode_inputs, config)
    decode_out = decode_session.run(decode_inputs)
    decode_session.skip_buffers(
        [x for x in decode_session.input_names + decode_session.output_names if x.startswith("past_")]
    )
    pos_id = np.max(decode_inputs["position_ids"]).reshape(1, 1) + 1
    for _ in range(generation_len - 1):
        loop_inputs = {
            "input_ids": np.argmax(decode_out["logits"]).reshape(1, 1),
            "position_ids": pos_id,
        }
        qpc_outputs.append(loop_inputs["input_ids"][0][0])
        decode_out = decode_session.run(loop_inputs)
        pos_id += 1
    print("QPC Outputs (AIC): \n")
    print("Prompt:", repr(prompt))
    print("Completion:", repr(tokenizer.decode(qpc_outputs)))


@pytest.mark.qaic
@pytest.mark.llm  # FIXME split into llm and vllm later
@pytest.mark.parametrize("model_id", model_id_blocking)
@pytest.mark.parametrize("prompt", [prompt1])
def test_disagg_mode_prefix_caching(model_id, prompt):
    PREFILL_SEQ_LEN = 128
    CTX_LEN = 128 * 3
    config = AutoConfig.from_pretrained(model_id)

    # QPC prefill compile
    prefill_qeff_model = QEFFAutoModelForCausalLM.from_pretrained(model_id, continuous_batching=True)
    prefill_qeff_model.prefill(enable=True, enable_chunking=True)
    prefill_qpc_path = prefill_qeff_model.compile(
        **_default_compile_kwargs(
            PREFILL_SEQ_LEN,
            CTX_LEN,
            16,
            prefill_only=True,
            enable_chunking=True,
            full_batch_size=1,
            kv_cache_batch_size=2,
        )
    )

    # QPC decode compile
    decode_qeff_model = QEFFAutoModelForCausalLM.from_pretrained(model_id, continuous_batching=True)
    decode_qeff_model.prefill(enable=False)
    decode_qpc_path = decode_qeff_model.compile(
        **_default_compile_kwargs(
            1,
            CTX_LEN,
            16,
            offload_pt_weights=False,  # weights must stay in memory for prefill compile
            full_batch_size=1,
            kv_cache_batch_size=2,
            retain_full_kv=True,
        )
    )

    # QPC runs on QAIC for two different batch slots with the same prompt
    kv_out_batch0, _ = _prefix_caching_inference(
        model_id, config, prefill_qpc_path, decode_qpc_path, prompt, decode_batch_id=0, prefill_seq_len=PREFILL_SEQ_LEN
    )
    kv_out_batch1, _ = _prefix_caching_inference(
        model_id, config, prefill_qpc_path, decode_qpc_path, prompt, decode_batch_id=1, prefill_seq_len=PREFILL_SEQ_LEN
    )
    for i in range(config.num_hidden_layers):
        assert (
            np.abs(
                kv_out_batch0[f"past_key.{i}_RetainedState"][0] - kv_out_batch1[f"past_key.{i}_RetainedState"][1]
            ).max()
            < QAIC_ATOL
        )
        assert (
            np.abs(
                kv_out_batch0[f"past_value.{i}_RetainedState"][0] - kv_out_batch1[f"past_value.{i}_RetainedState"][1]
            ).max()
            < QAIC_ATOL
        )

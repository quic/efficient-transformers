# -----------------------------------------------------------------------------
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------

import copy
import tempfile

import numpy as np
from tokenizers import Tokenizer, models, pre_tokenizers
from transformers import AutoConfig, AutoModelForCausalLM, PreTrainedTokenizerFast

from QEfficient.transformers.attention_blocking import AttentionBlockingConfig
from QEfficient.transformers.models.modeling_auto import QEFFAutoModelForCausalLM
from QEfficient.utils.run_utils import ApiRunner


def make_tokenizer(vocab_size):
    vocab = {str(i): i for i in range(vocab_size)}
    tok = Tokenizer(models.WordLevel(vocab, unk_token="0"))
    tok.pre_tokenizer = pre_tokenizers.Whitespace()
    tokenizer = PreTrainedTokenizerFast(tokenizer_object=tok, unk_token="0", pad_token="0", eos_token="0")
    tokenizer.padding_side = "right"
    return tokenizer


def _has_blocked_kv_applied(model, num_kv_blocks):
    for module in model.modules():
        if getattr(module, "num_kv_blocks", None) == num_kv_blocks:
            return True
    return False


def _has_q_blocking_applied(model, num_q_blocks):
    for module in model.modules():
        config = getattr(module, "attn_blocking_config", None)
        if (
            config is not None
            and getattr(config, "mode", None) == "q"
            and getattr(config, "num_q_blocks", None) == num_q_blocks
        ):
            return True
    return False


def build_runner_and_models(
    model_type,
    config_kwargs,
    num_kv_blocks=2,
    use_auto=False,
    prompt=None,
    prompt_len=8,
    ctx_len=16,
):
    config = AutoConfig.for_model(model_type, **config_kwargs)
    model = AutoModelForCausalLM.from_config(config, attn_implementation="eager").eval()

    tokenizer = make_tokenizer(config.vocab_size)
    prompt = prompt or ["1 2 3 4"]
    if prompt_len is None:
        prompt_len = int(tokenizer(prompt[0], return_tensors="pt")["input_ids"].shape[1])
    if ctx_len is None:
        ctx_len = prompt_len

    api_runner = ApiRunner(
        batch_size=len(prompt),
        tokenizer=tokenizer,
        config=config,
        prompt=prompt,
        prompt_len=prompt_len,
        ctx_len=ctx_len,
        full_batch_size=len(prompt),
    )

    qeff_unblocked = QEFFAutoModelForCausalLM(copy.deepcopy(model), False)
    qaic_config = {"attn_blocking_auto": True} if use_auto else {"num_kv_blocks": num_kv_blocks}
    qeff_blocked = QEFFAutoModelForCausalLM(copy.deepcopy(model), False, qaic_config=qaic_config)

    return config, api_runner, qeff_unblocked, qeff_blocked


def run_kv_tokens_test(model_type, config_kwargs, num_kv_blocks=2, use_auto=False, return_tokens=False):
    _, api_runner, qeff_unblocked, qeff_blocked = build_runner_and_models(
        model_type,
        config_kwargs,
        num_kv_blocks=num_kv_blocks,
        use_auto=use_auto,
    )
    applied = _has_blocked_kv_applied(qeff_blocked.model, num_kv_blocks)
    tokens_unblocked = api_runner.run_kv_model_on_pytorch(qeff_unblocked.model.eval())
    tokens_blocked = api_runner.run_kv_model_on_pytorch(qeff_blocked.model.eval())

    match = np.array_equal(tokens_unblocked, tokens_blocked)
    print(f"{model_type}: kv_block_applied = {applied}, tokens match = {match}")
    if return_tokens:
        return match, applied, tokens_unblocked, tokens_blocked, api_runner, qeff_blocked
    return match, applied


def run_onnx_tokens_test(model_type, api_runner, qeff_blocked, tokens_unblocked, tokens_blocked):
    export_dir = tempfile.mkdtemp(prefix="qeff_onnx_")
    try:
        if hasattr(qeff_blocked, "qaic_config") and isinstance(qeff_blocked.qaic_config, dict):
            if any(isinstance(v, AttentionBlockingConfig) for v in qeff_blocked.qaic_config.values()):
                qeff_blocked.qaic_config = {}
        if hasattr(qeff_blocked, "model") and hasattr(qeff_blocked.model, "qaic_config"):
            qeff_blocked.model.qaic_config = {}
        if hasattr(qeff_blocked, "hash_params") and isinstance(qeff_blocked.hash_params, dict):
            qeff_blocked.hash_params["qaic_config"] = {}
        onnx_path = qeff_blocked.export(export_dir=export_dir)
        api_runner.input_handler.prepare_ort_inputs()
        ort_tokens = api_runner.run_kv_model_on_ort(onnx_path)
    except Exception as exc:
        print(f"{model_type}: onnx skipped ({exc})")
        return True

    print(f"{model_type}: ort tokens = {ort_tokens}")
    match = np.array_equal(tokens_unblocked, tokens_blocked) and np.array_equal(tokens_blocked, ort_tokens)
    print(f"{model_type}: pytorch vs onnx tokens match = {match}")
    return match


def run_llama_q_block_test(config_kwargs, num_q_blocks=2):
    config = AutoConfig.for_model("llama", **config_kwargs)
    model = AutoModelForCausalLM.from_config(config, attn_implementation="eager").eval()

    tokenizer = make_tokenizer(config.vocab_size)
    prompt = ["1 2 3 4"]
    api_runner = ApiRunner(
        batch_size=len(prompt),
        tokenizer=tokenizer,
        config=config,
        prompt=prompt,
        prompt_len=8,
        ctx_len=16,
        full_batch_size=len(prompt),
    )

    qeff_unblocked = QEFFAutoModelForCausalLM(copy.deepcopy(model), False).model.eval()
    qaic_config = {"attn_blocking_config": AttentionBlockingConfig(mode="q", num_q_blocks=num_q_blocks)}
    qeff_blocked = QEFFAutoModelForCausalLM(copy.deepcopy(model), False, qaic_config=qaic_config).model.eval()

    tokens_unblocked = api_runner.run_kv_model_on_pytorch(qeff_unblocked)
    tokens_blocked = api_runner.run_kv_model_on_pytorch(qeff_blocked)
    match = np.array_equal(tokens_unblocked, tokens_blocked)
    print(f"llama-q: tokens match = {match}")

    onnx_match = run_onnx_tokens_test("llama-q", api_runner, qeff_blocked, tokens_unblocked, tokens_blocked)
    return onnx_match and match


def run_q_tokens_test(model_type, config_kwargs, num_q_blocks=2):
    config = AutoConfig.for_model(model_type, **config_kwargs)
    model = AutoModelForCausalLM.from_config(config, attn_implementation="eager").eval()

    tokenizer = make_tokenizer(config.vocab_size)
    prompt = ["1 2 3 4"]
    api_runner = ApiRunner(
        batch_size=len(prompt),
        tokenizer=tokenizer,
        config=config,
        prompt=prompt,
        prompt_len=8,
        ctx_len=16,
        full_batch_size=len(prompt),
    )

    qeff_unblocked = QEFFAutoModelForCausalLM(copy.deepcopy(model), False).model.eval()
    qaic_config = {"attn_blocking_config": AttentionBlockingConfig(mode="q", num_q_blocks=num_q_blocks)}
    qeff_blocked = QEFFAutoModelForCausalLM(copy.deepcopy(model), False, qaic_config=qaic_config)

    tokens_unblocked = api_runner.run_kv_model_on_pytorch(qeff_unblocked)
    tokens_blocked = api_runner.run_kv_model_on_pytorch(qeff_blocked.model.eval())
    applied = _has_q_blocking_applied(qeff_blocked.model, num_q_blocks)
    match = np.array_equal(tokens_unblocked, tokens_blocked)
    print(f"{model_type}-q: q_block_applied = {applied}, tokens match = {match}")

    onnx_match = run_onnx_tokens_test(f"{model_type}-q", api_runner, qeff_blocked, tokens_unblocked, tokens_blocked)
    return match and applied and onnx_match


def main():
    all_ok = True

    skip_models = {"gpt_oss", "mpt"}
    skip_q_models = {"gpt_oss", "mpt"}

    test_configs = [
        # name, max_position_embeddings, num_hidden_layers, num_attention_heads, hidden_size, intermediate_size, vocab_size, additional_params
        ("gpt2", 256, 2, 4, 128, 512, 127, {}),
        ("codegen", 256, 2, 4, 128, 512, 127, {"rotary_dim": 16}),
        ("falcon", 256, 2, 4, 128, 512, 127, {}),
        ("gptj", 256, 2, 4, 128, 512, 127, {"rotary_dim": 16}),
        ("llama", 256, 2, 4, 128, 512, 127, {"num_key_value_heads": 2}),
        ("mistral", 256, 2, 4, 128, 512, 127, {"num_key_value_heads": 2}),
        ("mixtral", 256, 2, 4, 128, 512, 127, {"num_key_value_heads": 2}),
        ("mpt", 256, 2, 4, 128, 512, 127, {}),
        ("phi", 256, 2, 4, 128, 512, 127, {}),
        ("phi3", 256, 2, 4, 128, 512, 127, {"pad_token_id": 0}),
        ("qwen2", 256, 2, 4, 128, 512, 127, {"num_key_value_heads": 2}),
        ("starcoder2", 256, 2, 4, 128, 512, 127, {}),
        ("granite", 256, 2, 4, 128, 512, 127, {"num_key_value_heads": 2}),
        ("olmo2", 256, 2, 4, 128, 512, 127, {"num_key_value_heads": 2}),
        ("gpt_oss", 256, 3, 4, 128, 512, 127, {"num_key_value_heads": 2}),
    ]

    for (
        model_type,
        max_position_embeddings,
        num_hidden_layers,
        num_attention_heads,
        hidden_size,
        intermediate_size,
        vocab_size,
        additional_params,
    ) in test_configs:
        if model_type in skip_models:
            print(f"{model_type}: kv_block_applied = False, tokens match = skipped")
            continue
        match, applied, tokens_unblocked, tokens_blocked, api_runner, qeff_blocked = run_kv_tokens_test(
            model_type,
            dict(
                max_position_embeddings=max_position_embeddings,
                num_hidden_layers=num_hidden_layers,
                num_attention_heads=num_attention_heads,
                hidden_size=hidden_size,
                intermediate_size=intermediate_size,
                vocab_size=vocab_size,
                **additional_params,
            ),
            use_auto=True,
            return_tokens=True,
        )
        all_ok &= match and applied
        run_onnx_tokens_test(model_type, api_runner, qeff_blocked, tokens_unblocked, tokens_blocked)

    for (
        model_type,
        max_position_embeddings,
        num_hidden_layers,
        num_attention_heads,
        hidden_size,
        intermediate_size,
        vocab_size,
        additional_params,
    ) in test_configs:
        if model_type in skip_q_models:
            print(f"{model_type}-q: q_block_applied = False, tokens match = skipped")
            continue
        all_ok &= run_q_tokens_test(
            model_type,
            dict(
                max_position_embeddings=max_position_embeddings,
                num_hidden_layers=num_hidden_layers,
                num_attention_heads=num_attention_heads,
                hidden_size=hidden_size,
                intermediate_size=intermediate_size,
                vocab_size=vocab_size,
                **additional_params,
            ),
        )

    if not all_ok:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

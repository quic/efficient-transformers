# -----------------------------------------------------------------------------
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------
import copy
import os
from typing import Optional

import numpy as np
import torch
from transformers import AutoConfig, AutoTokenizer

from QEfficient.generation.text_generation_inference import TextGeneration
from QEfficient.transformers.models.modeling_auto import QEFFAutoModelForCausalLM
from QEfficient.transformers.quantizers.auto import replace_transformers_quantizers
from QEfficient.utils._utils import load_hf_tokenizer
from QEfficient.utils.config_utils import get_first_config_value
from QEfficient.utils.constants import ATTENTION_HEAD_CONFIG_KEYS, KV_HEAD_CONFIG_KEYS, Constants
from QEfficient.utils.run_utils import ApiRunner
from QEfficient.utils.test_utils import ModelConfig, load_hf_causal_lm_model, load_qeff_causal_lm_model


def get_custom_n_layers(model_name):
    """
    Function to set number layers of the variuos types of models such as swiftkv models and others
    --------

    :model_name: str

    :return n_layer
    """
    if model_name in {"microsoft/Phi-3-mini-4k-instruct", "neuralmagic/Qwen2-0.5B-Instruct-FP8", "openai/gpt-oss-20b"}:
        return 2
    elif model_name in ModelConfig.SWIFTKV_MODELS:
        return -1
    return 1


def check_kv_repeat_causal_lm_pytorch_vs_ai100(
    model_name: str,
    manual_cleanup: callable,
    prompt_len: int = Constants.PROMPT_LEN,
    ctx_len: int = Constants.CTX_LEN,
    n_layer: int = -1,
    config: Optional[AutoConfig] = None,
):
    """
    Validate causal LM flow with repeated KV heads configuration.
    """
    if config is None:
        model_config = AutoConfig.from_pretrained(
            model_name,
            trust_remote_code=model_name in ModelConfig.EXTERNAL_MODELS,
        )
    else:
        model_config = config

    num_attention_heads = get_first_config_value(model_config, ATTENTION_HEAD_CONFIG_KEYS, default=1, cast_int=True)
    num_key_value_heads = get_first_config_value(model_config, KV_HEAD_CONFIG_KEYS, default=None, cast_int=True)
    if num_key_value_heads is None:
        num_key_value_heads = num_attention_heads
    if num_attention_heads < 1 or num_key_value_heads < 1:
        raise ValueError(
            f"Invalid heads in config for RepeatKV: "
            f"num_attention_heads={num_attention_heads}, num_key_value_heads={num_key_value_heads}"
        )
    if num_attention_heads % num_key_value_heads != 0:
        raise ValueError(
            f"Invalid heads in config for RepeatKV: num_attention_heads ({num_attention_heads}) "
            f"is not divisible by num_key_value_heads ({num_key_value_heads})."
        )
    num_replicate_kv_heads = num_attention_heads // num_key_value_heads

    check_causal_lm_pytorch_vs_kv_vs_ort_vs_ai100(
        model_name=model_name,
        manual_cleanup=manual_cleanup,
        prompt_len=prompt_len,
        ctx_len=ctx_len,
        n_layer=n_layer,
        config=config,
        qaic_config={"num_replicate_kv_heads": num_replicate_kv_heads},
    )


def check_causal_lm_pytorch_vs_kv_vs_ort_vs_ai100(
    model_name: str,
    continuous_batching: bool = False,
    n_layer: int = -1,
    config: Optional[AutoConfig] = None,
    transform_params: Optional[dict] = None,
    export_params: Optional[dict] = None,
    compile_params: Optional[dict] = None,
    generate_params: Optional[dict] = None,
    export_compile_only: bool = False,
):
    torch.manual_seed(42)
    replace_transformers_quantizers()
    torch_dtype = transform_params.get("torch_dtype", torch.float32)
    model_hf = load_hf_causal_lm_model(model_name, num_hidden_layers=n_layer, config=config, torch_dtype=torch_dtype)
    print(model_hf)
    tokenizer = load_hf_tokenizer(pretrained_model_name_or_path=model_name)
    config = model_hf.config
    prompt = generate_params.get("prompt", Constants.INPUT_STR)
    prompt_len = compile_params.get("prefill_seq_len", Constants.PROMPT_LEN)
    ctx_len = compile_params.get("ctx_len", Constants.CTX_LEN)
    num_devices = compile_params.get("num_devices", 1)
    batch_size = len(prompt)
    prompts = prompt * 4 if continuous_batching else prompt
    full_batch_size = 4
    # generation_len = generate_params.get("generation_len", 25)
    num_speculative_tokens = compile_params.get("num_speculative_tokens", None)
    is_tlm = False if num_speculative_tokens is None else True
    qaic_config = transform_params.get("qaic_config", None)
    prefill_only = compile_params.get("prefill_only", None)

    pytorch_hf_tokens = None
    pytorch_kv_tokens = None

    qeff_model = QEFFAutoModelForCausalLM(
        copy.deepcopy(model_hf),
        is_tlm=is_tlm,
        pretrained_model_name_or_path=model_name,
        continuous_batching=continuous_batching,
        qaic_config=qaic_config,
    )
    qeff_model.transform(
        ctx_len=ctx_len,
        seq_len=prompt_len,
        batch_size=full_batch_size if continuous_batching else batch_size,
        num_devices=num_devices,
        qaic_config=qaic_config,
    )
    api_runner = ApiRunner(
        batch_size,
        tokenizer,
        qeff_model.config,
        prompts,
        Constants.PROMPT_LEN,
        Constants.CTX_LEN,
        full_batch_size if continuous_batching else None,
    )
    if continuous_batching is False:
        pytorch_kv_tokens = api_runner.run_kv_model_on_pytorch(qeff_model.model)
    if model_name not in ModelConfig.SWIFTKV_MODELS and model_name not in ModelConfig.EXTERNAL_MODELS:
        if continuous_batching:
            pytorch_hf_tokens = api_runner.run_hf_model_on_pytorch_CB(model_hf)
            pytorch_hf_tokens = np.vstack(pytorch_hf_tokens)
        else:
            pytorch_hf_tokens = api_runner.run_hf_model_on_pytorch(model_hf)

    _ = qeff_model.export(**export_params)

    compiler_options = {}
    if continuous_batching and prompt_len == 1:
        prefill_spec = {
            "batch_size": batch_size,
            "seq_len": 1,
            "ctx_len": ctx_len,
            "full_batch_size": full_batch_size,
            "sliding_window": 128,
        }
        decode_spec = {
            "batch_size": full_batch_size,
            "seq_len": 1,
            "ctx_len": ctx_len,
            "full_batch_size": full_batch_size,
            "sliding_window": 128,
        }
        compiler_options["specializations"] = [prefill_spec, decode_spec]

    mdp_compile_kwargs = {}
    mdp_num_partitions = compile_params.pop("mdp_num_partitions", None)
    mdp_strategy = compile_params.pop("mdp_strategy", None)
    if mdp_num_partitions is not None:
        mdp_compile_kwargs["mdp_num_partitions"] = mdp_num_partitions
    if mdp_strategy is not None:
        mdp_compile_kwargs["mdp_strategy"] = mdp_strategy

    qpc_path = qeff_model.compile(
        **compile_params,
        batch_size=batch_size if continuous_batching else 1,
        full_batch_size=full_batch_size if continuous_batching else None,
        **compiler_options,
        **mdp_compile_kwargs,
    )
    assert os.path.isfile(os.path.join(os.path.dirname(qpc_path), "qconfig.json"))

    if export_compile_only:
        return

    # Generate
    exec_info = qeff_model.generate(tokenizer, prompts=prompts)

    # if pytorch_hf_tokens is not None and pytorch_kv_tokens is not None:
    #     assert (pytorch_hf_tokens == pytorch_kv_tokens).all(), (
    #         "Tokens don't match for HF PyTorch model output and KV PyTorch model output."
    #     )

    if continuous_batching:
        cloud_ai_100_tokens = exec_info.generated_ids
        if cloud_ai_100_tokens is not None and pytorch_hf_tokens is not None:
            assert all(
                [
                    all(pt_token[:24] == cloud_token[:24])
                    for pt_token, cloud_token in zip(pytorch_hf_tokens, cloud_ai_100_tokens)
                ]
            ), "Tokens don't match for  HF PyTorch model output and Cloud AI 100 output."
        if pytorch_hf_tokens is not None and cloud_ai_100_tokens is not None:
            assert all(
                [
                    all(pt_token[:24] == cloud_token[:24])
                    for pt_token, cloud_token in zip(pytorch_hf_tokens, cloud_ai_100_tokens)
                ]
            ), "Tokens don't match for  HF PyTorch model output and Cloud AI 100 output."
    else:
        gen_len = pytorch_kv_tokens.shape[-1]
        cloud_ai_100_tokens = exec_info.generated_ids[0][:, :gen_len]
        if prefill_only:
            assert (pytorch_hf_tokens[0][0] == cloud_ai_100_tokens[0][0]).all(), (
                "prefill run output tokens don't match for HF PyTorch output and Cloud AI 100 output."
            )
        else:
            assert (pytorch_hf_tokens == cloud_ai_100_tokens).all(), (
                "Tokens don't match for HF PyTorch output and Cloud AI 100 output."
            )


def check_prefix_caching_inference(
    model_name: str,
    continuous_batching: bool = False,
    n_layer: int = -1,
    config: Optional[AutoConfig] = None,
    transform_params: Optional[dict] = None,
    export_params: Optional[dict] = None,
    compile_params: Optional[dict] = None,
    generate_params: Optional[dict] = None,
    export_compile_only: bool = False,
):

    torch.manual_seed(42)
    replace_transformers_quantizers()
    torch_dtype = transform_params.get("torch_dtype", torch.float32)
    qeff_model = load_qeff_causal_lm_model(
        model_name=model_name,
        num_hidden_layers=n_layer,
        continuous_batching=continuous_batching,
        config=config,
        torch_dtype=torch_dtype,
    )
    qeff_model.compile(
        **compile_params,
    )
    if export_compile_only:
        return

    qpc_path = qeff_model.qpc_path
    assert os.path.isfile(os.path.join(os.path.dirname(qpc_path), "qconfig.json"))

    full_batch_size = compile_params.get("full_batch_size", 2)
    ctx_len = compile_params.get("ctx_len", Constants.CTX_LEN)
    prefixes = generate_params.get("prefixes", ["Once upon a time ", "Once upon a time "])
    suffixes1 = generate_params.get("suffixes1", ["in a land far away", "there was a small village"])
    suffixes2 = generate_params.get("suffixes2", ["a little girl", "in a bustling city"])

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    generator = TextGeneration(tokenizer=tokenizer, qpc_path=qpc_path, full_batch_size=full_batch_size, ctx_len=ctx_len)

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

    assert np.all(generator._qaic_model.generated_ids[0, :gen_len2] == [int(val[0, 0]) for val in generation_outputs])
    assert np.all(generator._qaic_model.generated_ids[1, :gen_len2] == [int(val[1, 0]) for val in generation_outputs])

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
        prompts_exec_info.generated_ids[1][:247] == [int(val[1, 0]) for val in generation_outputs_prefill_cached][:247]
    )

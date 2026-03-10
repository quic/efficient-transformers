# -----------------------------------------------------------------------------
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------

from typing import List, Optional, Tuple, Union

import numpy as np
import pytest
from transformers import AutoConfig, AutoModelForCausalLM, AutoProcessor, AutoTokenizer

from QEfficient import QEFFAutoModelForCausalLM, QEFFAutoModelForImageTextToText
from QEfficient.generation.cloud_infer import QAICInferenceSession
from QEfficient.utils import load_hf_tokenizer
from QEfficient.utils.constants import Constants
from QEfficient.utils.test_utils import InternProcessor
from tests.transformers.models.image_text_to_text.test_continuous_batching import set_num_layers

test_configs = [
    pytest.param(
        "TinyLlama/TinyLlama-1.1B-Chat-v1.0",  # model
        Constants.INPUT_STR * 2,  # prompts
        32,  # prefill_seq_len
        64,  # ctx_len
        20,  # generation_len
        2,  # full_batch_size
        1,  # spec_length
        False,  # is_vlm
    ),
    pytest.param(
        "OpenGVLab/InternVL2_5-1B",  # model
        (
            ["https://picsum.photos/id/237/536/354"] * 2,
            ["Can you describe the image in detail."] * 2,
        ),  # images and prompts
        128,  # prefill_seq_len
        4096,  # ctx_len
        20,  # generation_len
        2,  # full_batch_size
        None,  # spec_length
        True,  # is_vlm
    ),
]


def prepare_model_setup(
    model: str, is_vlm: bool, num_hidden_layers: int, prompts: Union[List, Tuple], spec_length: Optional[int]
):
    additional_configs = {}
    additional_params = {}
    if is_vlm:
        config = AutoConfig.from_pretrained(model, trust_remote_code=True)
        config = set_num_layers(config, n_layer=num_hidden_layers)
        additional_configs["config"] = config
        additional_configs["kv_offload"] = True
        assert isinstance(prompts, tuple), "For VLMs, both image and text prompts must be provided."
        additional_params["images"] = prompts[0]
        prompts = prompts[1]

        if "InternVL" in model:
            additional_configs["trust_remote_code"] = True
            model_hf = AutoModelForCausalLM.from_pretrained(
                model,
                config=config,
                trust_remote_code=True,
            )
            tokenizer = AutoTokenizer.from_pretrained(model, trust_remote_code=True, use_fast=False)
            additional_params["processor"] = InternProcessor(model_hf, tokenizer)
            qeff_class = QEFFAutoModelForCausalLM
        else:
            additional_params["processor"] = AutoProcessor.from_pretrained(model)
            qeff_class = QEFFAutoModelForImageTextToText
    else:
        if num_hidden_layers != -1:
            additional_configs["num_hidden_layers"] = num_hidden_layers
        spec_length = (spec_length or 1) - 1
        qeff_class = QEFFAutoModelForCausalLM
    return additional_configs, additional_params, prompts, spec_length, qeff_class


@pytest.mark.on_qaic
@pytest.mark.feature
@pytest.mark.parametrize(
    "model, prompts, prefill_seq_len, ctx_len, generation_len, full_batch_size, spec_length, is_vlm",
    test_configs,
)
def test_sampler_transform(
    model: str,
    prompts: Union[List[str], tuple[List[str], List[str]]],
    prefill_seq_len: int,
    ctx_len: int,
    generation_len: int,
    full_batch_size: int,
    spec_length: Optional[int],
    is_vlm: bool,
):
    """
    Test if `SamplerTransform` adds nodes at the output of a `QEffForCausalLM model` to enable the
    sampling of next tokens at the device (instead of the host) and returns the
    next tokens and/or probability distributions.
    """
    # Export and compile QEfficient models
    num_hidden_layers = 2
    additional_configs, additional_params, prompts, spec_length, qeff_class = prepare_model_setup(
        model, is_vlm, num_hidden_layers, prompts, spec_length
    )
    model_w_sampler = qeff_class.from_pretrained(
        model,
        continuous_batching=True,
        qaic_config={
            "include_sampler": True,
            "return_pdfs": False,
            "max_top_k_ids": 512,
        },
        **additional_configs,
    )
    model_w_sampler_w_guided_decoding = qeff_class.from_pretrained(
        model,
        continuous_batching=True,
        qaic_config={
            "include_sampler": True,
            "return_pdfs": False,
            "max_top_k_ids": 512,
            "include_guided_decoding": True,
        },
        **additional_configs,
    )
    model_wo_sampler = qeff_class.from_pretrained(
        model,
        continuous_batching=True,
        qaic_config={
            "include_sampler": False,
            "return_pdfs": False,
        },
        **additional_configs,
    )
    model_w_sampler_qpc_path = model_w_sampler.compile(
        prefill_seq_len=prefill_seq_len,
        ctx_len=ctx_len,
        full_batch_size=full_batch_size,
        num_devices=1,
        num_cores=16,
        num_speculative_tokens=spec_length,
        mxint8_kv_cache=True,
        mxfp6_matmul=True,
    )
    model_w_sampler_w_guided_decoding_qpc_path = model_w_sampler_w_guided_decoding.compile(
        prefill_seq_len=prefill_seq_len,
        ctx_len=ctx_len,
        full_batch_size=full_batch_size,
        num_devices=1,
        num_cores=16,
        num_speculative_tokens=spec_length,
        mxint8_kv_cache=True,
        mxfp6_matmul=True,
    )
    model_wo_sampler_qpc_path = model_wo_sampler.compile(
        prefill_seq_len=prefill_seq_len,
        ctx_len=ctx_len,
        full_batch_size=full_batch_size,
        num_devices=1,
        num_cores=16,
        num_speculative_tokens=spec_length,
        mxint8_kv_cache=True,
        mxfp6_matmul=True,
    )
    if is_vlm:
        model_w_sampler_qpc_path = model_w_sampler_qpc_path[1]
        model_w_sampler_w_guided_decoding_qpc_path = model_w_sampler_w_guided_decoding_qpc_path[1]
        model_wo_sampler_qpc_path = model_wo_sampler_qpc_path[1]

    # Init qaic session
    model_w_sampler_session = QAICInferenceSession(model_w_sampler_qpc_path)
    model_w_sampler_w_guided_decoding_session = QAICInferenceSession(model_w_sampler_w_guided_decoding_qpc_path)
    model_wo_sampler_session = QAICInferenceSession(model_wo_sampler_qpc_path)

    # Skip inputs/outputs buffers
    model_w_sampler_session.skip_buffers(set([x for x in model_w_sampler_session.input_names if x.startswith("past_")]))
    model_w_sampler_session.skip_buffers(
        set([x for x in model_w_sampler_session.output_names if x.endswith("_RetainedState")])
    )
    model_w_sampler_w_guided_decoding_session.skip_buffers(
        set([x for x in model_w_sampler_w_guided_decoding_session.input_names if x.startswith("past_")])
    )
    model_w_sampler_w_guided_decoding_session.skip_buffers(
        set([x for x in model_w_sampler_w_guided_decoding_session.output_names if x.endswith("_RetainedState")])
    )
    model_wo_sampler_session.skip_buffers(
        set([x for x in model_wo_sampler_session.input_names if x.startswith("past_")])
    )
    model_wo_sampler_session.skip_buffers(
        set([x for x in model_wo_sampler_session.output_names if x.endswith("_RetainedState")])
    )

    # Validate sampler inputs
    sampler_inputs = Constants.SAMPLER_INPUTS
    for input_name in sampler_inputs:
        assert input_name in model_w_sampler_session.input_names, (
            f"Sampler input {input_name} not found in QPC compiled with On Device Sampler"
        )
        assert input_name in model_w_sampler_w_guided_decoding_session.input_names, (
            f"Sampler input {input_name} not found in QPC compiled with On Device Sampler and Guided Decoding"
        )
        assert input_name not in model_wo_sampler_session.input_names, (
            f"Sampler input {input_name} found in QPC compiled without On Device Sampler"
        )
    assert "token_bitmasks" in model_w_sampler_w_guided_decoding_session.input_names, (
        "Sampler input token_bitmasks not found in QPC compiled with On Device Sampler and Guided Decoding"
    )


@pytest.mark.on_qaic
@pytest.mark.feature
@pytest.mark.parametrize(
    "model, prompts, prefill_seq_len, ctx_len, generation_len, full_batch_size, spec_length, is_vlm",
    test_configs,
)
def test_greedy_sampling(
    model: str,
    prompts: Union[List[str], tuple[List[str], List[str]]],
    prefill_seq_len: int,
    ctx_len: int,
    generation_len: int,
    full_batch_size: int,
    spec_length: Optional[int],
    is_vlm: bool,
):
    """
    Test greedy sampling with QPCs compiled with and without On Device Sampling.
    """
    # Export and compile QEfficient models
    num_hidden_layers = 4
    additional_configs, additional_params, prompts, spec_length, qeff_class = prepare_model_setup(
        model, is_vlm, num_hidden_layers, prompts, spec_length
    )
    model_w_sampler = qeff_class.from_pretrained(
        model,
        continuous_batching=True,
        qaic_config={
            "include_sampler": True,
            "return_pdfs": False,
            "max_top_k_ids": 512,
        },
        **additional_configs,
    )
    model_wo_sampler = qeff_class.from_pretrained(
        model,
        continuous_batching=True,
        qaic_config={
            "include_sampler": False,
            "return_pdfs": False,
        },
        **additional_configs,
    )
    model_w_sampler.compile(
        prefill_seq_len=prefill_seq_len,
        ctx_len=ctx_len,
        full_batch_size=full_batch_size,
        num_devices=1,
        num_cores=16,
        num_speculative_tokens=spec_length,
        mxint8_kv_cache=True,
        mxfp6_matmul=True,
    )
    model_wo_sampler.compile(
        prefill_seq_len=prefill_seq_len,
        ctx_len=ctx_len,
        full_batch_size=full_batch_size,
        num_devices=1,
        num_cores=16,
        num_speculative_tokens=spec_length,
        mxint8_kv_cache=True,
        mxfp6_matmul=True,
    )

    # Generate texts from prompts
    tokenizer = load_hf_tokenizer(pretrained_model_name_or_path=model)
    model_w_sampler_exec_info = model_w_sampler.generate(
        tokenizer=tokenizer,
        prompts=prompts,
        generation_len=generation_len,
        include_sampler=True,
        return_pdfs=False,
        sampling_params={
            "repetition_penalties": np.array(1.0, dtype=np.float32).repeat(full_batch_size).reshape(-1, 1),
            "presence_penalties": np.array(0.0, dtype=np.float32).repeat(full_batch_size).reshape(-1, 1),
            # "frequency_penalties": np.array(0.0, dtype=np.float32).repeat(full_batch_size).reshape(-1, 1),
            "temperatures": np.array(0.0, dtype=np.float32).repeat(full_batch_size).reshape(-1, 1),
            "top_ks": np.array(512, dtype=np.int32).repeat(full_batch_size).reshape(-1, 1),
            "top_ps": np.array(1.0, dtype=np.float32).repeat(full_batch_size).reshape(-1, 1),
            "min_ps": np.array(0.0, dtype=np.float32).repeat(full_batch_size).reshape(-1, 1),
            "random_numbers": np.zeros((full_batch_size, 512), dtype=np.float32),
        },
        **additional_params,
    )
    model_wo_sampler_exec_info = model_wo_sampler.generate(
        tokenizer=tokenizer,
        prompts=prompts,
        generation_len=generation_len,
        include_sampler=False,
        return_pdfs=False,
        sampling_params=None,
        **additional_params,
    )

    # Compare generated texts and ids
    assert model_w_sampler_exec_info.generated_texts == model_wo_sampler_exec_info.generated_texts, (
        "Generated texts do not match"
    )
    assert (model_w_sampler_exec_info.generated_ids == model_wo_sampler_exec_info.generated_ids).all(), (
        "Generated ids do not match"
    )


@pytest.mark.on_qaic
@pytest.mark.feature
@pytest.mark.parametrize(
    "model, prompts, prefill_seq_len, ctx_len, generation_len, full_batch_size, spec_length, is_vlm",
    test_configs,
)
def test_random_sampling(
    model: str,
    prompts: Union[List[str], tuple[List[str], List[str]]],
    prefill_seq_len: int,
    ctx_len: int,
    generation_len: int,
    full_batch_size: int,
    spec_length: Optional[int],
    is_vlm: bool,
):
    """
    Test random sampling with QPCs compiled with and without On Device Sampling.
    """
    # Export and compile QEfficient models
    num_hidden_layers = -1
    additional_configs, additional_params, prompts, spec_length, qeff_class = prepare_model_setup(
        model, is_vlm, num_hidden_layers, prompts, spec_length
    )
    model_w_sampler = qeff_class.from_pretrained(
        model,
        continuous_batching=True,
        qaic_config={
            "include_sampler": True,
            "return_pdfs": False,
            "max_top_k_ids": 512,
        },
        **additional_configs,
    )
    model_wo_sampler = qeff_class.from_pretrained(
        model,
        continuous_batching=True,
        qaic_config={
            "include_sampler": False,
            "return_pdfs": False,
        },
        **additional_configs,
    )
    model_w_sampler.compile(
        prefill_seq_len=prefill_seq_len,
        ctx_len=ctx_len,
        full_batch_size=full_batch_size,
        num_devices=1,
        num_cores=16,
        num_speculative_tokens=spec_length,
        mxint8_kv_cache=True,
        mxfp6_matmul=True,
    )
    model_wo_sampler.compile(
        prefill_seq_len=prefill_seq_len,
        ctx_len=ctx_len,
        full_batch_size=full_batch_size,
        num_devices=1,
        num_cores=16,
        num_speculative_tokens=spec_length,
        mxint8_kv_cache=True,
        mxfp6_matmul=True,
    )

    # Generate texts from prompts
    tokenizer = load_hf_tokenizer(pretrained_model_name_or_path=model)
    np.random.seed(0)
    model_w_sampler_exec_info = model_w_sampler.generate(
        tokenizer=tokenizer,
        prompts=prompts,
        generation_len=generation_len,
        include_sampler=True,
        return_pdfs=False,
        sampling_params={
            "repetition_penalties": np.array(20.2, dtype=np.float32).repeat(full_batch_size).reshape(-1, 1),
            "presence_penalties": np.array(10.5, dtype=np.float32).repeat(full_batch_size).reshape(-1, 1),
            # "frequency_penalties": np.array(0.5, dtype=np.float32).repeat(full_batch_size).reshape(-1, 1),
            "temperatures": np.array(4.0, dtype=np.float32).repeat(full_batch_size).reshape(-1, 1),
            "top_ks": np.array(512, dtype=np.int32).repeat(full_batch_size).reshape(-1, 1),
            "top_ps": np.array(0.89, dtype=np.float32).repeat(full_batch_size).reshape(-1, 1),
            "min_ps": np.array(0.6, dtype=np.float32).repeat(full_batch_size).reshape(-1, 1),
            "random_numbers": np.tile(np.random.uniform(low=0.0, high=1.0, size=512), (full_batch_size, 1)).astype(
                np.float32
            ),
        },
        **additional_params,
    )
    model_wo_sampler_exec_info = model_wo_sampler.generate(
        tokenizer=tokenizer,
        prompts=prompts,
        generation_len=generation_len,
        include_sampler=False,
        return_pdfs=False,
        sampling_params=None,
        **additional_params,
    )

    # Compare generated texts
    if model == "TinyLlama/TinyLlama-1.1B-Chat-v1.0":
        golden_texts = {
            "w_sampler": "Aiden and I am a freelance writer who loves to explore the world. With over",
            "wo_sampler": "John Smith and I am a software engineer. I have been working in the industry for the past ",
        }
        golden_ids = {
            "w_sampler": [
                [
                    319,
                    3615,
                    322,
                    306,
                    626,
                    263,
                    3005,
                    295,
                    749,
                    9227,
                    1058,
                    12355,
                    267,
                    304,
                    26987,
                    278,
                    3186,
                    29889,
                    2973,
                    975,
                ]
            ],
            "wo_sampler": [
                [
                    2259,
                    7075,
                    322,
                    306,
                    626,
                    263,
                    7047,
                    22055,
                    29889,
                    306,
                    505,
                    1063,
                    1985,
                    297,
                    278,
                    13661,
                    363,
                    278,
                    4940,
                    29871,
                ]
            ],
        }
    elif model == "OpenGVLab/InternVL2_5-1B":
        golden_texts = {
            "w_sampler": "The description of this vivid scene is as follows:\n\nIn a sepia-toned photograph, we see",
            "wo_sampler": "The image features a black puppy lying on a wooden surface. The puppy has a shiny, glossy coat",
        }
        golden_ids = {
            "w_sampler": [
                [
                    785,
                    4008,
                    315,
                    419,
                    42020,
                    6109,
                    374,
                    438,
                    11017,
                    1447,
                    641,
                    264,
                    21017,
                    685,
                    74635,
                    291,
                    10300,
                    11,
                    582,
                    1490,
                ]
            ],
            "wo_sampler": [
                [
                    785,
                    2168,
                    4419,
                    264,
                    3691,
                    41189,
                    20446,
                    389,
                    264,
                    22360,
                    7329,
                    13,
                    576,
                    41189,
                    702,
                    264,
                    41199,
                    11,
                    73056,
                    22875,
                ]
            ],
        }
    for i in range(full_batch_size):
        assert (
            tokenizer.decode(model_w_sampler_exec_info.generated_ids[i][:generation_len]) == golden_texts["w_sampler"]
        ), "Sampler generated texts does not match"
        assert (model_w_sampler_exec_info.generated_ids[i][:generation_len] == golden_ids["w_sampler"]).all(), (
            "Sampler generated ids do not match"
        )
        assert (
            tokenizer.decode(model_wo_sampler_exec_info.generated_ids[i][:generation_len]) == golden_texts["wo_sampler"]
        ), "Without sampler generated texts does not match"
        assert (model_wo_sampler_exec_info.generated_ids[i][:generation_len] == golden_ids["wo_sampler"]).all(), (
            "Without sampler generated ids do not match"
        )


@pytest.mark.on_qaic
@pytest.mark.feature
@pytest.mark.parametrize(
    "model, prompts, prefill_seq_len, ctx_len, generation_len, full_batch_size, spec_length, is_vlm",
    test_configs,
)
def test_guided_decoding(
    model: str,
    prompts: Union[List[str], tuple[List[str], List[str]]],
    prefill_seq_len: int,
    ctx_len: int,
    generation_len: int,
    full_batch_size: int,
    spec_length: Optional[int],
    is_vlm: bool,
):
    """
    Test QPCs compiled with and without guided decoding.
    """
    # Export and compile QEfficient models
    num_hidden_layers = 2
    additional_configs, additional_params, prompts, spec_length, qeff_class = prepare_model_setup(
        model, is_vlm, num_hidden_layers, prompts, spec_length
    )
    model_w_sampler_w_guided_decoding = qeff_class.from_pretrained(
        model,
        continuous_batching=True,
        qaic_config={
            "include_sampler": True,
            "return_pdfs": False,
            "max_top_k_ids": 1024,
            "include_guided_decoding": True,
        },
        **additional_configs,
    )
    model_w_sampler_wo_guided_decoding = qeff_class.from_pretrained(
        model,
        continuous_batching=True,
        qaic_config={
            "include_sampler": True,
            "return_pdfs": False,
            "max_top_k_ids": 1024,
        },
        **additional_configs,
    )
    model_w_sampler_w_guided_decoding.compile(
        prefill_seq_len=prefill_seq_len,
        ctx_len=ctx_len,
        full_batch_size=full_batch_size,
        num_devices=1,
        num_cores=16,
        num_speculative_tokens=spec_length,
        mxint8_kv_cache=True,
        mxfp6_matmul=True,
    )
    model_w_sampler_wo_guided_decoding.compile(
        prefill_seq_len=prefill_seq_len,
        ctx_len=ctx_len,
        full_batch_size=full_batch_size,
        num_devices=1,
        num_cores=16,
        num_speculative_tokens=spec_length,
        mxint8_kv_cache=True,
        mxfp6_matmul=True,
    )

    # Generate texts from prompts
    tokenizer = load_hf_tokenizer(pretrained_model_name_or_path=model)
    np.random.seed(0)
    sampling_params = {
        "repetition_penalties": np.array(1.0, dtype=np.float32).repeat(full_batch_size).reshape(-1, 1),
        "presence_penalties": np.array(0.0, dtype=np.float32).repeat(full_batch_size).reshape(-1, 1),
        # "frequency_penalties": np.array(0.0, dtype=np.float32).repeat(full_batch_size).reshape(-1, 1),
        "temperatures": np.array(0.0, dtype=np.float32).repeat(full_batch_size).reshape(-1, 1),
        "top_ks": np.array(1024, dtype=np.int32).repeat(full_batch_size).reshape(-1, 1),
        "top_ps": np.array(1.0, dtype=np.float32).repeat(full_batch_size).reshape(-1, 1),
        "min_ps": np.array(0.0, dtype=np.float32).repeat(full_batch_size).reshape(-1, 1),
        "random_numbers": np.zeros((full_batch_size, 1024), dtype=np.float32),
    }
    if is_vlm:
        vocab_size = model_w_sampler_w_guided_decoding.model.language_model.config.vocab_size
    else:
        vocab_size = model_w_sampler_w_guided_decoding.model.config.vocab_size
    model_w_sampler_w_guided_decoding_exec_info = model_w_sampler_w_guided_decoding.generate(
        tokenizer=tokenizer,
        prompts=prompts,
        generation_len=generation_len,
        include_sampler=True,
        return_pdfs=False,
        include_guided_decoding=True,
        sampling_params={
            **sampling_params,
            **{
                "token_bitmasks": np.tile(
                    np.random.choice([True, False], size=(vocab_size,)),
                    (full_batch_size, 1),
                )
            },
        },
        **additional_params,
    )
    model_w_sampler_wo_guided_decoding_exec_info = model_w_sampler_wo_guided_decoding.generate(
        tokenizer=tokenizer,
        prompts=prompts,
        generation_len=generation_len,
        include_sampler=True,
        return_pdfs=False,
        sampling_params=sampling_params,
        **additional_params,
    )
    assert (
        model_w_sampler_w_guided_decoding_exec_info.generated_ids
        != model_w_sampler_wo_guided_decoding_exec_info.generated_ids
    ).any(), "Sampler outputs with and without guided decoding should not match"

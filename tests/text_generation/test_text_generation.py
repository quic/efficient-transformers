# -----------------------------------------------------------------------------
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------

import os

import pytest
from transformers import AutoModelForCausalLM

from QEfficient.generation.text_generation_inference import TextGeneration
from QEfficient.transformers.models.modeling_auto import QEFFAutoModelForCausalLM
from QEfficient.utils import hf_download
from QEfficient.utils._utils import load_hf_tokenizer
from QEfficient.utils.constants import Constants
from QEfficient.utils.device_utils import get_available_device_id

configs = [pytest.param("gpt2", 2, None, 32, id="gpt2_config")]


def load_causal_lm_model(model_config):
    """
    Function to load model from huggingface and transform to KV model
    --------

    :model_config: Dict

    :return model_hf, params
    """
    model_path = hf_download(
        repo_id=model_config["model_name"],
        ignore_patterns=["*.onnx", "*.ot", "*.md", "*.tflite", "*.pdf", "*.h5", "*.msgpack"],
    )
    model_hf = AutoModelForCausalLM.from_pretrained(
        model_path,
        use_cache=True,
        num_hidden_layers=model_config["n_layer"],
        attn_implementation="eager",
        low_cpu_mem_usage=False,
    )  # Run models for single layers only
    params = sum(p.numel() for p in model_hf.parameters())
    model_hf.eval()
    return model_hf, params


# Use @pytest.mark.parametrize to apply the configurations
@pytest.mark.on_qaic
@pytest.mark.llm_model
@pytest.mark.parametrize("model_name, n_layer, full_batch_size, max_gen_len", configs)
def test_generate_text_stream(
    model_name: str,
    n_layer: int,
    full_batch_size: int,
    max_gen_len: int,
    prompt_len: int = Constants.PROMPT_LEN,
    ctx_len: int = Constants.CTX_LEN,
):
    """
    Validate the PyTorch model, the PyTorch model after KV changes, the ONNX model, and the Cloud AI 100 model, both with and without continuous batching.
    ``Mandatory`` Args:
        :model_name (str): Hugging Face Model Card name, Example: ``gpt2``
        :prompt_len (int): Prompt length for the model to compile.
        :ctx_len (int): Maximum context length to compile the model.
        :n_layers (int): Number of layers for the Model.
    """
    model_config = {"model_name": model_name, "n_layer": n_layer}
    model_hf, _ = load_causal_lm_model(model_config)

    tokenizer = load_hf_tokenizer(pretrained_model_name_or_path=model_name)

    qeff_model = QEFFAutoModelForCausalLM(model_hf)

    qeff_model.export()
    device_id = get_available_device_id()

    if not device_id:
        pytest.skip("No available devices to run model on Cloud AI 100")

    qpc_path = qeff_model.compile(
        prefill_seq_len=prompt_len,
        ctx_len=ctx_len,
        num_cores=14,
        mxfp6=False,
        aic_enable_depth_first=False,
        full_batch_size=full_batch_size,
    )

    exec_info = qeff_model.generate(tokenizer, prompts=Constants.INPUT_STR, generation_len=max_gen_len)
    cloud_ai_100_tokens = exec_info.generated_ids[0]  # Because we always run for single input and single batch size
    cloud_ai_100_output = [tokenizer.decode(token, skip_special_tokens=True) for token in cloud_ai_100_tokens[0]]

    text_generator = TextGeneration(
        tokenizer=tokenizer,
        qpc_path=qpc_path,
        ctx_len=ctx_len,
        full_batch_size=full_batch_size,
    )
    stream_tokens = []
    for decoded_tokens in text_generator.generate_stream_tokens(Constants.INPUT_STR, generation_len=max_gen_len):
        stream_tokens.extend(decoded_tokens)

    assert cloud_ai_100_output == stream_tokens, (
        f"Deviation in output observed while comparing regular execution and streamed output: {cloud_ai_100_output} != {stream_tokens}"
    )
    assert os.path.isfile(os.path.join(os.path.dirname(qpc_path), "qconfig.json"))

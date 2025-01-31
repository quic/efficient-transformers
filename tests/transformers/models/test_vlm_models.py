# -----------------------------------------------------------------------------
#
# Copyright (c) 2024 Qualcomm Innovation Center, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------


import pytest
import requests
from PIL import Image
from transformers import AutoModelForCausalLM, AutoProcessor, TextStreamer

# from QEfficient.exporter.export_hf_to_cloud_ai_100 import qualcomm_efficient_converter
from QEfficient.transformers.models.modeling_auto import QEFFAutoModelForImageTextToText

# from QEfficient.transformers.quantizers.auto import replace_transformers_quantizers
from QEfficient.utils import hf_download

# from QEfficient.utils._utils import load_hf_processor
from QEfficient.utils.constants import VlmConstants
from QEfficient.utils.device_utils import get_available_device_id

test_models = [
    "microsoft/Phi-3.5-vision-instruct",
]


def load_vlm_model(model_config):
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
        _attn_implementation="eager",
        low_cpu_mem_usage=False,
    )  # Run models for single layers only
    params = sum(p.numel() for p in model_hf.parameters())
    model_hf.eval()
    return model_hf, params


def _generate_inputs(model, processor):
    ## PREPROCESSING THE MULTI-MODAL INPUTS
    images = []
    placeholder = ""

    # Note: if OOM, you might consider reduce number of frames in this example.
    for i in range(1, 2):
        url = f"https://image.slidesharecdn.com/azureintroduction-191206101932/75/Introduction-to-Microsoft-Azure-Cloud-{i}-2048.jpg"
        images.append(Image.open(requests.get(url, stream=True).raw))
        placeholder += f"<|image_{1}|>\n"

    messages = [
        {"role": "user", "content": placeholder + "Summarize the deck of slides."},
    ]

    prompt = processor.tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    model.config.hidden_size // model.config.num_attention_heads
    # ctx_len = 1280  # FIXME: Pass it a ssome arguement later on
    inputs = dict(processor(images=images, text=prompt, return_tensors="pt"))
    return inputs


def check_vlm_pytorch_vs_kv_vs_ort_vs_ai100(
    model_name: str,
    prompt_len: int = VlmConstants.SEQ_LEN,
    ctx_len: int = VlmConstants.CTX_LEN,
    n_layer: int = 1,
):
    """
    Validate the PyTorch model, the PyTorch model after KV changes, the ONNX model, and the Cloud AI 100 model, both with and without continuous batching.
    ``Mandatory`` Args:
        :model_name (str): Hugging Face Model Card name, Example: ``Phi-3.5-vision-instruct``
        :prompt_len (int): Prompt length for the model to compile.
        :ctx_len (int): Maximum context length to compile the model.
        :n_layers (int): Number of layers for the Model.
    """
    # replace_transformers_quantizers()
    model_config = {"model_name": model_name}
    model_config["n_layer"] = n_layer

    model_hf, _ = load_vlm_model(model_config)
    # Load processor instead
    processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)
    streamer = TextStreamer(processor)
    streamer.on_finalized_text("<  ")
    inputs = _generate_inputs(model_hf, processor)
    # Original PyTorch model
    pt_model = AutoModelForCausalLM.from_pretrained(
        model_name,
        num_hidden_layers=n_layer,
        _attn_implementation="eager",
        trust_remote_code=True,
        rope_scaling=None,
    )

    qeff_model = QEFFAutoModelForImageTextToText(pt_model, processor, is_tlm=False)
    qeff_model.export()
    if not get_available_device_id():
        pytest.skip("No available devices to run model on Cloud AI 100")

    _ = qeff_model.compile(
        prefill_seq_len=prompt_len,
        ctx_len=ctx_len,
        num_cores=14,
        mxfp6=False,
        aic_enable_depth_first=False,
    )
    exec_info = qeff_model.generate(inputs, streamer, device_ids=None, runtime_ai100=True)
    exec_info[0]  # Because we always run for single input and single batch size


@pytest.mark.on_qaic
@pytest.mark.parametrize("model_name", test_models)
def test_causal_lm_pytorch_vs_kv_vs_ort_vs_ai100(model_name):
    """
    Test function to validate the PyTorch model, the PyTorch model after KV changes, the ONNX model, and the Cloud AI 100 model, both with and without continuous batching.
    ``Mandatory`` Args:
        :model_name (str): Hugging Face Model Card name, Example: ``gpt2``
    """
    if model_name == "microsoft/Phi-3-mini-4k-instruct":
        n_layer = 2  # test only 2 layer models
    else:
        n_layer = 32

    check_vlm_pytorch_vs_kv_vs_ort_vs_ai100(model_name=model_name, n_layer=n_layer)

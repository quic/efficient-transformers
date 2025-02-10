# -----------------------------------------------------------------------------
#
# Copyright (c) 2024 Qualcomm Innovation Center, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------


# from PIL import Image
import pytest
import requests
import torch

# For intern Specific
from PIL import Image
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoModelForImageTextToText,
    AutoModelForVision2Seq,
    AutoProcessor,
    AutoTokenizer,
    TextStreamer,
)
from transformers.image_utils import load_image

from QEfficient.transformers.models.InternVL.internprocessor import InternProcessor
from QEfficient.transformers.models.modeling_auto import QEFFAutoModelForImageTextToText
from QEfficient.utils import hf_download
from QEfficient.utils.constants import Constants
from QEfficient.utils.device_utils import get_available_device_id

test_models = [
    "llava-hf/llava-1.5-7b-hf",
    "OpenGVLab/InternVL2_5-1B",
]


def load_vlm_model(model_config):
    """
    Function to load model from huggingface and transform to KV model
    --------

    :model_config: Dict

    :return model_hf, params
    """

    if model_config["model_name"] == "OpenGVLab/InternVL2_5-1B":
        config = AutoConfig.from_pretrained(model_config["model_path"])
        config.llm_config.use_cache = True
        # config.llm_config.num_hidden_layers = model_config["n_layer_text"]
        # config.vision_config.num_hidden_layers = model_config["n_layer_vision"]
        config.llm_config._attn_implementation = "eager"
        config.vision_config.use_flash_attn = "false"
        model_hf = AutoModelForCausalLM.from_pretrained(
            model_config["model_path"], low_cpu_mem_usage=False, config=config
        )
    elif model_config["model_name"] == "llava-hf/llava-1.5-7b-hf":
        config = AutoConfig.from_pretrained(model_config["model_path"])
        # config.text_config.num_hidden_layers = model_config["n_layer_text"]
        # config.vision_config.num_hidden_layers = model_config["n_layer_vision"]
        config._attn_implementation = "eager"
        config.vision_config.use_flash_attn = "false"
        model_hf = AutoModelForImageTextToText.from_pretrained(
            model_config["model_path"], low_cpu_mem_usage=False, config=config
        )
    
    params = sum(p.numel() for p in model_hf.parameters())
    model_hf.eval()
    return model_hf, params

def generate_hf_inputs_intern(model_name, model, processor):
    pixel_values = []
    for i in range(1, 2):
        url = f"https://image.slidesharecdn.com/azureintroduction-191206101932/75/Introduction-to-Microsoft-Azure-Cloud-{i}-2048.jpg"
        img = requests.get(url, stream=True).raw
        pixel_values.append(processor.load_image(img, max_num=12))

    question = "<image>\nPlease describe the image in detail."
    pixel_values = torch.cat(pixel_values, dim=0)
    query = processor(processor.tokenizer, pixel_values, question)
    inputs = dict(processor.tokenizer(query, return_tensors="pt"))
    inputs["pixel_values"] = pixel_values
    return inputs


def generate_hf_inputs_llava(model_name, model, processor=None):
    img = Image.open(requests.get(Constants.BASE_URL_LLAVA, stream=True).raw)
    prompt = processor.apply_chat_template(
        [{"role": "user", "content": [{"type": "text", "text": Constants.PROMPT_LLAVA}, {"type": "image"}]}],
        add_generation_prompt=True,
    )
    inputs = processor(images=img, text=prompt, return_tensors="pt")
    return inputs


# ---------------------------------------------
# Please Add new models here inside the map
# {model_name:generate_hf_inputs_<model_name>}
# ---------------------------------------------
generate_hf_inputs_func_map = {
    "llava-hf/llava-1.5-7b-hf": generate_hf_inputs_llava,
    "OpenGVLab/InternVL2_5-1B": generate_hf_inputs_intern,
}


def generate_hf_inputs(model_name, model, processor=None):
    generate_func = generate_hf_inputs_func_map.get(model_name)
    if not generate_func:
        raise ValueError(f"Input generation function for model {model_name} not found.")

    return generate_func(model_name, model, processor)


def check_vlm_pytorch_vs_kv_vs_ort_vs_ai100(
    model_name: str,
    prompt_len: int = Constants.SEQ_LEN_VLM,
    ctx_len: int = Constants.CTX_LEN_VLM_INTERN,
    n_layer_text: int = 1,
    n_layer_vision: int = 1,
    # num_speculative_tokens: Optional[int] = None,
):
    """
    Validate the PyTorch model, the PyTorch model after KV changes, the ONNX model, and the Cloud AI 100 model, both with and without continuous batching.
    ``Mandatory`` Args:
        :model_name (str): Hugging Face Model Card name, Example: ``Phi-3.5-vision-instruct``
        :prompt_len (int): Prompt length for the model to compile.
        :ctx_len (int): Maximum context length to compile the model.
        :n_layers (int): Number of layers for the Model.
    """

    model_config = {"model_name": model_name}
    model_config["n_layer_text"] = n_layer_text
    model_config["n_layer_vision"] = n_layer_vision

    model_path = hf_download(
        repo_id=model_config["model_name"],
        ignore_patterns=["*.onnx", "*.ot", "*.md", "*.tflite", "*.pdf", "*.h5", "*.msgpack"],
    )

    model_config["model_path"] = model_path

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True, use_fast=False)
    model_hf, _ = load_vlm_model(model_config)

    # Load processor for models

    if model_name == "OpenGVLab/InternVL2_5-1B":
        processor = InternProcessor(model_hf, tokenizer)
    else:
        processor = AutoProcessor.from_pretrained(model_name, padding_side="right", trust_remote_code=True)

    streamer = TextStreamer(tokenizer)
    inputs = generate_hf_inputs(model_name, model_hf, processor)

    qeff_model = QEFFAutoModelForImageTextToText.from_pretrained(model_name)
    qeff_model.export()
    if not get_available_device_id():
        pytest.skip("No available devices to run model on Cloud AI 100")

    qpc_path = qeff_model.compile(
        prefill_seq_len=prompt_len,
        ctx_len=ctx_len,
        num_cores=14,
        mxfp6=False,
        aic_enable_depth_first=False,
    )
    qeff_model.qpc_path = qpc_path
    qeff_model.generate(inputs, streamer, device_ids=None, runtime_ai100=True)


@pytest.mark.on_qaic
@pytest.mark.parametrize("model_name", test_models)
def test_causal_lm_pytorch_vs_kv_vs_ort_vs_ai100(model_name):
    """
    Test function to validate the PyTorch model, the PyTorch model after KV changes, the ONNX model, and the Cloud AI 100 model, both with and without continuous batching.
    ``Mandatory`` Args:
        :model_name (str): Hugging Face Model Card name, Example: ``gpt2``
    """

    n_layer_text = 1
    n_layer_vision = 1

    check_vlm_pytorch_vs_kv_vs_ort_vs_ai100(
        model_name=model_name, n_layer_text=n_layer_text, n_layer_vision=n_layer_vision
    )

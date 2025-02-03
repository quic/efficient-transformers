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
import torch.nn as nn
import torchvision.transforms as T

# For intern Specific
from PIL import Image
from torchvision.transforms.functional import InterpolationMode
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

from QEfficient.transformers.models.InternVL import get_conv_template

# from QEfficient.exporter.export_hf_to_cloud_ai_100 import qualcomm_efficient_converter
from QEfficient.transformers.models.modeling_auto import QEFFAutoModelForImageTextToText

# from QEfficient.transformers.quantizers.auto import replace_transformers_quantizers
from QEfficient.utils import hf_download

# from QEfficient.utils._utils import load_hf_processor
from QEfficient.utils.constants import Constants
from QEfficient.utils.device_utils import get_available_device_id

test_models = [
    "llava-hf/llava-1.5-7b-hf",
    "OpenGVLab/InternVL2_5-1B",
]

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


class InternProcessor:
    def __init__(self, model: nn.Module, tokenizer):
        self.model = model
        image_size = self.model.config.force_image_size or self.model.config.vision_config.image_size
        patch_size = self.model.config.vision_config.patch_size
        self.template = model.config.template
        self.conv_template = get_conv_template(self.template)
        self.system_message = self.conv_template.system_message
        self.num_image_token = int((image_size // patch_size) ** 2 * (self.model.config.downsample_ratio**2))
        self.tokenizer = tokenizer

    def build_transform(self, input_size):
        MEAN, STD = IMAGENET_MEAN, IMAGENET_STD
        transform = T.Compose(
            [
                T.Lambda(lambda img: img.convert("RGB") if img.mode != "RGB" else img),
                T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
                T.ToTensor(),
                T.Normalize(mean=MEAN, std=STD),
            ]
        )
        return transform

    def find_closest_aspect_ratio(self, aspect_ratio, target_ratios, width, height, image_size):
        best_ratio_diff = float("inf")
        best_ratio = (1, 1)
        area = width * height
        for ratio in target_ratios:
            target_aspect_ratio = ratio[0] / ratio[1]
            ratio_diff = abs(aspect_ratio - target_aspect_ratio)
            if ratio_diff < best_ratio_diff:
                best_ratio_diff = ratio_diff
                best_ratio = ratio
            elif ratio_diff == best_ratio_diff:
                if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                    best_ratio = ratio
        return best_ratio

    def dynamic_preprocess(self, image, min_num=1, max_num=12, image_size=448, use_thumbnail=False):
        orig_width, orig_height = image.size
        aspect_ratio = orig_width / orig_height

        # calculate the existing image aspect ratio
        target_ratios = set(
            (i, j)
            for n in range(min_num, max_num + 1)
            for i in range(1, n + 1)
            for j in range(1, n + 1)
            if i * j <= max_num and i * j >= min_num
        )
        target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])

        # find the closest aspect ratio to the target
        target_aspect_ratio = self.find_closest_aspect_ratio(
            aspect_ratio, target_ratios, orig_width, orig_height, image_size
        )

        # calculate the target width and height
        target_width = image_size * target_aspect_ratio[0]
        target_height = image_size * target_aspect_ratio[1]
        blocks = target_aspect_ratio[0] * target_aspect_ratio[1]

        # resize the image
        resized_img = image.resize((target_width, target_height))
        processed_images = []
        for i in range(blocks):
            box = (
                (i % (target_width // image_size)) * image_size,
                (i // (target_width // image_size)) * image_size,
                ((i % (target_width // image_size)) + 1) * image_size,
                ((i // (target_width // image_size)) + 1) * image_size,
            )
            # split the image
            split_img = resized_img.crop(box)
            processed_images.append(split_img)
        assert len(processed_images) == blocks
        if use_thumbnail and len(processed_images) != 1:
            thumbnail_img = image.resize((image_size, image_size))
            processed_images.append(thumbnail_img)
        return processed_images

    def load_image(self, image_file, input_size=448, max_num=12):
        image = Image.open(image_file).convert("RGB")
        # import ipdb; ipdb.set_trace()
        transform = self.build_transform(input_size=input_size)
        images = self.dynamic_preprocess(image, image_size=input_size, use_thumbnail=True, max_num=max_num)
        pixel_values = [transform(image) for image in images]
        pixel_values = torch.stack(pixel_values)
        return pixel_values

    def __call__(
        self,
        tokenizer,
        pixel_values,
        question,
        history=None,
        return_history=False,
        num_patches_list=None,
        IMG_START_TOKEN="<img>",
        IMG_END_TOKEN="</img>",
        IMG_CONTEXT_TOKEN="<IMG_CONTEXT>",
        verbose=False,
    ) -> str:
        if history is None and pixel_values is not None and "<image>" not in question:
            question = "<image>\n" + question

        if num_patches_list is None:
            num_patches_list = [pixel_values.shape[0]] if pixel_values is not None else []
        assert pixel_values is None or len(pixel_values) == sum(num_patches_list)

        img_context_token_id = tokenizer.convert_tokens_to_ids(IMG_CONTEXT_TOKEN)
        self.model.img_context_token_id = img_context_token_id

        template = get_conv_template(self.template)
        template.system_message = self.system_message

        history = [] if history is None else history
        for old_question, old_answer in history:
            template.append_message(template.roles[0], old_question)
            template.append_message(template.roles[1], old_answer)
        template.append_message(template.roles[0], question)
        template.append_message(template.roles[1], None)
        query = template.get_prompt()

        if verbose and pixel_values is not None:
            image_bs = pixel_values.shape[0]
            print(f"dynamic ViT batch size: {image_bs}")

        for num_patches in num_patches_list:
            image_tokens = IMG_START_TOKEN + IMG_CONTEXT_TOKEN * self.num_image_token * num_patches + IMG_END_TOKEN
            query = query.replace("<image>", image_tokens, 1)

        return query


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
        config.llm_config.num_hidden_layers = model_config["n_layer_text"]
        config.vision_config.num_hidden_layers = model_config["n_layer_vision"]
        config.llm_config._attn_implementation = "eager"
        config.vision_config.use_flash_attn = "false"
        model_hf = AutoModelForCausalLM.from_pretrained(
            model_config["model_path"], low_cpu_mem_usage=False, config=config
        )
    elif model_config["model_name"] == "llava-hf/llava-1.5-7b-hf":
        config = AutoConfig.from_pretrained(model_config["model_path"])
        config.text_config.num_hidden_layers = model_config["n_layer_text"]
        config.vision_config.num_hidden_layers = model_config["n_layer_vision"]
        config._attn_implementation = "eager"
        config.vision_config.use_flash_attn = "false"
        model_hf = AutoModelForImageTextToText.from_pretrained(
            model_config["model_path"], low_cpu_mem_usage=False, config=config
        )
    elif model_config["model_name"] == "HuggingFaceTB/SmolVLM-256M-Instruct":
        config = AutoConfig.from_pretrained(model_config["model_path"])
        config.text_config.num_hidden_layers = model_config["n_layer_text"]
        config.vision_config.num_hidden_layers = model_config["n_layer_vision"]
        config._attn_implementation = "eager"
        config.vision_config.use_flash_attn = "false"
        model_hf = AutoModelForVision2Seq.from_pretrained(
            model_config["model_path"], low_cpu_mem_usage=False, config=config
        )
    params = sum(p.numel() for p in model_hf.parameters())
    model_hf.eval()
    return model_hf, params


def generate_hf_inputs_smol(model_name, model, processor):
    image = load_image("https://cdn.britannica.com/61/93061-050-99147DCE/Statue-of-Liberty-Island-New-York-Bay.jpg")
    messages = [
        {"role": "user", "content": [{"type": "image"}, {"type": "text", "text": "Can you describe this image?"}]},
    ]
    # Prepare inputs
    prompt = processor.apply_chat_template(messages, add_generation_prompt=True)
    inputs = processor(text=prompt, images=[image], return_tensors="pt")
    return inputs


def generate_hf_inputs_intern(model_name, model, processor):
    ## PREPROCESSING THE MULTI-MODAL INPUTS
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
    # inputs["processor"] = processor
    return inputs


# ---------------------------------------------
# Please Add new models here inside the map
# {model_name:generate_hf_inputs_<model_name>}
# ---------------------------------------------
generate_hf_inputs_func_map = {
    "llava-hf/llava-1.5-7b-hf": generate_hf_inputs_llava,
    "OpenGVLab/InternVL2_5-1B": generate_hf_inputs_intern,
    "HuggingFaceTB/SmolVLM-256M-Instruct": generate_hf_inputs_smol,
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

    qeff_model = QEFFAutoModelForImageTextToText(model_hf, processor, is_tlm=False)

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


# if __name__ == "__main__":
#     # model_name = "OpenGVLab/InternVL2_5-1B"
#     # model_name="llava-hf/llava-1.5-7b-hf"
#     # model_name="HuggingFaceTB/SmolVLM-256M-Instruct"
#     check_vlm_pytorch_vs_kv_vs_ort_vs_ai100(model_name=model_name)

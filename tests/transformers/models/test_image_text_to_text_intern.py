# -----------------------------------------------------------------------------
#
# Copyright (c) 2025 Qualcomm Innovation Center, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------

import numpy as np
import pytest
import torch
import torch.nn as nn
import torchvision.transforms as T
from PIL import Image
from torchvision.transforms.functional import InterpolationMode
from transformers import AutoConfig, AutoTokenizer, TextStreamer

from conversation import get_conv_template
from QEfficient import QEFFAutoModelForCausalLM

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
        transform = self.build_transform(input_size=input_size)
        images = self.dynamic_preprocess(image, image_size=input_size, use_thumbnail=True, max_num=max_num)
        pixel_values = [transform(image) for image in images]
        pixel_values = torch.stack(pixel_values)
        return pixel_values

    def __call__(
        self,
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
        img_context_token_id = self.tokenizer.convert_tokens_to_ids(IMG_CONTEXT_TOKEN)
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


@pytest.mark.on_qaic
def test_image_text_to_text_intern():
    model_name = "OpenGVLab/InternVL2_5-1B"
    config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)  # noqa: F841
    # config.llm_config.num_hidden_layers = 1
    # config.vision_config.num_hidden_layers = 1
    # model = QEFFAutoModelForCausalLM.from_pretrained(model_name, kv_offload=False, config=config, trust_remote_code=True)
    model = QEFFAutoModelForCausalLM.from_pretrained(model_name, kv_offload=False, trust_remote_code=True)

    model.export()
    model.compile(num_cores=14)

    ### Pytorch execution
    qeff_pt_model = model.model

    prompt = "Please describe the image and generate a short story around it"
    ctx_len = 4096
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True, use_fast=False)

    internProcessor = InternProcessor(qeff_pt_model, tokenizer)
    pixel_values = internProcessor.load_image(
        "/local/mnt/workspace/open-source/efficient-transformers/image1.jpg", max_num=12
    )
    question = "<image>\n" + prompt
    query = internProcessor(pixel_values, question)
    pad_inputs = tokenizer(query, return_tensors="pt", padding="max_length", max_length=3840, padding_side="right")

    inputs = tokenizer(query, return_tensors="pt")
    inputs = dict(inputs)

    batch_size, prompt_len = inputs["input_ids"].shape
    inputs["pixel_values"] = pixel_values.clone()
    pad_inputs["pixel_values"] = pixel_values.clone()
    import copy  # noqa: E402

    orig_inputs = copy.deepcopy(pad_inputs)
    inputs["position_ids"] = torch.arange(prompt_len).view(1, -1)
    inputs.pop("attention_mask")

    head_dim = (
        qeff_pt_model.language_model.config.hidden_size // qeff_pt_model.language_model.config.num_attention_heads
    )
    inputs["past_key_values"] = [
        tuple(
            [
                torch.zeros(
                    batch_size,
                    qeff_pt_model.language_model.config.num_key_value_heads,
                    ctx_len,
                    head_dim,
                    dtype=torch.float32,
                )
                for _ in range(2)
            ]
        )
        for _ in range(qeff_pt_model.language_model.config.num_hidden_layers)
    ]

    streamer = TextStreamer(tokenizer)
    generation_len = 10
    generated_ids = np.full((batch_size, generation_len + 1), tokenizer.pad_token_id)
    pt_outputs = qeff_pt_model(**inputs)
    inputs["input_ids"] = pt_outputs[0].argmax(2)
    inputs["position_ids"] = inputs["position_ids"].max(1, keepdim=True).values + 1
    streamer.put(inputs["input_ids"])
    generated_ids[:, 0] = inputs["input_ids"].squeeze(1)
    finished_sequences = inputs["input_ids"] == tokenizer.eos_token_id
    for i in range(1, generation_len):
        outputs = qeff_pt_model(**inputs)
        inputs["input_ids"] = outputs[0].argmax(2)
        print(inputs["input_ids"])
        # print(tokenizer.decode(inputs["input_ids"]))
        inputs["position_ids"] += 1
        generated_ids[:, i] = inputs["input_ids"].squeeze(1)
        finished_sequences |= inputs["input_ids"] == tokenizer.eos_token_id
        if finished_sequences.all():
            break

    streamer.end()

    generated_texts = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
    print(generated_texts)

    exec_info = model.generate(inputs=orig_inputs, generation_len=128)
    print(exec_info)
    generated_ids_aic = exec_info.generated_ids
    print(generated_ids_aic)
    generated_texts = tokenizer.batch_decode(generated_ids_aic, skip_special_tokens=True)
    print(generated_texts)

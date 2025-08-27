# -----------------------------------------------------------------------------
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------

from io import BytesIO
from typing import List

import requests
import torch
import torch.nn as nn
import torchvision.transforms as T
from PIL import Image
from torchvision.transforms.functional import InterpolationMode
from transformers import AutoTokenizer, TextStreamer

import decord
import numpy as np
from decord import VideoReader, cpu

from QEfficient import QEFFAutoModelForCausalLM
from QEfficient.utils.logging_utils import logger

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


# Process the input messages to generate prompt for the model.
def get_prompt(messages) -> str:
    """Get the prompt for generation."""
    ## Chat template used for InternVL
    system_prompt = "<|im_start|>system\n你是书生·万象，英文名是InternVL，是由上海人工智能实验室、清华大学及多家合作单位联合开发的多模态大语言模型。"
    sep = "<|im_end|>\n"

    ret = system_prompt + sep
    for role, message in messages:
        if message:
            if type(message) is tuple:
                message, _, _ = message
            ret += role + message + sep
        else:
            ret += role
    return ret


# Processor class for InternVL models
class InternProcessor:
    """
    InternVL model only has an AutoTokenizer so this class performs the processing tasks similar to an AutoProcessor.
    The methods used here are borrowed from the original InternVL modelling files.
    "https://huggingface.co/OpenGVLab/InternVL2_5-1B/"
    """

    def __init__(self, model: nn.Module, tokenizer):
        self.model = model
        image_size = self.model.config.force_image_size or self.model.config.vision_config.image_size
        patch_size = self.model.config.vision_config.patch_size
        self.template = model.config.template
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

    def load_image(self, image, input_size=448, max_num=12):
        transform = self.build_transform(input_size=input_size)
        images = self.dynamic_preprocess(image, image_size=input_size, use_thumbnail=True, max_num=max_num)
        pixel_values = [transform(image) for image in images]
        pixel_values = torch.stack(pixel_values)
        return pixel_values

    def get_index(self, bound, fps, max_frame, first_idx=0, num_segments=13):
        start, end = -100000, 100000
        start_idx = max(first_idx, round(start * fps))
        end_idx = min(round(end * fps), max_frame)
        seg_size = float(end_idx - start_idx) / num_segments
        frame_indices = np.array([
            int(start_idx + (seg_size / 2) + np.round(seg_size * idx))
            for idx in range(num_segments)
        ])
        return frame_indices

    def load_video(self, video_path:str, bound=None, input_size=448, max_num=1, num_segments=13):
        vr = VideoReader(video_path, ctx=cpu(0)) 
        max_frame = len(vr) - 1
        fps = float(vr.get_avg_fps())
        pixel_values_list, num_patches_list = [], []
        transform = self.build_transform(input_size=input_size) 
        frame_indices = self.get_index(bound, fps, max_frame, first_idx=0, num_segments=num_segments) 
        for frame_index in frame_indices:
            img = Image.fromarray(vr[frame_index].asnumpy()).convert('RGB')
            img = self.dynamic_preprocess(img, image_size=input_size, use_thumbnail=True, max_num=max_num) 
            pixel_values = [transform(tile) for tile in img]
            pixel_values = torch.stack(pixel_values)
            num_patches_list.append(pixel_values.shape[0])
            pixel_values_list.append(pixel_values)
        pixel_values = torch.cat(pixel_values_list)
        return pixel_values, num_patches_list

    def __call__(
        self,
        pixel_values,
        question,
        messages,
        roles,
        history=None,
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

        messages.append([roles[0], question])
        messages.append([roles[1], None])
        query = get_prompt(messages)
        if verbose and pixel_values is not None:
            image_bs = pixel_values.shape[0]
            logger.info(f"dynamic ViT batch size: {image_bs}")

        for num_patches in num_patches_list:
            image_tokens = IMG_START_TOKEN + IMG_CONTEXT_TOKEN * self.num_image_token * num_patches + IMG_END_TOKEN
            query = query.replace("<image>", image_tokens, 1)
        return query


def run_intern_on_aic(
    model_name,
    prompt,
    messages,
    roles,
    kv_offload=False,
    prefill_seq_len=3840,
    num_devices=1,
    num_cores=16,
    multi_frame_inference=False,
    image_url=None,
    video_path=None,
):
    ## STEP 1 -- LOAD THE MODEL

    # The original Intern-VL model, despite being multimodal, is loaded using `AutoModelForCausalLM` in Huggingface.
    # To maintain compatibility, we load this model using `QEFFAutoModelForCausalLM`.

    model = QEFFAutoModelForCausalLM.from_pretrained(model_name, kv_offload=kv_offload, trust_remote_code=True)

    ## STEP 2 -- EXPORT & COMPILE THE MODEL

    model.compile(
        num_cores=num_cores,
        num_devices=num_devices,
        prefill_seq_len=prefill_seq_len,
        mxfp6_matmul=True,
        mxint8_kv_cache=True,
        allow_mxint8_mdp_io=True,
        aic_enable_depth_first=True,
    )

    ## STEP 3 -- SETUP THE PROCESSOR

    # InternVL doesn't have an AutoProcessor yet, so we will use our own processor class "InternProcessor"
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True, use_fast=False)
    internProcessor = InternProcessor(model.model, tokenizer)

    ## STEP 4 -- PREPROCESS THE INPUTS
    if multi_frame_inference:
        pixel_values, num_patches_list = internProcessor.load_video(video_path)
        video_prefix = ''.join([f'Frame{i+1}: <image>\n' for i in range(len(num_patches_list))])
        question = video_prefix + prompt
    else:
        response = requests.get(image_url, stream=True)
        img = Image.open(BytesIO(response.content)).convert("RGB")
        # img = Image.open(image_url).convert("RGB")
        # Images are resized to (1000, 747) for inference
        image = img.resize((1000, 747))
        # preprocess the resized image
        pixel_values = internProcessor.load_image(image, max_num=12)
        question = "<image>\n" + prompt

    query = internProcessor(pixel_values, question, messages, roles)
    inputs = tokenizer(
        query, return_tensors="pt", padding="max_length", max_length=prefill_seq_len, padding_side="right"
    )

    inputs["pixel_values"] = pixel_values

    ## STEP 5 -- RUN INFERENCE VIA GENERATE FUNCTION
    streamer = TextStreamer(tokenizer)
    if kv_offload:
        outputs=model.generate(inputs=inputs, streamer=streamer,device_id_lang=[16,17,18,19], device_id_vision=[20,21,22,23], generation_len=128)
    else:    
        outputs=model.generate(inputs=inputs, streamer=streamer,device_ids=[24,25,26,27], generation_len=128)
    print(outputs)


if __name__ == "__main__":
    model_name = "OpenGVLab/InternVL3-8B"

    # Chat Template information for prompt preprocessing
    messages: List[List[str]] = []
    roles = ("<|im_start|>user\n", "<|im_start|>assistant\n")

    ## Compilation parameters

    # `kv_offload` is used to compile the model in a Single QPC or 2 QPCs.
    # The Dual QPC approach splits the model to perform Image Encoding and Output generation in 2 different QPCs.
    # The outputs of the Vision Encoder are then passed to the Language model via host in this case.

    kv_offload = True
    multi_frame_inference=True

    # InternVL is an Early-Fusion model that uses placeholder tokens within the input_ids to interleave text_embeddings with
    # Image embeddings and generate final input_embeds for outout generation. Hence we need very large prefill_seq_len (3840 in this case) to
    # incorporate the memory for the merged embeddings.

    prefill_seq_len = 3840
    num_devices = 4
    num_cores = 16

    # Inputs for the model
    if multi_frame_inference:
        video_path = "/local/mnt/workspace/aditjadh/aisyssol/red-panda.mp4"
        prompt="What is happening in this video" 
        run_intern_on_aic(
            model_name=model_name,
            prompt=prompt,
            messages=messages,
            roles=roles,
            kv_offload=kv_offload,
            prefill_seq_len=prefill_seq_len,
            num_devices=num_devices,
            num_cores=num_cores,
            multi_frame_inference=multi_frame_inference,
            video_path=video_path,
        )
    else:
        image_url = "https://image.slidesharecdn.com/azureintroduction-191206101932/75/Introduction-to-Microsoft-Azure-Cloud-1-2048.jpg"
        prompt="Describe the image"
        run_intern_on_aic(
            model_name=model_name,
            prompt=prompt,
            image_url=image_url,
            messages=messages,
            roles=roles,
            kv_offload=kv_offload,
            prefill_seq_len=prefill_seq_len,
            num_devices=num_devices,
            num_cores=num_cores,
            multi_frame_inference=multi_frame_inference,
        )


"""
Expected Response:

The image is a promotional graphic for Microsoft Azure. It features a blue background with a hexagonal pattern on the left side. The hexagons are white and are arranged in a way that suggests a network or connectivity theme. 

On the right side of the image, the Microsoft Azure logo is prominently displayed. The logo consists of the Azure name in white, with the Microsoft logo above it, which includes four colored squares (blue, green, yellow, and red). Below the logo, the word "Azure" is written in large white letters.

Below the logo, there is text that reads:
- "By Dinesh Kumar Wick
"""

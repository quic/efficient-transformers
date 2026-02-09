# -----------------------------------------------------------------------------
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
#
# ----------------------------------------------------------------------------

from io import BytesIO
from typing import List, Optional

import requests
import torch
from PIL import Image
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoModelForImageTextToText,
    AutoProcessor,
    AutoTokenizer,
    TextStreamer,
)

from QEfficient.transformers.models.modeling_auto import QEFFAutoModelForCausalLM, QEFFAutoModelForImageTextToText
from QEfficient.utils import hf_download
from QEfficient.utils._utils import get_num_layers_vlm
from QEfficient.utils.test_utils import InternProcessor

NEW_GENERATION_TOKENS = 10


def load_image_text_to_text_model(model_config):
    model_path = hf_download(
        repo_id=model_config._name_or_path,
        ignore_patterns=["*.onnx", "*.ot", "*.md", "*.tflite", "*.pdf", "*.h5", "*.msgpack"],
    )
    try:
        model_hf = AutoModelForImageTextToText.from_pretrained(
            model_path,
            low_cpu_mem_usage=False,
            config=model_config,
        )
    except ValueError:
        model_hf = AutoModelForCausalLM.from_pretrained(
            model_path,
            low_cpu_mem_usage=False,
            trust_remote_code=True,
            config=model_config,
        )
    params = sum(p.numel() for p in model_hf.parameters())
    model_hf.eval()
    return model_hf, params


def set_num_layers(config, n_layer=1):
    ## -1 indicates use all the layers of the model.
    if n_layer == -1:
        return config
    elif hasattr(config, "model_type") and "mllama" in config.model_type:
        config.text_config.num_hidden_layers = n_layer
        config.text_config.cross_attention_layers = [
            x for x in config.text_config.cross_attention_layers if x < n_layer
        ]
    elif hasattr(config, "text_config"):
        config.text_config.num_hidden_layers = n_layer
        config.vision_config.num_hidden_layers = n_layer
    elif hasattr(config, "llm_config"):
        config.llm_config.num_hidden_layers = n_layer
        config.vision_config.num_hidden_layers = n_layer
    else:
        config.num_hidden_layers = n_layer
    return config


def check_image_text_to_text_pytorch_vs_kv_vs_ort_vs_ai100(
    model_name: str = "llava-hf/llava-1.5-7b-hf",
    img_url: str = "https://huggingface.co/datasets/huggingface/documentation-images/resolve/0052a70beed5bf71b92610a43a52df6d286cd5f3/diffusers/rabbit.jpg",
    query: str = "Describe this image.",
    prompt_len: int = 128,
    ctx_len: int = 2048,
    max_gen_len: int = 20,
    batch_size: int = 1,
    n_layer: int = 1,
    kv_offload: bool = True,
    num_devices: int = 1,
    enable_qnn: Optional[bool] = False,
    qnn_config: Optional[str] = None,
    config: Optional[AutoConfig] = None,
    img_size: Optional[int] = None,
):
    """
    Unified function to test PyTorch model, PyTorch KV model, ONNX model, and Cloud AI 100 model.
    Handles standard VLM models, InternVL models, and Molmo models.

    Args:
        model_name: Hugging Face model identifier
        img_url: URL to image for testing
        query: Text query for the model
        prompt_len: Prompt sequence length
        ctx_len: Context length
        max_gen_len: Maximum generation length
        batch_size: Batch size for processing
        n_layer: Number of layers to use
        kv_offload: Whether to use KV offloading
        num_devices: Number of devices to use
        enable_qnn: Enable QNN compilation
        qnn_config: Path to QNN config file
        config: Pre-configured model config (optional)
        img_size: Image size for standard models (optional)
    """

    is_intern_model = model_name == "OpenGVLab/InternVL2_5-1B" or model_name == "OpenGVLab/InternVL3_5-1B"
    is_molmo_model = model_name == "allenai/Molmo-7B-D-0924"

    # ========== Config and Model Loading ==========
    if config is None:
        config = AutoConfig.from_pretrained(model_name, trust_remote_code=True, padding=not is_molmo_model)
        config._attn_implementation = "eager" if (is_intern_model or is_molmo_model) else None
        config = set_num_layers(config, n_layer=n_layer)

    if is_intern_model:
        model_hf = AutoModelForCausalLM.from_pretrained(
            model_name,
            low_cpu_mem_usage=False,
            trust_remote_code=True,
            config=config,
        )
        n_layer = get_num_layers_vlm(config)

    elif is_molmo_model:
        model_hf, _ = load_image_text_to_text_model(config)
        n_layer = (n_layer, n_layer)
    else:
        model_hf, _ = load_image_text_to_text_model(config)
        n_layer = get_num_layers_vlm(config)

    print("\n============HF Model===============\n")
    print(model_hf)
    print("\n=====================================\n")
    # ========== Processor and Image Loading ==========
    if is_intern_model:
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True, use_fast=False)
        processor = InternProcessor(model_hf, tokenizer)
    else:
        processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True, padding=True)

    if is_intern_model:
        prompt = [query]
        img_url_list = [img_url]
        pixel_values = []
        num_patches_list = []
        questions = []
        for i in range(len(prompt)):
            img = requests.get(img_url_list[i], stream=True)
            image = Image.open(BytesIO(img.content)).convert("RGB")
            image = image.resize((448, 448))
            pixel_value = processor.load_image(image, max_num=12)
            num_patches_list.append(pixel_value.shape[0])
            pixel_values.append(pixel_value)
            question = "<image>\n" + prompt[i]
            questions.append(question)
        pixel_values = torch.cat(pixel_values, dim=0)
    else:
        if is_molmo_model:
            img = requests.get(img_url, stream=True)
            image = Image.open(BytesIO(img.content)).convert("RGB")
            image = image.resize((536, 354))
        else:
            image = Image.open(requests.get(img_url, stream=True).raw)
            if model_name == "mistralai/Mistral-Small-3.1-24B-Instruct-2503":
                image = image.resize((1540, 1540))

    # ========== Prepare Inputs and Get PyTorch HF Tokens ==========
    if is_intern_model:
        messages: List[List[str]] = []
        roles = ("<|im_start|>user\n", "<|im_start|>assistant\n")
        prompt = processor(pixel_values, questions, messages, roles, num_patches_list=num_patches_list)
        inputs = tokenizer(prompt, return_tensors="pt")
        batch_size, prompt_len = inputs["input_ids"].shape
        inputs["pixel_values"] = pixel_values.clone()

    elif is_molmo_model:
        inputs = processor.process(images=[image], text=query)
        inputs = {k: v.unsqueeze(0) for k, v in inputs.items()}
        batch_size, prompt_len = inputs["input_ids"].shape
        inputs["attention_mask"] = torch.ones((inputs["input_ids"].shape), dtype=torch.int64)
        valid = inputs["image_input_idx"] > 0
        valid = valid.reshape(1, -1)
        inputs["valid_idx"] = torch.nonzero(valid)[:, 1].unsqueeze(0)
        inputs["pixel_values"] = inputs.pop("images")
    else:
        conversation = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": query},
                    {"type": "image"},
                ],
            },
        ]
        prompt = processor.apply_chat_template(conversation, add_generation_prompt=True)

        inputs = processor(images=image, text=prompt, return_tensors="pt")
        if "pixel_values" in inputs:
            inputs["pixel_values"] = inputs["pixel_values"].to(torch.float32)

    streamer = TextStreamer(processor.tokenizer)

    # ========== Export and Compile Model ==========
    if is_intern_model or is_molmo_model:
        qeff_model = QEFFAutoModelForCausalLM.from_pretrained(
            model_name,
            kv_offload=kv_offload,
            config=config,
            enable_proxy=True,
        )
    else:
        qeff_model = QEFFAutoModelForImageTextToText(model_hf, kv_offload=kv_offload, enable_proxy=True)
    print("\n============Qeff Proxy Model===============\n")
    print(qeff_model.model)
    print("\n=====================================\n")
    qeff_model.export()

    compile_kwargs = {
        "num_devices": num_devices,
        "prefill_seq_len": prompt_len,
        "ctx_len": ctx_len,
        "mxfp6": False,
        "enable_qnn": enable_qnn,
        "qnn_config": qnn_config,
    }

    if is_intern_model:
        compile_kwargs["num_patches"] = 1
    elif not is_molmo_model and img_size is not None:
        compile_kwargs["img_size"] = img_size

    qeff_model.compile(**compile_kwargs)

    # ========== Generate and Verify Output ==========

    if not is_intern_model and not is_molmo_model:
        inputs = processor(images=image, text=prompt, return_tensors="pt")
        if hasattr(qeff_model.model.config, "model_type") and qeff_model.model.config.model_type == "qwen2_5_vl":
            inputs = qeff_model.model.prepare_inputs_for_generation(
                inputs=inputs, prefill_seq_len=prompt_len, batch_size=batch_size
            )
        if "pixel_values" in inputs:
            inputs["pixel_values"] = inputs["pixel_values"].to(torch.float32)

    print("QPC Outputs (QAIC):")
    qeff_model.generate(
        inputs=inputs,
        generation_len=NEW_GENERATION_TOKENS,
        streamer=streamer,
        write_io=True,
    )


image_models = [
    "llava-hf/llava-1.5-7b-hf",
    # "meta-llama/Llama-3.2-11B-Vision-Instruct",
    # "meta-llama/Llama-3.2-90B-Vision-Instruct",
    # "ibm-granite/granite-vision-3.2-2b",
    # "Llama-4-Scout-17B-16E-Instruct",
    # "google/gemma-3-4b-it",
    # "Qwen/Qwen2.5-VL-3B-Instruct",
    # "mistralai/Mistral-Small-3.1-24B-Instruct-2503",
]

for model_name in image_models:
    print(f"\n\nTesting model: {model_name}")
    check_image_text_to_text_pytorch_vs_kv_vs_ort_vs_ai100(model_name=model_name)


"""
Testing model: llava-hf/llava-1.5-7b-hf

============HF Model===============

LlavaForConditionalGeneration(
  (model): LlavaModel(
    (vision_tower): CLIPVisionModel(
      (vision_model): CLIPVisionTransformer(
        (embeddings): CLIPVisionEmbeddings(
          (patch_embedding): Conv2d(3, 1024, kernel_size=(14, 14), stride=(14, 14), bias=False)
          (position_embedding): Embedding(577, 1024)
        )
        (pre_layrnorm): LayerNorm((1024,), eps=1e-05, elementwise_affine=True)
        (encoder): CLIPEncoder(
          (layers): ModuleList(
            (0-23): 24 x CLIPEncoderLayer(
              (self_attn): CLIPAttention(
                (k_proj): Linear(in_features=1024, out_features=1024, bias=True)
                (v_proj): Linear(in_features=1024, out_features=1024, bias=True)
                (q_proj): Linear(in_features=1024, out_features=1024, bias=True)
                (out_proj): Linear(in_features=1024, out_features=1024, bias=True)
              )
              (layer_norm1): LayerNorm((1024,), eps=1e-05, elementwise_affine=True)
              (mlp): CLIPMLP(
                (activation_fn): QuickGELUActivation()
                (fc1): Linear(in_features=1024, out_features=4096, bias=True)
                (fc2): Linear(in_features=4096, out_features=1024, bias=True)
              )
              (layer_norm2): LayerNorm((1024,), eps=1e-05, elementwise_affine=True)
            )
          )
        )
        (post_layernorm): LayerNorm((1024,), eps=1e-05, elementwise_affine=True)
      )
    )
    (multi_modal_projector): LlavaMultiModalProjector(
      (linear_1): Linear(in_features=1024, out_features=4096, bias=True)
      (act): GELUActivation()
      (linear_2): Linear(in_features=4096, out_features=4096, bias=True)
    )
    (language_model): LlamaModel(
      (embed_tokens): Embedding(32064, 4096)
      (layers): ModuleList(
        (0-31): 32 x LlamaDecoderLayer(
          (self_attn): LlamaAttention(
            (q_proj): Linear(in_features=4096, out_features=4096, bias=False)
            (k_proj): Linear(in_features=4096, out_features=4096, bias=False)
            (v_proj): Linear(in_features=4096, out_features=4096, bias=False)
            (o_proj): Linear(in_features=4096, out_features=4096, bias=False)
          )
          (mlp): LlamaMLP(
            (gate_proj): Linear(in_features=4096, out_features=11008, bias=False)
            (up_proj): Linear(in_features=4096, out_features=11008, bias=False)
            (down_proj): Linear(in_features=11008, out_features=4096, bias=False)
            (act_fn): SiLU()
          )
          (input_layernorm): LlamaRMSNorm((4096,), eps=1e-05)
          (post_attention_layernorm): LlamaRMSNorm((4096,), eps=1e-05)
        )
      )
      (norm): LlamaRMSNorm((4096,), eps=1e-05)
      (rotary_emb): LlamaRotaryEmbedding()
    )
  )
  (lm_head): Linear(in_features=4096, out_features=32064, bias=False)
)

=====================================

============Qeff Model===============

_QEFFAutoModelForImageTextToTextSingleQPC
QEffLlavaForConditionalGeneration(
  (model): LlavaModel(
    (vision_tower): CLIPVisionModel(
      (vision_model): CLIPVisionTransformer(
        (embeddings): CLIPVisionEmbeddings(
          (patch_embedding): Conv2d(3, 1024, kernel_size=(14, 14), stride=(14, 14), bias=False)
          (position_embedding): Embedding(577, 1024)
        )
        (pre_layrnorm): LayerNorm((1024,), eps=1e-05, elementwise_affine=True)
        (encoder): CLIPEncoder(
          (layers): ModuleList(
            (0-23): 24 x CLIPEncoderLayer(
              (self_attn): CLIPAttention(
                (k_proj): Linear(in_features=1024, out_features=1024, bias=True)
                (v_proj): Linear(in_features=1024, out_features=1024, bias=True)
                (q_proj): Linear(in_features=1024, out_features=1024, bias=True)
                (out_proj): Linear(in_features=1024, out_features=1024, bias=True)
              )
              (layer_norm1): LayerNorm((1024,), eps=1e-05, elementwise_affine=True)
              (mlp): CLIPMLP(
                (activation_fn): QuickGELUActivation()
                (fc1): Linear(in_features=1024, out_features=4096, bias=True)
                (fc2): Linear(in_features=4096, out_features=1024, bias=True)
              )
              (layer_norm2): LayerNorm((1024,), eps=1e-05, elementwise_affine=True)
            )
          )
        )
        (post_layernorm): LayerNorm((1024,), eps=1e-05, elementwise_affine=True)
      )
    )
    (multi_modal_projector): LlavaMultiModalProjector(
      (linear_1): Linear(in_features=1024, out_features=4096, bias=True)
      (act): GELUActivation()
      (linear_2): Linear(in_features=4096, out_features=4096, bias=True)
    )
    (language_model): QEffLlamaModel(
      (embed_tokens): Embedding(32064, 4096)
      (layers): ModuleList(
        (0-31): 32 x QEffLlamaDecoderLayer(
          (self_attn): QEffLlamaAttention(
            (q_proj): Linear(in_features=4096, out_features=4096, bias=False)
            (k_proj): Linear(in_features=4096, out_features=4096, bias=False)
            (v_proj): Linear(in_features=4096, out_features=4096, bias=False)
            (o_proj): Linear(in_features=4096, out_features=4096, bias=False)
            (rotary_emb): QEffLlamaRotaryEmbedding()
          )
          (mlp): LlamaMLP(
            (gate_proj): Linear(in_features=4096, out_features=11008, bias=False)
            (up_proj): Linear(in_features=4096, out_features=11008, bias=False)
            (down_proj): Linear(in_features=11008, out_features=4096, bias=False)
            (act_fn): SiLU()
          )
          (input_layernorm): CustomRMSNormAIC()
          (post_attention_layernorm): CustomRMSNormAIC()
        )
      )
      (norm): CustomRMSNormAIC()
      (rotary_emb): QEffLlamaRotaryEmbedding()
    )
  )
  (lm_head): Linear(in_features=4096, out_features=32064, bias=False)
)

=====================================
"""

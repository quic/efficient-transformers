# -----------------------------------------------------------------------------
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
#
# ----------------------------------------------------------------------------

import copy
from io import BytesIO
from typing import List, Optional

import requests
import torch
from PIL import Image
from transformers import (
    AutoConfig,
    AutoProcessor,
    AutoTokenizer,
    GenerationConfig,
    TextStreamer,
)

from QEfficient import QEFFAutoModelForCausalLM, QEFFAutoModelForImageTextToText
from QEfficient.utils.run_utils import ApiRunnerInternVL, ApiRunnerMolmo, ApiRunnerVlm
from QEfficient.utils.test_utils import (
    InternProcessor,
    ModelConfig,
    load_vlm_model,
    load_vlm_model_from_config,
    set_num_layers_vlm,
)

from ..check_model_results import dump_and_compare_results


def check_image_text_to_text_pytorch_vs_kv_vs_ort_vs_ai100(
    model_name: str,
    num_hidden_layers: Optional[int] = -1,
    kv_offload: Optional[bool] = False,
    num_devices: Optional[int] = 1,
    enable_qnn: Optional[bool] = False,
    qnn_config: Optional[str] = None,
    config: Optional[AutoConfig] = None,
    torch_dtype: Optional[torch.dtype] = torch.float32,
    compare_results: Optional[bool] = False,
    export_compile_only: Optional[bool] = False,
):

    prompt_len = 128
    ctx_len = 4096
    img_size = 1540
    img_url = "https://picsum.photos/id/237/536/354"
    query = "Can you describe the image in detail."
    batch_size = 1
    max_gen_len = 20

    pytorch_kv_tokens = None
    ort_tokens = None
    n_layer = num_hidden_layers

    if config is None:
        config = AutoConfig.from_pretrained(
            model_name, trust_remote_code=True, padding=model_name not in ModelConfig.MOLMO_MODELS
        )
        config = set_num_layers_vlm(config, n_layer=n_layer)
        if hasattr(config, "model_type") and config.model_type in ["gemma3"]:
            config.text_config._sliding_window_pattern = 2
            config.text_config.layer_types = ["sliding_attention", "full_attention"]
        if hasattr(config, "model_type") and config.model_type in [
            "qwen3_vl",
            "qwen3_vl_moe",
        ]:
            config.vision_config.depth = 9
            config.text_config.num_hidden_layers = 1
            config.vision_config.deepstack_visual_indexes = [8]
        if model_name in ModelConfig.INTERNVL_MODELS or model_name in ModelConfig.MOLMO_MODELS:
            config._attn_implementation = "eager"
            model_hf = load_vlm_model(config)
            qeff_model = QEFFAutoModelForCausalLM.from_pretrained(
                model_name,
                kv_offload=kv_offload,
                config=config,
                torch_dtype=torch_dtype,
            )
        else:
            model_hf = load_vlm_model(config)
            qeff_model = QEFFAutoModelForImageTextToText.from_pretrained(
                model_name,
                kv_offload=kv_offload,
                config=config,
                torch_dtype=torch_dtype,
            )
    else:
        model_hf = load_vlm_model_from_config(config)
        qeff_model = QEFFAutoModelForImageTextToText(
            copy.deepcopy(model_hf),
            kv_offload=kv_offload,
            config=model_hf.config,
            torch_dtype=torch_dtype,
        )
    compile_kwargs = {
        "num_devices": num_devices,
        "prefill_seq_len": prompt_len,
        "ctx_len": ctx_len,
        "mxfp6": False,
        "enable_qnn": enable_qnn,
        "qnn_config": qnn_config,
    }

    if model_name in ModelConfig.INTERNVL_MODELS:
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True, use_fast=False)
        processor = InternProcessor(model_hf, tokenizer)
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
        messages: List[List[str]] = []
        roles = ("<|im_start|>user\n", "<|im_start|>assistant\n")
        prompt = processor(pixel_values, questions, messages, roles, num_patches_list=num_patches_list)
        inputs = tokenizer(prompt, return_tensors="pt")
        batch_size, prompt_len = inputs["input_ids"].shape
        inputs["pixel_values"] = pixel_values.clone()
        generation_config = dict(max_new_tokens=max_gen_len, do_sample=False)
        generation_config["eos_token_id"] = tokenizer.convert_tokens_to_ids("<|im_end|>\n".strip())
        api_runner = ApiRunnerInternVL(
            batch_size,
            processor,
            config,
            image,
            query,
            prompt_len,
            ctx_len,
            max_gen_len,
            num_hidden_layers,
        )
        pytorch_hf_tokens = api_runner.run_vlm_hf_model_on_pytorch(model_hf, inputs, generation_config)
        compile_kwargs["num_patches"] = 1

    elif model_name in ModelConfig.MOLMO_MODELS:
        processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True, padding=True)
        img = requests.get(img_url, stream=True)
        image = Image.open(BytesIO(img.content)).convert("RGB")
        image = image.resize((536, 354))
        inputs = processor.process(images=[image], text=query)
        inputs = {k: v.unsqueeze(0) for k, v in inputs.items()}
        generation_config = GenerationConfig(max_new_tokens=max_gen_len, stop_strings="<|endoftext|>")
        api_runner = ApiRunnerMolmo(
            batch_size,
            processor,
            config,
            image,
            query,
            prompt_len,
            ctx_len,
            max_gen_len,
            (num_hidden_layers, num_hidden_layers),
        )
        pytorch_hf_tokens = api_runner.run_vlm_hf_model_on_pytorch(model_hf, inputs, generation_config)
        batch_size, prompt_len = inputs["input_ids"].shape
        inputs["attention_mask"] = torch.ones((inputs["input_ids"].shape), dtype=torch.int64)
        valid = inputs["image_input_idx"] > 0
        valid = valid.reshape(1, -1)
        inputs["valid_idx"] = torch.nonzero(valid)[:, 1].unsqueeze(0)
        inputs["pixel_values"] = inputs.pop("images")
        compile_kwargs["img_size"] = img_size

    else:
        processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True, padding=True)
        image = Image.open(requests.get(img_url, stream=True).raw)
        if model_name == "mistralai/Mistral-Small-3.1-24B-Instruct-2503":
            image = image.resize((1540, 1540))
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
        api_runner = ApiRunnerVlm(
            batch_size,
            processor,
            config,
            image,
            conversation,
            prompt,
            prompt_len,
            ctx_len,
            max_gen_len,
            num_hidden_layers,
        )
        inputs = processor(images=image, text=prompt, return_tensors="pt")
        if "pixel_values" in inputs:
            inputs["pixel_values"] = inputs["pixel_values"].to(qeff_model.model.config.torch_dtype)
        pytorch_hf_tokens = api_runner.run_vlm_hf_model_on_pytorch(model_hf, inputs)
        inputs = processor(images=image, text=prompt, return_tensors="pt")
        if hasattr(qeff_model.model.config, "model_type") and qeff_model.model.config.model_type in [
            "qwen2_5_vl",
            "qwen3_vl",
            "qwen3_vl_moe",
            "qwen3_5",
            "qwen3_5_moe",
        ]:
            inputs = qeff_model.model.prepare_inputs_for_generation(
                inputs=inputs, prefill_seq_len=prompt_len, batch_size=batch_size
            )
        if "pixel_values" in inputs:
            inputs["pixel_values"] = inputs["pixel_values"].to(qeff_model.model.config.torch_dtype)
        compile_kwargs["img_size"] = img_size

    # pytorch_kv_tokens = api_runner.run_vlm_kv_model_on_pytorch(qeff_model.model)
    # assert (pytorch_kv_tokens == pytorch_hf_tokens).all(), (
    #     "Tokens don't match for pytorch HF output and pytorch KV output"
    # )

    _ = qeff_model.export()
    # ort_tokens = api_runner.run_vlm_kv_model_on_ort(onnx_model_path)
    # assert (pytorch_hf_tokens == ort_tokens).all(), "Tokens don't match for pytorch HF output and ORT output"

    qeff_model.compile(**compile_kwargs)

    if export_compile_only:
        return

    streamer = TextStreamer(processor.tokenizer)
    print("QPC Outputs (QAIC):")
    exec_info = qeff_model.generate(inputs=inputs, generation_len=max_gen_len, streamer=streamer)
    print(exec_info)
    cloud_ai_100_tokens = exec_info.generated_ids[:, :-1]
    from tests.transformers.models.causal_lm_models.check_causal_models import _tokens_match_or_first_token

    _tokens_match_or_first_token(
        pytorch_hf_tokens,
        cloud_ai_100_tokens,
        label="Tokens don't match for pytorch HF output and QPC output",
        vlm=True,
    )

    if compare_results is False:
        return

    dump_and_compare_results(
        model_name=model_name,
        compile_params=compile_kwargs,
        json_file_path="image_text_to_text_model_results.json",
        cloud_ai_100_tokens=cloud_ai_100_tokens.tolist(),
        pytorch_hf_tokens=pytorch_hf_tokens.tolist(),
        pytorch_kv_tokens=pytorch_kv_tokens.tolist() if pytorch_kv_tokens is not None else None,
        ort_tokens=ort_tokens.cpu().tolist() if ort_tokens is not None else None,
        exec_info=exec_info,
    )


def check_image_text_to_text_pytorch_vs_kv_vs_ort_vs_ai100_CB(
    model_name: str,
    num_hidden_layers: int = -1,
    kv_offload: bool = False,
    num_devices: int = 1,
    enable_qnn: Optional[bool] = False,
    qnn_config: Optional[str] = None,
    config: Optional[AutoConfig] = None,
    export_compile_only: Optional[bool] = False,
):

    prompt_len = 128
    ctx_len = 4096
    img_size = 1540
    img_url = "https://picsum.photos/id/237/536/354"
    batch_size = 1
    max_gen_len = 20
    image_urls = ["https://picsum.photos/id/237/536/354", "https://picsum.photos/id/238/536/354"]
    queries = ["Can you describe the image in detail?", "What are the objects in the image?"]

    n_layer = num_hidden_layers
    batch_size = 1
    full_batch_size = 2

    if config is None:
        config = AutoConfig.from_pretrained(
            model_name, trust_remote_code=True, padding=model_name not in ModelConfig.MOLMO_MODELS
        )
        config = set_num_layers_vlm(config, n_layer=n_layer)
        if hasattr(config, "model_type") and config.model_type in ["gemma3"]:
            config.text_config._sliding_window_pattern = 2
            config.text_config.layer_types = ["sliding_attention", "full_attention"]
        if hasattr(config, "model_type") and config.model_type in [
            "qwen3_vl",
            "qwen3_vl_moe",
        ]:
            config.vision_config.depth = 9
            config.text_config.num_hidden_layers = 1
            config.vision_config.deepstack_visual_indexes = [8]
        if model_name in ModelConfig.INTERNVL_MODELS or model_name in ModelConfig.MOLMO_MODELS:
            config._attn_implementation = "eager"
            model_hf = load_vlm_model(config)
            qeff_model = QEFFAutoModelForCausalLM.from_pretrained(
                model_name,
                kv_offload=kv_offload,
                config=config,
                continuous_batching=True,
            )
        else:
            model_hf = load_vlm_model(config)
            qeff_model = QEFFAutoModelForImageTextToText.from_pretrained(
                model_name,
                kv_offload=kv_offload,
                config=config,
                continuous_batching=True,
            )
    else:
        model_hf = load_vlm_model_from_config(config)
        qeff_model = QEFFAutoModelForImageTextToText(
            copy.deepcopy(model_hf),
            kv_offload=kv_offload,
            config=model_hf.config,
            continuous_batching=True,
        )

    compile_kwargs = {
        "num_cores": 16,
        "num_devices": num_devices,
        "prefill_seq_len": prompt_len,
        "ctx_len": ctx_len,
        "batch_size": batch_size,
        "full_batch_size": full_batch_size,
        "mxfp6_matmul": False,
    }

    images = []
    generation_config = None
    if model_name in ModelConfig.INTERNVL_MODELS:
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True, use_fast=False)
        processor = InternProcessor(model_hf, tokenizer)
        image_height = 448
        image_width = 448
        for img_url in image_urls:
            img = requests.get(img_url, stream=True)
            image = Image.open(BytesIO(img.content)).convert("RGB")
            image = image.resize((image_height, image_width))
            images.append(image)
        generation_config = dict(max_new_tokens=max_gen_len, do_sample=False)
        generation_config["eos_token_id"] = tokenizer.convert_tokens_to_ids("<|im_end|>\n".strip())
        api_runner = ApiRunnerInternVL(
            batch_size,
            processor,
            config,
            images[0],
            queries[0],
            prompt_len,
            ctx_len,
            max_gen_len,
            n_layer,
        )
        # For same prompt
        image_list = [images[0]] * full_batch_size
        prompt_list = [queries[0]] * full_batch_size
        pytorch_hf_tokens = api_runner.run_vlm_hf_model_on_pytorch_CB(model_hf, image_list, prompt_list)
        compile_kwargs["num_patches"] = 1
    elif model_name in ModelConfig.MOLMO_MODELS:
        processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True, padding=True)
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        image_height = 536
        image_width = 354
        for img_url in image_urls:
            img = requests.get(img_url, stream=True)
            image = Image.open(BytesIO(img.content)).convert("RGB")
            image = image.resize((image_height, image_width))
            images.append(image)
        api_runner = ApiRunnerMolmo(
            batch_size,
            processor,
            config,
            images[0],
            queries[0],
            prompt_len,
            ctx_len,
            max_gen_len,
            n_layer,
        )
        generation_config = GenerationConfig(max_new_tokens=max_gen_len, stop_strings="<|endoftext|>")
        image_list = [images[0]] * full_batch_size
        prompt_list = [queries[0]] * full_batch_size
        pytorch_hf_tokens = api_runner.run_vlm_hf_model_on_pytorch_CB(
            model_hf, image_list, prompt_list, generation_config
        )
        compile_kwargs["img_size"] = img_size
    else:
        processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True, padding=True)
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        image_height = None
        image_width = None
        for img_url in image_urls:
            image = Image.open(requests.get(img_url, stream=True).raw)
            if model_name == "mistralai/Mistral-Small-3.1-24B-Instruct-2503":
                image_height = 1540
                image_width = 1540
                image = image.resize((image_height, image_width))
            images.append(image)

        conversation = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": queries[0]},
                    {"type": "image"},
                ],
            },
        ]
        prompt = processor.apply_chat_template(conversation, add_generation_prompt=True)
        api_runner = ApiRunnerVlm(
            batch_size,
            processor,
            config,
            images[0],
            conversation,
            prompt,
            prompt_len,
            ctx_len,
            max_gen_len,
            n_layer,
        )
        image_list = [images[0]] * full_batch_size
        prompt_list = [queries[0]] * full_batch_size
        pytorch_hf_tokens = api_runner.run_vlm_hf_model_on_pytorch_CB(model_hf, image_list, prompt_list)
        compile_kwargs["img_size"] = img_size

    qeff_model.export()
    qeff_model.compile(**compile_kwargs)

    if export_compile_only:
        return

    print("QPC Outputs (QAIC):")
    exec_info = qeff_model.generate(
        tokenizer=tokenizer,
        processor=processor,
        images=[image_urls[0]] * full_batch_size,
        prompts=prompt_list,
        generation_len=max_gen_len,
        image_height=image_height,
        image_width=image_width,
    )
    qpc_tokens = exec_info.generated_ids[:, :max_gen_len]
    print("QPC Outputs (QAIC) for Continuous Batching with same prompt:")
    print(exec_info.generated_texts)
    from tests.transformers.models.causal_lm_models.check_causal_models import _tokens_match_or_first_token

    for i in range(full_batch_size):
        _tokens_match_or_first_token(
            pytorch_hf_tokens[i],
            qpc_tokens[i],
            label=f"Tokens don't match for prompt {i} between HF and QPC output for same prompts",
            vlm=True,
        )
    if model_name in ModelConfig.MOLMO_MODELS:
        pytorch_hf_tokens = api_runner.run_vlm_hf_model_on_pytorch_CB(
            model_hf, images, queries, generation_config=generation_config
        )
    else:
        pytorch_hf_tokens = api_runner.run_vlm_hf_model_on_pytorch_CB(model_hf, images, queries)

    print("QPC Outputs (QAIC):")
    exec_info = qeff_model.generate(
        tokenizer=tokenizer,
        processor=processor,
        images=image_urls,
        prompts=queries,
        generation_len=max_gen_len,
        image_height=image_height,
        image_width=image_width,
    )
    qpc_tokens = exec_info.generated_ids[:, :max_gen_len]
    print("QPC Outputs (QAIC) for Continuous Batching with different prompt:")
    print(exec_info.generated_texts)
    for i in range(full_batch_size):
        _tokens_match_or_first_token(
            pytorch_hf_tokens[i],
            qpc_tokens[i],
            label=f"Tokens don't match for prompt {i} between HF and QPC output for different prompts",
            vlm=True,
        )

# -----------------------------------------------------------------------------
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
#
# ----------------------------------------------------------------------------

import copy
import inspect
import json
import os
import sys
from io import BytesIO
from typing import List, Optional

import pytest
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
from transformers.dynamic_module_utils import get_class_from_dynamic_module
from transformers.utils import import_utils as hf_import_utils

from QEfficient import QEFFAutoModelForCausalLM, QEFFAutoModelForImageTextToText
from QEfficient.utils._utils import create_json
from QEfficient.utils.constants import QnnConstants
from QEfficient.utils.run_utils import ApiRunnerInternVL, ApiRunnerMolmo, ApiRunnerVlm
from QEfficient.utils.test_utils import (
    InternProcessor,
    ModelConfig,
    load_vlm_model,
    load_vlm_model_from_config,
    set_num_layers_vlm,
)

from ..check_model_results import dump_and_compare_results

CONFIG_PATH = os.path.join(os.path.dirname(__file__), "../../../configs/image_text_model_configs.json")
with open(CONFIG_PATH, "r") as f:
    config_data = json.load(f)
    multimodal_models = config_data["image_text_models"]
test_mm_models = [model_config["model_name"] for model_config in multimodal_models]
model_config_dict = {model["model_name"]: model for model in multimodal_models}

NEW_GENERATION_TOKENS = 10
KIMI_K25_MODEL_NAME = "moonshotai/Kimi-K2.5"
KIMI_K25_QAIC_CONFIG = {"mla_absorption": {"cache_compressed": True, "absorption": False, "online": False}}


def _is_kimi_k25(model_name: str) -> bool:
    return model_name == KIMI_K25_MODEL_NAME


def _ensure_torch_fx_import_compatibility():
    if hasattr(hf_import_utils, "is_torch_fx_available"):
        return

    def _is_torch_fx_available() -> bool:
        if not hf_import_utils.is_torch_available():
            return False
        try:
            import torch.fx  # noqa: F401

            return True
        except Exception:
            return False

    hf_import_utils.is_torch_fx_available = _is_torch_fx_available


def _patch_kimi_k25_tie_weights_compat(kimi_cls):
    tie_signature = inspect.signature(kimi_cls.tie_weights)
    if tuple(tie_signature.parameters) != ("self",):
        return

    def _tie_weights_compat(self, missing_keys=None, recompute_mapping=True):
        lm_tie_weights = getattr(self.language_model, "tie_weights")
        try:
            return lm_tie_weights(missing_keys=missing_keys, recompute_mapping=recompute_mapping)
        except TypeError:
            return lm_tie_weights()

    kimi_cls.tie_weights = _tie_weights_compat


def _patch_kimi_k25_deepseek_init_weights_compat(kimi_cls):
    module_prefix, _ = kimi_cls.__module__.rsplit(".", maxsplit=1)
    deepseek_module = sys.modules.get(f"{module_prefix}.modeling_deepseek")
    if deepseek_module is None or not hasattr(deepseek_module, "DeepseekV3PreTrainedModel"):
        return

    deepseek_cls = deepseek_module.DeepseekV3PreTrainedModel
    if getattr(deepseek_cls, "_qeff_test_init_weights_patched", False):
        return

    def _init_weights_compat(self, module):
        std = self.config.initializer_range
        if isinstance(module, torch.nn.Linear):
            if hasattr(module, "weight") and module.weight is not None:
                module.weight.data.normal_(mean=0.0, std=std)
            if hasattr(module, "bias") and module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, torch.nn.Embedding):
            if hasattr(module, "weight") and module.weight is not None:
                module.weight.data.normal_(mean=0.0, std=std)
                if module.padding_idx is not None:
                    module.weight.data[module.padding_idx].zero_()

    deepseek_cls._init_weights = _init_weights_compat
    deepseek_cls._qeff_test_init_weights_patched = True


def _patch_kimi_k25_remote_code_compat(config):
    _ensure_torch_fx_import_compatibility()
    kimi_cls = get_class_from_dynamic_module("modeling_kimi_k25.KimiK25ForConditionalGeneration", config._name_or_path)
    _patch_kimi_k25_tie_weights_compat(kimi_cls)
    _patch_kimi_k25_deepseek_init_weights_compat(kimi_cls)


def _get_kimi_k25_test_config(model_name: str):
    config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
    config._attn_implementation = "eager"
    config.torch_dtype = torch.float32
    config.dtype = torch.float32
    additional_params = model_config_dict[model_name]["additional_params"]

    for attr, value in additional_params["text_config"].items():
        setattr(config.text_config, attr, value)
    config.text_config._attn_implementation = "eager"
    config.text_config.torch_dtype = torch.float32
    config.text_config.dtype = torch.float32

    for attr, value in additional_params["vision_config"].items():
        setattr(config.vision_config, attr, value)
    config.vision_config._attn_implementation = "eager"
    config.vision_config.torch_dtype = torch.float32
    config.vision_config.dtype = torch.float32

    _patch_kimi_k25_remote_code_compat(config)
    return config


def _get_kimi_k25_num_image_tokens(config, grid_thws):
    merge_height, merge_width = config.vision_config.merge_kernel_size
    return int(grid_thws[0, 1].item() // merge_height) * int(grid_thws[0, 2].item() // merge_width)


@torch.no_grad()
def _run_kimi_k25_hf_model_on_pytorch(model, processor, inputs, max_gen_len):
    generated_ids = inputs["input_ids"]
    attention_mask = inputs["attention_mask"]
    pixel_values = inputs["pixel_values"]
    grid_thws = inputs["grid_thws"]
    new_tokens = []

    eos_token_id = getattr(model.config, "eos_token_id", None)
    if eos_token_id is None and hasattr(model.config, "text_config"):
        eos_token_id = getattr(model.config.text_config, "eos_token_id", None)

    for _ in range(max_gen_len):
        outputs = model(
            input_ids=generated_ids,
            attention_mask=attention_mask,
            pixel_values=pixel_values,
            grid_thws=grid_thws,
            use_cache=False,
            return_dict=True,
        )
        logits = outputs[0] if isinstance(outputs, tuple) else outputs.logits
        next_token = logits[:, -1, :].argmax(dim=-1, keepdim=True)
        new_tokens.append(next_token)

        generated_ids = torch.cat([generated_ids, next_token], dim=1)
        attention_mask = torch.cat(
            [
                attention_mask,
                torch.ones((attention_mask.shape[0], 1), dtype=attention_mask.dtype, device=attention_mask.device),
            ],
            dim=1,
        )

        if eos_token_id is not None and torch.all(next_token == eos_token_id):
            break

    output_tokens = torch.cat(new_tokens, dim=1).squeeze(0)
    py_output = processor.tokenizer.decode(output_tokens.tolist()).strip()
    print("Original HF Model Outputs (Torch CPU):")
    print("Completion:", repr(py_output))
    return output_tokens


def check_image_text_to_text_pytorch_vs_kv_vs_ort_vs_ai100(
    model_name: str,
    manual_cleanup: callable,
    num_hidden_layers: Optional[int] = -1,
    kv_offload: Optional[bool] = False,
    num_devices: Optional[int] = 1,
    enable_qnn: Optional[bool] = False,
    qnn_config: Optional[str] = None,
    config: Optional[AutoConfig] = None,
    torch_dtype: Optional[torch.dtype] = torch.float32,
    compare_results: Optional[bool] = False,
):
    prompt_len = model_config_dict[model_name]["prompt_len"]
    ctx_len = model_config_dict[model_name]["ctx_len"]
    img_size = model_config_dict[model_name].get("img_size")
    img_url = model_config_dict[model_name]["img_url"]
    query = model_config_dict[model_name]["text_prompt"]
    batch_size = model_config_dict[model_name]["batch_size"]

    max_gen_len = NEW_GENERATION_TOKENS
    pytorch_kv_tokens = None
    ort_tokens = None
    n_layer = num_hidden_layers
    if _is_kimi_k25(model_name) and config is None:
        config = _get_kimi_k25_test_config(model_name)
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
        qaic_config = KIMI_K25_QAIC_CONFIG if _is_kimi_k25(model_name) else None
        qeff_model = QEFFAutoModelForImageTextToText(
            copy.deepcopy(model_hf),
            kv_offload=kv_offload,
            qaic_config=qaic_config,
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
        generation_config = GenerationConfig(max_new_tokens=NEW_GENERATION_TOKENS, stop_strings="<|endoftext|>")
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

    elif _is_kimi_k25(model_name):
        processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True, padding=True)
        image = Image.open(requests.get(img_url, stream=True).raw).convert("RGB")
        conversation = [
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": image},
                    {"type": "text", "text": query},
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
        inputs = processor(
            messages=conversation,
            add_generation_prompt=True,
            tokenize=False,
            return_tensors="pt",
        )
        pytorch_hf_tokens = _run_kimi_k25_hf_model_on_pytorch(model_hf, processor, inputs, max_gen_len)
        inputs = processor(
            messages=conversation,
            add_generation_prompt=True,
            tokenize=False,
            return_tensors="pt",
        )
        compile_kwargs.update(
            {
                "num_patches": int(inputs["pixel_values"].shape[0]),
                "h": int(inputs["grid_thws"][0, 1].item()),
                "w": int(inputs["grid_thws"][0, 2].item()),
                "num_image_tokens": _get_kimi_k25_num_image_tokens(config, inputs["grid_thws"]),
            }
        )

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
    streamer = TextStreamer(processor.tokenizer)
    print("QPC Outputs (QAIC):")
    exec_info = qeff_model.generate(inputs=inputs, generation_len=NEW_GENERATION_TOKENS, streamer=streamer)
    print(exec_info)
    cloud_ai_100_tokens = exec_info.generated_ids[:, :-1]
    assert (pytorch_hf_tokens == cloud_ai_100_tokens).all(), "Tokens don't match for pytorch HF output and QPC output"
    manual_cleanup(qeff_model.onnx_path)  # Clean up the model files after the tests are done.
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


@pytest.mark.full_layers
@pytest.mark.on_qaic
@pytest.mark.multimodal
@pytest.mark.parametrize("model_name", test_mm_models)
@pytest.mark.parametrize("kv_offload", [True, False])
def test_full_image_text_to_text_pytorch_vs_kv_vs_ort_vs_ai100(model_name, kv_offload, manual_cleanup):
    if model_name in ModelConfig.SKIPPED_MODELS:
        pytest.skip("Test skipped for this model due to some issues.")
    if model_name in ModelConfig.DUAL_QPC_MODELS and not kv_offload:
        pytest.skip("These models require kv_offload=True for testing.")

    torch.manual_seed(42)
    check_image_text_to_text_pytorch_vs_kv_vs_ort_vs_ai100(
        model_name,
        kv_offload=kv_offload,
        compare_results=True,
        manual_cleanup=manual_cleanup,
        num_devices=4,
    )


@pytest.mark.few_layers
@pytest.mark.on_qaic
@pytest.mark.multimodal
@pytest.mark.parametrize("model_name", test_mm_models)
@pytest.mark.parametrize("kv_offload", [True, False])
def test_few_image_text_to_text_pytorch_vs_kv_vs_ort_vs_ai100(model_name, kv_offload, manual_cleanup):
    if model_name in ModelConfig.SKIPPED_MODELS:
        pytest.skip("Test skipped for this model due to some issues.")
    if model_name in ModelConfig.DUAL_QPC_MODELS and not kv_offload:
        pytest.skip("These models require kv_offload=True for testing.")

    torch.manual_seed(42)
    check_image_text_to_text_pytorch_vs_kv_vs_ort_vs_ai100(
        model_name,
        num_hidden_layers=model_config_dict[model_name]["num_layers"],
        kv_offload=kv_offload,
        manual_cleanup=manual_cleanup,
    )


@pytest.mark.dummy_layers
@pytest.mark.on_qaic
@pytest.mark.multimodal
@pytest.mark.parametrize("model_name", test_mm_models)
@pytest.mark.parametrize("kv_offload", [True, False])
def test_dummy_image_text_to_text_pytorch_vs_kv_vs_ort_vs_ai100(model_name, kv_offload, manual_cleanup):
    if model_name in ModelConfig.SKIPPED_MODELS:
        pytest.skip("Test skipped for this model due to some issues.")
    if model_name in ModelConfig.DUAL_QPC_MODELS and not kv_offload:
        pytest.skip("These models require kv_offload=True for testing.")

    torch.manual_seed(42)
    hf_config = None
    if model_name in ModelConfig.STANDARD_VLM_MODELS:
        model_type = model_config_dict[model_name].get("model_type", None)
        custom_config = model_config_dict[model_name].get("additional_params", {})
        hf_config = AutoConfig.for_model(model_type, trust_remote_code=True, **custom_config)
        hf_config.name_or_path = model_name
        check_image_text_to_text_pytorch_vs_kv_vs_ort_vs_ai100(
            model_name, kv_offload=kv_offload, config=hf_config, manual_cleanup=manual_cleanup
        )
    else:
        check_image_text_to_text_pytorch_vs_kv_vs_ort_vs_ai100(
            model_name,
            num_hidden_layers=model_config_dict[model_name]["num_layers"],
            kv_offload=kv_offload,
            manual_cleanup=manual_cleanup,
        )


################################ QNN Tests ################################


@pytest.mark.on_qaic
@pytest.mark.qnn
@pytest.mark.multimodal
@pytest.mark.parametrize("model_name", test_mm_models)
@pytest.mark.parametrize("kv_offload", [True, False])
def test_image_text_to_text_pytorch_vs_kv_vs_ort_vs_ai100_qnn(model_name, kv_offload, manual_cleanup):
    """
    Test function to validate the PyTorch model, the PyTorch model after KV changes, the ONNX model, and the Cloud AI 100 model,  without continuous batching.
    ``Mandatory`` Args:
        :model_name (str): Hugging Face Model Card name, Example: ``gpt2``
    """
    if model_name in {
        "meta-llama/Llama-4-Scout-17B-16E-Instruct",
        "google/gemma-3-4b-it",
        KIMI_K25_MODEL_NAME,
    }:
        pytest.skip("QNN is not supported for these models yet.")

    qnn_config_json_path = os.path.join(os.getcwd(), "qnn_config.json")
    create_json(qnn_config_json_path, QnnConstants.QNN_SAMPLE_CONFIG)

    check_image_text_to_text_pytorch_vs_kv_vs_ort_vs_ai100(
        model_name=model_name,
        kv_offload=kv_offload,
        enable_qnn=True,
        qnn_config=qnn_config_json_path,
        manual_cleanup=manual_cleanup,
    )

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
from typing import Optional

import pytest
import requests
import torch
from PIL import Image
from transformers import (
    AutoConfig,
    AutoProcessor,
    AutoTokenizer,
    GenerationConfig,
)
from transformers.dynamic_module_utils import get_class_from_dynamic_module
from transformers.utils import import_utils as hf_import_utils

from QEfficient import QEFFAutoModelForCausalLM, QEFFAutoModelForImageTextToText
from QEfficient.utils.run_utils import ApiRunnerInternVL, ApiRunnerMolmo, ApiRunnerVlm
from QEfficient.utils.test_utils import (
    InternProcessor,
    ModelConfig,
    load_vlm_model,
    load_vlm_model_from_config,
    set_num_layers_vlm,
)

CONFIG_PATH = os.path.join(os.path.dirname(__file__), "../../../configs/image_text_model_configs.json")
with open(CONFIG_PATH, "r") as f:
    config_data = json.load(f)
    multimodal_models = config_data["image_text_models"]
test_mm_models = [model_config["model_name"] for model_config in multimodal_models]
model_config_dict = {model["model_name"]: model for model in multimodal_models}

NEW_GENERATION_TOKENS = 10
KIMI_K25_MODEL_NAME = "moonshotai/Kimi-K2.5"


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
    return kimi_cls


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


def _load_kimi_k25_model_from_config(config):
    kimi_cls = _patch_kimi_k25_remote_code_compat(config)
    model = kimi_cls._from_config(config)
    torch_dtype = getattr(model.config, "torch_dtype", None)
    if torch_dtype == torch.bfloat16 or torch_dtype == torch.float16:
        model = model.to(torch.float32)
    return model.eval()


def _get_kimi_k25_num_image_tokens(config, grid_thws):
    merge_height, merge_width = config.vision_config.merge_kernel_size
    return int(grid_thws[0, 1].item() // merge_height) * int(grid_thws[0, 2].item() // merge_width)


@torch.no_grad()
def _run_kimi_k25_hf_model_on_pytorch_CB(model, processor, images, queries, max_gen_len):
    generated_tokens = []

    eos_token_id = getattr(model.config, "eos_token_id", None)
    if eos_token_id is None and hasattr(model.config, "text_config"):
        eos_token_id = getattr(model.config.text_config, "eos_token_id", None)

    for idx, (image, query) in enumerate(zip(images, queries)):
        conversation = [
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": image},
                    {"type": "text", "text": query},
                ],
            },
        ]
        inputs = processor(messages=conversation, add_generation_prompt=True, tokenize=False, return_tensors="pt")
        generated_ids = inputs["input_ids"]
        attention_mask = inputs["attention_mask"]
        pixel_values = inputs["pixel_values"]
        grid_thws = inputs["grid_thws"]
        new_tokens = []

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
        print(f"Original HF Model Outputs (Torch CPU) for prompt {idx}:")
        print("Query:", repr(query))
        print("Completion:", repr(py_output))
        generated_tokens.append(output_tokens.numpy())

    return generated_tokens


def _load_kimi_k25_layer_subset_model():
    examples_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../../examples/kimi_k2"))
    if examples_dir not in sys.path:
        sys.path.insert(0, examples_dir)

    from test_kimi_k25 import (  # noqa: PLC0415
        LOADED_EXPERT_IDS,
        NUM_EXPERTS_PER_TOKEN,
        NUM_TEXT_LAYERS,
        NUM_VISION_LAYERS,
        _load_layer_subset_model,
        _patch_deepseek_init_weights_compat,
        _patch_kimi_tie_weights_compat,
        _prepare_config,
        _resolve_model_path,
        _set_deterministic,
    )

    _set_deterministic(1234)
    _ensure_torch_fx_import_compatibility()
    model_path = _resolve_model_path()
    config = _prepare_config(model_path)
    kimi_cls = get_class_from_dynamic_module("modeling_kimi_k25.KimiK25ForConditionalGeneration", str(model_path))
    _patch_kimi_tie_weights_compat(kimi_cls)
    _patch_deepseek_init_weights_compat(kimi_cls)

    model, tokenizer, processor = _load_layer_subset_model(
        model_path=model_path,
        kimi_cls=kimi_cls,
        config=config,
        num_vision_layers=min(NUM_VISION_LAYERS, 1),
        num_text_layers=min(NUM_TEXT_LAYERS, 1),
        loaded_expert_ids=LOADED_EXPERT_IDS,
        num_experts_per_tok=NUM_EXPERTS_PER_TOKEN,
        dtype=torch.float32,
    )
    model.vision_tower.patch_embed.pos_emb.interpolation_mode = "bilinear"
    model = model.eval().to("cpu")
    return model, tokenizer, processor


def check_image_text_to_text_pytorch_vs_kv_vs_ort_vs_ai100_CB(
    model_name: str,
    manual_cleanup: callable,
    num_hidden_layers: int = -1,
    kv_offload: bool = False,
    num_devices: int = 1,
    enable_qnn: Optional[bool] = False,
    qnn_config: Optional[str] = None,
    config: Optional[AutoConfig] = None,
):
    prompt_len = model_config_dict[model_name]["prompt_len"]
    ctx_len = model_config_dict[model_name]["ctx_len"]
    max_gen_len = (NEW_GENERATION_TOKENS,)
    img_size = model_config_dict[model_name].get("img_size")
    image_urls = model_config_dict[model_name]["img_url_list"]
    queries = model_config_dict[model_name]["text_prompt_list"]
    n_layer = num_hidden_layers
    batch_size = model_config_dict[model_name]["batch_size"]
    full_batch_size = model_config_dict[model_name]["full_batch_size"]
    max_gen_len = NEW_GENERATION_TOKENS
    kimi_tokenizer = None
    kimi_processor = None
    if _is_kimi_k25(model_name):
        full_batch_size = 1

    if _is_kimi_k25(model_name) and config is None:
        model_hf, kimi_tokenizer, kimi_processor = _load_kimi_k25_layer_subset_model()
        config = model_hf.config
        qeff_model = QEFFAutoModelForImageTextToText(
            copy.deepcopy(model_hf),
            kv_offload=kv_offload,
            config=model_hf.config,
            torch_dtype=torch.float32,
            continuous_batching=True,
        )
    elif _is_kimi_k25(model_name):
        if config is None:
            config = _get_kimi_k25_test_config(model_name)
        model_hf = _load_kimi_k25_model_from_config(config)
        model_hf.vision_tower.patch_embed.pos_emb.interpolation_mode = "bilinear"
        qeff_model = QEFFAutoModelForImageTextToText(
            copy.deepcopy(model_hf),
            kv_offload=kv_offload,
            config=model_hf.config,
            continuous_batching=True,
        )
    elif config is None:
        config = AutoConfig.from_pretrained(
            model_name, trust_remote_code=True, padding=model_name not in ModelConfig.MOLMO_MODELS
        )
        config = set_num_layers_vlm(config, n_layer=n_layer)
        if hasattr(config, "model_type") and config.model_type in ["gemma3"]:
            config.text_config._sliding_window_pattern = 2
            config.text_config.layer_types = ["sliding_attention", "full_attention"]
        if hasattr(config, "model_type") and config.model_type in ["gemma4"]:
            config.text_config.num_kv_shared_layers = 0
            config.text_config.layer_types = ["sliding_attention"]
        if hasattr(config, "model_type") and config.model_type in ["qwen3_5"]:
            config.text_config.layer_types = [
                "linear_attention",
                "linear_attention",
                "linear_attention",
                "full_attention",
            ]
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
        "split-model-io": True,
    }
    if model_name in ["qwen2_5_vl", "qwen3_vl", "qwen3_vl_moe", "qwen3_5", "qwen3_5_moe", "gemma4"]:
        compile_kwargs["use_onnx_subfunctions"] = True

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
        generation_config = GenerationConfig(max_new_tokens=NEW_GENERATION_TOKENS, stop_strings="<|endoftext|>")
        image_list = [images[0]] * full_batch_size
        prompt_list = [queries[0]] * full_batch_size
        pytorch_hf_tokens = api_runner.run_vlm_hf_model_on_pytorch_CB(
            model_hf, image_list, prompt_list, generation_config
        )
        compile_kwargs["img_size"] = img_size
    elif _is_kimi_k25(model_name):
        processor = kimi_processor or AutoProcessor.from_pretrained(model_name, trust_remote_code=True)
        tokenizer = kimi_tokenizer or AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        image_height = None
        image_width = None
        image_urls = [image_urls[0]] * len(queries)
        num_patches = []
        image_heights = []
        image_widths = []
        num_image_tokens = []
        for img_url in image_urls:
            image = Image.open(requests.get(img_url, stream=True).raw).convert("RGB")
            images.append(image)

        for image, query in zip(images, queries):
            conversation = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image_url", "image_url": image},
                        {"type": "text", "text": query},
                    ],
                },
            ]
            inputs = processor(messages=conversation, add_generation_prompt=True, tokenize=False, return_tensors="pt")
            num_patches.append(int(inputs["pixel_values"].shape[0]))
            image_heights.append(int(inputs["grid_thws"][0, 1].item()))
            image_widths.append(int(inputs["grid_thws"][0, 2].item()))
            num_image_tokens.append(_get_kimi_k25_num_image_tokens(config, inputs["grid_thws"]))

        image_list = [images[0]] * full_batch_size
        prompt_list = [queries[0]] * full_batch_size
        pytorch_hf_tokens = _run_kimi_k25_hf_model_on_pytorch_CB(
            copy.deepcopy(model_hf), processor, image_list, prompt_list, max_gen_len
        )
        compile_kwargs.update(
            {
                "prefill_seq_len": 1,
                "num_patches": num_patches[0],
                "h": image_heights[0],
                "w": image_widths[0],
                "num_image_tokens": num_image_tokens[0],
            }
        )
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
    # if _is_kimi_k25(model_name):
    #    manual_cleanup(qeff_model.onnx_path)
    #    return
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
    for i in range(full_batch_size):
        assert (pytorch_hf_tokens[i] == qpc_tokens[i]).all(), (
            f"Tokens don't match for prompt {i} between HF and QPC output for same prompts"
        )
    if model_name in ModelConfig.MOLMO_MODELS:
        pytorch_hf_tokens = api_runner.run_vlm_hf_model_on_pytorch_CB(
            model_hf, images, queries, generation_config=generation_config
        )
    else:
        if _is_kimi_k25(model_name):
            pytorch_hf_tokens = _run_kimi_k25_hf_model_on_pytorch_CB(
                copy.deepcopy(model_hf), processor, images, queries, max_gen_len
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
        assert (pytorch_hf_tokens[i] == qpc_tokens[i]).all(), (
            f"Tokens don't match for prompt {i} between HF and QPC output for different prompts"
        )
    manual_cleanup(qeff_model.onnx_path)  # Clean up the model files after the tests are done.


@pytest.mark.skip("Token Mismatch for full models")
@pytest.mark.full_layers
@pytest.mark.on_qaic
@pytest.mark.multimodal
@pytest.mark.parametrize("model_name", test_mm_models)
@pytest.mark.parametrize("kv_offload", [True])  # TODO: Add support for kv_offload=False
def test_full_image_text_to_text_pytorch_vs_ai100_continuous_batching(model_name, kv_offload, manual_cleanup):
    if model_name in ModelConfig.SKIPPED_MODELS:
        pytest.skip("Test skipped for this model due to some issues.")
    if model_name in ModelConfig.DUAL_QPC_MODELS and not kv_offload:
        pytest.skip("These models require kv_offload=True for testing.")

    torch.manual_seed(42)
    check_image_text_to_text_pytorch_vs_kv_vs_ort_vs_ai100_CB(
        model_name=model_name,
        kv_offload=kv_offload,
        manual_cleanup=manual_cleanup,
        num_devices=4,
    )


@pytest.mark.few_layers
@pytest.mark.on_qaic
@pytest.mark.multimodal
@pytest.mark.parametrize("model_name", test_mm_models)
@pytest.mark.parametrize("kv_offload", [True])  # TODO: Add support for kv_offload=False
def test_few_image_text_to_text_pytorch_vs_ai100_continuous_batching(model_name, kv_offload, manual_cleanup):
    if model_name in ModelConfig.SKIPPED_MODELS:
        pytest.skip("Test skipped for this model due to some issues.")
    if model_name in ModelConfig.DUAL_QPC_MODELS and not kv_offload:
        pytest.skip("These models require kv_offload=True for testing.")

    torch.manual_seed(42)
    check_image_text_to_text_pytorch_vs_kv_vs_ort_vs_ai100_CB(
        model_name=model_name,
        num_hidden_layers=model_config_dict[model_name]["num_layers"],
        kv_offload=kv_offload,
        manual_cleanup=manual_cleanup,
    )


@pytest.mark.dummy_layers
@pytest.mark.on_qaic
@pytest.mark.multimodal
@pytest.mark.parametrize("model_name", test_mm_models)
@pytest.mark.parametrize("kv_offload", [True])  # TODO: Add support for kv_offload=False
def test_dummy_image_text_to_text_pytorch_vs_ai100_continuous_batching(model_name, kv_offload, manual_cleanup):
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
        check_image_text_to_text_pytorch_vs_kv_vs_ort_vs_ai100_CB(
            model_name, kv_offload=kv_offload, config=hf_config, manual_cleanup=manual_cleanup
        )
    else:
        check_image_text_to_text_pytorch_vs_kv_vs_ort_vs_ai100_CB(
            model_name,
            num_hidden_layers=model_config_dict[model_name]["num_layers"],
            kv_offload=kv_offload,
            manual_cleanup=manual_cleanup,
        )

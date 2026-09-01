# -----------------------------------------------------------------------------
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
#
# ----------------------------------------------------------------------------

import copy
import json
import os
from typing import Optional

import onnx
import pytest
import torch
from transformers import (
    AutoConfig,
    AutoProcessor,
    TextStreamer,
)

from QEfficient import QEFFAutoModelForImageTextToText
from QEfficient.utils.run_utils import ApiRunnerVlm
from QEfficient.utils.test_utils import (
    ModelConfig,
    load_vlm_model,
    load_vlm_model_from_config,
    set_num_layers_vlm,
)
from tests.transformers.models.image_text_to_text._image_utils import load_test_image
from tests.two_phase import model_export_compile_lock, resolve_two_phase_cleanup

NEW_GENERATION_TOKENS = 10


CONFIG_PATH = os.path.join(os.path.dirname(__file__), "../../configs/image_text_model_configs.json")

with open(CONFIG_PATH, "r") as f:
    config_data = json.load(f)
    multimodal_models = config_data["image_text_subfunction_models"]

test_mm_models = [model_config["model_name"] for model_config in multimodal_models]
model_config_dict = {model["model_name"]: model for model in multimodal_models}

# The tiny-random qwen3.5 hybrids ship a degenerate ``hidden_size=8`` checkpoint that stores
# attention weights at only a couple of layer indices. Loading it via ``from_pretrained`` is
# fp16-marginal at that width and, worse, random-initializes the checkpoint-missing weights
# independently for the HF reference and the QEff model under test, so the two become different
# networks and token parity is meaningless. Build these from their authored config (random init
# at the intended sizes) with weights shared via ``deepcopy`` instead -- the same synthesized-config
# path the main VLM test uses for ``ModelConfig.STANDARD_VLM_MODELS``.
SYNTHESIZED_CONFIG_MODELS = frozenset({"tiny-random/qwen3.5", "tiny-random/qwen3.5-moe"})


def has_decoder_layer_function(onnx_path, expected_function_tokens):
    """Check if ONNX model contains expected decoder-layer function definition."""
    model = onnx.load(onnx_path, load_external_data=False)
    function_names = [f.name for f in model.functions]
    decoder_layer_functions = [
        name for name in function_names if any(token in name for token in expected_function_tokens)
    ]
    return len(decoder_layer_functions) > 0, decoder_layer_functions


def check_image_text_to_text_subfunction_core(
    model_name: str,
    manual_cleanup: callable,
    kv_offload: bool = False,
    num_hidden_layers: int = -1,
    num_devices: int = 1,
    config: Optional[AutoConfig] = None,
    torch_dtype: Optional[torch.dtype] = torch.float32,
    compile_only: bool = False,
):
    # Two-phase compile/execute split: suppress per-test cleanup in both phases (model variants
    # share a content-addressed export dir, so one variant's rmtree would destroy its siblings'
    # warm QPCs) and force compile-only in the warm phase. A no-op in normal runs.
    manual_cleanup, compile_only = resolve_two_phase_cleanup(manual_cleanup, compile_only)
    img_size = model_config_dict[model_name]["img_size"]
    img_url = model_config_dict[model_name]["img_url"]
    query = model_config_dict[model_name]["text_prompt"]
    prompt_len = model_config_dict[model_name]["prompt_len"]
    ctx_len = model_config_dict[model_name]["ctx_len"]
    batch_size = model_config_dict[model_name]["batch_size"]
    enable_qnn = False
    qnn_config = None
    max_gen_len = NEW_GENERATION_TOKENS

    if config is None and model_name in SYNTHESIZED_CONFIG_MODELS:
        # Synthesize the config at the authored sizes so both models are built from it (random
        # init) with byte-identical weights; see ``SYNTHESIZED_CONFIG_MODELS`` for the rationale.
        model_type = model_config_dict[model_name].get("model_type", None)
        custom_config = model_config_dict[model_name].get("additional_params", {})
        config = AutoConfig.for_model(model_type, trust_remote_code=True, **custom_config)
        config.name_or_path = model_name

    if config is None:
        config = AutoConfig.from_pretrained(
            model_name, trust_remote_code=True, padding=model_name not in ModelConfig.MOLMO_MODELS
        )
        config = set_num_layers_vlm(config, n_layer=num_hidden_layers)
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

        # A truncated hybrid stack must keep at least one full_attention layer: the hybrid
        # cache indexes its linear-layer state off the attention layers, so an all-linear
        # pattern makes ``get_seq_length`` raise. ``set_num_layers_vlm`` only rewrites the
        # layer count, and a checkpoint's own pattern can be all-linear at this depth, so
        # pin the pattern authored alongside the layer count in the model-config JSON.
        authored_layer_types = (
            model_config_dict[model_name].get("additional_params", {}).get("text_config", {}).get("layer_types")
        )
        if authored_layer_types is not None and getattr(config.text_config, "layer_types", None) is not None:
            config.text_config.layer_types = authored_layer_types
            config.text_config.num_hidden_layers = len(authored_layer_types)

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
        "img_size": img_size,
        "num_devices": num_devices,
        "prefill_seq_len": prompt_len,
        "ctx_len": ctx_len,
        "mxfp6": False,
        "enable_qnn": enable_qnn,
        "qnn_config": qnn_config,
        "use_onnx_subfunctions": True,
    }

    processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True, padding=True)
    image = load_test_image(img_url)
    if model_name == "tiny-random/mistral-3":
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
    if not compile_only:
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

    # One lock spans export and compile: both write into the same content-addressed dir, so a
    # concurrent worker must not slip in between them.
    with model_export_compile_lock(model_name):
        with_sub_func_onnx = qeff_model.export(use_onnx_subfunctions=True, offload_pt_weights=False)

        model_type = getattr(qeff_model.model.config, "model_type", "")
        expected_function_tokens = {
            "qwen2_5_vl": ("QEffQwen2_5_VLDecoderLayer",),
            "qwen3_5": ("QEffQwen3_5DecoderLayer",),
            "qwen3_5_moe": ("QEffQwen3_5MoeDecoderLayer",),
        }.get(model_type, tuple())
        assert expected_function_tokens, f"Unsupported model_type for VLM subfunction test: {model_type}"

        has_decoder_layer, decoder_layer_names = has_decoder_layer_function(
            with_sub_func_onnx[-1], expected_function_tokens
        )
        assert has_decoder_layer, (
            "Model exported with use_onnx_subfunctions=True should contain expected decoder-layer "
            f"function definition. model_type={model_type}, expected_any={expected_function_tokens}"
        )
        print(f"\nDecoder-layer functions found: {decoder_layer_names}")
        qeff_model.compile(**compile_kwargs)

    if compile_only:
        manual_cleanup(qeff_model.onnx_path)
        return

    streamer = TextStreamer(processor.tokenizer)
    print("QPC Outputs (QAIC):")
    exec_info = qeff_model.generate(inputs=inputs, generation_len=NEW_GENERATION_TOKENS, streamer=streamer)
    print(exec_info)
    cloud_ai_100_tokens = exec_info.generated_ids[:, :-1]
    assert (pytorch_hf_tokens == cloud_ai_100_tokens).all(), "Tokens don't match for pytorch HF output and QPC output"
    manual_cleanup(qeff_model.onnx_path)


@pytest.mark.full_layers
@pytest.mark.on_qaic
@pytest.mark.multimodal
@pytest.mark.feature
@pytest.mark.parametrize("model_name", test_mm_models)
@pytest.mark.parametrize("kv_offload", [True])
def test_full_image_text_to_text_subfunction(model_name, kv_offload, manual_cleanup):
    torch.manual_seed(42)
    check_image_text_to_text_subfunction_core(model_name, kv_offload=kv_offload, manual_cleanup=manual_cleanup)


@pytest.mark.few_layers
@pytest.mark.on_qaic
@pytest.mark.multimodal
@pytest.mark.feature
@pytest.mark.parametrize("model_name", test_mm_models)
@pytest.mark.parametrize("kv_offload", [True])
def test_few_image_text_to_text_subfunction(model_name, kv_offload, manual_cleanup):
    torch.manual_seed(42)
    check_image_text_to_text_subfunction_core(
        model_name,
        kv_offload=kv_offload,
        num_hidden_layers=model_config_dict[model_name].get("n_layers", 2),
        manual_cleanup=manual_cleanup,
    )


@pytest.mark.dummy_layers
@pytest.mark.on_qaic
@pytest.mark.multimodal
@pytest.mark.feature
@pytest.mark.parametrize("model_name", test_mm_models)
@pytest.mark.parametrize("kv_offload", [True])
def test_dummy_image_text_to_text_subfunction(model_name, kv_offload, manual_cleanup):
    torch.manual_seed(42)
    check_image_text_to_text_subfunction_core(
        model_name,
        num_hidden_layers=model_config_dict[model_name]["num_layers"],
        kv_offload=kv_offload,
        manual_cleanup=manual_cleanup,
    )

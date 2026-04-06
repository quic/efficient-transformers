# -----------------------------------------------------------------------------
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------

import os
from time import perf_counter

import numpy as np
import onnx
import pytest
import torch
from peft import IA3Config, LoraConfig, get_peft_model
from transformers import AutoConfig, AutoModelForCausalLM

from QEfficient import QEffAutoPeftModelForCausalLM

configs = [
    pytest.param(
        AutoConfig.for_model(
            "llama",
            num_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=2,
            hidden_size=128,
            architectures=["LlamaForCausalLM"],
        ),
        LoraConfig(target_modules=["q_proj", "v_proj"], task_type="CAUSAL_LM", lora_alpha=8),
        id="llama-2l-4h-2kvh-128d-qv",
    ),
    pytest.param(
        AutoConfig.for_model(
            "mistral",
            num_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=2,
            hidden_size=128,
            architectures=["MistralForCausalLM"],
        ),
        LoraConfig(target_modules=["q_proj", "v_proj"], task_type="CAUSAL_LM", lora_alpha=6),
        id="mistral-2l-4h-128d-qv",
    ),
]


def create_peft_model(base_config, adapter_config, adapter_name="default"):
    base_model = AutoModelForCausalLM.from_config(base_config, attn_implementation="eager")
    adapted_model = get_peft_model(base_model, adapter_config, adapter_name)

    # Add random noise to adapter weights to avoid onnx export deduplicating all-zero weights
    for name, param in adapted_model.named_parameters():
        if name.endswith(f".{adapter_name}.weight") and torch.all(param == 0.0):
            param.data.add_(torch.rand(param.shape))

    return base_model, adapted_model


@pytest.mark.parametrize("base_config,adapter_config", configs)
def test_auto_peft_model_for_causal_lm_init(base_config, adapter_config):
    base_model, ia3_model = create_peft_model(base_config, IA3Config(task_type="CAUSAL_LM"))

    with pytest.raises(TypeError):
        QEffAutoPeftModelForCausalLM(base_model)

    with pytest.raises(NotImplementedError):
        QEffAutoPeftModelForCausalLM(ia3_model)

    base_model, lora_model = create_peft_model(base_config, adapter_config, "testAdapter101")
    lora_model.add_adapter("testAdapter102", adapter_config)
    qeff_model = QEffAutoPeftModelForCausalLM(lora_model)
    assert set(qeff_model.adapter_weights.keys()) == {"testAdapter101", "testAdapter102"}


@pytest.mark.parametrize("base_config,adapter_config", configs)
def test_auto_peft_model_for_causal_lm_from_pretrained(base_config, adapter_config, tmp_path):
    base_path = tmp_path / "base"
    adapter_path = tmp_path / "adapter"
    adapter_name = "testAdapter103"

    base_model = AutoModelForCausalLM.from_config(base_config, attn_implementation="eager")
    base_model.save_pretrained(base_path)

    base_model = AutoModelForCausalLM.from_pretrained(base_path)
    lora_model = get_peft_model(base_model, adapter_config, adapter_name)
    lora_model.save_pretrained(adapter_path)

    qeff_model = QEffAutoPeftModelForCausalLM.from_pretrained(adapter_path / adapter_name, adapter_name)
    assert set(qeff_model.adapter_weights.keys()) == {"testAdapter103"}

    with pytest.raises(NotImplementedError):
        QEffAutoPeftModelForCausalLM.from_pretrained(adapter_path / adapter_name, full_batch_size=4)


# This test isn't required anymore as different adapter names should generate different hashes. We'll
# phase out this test in some time.
@pytest.mark.skip(reason="Different adapter names will create different hashes so we'll skip this test.")
def test_auto_peft_model_for_causal_lm_hash():
    base_config_0, adapter_config_0 = configs[0].values
    base_config_1, adapter_config_1 = configs[1].values

    base_model_0, lora_model_0 = create_peft_model(base_config_0, adapter_config_0, "adapter_0_0")
    lora_model_0.add_adapter("adapter_0_1", adapter_config_0)
    lora_model_0.add_adapter("adapter_1_0", adapter_config_1)
    lora_model_0.add_adapter("adapter_1_1", adapter_config_1)

    qeff_model_0 = QEffAutoPeftModelForCausalLM(lora_model_0)

    qeff_model_0.set_adapter("adapter_0_0")
    hash_0_0_0 = qeff_model_0.model_hash
    qeff_model_0.set_adapter("adapter_0_1")
    hash_0_0_1 = qeff_model_0.model_hash
    assert hash_0_0_0 == hash_0_0_1
    qeff_model_0.set_adapter("adapter_1_0")
    hash_0_1_0 = qeff_model_0.model_hash
    qeff_model_0.set_adapter("adapter_1_1")
    hash_0_1_1 = qeff_model_0.model_hash
    assert hash_0_1_0 == hash_0_1_1
    assert hash_0_0_0 != hash_0_1_0

    base_model_1, lora_model_1 = create_peft_model(base_config_1, adapter_config_0, "adapter_0")
    lora_model_1.add_adapter("adapter_1", adapter_config_1)

    qeff_model_1 = QEffAutoPeftModelForCausalLM(lora_model_1)

    qeff_model_1.set_adapter("adapter_0")
    hash_1_0 = qeff_model_1.model_hash
    qeff_model_1.set_adapter("adapter_1")
    hash_1_1 = qeff_model_1.model_hash
    assert hash_1_0 != hash_1_1

    assert hash_0_0_0 != hash_1_0
    assert hash_0_1_0 != hash_1_1


@pytest.mark.parametrize("base_config,adapter_config", configs)
def test_auto_peft_model_for_causal_lm_export(base_config, adapter_config, tmp_path):
    _, lora_model = create_peft_model(base_config, adapter_config)
    qeff_model = QEffAutoPeftModelForCausalLM(lora_model)
    start = perf_counter()
    qeff_model.export(tmp_path)
    end = perf_counter()
    export_time_0 = end - start
    model_path = tmp_path.with_name(tmp_path.name + "-" + qeff_model.export_hash)
    assert model_path.is_dir()
    assert qeff_model.onnx_path.is_file()

    # Check if all the LoRA weights are converted to inputs and outputs
    onnx_model = onnx.load(qeff_model.onnx_path, load_external_data=False)
    input_names = {x.name for x in onnx_model.graph.input}
    output_names = {x.name for x in onnx_model.graph.output}
    for weight_name in qeff_model.adapter_weights[qeff_model.active_adapter]:
        assert weight_name in input_names
        assert weight_name + "_RetainedState" in output_names

    start = perf_counter()
    qeff_model.export(tmp_path)
    end = perf_counter()
    export_time_1 = end - start
    assert export_time_1 < 0.01 * export_time_0


@pytest.mark.parametrize("base_config,adapter_config", configs)
def test_auto_peft_model_for_causal_lm_activate_invalid(base_config, adapter_config, tmp_path):
    _, lora_model = create_peft_model(base_config, adapter_config)
    lora_model.add_adapter("invalid", LoraConfig(target_modules=["q_proj"], task_type="CAUSAL_LM"))
    qeff_model = QEffAutoPeftModelForCausalLM(lora_model)
    qeff_model.export(tmp_path)

    with pytest.raises(ValueError):
        qeff_model.set_adapter("invalid")


@pytest.mark.feature
@pytest.mark.on_qaic
@pytest.mark.parametrize("batch_size", [1, 4], ids=["bs1", "bs4"])
@pytest.mark.parametrize("base_config,adapter_config", configs)
def test_auto_peft_model_for_causal_lm_compile_generate(base_config, adapter_config, batch_size, tmp_path):
    _, lora_model = create_peft_model(base_config, adapter_config)
    qeff_model = QEffAutoPeftModelForCausalLM(lora_model)
    onnx_path = qeff_model.export(tmp_path)
    start = perf_counter()
    qeff_model.compile(onnx_path=onnx_path, batch_size=batch_size, prefill_seq_len=32, ctx_len=128)
    end = perf_counter()
    compile_time_0 = end - start

    qeff_model.generate(
        input_ids=np.zeros((batch_size, 32), dtype="int64"),
        attention_mask=np.concatenate(
            [
                np.ones((batch_size, 10), dtype="int64"),
                np.zeros((batch_size, 22), dtype="int64"),
            ],
            axis=1,
        ),
        max_new_tokens=10,
    )

    start = perf_counter()
    qeff_model.compile(onnx_path=onnx_path, batch_size=batch_size, prefill_seq_len=32, ctx_len=128)
    end = perf_counter()
    compile_time_1 = end - start
    assert compile_time_1 < 0.01 * compile_time_0
    assert os.path.isfile(os.path.join(os.path.dirname(qeff_model.qpc_path), "qconfig.json"))

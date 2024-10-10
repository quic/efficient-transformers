# -----------------------------------------------------------------------------
#
# Copyright (c) 2024 Qualcomm Innovation Center, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------
import sys
from pathlib import Path
from time import perf_counter

import numpy as np
import pytest
from peft import LoraConfig
from transformers import AutoConfig, AutoModelForCausalLM

from QEfficient import QEffAutoLoraModelForCausalLM

INTMAX = sys.maxsize

configs = [
    pytest.param(
        AutoConfig.for_model(
            "llama", num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2, hidden_size=128
        ),
        LoraConfig(target_modules=["q_proj", "v_proj"], task_type="CAUSAL_LM", lora_alpha=8),
        id="llama-2l-4h-2kvh-128d-qv",
    ),
    pytest.param(
        AutoConfig.for_model(
            "mistral", num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2, hidden_size=128
        ),
        LoraConfig(target_modules=["q_proj", "v_proj"], task_type="CAUSAL_LM", lora_alpha=6),
        id="mistral-2l-4h-128d-qv",
    ),
]

model_samples = [
    pytest.param("mistralai/Mistral-7B-v0.1", "predibase/gsm8k", "predibase/dbpedia"),
    pytest.param(
        "meta-llama/Meta-Llama-3-8B",
        "hallisky/lora-type-narrative-llama-3-8b",
        "hallisky/lora-grade-elementary-llama-3-8b",
    ),
]


def create_lora_base_model(base_config):
    base_model = AutoModelForCausalLM.from_config(base_config, attn_implementation="eager")
    lora_base_model = QEffAutoLoraModelForCausalLM(
        base_model, pretrained_model_name_or_path=str(base_config.model_type)
    )

    return lora_base_model


def load_adapter_with_random_weights(lora_base_model, adapter_config, adapter_name):
    lora_base_model.adapter_configs[adapter_name] = adapter_config
    lora_base_model.adapter_weights[adapter_name] = {"weights": np.ones((3, 3))}

    return lora_base_model


# test model initialization using __init__ approach
@pytest.mark.parametrize("base_model_name,adapter_id_0,adapter_id_1", model_samples)
def test_auto_lora_model_for_causal_lm_init(base_model_name, adapter_id_0, adapter_id_1):
    model_hf = AutoModelForCausalLM.from_pretrained(base_model_name, num_hidden_layers=1)
    qeff_model = QEffAutoLoraModelForCausalLM(model_hf, pretrained_model_name_or_path=base_model_name)

    assert qeff_model.base_model_name == base_model_name
    assert len(qeff_model.adapter_weights) == 0
    assert len(qeff_model.adapter_configs) == 0
    assert len(qeff_model.active_adapters) == 0
    assert qeff_model.max_num_adapters == 0
    assert len(qeff_model.active_adapter_to_id) == 0


# test model initialization using from_pretrained approach
@pytest.mark.parametrize("base_model_name,adapter_id_0,adapter_id_1", model_samples)
def test_auto_lora_model_for_causal_lm_from_pretrained(base_model_name, adapter_id_0, adapter_id_1):
    qeff_model = QEffAutoLoraModelForCausalLM.from_pretrained(base_model_name, num_hidden_layers=1)

    assert qeff_model.base_model_name == base_model_name
    assert len(qeff_model.adapter_weights) == 0
    assert len(qeff_model.adapter_configs) == 0
    assert len(qeff_model.active_adapters) == 0
    assert qeff_model.max_num_adapters == 0
    assert len(qeff_model.active_adapter_to_id) == 0


# test model hash
def test_auto_lora_model_for_causal_lm_hash():
    base_config_0, adapter_config_0 = configs[0].values
    base_config_1, adapter_config_1 = configs[1].values

    qeff_model_0 = create_lora_base_model(base_config_0)
    qeff_model_0 = load_adapter_with_random_weights(qeff_model_0, adapter_config_0, "adapter_0")
    qeff_model_0 = load_adapter_with_random_weights(qeff_model_0, adapter_config_1, "adapter_1")

    qeff_model_1 = create_lora_base_model(base_config_1)
    qeff_model_1 = load_adapter_with_random_weights(qeff_model_1, adapter_config_0, "adapter_0")
    qeff_model_1 = load_adapter_with_random_weights(qeff_model_1, adapter_config_1, "adapter_1")

    qeff_model_0_1 = create_lora_base_model(base_config_0)
    qeff_model_0_1 = load_adapter_with_random_weights(qeff_model_0_1, adapter_config_0, "adapter_0")
    qeff_model_0_1 = load_adapter_with_random_weights(qeff_model_0_1, adapter_config_1, "adapter_1")
    qeff_model_0_1.set_adapter("adapter_0")
    model_hash_0_1_0 = qeff_model_0_1.model_hash
    qeff_model_0_1.set_adapter("adapter_1")
    model_hash_0_1_1 = qeff_model_0_1.model_hash

    # check num of adapter config matters
    qeff_model_0.set_adapter("adapter_0")
    model_hash_0_0 = qeff_model_0.model_hash
    qeff_model_0.set_adapter("adapter_1")
    model_hash_0_1 = qeff_model_0.model_hash
    assert model_hash_0_0 != model_hash_0_1

    # check if same model, same adapter result in same hash
    assert model_hash_0_1_0 == model_hash_0_0
    assert model_hash_0_1_1 == model_hash_0_1

    # check base model configs matters
    qeff_model_1.set_adapter("adapter_0")
    qeff_model_1.set_adapter("adapter_1")
    model_hash_1_1 = qeff_model_1.model_hash
    assert model_hash_1_1 != model_hash_0_1

    # check adapter orders matters
    qeff_model_1.delete_adapter("adapter_0")
    qeff_model_1.delete_adapter("adapter_1")
    qeff_model_1.set_adapter("adapter_1")
    qeff_model_1.set_adapter("adapter_0")
    model_hash_1_2 = qeff_model_1.model_hash
    assert model_hash_1_2 != model_hash_1_1

    # check if different adapter config, but same adapter name matters
    qeff_model_0 = load_adapter_with_random_weights(qeff_model_0, adapter_config_0, "adapter_1")
    model_hash_0_2 = qeff_model_0.model_hash
    assert model_hash_0_2 != model_hash_0_1


# test load_adapter() and set_adapter() and get_adapter_id()
@pytest.mark.parametrize("base_model_name,adapter_id_0,adapter_id_1", model_samples[:1])
def test_auto_lora_model_for_causal_lm_load_set_adapter_id_check(base_model_name, adapter_id_0, adapter_id_1):
    qeff_model = QEffAutoLoraModelForCausalLM.from_pretrained(base_model_name, num_hidden_layers=1)

    qeff_model.load_adapter(adapter_id_0, "adapter_0")
    qeff_model.load_adapter(adapter_id_1, "adapter_1")
    set_id_0 = qeff_model.set_adapter("adapter_0")
    set_id_1 = qeff_model.set_adapter("adapter_1")

    assert set_id_1 == set_id_0 + 1

    qeff_model.load_adapter(adapter_id_1, "adapter_2")
    qeff_model.set_adapter("adapter_2")
    qeff_model.delete_adapter("adapter_1")

    update_id_0 = qeff_model.get_adapter_id("adapter_0")
    update_id_2 = qeff_model.get_adapter_id("adapter_2")
    assert set_id_0 == update_id_0
    assert set_id_1 == update_id_2

    with pytest.raises(KeyError):
        qeff_model.get_adapter_id("adapter_1")


# test unload_adapter() and delete_adapter()
@pytest.mark.parametrize("base_model_name,adapter_id_0,adapter_id_1", model_samples[1:])
def test_auto_lora_model_for_causal_lm_unload_delete_adapter(base_model_name, adapter_id_0, adapter_id_1):
    qeff_model = QEffAutoLoraModelForCausalLM.from_pretrained(base_model_name, num_hidden_layers=1)

    qeff_model.load_adapter(adapter_id_0, "adapter_0")
    qeff_model.load_adapter(adapter_id_1, "adapter_1")
    qeff_model.set_adapter("adapter_0")

    assert not qeff_model.unload_adapter("adapter_0")  # active adapter
    assert qeff_model.unload_adapter("adapter_1")  # valid unload
    assert not qeff_model.unload_adapter("adapter_2")  # not loaded adapter

    assert qeff_model.delete_adapter("adapter_0")  # active adapter
    assert not qeff_model.delete_adapter("adapter_1")  # not active adapter


# test the export, export caching, compile, generate workflow
@pytest.mark.parametrize("base_model_name,adapter_id_0,adapter_id_1", model_samples[:1])
def test_auto_lora_model_for_causal_lm_export_compile_generate(base_model_name, adapter_id_0, adapter_id_1, tmp_path):
    qeff_model = QEffAutoLoraModelForCausalLM.from_pretrained(base_model_name, num_hidden_layers=1)

    qeff_model.load_adapter(adapter_id_0, "adapter_0")
    qeff_model.load_adapter(adapter_id_1, "adapter_1")
    qeff_model.set_adapter("adapter_0")
    qeff_model.set_adapter("adapter_1")

    # export
    start = perf_counter()
    qeff_model.export(export_dir=tmp_path, full_batch_size=1)  # NOTE: should export with full_batch_size enabled
    end = perf_counter()
    export_time_0 = end - start
    model_path = tmp_path.with_name(tmp_path.name + "-" + qeff_model.model_hash)
    assert model_path.is_dir()
    assert Path(qeff_model.onnx_path).is_file()

    # test export caching
    start = perf_counter()
    qeff_model.export(export_dir=tmp_path, full_batch_size=1)
    end = perf_counter()
    export_time_1 = end - start
    assert export_time_1 < 0.01 * export_time_0

    # test compile
    qeff_model.compile(num_cores=16, device_group=[0])
    assert Path(qeff_model.qpc_path).is_dir()

    # test generate
    prompts = ["hello!", "hi", "hello, my name is", "hey"]
    qeff_model.generate(prompts, [0], prompt_to_lora_id_mapping=[0, 1, 0, INTMAX])

# -----------------------------------------------------------------------------
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------

import os
from pathlib import Path
from time import perf_counter

import numpy as np
import pytest
from peft import LoraConfig
from transformers import AutoConfig, AutoModelForCausalLM

from QEfficient import QEffAutoPeftModelForCausalLM
from QEfficient.peft.lora import QEffAutoLoraModelForCausalLM
from QEfficient.utils import load_hf_tokenizer

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
    lora_base_model = QEffAutoLoraModelForCausalLM(base_model)

    return lora_base_model


# test model initialization using __init__ approach
@pytest.mark.parametrize("base_model_name,adapter_id_0,adapter_id_1", model_samples)
def test_auto_lora_model_for_causal_lm_init(base_model_name, adapter_id_0, adapter_id_1):
    model_hf = AutoModelForCausalLM.from_pretrained(base_model_name, num_hidden_layers=1)
    qeff_model = QEffAutoLoraModelForCausalLM(model_hf)

    assert len(qeff_model.adapter_weights) == 0
    assert len(qeff_model.adapter_configs) == 0
    assert len(qeff_model.active_adapter_to_id) == 0


# test model initialization using from_pretrained approach
@pytest.mark.parametrize("base_model_name,adapter_id_0,adapter_id_1", model_samples)
def test_auto_lora_model_for_causal_lm_from_pretrained(base_model_name, adapter_id_0, adapter_id_1):
    qeff_model = QEffAutoLoraModelForCausalLM.from_pretrained(
        pretrained_model_name_or_path=base_model_name, num_hidden_layers=1
    )

    assert len(qeff_model.adapter_weights) == 0
    assert len(qeff_model.adapter_configs) == 0
    assert len(qeff_model.active_adapter_to_id) == 0


# test peft model initialization using from_pretrained approach
@pytest.mark.parametrize("base_model_name,adapter_id_0,adapter_id_1", model_samples)
def test_auto_peft_model_for_causal_lm_from_pretrained(base_model_name, adapter_id_0, adapter_id_1):
    qeff_model = QEffAutoPeftModelForCausalLM.from_pretrained(
        adapter_id_0, "id_0", finite_adapters=True, num_hidden_layers=1
    )
    qeff_model_tmp = QEffAutoPeftModelForCausalLM.from_pretrained(
        adapter_id_0, adapter_name="id_0", finite_adapters=True, num_hidden_layers=1
    )

    assert qeff_model.active_adapter_to_id == qeff_model_tmp.active_adapter_to_id
    del qeff_model_tmp
    assert isinstance(qeff_model, QEffAutoLoraModelForCausalLM)
    assert len(qeff_model.adapter_weights) == 1
    assert len(qeff_model.adapter_configs) == 1
    assert len(qeff_model.active_adapter_to_id) == 1

    # test pass without adapter name
    with pytest.raises(TypeError):
        QEffAutoLoraModelForCausalLM.from_pretrained(adapter_id_0, finite_adapters=True, num_hidden_layers=1)

    # test pass with adapter name as integer
    with pytest.raises(TypeError):
        QEffAutoLoraModelForCausalLM.from_pretrained(adapter_id_0, 0, finite_adapters=True, num_hidden_layers=1)


# test the init assertion for models that are not supported
@pytest.mark.parametrize("base_model_name", ["distilbert/distilgpt2"])
def test_auto_lora_model_for_causal_lm_init_from_unsupported_model(base_model_name):
    model_hf = AutoModelForCausalLM.from_pretrained(base_model_name, num_hidden_layers=1)
    with pytest.raises(NotImplementedError):
        QEffAutoLoraModelForCausalLM(model_hf)

    with pytest.raises(NotImplementedError):
        QEffAutoLoraModelForCausalLM.from_pretrained(base_model_name, num_hidden_layers=1)


# This test isn't required anymore as different adapter names should generate different hashes. We'll
# phase out this test in some time.
@pytest.mark.skip(reason="Different adapter names will create different hashes so we'll skip this test.")
def test_auto_lora_model_for_causal_lm_hash():
    base_config_0, adapter_config_0 = configs[0].values
    base_config_1, adapter_config_1 = configs[1].values

    qeff_model_0 = create_lora_base_model(base_config_0)
    qeff_model_0.load_adapter(
        "dummy_id", "adapter_0", adapter_config=adapter_config_0, adapter_weight={"weights": np.ones((3, 3))}
    )
    qeff_model_0.load_adapter(
        "dummy_id", "adapter_1", adapter_config=adapter_config_1, adapter_weight={"weights": np.ones((3, 3))}
    )
    model_hash_0_0 = qeff_model_0.model_hash

    qeff_model_1 = create_lora_base_model(base_config_1)
    qeff_model_1.load_adapter(
        "dummy_id", "adapter_0", adapter_config=adapter_config_0, adapter_weight={"weights": np.ones((3, 3))}
    )
    qeff_model_1.load_adapter(
        "dummy_id", "adapter_1", adapter_config=adapter_config_1, adapter_weight={"weights": np.ones((3, 3))}
    )
    model_hash_1_0 = qeff_model_1.model_hash

    qeff_model_0_1 = create_lora_base_model(base_config_0)
    qeff_model_0_1.load_adapter(
        "dummy_id", "adapter_0", adapter_config=adapter_config_0, adapter_weight={"weights": np.ones((3, 3))}
    )
    qeff_model_0_1.load_adapter(
        "dummy_id", "adapter_1", adapter_config=adapter_config_1, adapter_weight={"weights": np.ones((3, 3))}
    )
    model_hash_0_1_0 = qeff_model_0_1.model_hash

    # check if same model, same adapter config, same adapter weight, result in same hash
    assert model_hash_0_1_0 == model_hash_0_0

    # check if same model, same adapter config, but different weight, result in different hash
    qeff_model_0_1.unload_adapter("adapter_1")
    qeff_model_0_1.unload_adapter("adapter_0")
    qeff_model_0_1.load_adapter(
        "dummy_id", "adapter_0", adapter_config=adapter_config_0, adapter_weight={"weights": np.random.randn(3, 3)}
    )
    qeff_model_0_1.load_adapter(
        "dummy_id", "adapter_1", adapter_config=adapter_config_1, adapter_weight={"weights": np.random.randn(3, 3)}
    )
    model_hash_0_1_1 = qeff_model_0_1.model_hash
    assert model_hash_0_1_1 != model_hash_0_0

    # check base model configs difference result in different hash
    assert model_hash_0_0 != model_hash_1_0

    # check different adapter orders, result in different hash
    qeff_model_1.unload_adapter("adapter_0")
    qeff_model_1.unload_adapter("adapter_1")
    qeff_model_1.load_adapter(
        "dummy_id", "adapter_1", adapter_config=adapter_config_1, adapter_weight={"weights": np.ones((3, 3))}
    )
    qeff_model_1.load_adapter(
        "dummy_id", "adapter_0", adapter_config=adapter_config_0, adapter_weight={"weights": np.ones((3, 3))}
    )
    model_hash_1_1 = qeff_model_1.model_hash
    assert model_hash_1_1 != model_hash_1_0

    # check if same adapter name, but different config, result in different hash
    qeff_model_0.unload_adapter("adapter_1")
    qeff_model_0.load_adapter(
        "dummy_id", "adapter_1", adapter_config=adapter_config_0, adapter_weight={"weights": np.ones((3, 3))}
    )
    model_hash_0_1 = qeff_model_0.model_hash
    assert model_hash_0_1 != model_hash_0_0


# test download_adapter(), load_adapter() and unload_adapter()
@pytest.mark.parametrize("base_model_name,adapter_id_0,adapter_id_1", model_samples[1:])
def test_auto_lora_model_for_causal_lm_load_unload_adapter(base_model_name, adapter_id_0, adapter_id_1):
    qeff_model = QEffAutoLoraModelForCausalLM.from_pretrained(base_model_name, num_hidden_layers=1)

    qeff_model.download_adapter(adapter_id_0, "adapter_0")
    qeff_model.download_adapter(adapter_id_1, "adapter_1")

    qeff_model.load_adapter(adapter_id_0, "adapter_0")

    assert not qeff_model.unload_adapter("adapter_1")  # not active adapter
    assert qeff_model.unload_adapter("adapter_0")  # valid unload


# test the export, export caching, compile and generate workflow in noncb mode
@pytest.mark.on_qaic
@pytest.mark.feature
@pytest.mark.parametrize("base_model_name,adapter_id_0,adapter_id_1", model_samples[:1])
def test_auto_lora_model_for_causal_lm_noncb_export_compile_generate(
    base_model_name, adapter_id_0, adapter_id_1, tmp_path
):
    qeff_model = QEffAutoLoraModelForCausalLM.from_pretrained(base_model_name, num_hidden_layers=1)

    qeff_model.load_adapter(adapter_id_0, "adapter_0")
    qeff_model.load_adapter(adapter_id_1, "adapter_1")

    # export
    start = perf_counter()
    onnx_path = qeff_model.export(export_dir=tmp_path)
    end = perf_counter()
    export_time_0 = end - start
    model_path = tmp_path.with_name(tmp_path.name + "-" + qeff_model.export_hash)
    assert model_path.is_dir()
    assert Path(qeff_model.onnx_path).is_file()

    # test export caching
    start = perf_counter()
    qeff_model.export(export_dir=tmp_path)
    end = perf_counter()
    export_time_1 = end - start
    assert export_time_1 < export_time_0

    # test compile
    qeff_model.compile(onnx_path=onnx_path, prefill_seq_len=32, ctx_len=64)
    assert Path(qeff_model.qpc_path).is_dir()
    assert os.path.isfile(os.path.join(os.path.dirname(qeff_model.qpc_path), "qconfig.json"))

    # test generate
    prompts = ["hello!", "hi", "hello, my name is", "hey"]
    qeff_model.generate(
        tokenizer=load_hf_tokenizer(pretrained_model_name_or_path=base_model_name),
        prompts=prompts,
        prompt_to_adapter_mapping=["adapter_0", "adapter_1", "adapter_0", "base"],
    )


# test the compile and generate workflow in cb mode
@pytest.mark.on_qaic
@pytest.mark.feature
@pytest.mark.parametrize("base_model_name,adapter_id_0,adapter_id_1", model_samples[:1])
def test_auto_lora_model_for_causal_lm_cb_compile_generate(base_model_name, adapter_id_0, adapter_id_1, tmp_path):
    qeff_model = QEffAutoLoraModelForCausalLM.from_pretrained(
        base_model_name, continuous_batching=True, num_hidden_layers=1
    )

    qeff_model.load_adapter(adapter_id_0, "adapter_0")
    qeff_model.load_adapter(adapter_id_1, "adapter_1")

    # test compile
    qeff_model.compile(prefill_seq_len=32, ctx_len=512, full_batch_size=2)
    assert Path(qeff_model.qpc_path).is_dir()
    assert os.path.isfile(os.path.join(os.path.dirname(qeff_model.qpc_path), "qconfig.json"))

    # test generate
    prompts = ["hello!", "hi", "hello, my name is", "hey"]
    qeff_model.generate(
        tokenizer=load_hf_tokenizer(pretrained_model_name_or_path=base_model_name),
        prompts=prompts,
        prompt_to_adapter_mapping=["adapter_0", "adapter_1", "adapter_0", "base"],
    )

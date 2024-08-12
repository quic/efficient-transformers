import pytest
import torch
from transformers.models.codegen.modeling_codegen import CodeGenConfig, CodeGenForCausalLM
from transformers.models.falcon.modeling_falcon import FalconConfig, FalconForCausalLM
from transformers.models.gpt2.modeling_gpt2 import GPT2Config, GPT2LMHeadModel
from transformers.models.gptj.modeling_gptj import GPTJConfig, GPTJForCausalLM
from transformers.models.llama.modeling_llama import LlamaConfig, LlamaForCausalLM
from transformers.models.mistral.modeling_mistral import MistralConfig, MistralForCausalLM
from transformers.models.mixtral.modeling_mixtral import MixtralConfig, MixtralForCausalLM
from transformers.models.mpt.modeling_mpt import MptConfig, MptForCausalLM
from transformers.models.phi.modeling_phi import PhiConfig, PhiForCausalLM
from transformers.models.phi3.modeling_phi3 import Phi3Config, Phi3ForCausalLM
from transformers.models.qwen2.modeling_qwen2 import Qwen2Config, Qwen2ForCausalLM
from transformers.models.starcoder2.modeling_starcoder2 import Starcoder2Config, Starcoder2ForCausalLM

from QEfficient.transformers.pytorch_transforms import CustomOpsTransform, KVCacheTransform
from QEfficient.utils._utils import get_padding_shape_from_config
from QEfficient.utils.logging_utils import logger

KVCacheTransformTestConfigs = [
    (
        LlamaConfig,
        LlamaForCausalLM,
        8,
        32,
        {
            "num_hidden_layers": 3,
            "num_key_value_heads": 8,
            "num_attention_heads": 32,
            "hidden_size": 128,
            "intermediate_size": 512,
        },
        0.8,
    ),
    (
        LlamaConfig,
        LlamaForCausalLM,
        8,
        32,
        {
            "num_hidden_layers": 1,
            "num_key_value_heads": 8,
            "num_attention_heads": 32,
            "hidden_size": 128,
            "intermediate_size": 512,
        },
        0.8,
    ),
    (
        LlamaConfig,
        LlamaForCausalLM,
        8,
        32,
        {
            "num_hidden_layers": 3,
            "num_key_value_heads": 32,
            "num_attention_heads": 32,
            "hidden_size": 128,
            "intermediate_size": 512,
        },
        0.8,
    ),
    (
        LlamaConfig,
        LlamaForCausalLM,
        8,
        32,
        {
            "num_hidden_layers": 1,
            "num_key_value_heads": 32,
            "num_attention_heads": 32,
            "hidden_size": 128,
            "intermediate_size": 512,
        },
        0.8,
    ),
    (GPT2Config, GPT2LMHeadModel, 8, 32, {"n_layer": 3, "n_head": 12, "n_embd": 192, "n_inner": 512}, 0.8),
    (GPT2Config, GPT2LMHeadModel, 8, 32, {"n_layer": 1, "n_head": 12, "n_embd": 192, "n_inner": 512}, 0.8),
    (CodeGenConfig, CodeGenForCausalLM, 8, 32, {"n_layer": 1, "n_head": 16, "n_embd": 1024, "n_inner": 2048}, 0.8),
    (CodeGenConfig, CodeGenForCausalLM, 8, 32, {"n_layer": 3, "n_head": 16, "n_embd": 1024, "n_inner": 2048}, 0.8),
    (
        FalconConfig,
        FalconForCausalLM,
        8,
        32,
        {"num_hidden_layers": 1, "multi_query": True, "num_attention_heads": 71, "hidden_size": 4544},
        1.5,
    ),
    (
        FalconConfig,
        FalconForCausalLM,
        8,
        32,
        {"num_hidden_layers": 3, "multi_query": False, "num_attention_heads": 71, "hidden_size": 4544},
        1.5,
    ),
    (
        FalconConfig,
        FalconForCausalLM,
        8,
        32,
        {"num_hidden_layers": 1, "multi_query": False, "num_attention_heads": 71, "hidden_size": 4544},
        1.5,
    ),
    (
        FalconConfig,
        FalconForCausalLM,
        8,
        32,
        {"num_hidden_layers": 3, "multi_query": True, "num_attention_heads": 71, "hidden_size": 4544},
        1.5,
    ),
    (GPTJConfig, GPTJForCausalLM, 8, 32, {"n_layer": 3, "n_head": 16, "n_embd": 4096, "n_inner": 512}, 1),
    (GPTJConfig, GPTJForCausalLM, 8, 32, {"n_layer": 1, "n_head": 16, "n_embd": 4096, "n_inner": 512}, 1.2),
    (
        MistralConfig,
        MistralForCausalLM,
        8,
        32,
        {
            "num_hidden_layers": 1,
            "num_key_value_heads": 8,
            "num_attention_heads": 32,
            "hidden_size": 128,
            "intermediate_size": 512,
        },
        0.8,
    ),
    (
        MistralConfig,
        MistralForCausalLM,
        8,
        32,
        {
            "num_hidden_layers": 1,
            "num_key_value_heads": 32,
            "num_attention_heads": 32,
            "hidden_size": 128,
            "intermediate_size": 512,
        },
        0.8,
    ),
    (
        MistralConfig,
        MistralForCausalLM,
        8,
        32,
        {
            "num_hidden_layers": 3,
            "num_key_value_heads": 8,
            "num_attention_heads": 32,
            "hidden_size": 128,
            "intermediate_size": 512,
        },
        0.8,
    ),
    (
        MistralConfig,
        MistralForCausalLM,
        8,
        32,
        {
            "num_hidden_layers": 3,
            "num_key_value_heads": 32,
            "num_attention_heads": 32,
            "hidden_size": 128,
            "intermediate_size": 512,
        },
        0.8,
    ),
    (
        MixtralConfig,
        MixtralForCausalLM,
        8,
        32,
        {
            "num_hidden_layers": 1,
            "num_key_value_heads": 8,
            "num_attention_heads": 32,
            "hidden_size": 128,
            "intermediate_size": 512,
        },
        0.8,
    ),
    (
        MixtralConfig,
        MixtralForCausalLM,
        8,
        32,
        {
            "num_hidden_layers": 1,
            "num_key_value_heads": 32,
            "num_attention_heads": 32,
            "hidden_size": 128,
            "intermediate_size": 512,
        },
        0.8,
    ),
    (
        MixtralConfig,
        MixtralForCausalLM,
        8,
        32,
        {
            "num_hidden_layers": 3,
            "num_key_value_heads": 8,
            "num_attention_heads": 32,
            "hidden_size": 128,
            "intermediate_size": 512,
        },
        0.8,
    ),
    (
        MixtralConfig,
        MixtralForCausalLM,
        8,
        32,
        {
            "num_hidden_layers": 3,
            "num_key_value_heads": 32,
            "num_attention_heads": 32,
            "hidden_size": 128,
            "intermediate_size": 512,
        },
        0.8,
    ),
    (MptConfig, MptForCausalLM, 8, 32, {"n_layers": 1, "n_heads": 16, "d_model": 2048}, 0.8),
    (MptConfig, MptForCausalLM, 8, 32, {"n_layers": 3, "n_heads": 16, "d_model": 2048}, 0.8),
    (
        PhiConfig,
        PhiForCausalLM,
        8,
        32,
        {
            "num_hidden_layers": 3,
            "num_key_value_heads": 8,
            "num_attention_heads": 32,
            "hidden_size": 128,
            "intermediate_size": 512,
        },
        0.8,
    ),
    (
        PhiConfig,
        PhiForCausalLM,
        8,
        32,
        {
            "num_hidden_layers": 3,
            "num_key_value_heads": 32,
            "num_attention_heads": 32,
            "hidden_size": 128,
            "intermediate_size": 512,
        },
        0.8,
    ),
    (
        PhiConfig,
        PhiForCausalLM,
        8,
        32,
        {
            "num_hidden_layers": 1,
            "num_key_value_heads": 8,
            "num_attention_heads": 32,
            "hidden_size": 128,
            "intermediate_size": 512,
        },
        0.8,
    ),
    (
        PhiConfig,
        PhiForCausalLM,
        8,
        32,
        {
            "num_hidden_layers": 1,
            "num_key_value_heads": 32,
            "num_attention_heads": 32,
            "hidden_size": 128,
            "intermediate_size": 512,
        },
        0.8,
    ),
    (
        Phi3Config,
        Phi3ForCausalLM,
        8,
        32,
        {
            "num_hidden_layers": 1,
            "num_key_value_heads": 32,
            "num_attention_heads": 32,
            "hidden_size": 128,
            "intermediate_size": 512,
        },
        0.8,
    ),
    (
        Phi3Config,
        Phi3ForCausalLM,
        8,
        32,
        {
            "num_hidden_layers": 1,
            "num_key_value_heads": 8,
            "num_attention_heads": 32,
            "hidden_size": 128,
            "intermediate_size": 512,
        },
        0.8,
    ),
    (
        Phi3Config,
        Phi3ForCausalLM,
        8,
        32,
        {
            "num_hidden_layers": 3,
            "num_key_value_heads": 32,
            "num_attention_heads": 32,
            "hidden_size": 128,
            "intermediate_size": 512,
        },
        0.8,
    ),
    (
        Phi3Config,
        Phi3ForCausalLM,
        8,
        32,
        {
            "num_hidden_layers": 3,
            "num_key_value_heads": 8,
            "num_attention_heads": 32,
            "hidden_size": 128,
            "intermediate_size": 512,
        },
        0.8,
    ),
    (
        Qwen2Config,
        Qwen2ForCausalLM,
        8,
        32,
        {
            "num_hidden_layers": 1,
            "num_key_value_heads": 8,
            "num_attention_heads": 32,
            "hidden_size": 128,
            "intermediate_size": 512,
        },
        0.8,
    ),
    (
        Qwen2Config,
        Qwen2ForCausalLM,
        8,
        32,
        {
            "num_hidden_layers": 1,
            "num_key_value_heads": 32,
            "num_attention_heads": 32,
            "hidden_size": 128,
            "intermediate_size": 512,
        },
        0.8,
    ),
    (
        Qwen2Config,
        Qwen2ForCausalLM,
        8,
        32,
        {
            "num_hidden_layers": 3,
            "num_key_value_heads": 8,
            "num_attention_heads": 32,
            "hidden_size": 128,
            "intermediate_size": 512,
        },
        0.8,
    ),
    (
        Qwen2Config,
        Qwen2ForCausalLM,
        8,
        32,
        {
            "num_hidden_layers": 3,
            "num_key_value_heads": 32,
            "num_attention_heads": 32,
            "hidden_size": 128,
            "intermediate_size": 512,
        },
        0.8,
    ),
    (
        Starcoder2Config,
        Starcoder2ForCausalLM,
        8,
        32,
        {
            "num_hidden_layers": 3,
            "num_key_value_heads": 2,
            "num_attention_heads": 24,
            "hidden_size": 192,
            "intermediate_size": 512,
        },
        0.8,
    ),
    (
        Starcoder2Config,
        Starcoder2ForCausalLM,
        8,
        32,
        {
            "num_hidden_layers": 1,
            "num_key_value_heads": 2,
            "num_attention_heads": 24,
            "hidden_size": 192,
            "intermediate_size": 512,
        },
        0.8,
    ),
    (
        Starcoder2Config,
        Starcoder2ForCausalLM,
        8,
        32,
        {
            "num_hidden_layers": 1,
            "num_key_value_heads": 24,
            "num_attention_heads": 24,
            "hidden_size": 192,
            "intermediate_size": 512,
        },
        0.8,
    ),
    (
        Starcoder2Config,
        Starcoder2ForCausalLM,
        8,
        32,
        {
            "num_hidden_layers": 3,
            "num_key_value_heads": 24,
            "num_attention_heads": 24,
            "hidden_size": 192,
            "intermediate_size": 512,
        },
        0.8,
    ),
]


def compare_original_vs_kv_model_pt_outputs(original_val, kv_val, tolerance=1e-6) -> bool:
    # Base case
    if original_val is None:
        assert kv_val is None
        return True
    elif isinstance(original_val, torch.Tensor):
        mae = torch.mean(torch.abs(original_val - kv_val))
        if mae >= tolerance:
            logger.critical(f"MAE={mae} is greater than expected tolerance={tolerance}")
            return False
        return True

    # Call recursively if tuple/list
    elif isinstance(original_val, (tuple, list)):
        for sub_orig_val, sub_kv_val in zip(original_val, kv_val):
            if not compare_original_vs_kv_model_pt_outputs(sub_orig_val, sub_kv_val, tolerance):
                return False
        return True
    else:
        raise TypeError(f"got unexpected type inputs {type(original_val)}")


def run_kv_cache_transform_and_test(
    hf_model,
    num_hidden_layers,
    padding_shape,
    vocab_size,
    input_len,
    logits_tolerance=0.8,
):
    hf_model.eval()
    # Run original model
    input_ids = torch.randint(0, vocab_size, size=(1, input_len))
    with torch.inference_mode():
        original_model_outputs = hf_model(input_ids=input_ids, output_hidden_states=True)

    # Apply transform
    hf_model, transformed = KVCacheTransform.apply(hf_model)
    assert transformed

    # Prepare KV model inputs
    past_key_values = []
    for _ in range(num_hidden_layers):
        past_key = torch.zeros((padding_shape), dtype=torch.float32)
        past_value = torch.zeros((padding_shape), dtype=torch.float32)
        pkv = (past_key, past_value)
        past_key_values.append(pkv)

    # Run KV model
    with torch.inference_mode():
        transformed_model_outputs = hf_model(
            input_ids=input_ids,
            position_ids=torch.Tensor([range(input_ids.shape[1])]).long(),
            past_key_values=tuple(past_key_values),
            output_hidden_states=True,
        )

    assert original_model_outputs.keys() == transformed_model_outputs.keys()

    # FIXME: Tolerance should not be so high for logits
    assert compare_original_vs_kv_model_pt_outputs(
        original_model_outputs["logits"], transformed_model_outputs["logits"], tolerance=logits_tolerance
    ), "Logits are not matching with tolerance=0.8"
    assert compare_original_vs_kv_model_pt_outputs(
        original_model_outputs["hidden_states"], transformed_model_outputs["hidden_states"], tolerance=1e-6
    )

    # Slice Past key values based on input_len
    pkv = transformed_model_outputs["past_key_values"][0]
    new_pkv = []
    for past_key_value in pkv:
        new_pkv.append(past_key_value[:, :, :input_len, :])
    transformed_model_outputs["past_key_values"] = (tuple(new_pkv),)

    assert compare_original_vs_kv_model_pt_outputs(
        original_model_outputs["past_key_values"], transformed_model_outputs["past_key_values"], tolerance=1e-10
    )


@pytest.mark.parametrize("input_size", [2, 5], ids=lambda x: "input_size=" + str(x))
@pytest.mark.parametrize("hidden_size", [64, 1024], ids=lambda x: "hidden_size=" + str(x))
@pytest.mark.parametrize("module", CustomOpsTransform._module_mapping.keys(), ids=lambda x: "module=" + x.__name__)
def test_rms_norm_ops_transform(module: torch.nn.Module, hidden_size: int, input_size: int) -> None:
    """Test custom Ops transform individually

    Args:
        module (nn.Module): Pytorch module
        hidden_size (int): hidden_size for RMSNorm operation
        input_size (int): Random inputs shape for testing
    """
    model = module(hidden_size=hidden_size)
    rand_data = torch.rand(input_size, hidden_size)

    original_output = model(rand_data)

    model, transformed = CustomOpsTransform.apply(model)
    assert transformed

    transformed_model_output = model(rand_data)

    assert not isinstance(model, module)
    assert torch.all(original_output == transformed_model_output)


@pytest.mark.parametrize(
    "config_class,model_class,input_len,ctx_len,kwargs,logits_tolerance", KVCacheTransformTestConfigs
)
def test_kv_cache_transform(config_class, model_class, input_len, ctx_len, kwargs, logits_tolerance):
    kwargs.update({"attn_implementation": "eager", "use_cache": True})
    config = config_class(**kwargs)
    hf_model = model_class(config=config)

    num_hidden_layers = kwargs.get(
        "num_hidden_layers", kwargs.get("n_layer", kwargs.get("n_layers"))
    )  # Not all configs have this params e.g. gpt2
    padding_shape = get_padding_shape_from_config(config=config, batch_size=1, seq_len=ctx_len)

    run_kv_cache_transform_and_test(
        hf_model,
        num_hidden_layers=num_hidden_layers,
        padding_shape=padding_shape,
        vocab_size=config.vocab_size,
        input_len=input_len,
        logits_tolerance=logits_tolerance,
    )

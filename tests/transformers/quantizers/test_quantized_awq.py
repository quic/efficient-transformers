import pytest
from transformers import AutoModelForCausalLM, AutoTokenizer

from QEfficient import QEFFAutoModelForCausalLM
from QEfficient.transformers.quantizers import replace_transformers_quantizers
from QEfficient.utils.constants import Constants
from QEfficient.utils.run_utils import ApiRunner

quant_models_test = ["casperhansen/qwen2-0.5b-instruct-awq"]


@pytest.mark.parametrize("model_card_name", quant_models_test)
def test_quant_models_load_and_run(model_card_name: str):
    replace_transformers_quantizers()
    model_hf = AutoModelForCausalLM.from_pretrained(
        model_card_name, attn_implementation="eager", low_cpu_mem_usage=False
    )
    tokenizer = AutoTokenizer.from_pretrained(model_card_name)
    api_runner = ApiRunner(
        1,
        tokenizer,
        model_hf.config,
        Constants.INPUT_STR,
        Constants.PROMPT_LEN,
        Constants.CTX_LEN,
    )

    pytorch_hf_tokens = api_runner.run_hf_model_on_pytorch(model_hf)
    
    qeff_model = QEFFAutoModelForCausalLM(model_hf, model_card_name)

    pytorch_kv_tokens = api_runner.run_kv_model_on_pytorch(qeff_model.model)
    
    assert (pytorch_hf_tokens == pytorch_kv_tokens).all()
    
    print("done")
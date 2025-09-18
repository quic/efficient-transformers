from transformers import AutoModelForCausalLM

from QEfficient import QEFFAutoModelForCausalLM
from QEfficient.utils.run_utils import ApiRunner
from QEfficient.utils._utils import create_json, load_hf_tokenizer
from QEfficient.transformers.quantizers.auto import replace_transformers_quantizers

from QEfficient.utils import hf_download

def load_causal_lm_model(model_config):
    """
    Function to load model from huggingface and transform to KV model
    --------

    :model_config: Dict

    :return model_hf, params
    """
    model_path = hf_download(
        repo_id=model_config["model_name"],
        ignore_patterns=["*.onnx", "*.ot", "*.md", "*.tflite", "*.pdf", "*.h5", "*.msgpack"],
    )
    model_hf = AutoModelForCausalLM.from_pretrained(
        model_path,
        use_cache=True,
        num_hidden_layers=model_config["n_layer"],
        attn_implementation="eager",
        low_cpu_mem_usage=False,
    )  # Run models for single layers only
    # params = sum(p.numel() for p in model_hf.parameters())
    params=""
    # model_hf.eval()
    return model_hf, params


def check_llama_onnx():
    MODEL_ID = "meta-llama/Llama-3.2-1B"
    # replace_transformers_quantizers()
    tokenizer = load_hf_tokenizer(pretrained_model_name_or_path=MODEL_ID)
    # config = AutoConfig.from_pretrained(MODEL_ID)
    # config.num_hidden_layers = 1
    model_config = {"model_name": MODEL_ID}
    model_config["n_layer"] = 2

    model_hf, _ = load_causal_lm_model(model_config)
    config = model_hf.config
    api_runner = ApiRunner(
        1,
        tokenizer,
        model_hf.config,
        "Where is the Thomas J. Watson Research Center located?",
        32, #prompt_len
        64, #ctx_len

    )
    pytorch_hf_tokens = api_runner.run_hf_model_on_pytorch(model_hf)
    qeff_model = QEFFAutoModelForCausalLM(model_hf)
    pytorch_kv_tokens = api_runner.run_kv_model_on_pytorch(qeff_model.model)
    onnx_model_path = qeff_model.export()
    ort_tokens = api_runner.run_kv_model_on_ort(onnx_model_path, is_tlm=False)

if __name__ == "__main__":
    # run()
    check_llama_onnx()
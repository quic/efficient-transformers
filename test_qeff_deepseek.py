from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
from QEfficient import QEFFAutoModelForCausalLM
from transformers import DeepseekV3ForCausalLM
from QEfficient.utils.run_utils import ApiRunner
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
    params = sum(p.numel() for p in model_hf.parameters())
    model_hf.eval()
    return model_hf, params


def check_deepseek():
    MODEL_ID = "unsloth/DeepSeek-V3-bf16"
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    config = AutoConfig.from_pretrained(MODEL_ID)
    config.num_hidden_layers = 4
    model_config = {"model_name": MODEL_ID}
    model_config["n_layer"] = 4

    model_hf, _ = load_causal_lm_model(model_config)
    api_runner = ApiRunner(
        1,
        tokenizer,
        model_hf.config,
        "Hello my name is",
        32, #prompt_len
        64, #ctx_len
        
    )
    pytorch_hf_tokens = api_runner.run_hf_model_on_pytorch(model_hf)
    # qeff_model = QEFFAutoModelForCausalLM.from_pretrained(MODEL_ID, num_hidden_layers=4)
    qeff_model = QEFFAutoModelForCausalLM(model_hf)
    pytorch_kv_tokens = api_runner.run_kv_model_on_pytorch(qeff_model.model)
    onnx_model_path = qeff_model.export()
    ort_tokens = api_runner.run_kv_model_on_ort(onnx_model_path, is_tlm=False)
    qpc_path = qeff_model.compile(
        prefill_seq_len=32,
        ctx_len=64,
        num_cores=16,
        mxfp6=False,
        aic_enable_depth_first=False,
    )
    exec_info = qeff_model.generate(tokenizer, prompts="Hello my name is")
    cloud_ai_100_tokens = exec_info.generated_ids[0]
    return

if __name__ == "__main__":
    check_deepseek()

import os

from transformers import AutoConfig, AutoTokenizer

from QEfficient.transformers.models.modeling_auto import QEFFAutoModelForCausalLM
from QEfficient.utils.run_utils import ApiRunner

# -----------------------------
# Patches to stabilize tracing
# -----------------------------
# import transformers.cache_utils as cache_utils
# _original_parse_processor_args = cache_utils.parse_processor_args
# def _patched_parse_processor_args(processor_class, kwargs):
#     if processor_class is None:
#         return {}, kwargs
#     try:
#         return _original_parse_processor_args(processor_class, kwargs)
#     except Exception:
#         return {}, kwargs
# cache_utils.parse_processor_args = _patched_parse_processor_args

# model_name = "meta-llama/Llama-3.2-1B"
model_name = "gpt2"
tokenizer = AutoTokenizer.from_pretrained(model_name)
config = AutoConfig.from_pretrained(model_name)
config.num_hidden_layers = 4
print(config)
runner = ApiRunner(
    batch_size=1,
    tokenizer=tokenizer,
    config=config,
    prompt=["My name is"],
    prompt_len=8,
    ctx_len=32,
)

# PyTorch (KV) output
qeff_model = QEFFAutoModelForCausalLM.from_pretrained(model_name, num_hidden_layers=4)
pt_tokens = runner.run_kv_model_on_pytorch(qeff_model.model)
print(pt_tokens)
# ONNXRuntime output with your ONNX path
onnx_path = qeff_model.export(use_dynamo=False, use_onnx_subfunctions=True)
print("export done")

dbg_dir = "acc_analyzer_inputs"
os.makedirs(dbg_dir, exist_ok=True)

ort_inputs = runner.input_handler.prepare_ort_inputs()

ort_tokens = runner.run_kv_model_on_ort(onnx_path)
print(ort_tokens)

qeff_model.compile(
    prefill_seq_len=8,
    ctx_len=32,
)
print("compile done")
print("QEff Transformed Onnx Model Outputs(AIC Backend)")
output = qeff_model.generate(prompts=["My name is"], tokenizer=tokenizer, automation=True)
print(output)
print(output.generated_ids)

from transformers import AutoTokenizer

from QEfficient import QEFFAutoModelForCausalLM

qeff_model = QEFFAutoModelForCausalLM.from_pretrained("zai-org/GLM-4.7")

qeff_model.compile(
    prefill_seq_len=1,
    num_devices=12,
    use_onnx_subfunctions=True,
    ctx_len=8192,
    mxfp6_matmul=True,
    mxint8_kv_cache=True,
    mos=1,
    aic_enable_depth_first=True,
    num_cores=16,
    offload_pt_weights=True,
)
tokenizer = AutoTokenizer.from_pretrained("zai-org/GLM-4.7")
qeff_model.generate(prompt=["Once upon a time,"], tokenizer=tokenizer)

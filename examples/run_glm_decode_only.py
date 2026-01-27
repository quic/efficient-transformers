import torch
from transformers import AutoTokenizer

from QEfficient import QEFFAutoModelForCausalLM

qeff_model = QEFFAutoModelForCausalLM.from_pretrained("zai-org/GLM-4.7", torch_dtype=torch.float16)

qeff_model.compile(
    prefill_seq_len=1,
    num_devices=16,
    use_onnx_subfunctions=True,
    ctx_len=4096,
    mxfp6_matmul=True,
    mxint8_kv_cache=False,
    mos=1,
    aic_enable_depth_first=True,
    num_cores=16,
    offload_pt_weights=True,
)
tokenizer = AutoTokenizer.from_pretrained("zai-org/GLM-4.7")
qeff_model.generate(prompts=["Once upon a time,"], tokenizer=tokenizer)

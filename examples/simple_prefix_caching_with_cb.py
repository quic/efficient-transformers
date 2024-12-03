from QEfficient import QEFFAutoModelForCausalLM

model_name = "meta-llama/Llama-3.1-8B-Instruct"

model = QEFFAutoModelForCausalLM.from_pretrained(
    model_name,
    #  num_hidden_layers=1,
    continuous_batching=True,
)

qpc_path = model.compile(
    prefill_seq_len=4096,
    ctx_len=4608,
    full_batch_size=8,
    cache_size_multiplier=4,
    num_devices=4,
    num_cores=14,
    mxfp6_matmul=True,
)
model.generate(prompts=["My name is"] * 32)
print(qpc_path)

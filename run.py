from QEfficient import QEFFAutoModelForCausalLM
from transformers import AutoTokenizer

model_name = "meta-llama/Llama-3.1-8B-Instruct"
model = QEFFAutoModelForCausalLM.from_pretrained(model_name, num_hidden_layers=2, continuous_batching=True)
# model = QEFFAutoModelForCausalLM.from_pretrained(model_name, num_hidden_layers=2)
CCL = [512, 1024, 2048, 4096, 8192, 16384, 32768]
model.compile(prefill_seq_len=128, ctx_len=32768, CCL=CCL, num_cores=16, num_devices=1, full_batch_size=4)
# model.compile(prefill_seq_len=128, ctx_len=32768, CCL=CCL, num_cores=16, num_devices=1,batch_size=4)

tokenizer = AutoTokenizer.from_pretrained(model_name)
model.generate(
    prompts=["Hi there!!", "Hi there!!", "Hi there!!", "Hi there!!"], tokenizer=tokenizer, CCL=CCL, generation_len=128
)

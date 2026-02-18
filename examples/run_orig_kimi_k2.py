import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained(
    "/home/huggingface_hub/models--moonshotai--Kimi-K2-Thinking/snapshots/612681931a8c906ddb349f8ad0f582cb552189cd",
    torch_dtype=torch.float32,
    num_hidden_layers=2,
    trust_remote_code=True,
)
tokenizer = AutoTokenizer.from_pretrained("moonshotai/Kimi-K2-Thinking", trust_remote_code=True)

prompt = "Once upon a time,"
inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
with torch.no_grad():
    outputs = model.generate(
        **inputs,
        max_new_tokens=10,
        do_sample=False,
        use_cache=False,
    )

response = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(response)

"""
Original Pytorch, kimi-k2 thinking:
Prompt: Once upon a time,
Completion : ?? branchesrupt??? flushedakislottery rehearsallesi
"""

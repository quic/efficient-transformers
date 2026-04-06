import torch
from transformers import AutoModelForImageTextToText, AutoProcessor

torch.manual_seed(42)

MODEL_ID = "tiny-random/gemma-4-dense"

# Load model
processor = AutoProcessor.from_pretrained(MODEL_ID)
model = AutoModelForImageTextToText.from_pretrained(
    MODEL_ID,
    dtype=torch.float16,
    device_map="auto",
)

# Prompt
messages = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "Hi, Who are you?"},
]

# Process input
text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
inputs = processor(text=text, return_tensors="pt").to(model.device)
input_len = inputs["input_ids"].shape[-1]

# Generate output
outputs = model.generate(**inputs, max_new_tokens=32)
response = processor.decode(outputs[0][input_len:], skip_special_tokens=False)

# Parse output
print(processor.parse_response(response))

import torch
import transformers
from transformers import AutoConfig, AutoProcessor, TextStreamer

from QEfficient import QEFFAutoModelForImageTextToText

model_id = "meta-llama/Llama-4-Scout-17B-16E-Instruct"
config = AutoConfig.from_pretrained(model_id)
# For Testing Purpose Only
config.text_config.num_hidden_layers = 4
config.vision_config.num_hidden_layers = 2

qeff_model = QEFFAutoModelForImageTextToText.from_pretrained(
    model_id,
    attn_implementation="eager",
    kv_offload=True,
    config=config,
    continuous_batching=True,
)
tokenizer = transformers.AutoTokenizer.from_pretrained(model_id)
processor = AutoProcessor.from_pretrained(model_id)

qeff_model.compile(
    prefill_seq_len=128,
    ctx_len=3072,
    img_size=336,
    num_cores=16,
    num_devices=4,
    max_num_tiles=17,
    batch_size=1,
    full_batch_size=4,
    mxfp6_matmul=True,
    mxint8_kv_cache=True,
    aic_enable_depth_first=True,
    mos=1,
)

image_url = (
        "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/datasets/cat_style_layout.png"
    )

prompts = [
    "Can you describe the image in detail?",
    # "What are the objects in the image?",
    # "What is the main subject of the image?",
    # "What colors are predominant in the image?",
]

all_inputs = []
for prompt in prompts:
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "url": image_url},
                {"type": "text", "text": prompt},
            ],
        },
    ]

    inputs = processor.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
    )
    inputs["pixel_values"] = inputs["pixel_values"].to(torch.float32)
    all_inputs.append(inputs)


output = qeff_model.generate(inputs=all_inputs[0], tokenizer=tokenizer, device_ids = [0,1,2,3], prompts=prompts, generation_len=100)

if hasattr(output, 'generated_texts'):
    for i, (prompt, response) in enumerate(zip(prompts, output.generated_texts)):
        print(f"Prompt {i+1}: {prompt}")
        print(f"Response {i+1}: {response}")
        print("-" * 30)
else:
    print("Generated IDs:", output.generated_ids)
    decoded_responses = tokenizer.batch_decode(output.generated_ids, skip_special_tokens=True)
    for i, (prompt, response) in enumerate(zip(prompts, decoded_responses)):
        print(f"Prompt {i+1}: {prompt}")
        print(f"Response {i+1}: {response}")
        print("-" * 30)

# print(output.generated_ids)
# print(tokenizer.batch_decode(output.generated_ids))
print(output)
print()

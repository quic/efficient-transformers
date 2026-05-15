import torch
from transformers import AutoTokenizer

from QEfficient import QEFFAutoModelForCausalLM



def duplicate_weights_for_linear_layer(
    layer: torch.nn.Module, orig_kv_heads: int, repeat: int, head_dim: int, hidden_size: int
):
    new_kv_heads = repeat * orig_kv_heads
    layer.weight.data = torch.repeat_interleave(
        layer.weight.data.view(orig_kv_heads, head_dim, hidden_size), repeat, 0
    ).view(new_kv_heads * head_dim, hidden_size)
    if layer.bias is not None:
        layer.bias.data = torch.repeat_interleave(layer.bias.data.view(orig_kv_heads, head_dim), repeat, 0).view(
            new_kv_heads * head_dim
        )


qeff_model = QEFFAutoModelForCausalLM.from_pretrained("/home/huggingface_hub/models--zai-org--GLM-4.7/snapshots/475d85cda16beac79cde7f4cf4cae8d1260566f5")

repeat=3
# Modify the number of key-value heads
orig_kv_heads = qeff_model.model.config.num_key_value_heads
new_kv_heads = repeat * orig_kv_heads
qeff_model.model.config.num_key_value_heads = new_kv_heads

print("Original KV heads:", orig_kv_heads)
print("Modified KV heads:", new_kv_heads)
num_attention_heads = qeff_model.model.config.num_attention_heads

hidden_size = qeff_model.model.config.hidden_size


# Update the model's attention layers with new key-value heads
for block in qeff_model.model.model.layers:
    attn = block.self_attn
    attn.num_key_value_heads = new_kv_heads
    attn.num_key_value_groups = num_attention_heads // new_kv_heads
    duplicate_weights_for_linear_layer(attn.k_proj, orig_kv_heads, repeat, attn.head_dim, hidden_size)
    duplicate_weights_for_linear_layer(attn.v_proj, orig_kv_heads, repeat, attn.head_dim, hidden_size)



import ipdb; ipdb.set_trace()
qeff_model.compile(
    prefill_seq_len=1,
    num_devices=12,
    use_onnx_subfunctions=True,
    ctx_len=8192,
    mxfp6_matmul=True,
    # mxint8_kv_cache=True,
    aic_enable_depth_first=True,
    num_cores=4,
    offload_pt_weights=True,
)
tokenizer = AutoTokenizer.from_pretrained("zai-org/GLM-4.7")
qeff_model.generate(prompt=["Once upon a time,"], tokenizer=tokenizer)

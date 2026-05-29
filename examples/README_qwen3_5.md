
## Installation steps ##

`pip install -e . inside efficient_transformers`
`pip uninstall transformers`
`pip install transformers==5.3.0`
`pip install peft==0.19.1`

## Example scripts ##

`examples/qwen3_5_example.py` -> single QPC (`kv_offload=False`)
`examples/image_text_to_text/models/qwen3_5/qwen3_5.py` -> dual QPC (`kv_offload=True`)
`examples/qwen3_5_moe_example_singleqpc.py` -> qwen3_5_moe single QPC

## To change the truncation order update /efficient_transformers/QEfficient/transformers/models/qwen3_5/modeling_qwen3_5.py:557 and then re-export 

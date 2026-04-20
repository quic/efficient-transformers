
## Installation steps ##

`pip install -e . inside efficient_transformers`
`pip uninstall transformers`
`pip install transformers==5.3.0`
`pip install peft==0.19.1`

## To change the truncation order update /efficient_transformers/QEfficient/transformers/models/qwen3_5/modeling_qwen3_5.py:557 and then re-export 
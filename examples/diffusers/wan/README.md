# Wan Lightning

## Prerequsites

Download Qefficient and install requirements

```bash
git clone https://github.com/quic/efficient-transformers.git

# create venv
python3.10 -m venv wan_env
source wan_env/bin/activate

cd efficient-transformers

# to install dependencies
pip install -e ".[diffusers]"

```
Export HF_HOME, QEFF_HOOME
```bash
export HF_HOME=/home/HF_HOME/
export QEFF_HOME=/home/qeff_home/
```

## To run Wan Lightning

```bash
# To run with HQKV blocking
ATTENTION_BLOCKING_MODE=qkv head_block_size=16 num_kv_blocks=16 num_q_blocks=2  python -W ignore examples/diffusers/wan/wan_lightning.py
```


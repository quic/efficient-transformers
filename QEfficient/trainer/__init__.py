from QEfficient.trainer.config import QEffConfig, QEffTrainingArguments
from QEfficient.trainer.models import QEfficient

# python -m QEfficient.refine --model_id facebook/opt-125m --dataset_name xiyuez/red-dot-design-award-product-description --train_frac 0.85 --lora_r 8 --max_ctx_len 512 --num_train_epochs 3 --per_device_train_batch_size 8 --output_dir ./output

__all__ = [
    "QEffConfig",
    "QEffTrainingArguments",
    "QEfficient",
]

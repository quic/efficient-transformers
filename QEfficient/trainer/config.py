from dataclasses import dataclass, field
from typing import Any, Dict


@dataclass
class QEffConfig:
    model_id: str = field(default="facebook/opt-125m", metadata={"help": "Pre-trained model to fine-tune"})
    dataset_name: str = field(
        default="xiyuez/red-dot-design-award-product-description", metadata={"help": "Dataset to fine-tune on"}
    )
    train_frac: float = field(default=0.85, metadata={"help": "Fraction of training dataset to split"})
    lora_config: Dict[str, Any] = field(
        default_factory=lambda: {"r": 8, "target_modules": ["q_proj", "k_proj", "v_proj", "out_proj", "fc1", "fc2"]}
    )
    max_ctx_len: int = field(default=512, metadata={"help": "Maximum context length for tokenization"})


@dataclass
class QEffTrainingArguments:
    output_dir: str = field(default="./output")
    num_train_epochs: int = field(default=3)
    per_device_train_batch_size: int = field(default=8)
    per_device_eval_batch_size: int = field(default=8)
    warmup_steps: int = field(default=500)
    weight_decay: float = field(default=0.01)
    logging_dir: str = field(default="./logs")

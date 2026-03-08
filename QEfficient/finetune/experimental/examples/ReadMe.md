
# Custom Dataset Example

This example demonstrates how to register a custom dataset type with the fine-tuning framework
by mirroring the structure of the built-in `SFTDataset`.

---

## Files to Create


```text
examples/
├── custom_dataset.py       # Custom dataset class
├── example_config.yaml     # Training configuration
└── example_finetune.py     # Entry point
```

---

## 1. `custom_dataset.py`

Create your dataset class by subclassing `BaseDataset` and registering it with the component
registry using the `@registry.dataset(<name>)` decorator.

The SeqCompletionDataset class in custom_dataset.py mirrors `SFTDataset` in structure.
---

## 2. `config.yaml`

```yaml
dataset:
  dataset_type: "my_custom_dataset"    # Must match @registry.dataset(<name>)
  dataset_name: "Salesforce/wikitext"
  config_name: "wikitext-103-raw-v1"
  prompt_template: "{text}"        
  train_split: "train"
  test_split: "test"
  seed: 42
  dataset_num_samples: 100
```

**dataset_type must exactly match the name passed to `@registry.dataset(...)` in your custom dataset file.**

---

## 3. `finetune_custom.py`

```python
from QEfficient.finetune.experimental.examples.custom_dataset import CustomDataset  # noqa: F401
from QEfficient.cloud.finetune_experimental import main

if __name__ == "__main__":
    main()
```


---

## Run

```bash
python examples/example_finetune.py examples/example_config.yaml
```

# Model Weight Loading in bfloat16/float16 for AIC Deployment

## Loading Model Weights with Specific Precision
To load model weights directly in `bfloat16` or `float16` precision, use the `torch_dtype` parameter in `from_pretrained()`:

### The available `torch_dtype` can be:
- `torch.bfloat16`
- `torch.float16`
- `torch.float32`

## Example Script `test_bf16_export.py` can be used to export and compile Llama-3.2-1B model in bfloat16/float16 format.
```bash
python test_bf16_export.py --model_name meta-llama/Llama-3.2-1B --precision bf16 --n_layer 1 --output_dir ./exported_model
```
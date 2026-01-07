import argparse

import torch
from transformers import AutoConfig, AutoTokenizer

from QEfficient import QEFFAutoModelForCausalLM


def parse_args():
    parser = argparse.ArgumentParser(description="Export and compile model in bfloat16/float16 format")
    parser.add_argument(
        "--model_name",
        type=str,
        default="meta-llama/Llama-3.2-1B",
        help="Model name or path (default: meta-llama/Llama-3.2-1B)",
    )
    parser.add_argument(
        "--precision",
        type=str,
        choices=["bf16", "fp16", "fp32"],
        default="bf16",
        help="Precision for model weights: bf16 (bfloat16), fp16 (float16), or fp32 (float32) (default: bf16)",
    )
    parser.add_argument("--output_dir", type=str, default=None, help="Output directory for exported model (optional)")
    parser.add_argument("--n_layer", type=int, default=1, help="Number of hidden layers to use (default: 1)")
    parser.add_argument(
        "--prompt", type=str, default="Hi, my name is", help="Prompt for text generation (default: 'Hi, my name is')"
    )
    parser.add_argument("--generation_len", type=int, default=20, help="Length of generated text (default: 20)")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    # Map precision string to torch dtype
    precision_map = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}
    LOAD_DTYPE = precision_map[args.precision]

    # Model config for export
    MODEL_ID = args.model_name
    PROMPT = args.prompt
    N_LAYER = args.n_layer

    # Load Config first to load custom layered model
    config = AutoConfig.from_pretrained(MODEL_ID, trust_remote_code=True)
    config.num_hidden_layers = N_LAYER

    # Load the Model with required Torch Dtype
    qeff_model = QEFFAutoModelForCausalLM.from_pretrained(
        pretrained_model_name_or_path=MODEL_ID, torch_dtype=LOAD_DTYPE, config=config, trust_remote_code=True
    )

    # Export & Compile the model
    onnx_model_path = qeff_model.export()

    qpc_path = qeff_model.compile(
        prefill_seq_len=128,
        ctx_len=512,
        num_cores=16,
        num_devices=1,
        mxfp6=False,
        aic_enable_depth_first=False,
        num_speculative_tokens=None,
    )

    print("\nModel exported and compiled successfully!")
    print(f"Model: {MODEL_ID}")
    print(f"Precision: {args.precision} ({LOAD_DTYPE})")
    print(f"Number of layers: {N_LAYER}")
    if args.output_dir:
        print(f"Output directory: {args.output_dir}")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    exec_info = qeff_model.generate(tokenizer, prompts=PROMPT, generation_len=args.generation_len)

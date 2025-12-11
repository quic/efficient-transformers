# -----------------------------------------------------------------------------
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------
import argparse
import re
from pprint import pprint

import numpy as np

from QEfficient import QEFFAutoModelForCausalLM as AutoModelForCausalLM
from QEfficient.utils import load_hf_tokenizer


def main(args, **kwargs):
    pprint(args.__dict__)

    # Get sampling inputs
    include_sampler = None
    return_pdfs = None
    max_top_k_ids = None
    include_guided_decoding = None
    sampling_params = None
    bs = args.full_batch_size if args.full_batch_size is not None else args.batch_size
    if args.override_qaic_config is not None:
        include_sampler = args.override_qaic_config.get("aic_include_sampler", None) == "true"
        if include_sampler is not None:
            return_pdfs = args.override_qaic_config.get("aic_return_pdfs", None) == "true"
            max_top_k_ids = int(args.override_qaic_config.get("max_top_k_ids", 512))
            np.random.seed(int(args.random_number))
            include_guided_decoding = args.override_qaic_config.get("aic_include_guided_decoding", None) == "true"
            sampling_params = {
                "repetition_penalties": np.array(args.repetition_penalty, dtype=np.float32).repeat(bs).reshape(-1, 1),
                "presence_penalties": np.array(args.presence_penalty, dtype=np.float32).repeat(bs).reshape(-1, 1),
                # "frequency_penalties": np.array(args.frequency_penalty, dtype=np.float32).repeat(bs).reshape(-1, 1),
                "temperatures": np.array(args.temperature, dtype=np.float32).repeat(bs).reshape(-1, 1),
                "top_ks": np.array(args.top_k, dtype=np.int32).repeat(bs).reshape(-1, 1),
                "top_ps": np.array(args.top_p, dtype=np.float32).repeat(bs).reshape(-1, 1),
                "min_ps": np.array(args.min_p, dtype=np.float32).repeat(bs).reshape(-1, 1),
                "random_numbers": np.tile(np.random.uniform(low=0.0, high=1.0, size=max_top_k_ids), (bs, 1)).astype(
                    np.float32
                ),
            }
    qaic_config = {
        k: v
        for k, v in {
            "include_sampler": include_sampler,
            "return_pdfs": return_pdfs,
            "max_top_k_ids": max_top_k_ids,
            "include_guided_decoding": include_guided_decoding,
        }.items()
        if v is not None
    }
    print("qaic_config:")
    pprint(qaic_config)

    # Load model with On Device Sampler enabled
    qeff_model = AutoModelForCausalLM.from_pretrained(
        pretrained_model_name_or_path=args.model_name,
        continuous_batching=args.full_batch_size is not None,
        qaic_config=qaic_config,
    )
    print(f"{args.model_name} optimized for AI 100 \n", qeff_model)

    if include_guided_decoding:
        # Ideally this should come from a logits processor like xgrammar, but for the sake of the
        # example, we generate a random bitmask
        sampling_params.update(
            {
                "token_bitmasks": np.tile(
                    np.random.choice([True, False], size=(qeff_model.model.config.vocab_size,)), (bs, 1)
                )
            }
        )
    print("sampling_params:")
    pprint(sampling_params)

    # Compile the model for inference
    generated_qpc_path = qeff_model.compile(
        prefill_seq_len=args.prompt_len,
        ctx_len=args.ctx_len,
        batch_size=args.batch_size,
        full_batch_size=args.full_batch_size,
        num_cores=args.num_cores,
        num_devices=(0 if args.device_group is None else len(args.device_group)),
        mxfp6_matmul=args.mxfp6,
        mxint8_kv_cache=args.mxint8,
        num_speculative_tokens=0,
        **kwargs,
    )
    print(f"Generated QPC file path: {generated_qpc_path}")

    # Generate texts from prompts
    if not args.prompt:
        args.prompt = [
            "Hi",
        ] * bs
    qeff_model.generate(
        tokenizer=load_hf_tokenizer(pretrained_model_name_or_path=args.model_name),
        prompts=args.prompt,
        prompts_txt_file_path=args.prompts_txt_file_path,
        device_id=args.device_group,
        generation_len=args.generation_len,
        include_sampler=include_sampler,
        return_pdfs=return_pdfs,
        include_guided_decoding=include_guided_decoding,
        sampling_params=sampling_params,
    )


if __name__ == "__main__":
    """
    Example usage:
    1. For continuous batching:
        python3.10 examples/on_device_sampling.py \
            --model-name 'meta-llama/Llama-3.1-8B' \
            --prompt-len 128 \
            --ctx-len 256 \
            --generation-len 20 \
            --full-batch-size 2 \
            --device-group [0,1,2,3] \
            --num-cores 16 \
            --mxint8-kv-cache \
            --mxfp6-matmul \
            --override-qaic-config "aic_include_sampler:true aic_return_pdfs:false max_top_k_ids:512 aic_include_guided_decoding:false" \
            --repetition-penalty 1.9 \
            --presence-penalty 0.8 \
            --temperature 0.67 \
            --top-k 54 \
            --top-p 0.89 \
            --min-p 0.6 \
            --random-number 26

    2. For non-continuous batching:
        python3.10 examples/on_device_sampling.py \
            --model-name 'meta-llama/Llama-3.1-8B' \
            --prompt-len 128 \
            --ctx-len 256 \
            --generation-len 20 \
            --batch-size 2 \
            --device-group [0,1,2,3] \
            --num-cores 16 \
            --mxint8-kv-cache \
            --mxfp6-matmul \
            --override-qaic-config "aic_include_sampler:true aic_return_pdfs:false max_top_k_ids:512 aic_include_guided_decoding:false" \
            --repetition-penalty 1.9 \
            --presence-penalty 0.8 \
            --temperature 0.67 \
            --top-k 54 \
            --top-p 0.89 \
            --min-p 0.6 \
            --random-number 26

    3. With guided decoding:
        python3.10 examples/on_device_sampling.py \
            --model-name 'meta-llama/Llama-3.1-8B' \
            --prompt-len 128 \
            --ctx-len 256 \
            --generation-len 20 \
            --full-batch-size 2 \
            --device-group [0,1,2,3] \
            --num-cores 16 \
            --mxint8-kv-cache \
            --mxfp6-matmul \
            --override-qaic-config "aic_include_sampler:true aic_return_pdfs:false max_top_k_ids:512 aic_include_guided_decoding:true" \
            --repetition-penalty 1.9 \
            --presence-penalty 0.8 \
            --temperature 0.67 \
            --top-k 54 \
            --top-p 0.89 \
            --min-p 0.6 \
            --random-number 26
    """

    parser = argparse.ArgumentParser(description="Run QEfficient model with On Device Sampling")
    parser.add_argument(
        "--model-name", "--model_name", required=True, default="meta-llama/Llama-3.1-8B", help="HF Model card name/id"
    )
    parser.add_argument("--batch-size", "--batch_size", type=int, default=1, help="Batch size for text generation")
    parser.add_argument(
        "--prompt-len", "--prompt_len", default=32, type=int, help="Sequence length for text generation."
    )
    parser.add_argument("--ctx-len", "--ctx_len", default=128, type=int, help="Context length for text generation.")
    parser.add_argument(
        "--mxfp6",
        "--mxfp6_matmul",
        "--mxfp6-matmul",
        action="store_true",
        help="Compress constant MatMul weights to MXFP6 E2M3, default is no compression",
    )
    parser.add_argument(
        "--mxint8",
        "--mxint8_kv_cache",
        "--mxint8-kv-cache",
        action="store_true",
        help="Compress Present/Past KV to MXINT8 using CustomIO config, default is False",
    )
    parser.add_argument(
        "--num_cores", "--num-cores", type=int, required=True, help="Number of cores to compile on Cloud AI 100"
    )
    parser.add_argument(
        "--device_group",
        "--device-group",
        type=lambda device_ids: [int(x) for x in device_ids.strip("[]").split(",")],
        help="Cloud AI 100 device ids (comma-separated) e.g. [0,1]",
    )
    parser.add_argument(
        "--prompt",
        type=lambda prompt: prompt.split("|"),
        help="Input prompt, if executing for batch size>1, pass input prompts in single string but separate with pipe (|) symbol",
    )
    parser.add_argument(
        "--prompts_txt_file_path",
        "--prompts-txt-file-path",
        type=str,
        help="File path for taking input prompts from txt file, sample prompts.txt file present in examples/sample_prompts folder",
    )
    parser.add_argument("--generation_len", "--generation-len", type=int, help="Number of tokens to generate")

    parser.add_argument(
        "--full_batch_size",
        "--full-batch-size",
        type=int,
        default=None,
        help="Set full batch size to enable continuous batching mode, default is None",
    )
    parser.add_argument(
        "--override-qaic-config",
        type=lambda configs: {
            str(value[0]): value[1] if len(value) > 1 else True
            for value in (re.split(r"[:=]", config.strip()) for config in re.split(r"[ ]+", configs.strip()))
        },
        default=None,
        help="override or set qaic device configuration.",
    )

    # ---On Device Sampling---
    sampling_group = parser.add_argument_group("Sampling parameters")
    sampling_group.add_argument(
        "--repetition-penalty",
        type=float,
        default=None,
        help="Sampling parameter that penalizes new tokens based on whether they appear in the "
        "prompt and the generated text so far. Values > 1 encourage the model to use new tokens, "
        "while values < 1 encourage the model to repeat tokens.",
    )
    sampling_group.add_argument(
        "--presence-penalty",
        type=float,
        default=None,
        help="Sampling parameter that penalizes new tokens based on whether they appear in the "
        "generated text so far. Values > 0 encourage the model to use new tokens, while values < "
        "0 encourage the model to repeat tokens.",
    )
    sampling_group.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Sampling parameter that controls the randomness of the sampling. Lower"
        "values make the model more deterministic, while higher values make"
        "the model more random. Zero means greedy sampling.",
    )
    sampling_group.add_argument(
        "--top-k",
        type=int,
        default=None,
        help="Sampling parameter that controls the number of top tokens to consider. Set to -1 to consider all tokens.",
    )
    sampling_group.add_argument(
        "--top-p",
        type=float,
        default=None,
        help="Sampling parameter that controls the cumulative probability of the top tokens to "
        "consider. Must be in (0, 1]. Set to 1.0 to consider all tokens.",
    )
    sampling_group.add_argument(
        "--min-p",
        type=float,
        default=None,
        help="Sampling parameter that represents the minumum probability for a token to be "
        "considered, relative to the probability of the most likely token. Must be in [0, 1]. "
        "Set to 0.0 to disable this.",
    )
    sampling_group.add_argument(
        "--random-number",
        type=float,
        default=None,
        help="Sampling parameter that represents the random seed to use for random sampling. Must be in [-1, 1].",
    )
    args, compiler_options = parser.parse_known_args()

    compiler_options_dict = {}
    for i in range(0, len(compiler_options)):
        if compiler_options[i].startswith("--"):
            key = compiler_options[i].lstrip("-").replace("-", "_")
            value = (
                compiler_options[i + 1]
                if i + 1 < len(compiler_options) and not compiler_options[i + 1].startswith("-")
                else True
            )
            compiler_options_dict[key] = value

    main(args, **compiler_options_dict)

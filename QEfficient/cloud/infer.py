# -----------------------------------------------------------------------------
#
# Copyright (c) 2024 Qualcomm Innovation Center, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------

import argparse
import logging
import sys
from typing import List, Optional

from QEfficient.base.common import QEFFCommonLoader
from QEfficient.utils import check_and_assign_cache_dir, load_hf_tokenizer
from QEfficient.utils.logging_utils import logger


def main(
    model_name: str,
    num_cores: int,
    device_group: Optional[List[int]] = None,
    prompt: Optional[str] = None,  # type: ignore
    prompts_txt_file_path: Optional[str] = None,
    aic_enable_depth_first: bool = False,
    mos: int = -1,
    batch_size: int = 1,
    full_batch_size: Optional[int] = None,
    prompt_len: int = 32,
    ctx_len: int = 128,
    generation_len: Optional[int] = None,
    mxfp6: bool = False,
    mxint8: bool = False,
    local_model_dir: Optional[str] = None,
    cache_dir: Optional[str] = None,
    hf_token: Optional[str] = None,
    allow_mxint8_mdp_io: bool = False,
    enable_qnn: Optional[bool] = False,
    qnn_config: Optional[str] = None,
    **kwargs,
) -> None:
    """
    1. Check if compiled qpc for given config already exists, if it does jump to execute, else
    2. Check if exported ONNX file already exists, if true, jump to compilation -> execution, else
    3. Check if HF model exists in cache, if true, start transform -> export -> compilation -> execution, else,
    4. Download HF model -> transform -> export -> compile -> execute
    ``Mandatory`` Args:
        :model_name (str): Hugging Face Model Card name, Example: ``gpt2``
        :num_cores (int): Number of cores to compile model on.
    ``Optional`` Args:
        :device_group (List[int]): Device Ids to be used for compilation. If ``len(device_group) > 1``, multiple Card setup is enabled. ``Defaults to None.``
        :prompt (str): Sample prompt for the model text generation. ``Defaults to None.``
        :prompts_txt_file_path (str): Path to txt file for multiple input prompts. ``Defaults to None.``
        :aic_enable_depth_first (bool): Enables ``DFS`` with default memory size. ``Defaults to False.``
        :mos (int): Effort level to reduce the on-chip memory. ``Defaults to -1.``
        :batch_size (int): Batch size to compile the model for. ``Defaults to 1.``
        :full_batch_size (int): Set full batch size to enable continuous batching mode. ``Default to None``
        :prompt_len (int): Prompt length for the model to compile. ``Defaults to 32.``
        :ctx_len (int): Maximum context length to compile the model. ``Defaults to 128.``
        :generation_len (int): Number of tokens to be generated. ``Defaults to False.``
        :mxfp6 (bool): Enable compilation for MXFP6 precision. ``Defaults to False.``
        :mxint8 (bool): Compress Present/Past KV to ``MXINT8`` using ``CustomIO`` config. ``Defaults to False.``
        :local_model_dir (str): Path to custom model weights and config files. ``Defaults to None.``
        :cache_dir (str): Cache dir where downloaded HuggingFace files are stored. ``Defaults to None.``
        :hf_token (str): HuggingFace login token to access private repos. ``Defaults to None.``
        :allow_mxint8_mdp_io (bool): Allows MXINT8 compression of MDP IO traffic. ``Defaults to False.``
        :enable_qnn (bool): Enables QNN Compilation. ``Defaults to False.``
        :qnn_config (str): Path of QNN Config parameters file. ``Defaults to None.``

    .. code-block:: bash

        python -m QEfficient.cloud.infer OPTIONS

    """
    cache_dir = check_and_assign_cache_dir(local_model_dir, cache_dir)
    tokenizer = load_hf_tokenizer(
        pretrained_model_name_or_path=(local_model_dir if local_model_dir else model_name),
        cache_dir=cache_dir,
        hf_token=hf_token,
    )

    if "--mxfp6" in sys.argv:
        if args.mxfp6:
            logger.warning("mxfp6 is going to be deprecated in a future release, use -mxfp6_matmul instead.")
    if "--mxint8" in sys.argv:
        if args.mxint8:
            logger.warning("mxint8 is going to be deprecated in a future release, use -mxint8_kv_cache instead.")

    qeff_model = QEFFCommonLoader.from_pretrained(
        pretrained_model_name_or_path=model_name,
        cache_dir=cache_dir,
        hf_token=hf_token,
        full_batch_size=full_batch_size,
        local_model_dir=local_model_dir,
    )

    #########
    # Compile
    #########
    _ = qeff_model.compile(
        prefill_seq_len=prompt_len,
        ctx_len=ctx_len,
        num_cores=num_cores,
        mxfp6_matmul=mxfp6,
        aic_enable_depth_first=aic_enable_depth_first,
        batch_size=batch_size,
        mos=mos,
        mxint8_kv_cache=mxint8,
        num_devices=(0 if device_group is None else len(device_group)),
        full_batch_size=full_batch_size,
        allow_mxint8_mdp_io=allow_mxint8_mdp_io,
        enable_qnn=enable_qnn,
        qnn_config=qnn_config,
        **kwargs,
    )

    #########
    # Execute
    #########
    _ = qeff_model.generate(
        tokenizer,
        prompts=prompt,
        device_id=device_group,
        prompt=prompt,
        prompts_txt_file_path=prompts_txt_file_path,
        generation_len=generation_len,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Inference command, the model will be downloaded from HF, optimized, compiled, executed on Cloud AI 100"
    )
    parser.add_argument("--model-name", "--model_name", required=True, help="HF Model card name/id")
    parser.add_argument(
        "--local-model-dir", "--local_model_dir", required=False, help="Path to custom model weights and config files"
    )
    parser.add_argument(
        "--cache-dir",
        "--cache_dir",
        default=None,
        required=False,
        help="Cache dir to store HF Downloads",
    )
    parser.add_argument(
        "--hf-token", "--hf_token", default=None, type=str, required=False, help="HF token id for private HF models"
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
        help="Cloud AI 100 device ids (comma-separated) e.g. [0,1]  ",
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
        help="File path for taking input prompts from txt file, sample prompts.txt file present in examples folder",
    )
    parser.add_argument("--generation_len", "--generation-len", type=int, help="Number of tokens to generate")
    parser.add_argument(
        "--aic_enable_depth_first",
        "--aic-enable-depth-first",
        action="store_true",
        help="If passed, this option will be enabled during compilation, disabled by default",
    )
    parser.add_argument(
        "--mos",
        type=int,
        default=-1,
        help="Effort level to reduce the on-chip memory",
    )
    # FIXME: Add verbose feature
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="pass to print info logs",
    )
    parser.add_argument(
        "--full_batch_size",
        "--full-batch-size",
        type=int,
        default=None,
        help="Set full batch size to enable continuous batching mode, default is None",
    )
    parser.add_argument(
        "--allow-mxint8-mdp-io",
        "--allow_mxint8_mdp_io",
        action="store_true",
        help="If passed, this option allows MXINT8 compression of MDP IO traffic",
    )
    parser.add_argument(
        "--enable_qnn",
        "--enable-qnn",
        action="store_true",
        default=False,
        help="Enables QNN. Optionally, a configuration file can be provided with [--enable_qnn CONFIG_FILE].\
             If not provided, the default configuration will be used.\
             Sample Config: QEfficient/cloud/compile/qnn_config.json",
    )
    parser.add_argument(
        "qnn_config",
        nargs="?",
        type=str,
    )

    args, compiler_options = parser.parse_known_args()
    compiler_options_dict = {}
    for i in range(0, len(compiler_options)):
        if compiler_options[i].startswith("--"):
            key = compiler_options[i].lstrip("-")
            value = (
                compiler_options[i + 1]
                if i + 1 < len(compiler_options) and not compiler_options[i + 1].startswith("-")
                else True
            )
            compiler_options_dict[key] = value
    if args.verbose:
        logger.setLevel(logging.INFO)
    del args.verbose  # type: ignore
    main(**args.__dict__, **compiler_options_dict)

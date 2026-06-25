# -----------------------------------------------------------------------------
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------

import argparse
import json
from typing import Optional

import torch
from datasets import Features, Sequence, Value, load_dataset
from huggingface_hub import hf_hub_download
from safetensors import safe_open

from QEfficient.utils.logging_utils import logger

# ─────────────────────────────────────────────────────────────────────────────
# model_name (TLM short)  →  (TLM HF repo, DLM HF repo)
# DLM column is the canonical DFlash repo. TLM column is the standard HF repo
# when known; otherwise None and must be supplied via --tlm_hf_path.
# ─────────────────────────────────────────────────────────────────────────────
MODEL_MAP = {
    "gemma-4-31B-it": (None, "z-lab/gemma-4-31B-it-DFlash"),
    "gemma-4-26B-A4B-it": (None, "z-lab/gemma-4-26B-A4B-it-DFlash"),
    "MiniMax-M2.7": (None, "z-lab/MiniMax-M2.7-DFlash"),
    "MiniMax-M2.5": (None, "z-lab/MiniMax-M2.5-DFlash"),
    "Kimi-K2.6": (None, "z-lab/Kimi-K2.6-DFlash"),
    "Kimi-K2.5": (None, "z-lab/Kimi-K2.5-DFlash"),
    "Qwen3.6-27B": (None, "z-lab/Qwen3.6-27B-DFlash"),
    "Qwen3.6-35B-A3B": (None, "z-lab/Qwen3.6-35B-A3B-DFlash"),
    "Qwen3.5-4B": (None, "z-lab/Qwen3.5-4B-DFlash"),
    "Qwen3.5-9B": (None, "z-lab/Qwen3.5-9B-DFlash"),
    "Qwen3.5-27B": (None, "z-lab/Qwen3.5-27B-DFlash"),
    "Qwen3.5-35B-A3B": (None, "z-lab/Qwen3.5-35B-A3B-DFlash"),
    "Qwen3.5-122B-A10B": (None, "z-lab/Qwen3.5-122B-A10B-DFlash"),
    "gpt-oss-20b": ("openai/gpt-oss-20b", "z-lab/gpt-oss-20b-DFlash"),
    "gpt-oss-120b": ("openai/gpt-oss-120b", "z-lab/gpt-oss-120b-DFlash"),
    "Qwen3-Coder-Next": (None, "z-lab/Qwen3-Coder-Next-DFlash"),
    "Qwen3-4B": ("Qwen/Qwen3-4B", "z-lab/Qwen3-4B-DFlash-b16"),
    "Qwen3-8B": ("Qwen/Qwen3-8B", "z-lab/Qwen3-8B-DFlash-b16"),
    "Qwen3-Coder-30B-A3B": ("Qwen/Qwen3-Coder-30B-A3B-Instruct", "z-lab/Qwen3-Coder-30B-A3B-DFlash"),
    "Llama-3.1-8B-Instruct": ("meta-llama/Llama-3.1-8B-Instruct", "z-lab/LLaMA3.1-8B-Instruct-DFlash-UltraChat"),
}


def _build_aliases(model_map):
    aliases = {}
    for short, (tlm_repo, _) in model_map.items():
        aliases[short.lower()] = short
        if tlm_repo:
            aliases[tlm_repo.lower()] = short
            aliases[tlm_repo.split("/", 1)[-1].lower()] = short
    return aliases


MODEL_ALIASES = _build_aliases(MODEL_MAP)


def resolve_model_name(name: str) -> str:
    """Map a user-supplied model name (short, full HF path, or basename) to
    the canonical short name used as a key in MODEL_MAP."""
    canonical = MODEL_ALIASES.get(name.lower())
    if canonical is None:
        raise argparse.ArgumentTypeError(
            f"unknown model_name '{name}'. Supported: " + ", ".join(sorted(MODEL_MAP.keys()))
        )
    return canonical


def build_target_layer_ids(num_target_layers: int, num_draft_layers: int):
    if num_draft_layers == 1:
        return [(num_target_layers // 2)]
    start = 1
    end = num_target_layers - 3
    span = end - start
    target_layer_ids = [int(round(start + (i * span) / (num_draft_layers - 1))) for i in range(num_draft_layers)]
    return target_layer_ids


def extract_context_feature(
    hidden_states: list[torch.Tensor],
    layer_ids: Optional[list[int]],
) -> torch.Tensor:
    offset = 1
    selected_states = []
    for layer_id in layer_ids:
        selected_states.append(hidden_states[layer_id + offset])
    target_hidden = torch.cat(selected_states, dim=-1)
    return target_hidden


def sample(logits: torch.Tensor, temperature: float = 0.0) -> torch.Tensor:
    if temperature < 1e-5:
        return torch.argmax(logits, dim=-1)
    bsz, seq_len, vocab_size = logits.shape
    logits = logits.view(-1, vocab_size)
    logits = logits / temperature
    probs = torch.softmax(logits, dim=-1)
    return torch.multinomial(probs, num_samples=1).view(bsz, seq_len)


def load_dflash_checkpoint(dflash_model_path: str) -> tuple[dict, dict]:
    """Download and load the DFlash safetensors checkpoint and config.

    Returns
    -------
    state_dict : dict[str, Tensor]  — all tensors in fp32
    cfg        : dict               — parsed config.json
    """
    bin_path = hf_hub_download(repo_id=dflash_model_path, filename="model.safetensors")
    config_path = hf_hub_download(repo_id=dflash_model_path, filename="config.json")

    with open(config_path, "r") as f:
        cfg = json.load(f)

    state_dict = {}
    with safe_open(bin_path, framework="pt", device="cpu") as f:
        for key in f.keys():
            state_dict[key] = f.get_tensor(key).to(torch.float32)

    return state_dict, cfg


def read_dlm_meta(dlm_repo: str, hf_token: Optional[str] = None):
    """Load a DFlash checkpoint and return (state_dict, target_layer_ids, block_size)."""
    state_dict, cfg = load_dflash_checkpoint(dlm_repo)
    target_layer_ids = cfg.get("dflash_config", {}).get("target_layer_ids", [])
    block_size = cfg.get("block_size", None)
    return state_dict, target_layer_ids, block_size


def compile_tlm_qpc(
    tlm_repo: str,
    dlm_repo: str,
    *,
    prefill_seq_len: int,
    ctx_len: int,
    num_cores: int,
    num_devices: int,
    hf_token: Optional[str] = None,
) -> str:
    """Build the TLM and compile it to a QPC. fc/hidden_norm are injected by
    QEfficient's DFlashTLMTransform (weights from dflash_dlm_repo)."""
    from QEfficient import QEFFAutoModelForCausalLM

    _, target_layer_ids, block_size = read_dlm_meta(dlm_repo, hf_token)
    tlm_target_ids = [i + 1 for i in target_layer_ids]

    logger.info(f"[compile_tlm] base={tlm_repo}  dlm={dlm_repo}  block_size={block_size}")
    tlm_qeff = QEFFAutoModelForCausalLM.from_pretrained(
        tlm_repo,
        torch_dtype=torch.float32,
        token=hf_token,
        qaic_config={"target_layer_ids": tlm_target_ids, "dflash_dlm_repo": dlm_repo},
    )
    qpc = tlm_qeff.compile(
        prefill_seq_len=prefill_seq_len,
        ctx_len=ctx_len,
        num_cores=num_cores,
        num_devices=num_devices,
        mxfp6_matmul=True,
        mxint8_kv_cache=True,
        mos=1,
        dflash_block_size=block_size,
    )
    qpc = str(qpc)
    logger.info(f"[compile_tlm] qpc={qpc}")
    return qpc


def compile_dlm_qpc(
    tlm_repo: str,
    dlm_repo: str,
    *,
    ctx_len: int,
    num_cores: int,
    num_devices: int,
    hf_token: Optional[str] = None,
) -> str:
    """Build the DLM and compile it to a QPC. lm_head/embed_tokens are injected
    by QEfficient's DFlashDLMTransform (weights from dflash_tlm_repo)."""
    from QEfficient import QEFFAutoModelForCausalLM

    _, _, block_size = read_dlm_meta(dlm_repo, hf_token)

    logger.info(f"[compile_dlm] dlm={dlm_repo}  block_size={block_size}")
    dlm_qeff = QEFFAutoModelForCausalLM.from_pretrained(
        dlm_repo,
        torch_dtype=torch.float32,
        token=hf_token,
        qaic_config={"dflash_dlm": True, "dflash_tlm_repo": tlm_repo},
    )
    qpc = dlm_qeff.compile(
        prefill_seq_len=block_size,
        ctx_len=ctx_len,
        num_cores=num_cores,
        num_devices=num_devices,
        mxfp6_matmul=True,
        mxint8_kv_cache=True,
        mos=1,
        prefill_only=True,
    )
    qpc = str(qpc)
    logger.info(f"[compile_dlm] qpc={qpc}")
    return qpc


def load_and_process_dataset(data_name: str):
    # Math datasets
    if data_name == "gsm8k":
        dataset = load_dataset("openai/gsm8k", "main", split="test")
        prompt_fmt = "{question}\nPlease reason step by step, and put your final answer within \\boxed{{}}."
        dataset = dataset.map(lambda x: {"turns": [prompt_fmt.format(**x)]})

    elif data_name == "math500":
        dataset = load_dataset("HuggingFaceH4/MATH-500", split="test")
        prompt_fmt = "{problem}\nPlease reason step by step, and put your final answer within \\boxed{{}}."
        dataset = dataset.map(lambda x: {"turns": [prompt_fmt.format(**x)]})

    elif data_name == "aime24":
        dataset = load_dataset("HuggingFaceH4/aime_2024", split="train")
        prompt_fmt = "{problem}\nPlease reason step by step, and put your final answer within \\boxed{{}}."
        dataset = dataset.map(lambda x: {"turns": [prompt_fmt.format(**x)]})

    elif data_name == "aime25":
        dataset = load_dataset("MathArena/aime_2025", split="train")
        prompt_fmt = "{problem}\nPlease reason step by step, and put your final answer within \\boxed{{}}."
        dataset = dataset.map(lambda x: {"turns": [prompt_fmt.format(**x)]})

    # Chat datasets
    elif data_name == "alpaca":
        dataset = load_dataset("tatsu-lab/alpaca", split="train")
        dataset = dataset.map(
            lambda x: {
                "formatted_input": (f"{x['instruction']}\n\nInput:\n{x['input']}" if x["input"] else x["instruction"])
            }
        )
        dataset = dataset.map(lambda x: {"turns": [x["formatted_input"]]})

    elif data_name == "mt-bench":
        dataset = load_dataset("HuggingFaceH4/mt_bench_prompts", split="train")
        dataset = dataset.map(lambda x: {"turns": x["prompt"]})

    # Coding datasets
    elif data_name == "humaneval":
        dataset = load_dataset("openai/openai_humaneval", split="test")
        prompt_fmt = "Write a solution to the following problem and make sure that it passes the tests:\n```python\n{prompt}\n```"
        dataset = dataset.map(lambda x: {"turns": [prompt_fmt.format(**x)]})

    elif data_name == "mbpp":
        dataset = load_dataset("google-research-datasets/mbpp", "sanitized", split="test")
        dataset = dataset.map(lambda x: {"turns": [x["prompt"]]})

    elif data_name == "lbpp":
        LBPP_PY_TEST_URL = "https://huggingface.co/datasets/CohereLabs/lbpp/resolve/main/python/test.parquet"
        dataset = load_dataset("parquet", data_files={"test": LBPP_PY_TEST_URL})["test"]
        dataset = dataset.map(lambda x: {"turns": [x["instruction"]]})

    elif data_name == "swe-bench":
        dataset = load_dataset("princeton-nlp/SWE-bench_Lite", split="test")
        prompt_fmt = "Problem Statement:\n{problem_statement}\nPlease fix the issue described above."
        dataset = dataset.map(lambda x: {"turns": [prompt_fmt.format(**x)]})

    elif data_name == "livecodebench":
        base = "https://huggingface.co/datasets/livecodebench/code_generation_lite/resolve/main/"
        allowed_files = ["test.jsonl", "test2.jsonl", "test3.jsonl", "test4.jsonl", "test5.jsonl", "test6.jsonl"]
        urls = [base + fn for fn in allowed_files]
        dataset = load_dataset("json", data_files={"test": urls})["test"]

        def format_lcb(doc):
            system_prompt = (
                "You are an expert Python programmer. You will be given a question (problem specification) "
                "and will generate a correct Python program that matches the specification and passes all tests. "
                "You will NOT return anything except for the program"
            )
            question_block = f"### Question:\n{doc['question_content']}"
            if doc.get("starter_code"):
                format_message = "### Format: Use the following code structure:"
                code_block = f"```python\n{doc['starter_code']}\n```"
            else:
                format_message = "### Format: Write your code in the following format:"
                code_block = "```python\n# YOUR CODE HERE\n```"
            answer_footer = "### Answer: (use the provided format with backticks)"
            return f"{system_prompt}\n\n{question_block}\n\n{format_message}\n{code_block}\n\n{answer_footer}"

        target_features = Features({"turns": Sequence(Value("large_string"))})
        dataset = dataset.map(
            lambda x: {"turns": [format_lcb(x)]}, remove_columns=dataset.column_names, features=target_features
        )

    return dataset


_DEFAULT_FMT = "{prompt}\nPlease reason step by step, and put your final answer within \\boxed{{}}."
_CODING_FMT = (
    "Write a solution to the following problem and make sure that it passes the tests:\n```python\n{prompt}\n```"
)

_CATEGORY_FMT = {
    "math": _DEFAULT_FMT,
    "math_reasoning": _DEFAULT_FMT,
    "coding": _CODING_FMT,
    "reasoning": _DEFAULT_FMT,
    "stem": _DEFAULT_FMT,
    "qa": _DEFAULT_FMT,
    "rag": _DEFAULT_FMT,
    "extraction": _DEFAULT_FMT,
    "humanities": _DEFAULT_FMT,
    "writing": _DEFAULT_FMT,
    "summarization": _DEFAULT_FMT,
    "translation": _DEFAULT_FMT,
    "roleplay": _DEFAULT_FMT,
}


def format_prompt(prompt: str, category: str = "") -> str:
    fmt = _CATEGORY_FMT.get(category, _DEFAULT_FMT)
    return fmt.format(prompt=prompt)


def reformat_jsonl_by_category(questions: list) -> list:
    """Apply instruction prefix to JSONL questions based on their category.

    Mirrors the prompt_fmt logic in load_and_process_dataset: for categories
    where a specific instruction is obvious (math, coding) a tailored prefix is
    used; for all others the same step-by-step reasoning instruction is applied.
    """
    for q in questions:
        category = q.get("category", "")
        q["turns"][0] = format_prompt(q["turns"][0], category)
    return questions

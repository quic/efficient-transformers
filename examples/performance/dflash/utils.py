# -----------------------------------------------------------------------------
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------

import json
import warnings
from typing import Optional

import numpy as np
import torch
from datasets import Features, Sequence, Value, load_dataset
from huggingface_hub import hf_hub_download
from safetensors import safe_open
from torch import nn
from transformers import AutoModelForCausalLM


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


_TARGET_ABSMAX = 128.0


def print_stats(x, name: str) -> None:
    if isinstance(x, torch.Tensor):
        x_np = x.detach().cpu().to(torch.float32).numpy()
    elif isinstance(x, np.ndarray):
        x_np = x.astype(np.float32)
    else:
        raise TypeError("Input must be a torch.Tensor or numpy.ndarray")
    print(f"[STATS] {name}")
    print(f"  Shape : {x_np.shape}")
    print(f"  Min   : {x_np.min():.6f}")
    print(f"  Max   : {x_np.max():.6f}")
    print(f"  Mean  : {x_np.mean():.6f}")
    print(f"  Median: {np.median(x_np):.6f}")
    print(f"  Std   : {x_np.std():.6f}")


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


def extract_lm_head(model: AutoModelForCausalLM) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Return (lm_head_weight, lm_head_bias) from a HuggingFace causal LM (fp32)."""
    sd = model.state_dict()
    weight = sd["lm_head.weight"].to(torch.float32)
    bias = sd.get("lm_head.bias")
    if bias is not None:
        bias = bias.to(torch.float32)
    return weight, bias


def build_tlm_model(
    base_model: AutoModelForCausalLM,
    dflash_state_dict: dict,
    target_layer_ids: list[int],
    target_absmax: float = _TARGET_ABSMAX,
) -> AutoModelForCausalLM:
    """Attach fc + hidden_norm to *base_model*, inject DFlash weights, and scale fc.

    Modifies *base_model* in-place and returns it.
    """
    inner = base_model.model
    hidden_size = base_model.config.hidden_size
    model_type = getattr(base_model.config, "model_type", "")
    n = len(target_layer_ids)

    # Add fc and hidden_norm
    inner.fc = nn.Linear(n * hidden_size, hidden_size, bias=False)

    if "qwen3" in model_type:
        from transformers.models.qwen3.modeling_qwen3 import Qwen3RMSNorm

        inner.hidden_norm = Qwen3RMSNorm(hidden_size, eps=base_model.config.rms_norm_eps)
    elif "llama" in model_type:
        from transformers.models.llama.modeling_llama import LlamaRMSNorm

        inner.hidden_norm = LlamaRMSNorm(hidden_size, eps=base_model.config.rms_norm_eps)
    else:
        warnings.warn(f"Unknown model_type '{model_type}'; using nn.RMSNorm for hidden_norm.")
        inner.hidden_norm = nn.RMSNorm(hidden_size, eps=getattr(base_model.config, "rms_norm_eps", 1e-6))

    # Inject weights from DFlash checkpoint
    fc_tensor = dflash_state_dict["fc.weight"].to(torch.float32)
    inner.fc.weight.data.copy_(fc_tensor)

    hn_tensor = dflash_state_dict["hidden_norm.weight"].to(torch.float32)
    inner.hidden_norm.weight.data.copy_(hn_tensor)

    # Scale fc weights so activations stay within fp16 range
    # RMSNorm(x/s) == RMSNorm(x), so this is zero-accuracy-cost
    with torch.no_grad():
        in_feat = inner.fc.in_features
        max_row_norm = inner.fc.weight.data.norm(dim=1).max().item()
        fc_out_bound = (in_feat**0.5) * max_row_norm
        s = max(fc_out_bound / target_absmax, 1.0)
        inner.fc.weight.data.div_(s)
        print(
            f"[TLM] fc scale: in_features={in_feat}, max_row_norm={max_row_norm:.4f}, "
            f"fc_out_bound={fc_out_bound:.2f}, s={s:.6f}"
        )

    print(f"[TLM] fc ({n * hidden_size} -> {hidden_size}) and hidden_norm attached and scaled")
    return base_model


def build_dlm_model(
    dflash_model_path: str,
    lm_head_weight: torch.Tensor,
    lm_head_bias: torch.Tensor | None = None,
) -> AutoModelForCausalLM:
    """Load the DFlash model and inject lm_head weights from the base TLM model.

    Also removes fc / hidden_norm if the DFlash checkpoint has them.
    """
    dlm_model = AutoModelForCausalLM.from_pretrained(dflash_model_path, torch_dtype=torch.float32)

    with torch.no_grad():
        dlm_model.lm_head.weight.copy_(lm_head_weight)

    if lm_head_bias is not None:
        if dlm_model.lm_head.bias is None:
            dlm_model.lm_head.bias = nn.Parameter(lm_head_bias)
        else:
            with torch.no_grad():
                dlm_model.lm_head.bias.copy_(lm_head_bias)

    # DFlash checkpoints occasionally carry fc / hidden_norm — strip them
    for attr in ("fc", "hidden_norm"):
        if hasattr(dlm_model, attr):
            delattr(dlm_model, attr)
            print(f"[DLM] Removed dlm_model.{attr}")

    print(f"[DLM] lm_head injected (shape: {lm_head_weight.shape})")
    return dlm_model

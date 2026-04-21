# -----------------------------------------------------------------------------
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------

"""Generic HF generation entrypoint for post-PEFT finetuning inference."""

from __future__ import annotations

import argparse
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import torch
import transformers
from peft import PeftConfig, PeftModel
from transformers import AutoTokenizer, StoppingCriteria, StoppingCriteriaList

from QEfficient.finetune.experimental.core.config_manager import ConfigManager, MasterConfig
from QEfficient.finetune.experimental.core.logger import Logger

logger = Logger(__name__)


def _resolve_torch_dtype(dtype_value: Optional[str]) -> Optional[torch.dtype]:
    """Map config dtype strings to `torch.dtype` values."""
    if dtype_value is None:
        return None
    if isinstance(dtype_value, torch.dtype):
        return dtype_value

    normalized = str(dtype_value).strip().lower()
    mapping = {
        "fp16": torch.float16,
        "float16": torch.float16,
        "half": torch.float16,
        "bf16": torch.bfloat16,
        "bfloat16": torch.bfloat16,
        "fp32": torch.float32,
        "float32": torch.float32,
        "auto": None,
    }
    if normalized not in mapping:
        raise ValueError(f"Unsupported torch dtype: {dtype_value}")
    return mapping[normalized]


def _is_adapter_checkpoint(path: Path) -> bool:
    """Check whether a directory looks like a PEFT adapter checkpoint."""
    return path.is_dir() and (path / "adapter_config.json").exists()


def _has_tokenizer_artifacts(path: Path) -> bool:
    """Check whether a directory contains tokenizer files."""
    if not path.is_dir():
        return False
    tokenizer_files = (
        "tokenizer.json",
        "tokenizer_config.json",
        "special_tokens_map.json",
        "vocab.json",
        "merges.txt",
    )
    return any((path / filename).exists() for filename in tokenizer_files)


def _resolve_tokenizer_source(
    *,
    adapter_path: Optional[str],
    tokenizer_name: Optional[str],
    base_model_path: str,
) -> str:
    """Pick tokenizer files from the adapter dir when they exist."""
    if adapter_path:
        adapter_tokenizer_path = Path(adapter_path)
        if _has_tokenizer_artifacts(adapter_tokenizer_path):
            return adapter_path
    if tokenizer_name:
        return tokenizer_name
    return base_model_path


def resolve_adapter_checkpoint(path: str) -> str:
    """Resolve a PEFT checkpoint directory from a path or training output root."""
    candidate = Path(path).expanduser().resolve()
    if candidate.is_file():
        candidate = candidate.parent

    if _is_adapter_checkpoint(candidate):
        return str(candidate)

    found: List[Path] = []
    for root, _, files in os.walk(candidate):
        if "adapter_config.json" in files:
            found.append(Path(root))

    if not found:
        return str(candidate)

    found.sort(key=lambda p: (p.stat().st_mtime, str(p)))
    return str(found[-1])


def _extract_base_model_name(adapter_path: str) -> Optional[str]:
    """Read the base model name from the PEFT adapter config."""
    try:
        peft_config = PeftConfig.from_pretrained(adapter_path)
    except Exception:
        return None
    base_model_name = getattr(peft_config, "base_model_name_or_path", None)
    return base_model_name or None


def _to_prompt_list(
    prompt: Optional[str] = None,
    prompts: Optional[Sequence[str]] = None,
    prompts_file: Optional[str] = None,
) -> List[str]:
    """Normalize prompt input from CLI strings or files."""
    if prompts:
        return [p for p in prompts if p and p.strip()]
    if prompts_file:
        with open(prompts_file, "r", encoding="utf-8") as f:
            return [line.strip() for line in f if line.strip()]
    if prompt:
        return [p.strip() for p in re.split(r"\s*\|\s*", prompt) if p.strip()]
    return []


def _to_string_list(value: Optional[str]) -> List[str]:
    """Split a delimited string into a clean list."""
    if not value:
        return []
    return [item.strip() for item in re.split(r"\s*[|,]\s*", value) if item.strip()]


def _format_prompt(
    prompt: str,
    *,
    prompt_template: Optional[str] = None,
    use_chat_template: bool = False,
    system_prompt: Optional[str] = None,
    tokenizer: Any = None,
) -> str:
    """Apply an optional prompt template or chat template."""
    if prompt_template:
        return prompt_template.format(prompt=prompt, input=prompt, text=prompt)

    if use_chat_template and tokenizer is not None and hasattr(tokenizer, "apply_chat_template"):
        messages: List[Dict[str, str]] = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    return prompt


def _infer_input_device(model: torch.nn.Module) -> torch.device:
    """Use the first model parameter to infer the execution device."""
    try:
        return next(model.parameters()).device
    except StopIteration:
        return torch.device("cpu")


def _build_stop_sequences(tokenizer: Any, stop_strings: Optional[Sequence[str]]) -> List[List[int]]:
    """Tokenize stop strings for generation-time early stopping."""
    stop_sequences: List[List[int]] = []
    for stop_string in stop_strings or []:
        token_ids = tokenizer.encode(stop_string, add_special_tokens=False)
        if token_ids:
            stop_sequences.append(token_ids)
    return stop_sequences


def _trim_at_stop_strings(text: str, stop_strings: Optional[Sequence[str]]) -> str:
    """Trim decoded text at the first matching stop string."""
    if not stop_strings:
        return text
    cut_index = len(text)
    for stop_string in stop_strings:
        index = text.find(stop_string)
        if index != -1:
            cut_index = min(cut_index, index)
    return text[:cut_index].rstrip()


class _StopOnTokenSequences(StoppingCriteria):
    """Stop generation when any configured token sequence appears."""
    def __init__(self, stop_sequences: Sequence[Sequence[int]]):
        self.stop_sequences = [tuple(sequence) for sequence in stop_sequences if sequence]

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        for sequence in input_ids:
            sequence_list = sequence.tolist()
            for stop_sequence in self.stop_sequences:
                if len(sequence_list) >= len(stop_sequence) and tuple(sequence_list[-len(stop_sequence) :]) == stop_sequence:
                    return True
        return False


def load_model_with_adapter(
    *,
    base_model_path: str,
    adapter_path: Optional[str] = None,
    tokenizer_name: Optional[str] = None,
    auto_class_name: str = "AutoModelForCausalLM",
    trust_remote_code: bool = False,
    device_map: Optional[Any] = None,
    torch_dtype: Optional[torch.dtype] = None,
    attn_implementation: Optional[str] = "sdpa",
    use_cache: bool = True,
) -> tuple[torch.nn.Module, Any]:
    """Load the base model, then attach the PEFT adapter if provided."""
    auto_class = getattr(transformers, auto_class_name)
    model = auto_class.from_pretrained(
        base_model_path,
        trust_remote_code=trust_remote_code,
        device_map=device_map,
        torch_dtype=torch_dtype,
        attn_implementation=attn_implementation,
        use_cache=use_cache,
        low_cpu_mem_usage=True,
    )

    if adapter_path:
        model = PeftModel.from_pretrained(model, adapter_path, is_trainable=False)

    tokenizer_source = _resolve_tokenizer_source(
        adapter_path=adapter_path,
        tokenizer_name=tokenizer_name,
        base_model_path=base_model_path,
    )
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_source, trust_remote_code=trust_remote_code)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    return model, tokenizer


def generate_text(
    model: torch.nn.Module,
    tokenizer: Any,
    prompts: Sequence[str],
    *,
    max_new_tokens: int = 128,
    temperature: float = 0.0,
    top_p: float = 1.0,
    top_k: int = 50,
    do_sample: bool = False,
    num_beams: int = 1,
    repetition_penalty: float = 1.0,
    prompt_template: Optional[str] = None,
    use_chat_template: bool = False,
    system_prompt: Optional[str] = None,
    stop_strings: Optional[Sequence[str]] = None,
) -> List[str]:
    """Run batched HF generation and return decoded continuations."""
    if not prompts:
        return []

    formatted_prompts = [
        _format_prompt(
            prompt,
            prompt_template=prompt_template,
            use_chat_template=use_chat_template,
            system_prompt=system_prompt,
            tokenizer=tokenizer,
        )
        for prompt in prompts
    ]

    encoded = tokenizer(
        formatted_prompts,
        return_tensors="pt",
        padding=True,
        truncation=True,
    )

    input_device = _infer_input_device(model)
    encoded = {k: v.to(input_device) for k, v in encoded.items()}
    input_width = encoded["input_ids"].shape[1]

    stop_sequences = _build_stop_sequences(tokenizer, stop_strings)
    generation_kwargs = {
        "max_new_tokens": max_new_tokens,
        "temperature": temperature,
        "top_p": top_p,
        "top_k": top_k,
        "do_sample": do_sample,
        "num_beams": num_beams,
        "repetition_penalty": repetition_penalty,
        "pad_token_id": tokenizer.pad_token_id,
        "eos_token_id": tokenizer.eos_token_id,
    }
    if stop_sequences:
        generation_kwargs["stopping_criteria"] = StoppingCriteriaList([_StopOnTokenSequences(stop_sequences)])

    model.eval()
    with torch.inference_mode():
        outputs = model.generate(**encoded, **generation_kwargs)

    generations: List[str] = []
    for output_ids in outputs:
        generated_tokens = output_ids[input_width:]
        generation = tokenizer.decode(generated_tokens, skip_special_tokens=True, clean_up_tokenization_spaces=True).strip()
        generations.append(_trim_at_stop_strings(generation, stop_strings))

    return generations


def main(
    config_path: Optional[str] = None,
    base_model_path: Optional[str] = None,
    adapter_path: Optional[str] = None,
    prompt: Optional[str] = None,
    prompts_file: Optional[str] = None,
    prompts: Optional[Sequence[str]] = None,
    prompt_template: Optional[str] = None,
    system_prompt: Optional[str] = None,
    use_chat_template: bool = False,
    max_new_tokens: int = 128,
    temperature: float = 0.0,
    top_p: float = 1.0,
    top_k: int = 50,
    do_sample: bool = False,
    num_beams: int = 1,
    repetition_penalty: float = 1.0,
    stop_strings: Optional[Sequence[str]] = None,
    trust_remote_code: bool = False,
    device: Optional[str] = None,
    device_map: Optional[Any] = None,
    torch_dtype: Optional[str] = None,
    auto_class_name: str = "AutoModelForCausalLM",
    attn_implementation: Optional[str] = "sdpa",
    use_cache: bool = True,
) -> List[str]:
    """Load a base checkpoint, attach PEFT adapters, and run HF generate."""
    config_manager = ConfigManager(config_path=config_path) if config_path else ConfigManager(config=MasterConfig())
    model_config = config_manager.get_model_config()
    dataset_config = config_manager.get_dataset_config()
    training_config = config_manager.get_training_config()

    resolved_adapter_path = adapter_path or training_config.get("resume_from_checkpoint")
    if resolved_adapter_path is None:
        resolved_adapter_path = training_config.get("output_dir")
    if resolved_adapter_path is not None:
        resolved_adapter_path = resolve_adapter_checkpoint(str(resolved_adapter_path))
        if not _is_adapter_checkpoint(Path(resolved_adapter_path)):
            raise FileNotFoundError(
                f"Could not find a PEFT adapter checkpoint under: {resolved_adapter_path}. "
                "Pass --adapter-path explicitly or point config.training.output_dir at a saved adapter."
            )

    resolved_base_model_path = base_model_path or model_config.get("model_name")
    if resolved_base_model_path is None and resolved_adapter_path:
        resolved_base_model_path = _extract_base_model_name(resolved_adapter_path)
    if resolved_base_model_path is None:
        raise ValueError("Unable to resolve a base model path.")

    resolved_tokenizer_name = dataset_config.get("tokenizer_name") or model_config.get("tokenizer_name")
    resolved_torch_dtype = _resolve_torch_dtype(torch_dtype or model_config.get("torch_dtype"))
    if resolved_torch_dtype is None and device and device.startswith("cuda"):
        resolved_torch_dtype = torch.float16

    resolved_device_map = None if device_map in (None, "", "none", "None") else device_map
    model, tokenizer = load_model_with_adapter(
        base_model_path=resolved_base_model_path,
        adapter_path=resolved_adapter_path,
        tokenizer_name=resolved_tokenizer_name,
        auto_class_name=auto_class_name,
        trust_remote_code=trust_remote_code,
        device_map=resolved_device_map,
        torch_dtype=resolved_torch_dtype,
        attn_implementation=attn_implementation,
        use_cache=use_cache,
    )

    prompts_list = _to_prompt_list(prompt=prompt, prompts=prompts, prompts_file=prompts_file)
    if not prompts_list:
        raise ValueError("Provide at least one prompt via prompt, prompts, or prompts_file.")

    if resolved_device_map is None and device is not None:
        model.to(device)

    logger.log_rank_zero(f"Base model: {resolved_base_model_path}")
    if resolved_adapter_path:
        logger.log_rank_zero(f"Adapter checkpoint: {resolved_adapter_path}")

    generations = generate_text(
        model,
        tokenizer,
        prompts_list,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        do_sample=do_sample,
        num_beams=num_beams,
        repetition_penalty=repetition_penalty,
        prompt_template=prompt_template,
        use_chat_template=use_chat_template,
        system_prompt=system_prompt,
        stop_strings=stop_strings,
    )

    for src, dst in zip(prompts_list, generations):
        logger.log_rank_zero(f"Prompt: {src}")
        logger.log_rank_zero(f"Generation: {dst}")

    return generations


def _build_parser() -> argparse.ArgumentParser:
    """Define the CLI for the experimental inference entrypoint."""
    parser = argparse.ArgumentParser(description="Run HF generation with a PEFT adapter on a finetuned checkpoint.")
    parser.add_argument("--config-path", "--config_path", dest="config_path", type=str, default=None, help="Path to a YAML config file.")
    parser.add_argument("--base-model-path", "--base_model_path", dest="base_model_path", type=str, default=None, help="Path or HF id of the base model.")
    parser.add_argument("--adapter-path", "--adapter_path", dest="adapter_path", type=str, default=None, help="Path to the PEFT adapter checkpoint.")
    parser.add_argument("--prompt", type=str, default=None, help="Single prompt or pipe-separated prompts.")
    parser.add_argument("--prompts-file", "--prompts_file", dest="prompts_file", type=str, default=None, help="File with one prompt per line.")
    parser.add_argument("--prompt-template", "--prompt_template", dest="prompt_template", type=str, default=None, help="Template used to format each prompt.")
    parser.add_argument("--system-prompt", "--system_prompt", dest="system_prompt", type=str, default=None, help="Optional system message for chat prompts.")
    parser.add_argument("--use-chat-template", "--use_chat_template", dest="use_chat_template", action="store_true", help="Format prompts with the tokenizer chat template.")
    parser.add_argument("--max-new-tokens", "--max_new_tokens", dest="max_new_tokens", type=int, default=128, help="Maximum number of generated tokens.")
    parser.add_argument("--temperature", type=float, default=0.0, help="Sampling temperature.")
    parser.add_argument("--top-p", "--top_p", dest="top_p", type=float, default=1.0, help="Nucleus sampling probability.")
    parser.add_argument("--top-k", "--top_k", dest="top_k", type=int, default=50, help="Top-k sampling cutoff.")
    parser.add_argument("--do-sample", "--do_sample", dest="do_sample", action="store_true", help="Enable stochastic sampling.")
    parser.add_argument("--num-beams", "--num_beams", dest="num_beams", type=int, default=1, help="Beam search width.")
    parser.add_argument(
        "--repetition-penalty",
        "--repetition_penalty",
        dest="repetition_penalty",
        type=float,
        default=1.0,
        help="Penalty for repeated tokens.",
    )
    parser.add_argument(
        "--stop-strings",
        "--stop_strings",
        dest="stop_strings",
        type=str,
        default='####',
        help="Pipe- or comma-separated stop strings, for example '####|</s>'.",
    )
    parser.add_argument("--trust-remote-code", "--trust_remote_code", dest="trust_remote_code", action="store_true", help="Allow loading custom model code.")
    parser.add_argument("--device", type=str, default=None, help="Target device for inference.")
    parser.add_argument("--device-map", "--device_map", dest="device_map", type=str, default=None, help="HF device map for model placement.")
    parser.add_argument("--torch-dtype", "--torch_dtype", dest="torch_dtype", type=str, default=None, help="Model dtype such as fp16 or bf16.")
    parser.add_argument("--auto-class-name", "--auto_class_name", dest="auto_class_name", type=str, default="AutoModelForCausalLM", help="HF auto model class name.")
    parser.add_argument("--attn-implementation", "--attn_implementation", dest="attn_implementation", type=str, default="sdpa", help="Attention backend to use.")
    parser.add_argument("--use-cache", "--use_cache", dest="use_cache", action="store_true", help="Enable KV cache during generation.")
    parser.add_argument("--no-use-cache", "--no_use_cache", dest="use_cache", action="store_false", help="Disable KV cache during generation.")
    parser.set_defaults(use_cache=True)
    return parser


if __name__ == "__main__":
    args = _build_parser().parse_args()
    main(
        config_path=args.config_path,
        base_model_path=args.base_model_path,
        adapter_path=args.adapter_path,
        prompt=args.prompt,
        prompts_file=args.prompts_file,
        prompt_template=args.prompt_template,
        system_prompt=args.system_prompt,
        use_chat_template=args.use_chat_template,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        do_sample=args.do_sample,
        num_beams=args.num_beams,
        repetition_penalty=args.repetition_penalty,
        stop_strings=_to_string_list(args.stop_strings),
        trust_remote_code=args.trust_remote_code,
        device=args.device,
        device_map=args.device_map,
        torch_dtype=args.torch_dtype,
        auto_class_name=args.auto_class_name,
        attn_implementation=args.attn_implementation,
        use_cache=args.use_cache,
    )

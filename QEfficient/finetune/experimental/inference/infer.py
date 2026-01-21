# -----------------------------------------------------------------------------
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------
import json
import os
import sys
from abc import ABC
from typing import Any, Dict, List, Optional, Union

import torch
from torch.utils.data import DataLoader
from transformers import (
    AutoModelForCausalLM,
    AutoModelForSeq2SeqLM,
    AutoProcessor,
    AutoTokenizer,
    HfArgumentParser,
    TextStreamer,
)

# QEfficient imports
from QEfficient.finetune.experimental.core.dataset import SFTDataset
from QEfficient.finetune.experimental.core.logger import Logger
from QEfficient.finetune.experimental.core.model import HFModel
from QEfficient.finetune.experimental.core.utils.dataset_utils import insert_pad_token

logger = Logger(__name__)

# Optional PEFT (LoRA)
try:
    from peft import PeftModel

    PEFT_AVAILABLE = True
except Exception:
    PEFT_AVAILABLE = False

import tempfile
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from pathlib import Path


@dataclass
class MasterConfig:
    # ---- Required (no default) ----
    model: str
    output: str

    # ---- Task & Model ----
    task: str = "text-generation"  # text-generation | text2text-generation | chat-generation | vlm-generation
    model_cls: Optional[str] = "AutoModelForCausalLM"
    tokenizer: Optional[str] = None  # defaults to model id if None
    processor: Optional[str] = None  # defaults to model id if None
    lora: Optional[str] = None
    hf_token: Optional[str] = None
    trust_remote_code: bool = False
    revision: Optional[str] = None

    # ---- Runtime ----
    device: str = "qaic"  # cpu | cuda | qaic
    tensor_parallel: bool = False
    fp16: bool = False
    bf16: bool = False

    # ---- Generation ----
    batch_size: int = 1
    max_new_tokens: int = 128
    do_sample: bool = False
    temperature: float = 0.8
    top_k: int = 50
    top_p: float = 0.95
    stream: bool = False
    repetition_penalty: float = 1.0  # >= 1.0
    no_repeat_ngram_size: int = 0  # >= 0

    # ---- Templates ----
    prompt_template: str = "{prompt}"
    completion_template: str = "{response}"
    stop_seq: List[str] = field(default_factory=lambda: ["### prompt:", "### response:"])

    # ---- Inputs ----
    input: Optional[str] = None  # path to .txt/.json/.jsonl, or raw string
    inputs: Optional[List[Any]] = None  # inline list of prompts


# -----------------------------------------------------------------------------
# Parse function in your requested style (uses HfArgumentParser(MasterConfig))
# -----------------------------------------------------------------------------
def parse_arguments(config_path: Optional[str] = None, args: Optional[List[str]] = None) -> MasterConfig:
    """Create argument parser for the new finetuning interface."""
    parser = HfArgumentParser(MasterConfig)

    # If an explicit config path was provided, parse it as YAML (only .yaml/.yml allowed)
    if config_path:
        config_path = os.path.abspath(config_path)
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Config file not found: {config_path}")
        if not (config_path.endswith(".yaml") or config_path.endswith(".yml")):
            raise ValueError(f"Expected a .yaml/.yml file, got: {config_path}")

        try:
            (master_config,) = parser.parse_yaml_file(yaml_file=config_path)
            return master_config
        except Exception as e:
            raise ValueError(f"Failed to parse YAML config '{config_path}': {e}")

    # If args not provided, default to empty list
    args = [] if args is None else args

    # If a single positional YAML file was passed via args, parse it as YAML
    if len(args) == 1 and (args[0].endswith(".yaml") or args[0].endswith(".yml")):
        yaml_path = os.path.abspath(args[0])
        (master_config,) = parser.parse_yaml_file(yaml_file=yaml_path)
    else:
        # Parse CLI args directly into dataclasses
        (master_config,) = parser.parse_args_into_dataclasses(args=args)
        # Ensure we return a proper MasterConfig object (not Frozen/namespace)
        master_config = MasterConfig(**asdict(master_config))

    return master_config


def _sanitize_prompt(x: Any) -> Optional[str]:
    if isinstance(x, dict):
        for key in ("prompt", "text"):
            val = x.get(key)
            if val is None:
                continue
            s = str(val).strip()
            if s:
                return s
    elif isinstance(x, str):
        s = x.strip()
        if s:
            return s
    else:
        s = str(x).strip()
        if s:
            return s
    return None


@contextmanager
def temporary_sft_json(items: List[Any]):
    """
    Write a temp JSON with canonical SFT keys:
      {"prompt": "<text>", "response": ""}
    Automatically deleted after use.
    """
    records = []
    for it in items:
        p = _sanitize_prompt(it)
        if p is None:
            continue
        records.append({"prompt": p, "response": ""})
    if not records:
        raise ValueError("No valid prompts in batch to build SFT JSON.")

    with tempfile.NamedTemporaryFile(delete=False, suffix=".json") as tmp:
        tmp_path = tmp.name
    Path(tmp_path).write_text(json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8")

    try:
        yield tmp_path
    finally:
        try:
            Path(tmp_path).unlink()
        except Exception:
            pass


def load_items_from_config(cfg: MasterConfig) -> List[Any]:
    """Load items from cfg.inputs (inline) or cfg.input (file/raw string)."""
    if cfg.inputs is not None:
        if not isinstance(cfg.inputs, list):
            raise ValueError("'inputs' must be a list.")
        items = [it for it in cfg.inputs if it is not None and str(it).strip()]
        if not items:
            raise ValueError("No non-empty prompts found in 'inputs'.")
        return items

    if cfg.input is None:
        raise ValueError("Provide either 'input' path or 'inputs' list in the config/CLI.")

    path = str(cfg.input)
    if os.path.isfile(path):
        ext = os.path.splitext(path)[1].lower()
        items: List[Any] = []
        if ext == ".json":
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, dict):
                # Try common container keys; else single item
                for key in ("data", "records", "samples", "items"):
                    val = data.get(key)
                    if isinstance(val, list) and val:
                        items = val
                        break
                if not items:
                    items = [data]
            elif isinstance(data, list):
                items = data
            else:
                raise ValueError("Unsupported JSON structure.")
        elif ext == ".jsonl":
            with open(path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    items.append(json.loads(line))
        else:
            # Treat as raw text file: each line a prompt (text-only)
            with open(path, "r", encoding="utf-8") as f:
                items = [ln.strip() for ln in f if ln.strip()]

        items = [it for it in items if it is not None and str(it).strip()]
        if not items:
            raise ValueError("No inputs found in the provided file.")
        return items
    else:
        # Raw string passed in cfg.input
        s = path.strip()
        if not s:
            raise ValueError("Provided 'input' string is empty.")
        return [s]


def save_jsonl(path: str, rows: List[Dict[str, Any]]):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def select_dtype(device: str, fp16: bool, bf16: bool) -> torch.dtype:
    if fp16:
        return torch.float16
    if bf16:
        return torch.bfloat16
    return torch.float16 if device == "cuda" else torch.float32


def resolve_device_map(tensor_parallel: bool) -> Optional[str]:
    return "auto" if tensor_parallel else None


def attach_lora_if_any(model, lora_path: Optional[str]):
    if not lora_path or not PEFT_AVAILABLE:
        return model
    model = PeftModel.from_pretrained(model, lora_path)
    try:
        model = model.merge_and_unload()
        logger.info("ℹ️ LoRA adapter merged into base model.")
    except Exception:
        logger.info("ℹ️ LoRA adapter attached without merge.")
    return model


def select_device_and_backend(device_opt: str, rank: int = 0):
    device = torch.device("cpu")
    backend = "gloo"
    os.environ["TORCH_DISTRIBUTED_BACKEND"] = backend

    if device_opt == "qaic":
        try:
            import torch_qaic  # noqa: F401

            logger.info("QAIC support is available")
            os.environ["CUDA_VISIBLE_DEVICES"] = ""
            device = torch.device(f"qaic:{rank}")
            backend = "qccl"
            os.environ["TORCH_DISTRIBUTED_BACKEND"] = backend
            logger.info(f"Using QAIC device {rank} with backend '{backend}'")
            return device, backend
        except ImportError as e:
            logger.info(f"QAIC support not available: {e}")
            logger.info("Falling back to CPU (gloo).")

    if device_opt == "cuda":
        if torch.cuda.is_available():
            device = torch.device("cuda:0")
            backend = "nccl"
            os.environ["TORCH_DISTRIBUTED_BACKEND"] = backend
            logger.info(f"Using CUDA device {device} with backend '{backend}'")
            return device, backend
        else:
            logger.info("CUDA is not available. Falling back to CPU (gloo).")

    logger.info("Using CPU device with backend 'gloo'")
    return device, backend


class Components:
    def __init__(self, model: Any, tokenizer: Optional[Any] = None, processor: Optional[Any] = None):
        self.model = model
        self.tokenizer = tokenizer
        self.processor = processor


def load_components(cfg: MasterConfig) -> Components:
    device_map = resolve_device_map(cfg.tensor_parallel)
    dtype = select_dtype(cfg.device, cfg.fp16, cfg.bf16)
    rank = int(os.environ.get("LOCAL_RANK", 0))
    device, _ = select_device_and_backend(cfg.device, rank)

    tokenizer = None
    processor = None

    # Decide model class
    model_cls = cfg.model_cls
    if model_cls is None:
        if cfg.task in ["text-generation", "chat-generation"]:
            model_cls = "AutoModelForCausalLM"
        elif cfg.task == "text2text-generation":
            model_cls = "AutoModelForSeq2SeqLM"
        elif cfg.task == "vlm-generation":
            model_cls = "AutoModelForCausalLM"
        else:
            raise ValueError(f"Unknown task: {cfg.task}")
    model_id = cfg.model
    tokenizer_id = cfg.tokenizer or model_id
    processor_id = cfg.processor or model_id
    load_kwargs = {
        "torch_dtype": dtype,
        "device_map": device_map,
        "trust_remote_code": cfg.trust_remote_code,
        "tokenizer_name": tokenizer_id,
    }
    if cfg.revision:
        load_kwargs["revision"] = cfg.revision

    # Try HFModel wrapper first; fallback to raw Transformers on mismatch
    try:
        model_obj = HFModel(model_name=model_id, auto_class_name=model_cls, **load_kwargs)
        model = model_obj.load_model()
        tokenizer = model_obj.load_tokenizer()
    except TypeError:
        load_kwargs.pop("tokenizer_name", None)
        if model_cls == "AutoModelForCausalLM":
            model = AutoModelForCausalLM.from_pretrained(model_id, **load_kwargs)
        elif model_cls == "AutoModelForSeq2SeqLM":
            model = AutoModelForSeq2SeqLM.from_pretrained(model_id, **load_kwargs)
        else:
            model = AutoModelForCausalLM.from_pretrained(model_id, **load_kwargs)
        try:
            tokenizer = AutoTokenizer.from_pretrained(tokenizer_id, use_fast=True)
        except Exception:
            tokenizer = AutoTokenizer.from_pretrained(tokenizer_id, use_fast=False)

    # Processor for VLMs or multi-modal models
    if cfg.task == "vlm-generation":
        try:
            processor = AutoProcessor.from_pretrained(processor_id)
        except Exception as e:
            logger.info(f"⚠️ AutoProcessor load failed: {e}\nFalling back to tokenizer.")
            if tokenizer is None:
                try:
                    tokenizer = AutoTokenizer.from_pretrained(tokenizer_id, use_fast=True)
                except Exception:
                    tokenizer = AutoTokenizer.from_pretrained(tokenizer_id, use_fast=False)

    # If not using device_map auto-sharding, move to explicit device
    if device_map is None:
        model.to(device)

    # Ensure PAD exists / right padding via project util
    if tokenizer:
        insert_pad_token(tokenizer)

    # Attach LoRA if any
    model = attach_lora_if_any(model, cfg.lora)

    return Components(model=model, tokenizer=tokenizer, processor=processor)


class BaseTask(ABC):
    name: str = "base"

    def preprocess_batch(self, batch: List[Any], cfg: MasterConfig, comps: Components) -> Any: ...
    def generate(self, enc, cfg: MasterConfig, comps: Components) -> Any: ...
    def postprocess_batch(
        self, outputs, batch: List[Any], cfg: MasterConfig, comps: Components
    ) -> List[Dict[str, Any]]: ...


class CausalLMTask(BaseTask):
    name = "text-generation"

    def preprocess_batch(self, batch: List[Union[str, Dict[str, Any]]], cfg: MasterConfig, comps: Components):
        prompt_template = cfg.prompt_template
        completion_template = cfg.completion_template

        with temporary_sft_json(batch) as json_path:
            dataset = SFTDataset(
                dataset_name="inference",
                split="inference",
                split_ratio=1.0,
                remove_samples_with_empty_columns=False,
                json_file_path=json_path,
                prompt_template=prompt_template,
                completion_template=completion_template,
                task=cfg.task,
            )

            n = len(dataset) if hasattr(dataset, "__len__") else None
            logger.info(f"SFTDataset length: {n}")
            if n and n > 0:
                logger.info(f"SFTDataset[0]: {dataset[0]}")

            collate = getattr(dataset, "collate_fn", None)
            if callable(collate):
                loader = DataLoader(dataset, batch_size=len(dataset), collate_fn=collate)
                enc = next(iter(loader))
            else:
                samples = [dataset[i] for i in range(len(dataset))]
                texts = []
                for s in samples:
                    for key in ("prompt", "input", "text"):
                        v = s.get(key)
                        if v is None:
                            continue
                        sv = str(v).strip()
                        if sv:
                            texts.append(sv)
                            break
                enc = safe_tokenize(comps.tokenizer, texts, return_tensors="pt", padding=True)

        device = next(comps.model.parameters()).device
        enc = {k: (v.to(device) if hasattr(v, "to") else v) for k, v in enc.items()}
        return enc

    def generate(self, enc, cfg: MasterConfig, comps: Components):
        pad_id = comps.tokenizer.pad_token_id if comps.tokenizer else None
        eos_id = comps.tokenizer.eos_token_id if comps.tokenizer else None
        gen_kwargs = {
            "max_new_tokens": cfg.max_new_tokens,
            "do_sample": cfg.do_sample,
            "temperature": cfg.temperature,
            "top_k": cfg.top_k,
            "top_p": cfg.top_p,
            "repetition_penalty": cfg.repetition_penalty,
            "no_repeat_ngram_size": cfg.no_repeat_ngram_size,
            "pad_token_id": pad_id,
            "eos_token_id": eos_id,
        }
        if cfg.stream and comps.tokenizer is not None:
            gen_kwargs["streamer"] = TextStreamer(comps.tokenizer, skip_prompt=True, skip_special_tokens=True)

        attn = enc.get("attention_mask", None)
        if attn is None:
            raise ValueError("attention_mask missing in enc; cannot compute prompt lengths.")
        prompt_lens = attn.sum(dim=1).tolist()

        with torch.inference_mode():
            out_ids = comps.model.generate(**enc, **gen_kwargs)

        logger.info(f"Generated sequences shape: {tuple(out_ids.shape)}; prompt_lens: {prompt_lens}")
        return {"sequences": out_ids, "prompt_lens": prompt_lens}

    def postprocess_batch(self, outputs, batch, cfg: MasterConfig, comps: Components):
        out_ids = outputs["sequences"]
        prompt_lens = outputs["prompt_lens"]

        results = []
        cont_lengths = []

        for i, raw in enumerate(batch):
            seq = out_ids[i]
            start = prompt_lens[i]
            if hasattr(seq, "shape"):
                start = max(0, min(start, seq.shape[0]))

            cont_ids = seq[start:]
            cont_lengths.append(cont_ids.shape[0] if hasattr(cont_ids, "shape") else len(cont_ids))

            if comps.tokenizer:
                assistant = comps.tokenizer.decode(cont_ids, skip_special_tokens=True)
            else:
                assistant = str(cont_ids.tolist()) if hasattr(cont_ids, "tolist") else str(cont_ids)

            assistant = assistant.strip()
            # Optional clamp: keep only first line/sentence (clean grammar outputs)
            assistant = assistant.splitlines()[0].strip()
            if "." in assistant:
                assistant = assistant.split(".", 1)[0].strip() + "."

            for stop in cfg.stop_seq or []:
                if stop and stop in assistant:
                    assistant = assistant.split(stop, 1)[0].strip()

            full_txt = (
                comps.tokenizer.decode(seq, skip_special_tokens=True)
                if comps.tokenizer
                else (str(seq.tolist()) if hasattr(seq, "tolist") else str(seq))
            )

            results.append({"input": raw, "full_text": full_txt, "response": assistant})

        logger.info(f"Continuation token lengths per sample: {cont_lengths}")
        return results


class Seq2SeqTask(BaseTask):
    name = "text2text-generation"

    def preprocess_batch(self, batch: List[Union[str, Dict[str, Any]]], cfg: MasterConfig, comps: Components):
        prompt_template = cfg.prompt_template
        completion_template = cfg.completion_template

        with temporary_sft_json(batch) as json_path:
            dataset = SFTDataset(
                dataset_name="inference",
                split="inference",
                json_file_path=json_path,
                prompt_template=prompt_template,
                completion_template=completion_template,
                tokenizer=comps.tokenizer,
                processor=getattr(comps, "processor", None),
                task=cfg.task,
            )

            collate = getattr(dataset, "collate_fn", None)
            if callable(collate):
                loader = DataLoader(dataset, batch_size=len(dataset), collate_fn=collate)
                enc = next(iter(loader))
            else:
                samples = [dataset[i] for i in range(len(dataset))]
                texts = []
                for s in samples:
                    for key in ("prompt", "input", "text"):
                        v = s.get(key)
                        if v is None:
                            continue
                        sv = str(v).strip()
                        if sv:
                            texts.append(sv)
                            break
                enc = safe_tokenize(comps.tokenizer, texts, return_tensors="pt", padding=True)

        device = next(comps.model.parameters()).device
        enc = {k: (v.to(device) if hasattr(v, "to") else v) for k, v in enc.items()}
        return enc

    def generate(self, enc, cfg: MasterConfig, comps: Components):
        pad_id = comps.tokenizer.pad_token_id if comps.tokenizer else None
        eos_id = comps.tokenizer.eos_token_id if comps.tokenizer else None
        gen_kwargs = {
            "max_new_tokens": cfg.max_new_tokens,
            "do_sample": cfg.do_sample,
            "temperature": cfg.temperature,
            "top_k": cfg.top_k,
            "top_p": cfg.top_p,
            "repetition_penalty": cfg.repetition_penalty,
            "no_repeat_ngram_size": cfg.no_repeat_ngram_size,
            "pad_token_id": pad_id,
            "eos_token_id": eos_id,
        }
        gen_kwargs = {k: v for k, v in gen_kwargs.items() if v is not None}
        with torch.inference_mode():
            out_ids = comps.model.generate(**enc, **gen_kwargs)
        return out_ids

    def postprocess_batch(self, outputs, batch, cfg: MasterConfig, comps: Components):
        full_texts = (
            comps.tokenizer.batch_decode(outputs, skip_special_tokens=True) if comps.tokenizer is not None else []
        )
        results = []
        for raw, txt in zip(batch, full_texts):
            ans = txt.strip().splitlines()[0].strip()
            if "." in ans:
                ans = ans.split(".", 1)[0].strip() + "."
            results.append({"input": raw, "full_text": txt, "response": ans})
        return results


class VLMTask(BaseTask):
    name = "vlm-generation"

    def _extract_query(self, item: Any) -> str:
        if isinstance(item, dict):
            if "prompt" in item:
                return str(item["prompt"])
            if "text" in item:
                return str(item["text"])
            if "messages" in item and isinstance(item["messages"], list):
                return "\n".join([f"{m.get('role', 'user')}: {m.get('content', '')}" for m in item["messages"]])
        if isinstance(item, str):
            return item
        return json.dumps(item, ensure_ascii=False)

    def _extract_images(self, item: Any) -> List[str]:
        paths = []
        if isinstance(item, dict):
            if "image" in item:
                paths = [item["image"]]
            elif "images" in item and isinstance(item["images"], list):
                paths = item["images"]
        return paths

    def preprocess_batch(self, batch: List[Any], cfg: MasterConfig, comps: Components):
        if comps.processor is None:
            raise RuntimeError("VLM requires a Processor (AutoProcessor). Provide 'processor' or use a VLM model.")
        queries = [self._extract_query(it) for it in batch]
        images_list = [self._extract_images(it) for it in batch]

        from PIL import Image

        enc_list = []
        for query, img_paths in zip(queries, images_list):
            formatted_query = (cfg.template or "{prompt}").format(prompt=query)
            if img_paths:
                pil_imgs = [Image.open(p).convert("RGB") for p in img_paths]
                enc = comps.processor(images=pil_imgs, text=formatted_query, return_tensors="pt")
            else:
                enc = comps.processor(text=formatted_query, return_tensors="pt")

            device = next(comps.model.parameters()).device
            enc = {k: (v.to(device) if hasattr(v, "to") else v) for k, v in enc.items()}
            enc_list.append(enc)
        return enc_list

    def generate(self, enc_list, cfg: MasterConfig, comps: Components):
        outputs = []
        for enc in enc_list:
            gen_kwargs = dict(
                max_new_tokens=cfg.max_new_tokens,
                temperature=cfg.temperature,
                top_k=cfg.top_k,
                top_p=cfg.top_p,
                do_sample=cfg.do_sample,
                repetition_penalty=cfg.repetition_penalty,
                no_repeat_ngram_size=cfg.no_repeat_ngram_size,
            )
            with torch.inference_mode():
                out_ids = comps.model.generate(**enc, **gen_kwargs)
            outputs.append(out_ids)
        return outputs

    def postprocess_batch(self, outputs, batch, cfg: MasterConfig, comps: Components):
        results = []
        for raw, out_ids in zip(batch, outputs):
            if comps.tokenizer is not None:
                txt = comps.tokenizer.decode(out_ids[0], skip_special_tokens=True)
            else:
                try:
                    txt = comps.processor.decode(out_ids[0], skip_special_tokens=True)  # may not exist
                except Exception:
                    txt = str(out_ids[0].tolist())
            assistant = txt.strip().splitlines()[0].strip()
            if "." in assistant:
                assistant = assistant.split(".", 1)[0].strip() + "."
            results.append({"input": raw, "full_text": txt, "response": assistant})
        return results


# Task registry
TASKS: Dict[str, BaseTask] = {
    "text-generation": CausalLMTask(),
    "text2text-generation": Seq2SeqTask(),
    "chat-generation": CausalLMTask(),  # reuse Causal LM
    "vlm-generation": VLMTask(),
}


# -----------------------------------------------------------------------------
# Batch slicing & tokenization
# -----------------------------------------------------------------------------
def slice_encodings(enc_all: Any, start: int, end: int) -> Any:
    if isinstance(enc_all, list):
        return enc_all[start:end]
    if isinstance(enc_all, dict):
        enc_batch = {}
        for k, v in enc_all.items():
            try:
                if isinstance(v, torch.Tensor):
                    enc_batch[k] = v[start:end]
                elif isinstance(v, (list, tuple)):
                    enc_batch[k] = v[start:end]
                else:
                    enc_batch[k] = v
            except Exception:
                enc_batch[k] = v
        return enc_batch
    return enc_all


def safe_tokenize(tokenizer, texts: List[str], **kwargs):
    clean = [t for t in (texts or []) if isinstance(t, str) and t.strip()]
    if not clean:
        raise ValueError("Tokenizer received an empty batch or only empty strings.")
    return tokenizer(clean, **kwargs)


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
def main():
    # Detect a --config path or positional YAML for parse_arguments()
    args = sys.argv[1:]
    config_path = None
    if "--config" in args:
        try:
            idx = args.index("--config")
            config_path = args[idx + 1]
            # Remove --config and its value from args passed to parse_arguments
            args = args[:idx] + args[idx + 2 :]
        except Exception:
            raise ValueError("Expected: --config <path/to/config.yaml>")

    # If first positional is a YAML file, we let parse_arguments handle it
    if not config_path and len(args) == 1 and (args[0].endswith(".yaml") or args[0].endswith(".yml")):
        config_path = os.path.abspath(args[0])
        args = []  # parse_arguments will read YAML only

    # Parse using Transformers HfArgumentParser with your requested function
    cfg = parse_arguments(config_path=config_path, args=args)

    # Load components & items
    comps = load_components(cfg)
    items = load_items_from_config(cfg)

    # Get task
    task = TASKS.get(cfg.task)
    if task is None:
        raise ValueError(f"Unsupported task: {cfg.task}")

    # 1) Preprocess once
    enc_all = task.preprocess_batch(items, cfg, comps)

    # 2) Batch generate + postprocess
    all_rows: List[Dict[str, Any]] = []
    bs = int(cfg.batch_size)

    for i in range(0, len(items), bs):
        batch_items = items[i : i + bs]
        enc_batch = slice_encodings(enc_all, i, i + bs)
        outputs = task.generate(enc_batch, cfg, comps)
        rows = task.postprocess_batch(outputs, batch_items, cfg, comps)
        all_rows.extend(rows)

    # 3) Save results
    save_jsonl(cfg.output, all_rows)

    # 4) Preview
    for i, row in enumerate(all_rows[:10]):
        logger.info(f"\n[{i}] Input:\n{row['input']}\n\nResponse:\n{row['response']}\n")


if __name__ == "__main__":
    main()

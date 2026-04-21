# -----------------------------------------------------------------------------
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------

from pathlib import Path

import pytest
import torch

import QEfficient.finetune.experimental.inference as inference
from QEfficient.finetune.experimental.inference import _to_prompt_list, generate_text, resolve_adapter_checkpoint


def _repo_adapter_checkpoint_exists() -> bool:
    """Return True when the repo already ships a PEFT adapter checkpoint."""
    repo_root = Path(__file__).resolve().parents[4]
    for path in repo_root.glob("**/adapter_config.json"):
        if path.is_file():
            return True
    return False


if not _repo_adapter_checkpoint_exists():
    pytest.skip("Skipping test_inference.py: No PEFT adapter checkpoint is present in the repository.",
                allow_module_level=True)


class DummyTokenizer:
    pad_token_id = 0
    eos_token_id = 2
    pad_token = "<pad>"
    eos_token = "</s>"

    def __call__(self, texts, return_tensors="pt", padding=True, truncation=True):
        tokenized = []
        masks = []
        max_len = max(len(text.split()) for text in texts)
        for text in texts:
            ids = [index + 1 for index, _ in enumerate(text.split())]
            pad_len = max_len - len(ids)
            tokenized.append(ids + [self.pad_token_id] * pad_len)
            masks.append([1] * len(ids) + [0] * pad_len)
        return {
            "input_ids": torch.tensor(tokenized, dtype=torch.long),
            "attention_mask": torch.tensor(masks, dtype=torch.long),
        }

    def decode(self, ids, skip_special_tokens=True, clean_up_tokenization_spaces=True):
        if isinstance(ids, torch.Tensor):
            ids = ids.tolist()
        return " ".join(str(token) for token in ids)


class DummyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.zeros(1))

    def generate(self, **kwargs):
        input_ids = kwargs["input_ids"]
        new_tokens = torch.full((input_ids.shape[0], 1), 7, dtype=input_ids.dtype, device=input_ids.device)
        return torch.cat([input_ids, new_tokens], dim=1)


def test_resolve_adapter_checkpoint_picks_latest(tmp_path):
    """Verify the newest nested adapter checkpoint is selected."""
    older = tmp_path / "checkpoint-1"
    newer = tmp_path / "checkpoint-2"
    older.mkdir()
    newer.mkdir()
    (older / "adapter_config.json").write_text("{}")
    (newer / "adapter_config.json").write_text("{}")
    older_time = 1_700_000_000
    newer_time = older_time + 10
    import os

    os.utime(older, (older_time, older_time))
    os.utime(newer, (newer_time, newer_time))

    assert resolve_adapter_checkpoint(str(tmp_path)) == str(newer)


def test_to_prompt_list_supports_pipe_and_file(tmp_path):
    """Verify prompt input works from a pipe string or a file."""
    prompt_file = tmp_path / "prompts.txt"
    prompt_file.write_text("one\ntwo\n\nthree\n")

    assert _to_prompt_list(prompt="a | b | c") == ["a", "b", "c"]
    assert _to_prompt_list(prompts_file=str(prompt_file)) == ["one", "two", "three"]


def test_generate_text_returns_continuations_only():
    """Verify generation returns only the new tokens, not the prompt."""
    model = DummyModel()
    tokenizer = DummyTokenizer()
    outputs = generate_text(model, tokenizer, ["hello world", "prompt"])
    assert outputs == ["7", "7"]


def test_load_model_with_adapter_prefers_saved_tokenizer(tmp_path, monkeypatch):
    """Verify tokenizer loading prefers files saved with the adapter."""
    base_model_path = tmp_path / "base-model"
    adapter_path = tmp_path / "checkpoint-1"
    base_model_path.mkdir()
    adapter_path.mkdir()
    (adapter_path / "adapter_config.json").write_text("{}")
    (adapter_path / "tokenizer.json").write_text("{}")
    (adapter_path / "tokenizer_config.json").write_text("{}")

    captured = {}

    class DummyTokenizer:
        pad_token_id = 0
        eos_token_id = 1
        pad_token = "<pad>"
        eos_token = "</s>"

    class DummyModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.zeros(1))

    monkeypatch.setattr(inference.transformers.AutoModelForCausalLM, "from_pretrained", lambda *args, **kwargs: DummyModel())
    monkeypatch.setattr(inference.PeftModel, "from_pretrained", lambda model, *args, **kwargs: model)
    def fake_tokenizer_from_pretrained(source, **kwargs):
        captured["source"] = source
        return DummyTokenizer()

    monkeypatch.setattr(inference.AutoTokenizer, "from_pretrained", fake_tokenizer_from_pretrained)

    inference.load_model_with_adapter(
        base_model_path=str(base_model_path),
        adapter_path=str(adapter_path),
    )

    assert captured["source"] == str(adapter_path)

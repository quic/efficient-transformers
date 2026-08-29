#!/usr/bin/env python3
"""One-shot Qwen3-ASR QPC example.

This intentionally reuses qpc_only_qwen3_asr.py so the example and the QPC-only
test exercise the same runtime path.
"""

from __future__ import annotations

import sys
from pathlib import Path

from qpc_only_qwen3_asr import main as qpc_only_main
from qpc_only_qwen3_asr import repo_root


DEFAULT_DEVICE_IDS = "0,1,2,3"
DEFAULT_GENERATION_LEN = "20"
DEFAULT_MANIFEST = (
    ".archon/artifacts/runs/qwen3_asr/"
    "regenerated-qpc/20260805T133135Z/qpc_manifest.json"
)


def existing_manifest_arg(root: Path) -> list[str]:
    manifest = root / DEFAULT_MANIFEST
    if manifest.is_file():
        return ["--manifest", str(manifest)]

    return []


def default_audio_file(root: Path) -> Path:
    return root / "audio_test.flac"


def main() -> int:
    root = repo_root()
    default_args = [
        *existing_manifest_arg(root),
        "--audio-file",
        str(default_audio_file(root)),
        "--generation-len",
        DEFAULT_GENERATION_LEN,
        "--device-ids",
        DEFAULT_DEVICE_IDS,
        "--no-stop-on-eos",
    ]

    sys.argv = [sys.argv[0], *default_args, *sys.argv[1:]]
    return qpc_only_main()


if __name__ == "__main__":
    raise SystemExit(main())

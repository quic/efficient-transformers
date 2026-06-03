"""CSV experiment logging helpers for auto-perf skill."""

from datetime import datetime, timezone
from pathlib import Path

import pandas as pd


LOG_COLUMNS = [
    "timestamp",
    "ttft_seconds",
    "decode_tokens_per_sec",
    "accuracy_match",
    "summary_of_changes",
]


def ensure_log_file(log_path: str) -> Path:
    p = Path(log_path).expanduser()
    if not p.is_absolute():
        p = (Path.cwd() / p).resolve()
    p.parent.mkdir(parents=True, exist_ok=True)
    if not p.exists():
        pd.DataFrame(columns=LOG_COLUMNS).to_csv(p, index=False)
    return p


def append_experiment_log(
    log_csv: Path,
    ttft_seconds: float,
    decode_tokens_per_sec: float,
    accuracy_match: bool,
    summary_of_changes: str,
) -> None:
    row = pd.DataFrame(
        [
            {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "ttft_seconds": ttft_seconds,
                "decode_tokens_per_sec": decode_tokens_per_sec,
                "accuracy_match": accuracy_match,
                "summary_of_changes": summary_of_changes,
            }
        ],
        columns=LOG_COLUMNS,
    )
    row.to_csv(log_csv, mode="a", header=False, index=False)

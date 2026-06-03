# Experiment Log Template

Use a single append-only CSV file backed by pandas.

## Required Schema (exact columns only)
- `timestamp`
- `ttft_seconds`
- `decode_tokens_per_sec`
- `accuracy_match`
- `summary_of_changes`

## Notes
- Create parent directories and the CSV file if they do not exist.
- Initialize new files with an empty pandas DataFrame using the required schema.
- Append one row per baseline/experiment attempt to the same file.

## Example Row
```csv
timestamp,ttft_seconds,decode_tokens_per_sec,accuracy_match,summary_of_changes
2026-03-17T06:30:00+00:00,0.246,119.8,true,fused rotary ops and removed redundant transpose
```

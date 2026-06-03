#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="${REPO_DIR:-$(git rev-parse --show-toplevel 2>/dev/null || pwd)}"
INTENDED_FILE_1="QEfficient/transformers/models/llama/modeling_llama.py"
INTENDED_FILE_2="QEfficient/transformers/models/pytorch_transforms.py"

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 \"<commit-message>\""
  exit 2
fi

COMMIT_MESSAGE="$1"

if [[ ! -d "${REPO_DIR}/.git" ]]; then
  echo "Error: git repo not found at ${REPO_DIR}"
  exit 1
fi

CURRENT_BRANCH="$(git -C "${REPO_DIR}" rev-parse --abbrev-ref HEAD)"
if [[ -z "${CURRENT_BRANCH}" || "${CURRENT_BRANCH}" == "HEAD" ]]; then
  echo "Error: detached HEAD; checkout a branch before pushing."
  exit 1
fi

git -C "${REPO_DIR}" add "${INTENDED_FILE_1}" "${INTENDED_FILE_2}"

if git -C "${REPO_DIR}" diff --cached --quiet -- "${INTENDED_FILE_1}" "${INTENDED_FILE_2}"; then
  echo "No staged changes found in intended files."
  exit 1
fi

git -C "${REPO_DIR}" commit -m "${COMMIT_MESSAGE}" -- "${INTENDED_FILE_1}" "${INTENDED_FILE_2}"


echo "Committed intended files on branch: ${CURRENT_BRANCH}"

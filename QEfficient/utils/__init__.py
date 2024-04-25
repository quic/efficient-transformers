# -----------------------------------------------------------------------------
#
# Copyright (c)  2023-2024 Qualcomm Innovation Center, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------

import os
import requests
from typing import Optional

from requests.exceptions import HTTPError
from huggingface_hub import snapshot_download


def hf_download(
    repo_id: Optional[str] = None,
    cache_dir: Optional[str] = None,
    hf_token: Optional[str] = None,
):
    # Setup cache and local dir
    local_dir = None
    if cache_dir is not None:
        cache_dir = f"{cache_dir}"
        local_dir = f"{cache_dir}/{repo_id}"

    os.makedirs(f"{cache_dir}/{repo_id}", exist_ok=True)
    max_retries = 5
    retry_count = 0
    while retry_count < max_retries:
        try:
            model_path = snapshot_download(
                repo_id,
                cache_dir=cache_dir,
                local_dir=local_dir,
                local_dir_use_symlinks=True,
                revision="main",
                resume_download=True,
                token=hf_token,
                ignore_patterns=["*.txt", "*.msgpack", "*.h5", "*.onnx", "*.ot", "*.md", "*.tflite", "*.pdf"],
            )
            break
        except requests.ReadTimeout as e:
            print(f"Read timeout: {e}")
            retry_count += 1

        except HTTPError as e:
            retry_count = max_retries
            if e.response.status_code == 401:
                print("You need to pass a valid `--hf_token=...` to download private checkpoints.")
            else:
                raise e

    return model_path

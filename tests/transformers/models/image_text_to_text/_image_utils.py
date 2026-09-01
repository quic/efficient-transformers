# -----------------------------------------------------------------------------
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
#
# ----------------------------------------------------------------------------

from io import BytesIO
from pathlib import Path

import requests
from PIL import Image
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

_ASSET_DIR = Path(__file__).resolve().parent / "assets"
_LOCAL_IMAGE_PATHS = {
    "https://picsum.photos/id/237/536/354": _ASSET_DIR / "picsum-237-536x354.jpg",
    "https://picsum.photos/id/238/536/354": _ASSET_DIR / "picsum-238-536x354.jpg",
}
LOCAL_TEST_IMAGE_URLS = tuple(_LOCAL_IMAGE_PATHS)

_SESSION = requests.Session()
_SESSION.mount(
    "https://",
    HTTPAdapter(
        max_retries=Retry(
            total=3,
            backoff_factor=1,
            status_forcelist=(429, 500, 502, 503, 504),
        )
    ),
)


def load_test_image(image_url: str) -> Image.Image:
    """Load a deterministic local snapshot when one exists, otherwise fetch the image safely."""
    image_path = _LOCAL_IMAGE_PATHS.get(image_url)
    if image_path is not None:
        with Image.open(image_path) as image:
            return image.convert("RGB")

    response = _SESSION.get(image_url, timeout=30)
    response.raise_for_status()
    content_type = response.headers.get("Content-Type", "")
    if not content_type.lower().startswith("image/"):
        raise ValueError(
            f"Expected an image response from {image_url}, got HTTP {response.status_code} "
            f"with Content-Type {content_type!r}"
        )
    with Image.open(BytesIO(response.content)) as image:
        return image.convert("RGB")

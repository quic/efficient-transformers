# -----------------------------------------------------------------------------
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------

import warnings

_DEPRECATION_WARNING_EMITTED = set()


def warn_deprecated_cloud_api(api_name: str, *, force: bool = False) -> None:
    """Warn once when a deprecated cloud API is used."""
    if api_name in _DEPRECATION_WARNING_EMITTED:
        return

    message = (
        f"QEfficient.cloud.{api_name} is deprecated. These APIs will no longer be usable and will be removed "
        "after the 1.23.0 release."
    )
    if force:
        with warnings.catch_warnings():
            warnings.simplefilter("default", FutureWarning)
            warnings.warn(message, FutureWarning, stacklevel=3)
    else:
        warnings.warn(message, FutureWarning, stacklevel=3)
    _DEPRECATION_WARNING_EMITTED.add(api_name)

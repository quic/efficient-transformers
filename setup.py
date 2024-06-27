# -----------------------------------------------------------------------------
#
# Copyright (c)  2023-2024 Qualcomm Innovation Center, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------

import re

from setuptools import setup

# Ensure we match the version set in QEfficient/__init__.py
filepath = "QEfficient/__init__.py"
with open(filepath) as version_file:
    (version,) = re.findall('__version__ = "(.*)"', version_file.read())

setup(version=version)

# -----------------------------------------------------------------------------
#
# Copyright (c)  2023-2024 Qualcomm Innovation Center, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------

import re

from setuptools import find_namespace_packages, setup

# Ensure we match the version set in QEfficient//version.py
try:
    filepath = "QEfficient/version.py"
    with open(filepath) as version_file:
        (__version__,) = re.findall('__version__ = "(.*)"', version_file.read())
except Exception as error:
    assert False, "Error: Could not open '%s' due %s\n" % (filepath, error)

INSTALL_REQUIRES = [
    "transformers==4.36.2",
    "torch@https://download.pytorch.org/whl/cpu/torch-2.0.0%2Bcpu-cp38-cp38-linux_x86_64.whl#sha256=354f281351cddb590990089eced60f866726415f7b287db5105514aa3c5f71ca",
    "datasets==2.7.0",
    "fsspec==2023.6.0",
    "multidict==6.0.4",
    "urllib3<2",
    "sentencepiece==0.1.98",
    "onnx==1.15.0",
    "onnxruntime==1.12.1",
    "numpy==1.23.0",
    "protobuf==3.20.2",
    "onnxscript==0.1.0.dev20240327",
    "pytest",
    "sympy",
]

QUALITY_REQUIRES = [
    "black",
    "ruff",
    "hf_doc_builder @ git+https://github.com/huggingface/doc-builder.git",
]

EXTRAS_REQUIRE = {
    "quality": QUALITY_REQUIRES,
}

setup(
    name="QEfficient Library",
    version=__version__,
    description=(
        "QEfficient is the library interface for the Hugging Face Transformer"
        "models for efficient inference on Qualcomm Cloud AI 100"
    ),
    long_description=open("README.md", "r", encoding="utf-8").read(),
    long_description_content_type="text/markdown",
    classifiers=[
        "Programming Language :: Python :: 3",
        "Development Status :: 5 - Development/Unstable",
        "Intended Audience :: Developers",
        "Intended Audience :: Education",
        "Operating System :: Linux",
        "Programming Language :: Python :: 3.8",
        "Topic :: Scientific/Engineering :: Artificial Intelligence for Inference Accelerator",
    ],
    keywords="transformers, Cloud AI 100, Inference",
    author="Qualcomm Cloud AI ML Team",
    license="BSD-3-Clause",
    packages=find_namespace_packages(include=["QEfficient*"]),
    install_requires=INSTALL_REQUIRES,
    extras_require=EXTRAS_REQUIRE,
    include_package_data=True,
    zip_safe=False,
    dependency_links=[
        "https://download.pytorch.org/whl/torch_stable.html",
        "https://download.pytorch.org/whl/cpu",
    ],
)

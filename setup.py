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

<<<<<<< HEAD
INSTALL_REQUIRES = [
    "transformers==4.41.2",
    "torch@https://download.pytorch.org/whl/cpu/torch-2.0.0%2Bcpu-cp38-cp38-linux_x86_64.whl#sha256=354f281351cddb590990089eced60f866726415f7b287db5105514aa3c5f71ca",
    "datasets==2.7.0",
    "fsspec==2023.6.0",
    "multidict==6.0.4",
    "urllib3<2",
    "sentencepiece==0.1.98",
    "onnx==1.16.0",
    "onnxruntime==1.16.3",
    "numpy==1.23.0",
    "protobuf==3.20.2",
    "onnxscript==0.1.0.dev20240327",
    "sympy",
]

QUALITY_REQUIRES = [
    "black",
    "ruff",
    "hf_doc_builder @ git+https://github.com/huggingface/doc-builder.git",
]

PYTEST_REQUIRES = [
    "pytest",
    "pytest-mock",
]

EXTRAS_REQUIRE = {
    "quality": QUALITY_REQUIRES,
    "test" : PYTEST_REQUIRES,
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
=======
setup(version=version)
>>>>>>> 30ce13b8426fae66ad8148e17a5776a221183049

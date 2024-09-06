import re
import shutil
from pathlib import Path

from setuptools import find_packages, setup
from setuptools.command.install import install


# Remove stale QEfficient.egg-info directory to avoid issues
stale_egg_info = Path(__file__).parent / "QEfficient.egg-info"
if stale_egg_info.exists():
    print(f"Removing stale {stale_egg_info}")
    shutil.rmtree(stale_egg_info)

_deps = [
    "filelock",
    "numpy>=1.23.0",
    "packaging",
    "protobuf==3.20.2",
    "pyyaml",
    "regex!=2019.12.17",
    "requests",
    "tqdm",
    "transformers==4.44.2",
    "torch==2.0.0+cpu",
    "peft==0.12.0",
    "datasets==2.7.0",
    "fsspec==2023.6.0",
    "multidict==6.0.4",
    "urllib3<2",
    "sentencepiece==0.1.98",
    "onnx==1.16.0",
    "onnxruntime==1.16.3",
    "onnxscript==0.1.0.dev20240327",
    "sympy",
]

deps = {b: a for a, b in (re.findall(r"^(([^!=<>~ ]+)(?:[!=<>~ ].*)?$)", x)[0] for x in _deps)}


def deps_list(*pkgs):
    return [deps[pkg] for pkg in pkgs]


extras = {}
extras["test"] = ["pytest", "pytest-mock"]
extras["docs"] = ["Sphinx==7.1.2", "sphinx-rtd-theme==2.0.0", "myst-parser==3.0.1"]
extras["quality"] = ["black", "ruff", "hf_doc_builder@git+https://github.com/huggingface/doc-builder.git"]

install_requires = [
    deps["filelock"],
    deps["numpy"],
    deps["packaging"],
    deps["protobuf"],
    deps["pyyaml"],
    deps["regex"],
    deps["requests"],
    deps["tqdm"],
    deps["torch"],  # Specify CPU version
    deps["transformers"],
    deps["peft"],
    deps["datasets"],
    deps["fsspec"],
    deps["multidict"],
    deps["urllib3"],
    deps["sentencepiece"],
    deps["onnx"],
    deps["onnxruntime"],
    deps["onnxscript"],
    deps["sympy"],
]


class CustomInstallCommand(install):
    def run(self):
        self.distribution.install_requires = install_requires
        install.run(self)


setup(
    name="QEfficient",
    version="0.1.0",
    author="Qualcomm Cloud AI ML Team",
    author_email="your_email@example.com",
    description="QEfficient is the library interface for the Hugging Face Transformer models for efficient inference on Qualcomm Cloud AI 100",
    long_description=open("README.md", "r", encoding="utf-8").read(),
    long_description_content_type="text/markdown",
    keywords="NLP vision speech deep learning transformer inference Qualcomm Cloud AI 100",
    license="Apache 2.0 License",
    url="https://github.com/quic/efficient-transformers/",
    packages=find_packages(),
    include_package_data=True,
    zip_safe=False,
    extras_require=extras,
    python_requires=">=3.8.0",
    install_requires=install_requires,
    cmdclass={
        "install": CustomInstallCommand,
    },
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Intended Audience :: Developers",
        "Intended Audience :: Education",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
)

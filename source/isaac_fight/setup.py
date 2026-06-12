# Copyright (c) 2026, Isaac Fight contributors.
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from pathlib import Path

from setuptools import find_packages, setup

ROOT = Path(__file__).resolve().parent

setup(
    name="isaac_fight",
    version="0.1.0",
    author="Isaac Fight contributors",
    description="Standalone Isaac Lab extension for Unitree humanoid 1v1 combat self-play.",
    long_description=(ROOT.parent.parent / "README.md").read_text(encoding="utf-8")
    if (ROOT.parent.parent / "README.md").exists()
    else "Isaac Fight",
    long_description_content_type="text/markdown",
    url="https://github.com/numan-ai/isaac-fight",
    license="BSD-3-Clause",
    packages=find_packages(),
    include_package_data=True,
    python_requires=">=3.10",
    install_requires=[
        "gymnasium>=0.29.0",
        "numpy>=1.24",
        "torch>=2.0",
        "packaging>=23.0",
        "pyyaml>=6.0",
    ],
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: BSD License",
        "Programming Language :: Python :: 3",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
)

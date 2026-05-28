#!/usr/bin/env python
# -*- coding: utf-8 -*-

from pathlib import Path

from setuptools import find_packages, setup


def load_requirements(file_name: str = "requirements.txt"):
    req_path = Path(__file__).parent / file_name
    if req_path.exists():
        with open(req_path, encoding="utf-8") as f:
            return [
                line.strip()
                for line in f
                if line.strip() and not line.strip().startswith("#")
            ]
    return []


setup(
    name="rex_omni",
    version="1.0.0",
    author="IDEA Research, Qing Jiang",
    author_email="jiangqing@idea.edu.cn",
    description="A high-level wrapper for Rex-Omni multimodal language model supporting various vision tasks",
    packages=find_packages(),
    python_requires=">=3.8",
    # install_requires=load_requirements(),
)

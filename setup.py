"""
Multi-Teacher LLM Distillation Engine

A production-ready, fully automated engine for knowledge distillation 
from multiple large language models into compact student models.
"""

from setuptools import setup, find_packages

setup(
    name="multi-teacher-distillation",
    version="1.0.0",
    author="Multi-Teacher LLM Distillation Engine Contributors",
    description="Automated engine for multi-teacher knowledge distillation",
    long_description=open("README.md", encoding="utf-8").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/harsh2025-sketch/multi_teacher_llm_distillation_engine",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: Apache Software License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires=">=3.10",
    install_requires=[
        "torch>=2.0.0",
        "transformers>=4.35.0",
        "accelerate>=0.24.0",
        "bitsandbytes>=0.41.0",
        "datasets>=2.14.0",
        "pandas>=2.0.0",
        "pyarrow>=13.0.0",
        "pyyaml>=6.0",
        "dataclasses-json>=0.6.0",
        "tqdm>=4.65.0",
        "tensorboard>=2.14.0",
        "huggingface-hub>=0.17.0",
        "safetensors>=0.4.0",
        "numpy>=1.24.0",
        "scipy>=1.11.0",
    ],
    extras_require={
        "webui": ["gradio>=4.0.0"],
        "dev": [
            "pytest>=7.4.0",
            "black>=23.0.0",
            "flake8>=6.0.0",
            "mypy>=1.0.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "distill-cli=src.cli:main",
            "distill-webui=src.web_ui:main",
        ],
    },
    include_package_data=True,
    zip_safe=False,
)

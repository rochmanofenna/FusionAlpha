#!/usr/bin/env python
"""
FusionAlpha Setup Script
"""

from setuptools import setup, find_packages
import os

# Read the README file
with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

# Read requirements
with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="fusionalpha",
    version="1.0.0",
    author="FusionAlpha Team",
    description="Market Contradiction Detection Trading System",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/FusionAlpha",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Financial and Insurance Industry",
        "Topic :: Office/Business :: Financial :: Investment",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    entry_points={
        "console_scripts": [
            "fusionalpha-backtest=fusion_alpha.pipelines.run_pipeline:main",
            "fusionalpha-paper=paper_trading.simulation:main",
            "fusionalpha-live=fusion_alpha.routers.live_trading:main",
        ],
    },
    include_package_data=True,
    package_data={
        "fusion_alpha": ["config/*.py"],
        "artifacts": ["*.pth", "*.db"],
    },
)
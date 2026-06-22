"""Compatibility shim for environments with pre-PEP-660 editable installs."""

from setuptools import find_packages, setup

setup(
    name="entity-bias-audit",
    version="0.1.0",
    package_dir={"": "src"},
    packages=find_packages("src"),
    python_requires=">=3.10",
    install_requires=["numpy>=1.24", "pandas>=2.0", "requests>=2.28", "scikit-learn>=1.2", "tqdm>=4.65"],
    extras_require={
        "dev": ["pytest>=7.4"],
        "inference": ["vllm"],
        "analysis": ["matplotlib>=3.7"],
    },
    entry_points={"console_scripts": ["bias-audit=bias_audit.cli:main"]},
)

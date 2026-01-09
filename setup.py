"""Setup script for anomaly-detection-toolkit."""

from setuptools import find_packages, setup

# Read the README file
with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="anomaly-detection-toolkit",
    version="0.1.1",
    author="Kyle Jones",
    author_email="kyletjones@gmail.com",
    description=(
        "A comprehensive Python library for detecting anomalies in time series "
        "and multivariate data"
    ),
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/kylejones200/anomaly-detection-toolkit",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.12",
        "Topic :: Scientific/Engineering",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    python_requires=">=3.12",
    install_requires=[
        "numpy>=1.20.0",
        "pandas>=1.3.0",
        "scikit-learn>=1.0.0",
        "PyWavelets>=1.3.0",
    ],
    extras_require={
        "deep": [
            "torch>=1.9.0",
            "tensorflow>=2.6.0",
        ],
        "all": [
            "torch>=1.9.0",
            "tensorflow>=2.6.0",
        ],
    },
)

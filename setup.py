from setuptools import setup, find_packages

with open("readme.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = fh.read().splitlines()

setup(
    name="quantitative-trading",
    version="0.1.0",
    author="Your Name",
    author_email="your.email@example.com",
    description="A comprehensive quantitative trading system with deep learning models",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/Quantitative-Trading",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Topic :: Office/Business :: Financial :: Investment",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires=">=3.7",
    install_requires=requirements,
    entry_points={
        "console_scripts": [
            "quant-trading=src.main:main",
        ],
    },
)
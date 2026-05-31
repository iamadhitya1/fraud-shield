from setuptools import setup, find_packages

setup(
    name="fraud-shield",
    version="1.0.0",
    author="M Adhitya",
    description="Train, save, and run fraud detection on transaction data. One class. Clean API.",
    long_description=open("README.md", encoding="utf-8").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/iamadhitya1/fraud-shield",
    packages=find_packages(),
    install_requires=[
        "scikit-learn>=1.0.0",
        "pandas>=1.3.0",
        "numpy>=1.21.0",
    ],
    python_requires=">=3.9",
    license="MIT",
    keywords=["fraud", "detection", "machine-learning", "random-forest", "finance", "classification"],
    classifiers=[
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Office/Business :: Financial",
    ],
)

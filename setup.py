import os
from glob import glob
from setuptools import setup, find_packages

dir = os.path.abspath(os.path.dirname(__file__))
with open(os.path.join(dir, "README.md")) as f:
    long_description = f.read()

setup(
    name="degrade",
    version="0.1.5",
    author="Samuel W. Remedios",
    description="Degrade a signal by blurring and downsampling",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author_email="samuel.remedios@jhu.edu",
    url="https://gitlab.com/iacl/degrade",
    license="MIT",
    packages=find_packages(),
    install_requires=[
        "numpy",
        "pytest",
        "pytest-cov",
        "scipy",
        "torch>=1.10.0",
        "sigpy",
        "resize[scipy] @ git+https://gitlab.com/shan-utils/resize",
    ],
    python_requires=">=3.7",
    include_package_data=True,
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)

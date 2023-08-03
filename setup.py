import os
from setuptools import setup, find_packages

pkg_dir = os.path.abspath(os.path.dirname(__file__))
with open(os.path.join(pkg_dir, "README.md")) as f:
    long_description = f.read()

setup(
    name="degrade",
    version="0.2",
    author="Samuel W. Remedios",
    description="Degrade a signal by blurring and downsampling",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author_email="samuel.remedios@jhu.edu",
    url="https://gitlab.com/iacl/degrade",
    license="MIT",
    packages=find_packages(),
    install_requires=[
        "numpy==1.23.5",
        "scipy",
        "torch>=1.10.0",
        "sigpy",
        "resize @ git+https://gitlab.com/shan-utils/resize@0.1.3",
        "nibabel",
        "transforms3d",
    ],
    extras_require={
        "dev": [
            "pytest",
            "pytest-cov",
        ]
    },
    entry_points={
        "console_scripts": [
            "apply-degrade=degrade.main:main",
            "remove-3D-inplane-interp=degrade.downsample_3D_inplane:main",
        ]
    },
    python_requires=">=3.7",
    include_package_data=True,
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)

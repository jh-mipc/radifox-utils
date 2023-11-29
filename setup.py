from pathlib import Path
from setuptools import setup, find_namespace_packages


__package_name__ = "radifox-utils"


def get_version_and_cmdclass(pkg_path):
    """Load version.py module without importing the whole package.

    Template code from miniver
    """
    import os
    from importlib.util import module_from_spec, spec_from_file_location

    spec = spec_from_file_location("version", os.path.join(pkg_path, "_version.py"))
    module = module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.__version__, module.get_cmdclass(pkg_path)


__version__, cmdclass = get_version_and_cmdclass('radifox/utils')

setup(
    name=__package_name__,
    version=__version__,
    cmdclass=cmdclass,
    author="JH-MIPC",
    author_email="jhmipc@jh.edu",
    url="https://github.com/jh-mipc/radifox-utils",
    description="Medical image utilities provided with RADIFOX",
    long_description=(Path(__file__).parent.resolve() / "README.md").read_text(),
    long_description_content_type="text/markdown",
    license="Apache License, 2.0",
    packages=find_namespace_packages(include=['radifox.*']),
    entry_points={
        "console_scripts": [
            "apply-degrade=radifox.utils.degrade.main:main",
            "resample-inplane-res=radifox.utils.degrade.inplane_res:main",
        ]
    },
    install_requires=[
        "nibabel",
        "numpy",
        "scipy",
        "sigpy-lite",
        "transforms3d",
    ],
    extras_require={
        "dev": [
            "pytest",
            "pytest-cov",
        ],
        "pytorch": "torch>=1.10.0",
    },
    python_requires=">=3.7",
    include_package_data=True,
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
    ],
)

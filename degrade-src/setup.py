import os
from setuptools import setup, find_packages


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


version, cmdclass = get_version_and_cmdclass(r"degrade")


pkg_dir = os.path.abspath(os.path.dirname(__file__))
with open(os.path.join(pkg_dir, "README.md")) as f:
    long_description = f.read()

setup(
    name="degrade",
    version=version,
    cmdclass=cmdclass,
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
        "sigpy",
        "resize @ git+https://gitlab.com/iacl/resize@v0.3.0",
        "nibabel",
        "transforms3d",
    ],
    extras_require={
        "dev": [
            "pytest",
            "pytest-cov",
        ],
        "pytorch": ["torch>=1.10.0"],
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

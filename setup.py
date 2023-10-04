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


version, cmdclass = get_version_and_cmdclass("resize")

with open(os.path.join(os.path.abspath(os.path.dirname(__file__)), "README.md")) as f:
    long_description = f.read()

setup(
    name="resize",
    version=version,
    cmdclass=cmdclass,
    author="IACL",
    description="Resize an image with correct sampling coordinates.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author_email="blake.dewey@jhu.edu",
    url="https://github.com/iacl/resize",
    license="MIT",
    packages=find_packages(),
    install_requires=[
        "numpy",
        "scipy",
        "transforms3d",
    ],
    extras_require={
        "pytorch": "torch>=1.10.0",
    },
    python_requires=">=3.7",
    include_package_data=True,
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)

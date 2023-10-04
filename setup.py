import os
from glob import glob
from setuptools import setup, find_packages

dir = os.path.abspath(os.path.dirname(__file__))
with open(os.path.join(dir, 'README.md')) as f:
    long_description = f.read()

setup(name='resize',
      version='0.2.0',
      author='IACL',
      description='Resize an image with correct sampling coordinates.',
      long_description=long_description,
      long_description_content_type='text/markdown',
      author_email='blake.dewey@jhu.edu',
      url='https://github.com/iacl/resize',
      license='MIT',
      packages=find_packages(),
      install_requires=[
            'numpy',
            'scipy',
            'transforms3d',
      ],
      extras_require={
            'pytorch': 'torch>=1.10.0',
      },
      python_requires='>=3.7',
      include_package_data=True,
      classifiers=['Programming Language :: Python :: 3',
                   'License :: OSI Approved :: MIT License',
                   'Operating System :: OS Independent']
)

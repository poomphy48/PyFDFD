from setuptools import find_packages, setup

import pyfdfd

setup(
    name="pyfdfd",
    version=pyfdfd.__version__,
    description="Finite diff ff",
    author=pyfdfd.__author__,
    author_email=pyfdfd.__email__,
    packages=find_packages(),
    install_requires=["scipy>=1.14.1", "numpy>=2.1.1"],
)
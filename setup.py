from setuptools import find_packages
from setuptools import setup

import DIPLOMv1


setup(
    author="Pupysheva Polina",
    description="Implementation incremental SVD with user and item offsets",
    install_requires=[
        "numpy>=1.16.4 ",
        "numba>=0.38.0",
        "pandas>=0.25.0"
    ],
    name="DIPLOMv1",
    packages=find_packages(),
    python_requires=">=3.6.9",
    version=DIPLOMv1.__version__,
)
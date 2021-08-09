"""

    Setup

"""
from typing import List

from setuptools import find_packages, setup


def _get_requirements() -> List[str]:
    with open("requirements.txt", "r") as f:
        return [i.strip().strip("\n") for i in f.readlines()]


setup(
    packages=find_packages(exclude=["tests", "experiments"]),
    install_requires=_get_requirements(),
)

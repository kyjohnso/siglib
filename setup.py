#!/usr/bin/env python
from setuptools import setup, find_packages


with open("README.rst") as readme_file:
    readme = readme_file.read()


with open("requirements.txt") as requirements_file:
    requirements = [line.strip() for line in requirements_file.readlines()]


setup(
    name="siglib",
    version="0.1.0",
    description="",
    long_description=readme,
    url="https://github.com/kyjohnso/siglib",
    maintainer="Kyle Johnson",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    install_requires=requirements,
)

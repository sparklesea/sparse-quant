#!/usr/bin/env python

from setuptools import setup
import setuptools
from torch.utils.cpp_extension import BuildExtension, CUDAExtension, CppExtension

setup(
    name="playground",
    version="0.2.0",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
)

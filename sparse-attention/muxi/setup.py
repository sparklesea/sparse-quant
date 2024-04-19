#!/usr/bin/env python

from setuptools import setup
import setuptools
from torch.utils.cpp_extension import BuildExtension, CUDAExtension, CppExtension

extra_compile_args = {
    "cxx": ["-g", "-O3", "-fopenmp", "-lgomp"],
    "mxcc": ["-x maca"],
}

setup(
    name="playground",
    version="0.2.0",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
    ext_modules=[
        CUDAExtension(
            name="block_sparse_ops",
            sources=[
                "playground/kernels/csrc/pybind.cpp",
                "playground/kernels/csrc/src/block_sparse_attention_lut.cu",
            ],
            extra_compile_args=extra_compile_args,
        ),
    ],
    cmdclass={
        'build_ext':
        BuildExtension.with_options(no_python_abi_suffix=True, use_ninja=False)
    },
)

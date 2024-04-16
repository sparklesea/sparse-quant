#!/usr/bin/env python

from setuptools import setup
import setuptools
from torch.utils.cpp_extension import BuildExtension, CUDAExtension, CppExtension

# extra_compile_args = {
#     "cxx": ["-g", "-O3", "-fopenmp", "-lgomp", "-std=c++17"],
#     "nvcc": ["-O3", "-std=c++17", "--generate-line-info", "-Xptxas=-v",
#              "-U__CUDA_NO_HALF_OPERATORS__", "-U__CUDA_NO_HALF_CONVERSIONS__"],
# }
extra_compile_args = {
    "cxx": ["-g", "-O3", "-fopenmp", "-lgomp"],
    "nvcc": ["-O3"],
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
                # "csrc/pybind.cpp", 
                # "csrc/gemm/gemm_cuda_gen.cu",
                # "csrc/gemm/dequant_op.cu", 
                # "csrc/gemm/dequant_gemv.cu",
                # "csrc/gemm/dequant_gemm.cu",
                # "csrc/gemm/dequant_gemv_quant.cu",
                # "csrc/gemm/dequant_batch_gemv.cu",
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

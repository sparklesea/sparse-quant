import os
import setuptools
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CppExtension

sources = [os.path.join('backend', f'pybind.cpp'),
            os.path.join('backend', f'gemm_quant.cu'), 
            # os.path.join('backend', f'ut.cu'),
            ]

# os.environ["CC"] = "mxcc"
# os.environ["CXX"] = "mxcc"

setup(
    name='quant',
    version='0.0.1',
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
    ext_modules=[
        CppExtension('quant',
            sources=sources,
            # extra_compile_args = ['-g', '-O3', '-fopenmp', '-lgomp'],
            extra_compile_args = {
                'cxx': ['-g', '-O3', '-fopenmp', '-lgomp'],
                'clang++': ['-O3', '-lcudart', '-std=c++11', '--cuda-gpu-arch=ivcore11', '-lcudart', '-L /usr/lib/gcc/x86_64-linux-gnu/11/']},
            include_dirs=['/usr/local/corex/include'],
        ),
    ],
    cmdclass={
        'build_ext': BuildExtension.with_options(no_python_abi_suffix=True, use_ninja=False)
    })

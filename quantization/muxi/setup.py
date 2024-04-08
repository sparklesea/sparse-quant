import os
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
    ext_modules=[
        CppExtension('quant',
            sources=sources,
            # extra_compile_args = ['-g', '-O3', '-fopenmp', '-lgomp'],
            extra_compile_args = {
                'cxx': ['-O3', '-fopenmp', '-lgomp'],
                '/opt/cu-bridge/CUDA_DIR/bin/nvcc': ['-O3', '--generate-line-info', '-Xptxas=-v','-U__CUDA_NO_HALF_OPERATORS__']},
            include_dirs=["/opt/cu-bridge/CUDA_DIR/include", "/opt/maca/tools/cu-bridge/include"],
        ),
    ],
    cmdclass={
        'build_ext': BuildExtension
    })

import os
import setuptools
import multiprocessing
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CppExtension, CUDAExtension

from setuptools.command.install import install

sources = [os.path.join('backend', f'pybind.cpp'),
            os.path.join('backend', f'gemm_quant.cu'), 
            # os.path.join('backend', f'ut.cu'),
            ]

current_directory = os.path.abspath(os.path.dirname(__file__))

class CustomBuildExt(BuildExtension):
    def build_extensions(self):
        num_cores = multiprocessing.cpu_count()

        for ext in self.extensions:
            ext.extra_compile_args = ["-j", str(num_cores)]  # 使用-j选项设置线程数
        super().build_extensions()

class CustomInstallCommand(install):
    def run(self):
        install.run(self)
        print("Running custom install command...")

setup(
    name='myquant',
    version='0.0.1',
    ext_modules=[
        CUDAExtension(
            'myquant',
            sources=sources,
            extra_compile_args={
                "cxx": [
                    "-O3",
                    "-fopenmp",
                    "-lgomp",
                    "-DBUILD_WITH_CUDA",
                ],
                "nvcc": [
                    "-O3",
                    "-std=c++17",
                    "--generate-line-info",
                    "-Xptxas=-v",
                    "-U__CUDA_NO_HALF_OPERATORS__",
                ],
            },
        ),
    ],
    cmdclass={
        'build_ext': BuildExtension,
        "install": CustomInstallCommand,
    })

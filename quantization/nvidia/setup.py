import os
import glob
import multiprocessing
from setuptools import setup

from torch.utils.cpp_extension import (
    BuildExtension,
    CUDAExtension,
)

from setuptools.command.install import install

name = "quant"
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

        # try:
        #     shutil.rmtree('build')
        #     print('Deleted build directory.')
        # except Exception as e:
        #     print(f"Error deleting build directory: {e}")

        # try:
        #     shutil.rmtree(f'{name}.egg-info')
        #     print(f'Deleted {name}.egg-info directory.')
        # except Exception as e:
        #     print(f"Error deleting {name}.egg-info directory: {e}")


ext_modules = []


def build_for_cuda():
    sources = [
        os.path.join("kernel", "pybind.cpp"),
        os.path.join("kernel", "gemm_s4_f16", f"format.cu"),
        os.path.join("kernel", "gemm_s4_f16", f"gemm_s4_f16.cu"),
        os.path.join("kernel", f"gemm_quant.cu"),
    ]

    ext_modules.append(
        CUDAExtension(
            name,
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
    )

build_for_cuda()

setup(
    name=name,
    version="0.1.0",
    ext_modules=ext_modules,
    cmdclass={
        "build_ext": BuildExtension,
        "install": CustomInstallCommand,
    },
)


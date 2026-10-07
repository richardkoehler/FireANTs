# Copyright (c) 2026 Rohit Jena. All rights reserved.
# 
# This file is part of FireANTs, distributed under the terms of
# the FireANTs License version 1.0. A copy of the license can be found
# in the LICENSE file at the root of this repository.
#
# IMPORTANT: This code is part of FireANTs and its use, reproduction, or
# distribution must comply with the full license terms, including:
# - Maintaining all copyright notices and bibliography references
# - Using only approved (re)-distribution channels 
# - Proper attribution in derivative works
#
# For full license details, see: https://github.com/rohitrango/FireANTs/blob/main/LICENSE 


from setuptools import setup, Extension
from torch.utils import cpp_extension
import torch
import os
import shutil
import subprocess
import sys

include_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'include')

# CPU kernels are always built; CUDA kernels are added when a CUDA toolkit is found
# (set FIREANTS_FUSED_OPS_CPU_ONLY=1 to skip them)
cpu_sources = [
    'src/src.cpp',
    'src/DeviceDispatch.cpp',
    'src/cpu/CrossCorrelationCPU.cpp',
    'src/cpu/FusedGridSamplerCPU.cpp',
    'src/cpu/MutualInformationCPU.cpp',
    'src/cpu/UtilsCPU.cpp',
]
cuda_sources = [
    'src/CrossCorrelation.cu',
    'src/FusedGridSampler.cu',
    'src/FusedGridSamplerGenericLabel.cu',
    'src/FusedGridComposer.cu',
    'src/FusedGenerateGrid.cu',
    'src/AdamUtils.cu',
    'src/GaussianBlurFFT.cu',
    'src/MutualInformation.cu'
]
cpu_only = os.environ.get('FIREANTS_FUSED_OPS_CPU_ONLY', '0') == '1'
build_cuda = cpp_extension.CUDA_HOME is not None and not cpu_only

if torch.version.cuda is not None and cpp_extension.CUDA_HOME is None and not cpu_only:
    stars = '*' * 79
    print('\n' + stars + '\n'
          'WARNING: PyTorch was built with CUDA but no CUDA toolkit was found (CUDA_HOME is not set),\n'
          'so fireants_fused_ops is built with CPU kernels only and CUDA tensors will use the PyTorch\n'
          'fallbacks. To build the CUDA kernels, install the CUDA toolkit and set CUDA_HOME. To silence\n'
          'this warning, set FIREANTS_FUSED_OPS_CPU_ONLY=1.\n' + stars + '\n')


def macos_openmp_flags():
    cflags, ldflags = os.environ.get('FIREANTS_OPENMP_CFLAGS'), os.environ.get('FIREANTS_OPENMP_LDFLAGS')
    if cflags is not None or ldflags is not None:
        return (cflags or '').split(), (ldflags or '').split()
    prefix = None
    if shutil.which('brew') is not None:
        try:
            prefix = subprocess.run(['brew', '--prefix', 'libomp'], capture_output=True, text=True, check=True).stdout.strip()
        except (subprocess.CalledProcessError, OSError):
            prefix = None
    if prefix and os.path.exists(os.path.join(prefix, 'lib', 'libomp.dylib')):
        return (['-Xpreprocessor', '-fopenmp', '-I' + os.path.join(prefix, 'include')],
                ['-L' + os.path.join(prefix, 'lib'), '-lomp', '-Wl,-rpath,' + os.path.join(prefix, 'lib')])
    print('NOTICE: OpenMP was not found on macOS, the CPU kernels of fireants_fused_ops will run single-threaded. '
          'Install it with `brew install libomp` or set FIREANTS_OPENMP_CFLAGS and FIREANTS_OPENMP_LDFLAGS.')
    return [], []


if sys.platform == 'win32':
    cxx_args, link_args = ['/O2', '/openmp'], []
elif sys.platform == 'darwin':
    omp_cflags, omp_ldflags = macos_openmp_flags()
    cxx_args, link_args = ['-O3'] + omp_cflags, omp_ldflags
else:
    cxx_args, link_args = ['-O3', '-fopenmp'], ['-fopenmp']

if build_cuda:
    extension = cpp_extension.CUDAExtension(
        name='fireants_fused_ops',
        sources=cpu_sources + cuda_sources,
        include_dirs=[include_dir] + torch.utils.cpp_extension.include_paths(),
        library_dirs=torch.utils.cpp_extension.library_paths(),
        define_macros=[('FIREANTS_WITH_CUDA', None)],
        extra_compile_args={
             'cxx': cxx_args,
             'nvcc': ['-lineinfo'],
        },
        extra_link_args=link_args,
    )
else:
    extension = cpp_extension.CppExtension(
        name='fireants_fused_ops',
        sources=cpu_sources,
        include_dirs=[include_dir] + torch.utils.cpp_extension.include_paths(),
        library_dirs=torch.utils.cpp_extension.library_paths(),
        extra_compile_args={'cxx': cxx_args},
        extra_link_args=link_args,
    )

setup(
    name='fireants_fused_ops',
    version='1.0.0',
    description='Fused CPU and CUDA operations for FireANTs',
    author='Rohit Jena',
    ext_modules=[extension],
    cmdclass={'build_ext': cpp_extension.BuildExtension},
    install_requires=['torch>=2.3.0'],
)

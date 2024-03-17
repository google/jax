# Copyright 2018 The JAX Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import importlib
import os
from setuptools import setup
from setuptools.dist import Distribution

__version__ = None
project_name = 'jaxlib'

def load_version_module(pkg_path):
  spec = importlib.util.spec_from_file_location(
    'version', os.path.join(pkg_path, 'version.py'))
  module = importlib.util.module_from_spec(spec)
  spec.loader.exec_module(module)
  return module

_version_module = load_version_module(project_name)
__version__ = _version_module._get_version_for_build()
_cmdclass = _version_module._get_cmdclass(project_name)

with open('README.md') as f:
  _long_description = f.read()

cuda_version = os.environ.get("JAX_CUDA_VERSION")
cudnn_version = os.environ.get("JAX_CUDNN_VERSION")
if cuda_version and cudnn_version:
  __version__ += f"+cuda{cuda_version.replace('.', '')}-cudnn{cudnn_version.replace('.', '')}"

rocm_version = os.environ.get("JAX_ROCM_VERSION")
if rocm_version:
    __version__ += f"+rocm{rocm_version.replace('.', '')}"

class BinaryDistribution(Distribution):
  """This class makes 'bdist_wheel' include an ABI tag on the wheel."""

  def has_ext_modules(self):
    return True

setup(
    name=project_name,
    version=__version__,
    cmdclass=_cmdclass,
    description='XLA library for JAX',
    long_description=_long_description,
    long_description_content_type='text/markdown',
    author='JAX team',
    author_email='jax-dev@google.com',
    packages=['jaxlib', 'jaxlib.xla_extension'],
    python_requires='>=3.9',
    install_requires=[
        'scipy>=1.9',
        "scipy>=1.11.1; python_version>='3.12'",
        'numpy>=1.23',
        'ml_dtypes>=0.2.0',
    ],
    extras_require={
      'cuda11_pip': [
        "nvidia-cublas-cu11>=11.11",
        "nvidia-cuda-cupti-cu11>=11.8",
        "nvidia-cuda-nvcc-cu11>=11.8",
        "nvidia-cuda-runtime-cu11>=11.8",
        "nvidia-cudnn-cu11>=8.8",
        "nvidia-cufft-cu11>=10.9",
        "nvidia-cusolver-cu11>=11.4",
        "nvidia-cusparse-cu11>=11.7",
      ],
      'cuda12_pip': [
        "nvidia-cublas-cu12",
        "nvidia-cuda-cupti-cu12",
        "nvidia-cuda-nvcc-cu12",
        "nvidia-cuda-runtime-cu12",
        "nvidia-cudnn-cu12>=8.9",
        "nvidia-cufft-cu12",
        "nvidia-cusolver-cu12",
        "nvidia-cusparse-cu12",
      ],
    },
    url='https://github.com/google/jax',
    license='Apache-2.0',
    classifiers=[
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    package_data={
        'jaxlib': [
            '*.so',
            '*.pyd*',
            'py.typed',
            'cpu/*',
            'cuda/*',
            'cuda/nvvm/libdevice/libdevice*',
            'mosaic/*.py',
            'mosaic/python/*.py',
            'mosaic/python/*.so',
            'mlir/*.py',
            'mlir/dialects/*.py',
            'mlir/extras/*.py',
            'mlir/_mlir_libs/*.dll',
            'mlir/_mlir_libs/*.dylib',
            'mlir/_mlir_libs/*.so',
            'mlir/_mlir_libs/*.pyd',
            'mlir/_mlir_libs/*.py',
            'rocm/*',
            'triton/*.py',
            'triton/*.pyi',
            'triton/*.pyd',
            'triton/*.so',
        ],
        'jaxlib.xla_extension': ['*.pyi'],
    },
    zip_safe=False,
    distclass=BinaryDistribution,
)

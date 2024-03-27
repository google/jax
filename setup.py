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

from distutils import spawn
import importlib
import os
import subprocess
import sys

from setuptools import setup, find_packages

project_name = 'jax'

_current_jaxlib_version = '0.4.25'
# The following should be updated with each new jaxlib release.
_latest_jaxlib_version_on_pypi = '0.4.25'
_available_cuda11_cudnn_versions = ['86']
_default_cuda11_cudnn_version = '86'
_default_cuda12_cudnn_version = '89'
_libtpu_version = '0.1.dev20240224'

def load_version_module(pkg_path):
  spec = importlib.util.spec_from_file_location(
    'version', os.path.join(pkg_path, 'version.py'))
  module = importlib.util.module_from_spec(spec)
  spec.loader.exec_module(module)
  return module

_version_module = load_version_module(project_name)
__version__ = _version_module._get_version_for_build()
_cmdclass = _version_module._get_cmdclass(project_name)
_minimum_jaxlib_version = _version_module._minimum_jaxlib_version

with open('README.md', encoding='utf-8') as f:
  _long_description = f.read()

if 'PROTOC' in os.environ and os.path.exists(os.environ['PROTOC']):
  protoc = os.environ['PROTOC']
else:
  protoc = spawn.find_executable('protoc')

def generate_proto(source):
  if not protoc or not os.path.exists(source):
    return
  protoc_command = [protoc, '-I.', '--python_out=.', source]
  if subprocess.call(protoc_command) != 0:
    sys.exit(-1)

generate_proto("jax/experimental/australis/executable.proto")
generate_proto("jax/experimental/australis/petri.proto")

setup(
    name=project_name,
    version=__version__,
    cmdclass=_cmdclass,
    description='Differentiate, compile, and transform Numpy code.',
    long_description=_long_description,
    long_description_content_type='text/markdown',
    author='JAX team',
    author_email='jax-dev@google.com',
    packages=find_packages(exclude=["examples", "jax/src/internal_test_util"]),
    package_data={'jax': ['py.typed', "*.pyi", "**/*.pyi"]},
    python_requires='>=3.9',
    install_requires=[
        'ml_dtypes>=0.2.0',
        'numpy>=1.22',
        "numpy>=1.23.2; python_version>='3.11'",
        "numpy>=1.26.0; python_version>='3.12'",
        'opt_einsum',
        'scipy>=1.9',
        "scipy>=1.11.1; python_version>='3.12'",
        # Required by xla_bridge.discover_pjrt_plugins for forwards compat with
        # Python versions < 3.10. Can be dropped when 3.10 is the minimum
        # required Python version.
        'importlib_metadata>=4.6;python_version<"3.10"',
    ],
    extras_require={
        # Minimum jaxlib version; used in testing.
        'minimum-jaxlib': [f'jaxlib=={_minimum_jaxlib_version}'],

        # CPU-only jaxlib can be installed via:
        # $ pip install jax[cpu]
        'cpu': [f'jaxlib=={_current_jaxlib_version}'],

        # Used only for CI builds that install JAX from github HEAD.
        'ci': [f'jaxlib=={_latest_jaxlib_version_on_pypi}'],

        # Cloud TPU VM jaxlib can be installed via:
        # $ pip install jax[tpu] -f https://storage.googleapis.com/jax-releases/libtpu_releases.html
        'tpu': [
          f'jaxlib=={_current_jaxlib_version}',
          f'libtpu-nightly=={_libtpu_version}',
          'requests',  # necessary for jax.distributed.initialize
        ],

        # $ pip install jax[australis]
        'australis': ['protobuf>=3.13,<4'],

        # CUDA installations require adding the JAX CUDA releases URL, e.g.,
        # Cuda installation defaulting to a CUDA and Cudnn version defined above.
        # $ pip install jax[cuda] -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
        'cuda': [f"jaxlib=={_current_jaxlib_version}+cuda11.cudnn{_default_cuda11_cudnn_version}"],

        'cuda11_pip': [
          f"jaxlib=={_current_jaxlib_version}+cuda11.cudnn{_default_cuda11_cudnn_version}",
          "nvidia-cublas-cu11>=11.11",
          "nvidia-cuda-cupti-cu11>=11.8",
          "nvidia-cuda-nvcc-cu11>=11.8",
          "nvidia-cuda-runtime-cu11>=11.8",
          "nvidia-cudnn-cu11>=8.8",
          "nvidia-cufft-cu11>=10.9",
          "nvidia-cusolver-cu11>=11.4",
          "nvidia-cusparse-cu11>=11.7",
          "nvidia-nccl-cu11>=2.18.3",
        ],

        'cuda12_pip': [
          f"jaxlib=={_current_jaxlib_version}+cuda12.cudnn{_default_cuda12_cudnn_version}",
          "nvidia-cublas-cu12>=12.3.4.1",
          "nvidia-cuda-cupti-cu12>=12.3.101",
          "nvidia-cuda-nvcc-cu12>=12.3.107",
          "nvidia-cuda-runtime-cu12>=12.3.101",
          "nvidia-cudnn-cu12>=8.9.7.29",
          "nvidia-cufft-cu12>=11.0.12.1",
          "nvidia-cusolver-cu12>=11.5.4.101",
          "nvidia-cusparse-cu12>=12.2.0.103",
          "nvidia-nccl-cu12>=2.19.3",
          # nvjitlink is not a direct dependency of JAX, but it is a transitive
          # dependency via, for example, cuSOLVER. NVIDIA's cuSOLVER packages
          # do not have a version constraint on their dependencies, so the
          # package doesn't get upgraded even though not doing that can cause
          # problems (https://github.com/google/jax/issues/18027#issuecomment-1756305196)
          # Until NVIDIA add version constraints, add an version constraint
          # here.
          "nvidia-nvjitlink-cu12>=12.3.101",
        ],

        'cuda12': [
          f"jaxlib=={_current_jaxlib_version}",
          f"jax-cuda12-plugin=={_current_jaxlib_version}",
          "nvidia-cublas-cu12>=12.3.4.1",
          "nvidia-cuda-cupti-cu12>=12.3.101",
          "nvidia-cuda-nvcc-cu12>=12.3.107",
          "nvidia-cuda-runtime-cu12>=12.3.101",
          "nvidia-cudnn-cu12>=8.9.7.29",
          "nvidia-cufft-cu12>=11.0.12.1",
          "nvidia-cusolver-cu12>=11.5.4.101",
          "nvidia-cusparse-cu12>=12.2.0.103",
          "nvidia-nccl-cu12>=2.19.3",
          # nvjitlink is not a direct dependency of JAX, but it is a transitive
          # dependency via, for example, cuSOLVER. NVIDIA's cuSOLVER packages
          # do not have a version constraint on their dependencies, so the
          # package doesn't get upgraded even though not doing that can cause
          # problems (https://github.com/google/jax/issues/18027#issuecomment-1756305196)
          # Until NVIDIA add version constraints, add an version constraint
          # here.
          "nvidia-nvjitlink-cu12>=12.3.101",
        ],

        # Target that does not depend on the CUDA pip wheels, for those who want
        # to use a preinstalled CUDA.
        'cuda11_local': [
          f"jaxlib=={_current_jaxlib_version}+cuda11.cudnn{_default_cuda11_cudnn_version}",
        ],
        'cuda12_local': [
          f"jaxlib=={_current_jaxlib_version}+cuda12.cudnn{_default_cuda12_cudnn_version}",
        ],

        # CUDA installations require adding jax releases URL; e.g.
        # $ pip install jax[cuda11_cudnn82] -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
        # $ pip install jax[cuda11_cudnn86] -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
        **{f'cuda11_cudnn{cudnn_version}': f"jaxlib=={_current_jaxlib_version}+cuda11.cudnn{cudnn_version}"
           for cudnn_version in _available_cuda11_cudnn_versions}
    },
    url='https://github.com/google/jax',
    license='Apache-2.0',
    classifiers=[
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    zip_safe=False,
)

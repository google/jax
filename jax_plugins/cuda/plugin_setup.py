# Copyright 2023 The JAX Authors.
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
cuda_version = 0  # placeholder
project_name = f"jax-cuda{cuda_version}-plugin"
package_name = f"jax_cuda{cuda_version}_plugin"

def load_version_module(pkg_path):
  spec = importlib.util.spec_from_file_location(
    'version', os.path.join(pkg_path, 'version.py'))
  module = importlib.util.module_from_spec(spec)
  spec.loader.exec_module(module)
  return module

_version_module = load_version_module(package_name)
__version__ = _version_module._get_version_for_build()
_cmdclass = _version_module._get_cmdclass(package_name)

class BinaryDistribution(Distribution):
  """This class makes 'bdist_wheel' include an ABI tag on the wheel."""

  def has_ext_modules(self):
    return True

setup(
    name=project_name,
    version=__version__,
    cmdclass=_cmdclass,
    description="JAX Plugin for NVIDIA GPUs",
    long_description="",
    long_description_content_type="text/markdown",
    author="JAX team",
    author_email="jax-dev@google.com",
    packages=[package_name],
    python_requires=">=3.10",
    install_requires=[f"jax-cuda{cuda_version}-pjrt=={__version__}"],
    extras_require={
      'with_cuda': [
          "nvidia-cublas-cu12>=12.1.3.1",
          "nvidia-cuda-cupti-cu12>=12.1.105",
          "nvidia-cuda-nvcc-cu12>=12.1.105",
          "nvidia-cuda-runtime-cu12>=12.1.105",
          "nvidia-cudnn-cu12>=9.0,<10.0",
          "nvidia-cufft-cu12>=11.0.2.54",
          "nvidia-cusolver-cu12>=11.4.5.107",
          "nvidia-cusparse-cu12>=12.1.0.106",
          "nvidia-nccl-cu12>=2.18.1",
          # nvjitlink is not a direct dependency of JAX, but it is a transitive
          # dependency via, for example, cuSOLVER. NVIDIA's cuSOLVER packages
          # do not have a version constraint on their dependencies, so the
          # package doesn't get upgraded even though not doing that can cause
          # problems (https://github.com/google/jax/issues/18027#issuecomment-1756305196)
          # Until NVIDIA add version constraints, add a version constraint
          # here.
          "nvidia-nvjitlink-cu12>=12.1.105",
      ],
    },
    url="https://github.com/google/jax",
    license="Apache-2.0",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
    ],
    package_data={
        package_name: [
            "*",
            "nvvm/libdevice/libdevice*",
        ],
    },
    zip_safe=False,
    distclass=BinaryDistribution,
)

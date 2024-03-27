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

# Script that builds a jax-cuda12-plugin wheel for cuda kernels, intended to be
# run via bazel run as part of the jax cuda plugin build process.

# Most users should not run this script directly; use build.py instead.

import argparse
import functools
import os
import pathlib
import tempfile

from bazel_tools.tools.python.runfiles import runfiles
from jax.tools import build_utils

parser = argparse.ArgumentParser()
parser.add_argument(
    "--output_path",
    default=None,
    required=True,
    help="Path to which the output wheel should be written. Required.",
)
parser.add_argument(
    "--jaxlib_git_hash",
    default="",
    required=True,
    help="Git hash. Empty if unknown. Optional.",
)
parser.add_argument(
    "--cpu", default=None, required=True, help="Target CPU architecture. Required."
)
parser.add_argument(
    "--cuda_version",
    default=None,
    required=True,
    help="Target CUDA version. Required.",
)
parser.add_argument(
    "--editable",
    action="store_true",
    help="Create an 'editable' jax cuda plugin build instead of a wheel.",
)
args = parser.parse_args()

r = runfiles.Create()
pyext = "pyd" if build_utils.is_windows() else "so"


def write_setup_cfg(sources_path, cpu):
  tag = build_utils.platform_tag(cpu)
  with open(sources_path / "setup.cfg", "w") as f:
    f.write(f"""[metadata]
license_files = LICENSE.txt

[bdist_wheel]
plat-name={tag}
""")


def prepare_wheel(
    sources_path: pathlib.Path, *, cpu, cuda_version
):
  """Assembles a source tree for the cuda kernel wheel in `sources_path`."""
  copy_runfiles = functools.partial(build_utils.copy_file, runfiles=r)

  copy_runfiles(
      "__main__/jax_plugins/cuda/plugin_pyproject.toml",
      dst_dir=sources_path,
      dst_filename="pyproject.toml",
  )
  copy_runfiles(
      "__main__/jax_plugins/cuda/plugin_setup.py",
      dst_dir=sources_path,
      dst_filename="setup.py",
  )
  build_utils.update_setup_with_cuda_version(sources_path, cuda_version)
  write_setup_cfg(sources_path, cpu)

  plugin_dir = sources_path / f"jax_cuda{cuda_version}_plugin"
  copy_runfiles(
      dst_dir=plugin_dir / "nvvm" / "libdevice",
      src_files=[
          "cuda_nvcc/nvvm/libdevice/libdevice.10.bc",
      ],
  )
  copy_runfiles(
      dst_dir=plugin_dir,
      src_files=[
          f"__main__/jaxlib/cuda/_solver.{pyext}",
          f"__main__/jaxlib/cuda/_blas.{pyext}",
          f"__main__/jaxlib/cuda/_linalg.{pyext}",
          f"__main__/jaxlib/cuda/_prng.{pyext}",
          f"__main__/jaxlib/cuda/_rnn.{pyext}",
          f"__main__/jaxlib/cuda/_sparse.{pyext}",
          f"__main__/jaxlib/cuda/_triton.{pyext}",
          f"__main__/jaxlib/cuda/_versions.{pyext}",
          f"__main__/jaxlib/cuda_plugin_extension.{pyext}",
          "__main__/jaxlib/version.py",
      ],
  )

# Build wheel for cuda kernels
tmpdir = tempfile.TemporaryDirectory(prefix="jax_cuda_plugin")
sources_path = tmpdir.name
try:
  os.makedirs(args.output_path, exist_ok=True)
  prepare_wheel(
      pathlib.Path(sources_path), cpu=args.cpu, cuda_version=args.cuda_version
  )
  package_name = f"jax cuda{args.cuda_version} plugin"
  if args.editable:
    build_utils.build_editable(sources_path, args.output_path, package_name)
  else:
    build_utils.build_wheel(
        sources_path,
        args.output_path,
        package_name,
        git_hash=args.jaxlib_git_hash,
    )
finally:
  tmpdir.cleanup()

# Copyright 2020 The JAX Authors.
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

# Script that builds a jaxlib wheel, intended to be run via bazel run as part
# of the jaxlib build process.

# Most users should not run this script directly; use build.py instead.

import argparse
import datetime
import functools
import glob
import os
import pathlib
import platform
import re
import shutil
import subprocess
import sys
import tempfile

from bazel_tools.tools.python.runfiles import runfiles

parser = argparse.ArgumentParser()
parser.add_argument(
  "--sources_path",
  default=None,
  help="Path in which the wheel's sources should be prepared. Optional. If "
       "omitted, a temporary directory will be used.")
parser.add_argument(
  "--output_path",
  default=None,
  required=True,
  help="Path to which the output wheel should be written. Required.")
parser.add_argument(
  "--cpu",
  default=None,
  required=True,
  help="Target CPU architecture. Required.")
args = parser.parse_args()

r = runfiles.Create()


def _is_mac():
  return platform.system() == "Darwin"


def _is_windows():
  return sys.platform.startswith("win32")


pyext = "pyd" if _is_windows() else "so"


def exists(src_file):
  return r.Rlocation(src_file) is not None


def copy_file(src_file, dst_dir, dst_filename=None, from_runfiles=True):
  if from_runfiles:
    src_file = r.Rlocation(src_file)

  src_filename = os.path.basename(src_file)
  dst_file = os.path.join(dst_dir, dst_filename or src_filename)
  if _is_windows():
    shutil.copyfile(src_file, dst_file)
  else:
    shutil.copy(src_file, dst_file)


_XLA_EXTENSION_STUBS = [
    "__init__.pyi",
    "jax_jit.pyi",
    "ops.pyi",
    "outfeed_receiver.pyi",
    "pmap_lib.pyi",
    "profiler.pyi",
    "pytree.pyi",
    "transfer_guard_lib.pyi",
    "program_eviction_lib.pyi",
]
_OPTIONAL_XLA_EXTENSION_STUBS = [
    "program_eviction_lib.pyi",  # Will be required on xla_extension_version >= 85.
]


def patch_copy_xla_extension_stubs(dst_dir):
  # This file is required by PEP-561. It marks jaxlib as package containing
  # type stubs.
  with open(os.path.join(dst_dir, "py.typed"), "w"):
    pass
  xla_extension_dir = os.path.join(dst_dir, "xla_extension")
  os.makedirs(xla_extension_dir)
  for stub_name in _XLA_EXTENSION_STUBS:
    stub_path = r.Rlocation(
        "org_tensorflow/tensorflow/compiler/xla/python/xla_extension/" + stub_name)
    stub_path = str(stub_path)  # Make pytype accept os.path.exists(stub_path).
    if stub_name in _OPTIONAL_XLA_EXTENSION_STUBS and not os.path.exists(stub_path):
      continue
    with open(stub_path) as f:
      src = f.read()
    src = src.replace(
        "from tensorflow.compiler.xla.python import xla_extension",
        "from .. import xla_extension"
    )
    with open(os.path.join(xla_extension_dir, stub_name), "w") as f:
      f.write(src)


def patch_copy_tpu_client_py(dst_dir):
  with open(r.Rlocation("org_tensorflow/tensorflow/compiler/xla/python/tpu_driver/client/tpu_client.py")) as f:
    src = f.read()
    src = src.replace("from tensorflow.compiler.xla.python import xla_extension as _xla",
                      "from . import xla_extension as _xla")
    src = src.replace("from tensorflow.compiler.xla.python import xla_client",
                      "from . import xla_client")
    src = src.replace(
        "from tensorflow.compiler.xla.python.tpu_driver.client import tpu_client_extension as _tpu_client",
        "from . import tpu_client_extension as _tpu_client")
    with open(os.path.join(dst_dir, "tpu_client.py"), "w") as f:
      f.write(src)

def verify_mac_libraries_dont_reference_chkstack():
  """Verifies that xla_extension.so doesn't depend on ____chkstk_darwin.

  We don't entirely know why this happens, but in some build environments
  we seem to target the wrong Mac OS version.
  https://github.com/google/jax/issues/3867

  This check makes sure we don't release wheels that have this dependency.
  """
  if not _is_mac():
    return
  nm = subprocess.run(
    ["nm", "-g",
     r.Rlocation("org_tensorflow/tensorflow/compiler/xla/python/xla_extension.so")
    ],
    capture_output=True, text=True,
    check=False)
  if nm.returncode != 0:
    raise RuntimeError(f"nm process failed: {nm.stdout} {nm.stderr}")
  if "____chkstk_darwin" in nm.stdout:
    raise RuntimeError(
      "Mac wheel incorrectly depends on symbol ____chkstk_darwin, which "
      "means that it isn't compatible with older MacOS versions.")


def prepare_wheel(sources_path):
  """Assembles a source tree for the wheel in `sources_path`."""
  jaxlib_dir = os.path.join(sources_path, "jaxlib")
  os.makedirs(jaxlib_dir)
  copy_to_jaxlib = functools.partial(copy_file, dst_dir=jaxlib_dir)

  verify_mac_libraries_dont_reference_chkstack()
  copy_file("__main__/build/LICENSE.txt", dst_dir=sources_path)
  copy_file("__main__/jaxlib/README.md", dst_dir=sources_path)
  copy_file("__main__/jaxlib/setup.py", dst_dir=sources_path)
  copy_file("__main__/jaxlib/setup.cfg", dst_dir=sources_path)
  copy_to_jaxlib("__main__/jaxlib/init.py", dst_filename="__init__.py")
  copy_to_jaxlib(f"__main__/jaxlib/cpu_feature_guard.{pyext}")
  copy_to_jaxlib("__main__/jaxlib/lapack.py")
  copy_to_jaxlib(f"__main__/jaxlib/_lapack.{pyext}")
  copy_to_jaxlib("__main__/jaxlib/mhlo_helpers.py")
  copy_to_jaxlib(f"__main__/jaxlib/_ducc_fft.{pyext}")
  copy_to_jaxlib("__main__/jaxlib/ducc_fft.py")
  copy_to_jaxlib("__main__/jaxlib/gpu_prng.py")
  copy_to_jaxlib("__main__/jaxlib/gpu_linalg.py")
  copy_to_jaxlib("__main__/jaxlib/gpu_solver.py")
  copy_to_jaxlib("__main__/jaxlib/gpu_sparse.py")
  copy_to_jaxlib("__main__/jaxlib/version.py")
  copy_to_jaxlib("__main__/jaxlib/xla_client.py")
  copy_to_jaxlib(f"__main__/jaxlib/xla_extension.{pyext}")

  cuda_dir = os.path.join(jaxlib_dir, "cuda")
  if exists(f"__main__/jaxlib/cuda/_cusolver.{pyext}"):
    libdevice_dir = os.path.join(cuda_dir, "nvvm", "libdevice")
    os.makedirs(libdevice_dir)
    copy_file("local_config_cuda/cuda/cuda/nvvm/libdevice/libdevice.10.bc", dst_dir=libdevice_dir)
    copy_file(f"__main__/jaxlib/cuda/_cusolver.{pyext}", dst_dir=cuda_dir)
    copy_file(f"__main__/jaxlib/cuda/_cublas.{pyext}", dst_dir=cuda_dir)
    copy_file(f"__main__/jaxlib/cuda/_cuda_linalg.{pyext}", dst_dir=cuda_dir)
    copy_file(f"__main__/jaxlib/cuda/_cuda_prng.{pyext}", dst_dir=cuda_dir)
  rocm_dir = os.path.join(jaxlib_dir, "rocm")
  if exists(f"__main__/jaxlib/rocm/_hipsolver.{pyext}"):
    os.makedirs(rocm_dir)
    copy_file(f"__main__/jaxlib/rocm/_hipsolver.{pyext}", dst_dir=rocm_dir)
    copy_file(f"__main__/jaxlib/rocm/_hipblas.{pyext}", dst_dir=rocm_dir)
    copy_file(f"__main__/jaxlib/rocm/_hip_linalg.{pyext}", dst_dir=rocm_dir)
    copy_file(f"__main__/jaxlib/rocm/_hip_prng.{pyext}", dst_dir=rocm_dir)
  if exists(f"__main__/jaxlib/cuda/_cusparse.{pyext}"):
    copy_file(f"__main__/jaxlib/cuda/_cusparse.{pyext}", dst_dir=cuda_dir)
  if exists(f"__main__/jaxlib/rocm/_hipsparse.{pyext}"):
    copy_file(f"__main__/jaxlib/rocm/_hipsparse.{pyext}", dst_dir=rocm_dir)


  mlir_dir = os.path.join(jaxlib_dir, "mlir")
  mlir_dialects_dir = os.path.join(jaxlib_dir, "mlir", "dialects")
  mlir_libs_dir = os.path.join(jaxlib_dir, "mlir", "_mlir_libs")
  os.makedirs(mlir_dir)
  os.makedirs(mlir_dialects_dir)
  os.makedirs(mlir_libs_dir)
  copy_file("__main__/jaxlib/mlir/ir.py", dst_dir=mlir_dir)
  copy_file("__main__/jaxlib/mlir/passmanager.py", dst_dir=mlir_dir)
  copy_file("__main__/jaxlib/mlir/dialects/_builtin_ops_ext.py", dst_dir=mlir_dialects_dir)
  copy_file("__main__/jaxlib/mlir/dialects/_builtin_ops_gen.py", dst_dir=mlir_dialects_dir)
  copy_file("__main__/jaxlib/mlir/dialects/_chlo_ops_gen.py", dst_dir=mlir_dialects_dir)
  copy_file("__main__/jaxlib/mlir/dialects/_mhlo_ops_gen.py", dst_dir=mlir_dialects_dir)
  copy_file("__main__/jaxlib/mlir/dialects/_ods_common.py", dst_dir=mlir_dialects_dir)
  copy_file("__main__/jaxlib/mlir/dialects/_func_ops_ext.py", dst_dir=mlir_dialects_dir)
  copy_file("__main__/jaxlib/mlir/dialects/_func_ops_gen.py", dst_dir=mlir_dialects_dir)
  copy_file("__main__/jaxlib/mlir/dialects/_ml_program_ops_ext.py", dst_dir=mlir_dialects_dir)
  copy_file("__main__/jaxlib/mlir/dialects/_ml_program_ops_gen.py", dst_dir=mlir_dialects_dir)
  copy_file("__main__/jaxlib/mlir/dialects/_sparse_tensor_ops_gen.py", dst_dir=mlir_dialects_dir)
  copy_file("__main__/jaxlib/mlir/dialects/sparse_tensor.py", dst_dir=mlir_dialects_dir)
  copy_file("__main__/jaxlib/mlir/dialects/builtin.py", dst_dir=mlir_dialects_dir)
  copy_file("__main__/jaxlib/mlir/dialects/chlo.py", dst_dir=mlir_dialects_dir)
  copy_file("__main__/jaxlib/mlir/dialects/mhlo.py", dst_dir=mlir_dialects_dir)
  copy_file("__main__/jaxlib/mlir/dialects/func.py", dst_dir=mlir_dialects_dir)
  copy_file("__main__/jaxlib/mlir/dialects/ml_program.py", dst_dir=mlir_dialects_dir)

  copy_file("__main__/jaxlib/mlir/_mlir_libs/__init__.py", dst_dir=mlir_libs_dir)
  copy_file(f"__main__/jaxlib/mlir/_mlir_libs/_mlir.{pyext}", dst_dir=mlir_libs_dir)
  copy_file(f"__main__/jaxlib/mlir/_mlir_libs/_chlo.{pyext}", dst_dir=mlir_libs_dir)
  copy_file(f"__main__/jaxlib/mlir/_mlir_libs/_mlirHlo.{pyext}", dst_dir=mlir_libs_dir)
  copy_file(f"__main__/jaxlib/mlir/_mlir_libs/_mlirDialectsSparseTensor.{pyext}", dst_dir=mlir_libs_dir)
  copy_file(f"__main__/jaxlib/mlir/_mlir_libs/_mlirSparseTensorPasses.{pyext}", dst_dir=mlir_libs_dir)
  copy_file(f"__main__/jaxlib/mlir/_mlir_libs/_stablehlo.{pyext}", dst_dir=mlir_libs_dir)
  copy_file(f"__main__/jaxlib/mlir/_mlir_libs/_site_initialize_0.{pyext}", dst_dir=mlir_libs_dir)
  if _is_windows():
    copy_file("__main__/jaxlib/mlir/_mlir_libs/jaxlib_mlir_capi.dll", dst_dir=mlir_libs_dir)
  elif _is_mac():
    copy_file("__main__/jaxlib/mlir/_mlir_libs/libjaxlib_mlir_capi.dylib", dst_dir=mlir_libs_dir)
  else:
    copy_file("__main__/jaxlib/mlir/_mlir_libs/libjaxlib_mlir_capi.so", dst_dir=mlir_libs_dir)
  patch_copy_xla_extension_stubs(jaxlib_dir)

  if exists("org_tensorflow/tensorflow/compiler/xla/python/tpu_driver/client/tpu_client_extension.so"):
    copy_to_jaxlib("org_tensorflow/tensorflow/compiler/xla/python/tpu_driver/client/tpu_client_extension.so")
    patch_copy_tpu_client_py(jaxlib_dir)


def edit_jaxlib_version(sources_path):
  version_regex = re.compile(r'__version__ = \"(.*)\"')

  version_file = pathlib.Path(sources_path) / "jaxlib" / "version.py"
  content = version_file.read_text()

  version_num = version_regex.search(content).group(1)

  datestring = datetime.datetime.now().strftime('%Y%m%d')
  nightly_version = f'{version_num}.dev{datestring}'

  content = content.replace(f'__version__ = "{version_num}"',
                            f'__version__ = "{nightly_version}"')
  version_file.write_text(content)


def build_wheel(sources_path, output_path, cpu):
  """Builds a wheel in `output_path` using the source tree in `sources_path`."""
  platform_name, cpu_name = {
    ("Linux", "x86_64"): ("manylinux2014", "x86_64"),
    ("Linux", "aarch64"): ("manylinux2014", "aarch64"),
    ("Linux", "ppc64le"): ("manylinux2014", "ppc64le"),
    ("Darwin", "x86_64"): ("macosx_10_14", "x86_64"),
    ("Darwin", "arm64"): ("macosx_11_0", "arm64"),
    ("Windows", "AMD64"): ("win", "amd64"),
  }[(platform.system(), cpu)]
  python_tag_arg = (f"--python-tag=cp{sys.version_info.major}"
                    f"{sys.version_info.minor}")
  platform_tag_arg = f"--plat-name={platform_name}_{cpu_name}"
  cwd = os.getcwd()
  if os.environ.get('JAXLIB_NIGHTLY'):
    edit_jaxlib_version(sources_path)
  os.chdir(sources_path)
  subprocess.run([sys.executable, "setup.py", "bdist_wheel",
                 python_tag_arg, platform_tag_arg], check=True)
  os.chdir(cwd)
  for wheel in glob.glob(os.path.join(sources_path, "dist", "*.whl")):
    output_file = os.path.join(output_path, os.path.basename(wheel))
    sys.stderr.write(f"Output wheel: {output_file}\n\n")
    sys.stderr.write("To install the newly-built jaxlib wheel, run:\n")
    sys.stderr.write(f"  pip install {output_file}\n\n")
    shutil.copy(wheel, output_path)


tmpdir = None
sources_path = args.sources_path
if sources_path is None:
  tmpdir = tempfile.TemporaryDirectory(prefix="jaxlib")
  sources_path = tmpdir.name

try:
  os.makedirs(args.output_path, exist_ok=True)
  prepare_wheel(sources_path)
  build_wheel(sources_path, args.output_path, args.cpu)
finally:
  if tmpdir:
    tmpdir.cleanup()

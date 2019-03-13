#!/usr/bin/python
#
# Copyright 2018 Google LLC
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
#
# Helper script for building JAX's libjax easily.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import collections
import hashlib
import os
import platform
import re
import shutil
import stat
import subprocess
import sys
import urllib

# pylint: disable=g-import-not-at-top
if hasattr(urllib, "urlretrieve"):
  urlretrieve = urllib.urlretrieve
else:
  import urllib.request
  urlretrieve = urllib.request.urlretrieve

if hasattr(shutil, "which"):
  which = shutil.which
else:
  from distutils.spawn import find_executable as which
# pylint: enable=g-import-not-at-top


def shell(cmd):
  output = subprocess.check_output(cmd)
  return output.decode("UTF-8").strip()


# Python

def get_python_bin_path(python_bin_path_flag):
  """Returns the path to the Python interpreter to use."""
  return python_bin_path_flag or sys.executable


# Bazel

BAZEL_BASE_URI = "https://github.com/bazelbuild/bazel/releases/download/0.22.0/"
BazelPackage = collections.namedtuple("BazelPackage", ["file", "sha256"])
bazel_packages = {
    "Linux":
        BazelPackage(
            file="bazel-0.22.0-linux-x86_64",
            sha256=
            "8474ed28ed4998e2f5671ddf3a9a80ae9e484a5de3b8b70c8b654c017c65d363"),
    "Darwin":
        BazelPackage(
            file="bazel-0.22.0-darwin-x86_64",
            sha256=
            "0fcd80d8a20b7c8b7b70ea9da4bea9b2ce3985c792d0cc7676a9e4c2f9600478"),
}


def download_and_verify_bazel():
  """Downloads a bazel binary from Github, verifying its SHA256 hash."""
  package = bazel_packages.get(platform.system())
  if package is None:
    return None

  if not os.access(package.file, os.X_OK):
    uri = BAZEL_BASE_URI + package.file
    sys.stdout.write("Downloading bazel from: {}\n".format(uri))

    def progress(block_count, block_size, total_size):
      if total_size <= 0:
        total_size = 170**6
      progress = (block_count * block_size) / total_size
      num_chars = 40
      progress_chars = int(num_chars * progress)
      sys.stdout.write("{} [{}{}] {}%\r".format(
          package.file, "#" * progress_chars,
          "." * (num_chars - progress_chars), int(progress * 100.0)))

    tmp_path, _ = urlretrieve(uri, None, progress)
    sys.stdout.write("\n")

    # Verify that the downloaded Bazel binary has the expected SHA256.
    downloaded_file = open(tmp_path, "rb")
    contents = downloaded_file.read()
    downloaded_file.close()
    digest = hashlib.sha256(contents).hexdigest()
    if digest != package.sha256:
      print(
          "Checksum mismatch for downloaded bazel binary (expected {}; got {})."
          .format(package.sha256, digest))
      sys.exit(-1)

    # Write the file as the bazel file name.
    out_file = open(package.file, "wb")
    out_file.write(contents)
    out_file.close()

    # Mark the file as executable.
    st = os.stat(package.file)
    os.chmod(package.file,
             st.st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)

  return "./" + package.file


def get_bazel_path(bazel_path_flag):
  """Returns the path to a Bazel binary, downloading Bazel if not found."""
  if bazel_path_flag:
    return bazel_path_flag

  bazel = which("bazel")
  if bazel:
    return bazel

  bazel = download_and_verify_bazel()
  if bazel:
    return bazel

  print("Cannot find or download bazel. Please install bazel.")
  sys.exit(-1)


def check_bazel_version(bazel_path, min_version):
  """Checks Bazel's version is at least `min_version`."""
  version_output = shell([bazel_path, "--bazelrc=/dev/null", "version"])
  match = re.search("Build label: *([0-9\\.]+)[^0-9\\.]", version_output)
  if match is None:
    print("Warning: bazel installation is not a release version. Make sure "
          "bazel is at least 0.19.2")
    return
  version = match.group(1)
  min_ints = [int(x) for x in min_version.split(".")]
  actual_ints = [int(x) for x in match.group(1).split(".")]
  if min_ints > actual_ints:
    print("Outdated bazel revision (>= {} required, found {})".format(
        min_version, version))
    sys.exit(0)


BAZELRC_TEMPLATE = """
build --action_env PYTHON_BIN_PATH="{python_bin_path}"
build --python_path="{python_bin_path}"
build --action_env TF_NEED_CUDA="{tf_need_cuda}"
build --action_env CUDA_TOOLKIT_PATH="{cuda_toolkit_path}"
build --action_env CUDNN_INSTALL_PATH="{cudnn_install_path}"
build --distinct_host_configuration=false
build --copt=-Wno-sign-compare
build -c opt
build:opt --copt=-march=native
build:opt --host_copt=-march=native
build:mkl_open_source_only --define=tensorflow_mkldnn_contraction_kernel=1

# Disable enabled-by-default TensorFlow features that we don't care about.
build --define=no_aws_support=true
build --define=no_gcp_support=true
build --define=no_hdfs_support=true
build --define=no_kafka_support=true
build --define=no_ignite_support=true

build:cuda --crosstool_top=@local_config_cuda//crosstool:toolchain
build:cuda --define=using_cuda=true --define=using_cuda_nvcc=true
"""


def write_bazelrc(**kwargs):
  f = open("../.bazelrc", "w")
  f.write(BAZELRC_TEMPLATE.format(**kwargs))
  f.close()


BANNER = r"""
     _   _  __  __
    | | / \ \ \/ /
 _  | |/ _ \ \  /
| |_| / ___ \/  \
 \___/_/   \/_/\_\

"""

EPILOG = """

From the 'build' directory in the JAX repository, run
    python build.py
or
    python3 build.py
to download and build JAX's XLA (jaxlib) dependency.
"""


def _parse_string_as_bool(s):
  """Parses a string as a boolean argument."""
  lower = s.lower()
  if lower == "true":
    return True
  elif lower == "false":
    return False
  else:
    raise ValueError("Expected either 'true' or 'false'; got {}".format(s))


def add_boolean_argument(parser, name, default=False, help_str=None):
  """Creates a boolean flag."""
  group = parser.add_mutually_exclusive_group()
  group.add_argument(
      "--" + name,
      nargs="?",
      default=default,
      const=True,
      type=_parse_string_as_bool,
      help=help_str)
  group.add_argument("--no" + name, dest=name, action="store_false")


def main():
  parser = argparse.ArgumentParser(
      description="Builds libjax from source.", epilog=EPILOG)
  parser.add_argument(
      "--bazel_path",
      help="Path to the Bazel binary to use. The default is to find bazel via "
      "the PATH; if none is found, downloads a fresh copy of bazel from "
      "GitHub.")
  parser.add_argument(
      "--python_bin_path",
      help="Path to Python binary to use. The default is the Python "
      "interpreter used to run the build script.")
  add_boolean_argument(
      parser,
      "enable_march_native",
      default=False,
      help_str="Generate code targeted to the current machine? This may "
          "increase performance, but may generate code that does not run on "
          "older machines.")
  add_boolean_argument(
      parser,
      "enable_mkl_dnn",
      default=True,
      help_str="Should we build with MKL-DNN enabled?")
  add_boolean_argument(
      parser,
      "enable_cuda",
      help_str="Should we build with CUDA enabled? Requires CUDA and CuDNN.")
  parser.add_argument(
      "--cuda_path",
      default="/usr/local/cuda",
      help="Path to the CUDA toolkit.")
  parser.add_argument(
      "--cudnn_path",
      default="/usr/local/cuda",
      help="Path to CUDNN libraries.")
  args = parser.parse_args()

  print(BANNER)
  os.chdir(os.path.dirname(__file__ or args.prog) or '.')

  # Find a working Bazel.
  bazel_path = get_bazel_path(args.bazel_path)
  check_bazel_version(bazel_path, "0.19.2")
  print("Bazel binary path: {}".format(bazel_path))

  python_bin_path = get_python_bin_path(args.python_bin_path)
  print("Python binary path: {}".format(python_bin_path))

  print("MKL-DNN enabled: {}".format("yes" if args.enable_mkl_dnn else "no"))
  print("-march=native: {}".format("yes" if args.enable_march_native else "no"))

  cuda_toolkit_path = args.cuda_path
  cudnn_install_path = args.cudnn_path
  print("CUDA enabled: {}".format("yes" if args.enable_cuda else "no"))
  if args.enable_cuda:
    print("CUDA toolkit path: {}".format(cuda_toolkit_path))
    print("CUDNN library path: {}".format(cudnn_install_path))
  write_bazelrc(
      python_bin_path=python_bin_path,
      tf_need_cuda=1 if args.enable_cuda else 0,
      cuda_toolkit_path=cuda_toolkit_path,
      cudnn_install_path=cudnn_install_path)

  print("\nBuilding XLA and installing it in the jaxlib source tree...")
  config_args = []
  if args.enable_march_native:
    config_args += ["--config=opt"]
  if args.enable_mkl_dnn:
    config_args += ["--config=mkl_open_source_only"]
  if args.enable_cuda:
    config_args += ["--config=cuda"]
  shell(
    [bazel_path, "run", "--verbose_failures=true"] +
    config_args +
    [":install_xla_in_source_tree", os.getcwd()])


if __name__ == "__main__":
  main()

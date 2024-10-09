# Copyright 2024 The JAX Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
# Helper script for setting up the tools used by the CLI.
import collections
import hashlib
import logging
import os
import platform
import shutil
import subprocess
import urllib.request

logger = logging.getLogger(__name__)

BAZELISK_BASE_URI = (
    "https://github.com/bazelbuild/bazelisk/releases/download/v1.21.0/"
)

BazeliskPackage = collections.namedtuple("BazeliskPackage", ["file", "sha256"])

BAZELISK_PACKAGES = {
    ("Linux", "x86_64"): BazeliskPackage(
        file="bazelisk-linux-amd64",
        sha256=(
            "655a5c675dacf3b7ef4970688b6a54598aa30cbaa0b9e717cd1412c1ef9ec5a7"
        ),
    ),
    ("Linux", "aarch64"): BazeliskPackage(
        file="bazelisk-linux-arm64",
        sha256=(
            "ff793b461968e30d9f954c080f4acaa557edbdeab1ce276c02e4929b767ead66"
        ),
    ),
    ("Darwin", "x86_64"): BazeliskPackage(
        file="bazelisk-darwin",
        sha256=(
            "07ba3d6b90c28984237a6273f6b7de2fd714a1e3a65d1e78f9b342675ecb75e4"
        ),
    ),
    ("Darwin", "arm64"): BazeliskPackage(
        file="bazelisk-darwin-arm64",
        sha256=(
            "17529faeed52219ee170d59bd820c401f1645a95f95ee4ac3ebd06972edfb6ff"
        ),
    ),
    ("Windows", "AMD64"): BazeliskPackage(
        file="bazelisk-windows-amd64.exe",
        sha256=(
            "3da1895614f460692635f8baa0cab6bb35754fc87d9badbd2b3b2ba55873cf89"
        ),
    ),
}

def guess_clang_paths(clang_path_flag):
  """
  Yields a sequence of guesses about Clang path. Some of sequence elements
  can be None. The resulting iterator is lazy and potentially has a side
  effects.
  """

  yield clang_path_flag
  yield shutil.which("clang")

def get_clang_path(clang_path_flag):
  for clang_path in guess_clang_paths(clang_path_flag):
    if clang_path:
      absolute_clang_path = os.path.realpath(clang_path)
      logger.debug("Found path to Clang: %s.", absolute_clang_path)
      return absolute_clang_path

def get_jax_supported_bazel_version(filename: str = ".bazelversion"):
  """Reads the contents of .bazelversion into a string.

  Args:
      filename: The path to ".bazelversion".

  Returns:
      The Bazel version as a string, or None if the file doesn't exist.
  """
  try:
    with open(filename, 'r') as file:
      content = file.read()
      return content.strip()
  except FileNotFoundError:
    print(f"Error: File '{filename}' not found.")
    return None

def get_bazel_path(bazel_path_flag):
  for bazel_path in guess_bazel_paths(bazel_path_flag):
    if bazel_path and verify_bazel_version(bazel_path):
      logger.debug("Found a compatible Bazel installation.")
      return bazel_path
  logger.debug("Unable not find a compatible Bazel installation. Downloading Bazelisk...")
  return download_and_verify_bazelisk()

def verify_bazel_version(bazel_path):
  """
  Verifies if the version of Bazel is compatible with JAX's required Bazel
  version.
  """
  system_bazel_version = subprocess.check_output([bazel_path, "--version"]).strip().decode('UTF-8')
  # `bazel --version` returns the version as "bazel a.b.c" so we split the
  # result to get only the version numbers.
  system_bazel_version = system_bazel_version.split(" ")[1]
  expected_bazel_version = get_jax_supported_bazel_version()
  if expected_bazel_version != system_bazel_version:
    logger.debug("Bazel version mismatch. JAX requires %s but got %s when `%s --version` was run", expected_bazel_version, system_bazel_version, bazel_path)
    return False
  return True

def guess_bazel_paths(bazel_path_flag):
  """
  Yields a sequence of guesses about bazel path. Some of sequence elements
  can be None. The resulting iterator is lazy and potentially has a side
  effects.
  """
  yield bazel_path_flag
  # For when Bazelisk was downloaded and is present on the root JAX directory
  yield shutil.which("./bazel")
  yield shutil.which("bazel")

def download_and_verify_bazelisk():
  """Downloads and verifies Bazelisk."""
  system  = platform.system()
  machine = platform.machine()
  downloaded_filename = "bazel"
  expected_sha256 = BAZELISK_PACKAGES[system, machine].sha256

  # Download Bazelisk and store it as "bazel".
  logger.debug("Downloading Bazelisk...")
  _, _ = urllib.request.urlretrieve(BAZELISK_BASE_URI + BAZELISK_PACKAGES[system, machine].file, downloaded_filename)

  with open(downloaded_filename, "rb") as downloaded_file:
    contents = downloaded_file.read()

  calculated_sha256 = hashlib.sha256(contents).hexdigest()

  # Verify checksum
  logger.debug("Verifying the checksum...")
  if calculated_sha256 != expected_sha256:
    raise ValueError("SHA256 checksum mismatch. Download may be corrupted.")
  logger.debug("Checksum verified!")

  logger.debug("Setting the Bazelisk binary to executable mode...")
  subprocess.run(["chmod", "+x", downloaded_filename], check=True)

  return os.path.realpath(downloaded_filename)


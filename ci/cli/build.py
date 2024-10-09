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
# CLI for building JAX artifacts.
import argparse
import asyncio
import logging
import os
import platform
import collections
import sys
import subprocess
from helpers import command, tools

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

BANNER = r"""
     _   _  __  __
    | | / \ \ \/ /
 _  | |/ _ \ \  /
| |_| / ___ \/  \
 \___/_/   \/_/\_\

"""

EPILOG = """
From the root directory of the JAX repository, run
    python ci/cli/build.py [jaxlib | jax-cuda-plugin | jax-cuda-pjrt | jax-rocm-plugin | jax-rocm-pjrt]
or
    python3 ci/cli/build.py [jaxlib | jax-cuda-plugin | jax-cuda-pjrt | jax-rocm-plugin | jax-rocm-pjrt]

to build one of: jaxlib, jax-cuda-plugin, jax-cuda-pjrt, jax-rocm-plugin, or jax-rocm-pjrt.
"""

ArtifactBuildSpec = collections.namedtuple(
    "ArtifactBuildSpec",
    ["bazel_build_target", "wheel_binary"],
)

# Define the build target and resulting wheel binary for each artifact.
ARTIFACT_BUILD_TARGET_DICT = {
    "jaxlib": ArtifactBuildSpec("//jaxlib/tools:build_wheel", "bazel-bin/jaxlib/tools/build_wheel"),
    "jax-cuda-plugin": ArtifactBuildSpec("//jaxlib/tools:build_gpu_kernels_wheel", "bazel-bin/jaxlib/tools/build_gpu_kernels_wheel"),
    "jax-cuda-pjrt": ArtifactBuildSpec("//jaxlib/tools:build_gpu_plugin_wheel", "bazel-bin/jaxlib/tools/build_gpu_plugin_wheel"),
    "jax-rocm-plugin": ArtifactBuildSpec("//jaxlib/tools:build_gpu_kernels_wheel", "bazel-bin/jaxlib/tools/build_gpu_kernels_wheel"),
    "jax-rocm-pjrt": ArtifactBuildSpec("//jaxlib/tools:build_gpu_plugin_wheel", "bazel-bin/jaxlib/tools/build_gpu_plugin_wheel"),
}

def get_bazelrc_config(os_name: str, arch: str, artifact: str, mode:str, use_rbe: bool):
  """
  Returns the bazelrc config for the given architecture, OS, and build mode.
  Args:
    os_name: The name of the OS.
    arch: The architecture of the host system.
    artifact: The artifact to build.
    mode: CLI build mode.
    use_rbe: Whether to use RBE.
  """

  # When building ROCm packages, we only inherit `--config=rocm` from .bazelrc
  if "rocm" in artifact:
    logger.debug("Building ROCm package. Using --config=rocm.")
    return "rocm"

  bazelrc_config = f"{os_name}_{arch}"

  # When the CLI is run by invoking ci/build_artifacts.sh, the CLI runs in CI
  # mode and will use one of the "ci_" configs in the .bazelrc. We want to run
  # certain CI builds with RBE and we also want to allow users the flexibility
  # to build JAX artifacts either by running the CLI or by running
  # ci/build_artifacts.sh. Because RBE requires permissions, we cannot enable it
  # by default in ci/build_artifacts.sh. Instead, we have the CI builds set
  # JAXCI_BUILD_ARTIFACT_WITH_RBE to 1 to enable RBE.
  if os.environ.get("JAXCI_BUILD_ARTIFACT_WITH_RBE", "0") == "1":
    use_rbe = True

  # In CI builds, we want to use RBE where possible. At the moment, RBE is only
  # supported on Linux x86 and Windows. If an user is requesting RBE, the CLI
  # will use RBE if the host system supports it, otherwise it will use the
  # local config.
  if use_rbe and ((os_name == "linux" and arch == "x86_64") \
      or (os_name == "windows" and arch == "amd64")):
    bazelrc_config = "rbe_" + bazelrc_config
  elif mode == "local":
    # Show warning if RBE is requested on an unsupported platform.
    if use_rbe:
      logger.warning("RBE is not supported on %s_%s. Using Local config instead.", os_name, arch)

    # If building `jaxlib` on Linux Aarch64, we use the default configs. No
    # custom local config is present in JAX's .bazelrc.
    if os_name == "linux" and arch == "aarch64" and artifact == "jaxlib":
      logger.debug("Linux Aarch64 CPU builds do not have custom local config in JAX's root .bazelrc. Running with default configs.")
      bazelrc_config = ""
      return bazelrc_config

    bazelrc_config = "local_" + bazelrc_config
  else:
    # Show warning if RBE is requested on an unsupported platform.
    if use_rbe:
      logger.warning("RBE is not supported on %s_%s. Using CI config instead.", os_name, arch)

    # Let user know that RBE is available for this platform.
    if (os_name == "linux" and arch == "x86_64")or (os_name == "windows" and arch == "amd64"):
      logger.info("RBE support is available for this platform. If you want to use RBE and have the required permissions, run the CLI with `--use_rbe` or set `JAXCI_BUILD_ARTIFACT_WITH_RBE=1`")

    bazelrc_config = "ci_" + bazelrc_config

  # When building jax-cuda-plugin or jax-cuda-pjrt, append "_cuda" to the
  # bazelrc config to use the CUDA specific configs.
  if artifact == "jax-cuda-plugin" or artifact == "jax-cuda-pjrt":
    bazelrc_config = bazelrc_config + "_cuda"

  return bazelrc_config

def get_jaxlib_git_hash():
  """Returns the git hash of the current repository."""
  res = subprocess.run(
      ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True
  )
  return res.stdout

# Set Clang as the C++ compiler if requested. CI builds use Clang by default
# via the toolchain used by the "ci_" configs in the .bazelrc. For Local builds,
# Bazel uses the default C++ compiler on the system which is GCC for Linux and
# MSVC for Windows.
def set_clang_as_compiler(bazel_command: command.CommandBuilder, clang_path: str):
  """
  Sets Clang as the C++ compiler in the Bazel command.
  Args:
    bazel_command: An instance of command.CommandBuilder.
    clang_path: The path to Clang.
  """
  # Find the path to Clang.
  absolute_clang_path = tools.get_clang_path(clang_path)
  if absolute_clang_path:
    logger.debug("Adding Clang as the C++ compiler to Bazel...")
    bazel_command.append(f"--action_env CLANG_COMPILER_PATH='{absolute_clang_path}'")
    bazel_command.append(f"--repo_env CC='{absolute_clang_path}'")
    bazel_command.append(f"--repo_env BAZEL_COMPILER='{absolute_clang_path}'")
    # Inherit Clang specific settings from the .bazelrc
    bazel_command.append("--config=clang")
  else:
    logger.debug("Could not find path to Clang. Continuing without Clang.")

def adjust_paths_for_windows(wheel_binary: str, output_dir: str, arch: str) -> tuple[str, str, str]:
  """
  Adjusts the paths to be compatible with Windows.
  Args:
    wheel_binary: The path to the wheel binary that was built by Bazel.
    output_dir: The output directory for the wheel.
    arch: The architecture of the host system.
  Returns:
    A tuple of the adjusted paths.
  """
  logger.debug("Adjusting paths for Windows...")
  # On Windows, the wheel binary has a .exe extension. and the path needs
  # to be adjusted to use backslashes.
  wheel_binary = wheel_binary.replace("/", "\\") + ".exe"
  output_dir = output_dir.replace("/", "\\")

  # Change to upper case to match the case in
  # "jax/tools/build_utils.py" for Windows.
  arch = arch.upper()

  return (wheel_binary, output_dir, arch)

def parse_and_append_bazel_options(bazel_command: command.CommandBuilder, bazel_options: str):
  """
  Parses the bazel options and appends them to the bazel command.
  Args:
    bazel_command: An instance of command.CommandBuilder.
    bazel_options: The bazel options to parse and append.
  """
  for option in bazel_options.split(" "):
    bazel_command.append(option)

def construct_requirements_update_command(bazel_command: command.CommandBuilder, additional_build_options: str, python_version: str, update_nightly: bool):
  """
  Constructs the Bazel command to run the requirements update.
  Args:
    bazel_command: An instance of command.CommandBuilder.
    additional_build_options: Additional build options to pass to Bazel.
    python_version: Hermetic Python version to use.
    update_nightly: Whether to update the nightly requirements file.
  """
  bazel_command.append("run")

  if python_version:
    logging.debug("Setting Hermetic Python version to %s", python_version)
    bazel_command.append(f"--repo_env=HERMETIC_PYTHON_VERSION={python_version}")

  if additional_build_options:
    logging.debug("Using additional build options: %s", additional_build_options)
    parse_and_append_bazel_options(bazel_command, additional_build_options)

  if update_nightly:
    bazel_command.append("//build:requirements_nightly.update")
  else:
    bazel_command.append("//build:requirements.update")

def add_python_version_argument(parser: argparse.ArgumentParser):
  """
  Add Python version argument to the parser.
  Args:
    parser: An instance of argparse.ArgumentParser.
  """
  parser.add_argument(
      "--python_version",
      type=str,
      choices=["3.10", "3.11", "3.12"],
      default="3.12",
      help="Python version to use",
  )

def add_cuda_version_argument(parser: argparse.ArgumentParser):
  """
  Add CUDA version argument to the parser.
  Args:
    parser: An instance of argparse.ArgumentParser.
  """
  parser.add_argument(
      "--cuda_version",
      type=str,
      default="12.3.2",
      help="CUDA version to use",
  )

def add_cudnn_version_argument(parser: argparse.ArgumentParser):
  """
  Add cuDNN version argument to the parser.
  Args:
    parser: An instance of argparse.ArgumentParser.
  """
  parser.add_argument(
      "--cudnn_version",
      type=str,
      default="9.1.1",
      help="cuDNN version to use",
  )

def add_disable_nccl_argument(parser: argparse.ArgumentParser):
  """
  Add an argument to allow disabling NCCL for CUDA/ROCM builds.
  Args:
    parser: An instance of argparse.ArgumentParser.
  """
  parser.add_argument(
      "--disable_nccl",
      action="store_true",
      help="Whether to disable NCCL for CUDA/ROCM builds.",
  )

def add_cuda_compute_capabilities_argument(parser: argparse.ArgumentParser):
  """
  Add an argument to set the CUDA compute capabilities.
  Args:
    parser: An instance of argparse.ArgumentParser.
  """
  parser.add_argument(
      "--cuda_compute_capabilities",
      type=str,
      default=None,
      help="A comma-separated list of CUDA compute capabilities to support.",
  )

def add_rocm_version_argument(parser: argparse.ArgumentParser):
  """
  Add ROCm version argument to the parser.
  Args:
    parser: An instance of argparse.ArgumentParser.
  """
  parser.add_argument(
      "--rocm_version",
      type=str,
      default="60",
      help="ROCm version to use",
  )


def add_rocm_amdgpu_targets_argument(parser: argparse.ArgumentParser):
  """
  Add an argument to set the ROCm amdgpu targets.
  Args:
    parser: An instance of argparse.ArgumentParser.
  """
  parser.add_argument(
      "--rocm_amdgpu_targets",
      type=str,
      default="gfx900,gfx906,gfx908,gfx90a,gfx1030",
      help="A comma-separated list of ROCm amdgpu targets to support.",
  )

def add_rocm_path_argument(parser: argparse.ArgumentParser):
  """
  Add an argument to set the ROCm toolkit path.
  Args:
    parser: An instance of argparse.ArgumentParser.
  """
  parser.add_argument(
      "--rocm_path",
      type=str,
      default="",
      help="Path to the ROCm toolkit.",
  )

def add_global_arguments(parser: argparse.ArgumentParser):
  """
  Add global arguments to the parser.
  Args:
    parser: An instance of argparse.ArgumentParser.
  """
  # Set the build mode. This is used to determine the Bazelrc config to use.
  # Local selects the "local_" config and CI selects the "ci_" config. CI
  # configs inherit local configs and set a custom C++ toolchain that needs to
  # be present on the system.
  parser.add_argument(
      "--mode",
      type=str,
      choices=["ci", "local"],
      default="local",
      help="""
        Sets the build mode to use.
        If set to "ci", the CLI will assume the build is being run in CI or CI
        like environment and will use the "ci_" configs in the .bazelrc.
        If set to "local", the CLI will use the "local_" configs in the
        .bazelrc.
        CI configs inherit the local configs and set a custom C++ toolchain to
        use Clang and specific versioned standard libraries. As a result, CI
        configs require the toolchain to be present on the system.
        When set to local, Bazel will use the default C++ compiler on the
        system which is GCC for Linux and MSVC for Windows. If you want to use
        Clang for local builds, use the `--use_clang` flag.
        """,
  )

  # If set, the build will create an 'editable' build instead of a wheel.
  parser.add_argument(
    "--editable",
    action="store_true",
    help=
      "Create an 'editable' build instead of a wheel.",
  )

  # Set Path to Bazel binary
  parser.add_argument(
      "--bazel_path",
      type=str,
      default="",
      help=
        """
        Path to the Bazel binary to use. The default is to find bazel via the
        PATH; if none is found, downloads a fresh copy of Bazelisk from GitHub.
        """,
  )

  # Use Clang as the C++ compiler. CI builds use Clang by default via the
  # toolchain used by the "ci_" configs in the .bazelrc.
  parser.add_argument(
    "--use_clang",
    action="store_true",
    help=
      """
      If set, the build will use Clang as the C++ compiler. Requires Clang to
      be present on the PATH or a path is given with --clang_path. CI builds use
      Clang by default.
      """,
  )

  # Set the path to Clang. If not set, the build will attempt to find Clang on
  # the PATH.
  parser.add_argument(
    "--clang_path",
    type=str,
    default="",
    help=
      """
      Path to the Clang binary to use. If not set and --use_clang is set, the
      build will attempt to find Clang on the PATH.
      """,
  )

  # Use RBE if available. Only available for Linux x86 and Windows and requires
  # permissions.
  parser.add_argument(
      "--use_rbe",
      action="store_true",
      help=
        """
        If set, the build will use RBE where possible. Currently, only Linux x86
        and Windows builds can use RBE. On other platforms, setting this flag will
        be a no-op. RBE requires permissions to JAX's remote worker pool. Only
        Googlers and CI builds can use RBE.
        """,
  )

  # Set the path to local XLA repository. If not set, the build will use the
  # XLA at the pinned version in workspace.bzl. CI builds set this via the
  # JAXCI_XLA_GIT_DIR environment variable.
  parser.add_argument(
    "--local_xla_path",
    type=str,
    default=os.environ.get("JAXCI_XLA_GIT_DIR", ""),
    help=
      """
      Path to local XLA repository to use. If not set, Bazel uses the XLA
      at the pinned version in workspace.bzl.
      """,
  )

  # Enabling native arch features will add --config=native_arch_posix to the
  # Bazel command. This enables -march=native, which generates code targeted to
  # use all features of the current machine. Not supported on Windows.
  parser.add_argument(
      "--enable_native_arch_features",
      action="store_true",
      help="Enables `-march=native` which generates code targeted to use all"
           "features of the current machine. (not supported on Windows)",
  )

  # Enabling MKL DNN will add --config=mkl_open_source_only to the Bazel
  # command.
  parser.add_argument(
      "--enable_mkl_dnn",
      action="store_true",
      help="Enables MKL-DNN.",
  )

  # Additional startup options to pass to Bazel.
  parser.add_argument(
      "--bazel_startup_options",
      type=str,
      default="",
      help="Space separated list of additional startup options to pass to Bazel."
           "E.g. --bazel_startup_options='--nobatch --noclient_debug'"
  )

  # Additional build options to pass to Bazel.
  parser.add_argument(
      "--bazel_build_options",
      type=str,
      default="",
      help="Space separated list of additional build options to pass to Bazel."
           "E.g. --bazel_build_options='--local_resources=HOST_CPUS --nosandbox_debug'"
  )

  # Directory in which artifacts should be stored.
  parser.add_argument(
      "--output_dir",
      type=str,
      default=os.environ.get("JAXCI_OUTPUT_DIR", os.path.join(os.getcwd(), "dist")),
      help="Directory in which artifacts should be stored."
  )

  parser.add_argument(
      "--requirements_update",
      action="store_true",
      help="If true, writes a .bazelrc and updates requirements_lock.txt for a"
            "corresponding version of Python but does not build any artifacts."
  )

  parser.add_argument(
      "--requirements_nightly_update",
      action="store_true",
      help="Same as update_requirements, but will consider dev, nightly and"
            "pre-release versions of packages."
  )

  # Use to invoke a dry run of the build. This will print the Bazel command that
  # will be invoked but will not execute it.
  parser.add_argument(
      "--dry_run",
      action="store_true",
      help="Prints the Bazel command that is going will be invoked.",
  )

  # Use to enable verbose logging.
  parser.add_argument(
      "--verbose",
      action="store_true",
      help="Prodcue verbose output for debugging.",
  )

async def main():
  parser = argparse.ArgumentParser(
      description=(
          "CLI for building one of the following packages from source: jaxlib, "
          "jax-cuda-plugin, jax-cuda-pjrt, jax-rocm-plugin, jax-rocm-pjrt."
      ),
      epilog=EPILOG,
  )

  # Create subparsers for jax, jaxlib, plugin, pjrt
  subparsers = parser.add_subparsers(
      dest="command", required=True, help="Artifact to build"
  )

  # jaxlib subcommand
  jaxlib_parser = subparsers.add_parser("jaxlib", help="Builds the jaxlib package.")
  add_global_arguments(jaxlib_parser)
  add_python_version_argument(jaxlib_parser)

  # jax-cuda-plugin subcommand
  cuda_plugin_parser = subparsers.add_parser("jax-cuda-plugin", help="Builds the jax-cuda-plugin package.")
  add_global_arguments(cuda_plugin_parser)
  add_python_version_argument(cuda_plugin_parser)
  add_cuda_version_argument(cuda_plugin_parser)
  add_cudnn_version_argument(cuda_plugin_parser)
  add_cuda_compute_capabilities_argument(cuda_plugin_parser)
  add_disable_nccl_argument(cuda_plugin_parser)

  # jax-cuda-pjrt subcommand
  cuda_pjrt_parser = subparsers.add_parser("jax-cuda-pjrt", help="Builds the jax-cuda-pjrt package.")
  add_global_arguments(cuda_pjrt_parser)
  add_cuda_version_argument(cuda_pjrt_parser)
  add_cudnn_version_argument(cuda_pjrt_parser)
  add_cuda_compute_capabilities_argument(cuda_pjrt_parser)
  add_disable_nccl_argument(cuda_pjrt_parser)

  # jax-rocm-plugin subcommand
  rocm_plugin_parser = subparsers.add_parser("jax-rocm-plugin", help="Builds the jax-rocm-plugin package.")
  add_global_arguments(rocm_plugin_parser)
  add_python_version_argument(rocm_plugin_parser)
  add_rocm_version_argument(rocm_plugin_parser)
  add_rocm_amdgpu_targets_argument(rocm_plugin_parser)
  add_rocm_path_argument(rocm_plugin_parser)
  add_disable_nccl_argument(rocm_plugin_parser)

  # jax-rocm-pjrt subcommand
  rocm_pjrt_parser = subparsers.add_parser("jax-rocm-pjrt", help="Builds the jax-rocm-pjrt package.")
  add_global_arguments(rocm_pjrt_parser)
  add_rocm_version_argument(rocm_pjrt_parser)
  add_rocm_amdgpu_targets_argument(rocm_pjrt_parser)
  add_rocm_path_argument(rocm_pjrt_parser)
  add_disable_nccl_argument(rocm_pjrt_parser)

  # Get the host systems architecture
  arch = platform.machine().lower()
  # Get the host system OS
  os_name = platform.system().lower()

  args = parser.parse_args()

  logger.info("%s", BANNER)

  if args.verbose:
    logging.getLogger().setLevel(logging.DEBUG)
    logger.info("Verbose logging enabled.")

  logger.info(
      "Building %s for %s %s...",
      args.command,
      os_name,
      arch,
  )

  # Find the path to Bazel
  bazel_path = tools.get_bazel_path(args.bazel_path)

  executor = command.SubprocessExecutor()

  # Start constructing the Bazel command
  bazel_command = command.CommandBuilder(bazel_path)

  if args.bazel_startup_options:
    logging.debug("Using additional Bazel startup options: %s", args.bazel_startup_options)
    parse_and_append_bazel_options(bazel_command, args.bazel_startup_options)

  # Temporary; when we make the new scripts as the default we can remove this.
  bazel_command.append("--bazelrc=ci/.bazelrc")

  # If the user requested a requirements update, construct the command and
  # execute it. Exit without building any artifacts.
  if args.requirements_update or args.requirements_nightly_update:
    python_version = args.python_version if hasattr(args, "python_version") else ""
    construct_requirements_update_command(bazel_command, args.bazel_build_options, python_version, args.requirements_nightly_update)
    await executor.run(bazel_command.command, args.dry_run)
    sys.exit(0)

  bazel_command.append("build")

  if args.enable_native_arch_features:
    logging.debug("Enabling native target CPU features.")
    bazel_command.append("--config=native_arch_posix")

  if args.enable_mkl_dnn:
    logging.debug("Enabling MKL DNN.")
    bazel_command.append("--config=mkl_open_source_only")

  if hasattr(args, "disable_nccl") and args.disable_nccl:
    logging.debug("Disabling NCCL.")
    bazel_command.append("--config=nonccl")

  if hasattr(args, "cuda_compute_capabilities"):
    logging.debug("Setting CUDA compute capabilities to %s", args.cuda_compute_capabilities)
    bazel_command.append(f"--repo_env HERMETIC_CUDA_COMPUTE_CAPABILITIES={args.cuda_compute_capabilities}")

  # Set Clang as the C++ compiler if requested. If Clang cannot be found, the
  # build will continue without Clang and instead use the system default.
  if args.use_clang or args.clang_path:
    set_clang_as_compiler(bazel_command, args.clang_path)

  if args.mode == "ci":
    logging.debug("Running in CI mode. Run the CLI with --help for more details on what this means.")

  # JAX's .bazelrc has custom configs for each build type, architecture, and
  # OS. Fetch the appropriate config and pass it to Bazel. A special case is
  # when building for Linux Aarch64, which does not have a custom local config
  # in JAX's .bazelrc. In this case, we build with the default configs.
  # When building ROCm packages, we only use `--config=rocm` from .bazelrc.
  bazelrc_config = get_bazelrc_config(os_name, arch, args.command, args.mode, args.use_rbe)
  if bazelrc_config:
    logging.debug("Using --config=%s from .bazelrc", bazelrc_config)
    bazel_command.append(f"--config={bazelrc_config}")

  # Check if a local XLA path is set.
  # When building artifacts for running tests, we use clone XLA at HEAD into
  # JAXCI_XLA_GIT_DIR and use that for building the artifacts.
  if args.local_xla_path:
    logging.debug("Setting local XLA path to %s", args.local_xla_path)
    bazel_command.append(f"--override_repository=xla={args.local_xla_path}")

  # Set the Hermetic Python version.
  if hasattr(args, "python_version"):
    logging.debug("Setting Hermetic Python version to %s", args.python_version)
    bazel_command.append(f"--repo_env=HERMETIC_PYTHON_VERSION={args.python_version}")
  else:
    # While pjrt packages do not use the Python version, we set the default
    # as 3.12 because Heremtic Python uses the system default if not Python
    # version is set. On the Linux Arm64 Docker image, the system default is
    # Python 3.9 which is not supported by JAX.
    # TODO(srnitin): Update the Docker images so that we can remove this.
    bazel_command.append("--repo_env=HERMETIC_PYTHON_VERSION=3.12")

  # Set the CUDA and cuDNN versions if they are not the default. Default values
  # are set in the .bazelrc.
  if "cuda" in args.command:
    if args.cuda_version != "12.3.2":
      logging.debug("Setting Hermetic CUDA version to %s", args.cuda_version)
      bazel_command.append(f"--repo_env=HERMETIC_CUDA_VERSION={args.cuda_version}")
    if args.cudnn_version != "9.1.1":
      logging.debug("Setting Hermetic cuDNN version to %s", args.cudnn_version)
      bazel_command.append(f"--repo_env=HERMETIC_CUDNN_VERSION={args.cudnn_version}")

  # If building ROCM packages, set the ROCm path and ROCm AMD GPU targets.
  if "rocm" in args.command:
    if args.rocm_path:
      logging.debug("Setting ROCm path to %s", args.rocm_path)
      bazel_command.append(f"--action_env ROCM_PATH='{args.rocm_path}'")
    if args.rocm_amdgpu_targets:
      logging.debug("Setting ROCm AMD GPU targets to %s", args.rocm_amdgpu_targets)
      bazel_command.append(f"--action_env TF_ROCM_AMDGPU_TARGETS={args.rocm_amdgpu_targets}")

  # Append any user specified Bazel build options.
  if args.bazel_build_options:
    logging.debug("Using additional Bazel build options: %s", args.bazel_build_options)
    parse_and_append_bazel_options(bazel_command, args.bazel_build_options)

  # Append the build target to the Bazel command.
  build_target, wheel_binary = ARTIFACT_BUILD_TARGET_DICT[args.command]
  bazel_command.append(build_target)

  # Execute the Bazel command.
  await executor.run(bazel_command.command, args.dry_run)

  # Construct the wheel build command.
  logger.info("Constructing wheel build command...")

  # Read output directory. Default is store the artifacts in the "dist/"
  # directory in JAX's GitHub repository root.
  output_dir = args.output_dir

  # If running on Windows, adjust the paths for compatibility.
  if os_name == "windows":
    wheel_binary, output_dir, arch = adjust_paths_for_windows(
        wheel_binary, output_dir, arch
    )

  logger.debug("Storing artifacts in %s", output_dir)

  run_wheel_binary = command.CommandBuilder(wheel_binary)

  if args.editable:
    logger.debug("Building an editable build.")
    output_dir = os.path.join(output_dir, args.command)
    run_wheel_binary.append("--editable")

  run_wheel_binary.append(f"--output_path={output_dir}")
  run_wheel_binary.append(f"--cpu={arch}")

  if "cuda" in args.command:
    run_wheel_binary.append("--enable-cuda=True")
    major_cuda_version = args.cuda_version.split(".")[0]
    run_wheel_binary.append(f"--platform_version={major_cuda_version}")

  if "rocm" in args.command:
    run_wheel_binary.append("--enable-rocm=True")
    run_wheel_binary.append(f"--platform_version={args.rocm_version}")

  jaxlib_git_hash = get_jaxlib_git_hash()
  run_wheel_binary.append(f"--jaxlib_git_hash={jaxlib_git_hash}")

  # Execute the wheel build command.
  await executor.run(run_wheel_binary.command, args.dry_run)

if __name__ == "__main__":
  asyncio.run(main())

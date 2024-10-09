#!/bin/bash
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
#
# Set up the build environment for JAX CI jobs. This script depends on the
# environment variables set in `setup_envs.sh`.
# -e: abort script if one command fails
# -u: error if undefined variable used
# -x: log all commands
# -o pipefail: entire command fails if pipe fails. watch out for yes | ...
# -o history: record shell history
# -o allexport: export all functions and variables to be available to subscripts
set -exuo pipefail -o history -o allexport

# Pre-emptively mark the git directory as safe. This is necessary for JAX CI
# jobs running on Linux runners in GitHub Actions. Without this, git complains
# that the directory has dubious ownership and refuses to run any commands.
# Avoid running on Windows runners as git runs into issues with not being able
# to lock the config file. Other git commands seem to work the on Windows
# runners so we can skip this step.
if [[ ! $(uname -s) =~ "MSYS_NT" ]]; then
  git config --global --add safe.directory $JAXCI_JAX_GIT_DIR
fi

# When building release artifacts, check out the release tag. JAX CI jobs build
# from the main branch by default.
if [[ -n "$JAXCI_RELEASE_TAG" ]]; then
  git checkout tags/"$JAXCI_RELEASE_TAG"
fi

# When running tests, we need to check out XLA at HEAD.
if [[ -z ${JAXCI_XLA_GIT_DIR} ]] && [[ "$JAXCI_CLONE_MAIN_XLA" == 1 ]]; then
    if [[ ! -d $(pwd)/xla ]]; then
      echo "Cloning XLA at HEAD to $(pwd)/xla"
      git clone --depth=1 https://github.com/openxla/xla.git $(pwd)/xla
    fi
    export JAXCI_XLA_GIT_DIR=$(pwd)/xla
fi

# If a path to XLA is provided, use that to build JAX or run tests.
if [[ ! -z ${JAXCI_XLA_GIT_DIR} ]]; then
  echo "Overriding XLA to be read from $JAXCI_XLA_GIT_DIR instead of the pinned"
  echo "version in the WORKSPACE."
  echo "If you would like to revert this behavior, unset JAXCI_XLA_GIT_DIR and"
  echo "JAXCI_CLONE_MAIN_XLA in your environment."

  # If a XLA commit is provided, check out XLA at that commit.
  if [[ ! -z "$JAXCI_XLA_COMMIT" ]]; then
    pushd "$JAXCI_XLA_GIT_DIR"

    git fetch --depth=1 origin "$JAXCI_XLA_COMMIT"
    echo "JAXCI_XLA_COMMIT is set. Checking out XLA at $JAXCI_XLA_COMMIT"
    git checkout "$JAXCI_XLA_COMMIT"

    popd
  fi
fi

# Setup check_if_to_run_in_docker, a helper function for executing steps that
# can either be run locally or run under Docker.
# run_docker_container.sh, below, redefines it as "docker exec".
# Important: "check_if_to_run_in_docker foo | bar" is
# "( check_if_to_run_in_docker foo ) | bar", and
# not "check_if_to_run_in_docker (foo | bar)".
# Therefore, "check_if_to_run_in_docker" commands cannot include pipes -- which
# is probably for the better. If a pipe is necessary for something, it is
# probably complex. Write a well-documented script under utilities/ to
# encapsulate the functionality instead.
check_if_to_run_in_docker() { "$@"; }

# For Windows, convert MSYS Linux-like paths to Windows paths.
if [[ $(uname -s) =~ "MSYS_NT" ]]; then
  echo 'Converting MSYS Linux-like paths to Windows paths (for Docker, Python, etc.)'
  # Convert all "_DIR" variables to Windows paths.
  source <(python3 ./ci/utilities/convert_msys_paths_to_win_paths.py)
fi

# Set up and and run the Docker container if needed.
# Jobs running on GitHub actions do not invoke this script. They define the
# Docker image via the `container` field in the workflow file.
if [[ "$JAXCI_RUN_DOCKER_CONTAINER" == 1 ]]; then
  echo "Setting up the Docker container..."
  source ./ci/utilities/run_docker_container.sh
fi

# When running Pytests, we need to install the wheels locally.
if [[ "$JAXCI_INSTALL_WHEELS_LOCALLY" == 1 ]]; then
   echo "Installing wheels locally..."
   source ./ci/utilities/install_wheels_locally.sh
fi

# TODO: cleanup steps
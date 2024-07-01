#!/usr/bin/env bash
# Copyright 2022 The JAX Authors.
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

# Usage: ci_build.sh [--dockerfile <DOCKERFILE_PATH> --keep_image --py_version <PYTHON_VERSION>]
#                    <COMMAND>
#
# DOCKERFILE_PATH: (Optional) Path to the Dockerfile used for docer build.
#                  If this optional value is not supplied (via the --dockerfile flag)
#                  Dockerfile.ms (located in the same directory as this script)
#                  will be used.
# KEEP_IMAGE: (Optional) If this flag is set, the container will be committed as an image
#
# PYTHON_VERSION: Python version to use
#
# COMMAND: Command to be executed in the docker container
#
# ROCM_VERSION: ROCm repo version
#
# ROCM_PATH: ROCM path in the docker container
#
# Environment variables read by this script
# WORKSPACE
# XLA_REPO
# XLA_BRANCH
# XLA_CLONE_DIR
# BUILD_TAG
#

set -eux

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/build_common.sh"
CONTAINER_TYPE="rocm"

DOCKERFILE_PATH="${SCRIPT_DIR}/Dockerfile.ms"    
DOCKER_CONTEXT_PATH="${SCRIPT_DIR}"
KEEP_IMAGE="--rm"
KEEP_CONTAINER="--rm"
PYTHON_VERSION="3.10.0"
ROCM_VERSION="6.1.0" #Point to latest release
BASE_DOCKER="ubuntu:20.04"
CUSTOM_INSTALL=""
POSITIONAL_ARGS=()

RUNTIME_FLAG=0

while [[ $# -gt 0 ]]; do
  case $1 in
    --py_version)
      PYTHON_VERSION="$2"
      shift 2
      ;;
    --dockerfile)
      DOCKERFILE_PATH="$2"
      DOCKER_CONTEXT_PATH=$(dirname "${DOCKERFILE_PATH}")
      shift 2
      ;;
    --keep_image)
      KEEP_IMAGE=""
      shift 1
      ;;
    --runtime)
      RUNTIME_FLAG=1
      shift 1
      ;;
    --keep_container)
      KEEP_CONTAINER=""
      shift 1
      ;;
    --rocm_version)
      ROCM_VERSION="$2"
      shift 2
      ;;
    --rocm_path)
      ROCM_PATH="$2"
      shift 2
      ;;
    *)
      POSITIONAL_ARGS+=("$1")
      shift
      ;;
  esac
done

if [[ ! -f "${DOCKERFILE_PATH}" ]]; then
  die "Invalid Dockerfile path: \"${DOCKERFILE_PATH}\""
fi

ROCM_EXTRA_PARAMS="--device=/dev/kfd --device=/dev/dri --group-add video \
  --cap-add=SYS_PTRACE --security-opt seccomp=unconfined --shm-size 16G"

# Helper function to traverse directories up until given file is found.
function upsearch (){
  test / == "$PWD" && return || \
    test -e "$1" && echo "$PWD" && return || \
    cd .. && upsearch "$1"
}

# Set up WORKSPACE. 
WORKSPACE="${WORKSPACE:-$(upsearch WORKSPACE)}"
BUILD_TAG="${BUILD_TAG:-jax}"

# Determine the docker image name and BUILD_TAG.
DOCKER_IMG_NAME="${BUILD_TAG}.${CONTAINER_TYPE}"

# Under Jenkins matrix build, the build tag may contain characters such as
# commas (,) and equal signs (=), which are not valid inside docker image names.
DOCKER_IMG_NAME=$(echo "${DOCKER_IMG_NAME}" | sed -e 's/=/_/g' -e 's/,/-/g')

# Convert to all lower-case, as per requirement of Docker image names
DOCKER_IMG_NAME=$(echo "${DOCKER_IMG_NAME}" | tr '[:upper:]' '[:lower:]')

# Print arguments.
echo "WORKSPACE: ${WORKSPACE}"
echo "COMMAND: ${POSITIONAL_ARGS[*]}"
echo "BUILD_TAG: ${BUILD_TAG}"
echo "  (docker container name will be ${DOCKER_IMG_NAME})"
echo ""

echo "Building container (${DOCKER_IMG_NAME})..."
echo "Python Version (${PYTHON_VERSION})"
if [[ "${RUNTIME_FLAG}" -eq 1  ]]; then
  echo "Building (runtime) container (${DOCKER_IMG_NAME}) with Dockerfile($DOCKERFILE_PATH)..."
  docker build --target rt_build --tag ${DOCKER_IMG_NAME} \
        --build-arg PYTHON_VERSION=$PYTHON_VERSION \
        --build-arg ROCM_VERSION=$ROCM_VERSION \
        --build-arg BASE_DOCKER=$BASE_DOCKER \
        --build-arg CUSTOM_INSTALL=$CUSTOM_INSTALL \
      -f "${DOCKERFILE_PATH}" "${DOCKER_CONTEXT_PATH}"
else
  echo "Building (build) container (${DOCKER_IMG_NAME}) with Dockerfile($DOCKERFILE_PATH)..."
  docker build --target b_build --tag ${DOCKER_IMG_NAME} \
        --build-arg PYTHON_VERSION=$PYTHON_VERSION \
        --build-arg ROCM_VERSION=$ROCM_VERSION \
        --build-arg BASE_DOCKER=$BASE_DOCKER \
        --build-arg CUSTOM_INSTALL=$CUSTOM_INSTALL \
      -f "${DOCKERFILE_PATH}" "${DOCKER_CONTEXT_PATH}"  
fi

# Check docker build status
if [[ $? != "0" ]]; then
  die "ERROR: docker build failed. Dockerfile is at ${DOCKERFILE_PATH}"
fi

# Run the command inside the container.
echo "Running '${POSITIONAL_ARGS[*]}' inside ${DOCKER_IMG_NAME}..."

export XLA_REPO="${XLA_REPO:-}"
export XLA_BRANCH="${XLA_BRANCH:-}"
export XLA_CLONE_DIR="${XLA_CLONE_DIR:-}"
export JAX_RENAME_WHL="${XLA_CLONE_DIR:-}"

if [ ! -z ${XLA_CLONE_DIR} ]; then
	ROCM_EXTRA_PARAMS=${ROCM_EXTRA_PARAMS}" -v ${XLA_CLONE_DIR}:${XLA_CLONE_DIR}"
fi

docker run ${KEEP_IMAGE} --name ${DOCKER_IMG_NAME} --pid=host \
  -v ${WORKSPACE}:/workspace \
  -w /workspace \
  -e XLA_REPO=${XLA_REPO} \
  -e XLA_BRANCH=${XLA_BRANCH} \
  -e XLA_CLONE_DIR=${XLA_CLONE_DIR} \
  -e PYTHON_VERSION=$PYTHON_VERSION \
  -e CI_RUN=1 \
  ${ROCM_EXTRA_PARAMS} \
  "${DOCKER_IMG_NAME}" \
  ${POSITIONAL_ARGS[@]}

if [[ "${KEEP_IMAGE}" != "--rm" ]] && [[ $? == "0" ]]; then
  echo "Committing the docker container as ${DOCKER_IMG_NAME}"
  docker stop ${DOCKER_IMG_NAME}
  docker commit ${DOCKER_IMG_NAME} ${DOCKER_IMG_NAME}
  docker rm ${DOCKER_IMG_NAME}    # remove this temp container
fi

echo "Jax-ROCm build was successful!"

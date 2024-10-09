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
# Source JAXCI environment variables.
source "ci/utilities/setup_envs.sh" "$1"
# Set up the build environment.
source "ci/utilities/setup_build_environment.sh"

if [[ $JAXCI_RUN_BAZEL_TEST_CPU == 1 ]]; then
      os=$(uname -s | awk '{print tolower($0)}')
      arch=$(uname -m)

      # If running on Mac or Linux Aarch64, we only build the test targets and
      # not run them. These platforms do not have native RBE support so we
      # cross-compile them on the Linux x86 RBE pool. As the tests still need
      # to be run on the host machine and because running the tests on a single
      # machine can take a long time, we skip running them on these platforms.
      if [[ $os == "darwin" ]] || ( [[ $os == "linux" ]] && [[ $arch == "aarch64" ]] ); then
            echo "Building RBE CPU tests..."
            check_if_to_run_in_docker bazel --bazelrc=ci/.bazelrc build --config=rbe_cross_compile_${os}_${arch} \
                  --repo_env=HERMETIC_PYTHON_VERSION="$JAXCI_HERMETIC_PYTHON_VERSION" \
                  --override_repository=xla="${JAXCI_XLA_GIT_DIR}" \
                  --test_env=JAX_NUM_GENERATED_CASES=25 \
                  //tests:cpu_tests //tests:backend_independent_tests
      else
            echo "Running RBE CPU tests..."
            check_if_to_run_in_docker bazel --bazelrc=ci/.bazelrc test --config=rbe_${os}_${arch} \
                  --repo_env=HERMETIC_PYTHON_VERSION="$JAXCI_HERMETIC_PYTHON_VERSION" \
                  --override_repository=xla="${JAXCI_XLA_GIT_DIR}" \
                  --test_env=JAX_NUM_GENERATED_CASES=25 \
                  //tests:cpu_tests //tests:backend_independent_tests
      fi
fi

# Run Bazel GPU tests locally.
if [[ $JAXCI_RUN_BAZEL_TEST_GPU_LOCAL == 1 ]]; then
      check_if_to_run_in_docker nvidia-smi
      echo "Running local GPU tests..."

      check_if_to_run_in_docker "$JAXCI_PYTHON" -c "import jax; print(jax.default_backend()); print(jax.devices()); print(len(jax.devices()))"

      # Only Linux x86 builds run GPU tests
      # Runs non-multiaccelerator tests with one GPU apiece.
      # It appears --run_under needs an absolute path.
      check_if_to_run_in_docker bazel --bazelrc=ci/.bazelrc test --config=ci_linux_x86_64_cuda \
            --config=non_multiaccelerator_local \
            --repo_env=HERMETIC_PYTHON_VERSION="$JAXCI_HERMETIC_PYTHON_VERSION" \
            --override_repository=xla="${JAXCI_XLA_GIT_DIR}" \
            --run_under "${JAXCI_JAX_GIT_DIR}/build/parallel_accelerator_execute.sh" \
            //tests:gpu_tests //tests:backend_independent_tests //tests/pallas:gpu_tests //tests/pallas:backend_independent_tests

      # Runs multiaccelerator tests with all GPUs.
      check_if_to_run_in_docker bazel --bazelrc=ci/.bazelrc test --config=ci_linux_x86_64_cuda \
            --config=multiaccelerator_local \
            --repo_env=HERMETIC_PYTHON_VERSION="$JAXCI_HERMETIC_PYTHON_VERSION" \
            --override_repository=xla="${JAXCI_XLA_GIT_DIR}" \
            //tests:gpu_tests //tests/pallas:gpu_tests
fi

# Run Bazel GPU tests with RBE.
if [[ $JAXCI_RUN_BAZEL_TEST_GPU_RBE == 1 ]]; then
      check_if_to_run_in_docker nvidia-smi
      echo "Running RBE GPU tests..."

      # Only Linux x86 builds run GPU tests
      # Runs non-multiaccelerator tests with one GPU apiece.
      check_if_to_run_in_docker bazel --bazelrc=ci/.bazelrc test --config=rbe_linux_x86_64_cuda \
            --config=non_multiaccelerator \
            --repo_env=HERMETIC_PYTHON_VERSION="$JAXCI_HERMETIC_PYTHON_VERSION" \
            --override_repository=xla="${JAXCI_XLA_GIT_DIR}" \
            //tests:gpu_tests //tests:backend_independent_tests //tests/pallas:gpu_tests //tests/pallas:backend_independent_tests //docs/...
fi
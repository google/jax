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
# Source JAXCI environment variables.

# If the user has not passed in an JAXCI_ENV_FILE, exit.
if [[ -z "$1" ]]; then
    echo "ERROR: No argument passed."
    echo "setup_envs.sh requires that a path to a JAX CI env file be passed as"
    echo "an argument when invoking the build scripts."
    echo "If you are looking to build JAX artifacts, please pass in a"
    echo "corresponding env file from the ci/envs/build_artifacts directory."
    echo "If you are looking to run JAX tests, please pass in a"
    echo "corresponding env file from the ci/envs/run_tests directory."
    exit 1
fi

# Get the current environment variables and any user set JAXCI_ environment
# variables. We store these in a file and pass them to the Docker container
# when setting up the container in `run_docker_container.sh`.
# Store the current environment variables.
envs_before=$(mktemp)
env > "$envs_before"

# Read any JAXCI_ environment variables set by the user.
user_set_jaxci_envs=$(mktemp)
env | grep ^JAXCI_ > "$user_set_jaxci_envs"

# -e: abort script if one command fails
# -u: error if undefined variable used
# -x: log all commands
# -o pipefail: entire command fails if pipe fails. watch out for yes | ...
# -o history: record shell history
# -o allexport: export all functions and variables to be available to subscripts
set -exuo pipefail -o history -o allexport
source "$1"
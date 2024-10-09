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
# Set up the Docker container and start it for JAX CI jobs.

# Keep the existing "jax" container if it's already present.
if ! docker container inspect jax >/dev/null 2>&1 ; then
  # Simple retry logic for docker-pull errors. Sleeps if a pull fails.
  # Pulling an already-pulled container image will finish instantly, so
  # repeating the command costs nothing.
  docker pull "$JAXCI_DOCKER_IMAGE" || sleep 15
  docker pull "$JAXCI_DOCKER_IMAGE"

  if [[ "$(uname -s)" =~ "MSYS_NT" ]]; then
    # Docker on Windows doesn't support the `host` networking mode, and so
    # port-forwarding is required for the container to detect it's running on GCE.
    export IP_ADDR=$(powershell -command "(Get-NetIPAddress -AddressFamily IPv4 -InterfaceAlias 'vEthernet (nat)').IPAddress")
    netsh interface portproxy add v4tov4 listenaddress=$IP_ADDR listenport=80 connectaddress=169.254.169.254 connectport=80
    JAXCI_DOCKER_ARGS="$JAXCI_DOCKER_ARGS -e GCE_METADATA_HOST=$IP_ADDR"
  else
    # The volume mapping flag below shares the user's gcloud credentials, if any,
    # with the container, in case the user has credentials stored there.
    # This would allow Bazel to authenticate for RBE.
    # Note: JAX's CI does not have any credentials stored there.
    JAXCI_DOCKER_ARGS="$JAXCI_DOCKER_ARGS -v $HOME/.config/gcloud:/root/.config/gcloud"
  fi

  # If XLA repository on the local system is to be used, map it to the container
  # and set the JAXCI_XLA_GIT_DIR environment variable to the container path.
  if [[ -n $JAXCI_XLA_GIT_DIR ]]; then
    JAXCI_DOCKER_ARGS="$JAXCI_DOCKER_ARGS -v $JAXCI_XLA_GIT_DIR:$JAXCI_DOCKER_WORK_DIR/xla -e JAXCI_XLA_GIT_DIR=$JAXCI_DOCKER_WORK_DIR/xla"
  fi

  # Set the output directory to the container path.
  export JAXCI_OUTPUT_DIR=$JAXCI_DOCKER_WORK_DIR/dist

  # Capture the environment variables that get set by JAXCI_ENV_FILE and store
  # them in a file. This is needed so that we know which envs to set when
  # setting up the Docker container in `setup_docker.sh`. An easier solution
  # would be to just grep for "JAXCI_" variables but unfortunately, this is not
  # robust as there are some variables such as `JAX_ENABLE_X64`, `NCCL_DEBUG`,
  # etc that are used by JAX but do not have the `JAXCI_` prefix.
  envs_after=$(mktemp)
  env > "$envs_after"

  jax_ci_envs=$(mktemp)

  # Only get the new environment variables set by JAXCI_ENV_FILE. Use
  # "env_before" that gets set in setup.sh for the initial environment
  # variables. diff exits with a return code. This can end the build abrupty so
  # we use "|| true" to ignore the return code and continue.
  diff <(sort "$envs_before") <(sort "$envs_after") | grep "^> " | sed 's/^> //' | grep -v "^BASH_FUNC" > "$jax_ci_envs" || true

  # Start the container. `user_set_jaxci_envs` is read after `jax_ci_envs` to
  # allow the user to override any environment variables set by JAXCI_ENV_FILE.
  docker run --env-file $jax_ci_envs --env-file "$user_set_jaxci_envs" $JAXCI_DOCKER_ARGS --name jax \
      -w $JAXCI_DOCKER_WORK_DIR -itd --rm \
      -v "$JAXCI_JAX_GIT_DIR:$JAXCI_DOCKER_WORK_DIR" \
      "$JAXCI_DOCKER_IMAGE" \
    bash

  if [[ "$(uname -s)" =~ "MSYS_NT" ]]; then
    # Allow requests from the container.
    CONTAINER_IP_ADDR=$(docker inspect -f '{{range .NetworkSettings.Networks}}{{.IPAddress}}{{end}}' jax)
    netsh advfirewall firewall add rule name="Allow Metadata Proxy" dir=in action=allow protocol=TCP localport=80 remoteip="$CONTAINER_IP_ADDR"
  fi
fi

# Update `check_if_to_run_in_docker` to execute the commands inside the Docker
# container.
check_if_to_run_in_docker() { docker exec jax "$@"; }

# Update `JAXCI_OUTPUT_DIR`, `JAXCI_JAX_GIT_DIR` and `JAXCI_XLA_GIT_DIR` with
# the new Docker path on the host shell environment. This is needed because when
# running in Docker with `docker exec`, the commands are run on the host shell
# environment and as such the following variables need to be updated with The
# Docker paths.
export JAXCI_OUTPUT_DIR=$JAXCI_DOCKER_WORK_DIR/dist
export JAXCI_JAX_GIT_DIR=$JAXCI_DOCKER_WORK_DIR
export JAXCI_XLA_GIT_DIR=$JAXCI_DOCKER_WORK_DIR/xla

check_if_to_run_in_docker git config --global --add safe.directory $JAXCI_DOCKER_WORK_DIR
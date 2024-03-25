#!/usr/bin/env bash
# Copyright 2024 The JAX Authors. All Rights Reserved.
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

# script to run pip-compile for each requirement.
# if there is a change in requirements.in then all lock files will be updated
# accordingly

# All commands run relative to this directory
cd "$(dirname "${BASH_SOURCE[0]}")"

SUPPORTED_VERSIONS=("3_9" "3_10" "3_11" "3_12")

cp ../test-requirements.txt test-requirements.txt

if [[ -z $1 ]]; then
  echo "Updating all dependencies"
  SUFFIX=""
else
  echo "Updating jaxlib only."
  SUFFIX="_jaxlib"
  WHL="$1"
  # If arg starts with a digit followed by a dot, it's jaxlib a version from pypi
  if [[ "ARG" =~ ^[0-9]\. ]]; then
    echo -e "\njaxlib==$WHL" >> requirements.in
  else
  # otherwise it's a local path to .whl
    echo -e "\njaxlib @ $WHL" >> requirements.in
  fi
fi

for VERSION in "${SUPPORTED_VERSIONS[@]}"
do
  touch "requirements_lock_$VERSION.txt"
  bazel run --experimental_convenience_symlinks=ignore //:requirements_"$VERSION""$SUFFIX".update --enable_bzlmod=false
  sed -i '/^#/d' requirements_lock_"$VERSION".txt
  mv requirements_lock_"$VERSION".txt ../requirements_lock_"$VERSION".txt
done

rm test-requirements.txt
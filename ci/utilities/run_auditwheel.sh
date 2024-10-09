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
# Runs auditwheel to ensure manylinux compatibility.

# Get a list of all the wheels in the output directory. Only look for wheels
# that need to be verified for manylinux compliance.
WHEELS=$(find "$JAXCI_OUTPUT_DIR/" -type f \( -name "*jaxlib*" -o -name "*jax*cuda*pjrt*" -o -name "*jax*cuda*plugin*" \))

if [[ -z "$WHEELS" ]]; then
  echo "ERROR: No wheels found under $JAXCI_OUTPUT_DIR"
  exit 1
fi

for wheel in $WHEELS; do
    printf "\nRunning auditwheel on the following wheel:"
    ls $wheel
    OUTPUT_FULL=$(python3 -m auditwheel show $wheel)
    # Remove the wheel name from the output to avoid false positives.
    wheel_name=$(basename $wheel)
    OUTPUT=${OUTPUT_FULL//${wheel_name}/}

    # If a wheel is manylinux2014 compliant, `auditwheel show` will return the
    # platform tag as manylinux_2_17. manylinux2014 is an alias for
    # manylinux_2_17.
    if echo "$OUTPUT" | grep -q "manylinux_2_17"; then
        printf "\n$wheel_name is manylinux2014 compliant.\n"
    else
        echo "$OUTPUT_FULL"
        printf "\n$wheel_name is NOT manylinux2014 compliant.\n"
        exit 1
    fi
done
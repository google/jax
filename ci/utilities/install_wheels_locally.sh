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
# Install wheels stored in `JAXCI_OUTPUT_DIR` locally using the Python binary
# set in JAXCI_PYTHON. Use the absolute path to the `find` utility to avoid
# using the Windows version of `find` on Windows.
WHEELS=$(/usr/bin/find "$JAXCI_OUTPUT_DIR/" -type f \( -name "*jaxlib*" -o -name "*jax*cuda*pjrt*" -o -name "*jax*cuda*plugin*" \))

for wheel in "$WHEELS"; do
  echo "Installing $(basename $wheel) ..."
  "$JAXCI_PYTHON" -m pip install "$wheel"
done

echo "Installing the JAX package in editable mode at the current commit..."
# Install JAX package at the current commit.
"$JAXCI_PYTHON" -m pip install -U -e .
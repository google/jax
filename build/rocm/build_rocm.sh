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

# NOTE(mrodden): ROCm JAX build and installs have moved to wheel based builds and installs,
# but some CI scripts still try to run this script. Nothing needs to be done here,
# but we print some debugging information for logs.

set -eux
python -V

printf "Detected jaxlib version: %s\n" $(pip3 list | grep jaxlib | tr -s ' ' | cut -d " " -f 2 | cut -d "+" -f 1)
printf "Detected ROCm version: %s\n" $(cat /opt/rocm/.info/version | cut -d "-" -f 1)

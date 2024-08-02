#!/usr/bin/env python3

# Copyright 2024 The JAX Authors.
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


# NOTE(mrodden): This file is part of the ROCm build scripts, and
# needs be compatible with Python 3.6. Please do not include these
# in any "upgrade" scripts


import pprint
import re
import sys
import subprocess

"""
Utility for examining GLIBC versioned symbols
for an object file (shared object or ELF binary)
"""


def main():
    sofile = sys.argv[1]

    s = highest_for_file(sofile)

    print("%s: %r" % (sofile, s))


def highest_for_file(sofile):
    output = subprocess.check_output(["objdump", "-T", sofile])

    r = re.compile("\(GLIBC_(.*)\)")
    versions = {}

    for line in output.decode("utf-8").split("\n"):
        line = line.strip()
        match = r.search(line)
        if match:
            version_str = match.group(1)
            count = versions.get(version_str, 0)
            versions[version_str] = count + 1

    vtups = list(map(lambda x: parse(x), versions.keys()))
    s = sorted(vtups)

    return s[-1]


def parse(version_str):
    return tuple(map(int, version_str.split(".")))


if __name__ == "__main__":
    main()

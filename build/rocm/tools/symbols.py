#!/usr/bin/env python3


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

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


import os
import sys


def get_libc_version():
    """
    Detect and return glibc version that the current Python is linked against.

    This mimics the detection behavior of the 'wheel' and 'auditwheel' projects,
    but without any PyPy or libmusl support.
    """

    try:
        version_str = os.confstr("CS_GNU_LIBC_VERSION")
        return version_str
    except Exception:
        print("WARN: lookup by confstr failed", file=sys.stderr)
        pass

    try:
        import ctypes
    except ImportError:
        return None

    pn = ctypes.CDLL(None)
    print(dir(pn))

    try:
        gnu_get_libc_version = pn.gnu_get_libc_version
    except AttributeError:
        return None

    gnu_get_libc_version.restype = ctypes.c_char_p
    version_str = gnu_get_libc_version()

    return version_str


if __name__ == "__main__":
    print(get_libc_version())

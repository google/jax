#!/usr/bin/env python3


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

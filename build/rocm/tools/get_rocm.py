#!/usr/bin/env python3


# NOTE(mrodden): This file is part of the ROCm build scripts, and
# needs be compatible with Python 3.6. Please do not include these
# in any "upgrade" scripts


import argparse
import logging
import subprocess


LOG = logging.getLogger(__name__)


def which_linux():
    try:
        os_rel = open("/etc/os-release").read()

        kvs = {}
        for line in os_rel.split("\n"):
            if line.strip():
                k, v = line.strip().split("=", 1)
                v = v.strip('"')
                kvs[k] = v

        print(kvs)
    except OSError:
        pass


rocm_package_names = [
    "libdrm-amdgpu",
    "rocm-dev",
    "rocm-ml-sdk",
    "miopen-hip ",
    "miopen-hip-devel",
    "rocblas",
    "rocblas-devel",
    "rocsolver-devel",
    "rocrand-devel",
    "rocfft-devel",
    "hipfft-devel",
    "hipblas-devel",
    "rocprim-devel",
    "hipcub-devel",
    "rccl-devel",
    "hipsparse-devel",
    "hipsolver-devel",
]


def install_rocm_el8(rocm_version_str):

    with open("/etc/yum.repos.d/rocm.repo", "w") as rfd:
        rfd.write(
            """
[ROCm]
name=ROCm
baseurl=http://repo.radeon.com/rocm/rhel8/%s/main
enabled=1
gpgcheck=1
gpgkey=https://repo.radeon.com/rocm/rocm.gpg.key
"""
            % rocm_version_str
        )

    with open("/etc/yum.repos.d/amdgpu.repo", "w") as afd:
        afd.write(
            """
[amdgpu]
name=amdgpu
baseurl=https://repo.radeon.com/amdgpu/latest/rhel/8.8/main/x86_64/
enabled=1
gpgcheck=1
gpgkey=https://repo.radeon.com/rocm/rocm.gpg.key
"""
        )

    cmd = ["dnf", "install", "-y"]
    cmd.extend(rocm_package_names)
    LOG.info("Running %r" % cmd)
    subprocess.run(cmd, check=True)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--rocm-version", help="ROCm version to install", default="6.1.1")
    return p.parse_args()


def main():
    args = parse_args()
    install_rocm_el8(args.rocm_version)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()

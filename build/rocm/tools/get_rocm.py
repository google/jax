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


import argparse
import json
import logging
import os
import sys
import subprocess
import urllib.request


LOG = logging.getLogger(__name__)


def latest_rocm():
    dat = urllib.request.urlopen(
        "https://api.github.com/repos/rocm/rocm/releases/latest"
    ).read()
    rd = json.loads(dat)
    _, ver_str = rd["tag_name"].split("-")
    return ver_str


def os_release_meta():
    try:
        os_rel = open("/etc/os-release").read()

        kvs = {}
        for line in os_rel.split("\n"):
            if line.strip():
                k, v = line.strip().split("=", 1)
                v = v.strip('"')
                kvs[k] = v

        return kvs
    except OSError:
        pass


class System(object):

    def __init__(self, pkgbin, rocm_package_list):
        self.pkgbin = pkgbin
        self.rocm_package_list = rocm_package_list

    def install_packages(self, package_specs):
        cmd = [
            self.pkgbin,
            "install",
            "-y",
        ]
        cmd.extend(package_specs)

        env = dict(os.environ)
        if self.pkgbin == "apt":
            env["DEBIAN_FRONTEND"] = "noninteractive"

        LOG.info("Running %r" % cmd)
        subprocess.check_call(cmd, env=env)

    def install_rocm(self):
        self.install_packages(self.rocm_package_list)


UBUNTU = System(
    pkgbin="apt",
    rocm_package_list=[
        "rocm-dev",
        "rocm-libs",
    ],
)


RHEL8 = System(
    pkgbin="dnf",
    rocm_package_list=[
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
    ],
)


def get_system():
    md = os_release_meta()

    if md["ID"] == "ubuntu":
        return UBUNTU

    if md["ID"] in ["almalinux", "rhel", "fedora", "centos"]:
        if md["PLATFORM_ID"] == "platform:el8":
            return RHEL8

    raise Exception("No system for %r" % md)


def _setup_internal_repo(system, rocm_version, job_name, build_num):
    # wget is required by amdgpu-repo
    system.install_packages(["wget"])

    install_amdgpu_installer_internal(rocm_version)

    amdgpu_build = (
        urllib.request.urlopen(
            "http://rocm-ci.amd.com/job/%s/%s/artifact/amdgpu_kernel_info.txt"
            % (job_name, build_num)
        )
        .read()
        .decode("utf8")
        .strip()
    )

    cmd = [
        "amdgpu-repo",
        "--amdgpu-build=%s" % amdgpu_build,
        "--rocm-build=%s/%s" % (job_name, build_num),
    ]
    LOG.info("Running %r" % cmd)
    subprocess.check_call(cmd)

    cmd = [
        "amdgpu-install",
        "--no-dkms",
        "--usecase=rocm",
        "-y",
    ]

    env = dict(os.environ)
    if system.pkgbin == "apt":
        env["DEBIAN_FRONTEND"] = "noninteractive"

    LOG.info("Running %r" % cmd)
    subprocess.check_call(cmd, env=env)


def install_rocm(rocm_version, job_name=None, build_num=None):
    s = get_system()

    if job_name and build_num:
        _setup_internal_repo(s, rocm_version, job_name, build_num)
    else:
        if s == RHEL8:
            setup_repos_el8(rocm_version)
        elif s == UBUNTU:
            setup_repos_ubuntu(rocm_version)
        else:
            raise Exception("Platform not supported")

    s.install_rocm()


def install_amdgpu_installer_internal(rocm_version):
    """
    Download and install the "amdgpu-installer" package from internal builds
    on the current system.
    """
    md = os_release_meta()
    url, fn = _build_installer_url(rocm_version, md)

    try:
        # download installer
        LOG.info("Downloading from %s", url)
        urllib.request.urlretrieve(url, filename=fn)

        system = get_system()

        cmd = [system.pkgbin, "install", "-y", "./%s" % fn]
        subprocess.check_call(cmd)
    finally:
        try:
            os.remove(fn)
        except FileNotFoundError:
            pass


def _build_installer_url(rocm_version, metadata):
    md = metadata

    if isinstance(rocm_version, str):
        parts = rocm_version.split(".")
        rv = type("Version", (), {})()
        rv.major = parts[0]
        rv.minor = parts[1]

        if len(parts) > 2:
            rv.rev = parts[2]
    else:
        rv = rocm_version

    base_url = "http://artifactory-cdn.amd.com/artifactory/list"

    if md["ID"] == "ubuntu":
        fmt = "amdgpu-install-internal_%(rocm_major)s.%(rocm_minor)s-%(os_version)s-1_all.deb"
        package_name = fmt % {
            "rocm_major": rv.major,
            "rocm_minor": rv.minor,
            "os_version": md["VERSION_ID"],
        }

        url = "%s/amdgpu-deb/%s" % (base_url, package_name)
    elif md.get("PLATFORM_ID") == "platform:el8":
        fmt = "amdgpu-install-internal-%(rocm_major)s.%(rocm_minor)s_%(os_version)s-1.noarch.rpm"
        package_name = fmt % {
            "rocm_major": rv.major,
            "rocm_minor": rv.minor,
            "os_version": "8",
        }

        url = "%s/amdgpu-rpm/rhel/%s" % (base_url, package_name)
    else:
        raise Exception("Platform not supported: %r" % md)

    return url, package_name


def setup_repos_ubuntu(rocm_version_str):

    s = get_system()
    s.install_packages(["wget", "sudo", "gnupg"])

    md = os_release_meta()
    codename = md["VERSION_CODENAME"]

    keyadd = "wget -qO - https://repo.radeon.com/rocm/rocm.gpg.key | sudo apt-key add -"
    subprocess.check_call(keyadd, shell=True)

    with open("/etc/apt/sources.list.d/amdgpu.list", "w") as fd:
        fd.write(
            ("deb [arch=amd64] " "https://repo.radeon.com/amdgpu/%s/ubuntu %s main\n")
            % (rocm_version_str, codename)
        )

    with open("/etc/apt/sources.list.d/rocm.list", "w") as fd:
        fd.write(
            ("deb [arch=amd64] " "https://repo.radeon.com/rocm/apt/%s %s main\n")
            % (rocm_version_str, codename)
        )

    # update indexes
    subprocess.check_call(["apt-get", "update"])


def setup_repos_el8(rocm_version_str):

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


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--rocm-version", help="ROCm version to install", default="latest")
    p.add_argument("--job-name", default=None)
    p.add_argument("--build-num", default=None)
    return p.parse_args()


def main():
    args = parse_args()
    if args.rocm_version == "latest":
        try:
            rocm_version = latest_rocm()
            print("Latest ROCm release: %s" % rocm_version)
        except Exception:
            print(
                "Latest ROCm lookup failed. Please use '--rocm-version' to specify a version instead.",
                file=sys.stderr,
            )
            sys.exit(-1)
    else:
        rocm_version = args.rocm_version

    install_rocm(rocm_version, job_name=args.job_name, build_num=args.build_num)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()

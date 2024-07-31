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
import logging
import os
from pprint import pprint
import subprocess

from auditwheel.lddtree import lddtree
from auditwheel.wheeltools import InWheelCtx
from auditwheel.elfutils import elf_file_filter
from auditwheel.policy import WheelPolicies
from auditwheel.wheel_abi import analyze_wheel_abi


LOG = logging.getLogger(__name__)


def tree(path):

    with InWheelCtx(path) as ctx:
        for sofile, fd in elf_file_filter(ctx.iter_files()):

            LOG.info("found SO file: %s" % sofile)
            elftree = lddtree(sofile)

            print(elftree)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("wheel_path")
    return p.parse_args()


def parse_wheel_name(path):
    wheel_name = os.path.basename(path)
    return wheel_name[:-4].split("-")


def fix_wheel(path):
    tup = parse_wheel_name(path)
    plat_tag = tup[4]
    if "manylinux2014" in plat_tag:
        # strip any manylinux tags from the current wheel first
        from wheel.cli import tags

        plat_mod_str = "linux_x86_64"
        new_wheel = tags.tags(
            path,
            python_tags=None,
            abi_tags=None,
            platform_tags=plat_mod_str,
            build_tag=None,
        )
        new_path = os.path.join(os.path.dirname(path), new_wheel)
        LOG.info("Stripped broken tags and created new wheel at %r" % new_path)
        path = new_path

    # build excludes, using auditwheels lddtree to find them
    wheel_pol = WheelPolicies()
    exclude = frozenset()
    abi = analyze_wheel_abi(wheel_pol, path, exclude)

    plat = "manylinux_2_28_x86_64"
    ext_libs = abi.external_refs.get(plat, {}).get("libs")
    exclude = list(ext_libs.keys())

    # call auditwheel repair with excludes
    cmd = ["auditwheel", "repair", "--plat", plat, "--only-plat"]

    for ex in exclude:
        cmd.append("--exclude")
        cmd.append(ex)

    cmd.append(path)

    LOG.info("running %r" % cmd)

    rc = subprocess.run(cmd, check=True)


def main():
    args = parse_args()
    path = args.wheel_path
    fix_wheel(path)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()

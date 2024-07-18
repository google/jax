#!/usr/bin/env python3


# NOTE(mrodden): This file is part of the ROCm build scripts, and
# needs be compatible with Python 3.6. Please do not include these
# in any "upgrade" scripts


import argparse
from collections import deque
import fcntl
import logging
import os
import re
import select
import subprocess
import shutil
import sys


LOG = logging.getLogger(__name__)


GPU_DEVICE_TARGETS = "gfx900 gfx906 gfx908 gfx90a gfx940 gfx941 gfx942 gfx1030 gfx1100"


def build_rocm_path(rocm_version_str):
    return "/opt/rocm-%s" % rocm_version_str


def update_rocm_targets(rocm_path, targets):
    target_fp = os.path.join(rocm_path, "bin/target.lst")
    version_fp = os.path.join(rocm_path, ".info/version")
    with open(target_fp, "w") as fd:
        fd.write("%s\n" % targets)

    # mimic touch
    open(version_fp, "a").close()


def build_jaxlib_wheel(jax_path, rocm_path, python_version, xla_path=None):
    cmd = [
        "python",
        "build/build.py",
        "--enable_rocm",
        "--build_gpu_plugin",
        "--gpu_plugin_rocm_version=60",
        "--rocm_path=%s" % rocm_path,
    ]

    if xla_path:
        cmd.append("--bazel_options=--override_repository=xla=%s" % xla_path)

    cpy = to_cpy_ver(python_version)
    py_bin = "/opt/python/%s-%s/bin" % (cpy, cpy)

    env = dict(os.environ)
    env["JAX_RELEASE"] = str(1)
    env["JAXLIB_RELEASE"] = str(1)
    env["PATH"] = "%s:%s" % (py_bin, env["PATH"])

    LOG.info("Running %r from cwd=%r" % (cmd, jax_path))
    pattern = re.compile("Output wheel: (.+)\n")

    return _run_scan_for_output(cmd, pattern, env=env, cwd=jax_path, capture="stderr")


def build_jax_wheel(jax_path, python_version):
    cmd = [
        "python",
        "-m",
        "build",
    ]

    cpy = to_cpy_ver(python_version)
    py_bin = "/opt/python/%s-%s/bin" % (cpy, cpy)

    env = dict(os.environ)
    env["JAX_RELEASE"] = str(1)
    env["JAXLIB_RELEASE"] = str(1)
    env["PATH"] = "%s:%s" % (py_bin, env["PATH"])

    LOG.info("Running %r from cwd=%r" % (cmd, jax_path))
    pattern = re.compile("Successfully built jax-.+ and (jax-.+\.whl)\n")

    wheels = _run_scan_for_output(cmd, pattern, env=env, cwd=jax_path, capture="stdout")

    paths = list(map(lambda x: os.path.join(jax_path, "dist", x), wheels))
    return paths


def _run_scan_for_output(cmd, pattern, env=None, cwd=None, capture=None):

    buf = deque(maxlen=20000)

    if capture == "stderr":
        p = subprocess.Popen(cmd, env=env, cwd=cwd, stderr=subprocess.PIPE)
        redir = sys.stderr
        cap_fd = p.stderr
    else:
        p = subprocess.Popen(cmd, env=env, cwd=cwd, stdout=subprocess.PIPE)
        redir = sys.stdout
        cap_fd = p.stdout

    flags = fcntl.fcntl(cap_fd, fcntl.F_GETFL)
    fcntl.fcntl(cap_fd, fcntl.F_SETFL, flags | os.O_NONBLOCK)

    eof = False
    while not eof:
        r, _, _ = select.select([cap_fd], [], [])
        for fd in r:
            dat = fd.read(512)
            if dat is None:
                continue
            elif dat:
                t = dat.decode("utf8")
                redir.write(t)
                buf.extend(t)
            else:
                eof = True

    # wait and drain pipes
    _, _ = p.communicate()

    if p.returncode != 0:
        raise Exception(
            "Child process exited with nonzero result: rc=%d" % p.returncode
        )

    text = "".join(buf)

    matches = pattern.findall(text)

    if not matches:
        LOG.error("No wheel name found in output: %r" % text)
        raise Exception("No wheel name found in output")

    wheels = []
    for match in matches:
        LOG.info("Found built wheel: %r" % match)
        wheels.append(match)

    return wheels


def to_cpy_ver(python_version):
    tup = python_version.split(".")
    return "cp%d%d" % (int(tup[0]), int(tup[1]))


def fix_wheel(path, jax_path):
    # NOTE(mrodden): fixwheel needs auditwheel 6.0.0, which has a min python of 3.8
    # so use one of the CPythons in /opt to run
    env = dict(os.environ)
    py_bin = "/opt/python/cp310-cp310/bin"
    env["PATH"] = "%s:%s" % (py_bin, env["PATH"])

    cmd = ["pip", "install", "auditwheel>=6"]
    subprocess.run(cmd, check=True, env=env)

    fixwheel_path = os.path.join(jax_path, "build/rocm/tools/fixwheel.py")
    cmd = ["python", fixwheel_path, path]
    subprocess.run(cmd, check=True, env=env)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--rocm-version", default="6.1.1", help="ROCM Version to build JAX against"
    )
    p.add_argument(
        "--python-versions",
        default=["3.10.19,3.12"],
        help="Comma separated CPython versions that wheels will be built and output for",
    )
    p.add_argument(
        "--xla-path",
        type=str,
        default=None,
        help="Optional directory where XLA source is located to use instead of JAX builtin XLA",
    )

    p.add_argument("jax_path", help="Directory where JAX source directory is located")

    return p.parse_args()


def main():
    args = parse_args()
    python_versions = args.python_versions.split(",")

    print("ROCM_VERSION=%s" % args.rocm_version)
    print("PYTHON_VERSIONS=%r" % python_versions)
    print("JAX_PATH=%s" % args.jax_path)
    print("XLA_PATH=%s" % args.xla_path)

    rocm_path = build_rocm_path(args.rocm_version)

    update_rocm_targets(rocm_path, GPU_DEVICE_TARGETS)

    for py in python_versions:
        wheel_paths = build_jaxlib_wheel(args.jax_path, rocm_path, py, args.xla_path)
        for wheel_path in wheel_paths:
            fix_wheel(wheel_path, args.jax_path)

    # build JAX wheel for completeness
    jax_wheels = build_jax_wheel(args.jax_path, python_versions[-1])

    # NOTE(mrodden): the jax wheel is a "non-platform wheel", so auditwheel will
    # do nothing, and in fact will throw an Exception. we just need to copy it
    # along with the jaxlib and plugin ones

    # copy jax wheel(s) to wheelhouse
    wheelhouse_dir = "/wheelhouse/"
    for whl in jax_wheels:
        LOG.info("Copying %s into %s" % (whl, wheelhouse_dir))
        shutil.copy(whl, wheelhouse_dir)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()

# Hermetic Python

Hermetic Python allows not to rely on system-installed Python, and
system-installed Python packages. \
Instead, an independent Python toolchain is registered, ensuring the right
dependencies are always used. \
See https://github.com/bazelbuild/rules_python/ for more details.

### Specifying the Python version

Note: Only a number of minor Python versions are supported at any given time.

By default, the lowest supported version is used.

To set a different version, use the `JAX_PYTHON_VERSION` environment variable,
e.g.

```
export JAX_PYTHON_VERSION=3.11
```

To specify the version via a Bazel command argument, use the following:

```
--repo_env=JAX_PYTHON_VERSION=3.11
```

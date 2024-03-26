# Notes on using Python with Bazel

If you are building JAX from source, JAX will use hermetic Python when running any Bazel commands.

Hermetic Python allows not to rely on system-installed Python, and
system-installed Python packages. \
Instead, an independent Python toolchain is registered, ensuring the right
dependencies are always used. \
See https://github.com/bazelbuild/rules_python/ for more details.

## Specifying the Python version

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

## Requirements updater

Requirements updater is a standalone tool, intended to simplify process of
updating requirements for multiple minor versions of Python.
It takes in a file with a set of dependencies(`requirements.in`), and produces
a more detailed requirements file for each version, with hashes specified for
each dependency required, as well as their sub-dependencies.

Runtime, build and test dependencies are specified in the `build/requirements_updater/requirements.in` file.`build/requirements_updater/updater.sh` is a script that runs the requirements_updater tool.

### Updating dependencies

To use a different dependency version, e.g. an older or newer version of `numpy`, one should:
1) Update the value in `build/requirements_updater/requirements.in`
2) Run
`bash updater.sh`
3) Verify that the version was updated in `requirements_lock` files.


### Adding new dependencies

To add a new runtime/build/test dependency, one should follow similar steps:
1) Add a new entry in `build/requirements_updater/requirements.in`
2) Run
`bash updater.sh`
3) Verify that the version was updated in `requirements_lock` files.
4) Add the entry to [py_deps](https://github.com/google/jax/blob/main/jaxlib/jax.bzl#L49)


## Using Prebuilt jaxlib with Bazel

To use prebuilt jaxlib from Pypi with bazel, pass the jaxlib version to
`updater.sh` script, e.g. to use jaxlib `0.4.25`:

`$ build/requirements_updater/updater.sh 0.4.25`

`$ bazel test --//jax:build_jaxlib=false //tests:cpu_tests`

`$ bazel test --//jax:build_jaxlib=false//tests:backend_independent_tests`

A local path to a `jaxlib` wheel can also be provided.


## How to add a new Python version

Note: Updating the
[rules-python](https://github.com/bazelbuild/rules_python/releases) version may
be required before going through the steps below. This is due to the new Python
versions becoming available through `rules-python`. \

All the files referenced below are located in the same directory as this README,
unless indicated otherwise.

1) Add the new version to the `VERSIONS` variable inside
   `jax/tools/py_version.bzl`. \
   While this isn't necessary for running the updater, it is required for
   actually using the new version with Jax.

2) In the `WORKSPACE` file, add the new version to the `python_versions`
   parameter of the `python_register_multi_toolchains` function.

3) In the `BUILD.bazel` file, add a load statement for the new version, e.g.

   ```
      load("@python//3.11:defs.bzl",
           compile_pip_requirements_3_11 = "compile_pip_requirements")
   ```

   Add a new entry for the loaded `compile_pip_requirements`, e.g.

   ```
      compile_pip_requirements_3_11(
          name = "requirements_3_11",
          extra_args = ["--allow-unsafe"],
          requirements_in = "requirements.in",
          requirements_txt = "requirements_lock_3_11.txt",
          data = ["test-requirements.txt"],
      )
   ```

4) Add the version to `SUPPORTED_VERSIONS` in `updater.sh`

5) Run the `updater.sh` shell script. \
   If the base requirements file hasn't yet been updated to account for the new
   Python version, which will require different versions for at least some
   dependencies, it will need to be updated now, for the script to run
   successfully.

6) A new `requirements_lock_3_11.txt` file should appear under the `build` of
   the `JAX` directory.

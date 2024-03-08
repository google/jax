## Requirements updater

Requirements updater is a standalone tool, intended to simplify process of
updating requirements for multiple minor versions of Python.

It takes in a file with a set of dependencies(`requirements.in`), and produces
a more detailed requirements file for each version, with hashes specified for 
each dependency required, as well as their sub-dependencies.


### How to run the updater
Update `requirements.in` accordingly, then run

```
bash updater.sh
```

To update a single dependency, e.g. `numpy`, add `"-P numpy"` to 
`extra_args` in `BUILD.bazel` file.

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
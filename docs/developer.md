(building-from-source)=
# Building from source

<!--* freshness: { reviewed: '2024-05-15' } *-->

First, obtain the JAX source code:

```
git clone https://github.com/google/jax
cd jax
```

Building JAX involves two steps:

1. Building or installing `jaxlib`, the C++ support library for `jax`.
2. Installing the `jax` Python package.

## Building or installing `jaxlib`

### Installing `jaxlib` with pip

If you're only modifying Python portions of JAX, we recommend installing
`jaxlib` from a prebuilt wheel using pip:

```
pip install jaxlib
```

See the [JAX readme](https://github.com/google/jax#installation) for full
guidance on pip installation (e.g., for GPU and TPU support).

### Building `jaxlib` from source

To build `jaxlib` from source, you must also install some prerequisites:

- a C++ compiler (g++, clang, or MSVC)

  On Ubuntu or Debian you can install the necessary prerequisites with:

  ```
  sudo apt install g++ python python3-dev
  ```

  If you are building on a Mac, make sure XCode and the XCode command line tools
  are installed.

  See below for Windows build instructions.

- there is no need to install Python dependencies locally, as your system
  Python will be ignored during the build; please check
  [Managing hermetic Python](#managing-hermetic-python) for details.

To build `jaxlib` for CPU or TPU, you can run:

```
python build/build.py
pip install dist/*.whl  # installs jaxlib (includes XLA)
```

To build a wheel for a version of Python different from your current system
installation pass `--python_version` flag to the build command:

```
python build/build.py --python_version=3.12
```

The rest of this document assumes that you are building for Python version
matching your current system installation. If you need to build for a different
version, simply append `--python_version=<py version>` flag every time you call
`python build/build.py`. Note, the Bazel build will always use a hermetic Python
installation regardless of whether the `--python_version` parameter is passed or
not.

There are two ways to build `jaxlib` with CUDA support: (1) use
`python build/build.py --enable_cuda` to generate a jaxlib wheel with cuda
support, or (2) use
`python build/build.py --enable_cuda --build_gpu_plugin --gpu_plugin_cuda_version=12`
to generate three wheels (jaxlib without cuda, jax-cuda-plugin,
and jax-cuda-pjrt). You can set `gpu_plugin_cuda_version` to 11 or 12.

See `python build/build.py --help` for configuration options, including ways to
specify the paths to CUDA and CUDNN, which you must have installed. Here
`python` should be the name of your Python 3 interpreter; on some systems, you
may need to use `python3` instead. Despite calling the script with `python`,
Bazel will always use its own hermetic Python interpreter and dependencies, only
the `build/build.py` script itself will be processed by your system Python
interpreter. By default, the wheel is written to the `dist/` subdirectory of the
current directory.

### Building jaxlib from source with a modified XLA repository.

JAX depends on XLA, whose source code is in the
[XLA GitHub repository](https://github.com/openxla/xla).
By default JAX uses a pinned copy of the XLA repository, but we often
want to use a locally-modified copy of XLA when working on JAX. There are two
ways to do this:

- use Bazel's `override_repository` feature, which you can pass as a command
  line flag to `build.py` as follows:

  ```
  python build/build.py --bazel_options=--override_repository=xla=/path/to/xla
  ```

- modify the `WORKSPACE` file in the root of the JAX source tree to point to
  a different XLA tree.

To contribute changes back to XLA, send PRs to the XLA repository.

The version of XLA pinned by JAX is regularly updated, but is updated in
particular before each `jaxlib` release.

### Additional Notes for Building `jaxlib` from source on Windows

On Windows, follow [Install Visual Studio](https://docs.microsoft.com/en-us/visualstudio/install/install-visual-studio?view=vs-2019)
to set up a C++ toolchain. Visual Studio 2019 version 16.5 or newer is required.
If you need to build with CUDA enabled, follow the
[CUDA Installation Guide](https://docs.nvidia.com/cuda/cuda-installation-guide-microsoft-windows/index.html)
to set up a CUDA environment.

JAX builds use symbolic links, which require that you activate
[Developer Mode](https://docs.microsoft.com/en-us/windows/apps/get-started/enable-your-device-for-development).

You can either install Python using its
[Windows installer](https://www.python.org/downloads/), or if you prefer, you
can use [Anaconda](https://docs.anaconda.com/anaconda/install/windows/)
or [Miniconda](https://docs.conda.io/en/latest/miniconda.html#windows-installers)
to set up a Python environment.

Some targets of Bazel use bash utilities to do scripting, so [MSYS2](https://www.msys2.org)
is needed. See [Installing Bazel on Windows](https://bazel.build/install/windows#install-compilers)
for more details. Install the following packages:

```
pacman -S patch coreutils
```

Once coreutils is installed, the realpath command should be present in your shell's path.

Once everything is installed. Open PowerShell, and make sure MSYS2 is in the
path of the current session. Ensure `bazel`, `patch` and `realpath` are
accessible. Activate the conda environment. The following command builds with
CUDA enabled, adjust it to whatever suitable for you:

```
python .\build\build.py `
  --enable_cuda `
  --cuda_path='C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v10.1' `
  --cudnn_path='C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v10.1' `
  --cuda_version='10.1' `
  --cudnn_version='7.6.5'
```

To build with debug information, add the flag `--bazel_options='--copt=/Z7'`.

### Additional notes for building a ROCM `jaxlib` for AMD GPUs

You need several ROCM/HIP libraries installed to build for ROCM. For
example, on a Ubuntu machine with
[AMD's `apt` repositories available](https://rocm.docs.amd.com/en/latest/deploy/linux/quick_start.html),
you need a number of packages installed:

```
sudo apt install miopen-hip hipfft-dev rocrand-dev hipsparse-dev hipsolver-dev \
    rccl-dev rccl hip-dev rocfft-dev roctracer-dev hipblas-dev rocm-device-libs
```

To build jaxlib with ROCM support, you can run the following build command,
suitably adjusted for your paths and ROCM version.

```
python build/build.py --enable_rocm --rocm_path=/opt/rocm-5.7.0
```

AMD's fork of the XLA repository may include fixes not present in the upstream
XLA repository. If you experience problems with the upstream repository, you can
try AMD's fork, by cloning their repository:

```
git clone https://github.com/ROCmSoftwarePlatform/xla.git
```

and override the XLA repository with which JAX is built:

```
python build/build.py --enable_rocm --rocm_path=/opt/rocm-5.7.0 \
  --bazel_options=--override_repository=xla=/path/to/xla-rocm
```

## Managing hermetic Python

To make sure that JAX's build is reproducible, behaves uniformly across
supported platforms (Linux, Windows, MacOS) and is properly isolated from
specifics of a local system, we rely on hermetic Python (see
[rules_python](https://github.com/bazelbuild/rules_python)) for all build
and test commands executed via Bazel. This means that your system Python
installation will be ignored during the build and Python interpreter itself
as well as all the Python dependencies will be managed by bazel directly.

### Specifying Python version

When you run `build/build.py` tool, the version of hermetic Python is set
automatically to match the version of the Python you used to
run `build/build.py` script. To choose a specific version explicitly you may
pass `--python_version` argument to the tool:

```
python build/build.py --python_version=3.12
```

Under the hood, the hermetic Python version is controlled
by `HERMETIC_PYTHON_VERSION` environment variable, which is set automatically
when you run `build/build.py`. In case you run bazel directly you may need to
set the variable explicitly in one of the following ways:

```
# Either add an entry to your `.bazelrc` file
build --repo_env=HERMETIC_PYTHON_VERSION=3.12

# OR pass it directly to your specific build command
bazel build <target> --repo_env=HERMETIC_PYTHON_VERSION=3.12

# OR set the environment variable globally in your shell:
export HERMETIC_PYTHON_VERSION=3.12
```

You may run builds and tests against different versions of Python sequentially
on the same machine by simply switching the value of `--python_version` between
the runs. All the python-agnostic parts of the build cache from the previous
build will be preserved and reused for the subsequent builds.

### Specifying Python dependencies

During bazel build all JAX's Python dependencies are pinned to their specific
versions. This is necessary to ensure reproducibility of the build.
The pinned versions of the full transitive closure of JAX's dependencies
together with their corresponding hashes are specified in
`build/requirements_lock_<python version>.txt` files (
e.g. `build/requirements_lock_3_12.txt` for `Python 3.12`).

To update the lock files, make sure `build/requirements.in` contains the desired 
direct dependencies list and then execute the following command (which will call 
[pip-compile](https://pypi.org/project/pip-tools/) under the hood):

```
python build/build.py --requirements_update --python_version=3.12
```

Alternatively, if you need more control, you may run the bazel command
directly (the two commands are equivalent):

```
bazel run //build:requirements.update --repo_env=HERMETIC_PYTHON_VERSION=3.12
```

where `3.12` is the `Python` version you wish to update.

Note, since it is still `pip` and `pip-compile` tools used under the hood, so
most of the command line arguments and features supported by those tools will be
acknowledged by the Bazel requirements updater command as well. For example, if
you wish the updater to consider pre-release versions simply pass `--pre`
argument to the bazel command:

```
bazel run //build:requirements.update --repo_env=HERMETIC_PYTHON_VERSION=3.12 -- --pre
```

### Specifying dependencies on local wheels

If you need to depend on a local .whl file, for example on your newly built
jaxlib wheel, you may add a path to the wheel in `build/requirements.in` and
re-run the requirements updater command for a selected version of Python. For
example:

```
echo -e "\n$(realpath jaxlib-0.4.27.dev20240416-cp312-cp312-manylinux2014_x86_64.whl)" >> build/requirements.in
python build/build.py --requirements_update --python_version=3.12
```

### Specifying dependencies on nightly wheels

To build and test against the very latest, potentially unstable, set of Python
dependencies we provide a special version of the dependency updater command as
follows:

```
python build/build.py --requirements_nightly_update --python_version=3.12
```

Or, if you run `bazel` directly (the two commands are equivalent):

```
bazel run //build:requirements_nightly.update --repo_env=HERMETIC_PYTHON_VERSION=3.12
```

The difference between this and the regular updater is that by default it would
accept pre-release, dev and nightly packages, it will also
search https://pypi.anaconda.org/scientific-python-nightly-wheels/simple as an
extra index url and will not put hashes in the resultant requirements lock file.

### Building with pre-release Python version

We support all of the current versions of Python out of the box, but if you need
to build and test against a different version (for example the latest unstable
version which hasn't been released officially yet) please follow the
instructions below.

1) Make sure you have installed necessary linux packages needed to build Python
   interpreter itself and key packages (like `numpy` or `scipy`) from source. On
   a typical Debian system you may need to install the following packages:

```
sudo apt-get update
sudo apt-get build-dep python3 -y
sudo apt-get install pkg-config zlib1g-dev libssl-dev -y
# to  build scipy
sudo apt-get install libopenblas-dev -y
```

2) Check your `WORKSPACE` file and make sure it
   has `custom_python_interpreter()` entry there, pointing to the version of
   Python you want to build.

3) Run `bazel build @python_dev//:python_dev` to build Python interpreter. By default it will
   be built with GCC compiler. If you wish to build with clang, you need to set
   corresponding env variables to do so (
   e.g. `--repo_env=CC=/usr/lib/llvm-17/bin/clang --repo_env=CXX=/usr/lib/llvm-17/bin/clang++`).

4) Check the output of the previous command. At the very end of it you will find
   a code snippet for `python_register_toolchains()` entry with your newly built
   Python in it. Copy that code snippet in your `WORKSPACE` file either right
   after  `python_init_toolchains()` entry (to add the new version of Python) or
   instead of it (to replace an existing version, like replacing 3.12 with
   custom built variant of 3.12). The code snippet is generated to match your
   actual setup, so it should work as is, but you can customize it if you choose
   so (for example to change location of Python's `.tgz` file so it could be
   downloaded remotely instead of being on local machine).

5) Make sure there is an entry for your Python's version in `requirements`
   parameter for `python_init_repositories()` in your WORKSPACE file. For
   example for `Python 3.13` it should have something
   like `"3.13": "//build:requirements_lock_3_13.txt"`.

6) For unstable versions of Python, optionally (but highly recommended)
   run `bazel build //build:all_py_deps --repo_env=HERMETIC_PYTHON_VERSION="3.13"`,
   where `3.13` is the version of Python interpreter you built on step 3.
   This will make `pip` pull and build from sources (for packages which don't
   have binaries published yet, for
   example `numpy`, `scipy`, `matplotlib`, `zstandard`) all of the JAX's python
   dependencies. It is recommended to do this step first (i.e. independently of
   actual JAX build) for all unstable versions of Python to avoid conflict
   between building JAX itself and building of its Python dependencies. For
   example, we normally build JAX with clang but building `matplotlib` from
   sources with clang fails out of the box due to differences in LTO behavior (
   Link Time Optimization, triggered by `-flto` flag) between GCC and clang, and
   matplotlib assumes GCC by default.
   If you build against a stable version of Python, or in general you do not
   expect any of your Python dependencies to be built from sources (i.e. binary
   distributions for the corresponding Python version already exist in the
   repository) this step is not needed.

7) Congrats, you've built and configured your custom Python for JAX project! You
   may now execute your built/test commands as usual, just make
   sure `HERMETIC_PYTHON_VERSION` environment variable is set and points to your
   new version.

8) Note, if you were building a pre-release version of Python, updating of
   `requirements_lock_<python_version>.txt` files with your newly built Python
   is likely to fail, because package repositories will not have matching
   binary packages. When there are no binary packages available `pip-compile`
   proceeds with building them from sources, which is likely to fail because it
   is more restrictive than doing the same thing during `pip` installation.
   The recommended way to update requirements lock file for unstable versions of
   Python is to update requirements for the latest stable version (e.g. `3.12`)
   without hashes (therefore special `//build:requirements_dev.update` target)
   and then copy the results to the unstable Python's lock file (e.g. `3.13`):
```
bazel run //build:requirements_dev.update --repo_env=HERMETIC_PYTHON_VERSION="3.12"
cp build/requirements_lock_3_12.txt build/requirements_lock_3_13.txt
bazel build //build:all_py_deps --repo_env=HERMETIC_PYTHON_VERSION="3.13"
# You may need to edit manually the resultant lock file, depending on how ready
# your dependencies are for the new version of Python.
```

## Installing `jax`

Once `jaxlib` has been installed, you can install `jax` by running:

```
pip install -e .  # installs jax
```

To upgrade to the latest version from GitHub, just run `git pull` from the JAX
repository root, and rebuild by running `build.py` or upgrading `jaxlib` if
necessary. You shouldn't have to reinstall `jax` because `pip install -e`
sets up symbolic links from site-packages into the repository.

(running-tests)=

## Running the tests

There are two supported mechanisms for running the JAX tests, either using Bazel
or using pytest.

### Using Bazel

First, configure the JAX build by running:

```
python build/build.py --configure_only
```

You may pass additional options to `build.py` to configure the build; see the
`jaxlib` build documentation for details.

By default the Bazel build runs the JAX tests using `jaxlib` built from source.
To run JAX tests, run:

```
bazel test //tests:cpu_tests //tests:backend_independent_tests
```

`//tests:gpu_tests` and `//tests:tpu_tests` are also available, if you have the
necessary hardware.

To use a preinstalled `jaxlib` instead of building it you first need to
make it available in the hermetic Python. To install a specific version of
`jaxlib` within hermetic Python run (using `jaxlib >= 0.4.26` as an example):

```
echo -e "\njaxlib >= 0.4.26" >> build/requirements.in
python build/build.py --requirements_update
```

Alternatively, to install `jaxlib` from a local wheel (assuming Python 3.12):

```
echo -e "\n$(realpath jaxlib-0.4.26-cp312-cp312-manylinux2014_x86_64.whl)" >> build/requirements.in
python build/build.py --requirements_update --python_version=3.12
```

Once you have `jaxlib` installed hermetically, run:

```
bazel test --//jax:build_jaxlib=false //tests:cpu_tests //tests:backend_independent_tests
```

A number of test behaviors can be controlled using environment variables (see
below). Environment variables may be passed to JAX tests using the
`--test_env=FLAG=value` flag to Bazel.

Some of JAX tests are for multiple accelerators (i.e. GPUs, TPUs). When JAX is
already installed, you can run GPUs tests like this:

```
bazel test //tests:gpu_tests --local_test_jobs=4 --test_tag_filters=multiaccelerator --//jax:build_jaxlib=false --test_env=XLA_PYTHON_CLIENT_ALLOCATOR=platform
```

You can speed up single accelerator tests by running them in parallel on
multiple accelerators. This also triggers multiple concurrent tests per
accelerator. For GPUs, you can do it like this:

```
NB_GPUS=2
JOBS_PER_ACC=4
J=$((NB_GPUS * JOBS_PER_ACC))
MULTI_GPU="--run_under $PWD/build/parallel_accelerator_execute.sh --test_env=JAX_ACCELERATOR_COUNT=${NB_GPUS} --test_env=JAX_TESTS_PER_ACCELERATOR=${JOBS_PER_ACC} --local_test_jobs=$J"
bazel test //tests:gpu_tests //tests:backend_independent_tests --test_env=XLA_PYTHON_CLIENT_PREALLOCATE=false --test_tag_filters=-multiaccelerator $MULTI_GPU
```

### Using `pytest`
First, install the dependencies by
running `pip install -r build/test-requirements.txt`.

To run all the JAX tests using `pytest`, we recommend using `pytest-xdist`,
which can run tests in parallel. It is installed as a part of
`pip install -r build/test-requirements.txt` command.

From the repository root directory run:

```
pytest -n auto tests
```

### Controlling test behavior

JAX generates test cases combinatorially, and you can control the number of
cases that are generated and checked for each test (default is 10) using the
`JAX_NUM_GENERATED_CASES` environment variable. The automated tests
currently use 25 by default.

For example, one might write

```
# Bazel
bazel test //tests/... --test_env=JAX_NUM_GENERATED_CASES=25`
```

or

```
# pytest
JAX_NUM_GENERATED_CASES=25 pytest -n auto tests
```

The automated tests also run the tests with default 64-bit floats and ints
(`JAX_ENABLE_X64`):

```
JAX_ENABLE_X64=1 JAX_NUM_GENERATED_CASES=25 pytest -n auto tests
```

You can run a more specific set of tests using
[pytest](https://docs.pytest.org/en/latest/usage.html#specifying-tests-selecting-tests)'s
built-in selection mechanisms, or alternatively you can run a specific test
file directly to see more detailed information about the cases being run:

```
JAX_NUM_GENERATED_CASES=5 python tests/lax_numpy_test.py
```

You can skip a few tests known to be slow, by passing environment variable
JAX_SKIP_SLOW_TESTS=1.

To specify a particular set of tests to run from a test file, you can pass a string
or regular expression via the `--test_targets` flag. For example, you can run all
the tests of `jax.numpy.pad` using:

```
python tests/lax_numpy_test.py --test_targets="testPad"
```

The Colab notebooks are tested for errors as part of the documentation build.

### Doctests

JAX uses pytest in doctest mode to test the code examples within the documentation.
You can run this using

```
pytest docs
```

Additionally, JAX runs pytest in `doctest-modules` mode to ensure code examples in
function docstrings will run correctly. You can run this locally using, for example:

```
pytest --doctest-modules jax/_src/numpy/lax_numpy.py
```

Keep in mind that there are several files that are marked to be skipped when the
doctest command is run on the full package; you can see the details in
[`ci-build.yaml`](https://github.com/google/jax/blob/main/.github/workflows/ci-build.yaml)

## Type checking

We use `mypy` to check the type hints. To check types locally the same way
as the CI checks them:

```
pip install mypy
mypy --config=pyproject.toml --show-error-codes jax
```

Alternatively, you can use the [pre-commit](https://pre-commit.com/) framework to run this
on all staged files in your git repository, automatically using the same mypy version as
in the GitHub CI:

```
pre-commit run mypy
```

## Linting

JAX uses the [ruff](https://docs.astral.sh/ruff/) linter to ensure code
quality. You can check your local changes by running:

```
pip install ruff
ruff jax
```

Alternatively, you can use the [pre-commit](https://pre-commit.com/) framework to run this
on all staged files in your git repository, automatically using the same ruff version as
the GitHub tests:

```
pre-commit run ruff
```

## Update documentation

To rebuild the documentation, install several packages:

```
pip install -r docs/requirements.txt
```

And then run:

```
sphinx-build -b html docs docs/build/html -j auto
```

This can take a long time because it executes many of the notebooks in the documentation source;
if you'd prefer to build the docs without executing the notebooks, you can run:

```
sphinx-build -b html -D nb_execution_mode=off docs docs/build/html -j auto
```

You can then see the generated documentation in `docs/build/html/index.html`.

The `-j auto` option controls the parallelism of the build. You can use a number
in place of `auto` to control how many CPU cores to use.

(update-notebooks)=

### Update notebooks

We use [jupytext](https://jupytext.readthedocs.io/) to maintain two synced copies of the notebooks
in `docs/notebooks`: one in `ipynb` format, and one in `md` format. The advantage of the former
is that it can be opened and executed directly in Colab; the advantage of the latter is that
it makes it much easier to track diffs within version control.

#### Editing `ipynb`

For making large changes that substantially modify code and outputs, it is easiest to
edit the notebooks in Jupyter or in Colab. To edit notebooks in the Colab interface,
open <http://colab.research.google.com> and `Upload` from your local repo.
Update it as needed, `Run all cells` then `Download ipynb`.
You may want to test that it executes properly, using `sphinx-build` as explained above.

#### Editing `md`

For making smaller changes to the text content of the notebooks, it is easiest to edit the
`.md` versions using a text editor.

#### Syncing notebooks

After editing either the ipynb or md versions of the notebooks, you can sync the two versions
using [jupytext](https://jupytext.readthedocs.io/) by running `jupytext --sync` on the updated
notebooks; for example:

```
pip install jupytext==1.16.0
jupytext --sync docs/notebooks/thinking_in_jax.ipynb
```

The jupytext version should match that specified in
[.pre-commit-config.yaml](https://github.com/google/jax/blob/main/.pre-commit-config.yaml).

To check that the markdown and ipynb files are properly synced, you may use the
[pre-commit](https://pre-commit.com/) framework to perform the same check used
by the github CI:

```
git add docs -u  # pre-commit runs on files in git staging.
pre-commit run jupytext
```

#### Creating new notebooks

If you are adding a new notebook to the documentation and would like to use the `jupytext --sync`
command discussed here, you can set up your notebook for jupytext by using the following command:

```
jupytext --set-formats ipynb,md:myst path/to/the/notebook.ipynb
```

This works by adding a `"jupytext"` metadata field to the notebook file which specifies the
desired formats, and which the `jupytext --sync` command recognizes when invoked.

#### Notebooks within the Sphinx build

Some of the notebooks are built automatically as part of the pre-submit checks and
as part of the [Read the docs](https://jax.readthedocs.io/en/latest) build.
The build will fail if cells raise errors. If the errors are intentional, you can either catch them,
or tag the cell with `raises-exceptions` metadata ([example PR](https://github.com/google/jax/pull/2402/files)).
You have to add this metadata by hand in the `.ipynb` file. It will be preserved when somebody else
re-saves the notebook.

We exclude some notebooks from the build, e.g., because they contain long computations.
See `exclude_patterns` in [conf.py](https://github.com/google/jax/blob/main/docs/conf.py).

### Documentation building on `readthedocs.io`

JAX's auto-generated documentation is at <https://jax.readthedocs.io/>.

The documentation building is controlled for the entire project by the
[readthedocs JAX settings](https://readthedocs.org/dashboard/jax). The current settings
trigger a documentation build as soon as code is pushed to the GitHub `main` branch.
For each code version, the building process is driven by the
`.readthedocs.yml` and the `docs/conf.py` configuration files.

For each automated documentation build you can see the
[documentation build logs](https://readthedocs.org/projects/jax/builds/).

If you want to test the documentation generation on Readthedocs, you can push code to the `test-docs`
branch. That branch is also built automatically, and you can
see the generated documentation [here](https://jax.readthedocs.io/en/test-docs/). If the documentation build
fails you may want to [wipe the build environment for test-docs](https://docs.readthedocs.io/en/stable/guides/wipe-environment.html).

For a local test, I was able to do it in a fresh directory by replaying the commands
I saw in the Readthedocs logs:

```
mkvirtualenv jax-docs  # A new virtualenv
mkdir jax-docs  # A new directory
cd jax-docs
git clone --no-single-branch --depth 50 https://github.com/google/jax
cd jax
git checkout --force origin/test-docs
git clean -d -f -f
workon jax-docs

python -m pip install --upgrade --no-cache-dir pip
python -m pip install --upgrade --no-cache-dir -I Pygments==2.3.1 setuptools==41.0.1 docutils==0.14 mock==1.0.1 pillow==5.4.1 alabaster>=0.7,<0.8,!=0.7.5 commonmark==0.8.1 recommonmark==0.5.0 'sphinx<2' 'sphinx-rtd-theme<0.5' 'readthedocs-sphinx-ext<1.1'
python -m pip install --exists-action=w --no-cache-dir -r docs/requirements.txt
cd docs
python `which sphinx-build` -T -E -b html -d _build/doctrees-readthedocs -D language=en . _build/html
```

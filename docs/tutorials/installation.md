(installation)=
# Installing JAX

This guide provides instructions for:

- Installing JAX binary packages for supported platforms using `pip` or `conda`
- Using Docker containers (for example {ref}`docker-containers-nvidia-gpu`)
- {ref}`building-jax-from-source`

**TL;DR** For most users, a typical JAX installation may look something like this:

* **CPU-only (Linux/macOS/Windows)**
  ```
  pip install -U "jax[cpu]"
  ```
* **GPU (NVIDIA, CUDA 12, x86_64)**
  ```
  pip install -U "jax[cuda12_pip]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
  ```

(install-supported-platforms)=
## Supported platforms

The table below shows all supported platforms and installation options. Check if your setup is supported; and if it says _"yes"_ or _"experimental"_, then click on the corresponding link to learn how to install JAX in greater detail.

|                  | Linux, x86_64                        | Linux, aarch64      | macOS, Intel x86_64, AMD GPU   | macOS, Apple Silicon, ARM-based       | Windows, x86_64         | Windows WSL2, x86_64           |
|------------------|---------------------------------------|--------------------------------|----------------------------------------|----------------------------------------|-------------------------|-----------------------------------------|
| CPU              | {ref}`yes <install-cpu>`              | {ref}`yes <install-cpu>`        | {ref}`yes <install-cpu>`| {ref}`yes <install-cpu>`| {ref}`yes <install-cpu>` | {ref}`yes <install-cpu>`|
| NVIDIA GPU       | {ref}`yes <install-nvidia-gpu>`       | {ref}`yes <install-nvidia-gpu>` | no | n/a | no | {ref}`experimental <install-nvidia-gpu>` |
| Google Cloud TPU | {ref}`yes <install-google-tpu>`       | n/a | n/a | n/a | n/a | n/a |
| AMD GPU          | {ref}`experimental <install-amd-gpu>` | no | no | n/a | no | no |
| Apple GPU    | n/a                                   | no | {ref}`experimental <install-apple-gpu>` | {ref}`experimental <install-apple-gpu>` | n/a |  n/a |


(install-cpu)=
## CPU

### pip installation: CPU

Currently, the JAX team releases `jaxlib` wheels for the following
operating systems and architectures:

- Linux, x86_64
- macOS, Intel
- macOS, Apple ARM-based
- Windows, x86_64 (*experimental*)

To install a CPU-only version of JAX, which might be useful for doing local
development on a laptop, you can run:

```bash
pip install --upgrade pip
pip install --upgrade "jax[cpu]"
```

On Windows, you may also need to install the
[Microsoft Visual Studio 2019 Redistributable](https://learn.microsoft.com/en-US/cpp/windows/latest-supported-vc-redist?view=msvc-170#visual-studio-2015-2017-2019-and-2022)
if it is not already installed on your machine.

Other operating systems and architectures require building from source. Trying
to pip install on other operating systems and architectures may lead to `jaxlib`
not being installed alongside `jax`, although `jax` may successfully install
(but fail at runtime).


(install-nvidia-gpu)=
## NVIDIA GPU

JAX supports NVIDIA GPUs that have SM version 5.2 (Maxwell) or newer.
Note that Kepler-series GPUs are no longer supported by JAX since
NVIDIA has dropped support for Kepler GPUs in its software.

You must first install the NVIDIA driver. You're
recommended to install the newest driver available from NVIDIA, but the driver
version must be >= 525.60.13 for CUDA 12 and >= 450.80.02 for CUDA 11 on Linux.

If you need to use a newer CUDA toolkit with an older driver, for example
on a cluster where you cannot update the NVIDIA driver easily, you may be
able to use the
[CUDA forward compatibility packages](https://docs.nvidia.com/deploy/cuda-compatibility/)
that NVIDIA provides for this purpose.

### pip installation: NVIDIA GPU (CUDA, installed via pip, easier)

There are two ways to install JAX with NVIDIA GPU support:

- Using NVIDIA CUDA and cuDNN installed from pip wheels
- Using a self-installed CUDA/cuDNN

The JAX team strongly recommends installing CUDA and cuDNN using the pip wheels,
since it is much easier!

This method is only supported on x86_64, because NVIDIA has not released aarch64
CUDA pip packages.

```bash
pip install --upgrade pip

# NVIDIA CUDA 12 installation
# Note: wheels only available on linux.
pip install --upgrade "jax[cuda12_pip]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html

# NVIDIA CUDA 11 installation
# Note: wheels only available on linux.
pip install --upgrade "jax[cuda11_pip]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
```

If JAX detects the wrong version of the NVIDIA CUDA libraries, there are several things
you need to check:

* Make sure that `LD_LIBRARY_PATH` is not set, since `LD_LIBRARY_PATH` can
  override the NVIDIA CUDA libraries.
* Make sure that the NVIDIA CUDA libraries installed are those requested by JAX.
  Rerunning the installation command above should work.

### pip installation: NVIDIA GPU (CUDA, installed locally, harder)

If you prefer to use a preinstalled copy of NVIDIA CUDA, you must first
install NVIDIA [CUDA](https://developer.nvidia.com/cuda-downloads) and
[cuDNN](https://developer.nvidia.com/CUDNN).

JAX provides pre-built CUDA-compatible wheels for **Linux x86_64 only**. Other
combinations of operating system and architecture are possible, but require
building from source (refer to {ref}`building-from-source` to learn more}.

You should use an NVIDIA driver version that is at least as new as your
[NVIDIA CUDA toolkit's corresponding driver version](https://docs.nvidia.com/cuda/cuda-toolkit-release-notes/index.html#cuda-major-component-versions__table-cuda-toolkit-driver-versions).
If you need to use a newer CUDA toolkit with an older driver, for example
on a cluster where you cannot update the NVIDIA driver easily, you may be
able to use the
[CUDA forward compatibility packages](https://docs.nvidia.com/deploy/cuda-compatibility/)
that NVIDIA provides for this purpose.

JAX currently ships two NVIDIA CUDA wheel variants:

- CUDA 12.2, cuDNN 8.9, NCCL 2.16
- CUDA 11.8, cuDNN 8.6, NCCL 2.16

You may use a JAX wheel provided the major version of your CUDA, cuDNN, and NCCL
installations match, and the minor versions are the same or newer.
JAX checks the versions of your libraries, and will report an error if they are
not sufficiently new.

NCCL is an optional dependency, required only if you are performing multi-GPU
computations.

To install, run:

```bash
pip install --upgrade pip

# Installs the wheel compatible with NVIDIA CUDA 12 and cuDNN 8.9 or newer.
# Note: wheels only available on linux.
pip install --upgrade "jax[cuda12_local]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html

# Installs the wheel compatible with NVIDIA CUDA 11 and cuDNN 8.6 or newer.
# Note: wheels only available on linux.
pip install --upgrade "jax[cuda11_local]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
```

**These `pip` installations do not work with Windows, and may fail silently; refer to the table
[above](#supported-platforms).**

You can find your CUDA version with the command:

```bash
nvcc --version
```

JAX uses `LD_LIBRARY_PATH` to find CUDA libraries and `PATH` to find binaries
(`ptxas`, `nvlink`). Please make sure that these paths point to the correct CUDA
installation.

Please let the JAX team know on [the GitHub issue tracker](https://github.com/google/jax/issues)
if you run into any errors or problems with the pre-built wheels.

(docker-containers-nvidia-gpu)=
### NVIDIA GPU Docker containers

NVIDIA provides the [JAX
Toolbox](https://github.com/NVIDIA/JAX-Toolbox) containers, which are
bleeding edge containers containing nightly releases of jax and some
models/frameworks.

## JAX nightly installation

Nightly releases reflect the state of the main JAX repository at the time they are
built, and may not pass the full test suite.

- `jax`:

```bash
pip install -U --pre jax -f https://storage.googleapis.com/jax-releases/jax_nightly_releases.html
```

- `jaxlib` CPU:

```bash
pip install -U --pre jaxlib -f https://storage.googleapis.com/jax-releases/jaxlib_nightly_releases.html
```

- `jaxlib` Google Cloud TPU:

```bash
pip install -U --pre jaxlib -f https://storage.googleapis.com/jax-releases/jaxlib_nightly_releases.html
pip install -U libtpu-nightly -f https://storage.googleapis.com/jax-releases/libtpu_releases.html
```

- `jaxlib` NVIDIA GPU (CUDA 12):

```bash
pip install -U --pre jaxlib -f https://storage.googleapis.com/jax-releases/jaxlib_nightly_cuda12_releases.html
```

- `jaxlib` NVIDIA GPU (CUDA 11):

```bash
pip install -U --pre jaxlib -f https://storage.googleapis.com/jax-releases/jaxlib_nightly_cuda_releases.html
```

(install-google-tpu)=
## Google Cloud TPU

### pip installation: Google Cloud TPU

JAX provides pre-built wheels for
[Google Cloud TPU](https://cloud.google.com/tpu/docs/users-guide-tpu-vm).
To install JAX along with appropriate versions of `jaxlib` and `libtpu`, you can run
the following in your cloud TPU VM:

```bash
pip install jax[tpu] -f https://storage.googleapis.com/jax-releases/libtpu_releases.html
```

For interactive notebook users: Colab TPUs no longer support JAX as of
JAX version 0.4. However, for an interactive TPU notebook in the cloud, you can
use [Kaggle TPU notebooks](https://www.kaggle.com/docs/tpu), which fully
support JAX.

(install-apple-gpu)=
## Apple Silicon GPU (ARM-based)

### pip installation: Apple ARM-based Silicon GPUs

Apple provides an experimental Metal plugin for Apple ARM-based GPU hardware. For details,
refer to
[Apple's JAX on Metal documentation](https://developer.apple.com/metal/jax/).

**Note:** There are several caveats with the Metal plugin:

* The Metal plugin is new and experimental and has a number of
  [known issues](https://github.com/google/jax/issues?q=is%3Aissue+is%3Aopen+label%3A%22Apple+GPU+%28Metal%29+plugin%22).
  Please report any issues on the JAX issue tracker.
* The Metal plugin currently requires very specific versions of `jax` and
  `jaxlib`. This restriction will be relaxed over time as the plugin API
  matures.

(install-amd-gpu)=
## AMD GPU

JAX has experimental ROCm support. There are two ways to install JAX:

* Use [AMD's Docker container](https://hub.docker.com/r/rocm/jax); or
* Build from source (refer to {ref}`building-from-source` — a section called _Additional notes for building a ROCM `jaxlib` for AMD GPUs_).

## Conda (community-supported)

### Conda installation

There is a community-supported Conda build of `jax`. To install it using `conda`,
simply run:

```bash
conda install jax -c conda-forge
```

To install it on a machine with an NVIDIA GPU, run:

```bash
conda install jaxlib=*=*cuda* jax cuda-nvcc -c conda-forge -c nvidia
```

Note the `cudatoolkit` distributed by `conda-forge` is missing `ptxas`, which
JAX requires. You must therefore either install the `cuda-nvcc` package from
the `nvidia` channel, or install CUDA on your machine separately so that `ptxas`
is in your path. The channel order above is important (`conda-forge` before
`nvidia`).

If you would like to override which release of CUDA is used by JAX, or to
install the CUDA build on a machine without GPUs, follow the instructions in the
[Tips & tricks](https://conda-forge.org/docs/user/tipsandtricks.html#installing-cuda-enabled-packages-like-tensorflow-and-pytorch)
section of the `conda-forge` website.

Go to the `conda-forge`
[jaxlib](https://github.com/conda-forge/jaxlib-feedstock#installing-jaxlib) and
[jax](https://github.com/conda-forge/jax-feedstock#installing-jax) repositories
for more details.


(building-jax-from-source)=
## Building JAX from source

Refer to {ref}`building-from-source`.

## Installing older `jaxlib` wheels

Due to storage limitations on the Python package index, the JAX team periodically removes
older `jaxlib` wheels from the releases on http://pypi.org/project/jax. These can
still be installed directly via the URLs here. For example:

```bash
# Install jaxlib on CPU via the wheel archive
pip install jax[cpu]==0.3.25 -f https://storage.googleapis.com/jax-releases/jax_releases.html

# Install the jaxlib 0.3.25 CPU wheel directly
pip install jaxlib==0.3.25 -f https://storage.googleapis.com/jax-releases/jax_releases.html
```
For specific older GPU wheels, be sure to use the `jax_cuda_releases.html` URL; for example
```bash
pip install jaxlib==0.3.25+cuda11.cudnn82 -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
```
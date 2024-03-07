# JAX on Cloud TPU examples

The same JAX code that runs on CPU and GPU can also be run on TPU. Cloud TPUs
have the advantage of quickly giving you access to multiple TPU accelerators,
including in [Colab](https://research.google.com/colaboratory/). All of the
example notebooks here use
[`jax.pmap`](https://jax.readthedocs.io/en/latest/jax.html#jax.pmap) to run JAX
computation across multiple TPU cores from Colab. You can also run the same code
directly on a [Cloud TPU
VM](https://cloud.google.com/tpu/docs/jax-quickstart-tpu-vm).

## Example Cloud TPU notebooks

The following notebooks showcase how to use and what you can do with Cloud TPUs on Colab:

### [Pmap Cookbook](https://colab.research.google.com/github/google/jax/blob/main/cloud_tpu_colabs/Pmap_Cookbook.ipynb)
A guide to getting started with `pmap`, a transform for easily distributing SPMD
computations across devices.

### [Lorentz ODE Solver](https://colab.research.google.com/github/google/jax/blob/main/cloud_tpu_colabs/Lorentz_ODE_Solver.ipynb)
Contributed by Alex Alemi (alexalemi@)

Solve and plot parallel ODE solutions with `pmap`.

<img src="https://raw.githubusercontent.com/google/jax/main/cloud_tpu_colabs/images/lorentz.png" width=65%></image>

### [Wave Equation](https://colab.research.google.com/github/google/jax/blob/main/cloud_tpu_colabs/Wave_Equation.ipynb)
Contributed by Stephan Hoyer (shoyer@)

Solve the wave equation with `pmap`, and make cool movies! The spatial domain is partitioned across the 8 cores of a Cloud TPU.

![](https://raw.githubusercontent.com/google/jax/main/cloud_tpu_colabs/images/wave_movie.gif)

### [JAX Demo](https://colab.research.google.com/github/google/jax/blob/main/cloud_tpu_colabs/JAX_demo.ipynb)
An overview of JAX presented at the [Program Transformations for ML workshop at NeurIPS 2019](https://program-transformations.github.io/) and the [Compilers for ML workshop at CGO 2020](https://www.c4ml.org/). Covers basic numpy usage, `grad`, `jit`, `vmap`, and `pmap`.

## Performance notes

The [guidance on running TensorFlow on TPUs](https://cloud.google.com/tpu/docs/performance-guide) applies to JAX as well, with the exception of TensorFlow-specific details. Here we highlight a few important details that are particularly relevant to using TPUs in JAX.

### Padding

One of the most common culprits for surprisingly slow code on TPUs is inadvertent padding:
- Arrays in the Cloud TPU are tiled. This entails padding one of the dimensions to a multiple of 8, and a different dimension to a multiple of 128.
- The matrix multiplication unit performs best with pairs of large matrices that minimize the need for padding.

### bfloat16 dtype

By default\*, matrix multiplication in JAX on TPUs [uses bfloat16](https://cloud.google.com/blog/products/ai-machine-learning/bfloat16-the-secret-to-high-performance-on-cloud-tpus) with float32 accumulation. This can be controlled with the `precision` keyword argument on relevant `jax.numpy` functions (`matmul`, `dot`, `einsum`, etc). In particular:
- `precision=jax.lax.Precision.DEFAULT`: uses mixed bfloat16 precision (fastest)
- `precision=jax.lax.Precision.HIGH`: uses multiple MXU passes to achieve higher precision
- `precision=jax.lax.Precision.HIGHEST`: uses even more MXU passes to achieve full float32 precision

JAX also adds the `bfloat16` dtype, which you can use to explicitly cast arrays to bfloat16, e.g., `jax.numpy.array(x, dtype=jax.numpy.bfloat16)`.

\* We might change the default precision in the future, since it is arguably surprising. Please comment/vote on [this issue](https://github.com/google/jax/issues/2161) if it affects you!

## Running JAX on a Cloud TPU VM

Refer to the [Cloud TPU VM
documentation](https://cloud.google.com/tpu/docs/jax-quickstart-tpu-vm).

## Reporting issues and getting help

If you run into Cloud TPU-specific issues (e.g. trouble creating a Cloud TPU
VM), please email <cloud-tpu-support@google.com>, or <trc-support@google.com> if
you are a [TRC](https://sites.research.google/trc/) member. You can also [file a
JAX issue](https://github.com/google/jax/issues) or [ask a discussion
question](https://github.com/google/jax/discussions) for any issues with these
notebooks or using JAX in general.

If you have any other questions or comments regarding JAX on Cloud TPUs, please
email <jax-cloud-tpu-team@google.com>. We’d like to hear from you!

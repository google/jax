---
jupytext:
  formats: ipynb,md:myst
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.16.4
kernelspec:
  display_name: Python 3 (ipykernel)
  language: python
  name: python3
---

+++ {"id": "-z6pOJwvn-_j"}

# Matrix Multiplication

In this guide, we'll write a matrix multiplication routine using Pallas. We'll also go over how to think about matmul performance on TPU and how to template a matmul kernel to fuse in operations.

```{code-cell} ipython3
:id: ejAVO6ikUUuF

#@title Imports
import functools
from typing import Callable

import jax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu
from jax import random
import jax.numpy as jnp
import numpy as np
```

+++ {"id": "58plJlycoPmT"}

## Background

Matrix multiplication is a fundamental linear algebra operation at heart of modern deep learning and language modeling. We'd like to make matmuls as speedy as possible using specialized accelerators like TPUs and GPUs, which both have specialized units for fast matrix multiplication.

To effectively utilize TPUs for matrix multiplication, we'll need to cover a few background concepts: block matrix multiplication, tiling and pipelining.

### Block Matrix Multiplication

Let's say we want to implement `matmul(x, y)` which generically multiplies an `(m, k)` array with a `(k, n)` array, but with a twist. We're only allowed to use the primitive `matmul_small` which multiples small matrices (say `m, k, n <= 256`). How could we do it?

A nice property of matrix multiplication is that each block of the output can be expressed as the sum of several smaller matrix multiplications of row blocks and column blocks of the inputs.
Formally, if we have input arrays $x \in \mathbb{R}^{m \times k}$ and $y \in \mathbb{R}^{k \times n}$ and output $z \in \mathbb{R}^{m \times n}$, we decompose them into blocks along the dimensions of size $b_m, b_k, b_n$.

For example, $x$ would be decomposed as:

$$
\begin{bmatrix}
x_{0, 0} & \cdots & x_{0, i_k} \\
x_{1, 0} & \cdots & x_{1, i_k} \\
\vdots & \ddots & \vdots \\
x_{i_m, 0} & \cdots & x_{i_m, i_k} \\
\end{bmatrix}
$$

where $x_{ik} \in \mathbb{R}^{b_m \times b_k}$. (We can similarly decompose $y$ and $z$.)

For a particular output block $z_{ij}$, we can compute it as

$$
z_{ij} = \sum_k x_{ik} y_{kj}
$$

Therefore, each output block $z_{ij}$ is the sum of several smaller block matrix multiplications $x_{ik} y_{kj}$. Here's how we'd implement this algorithm in NumPy:

```{code-cell} ipython3
:id: PACqDMtQrMOL

def matmul_small(x: np.ndarray, y: np.ndarray) -> np.ndarray:
  m, k, n = x.shape[0], x.shape[1], y.shape[0]
  assert m <= 256
  assert k <= 256
  assert n <= 256
  return np.matmul(x, y)

def block_matmul(
    x: np.ndarray,
    y: np.ndarray,
    *,
    bm: int = 256,
    bk: int = 256,
    bn: int = 256,
) -> np.ndarray:
  m, k = x.shape
  _, n = y.shape

  z = np.zeros((m, n), dtype=x.dtype)
  for m_i in range(m // bm):
    for n_i in range(n // bn):
      for k_i in range(k // bk):
        m_slice = slice(m_i * bm, (m_i + 1) * bm)
        k_slice = slice(k_i * bk, (k_i + 1) * bk)
        n_slice = slice(n_i * bn, (n_i + 1) * bn)
        x_block = x[m_slice, k_slice]
        y_block = y[k_slice, n_slice]
        z[m_slice, n_slice] += matmul_small(x_block, y_block)
  return z
```

+++ {"id": "TP49TV6q8so9"}

Our `block_matmul` function should now work on inputs larger than 256 (though we assume that our input dimensions evenly divide 256).

```{code-cell} ipython3
:id: 2SZFnWnurzEC

m, k, n = 4096, 4096, 4096
x = np.random.uniform(size=(m, k)).astype(np.float32)
y = np.random.uniform(size=(k, n)).astype(np.float32)
np.testing.assert_allclose(x @ y, block_matmul(x, y), atol=1e-6, rtol=1e-6)
```

+++ {"id": "GXtjEtEhtARN"}

`block_matmul` decomposes a matrix multiplication into many smaller ones by observing that each output chunk of size `(bm, bn)` can be computed by accumulating several `(bm, bk) x (bk, bn)` size matrix multiplications.

TPUs and GPUs do matmuls just like this! They natively support small matrix multiplication akin to `matmul_small`, so to utilize this hardware when doing bigger matrix multiplications, we will apply the `block_matmul` decomposition.

+++ {"id": "a0ESFoX1ID0z"}

### Tiling and Pipelining

In [the previous guide](pipelining), we covered how tiling up computations and pipelining in Pallas works. To make sure our compute units are always working and never stalled by memory transfers, we overlap the memory transfers for the next iteration of a kernel with the current one.

In Pallas, we specify that via `BlockSpec`s and a `grid`. Note that we already have a nested for loop in the block matrix multiplication algorithm. We can specify that in Pallas via a `grid`. The slices in the block matrix multiplication can also be specified via `BlockSpec`s.

+++ {"id": "FvYoyqlyIqo6"}

## Your first matrix multiplication kernel

+++ {"id": "umKZAlSvIt7x"}

Putting it all together, here's an implementation of a block matrix multiplication kernel that pipelines the memory transfers with the compute. We create a 3-d grid, corresponding to the 3-nested loop in the NumPy code. Note that while MXUs are only capable of multiplying small blocks, Pallas will automatically take bigger blocks and automatically tile them over the MXUs.

The last dimension of the grid corresponds to the contraction dimension of the matrix multiply and is a reduction dimension, so we need to be sure to initialize the accumulator.

```{code-cell} ipython3
:id: 75FBANKFbmQ5

def matmul_kernel(x_ref, y_ref, z_ref):
  @pl.when(pl.program_id(2) == 0)
  def _():
    z_ref[...] = jnp.zeros_like(z_ref)

  z_ref[...] += x_ref[...] @ y_ref[...]

def matmul(
    x: jax.Array,
    y: jax.Array,
    *,
    bm: int = 128,
    bk: int = 128,
    bn: int = 128,
):
  m, k = x.shape
  _, n = y.shape
  return pl.pallas_call(
      matmul_kernel,
      out_shape=jax.ShapeDtypeStruct((m, n), x.dtype),
      in_specs=[pl.BlockSpec((bm, bk), lambda i, j, k: (i, k)),
                pl.BlockSpec((bk, bn), lambda i, j, k: (k, j))],
      out_specs=pl.BlockSpec((bm, bn), lambda i, j, k: (i, j)),
      grid=(m // bm, n // bn, k // bk),
      compiler_params=pltpu.TPUCompilerParams(
          dimension_semantics=("parallel", "parallel", "arbitrary")),
  )(x, y)
```

```{code-cell} ipython3
:id: 0e8qTsimccGV

m, k, n = 4096, 4096, 4096
k1, k2 = random.split(random.key(0), 2)
x = random.normal(k1, (m, k), dtype=jnp.float32)
y = random.normal(k2, (k, n), dtype=jnp.float32)
np.testing.assert_array_equal(x @ y, matmul(x, y))
```

+++ {"id": "DycJX_-PJnnB"}

## Matrix multiplication performance

Let's think about how to analyze matrix multiplication performance. When we think about matmul performance, we typically care about two things: the total number of floating point operations (FLOPs) and the amount of memory bandwidth usage. From the [guide on TPUs and pipelining](pipelining), we see that in order to use the efficient compute units on TPUs (and on ML accelerators on general), we need to copy our inputs from HBM into VMEM, closer to the compute units. This copying to and from HBM takes time and an efficient kernel hopefully spends most of its time actually computing, as opposed to waiting for these transfers. Memory bandwidth measures the rate of this data transfer.

> Quick note: in this guide, we'll be discussing floating point operations, but want to make the distinction between FLOPs vs FLOP/s.
  When we say "FLOPs" we mean "floating point operations", as in a number of operations. When we say "FLOP/s", we refer to "floating point operations *per second*", as in a *rate* of performing floating point operations.

The number of FLOPs in a `(m, k) x (k, n)` matrix multiplication are (approximately) `2 * m * k * n`. (Technically it is `n * m * (2k - 1)` but for large enough `k` our approximation is sufficient.)

The minimum amount of memory bandwidth usage for a matrix multiply (assuming float32) is the total size of the inputs (copying into VMEM) plus the size of the output (copying into HBM). Thus the minimum bandwidth usage is `(m * k + k * n + m * n) * 4 bytes/float32`. Memory usage can be greater if we re-read the inputs multiple times, which is often the case.

One observation is that the number of matmul FLOPs is cubic in its inputs whereas the minimum bandwidth usage is quadratic in its inputs. Intuitively, this means that FLOPs grow faster than bandwidth usage, meaning that the bigger our matmul is, the more compute we have relative to copying.

```{code-cell} ipython3
:id: HZwmYZ61QZ5L
:outputId: 18505741-9254-4738-ec64-1660f6733d77

def matmul_flops(m: int, k: int, n: int):
  return 2 * m * k * n

def matmul_membw(m: int, k: int, n: int, dtype: jnp.dtype):
  return (m * k + k * n + m * n) * np.dtype(dtype).itemsize

print(matmul_flops(1024, 1024, 1024))
print(matmul_membw(1024, 1024, 1024, jnp.float32))
```

+++ {"id": "agCtb2GMQazl"}

Now that we can calculate the total number of FLOPs and (minimum) memory bandwidth usage of a matrix multiplication, let's see what a real TPU can handle.

This notebook was run on a TPU v5e chip so we'll use the v5e numbers (if you are running this notebook, your numbers may differ). TPU v5es have [197 TFLOP/s of bf16/f32 compute and 819 GB/s of memory bandwidth](https://cloud.google.com/tpu/docs/system-architecture-tpu-vm#tpu_v5e). By looking at the ratio of these numbers (called the arithmetic intensity), we can get a bound on how low this "FLOPs / memory bandwidth usage" ratio can get before we become IO bound (about 240 FLOPs/byte on TPU v5e).

```{code-cell} ipython3
:id: WUydNX2-K6Oy

v5e_flops = 197e12
v5e_membw = 819e9
v5e_op_intensity = v5e_flops / v5e_membw  # ~240.5
```

+++ {"id": "UjQIWq-9RJue"}

Roughly, these numbers tell us the FLOPs of a matmul should take `2 * m * k * n / (197 TFLOP/s)` seconds and the copies to/from VMEM should take `(m*k + k*n + m*n) * 4 bytes / 819GB/s` seconds.

```{code-cell} ipython3
:id: PiYobLc-RQSB

def matmul_flops_intensity(m: int, k: int, n: int, dtype: jnp.dtype):
  flops = matmul_flops(m, k, n)
  membw = matmul_membw(m, k, n, dtype)
  return flops / membw
```

+++ {"id": "q1y6dP00Sv9S"}

This basic calculation tells us roughly how efficiently we'll be able to use our MXUs. If our matmul op intensity is below what our chip is capable of, then our computation will be *memory bound*, i.e. our compute units will be idling while waiting for values to be transferred. If the matmul intensity is higher than what the chip is capable, then we will be *compute bound*.

Because matmul FLOPs are cubic in their input sizes and memory bandwidth usage is quadratic, we expect that we will get compute bound as we get bigger and bigger, but this crossing over point is really important! Let's say we are doing a `(1024, 1024) x (1024, 1024)` float32 matrix multiplication.

```{code-cell} ipython3
:id: NMcretZoTPjj
:outputId: 1a03e351-abcf-48d4-f81d-b8fcbe056619

print(f"{matmul_flops_intensity(1024, 1024, 1024, jnp.float32)} flops/byte")
```

+++ {"id": "U0CZSKwdTbqE"}

Our matmul flops intensity is less than what our chip is capable of. That's not good! We are likely going to be memory bound with this type of matrix multiplication. However, what if our inputs and outputs were bigger instead?  At some point when our matmuls get big enough, we will cross over from memory bound into compute bound. For example, if we have a matmul where `m = k = n`, we will cross over (on TPU v5e) when `2m**3 / 12m**2 > 240` or when `m = k = n > 1440`.

+++ {"id": "iw4c_CZIdSeV"}

### `bfloat16` matrix multiplication

+++ {"id": "7tACYDKIT3lq"}

To make it easier for matrix multiplication to be compute bound on TPU, we could also use a smaller dtype for our inputs and outputs. Our previous example used `float32` inputs and outputs but TPU v5e also supports the `bfloat16` data type (a 16-bit floating point format, also called `bf16`) for matrix multiplication as well. On TPU v5e, we will have the same FLOP/s but will *halve our memory bandwidth usage*. This makes it way easier to be compute bound for smaller matrices. Let's see what our intensity is with a 1024 x 1024 x 1024 `bf16` matrix multiply:

```{code-cell} ipython3
:id: mcuLdyDoTmnO
:outputId: 10c3dcf0-7421-49f5-a38e-e5772d791bc2

print(f"{matmul_flops_intensity(1024, 1024, 1024, jnp.bfloat16)} flops/byte")
```

+++ {"id": "XPPil1YSTn9Z"}

We now have a matmul that is compute bound!

Let's add `bf16` support to our matrix multiplication kernel.

The native MXU `bf16` matmul routine takes two input `bf16` matrices and accumulates it in `f32`. We will trigger this routine by passing `preferred_element_type=jnp.float32` into `jnp.matmul`. We will also need a accumulator `Ref` that is in `f32`. We will then downcast the output back to `bf16` before writing it back to HBM. This way we don't lose any precision, don't do any extra casting, and still retain the `bf16` memory bandwidth savings.

> Note that the only way of allocating scratch spaces right now is via `pltpu.PrefetchScalarGridSpec`. Don't worry about exactly what it does for now -- all you need to know for now is that it allows you to allocate scratch spaces in VMEM.

```{code-cell} ipython3
:id: tyMcZtA6dWDP

def matmul_kernel(x_ref, y_ref, z_ref, acc_ref, *, nsteps):
  @pl.when(pl.program_id(2) == 0)
  def _():
    acc_ref[...] = jnp.zeros_like(acc_ref)

  acc_ref[...] += jnp.dot(
      x_ref[...], y_ref[...], preferred_element_type=jnp.float32
  )

  @pl.when(pl.program_id(2) == nsteps - 1)
  def _():
    z_ref[...] = acc_ref[...].astype(z_ref.dtype)


@functools.partial(jax.jit, static_argnames=['bm', 'bk', 'bn'])
def matmul(
    x: jax.Array,
    y: jax.Array,
    *,
    bm: int = 128,
    bk: int = 128,
    bn: int = 128,
):
  m, k = x.shape
  _, n = y.shape
  return pl.pallas_call(
      functools.partial(matmul_kernel, nsteps=k // bk),
      grid_spec=pltpu.PrefetchScalarGridSpec(
        num_scalar_prefetch=0,
        in_specs=[
            pl.BlockSpec((bm, bk), lambda i, j, k: (i, k)),
            pl.BlockSpec((bk, bn), lambda i, j, k: (k, j)),
        ],
        out_specs=pl.BlockSpec((bm, bn), lambda i, j, k: (i, j)),
        scratch_shapes=[pltpu.VMEM((bm, bn), jnp.float32)],
        grid=(m // bm, n // bn, k // bk),
      ),
      out_shape=jax.ShapeDtypeStruct((m, n), x.dtype),
      compiler_params=pltpu.TPUCompilerParams(
          dimension_semantics=("parallel", "parallel", "arbitrary")),
  )(x, y)
```

```{code-cell} ipython3
:id: G3uHKEabVXep

m, k, n = 4096, 4096, 4096
k1, k2 = random.split(random.key(0), 2)
x = random.normal(k1, (m, k), dtype=jnp.bfloat16)
y = random.normal(k2, (k, n), dtype=jnp.bfloat16)
np.testing.assert_array_equal(x @ y, matmul(x, y))
```

+++ {"id": "fBL1NwXzVlWa"}

## Performance of pipelined kernels

Our above analysis about FLOPs vs memory usage applies at a coarse scale i.e. when we are looking at the the size of a the total matrix multiplication. However, remember that in practice, we are pipelining the execution of a blocked matrix multiplication, meaning we have a loop in which we are doing matrix multiplication with smaller blocks.

This means that we actually care about the FLOPs vs memory bandwidth usage of each individual instance of the kernel, not the global FLOPs vs memory bandwidth usage. Therefore, the block sizes `bm`, `bk`, `bn` are extremely important for performance. Even if we have the largest matrices in the world, if we pick very small `bm`, `bk`, and `bn`, we will be memory bound because each time we invoke the kernel we will have too few FLOPs to hide the memory transfers happening in the background.

The intuition should therefore be: to be compute bound, make the blocks as big as possible! There are two main constraints:

1. VMEM usage: The bigger our blocks, the more VMEM we use. With large enough blocks, we will run out.
2. Pipeline bubbles: The larger our blocks are relative to the matrix size, the fewer loop iterations we will have in our pipeline. This will make the size of the bubbles at the beginning and end of the pipeline larger relative to the total pipeline and this overhead can be nontrivial.

Getting good matrix multiplication performance in Pallas boils down to picking good block sizes to balance this optimization problem. In practice, we often sweep over a large set of candidate block sizes, profile the kernel, and pick the best one.

For now, let's do some very simple timing experiments. We'll use `timeit` to measure the amount of time running each kernel takes. Note that this is a upper bound on the actual runtime of the kernel since we are measuring Python dispatch and other overheads using `timeit`. We'll compute the amount of FLOP/s we obtained this way and compute the percentage utilization we get compared to what the chip offers and we'll use some reasonable block sizes to verify our intuition.

```{code-cell} ipython3
:id: RjU3sSTUWzIk
:outputId: 02b5793e-1ff3-41f4-ab45-4cf1393885ba

import timeit

def benchmark(f, ntrials: int = 100):
  def run(*args, **kwargs):
    # Compile function first
    jax.block_until_ready(f(*args, **kwargs))
    # Time function
    result = timeit.timeit(lambda: jax.block_until_ready(f(*args, **kwargs)),
                           number=ntrials)
    time = result / ntrials
    # print(f"Time: {time}")
    return time
  return run

def analyze_matmul(m: int, k: int, n: int, dtype: np.dtype,
                   mm_func):
  x = jnp.ones((m, k), dtype=dtype)
  y = jnp.ones((k, n), dtype=dtype)
  time = benchmark(mm_func)(x, y)
  print(f"----- {m} x {k} x {n} -----")
  print("Matmul time: ", time)
  mm_flops = matmul_flops(m, k, n) / time
  print("Matmul FLOP/s: ", mm_flops)
  print(f"FLOP/s utilization: {mm_flops / v5e_flops * 100:.4f}%")
  print()

print("================bm=128, bk=128, bn=128===================")
mm = functools.partial(matmul, bm=128, bk=128, bn=128)
analyze_matmul(1024, 1024, 1024, jnp.bfloat16, mm)
analyze_matmul(4096, 4096, 4096, jnp.bfloat16, mm)
analyze_matmul(8192, 8192, 8192, jnp.bfloat16, mm)

print("================bm=512, bk=1024, bn=1024===================")
mm = functools.partial(matmul, bm=512, bk=1024, bn=1024)
analyze_matmul(1024, 1024, 1024, jnp.bfloat16, mm)
analyze_matmul(4096, 4096, 4096, jnp.bfloat16, mm)
analyze_matmul(8192, 8192, 8192, jnp.bfloat16, mm)
```

+++ {"id": "mg1GMqcVan70"}

Bigger block sizes help a lot! We get pretty good utilization (80-90%) in the bigger matmuls, but the smallest matmul seems pretty hard to get good performance with.

Let's compare this with XLA's matmuls. We don't expect Pallas to do better than XLA because XLA is *very* good at generating matmuls but hopefully we are close.
With more careful block size tuning (left as future work), we can also reach XLA performance.

```{code-cell} ipython3
:id: OpU7I7BNXQYg
:outputId: 28c2c3cf-759e-465c-f969-0e2c9607b8a5

print("================ XLA matmul ===================")
mm = jnp.matmul
analyze_matmul(1024, 1024, 1024, jnp.bfloat16, mm)
analyze_matmul(4096, 4096, 4096, jnp.bfloat16, mm)
analyze_matmul(8192, 8192, 8192, jnp.bfloat16, mm)
```

+++ {"id": "L-KUG3lha-jm"}

Pallas, with some very basic tuning, gets pretty close to XLA's performance numbers! By trying out more block sizes, we should expect to close the gap entirely.

+++ {"id": "nbdHMJRObnZa"}

## Templating the matrix multiplication

+++ {"id": "qSfcMwtDg7Vn"}

Now that we have a basic matrix multiplication kernel, we can now try fusing operations into it.

### Fused right-hand-side transpose

A common first thing to do is to fuse a transpose. What do we mean by that? Suppose we wanted to compute `x @ y.T` instead of `x @ y`. Naively we could first compute `y.T` and then pass it into our efficient matrix multiply kernel. However, the operation `y.T` is not free on its own -- it involves copying `O(n^2)` data. Ideally, we could compute the transpose *while* doing the matrix multiply in just one kernel, i.e. "fusing" it with the matmul.

Accelerators often support native matrix multiplication routine that fuse a RHS transpose. For instance TPU v5e, the MXU allows us to do `x @ y.T` for small arrays. We can invoke this routine with `jax.lax.dot_general`, which will be more efficient than doing a transpose then a matmul separately.

```{code-cell} ipython3
:id: 1_6S_QnMbHAQ

def matmul_kernel(x_ref, y_ref, z_ref, acc_ref, *, nsteps, transpose_rhs):
  @pl.when(pl.program_id(2) == 0)
  def _():
    acc_ref[...] = jnp.zeros_like(acc_ref)

  # dot_general expects a data structure (contraction_dims, batch_dims),
  # where contraction_dims are the set of dimensions for LHS and RHS that will
  # be contracted (reduced) in the matmul; batch_dims, on the other hand, are
  # looped over. The remaining dimensions will be the input and output dimension
  # of the matmul.
  if transpose_rhs:
    dims = ((1,), (1,)), ((), ())
  else:
    dims = ((1,), (0,)), ((), ())

  acc_ref[...] += jax.lax.dot_general(
      x_ref[...], y_ref[...], dims, preferred_element_type=jnp.float32,
  )

  @pl.when(pl.program_id(2) == nsteps - 1)
  def _():
    z_ref[...] = acc_ref[...].astype(z_ref.dtype)


@functools.partial(jax.jit, static_argnames=['bm', 'bk', 'bn', 'transpose_rhs'])
def matmul(
    x: jax.Array,
    y: jax.Array,
    *,
    bm: int = 128,
    bk: int = 128,
    bn: int = 128,
    transpose_rhs: bool = False,
):
  if transpose_rhs:
    y = y.swapaxes(0, 1)
    y_block_spec = pl.BlockSpec((bn, bk), lambda i, j, k: (j, k))
  else:
    y_block_spec = pl.BlockSpec((bk, bn), lambda i, j, k: (k, j))
  m, k = x.shape
  _, n = y.shape
  return pl.pallas_call(
      functools.partial(matmul_kernel, nsteps=k // bk, transpose_rhs=transpose_rhs),
      grid_spec=pltpu.PrefetchScalarGridSpec(
        num_scalar_prefetch=0,
        in_specs=[
            pl.BlockSpec((bm, bk), lambda i, j, k: (i, k)),
            y_block_spec,
        ],
        out_specs=pl.BlockSpec((bm, bn), lambda i, j, k: (i, j)),
        scratch_shapes=[pltpu.VMEM((bm, bn), jnp.float32)],
        grid=(m // bm, n // bn, k // bk),
      ),
      out_shape=jax.ShapeDtypeStruct((m, n), x.dtype),
      compiler_params=pltpu.TPUCompilerParams(
          dimension_semantics=("parallel", "parallel", "arbitrary")),
  )(x, y)
```

+++ {"id": "eSmPJHSchuGX"}

We do a transpose inside of the `matmul` function (`y = y.swapaxes(0, 1)`). This is because inside of a JIT-ted JAX computation, dimension ordering is purely *logical*, not physical, so rearranging dimensions does not imply a
physical layout difference. However, when we pass an array into a `pallas_call`, we do enforce a major-to-minor dimension ordering constraint. By transposing `y` inside of the `matmul` function, we are requesting that `y` be in a
transposed layout `(n, k)` instead of the usual `(k, n)`. The user will still pass in the array in the (logical) `(k, n)` dimension, however.

Note: to benchmark the transpose, we actually want `y` to be in the physical transposed layout when we pass it into the kernel, so we don't measure relayout time. In the wrapper function, we will (logically) transpose it back to `(k, n)`
before passing it into `matmul` because `matmul` expects a logical `(k, n)` dimension ordering.

```{code-cell} ipython3
:id: AcBMHhKLhkDp
:outputId: 48f2f70b-c94d-44eb-c781-871c36cf457f

def analyze_matmul(m: int, k: int, n: int, dtype: np.dtype,
                   mm_func, transpose_rhs: bool = False):
  x = jnp.ones((m, k), dtype=dtype)
  if transpose_rhs:
    y = jnp.ones((n, k), dtype=dtype)
    @jax.jit
    def _wrapper(x, y):
      y = y.swapaxes(0, 1)
      return mm_func(x, y, transpose_rhs=True)
  else:
    y = jnp.ones((k, n), dtype=dtype)
    _wrapper = mm_func
  time = benchmark(_wrapper)(x, y)
  print(f"----- {m} x {k} x {n} -----")
  print("Matmul time: ", time)
  mm_flops = matmul_flops(m, k, n) / time
  print("Matmul FLOP/s: ", mm_flops)
  print(f"FLOP/s utilization: {mm_flops / v5e_flops * 100:.4f}%")
  print()

print("================bm=128, bk=128, bn=128===================")
mm = functools.partial(matmul, bm=128, bk=128, bn=128)
analyze_matmul(1024, 1024, 1024, jnp.bfloat16, mm, transpose_rhs=True)
analyze_matmul(4096, 4096, 4096, jnp.bfloat16, mm, transpose_rhs=True)
analyze_matmul(8192, 8192, 8192, jnp.bfloat16, mm, transpose_rhs=True)

print("================bm=512, bk=1024, bn=1024===================")
mm = functools.partial(matmul, bm=512, bk=1024, bn=1024)
analyze_matmul(1024, 1024, 1024, jnp.bfloat16, mm, transpose_rhs=True)
analyze_matmul(4096, 4096, 4096, jnp.bfloat16, mm, transpose_rhs=True)
analyze_matmul(8192, 8192, 8192, jnp.bfloat16, mm, transpose_rhs=True)
```

+++ {"id": "E0P8lWhskn3j"}

See how we get the same utilization despite the extra transpose!

+++ {"id": "DUYGnu7zkz8v"}

### Fused activation function

Fusing in an activation is also really common. This makes sure we don't follow an efficient, compute bound matmul kernel with a slow memory bound activation kernel.

```{code-cell} ipython3
:id: SANr6fyBiso_

def matmul_kernel(
    x_ref, y_ref, z_ref, acc_ref, *, nsteps, transpose_rhs, activation
):
  @pl.when(pl.program_id(2) == 0)
  def _():
    acc_ref[...] = jnp.zeros_like(acc_ref)

  if transpose_rhs:
    dims = ((1,), (1,)), ((), ())
  else:
    dims = ((1,), (0,)), ((), ())

  acc_ref[...] += jax.lax.dot_general(
      x_ref[...],
      y_ref[...],
      dims,
      preferred_element_type=jnp.float32,
  )

  @pl.when(pl.program_id(2) == nsteps - 1)
  def _():
    z_ref[...] = activation(acc_ref[...]).astype(z_ref.dtype)


@functools.partial(jax.jit, static_argnames=['bm', 'bk', 'bn', 'activation'])
def matmul(
    x: jax.Array,
    y: jax.Array,
    *,
    bm: int = 128,
    bk: int = 128,
    bn: int = 128,
    transpose_rhs: bool = False,
    activation: Callable[[jax.Array], jax.Array] = lambda x: x,
):
  if transpose_rhs:
    y = y.swapaxes(0, 1)
    y_block_spec = pl.BlockSpec((bn, bk), lambda i, j, k: (j, k))
  else:
    y_block_spec = pl.BlockSpec((bk, bn), lambda i, j, k: (k, j))
  m, k = x.shape
  _, n = y.shape
  return pl.pallas_call(
      functools.partial(
          matmul_kernel,
          nsteps=k // bk,
          transpose_rhs=transpose_rhs,
          activation=activation,
      ),
      grid_spec=pltpu.PrefetchScalarGridSpec(
          num_scalar_prefetch=0,
          in_specs=[
              pl.BlockSpec((bm, bk), lambda i, j, k: (i, k)),
              y_block_spec,
          ],
          out_specs=pl.BlockSpec((bm, bn), lambda i, j, k: (i, j)),
          scratch_shapes=[pltpu.VMEM((bm, bn), jnp.float32)],
          grid=(m // bm, n // bn, k // bk),
      ),
      out_shape=jax.ShapeDtypeStruct((m, n), x.dtype),
      compiler_params=pltpu.TPUCompilerParams(
          dimension_semantics=("parallel", "parallel", "arbitrary")),
  )(x, y)
```

```{code-cell} ipython3
:id: BOu7WBCBlHpN
:outputId: 4b7f72c4-f562-4a49-cc48-17bf0c845434

def analyze_matmul(m: int, k: int, n: int, dtype: np.dtype,
                   mm_func, transpose_rhs: bool = False,
                   activation = lambda x: x):
  x = jnp.ones((m, k), dtype=dtype)
  if transpose_rhs:
    y = jnp.ones((n, k), dtype=dtype)
    @jax.jit
    def _wrapper(x, y):
      y = y.swapaxes(0, 1)
      return mm_func(x, y, transpose_rhs=True, activation=activation)
  else:
    y = jnp.ones((k, n), dtype=dtype)
    _wrapper = functools.partial(mm_func, activation=activation)
  time = benchmark(_wrapper)(x, y)
  print(f"----- {m} x {k} x {n} -----")
  print("Matmul time: ", time)
  mm_flops = matmul_flops(m, k, n) / time
  print("Matmul FLOP/s: ", mm_flops)
  print(f"FLOP/s utilization: {mm_flops / v5e_flops * 100:.4f}%")
  print()


activation = jax.nn.relu
print("================bm=128, bk=128, bn=128===================")
mm = functools.partial(matmul, bm=128, bk=128, bn=128)
analyze_matmul(1024, 1024, 1024, jnp.bfloat16, mm, activation=activation)
analyze_matmul(4096, 4096, 4096, jnp.bfloat16, mm, activation=activation)
analyze_matmul(8192, 8192, 8192, jnp.bfloat16, mm, activation=activation)

print("================bm=512, bk=1024, bn=1024===================")
mm = functools.partial(matmul, bm=512, bk=1024, bn=1024)
analyze_matmul(1024, 1024, 1024, jnp.bfloat16, mm, activation=activation)
analyze_matmul(4096, 4096, 4096, jnp.bfloat16, mm, activation=activation)
analyze_matmul(8192, 8192, 8192, jnp.bfloat16, mm, activation=activation)
```

+++ {"id": "tIekGWFLmgtS"}

The additional fused activation barely affects our utilization at all!

+++ {"id": "faNZwx20mpJi"}

## Conclusion

In this guide, we covered how to write efficient matrix multiplications on TPU using Pallas. We discussed blocked matrix multiplication and pipelining, how to analyze the performance of a TPU matmul, and how to write an efficient `bf16` matrix multiplication. We concluded with templating the matrix multiplication to support a fused transpose and fused activation functions.

Exercises left to the reader:
* Add support for input fusions. Sometimes we want to fuse an operation into the inputs of the matmul. Try templating the matrix multiplication even more to support this.
* Add support for `int8` matrix multiplication. TPU v5 supports native `int8` matrix multiplication at twice the FLOPs of `bf16`. Try adding support for that and see what utilization is possible.
* Add backwards pass support for the `matmul` function. You can do this with `jax.custom_vjp`.

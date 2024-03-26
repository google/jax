---
jupytext:
  formats: ipynb,md:myst
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.16.0
kernelspec:
  display_name: Python 3
  language: python
  name: python3
---

+++ {"id": "xtWX4x9DCF5_"}

# JAX Quickstart

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/google/jax/blob/main/docs/notebooks/quickstart.ipynb) [![Open in Kaggle](https://kaggle.com/static/images/open-in-kaggle.svg)](https://kaggle.com/kernels/welcome?src=https://github.com/google/jax/blob/main/docs/notebooks/quickstart.ipynb)

**JAX is NumPy on the CPU, GPU, and TPU, with great automatic differentiation for high-performance machine learning research.**

With its updated version of [Autograd](https://github.com/hips/autograd), JAX
can automatically differentiate native Python and NumPy code. It can
differentiate through a large subset of Python’s features, including loops, ifs,
recursion, and closures, and it can even take derivatives of derivatives of
derivatives. It supports reverse-mode as well as forward-mode differentiation, and the two can be composed arbitrarily
to any order.

What’s new is that JAX uses
[XLA](https://www.tensorflow.org/xla)
to compile and run your NumPy code on accelerators, like GPUs and TPUs.
Compilation happens under the hood by default, with library calls getting
just-in-time compiled and executed. But JAX even lets you just-in-time compile
your own Python functions into XLA-optimized kernels using a one-function API.
Compilation and automatic differentiation can be composed arbitrarily, so you
can express sophisticated algorithms and get maximal performance without having
to leave Python.

```{code-cell} ipython3
:id: SY8mDvEvCGqk

import jax.numpy as jnp
from jax import grad, jit, vmap
from jax import random
```

+++ {"id": "FQ89jHCYfhpg"}

## Multiplying Matrices

+++ {"id": "Xpy1dSgNqCP4"}

We'll be generating random data in the following examples. One big difference between NumPy and JAX is how you generate random numbers. For more details, see [Common Gotchas in JAX].

[Common Gotchas in JAX]: https://jax.readthedocs.io/en/latest/notebooks/Common_Gotchas_in_JAX.html#%F0%9F%94%AA-Random-Numbers

```{code-cell} ipython3
:id: u0nseKZNqOoH
:outputId: 03e20e21-376c-41bb-a6bb-57431823691b

key = random.key(0)
x = random.normal(key, (10,))
print(x)
```

+++ {"id": "hDJF0UPKnuqB"}

Let's dive right in and multiply two big matrices.

```{code-cell} ipython3
:id: eXn8GUl6CG5N
:outputId: ffce6bdc-86e6-4af0-ab5d-65d235022db9

size = 3000
x = random.normal(key, (size, size), dtype=jnp.float32)
%timeit jnp.dot(x, x.T).block_until_ready()  # runs on the GPU
```

+++ {"id": "0AlN7EbonyaR"}

We added that `block_until_ready` because JAX uses asynchronous execution by default (see {ref}`async-dispatch`).

JAX NumPy functions work on regular NumPy arrays.

```{code-cell} ipython3
:id: ZPl0MuwYrM7t
:outputId: 71219657-b559-474e-a877-5441ee39f18f

import numpy as np
x = np.random.normal(size=(size, size)).astype(np.float32)
%timeit jnp.dot(x, x.T).block_until_ready()
```

+++ {"id": "_SrcB2IurUuE"}

That's slower because it has to transfer data to the GPU every time. You can ensure that an NDArray is backed by device memory using {func}`~jax.device_put`.

```{code-cell} ipython3
:id: Jj7M7zyRskF0
:outputId: a649a6d3-cf28-445e-c3fc-bcfe3069482c

from jax import device_put

x = np.random.normal(size=(size, size)).astype(np.float32)
x = device_put(x)
%timeit jnp.dot(x, x.T).block_until_ready()
```

+++ {"id": "clO9djnen8qi"}

The output of {func}`~jax.device_put` still acts like an NDArray, but it only copies values back to the CPU when they're needed for printing, plotting, saving to disk, branching, etc. The behavior of {func}`~jax.device_put` is equivalent to the function `jit(lambda x: x)`, but it's faster.

+++ {"id": "ghkfKNQttDpg"}

If you have a GPU (or TPU!) these calls run on the accelerator and have the potential to be much faster than on CPU.
See {ref}`faq-jax-vs-numpy` for more comparison of performance characteristics of NumPy and JAX

+++ {"id": "iOzp0P_GoJhb"}

JAX is much more than just a GPU-backed NumPy. It also comes with a few program transformations that are useful when writing numerical code. For now, there are three main ones:

 - {func}`~jax.jit`, for speeding up your code
 - {func}`~jax.grad`, for taking derivatives
 - {func}`~jax.vmap`, for automatic vectorization or batching.

Let's go over these, one-by-one. We'll also end up composing these in interesting ways.

+++ {"id": "bTTrTbWvgLUK"}

## Using {func}`~jax.jit` to speed up functions

+++ {"id": "YrqE32mvE3b7"}

JAX runs transparently on the GPU or TPU (falling back to CPU if you don't have one). However, in the above example, JAX is dispatching kernels to the GPU one operation at a time. If we have a sequence of operations, we can use the `jax.jit` function to compile multiple operations together using [XLA](https://www.tensorflow.org/xla). Let's try that.

```{code-cell} ipython3
:id: qLGdCtFKFLOR
:outputId: 870253fa-ba1b-47ec-c5a4-1c6f706be996

def selu(x, alpha=1.67, lmbda=1.05):
  return lmbda * jnp.where(x > 0, x, alpha * jnp.exp(x) - alpha)

x = random.normal(key, (1000000,))
%timeit selu(x).block_until_ready()
```

+++ {"id": "a_V8SruVHrD_"}

We can speed it up with `jax.jit`, which will jit-compile the first time `selu` is called and will be cached thereafter.

```{code-cell} ipython3
:id: fh4w_3NpFYTp
:outputId: 4d56b4f2-5d58-4689-ecc2-ac361c0245cd

selu_jit = jit(selu)
%timeit selu_jit(x).block_until_ready()
```

+++ {"id": "HxpBc4WmfsEU"}

## Taking derivatives with {func}`~jax.grad`

In addition to evaluating numerical functions, we also want to transform them. One transformation is [automatic differentiation](https://en.wikipedia.org/wiki/Automatic_differentiation). In JAX, just like in [Autograd](https://github.com/HIPS/autograd), you can compute gradients with the {func}`~jax.grad` function.

```{code-cell} ipython3
:id: IMAgNJaMJwPD
:outputId: 6646cc65-b52f-4825-ff7f-e50b67083493

def sum_logistic(x):
  return jnp.sum(1.0 / (1.0 + jnp.exp(-x)))

x_small = jnp.arange(3.)
derivative_fn = grad(sum_logistic)
print(derivative_fn(x_small))
```

+++ {"id": "PtNs881Ohioc"}

Let's verify with finite differences that our result is correct.

```{code-cell} ipython3
:id: JXI7_OZuKZVO
:outputId: 18c1f913-d5d6-4895-f71e-e62180c3ad1b

def first_finite_differences(f, x):
  eps = 1e-3
  return jnp.array([(f(x + eps * v) - f(x - eps * v)) / (2 * eps)
                   for v in jnp.eye(len(x))])


print(first_finite_differences(sum_logistic, x_small))
```

+++ {"id": "Q2CUZjOWNZ-3"}

Taking derivatives is as easy as calling {func}`~jax.grad`. {func}`~jax.grad` and {func}`~jax.jit` compose and can be mixed arbitrarily. In the above example we jitted `sum_logistic` and then took its derivative. We can go further:

```{code-cell} ipython3
:id: TO4g8ny-OEi4
:outputId: 1a0421e6-60e9-42e3-dc9c-e558a69bbf17

print(grad(jit(grad(jit(grad(sum_logistic)))))(1.0))
```

+++ {"id": "yCJ5feKvhnBJ"}

For more advanced autodiff, you can use {func}`jax.vjp` for reverse-mode vector-Jacobian products and {func}`jax.jvp` for forward-mode Jacobian-vector products. The two can be composed arbitrarily with one another, and with other JAX transformations. Here's one way to compose them to make a function that efficiently computes full Hessian matrices:

```{code-cell} ipython3
:id: Z-JxbiNyhxEW

from jax import jacfwd, jacrev
def hessian(fun):
  return jit(jacfwd(jacrev(fun)))
```

+++ {"id": "TI4nPsGafxbL"}

## Auto-vectorization with {func}`~jax.vmap`

+++ {"id": "PcxkONy5aius"}

JAX has one more transformation in its API that you might find useful: {func}`~jax.vmap`, the vectorizing map. It has the familiar semantics of mapping a function along array axes, but instead of keeping the loop on the outside, it pushes the loop down into a function’s primitive operations for better performance. When composed with {func}`~jax.jit`, it can be just as fast as adding the batch dimensions by hand.

+++ {"id": "TPiX4y-bWLFS"}

We're going to work with a simple example, and promote matrix-vector products into matrix-matrix products using {func}`~jax.vmap`. Although this is easy to do by hand in this specific case, the same technique can apply to more complicated functions.

```{code-cell} ipython3
:id: 8w0Gpsn8WYYj

mat = random.normal(key, (150, 100))
batched_x = random.normal(key, (10, 100))

def apply_matrix(v):
  return jnp.dot(mat, v)
```

+++ {"id": "0zWsc0RisQWx"}

Given a function such as `apply_matrix`, we can loop over a batch dimension in Python, but usually the performance of doing so is poor.

```{code-cell} ipython3
:id: KWVc9BsZv0Ki
:outputId: bea78b6d-cd17-45e6-c361-1c55234e77c0

def naively_batched_apply_matrix(v_batched):
  return jnp.stack([apply_matrix(v) for v in v_batched])

print('Naively batched')
%timeit naively_batched_apply_matrix(batched_x).block_until_ready()
```

+++ {"id": "qHfKaLE9stbA"}

We know how to batch this operation manually. In this case, `jnp.dot` handles extra batch dimensions transparently. Note that to jit-compile this function, we'll use the `@jit` decoratod (instead of calling `jax.jit`).

```{code-cell} ipython3
:id: ipei6l8nvrzH
:outputId: 335cdc4c-c603-497b-fc88-3fa37c5630c2

@jit
def batched_apply_matrix(v_batched):
  return jnp.dot(v_batched, mat.T)

print('Manually batched')
%timeit batched_apply_matrix(batched_x).block_until_ready()
```

+++ {"id": "1eF8Nhb-szAb"}

However, suppose we had a more complicated function without batching support. We can use {func}`~jax.vmap` to add batching support automatically.

```{code-cell} ipython3
:id: 67Oeknf5vuCl
:outputId: 9c680e74-ebb5-4563-ebfc-869fd82de091

@jit
def vmap_batched_apply_matrix(v_batched):
  return vmap(apply_matrix)(v_batched)

print('Auto-vectorized with vmap')
%timeit vmap_batched_apply_matrix(batched_x).block_until_ready()
```

+++ {"id": "pYVl3Z2nbZhO"}

Of course, {func}`~jax.vmap` can be arbitrarily composed with {func}`~jax.jit`, {func}`~jax.grad`, and any other JAX transformation.

+++ {"id": "WwNnjaI4th_8"}

This is just a taste of what JAX can do. We're really excited to see what you do with it!

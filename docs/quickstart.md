---
jupytext:
  formats: md:myst
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.16.1
kernelspec:
  display_name: Python 3
  language: python
  name: python3
---

# Quickstart

<!--* freshness: { reviewed: '2024-06-13' } *-->

**JAX a library for array-oriented numerical computation (*à la* [NumPy](https://numpy.org/)), with automatic differentiation and JIT compilation to enable high-performance machine learning research**.

This document provides a quick overview of essential JAX features, so you can get started with JAX quickly:

* JAX provides a unified NumPy-like interface to computations that run on CPU, GPU, or TPU, in local or distributed settings.
* JAX features built-in Just-In-Time (JIT) compilation via [Open XLA](https://github.com/openxla), an open-source machine learning compiler ecosystem.
* JAX functions support efficient evaluation of gradients via its automatic differentiation transformations.
* JAX functions can be automatically vectorized to efficiently map them over arrays representing batches of inputs.

## Installation

JAX can be installed for CPU on Linux, Windows, and macOS directly from the [Python Package Index](https://pypi.org/project/jax/):
```
pip install jax
```
or, for NVIDIA GPU:
```
pip install -U "jax[cuda12]"
```
For more detailed platform-specific installation information, check out {ref}`installation`.

## JAX as NumPy

Most JAX usage is through the familiar {mod}`jax.numpy` API, which is typically imported under the `jnp` alias:

```{code-cell}
import jax.numpy as jnp
```

With this import, you can immediately use JAX in a similar manner to typical NumPy programs,
including using NumPy-style array creation functions, Python functions and operators, and
array attributes and methods:

```{code-cell}
def selu(x, alpha=1.67, lmbda=1.05):
  return lmbda * jnp.where(x > 0, x, alpha * jnp.exp(x) - alpha)

x = jnp.arange(5.0)
print(selu(x))
```

You'll find a few differences between JAX arrays and NumPy arrays once you begin digging-in;
these are explored in  [🔪 JAX - The Sharp Bits 🔪](https://jax.readthedocs.io/en/latest/notebooks/Common_Gotchas_in_JAX.html).

## Just-in-time compilation with {func}`jax.jit`
JAX runs transparently on the GPU or TPU (falling back to CPU if you don't have one). However, in the above example, JAX is dispatching kernels to the chip one operation at a time. If we have a sequence of operations, we can use the {func}`jax.jit` function to compile this sequence of operations together using XLA.

We can use IPython's `%timeit` to quickly benchmark our `selu` function, using `block_until_ready()` to
account for JAX's dynamic dispatch (See {ref}`async-dispatch`):

```{code-cell}
from jax import random

key = random.key(1701)
x = random.normal(key, (1_000_000,))
%timeit selu(x).block_until_ready()
```

(notice we've used {mod}`jax.random` to generate some random numbers; for details on
how to generate random numbers in JAX, check out {ref}`pseudorandom-numbers`).

We can speed the execution of this function with the {func}`jax.jit` transformation,
which will jit-compile the first time `selu` is called and will be cached thereafter.

```{code-cell}
from jax import jit

selu_jit = jit(selu)
_ = selu_jit(x)  # compiles on first call
%timeit selu_jit(x).block_until_ready()
```

The above timing represent execution on CPU, but the same code can be run on GPU or TPU,
typically for an even greater speedup.

For more on JIT compilation in JAX, check out {ref}`jit-compilation`.

## Taking derivatives with {func}`jax.grad`

In addition to transforming functions via JIT compilation, JAX also provides other
transformations. One such transformation is {func}`jax.grad`, which performs
[automatic differentiation (autodiff)](https://en.wikipedia.org/wiki/Automatic_differentiation):

```{code-cell}
from jax import grad

def sum_logistic(x):
  return jnp.sum(1.0 / (1.0 + jnp.exp(-x)))

x_small = jnp.arange(3.)
derivative_fn = grad(sum_logistic)
print(derivative_fn(x_small))
```

Let's verify with finite differences that our result is correct.

```{code-cell}
def first_finite_differences(f, x, eps=1E-3):
  return jnp.array([(f(x + eps * v) - f(x - eps * v)) / (2 * eps)
                   for v in jnp.eye(len(x))])

print(first_finite_differences(sum_logistic, x_small))
```

The {func}`~jax.grad` and {func}`~jax.jit` transformations compose and can be mixed arbitrarily.
In the above example we jitted `sum_logistic` and then took its derivative. We can go further:

```{code-cell}
print(grad(jit(grad(jit(grad(sum_logistic)))))(1.0))
```

Beyond scalar-valued functions, the {func}`jax.jacobian` transformation can be
used to compute the full Jacobian matrix for vector-valued functions:

```{code-cell}
from jax import jacobian
print(jacobian(jnp.exp)(x_small))
```

For more advanced autodiff operations, you can use {func}`jax.vjp` for reverse-mode vector-Jacobian products,
and {func}`jax.jvp` and {func}`jax.linearize` for forward-mode Jacobian-vector products.
The two can be composed arbitrarily with one another, and with other JAX transformations.
For example, {func}`jax.jvp` and {func}`jax.vjp` are used to define the forward-mode {func}`jax.jacfwd` and reverse-mode {func}`jax.jacrev` for computing Jacobians in forward- and reverse-mode, respectively.
Here's one way to compose them to make a function that efficiently computes full Hessian matrices:

```{code-cell}
from jax import jacfwd, jacrev
def hessian(fun):
  return jit(jacfwd(jacrev(fun)))
print(hessian(sum_logistic)(x_small))
```

This kind of composition produces efficient code in practice; this is more-or-less how JAX's built-in {func}`jax.hessian` function is implemented.

For more on automatic differentiation in JAX, check out {ref}`automatic-differentiation`.

## Auto-vectorization with {func}`jax.vmap`

Another useful transformation is {func}`~jax.vmap`, the vectorizing map.
It has the familiar semantics of mapping a function along array axes, but instead of explicitly looping
over function calls, it transforms the function into a natively vectorized version for better performance.
When composed with {func}`~jax.jit`, it can be just as performant as manually rewriting your function
operate over an extra batch dimension.

We're going to work with a simple example, and promote matrix-vector products into matrix-matrix products using {func}`~jax.vmap`.
Although this is easy to do by hand in this specific case, the same technique can apply to more complicated functions.

```{code-cell}
key1, key2 = random.split(key)
mat = random.normal(key1, (150, 100))
batched_x = random.normal(key2, (10, 100))

def apply_matrix(x):
  return jnp.dot(mat, x)
```

The `apply_matrix` function maps a vector to a vector, but we may want to apply it row-wise across a matrix.
We could do this by looping over the batch dimension in Python, but this usually results in poor performance.

```{code-cell}
def naively_batched_apply_matrix(v_batched):
  return jnp.stack([apply_matrix(v) for v in v_batched])

print('Naively batched')
%timeit naively_batched_apply_matrix(batched_x).block_until_ready()
```

A programmer familiar with the the `jnp.dot` function might recognize that `apply_matrix` can
be rewritten to avoid explicit looping, using the built-in batching semantics of `jnp.dot`:

```{code-cell}
import numpy as np

@jit
def batched_apply_matrix(batched_x):
  return jnp.dot(batched_x, mat.T)

np.testing.assert_allclose(naively_batched_apply_matrix(batched_x),
                           batched_apply_matrix(batched_x), atol=1E-4, rtol=1E-4)
print('Manually batched')
%timeit batched_apply_matrix(batched_x).block_until_ready()
```

However, as functions become more complicated, this kind of manual batching becomes more difficult and error-prone.
The {func}`~jax.vmap` transformation is designed to automatically transform a function into a batch-aware version:

```{code-cell}
from jax import vmap

@jit
def vmap_batched_apply_matrix(batched_x):
  return vmap(apply_matrix)(batched_x)

np.testing.assert_allclose(naively_batched_apply_matrix(batched_x),
                           vmap_batched_apply_matrix(batched_x), atol=1E-4, rtol=1E-4)
print('Auto-vectorized with vmap')
%timeit vmap_batched_apply_matrix(batched_x).block_until_ready()
```

As you would expect, {func}`~jax.vmap` can be arbitrarily composed with {func}`~jax.jit`,
{func}`~jax.grad`, and any other JAX transformation.

For more on automatic vectorization in JAX, check out {ref}`automatic-vectorization`.

This is just a taste of what JAX can do. We're really excited to see what you do with it!

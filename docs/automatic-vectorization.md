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

(automatic-vectorization)=
# Automatic vectorization

<!--* freshness: { reviewed: '2024-05-03' } *-->

In the previous section we discussed JIT compilation via the {func}`jax.jit` function.
This notebook discusses another of JAX's transforms: vectorization via {func}`jax.vmap`.

## Manual vectorization

Consider the following simple code that computes the convolution of two one-dimensional vectors:

```{code-cell}
import jax
import jax.numpy as jnp

x = jnp.arange(5)
w = jnp.array([2., 3., 4.])

def convolve(x, w):
  output = []
  for i in range(1, len(x)-1):
    output.append(jnp.dot(x[i-1:i+2], w))
  return jnp.array(output)

convolve(x, w)
```

Suppose we would like to apply this function to a batch of weights `w` to a batch of vectors `x`.

```{code-cell}
xs = jnp.stack([x, x])
ws = jnp.stack([w, w])
```

The most naive option would be to simply loop over the batch in Python:

```{code-cell}
def manually_batched_convolve(xs, ws):
  output = []
  for i in range(xs.shape[0]):
    output.append(convolve(xs[i], ws[i]))
  return jnp.stack(output)

manually_batched_convolve(xs, ws)
```

This produces the correct result, however it is not very efficient.

In order to batch the computation efficiently, you would normally have to rewrite the function manually to ensure it is done in vectorized form. This is not particularly difficult to implement, but does involve changing how the function treats indices, axes, and other parts of the input.

For example, we could manually rewrite `convolve()` to support vectorized computation across the batch dimension as follows:

```{code-cell}
def manually_vectorized_convolve(xs, ws):
  output = []
  for i in range(1, xs.shape[-1] -1):
    output.append(jnp.sum(xs[:, i-1:i+2] * ws, axis=1))
  return jnp.stack(output, axis=1)

manually_vectorized_convolve(xs, ws)
```

Such re-implementation can be messy and error-prone as the complexity of a function increases; fortunately JAX provides another way.

## Automatic vectorization

In JAX, the {func}`jax.vmap` transformation is designed to generate such a vectorized implementation of a function automatically:

```{code-cell}
auto_batch_convolve = jax.vmap(convolve)

auto_batch_convolve(xs, ws)
```

It does this by tracing the function similarly to {func}`jax.jit`, and automatically adding batch axes at the beginning of each input.

If the batch dimension is not the first, you may use the `in_axes` and `out_axes` arguments to specify the location of the batch dimension in inputs and outputs. These may be an integer if the batch axis is the same for all inputs and outputs, or lists, otherwise.

```{code-cell}
auto_batch_convolve_v2 = jax.vmap(convolve, in_axes=1, out_axes=1)

xst = jnp.transpose(xs)
wst = jnp.transpose(ws)

auto_batch_convolve_v2(xst, wst)
```

{func}`jax.vmap` also supports the case where only one of the arguments is batched: for example, if you would like to convolve to a single set of weights `w` with a batch of vectors `x`; in this case the `in_axes` argument can be set to `None`:

```{code-cell}
batch_convolve_v3 = jax.vmap(convolve, in_axes=[0, None])

batch_convolve_v3(xs, w)
```

## Combining transformations

As with all JAX transformations, {func}`jax.jit` and {func}`jax.vmap` are designed to be composable, which means you can wrap a vmapped function with `jit`, or a jitted function with `vmap`, and everything will work correctly:

```{code-cell}
jitted_batch_convolve = jax.jit(auto_batch_convolve)

jitted_batch_convolve(xs, ws)
```

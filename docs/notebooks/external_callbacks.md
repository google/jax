---
jupytext:
  formats: ipynb,md:myst
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.16.1
kernelspec:
  display_name: Python 3
  name: python3
---

+++ {"id": "7XNMxdTwURqI"}

# External Callbacks in JAX

<!--* freshness: { reviewed: '2024-04-08' } *-->

+++ {"id": "h6lXo6bSUYGq"}

This guide outlines the uses of various callback functions, which allow JAX runtimes to execute Python code on the host, even while running under `jit`, `vmap`, `grad`, or another transformation.

+++ {"id": "Xi_nhfpnlmbm"}

## Why callbacks?

A callback routine is a way to perform **host-side** execution of code at runtime.
As a simple example, suppose you'd like to print the *value* of some variable during the course of a computation.
Using a simple Python `print` statement, it looks like this:

```{code-cell}
:id: lz8rEL1Amb4r
:outputId: bbd37102-19f2-46d2-b794-3d4952c6fe97

import jax

@jax.jit
def f(x):
  y = x + 1
  print("intermediate value: {}".format(y))
  return y * 2

result = f(2)
```

+++ {"id": "yEy41sFAmxOp"}

What is printed is not the runtime value, but the trace-time abstract value (if you're not famililar with *tracing* in JAX, a good primer can be found in [How To Think In JAX](https://jax.readthedocs.io/en/latest/notebooks/thinking_in_jax.html)).

To print the value at runtime we need a callback, for example `jax.debug.print`:

```{code-cell}
:id: wFfHmoQxnKDF
:outputId: 6bea21d9-9bb1-4d4d-f3ec-fcf1c691a46a

@jax.jit
def f(x):
  y = x + 1
  jax.debug.print("intermediate value: {}", y)
  return y * 2

result = f(2)
```

+++ {"id": "CvWv3pudn9X5"}

This works by passing the runtime value represented by `y` back to the host process, where the host can print the value.

+++ {"id": "X0vR078znuT-"}

## Flavors of Callback

In earlier versions of JAX, there was only one kind of callback available, implemented in `jax.experimental.host_callback`. The `host_callback` routines had some deficiencies, and are now deprecated in favor of several callbacks designed for different situations:

- {func}`jax.pure_callback`: appropriate for pure functions: i.e. functions with no side effect.
- {func}`jax.experimental.io_callback`: appropriate for impure functions: e.g. functions which read or write data to disk.
- {func}`jax.debug.callback`: appropriate for functions that should reflect the execution behavior of the compiler.

(The {func}`jax.debug.print` function we used above is a wrapper around {func}`jax.debug.callback`).

From the user perspective, these three flavors of callback are mainly distinguished by what transformations and compiler optimizations they allow.

|callback function | supports return value | `jit` | `vmap` | `grad` | `scan`/`while_loop` | guaranteed execution |
|-------------------------------------|----|----|----|----|----|----|
|`jax.pure_callback`            | ✅ | ✅ | ✅ | ❌¹ | ✅ | ❌ |
|`jax.experimental.io_callback` | ✅ | ✅ | ✅/❌² | ❌ | ✅³ | ✅ |
|`jax.debug.callback`           | ❌ | ✅ | ✅ | ✅ | ✅ | ❌ |

¹ `jax.pure_callback` can be used with `custom_jvp` to make it compatible with autodiff

² `jax.experimental.io_callback` is compatible with `vmap` only if `ordered=False`.

³ Note that `vmap` of `scan`/`while_loop` of `io_callback` has complicated semantics, and its behavior may change in future releases.

+++ {"id": "hE_M8DaPvoym"}

### Exploring `jax.pure_callback`

`jax.pure_callback` is generally the callback function you should reach for when you want host-side execution of a pure function: i.e. a function that has no side-effects (such as printing values, reading data from disk, updating a global state, etc.).

The function you pass to `jax.pure_callback` need not actually be pure, but it will be assumed pure by JAX's transformations and higher-order functions, which means that it may be silently elided or called multiple times.

```{code-cell}
:id: 4lQDzXy6t_-k
:outputId: 279e4daf-0540-4eab-f535-d3bcbac74c44

import jax
import jax.numpy as jnp
import numpy as np

def f_host(x):
  # call a numpy (not jax.numpy) operation:
  return np.sin(x).astype(x.dtype)

def f(x):
  result_shape = jax.ShapeDtypeStruct(x.shape, x.dtype)
  return jax.pure_callback(f_host, result_shape, x)

x = jnp.arange(5.0)
f(x)
```

+++ {"id": "q7YCIr8qMrDs"}

Because `pure_callback` can be elided or duplicated, it is compatible out-of-the-box with transformations like `jit` and `vmap`, as well as higher-order primitives like `scan` and `while_loop`:"

```{code-cell}
:id: bgoZ0fxsuoWV
:outputId: 901443bd-5cb4-4923-ce53-6f832ac22ca9

jax.jit(f)(x)
```

```{code-cell}
:id: ajBRGWGfupu2
:outputId: b28e31ee-7457-4b92-872b-52d819f53ddf

jax.vmap(f)(x)
```

```{code-cell}
:id: xe7AOGexvC13
:outputId: 8fa77977-1f2b-41c5-cc5e-11993ee5aa3e

def body_fun(_, x):
  return _, f(x)
jax.lax.scan(body_fun, None, jnp.arange(5.0))[1]
```

+++ {"id": "tMzAVs2VNj5G"}

However, because there is no way for JAX to introspect the content of the callback, `pure_callback` has undefined autodiff semantics:

```{code-cell}
:id: 4QAF4VhUu5bb
:outputId: f8a06d02-47e9-4240-8077-d7be81e5a480

%xmode minimal
```

```{code-cell}
:id: qUpKPxlOurfY
:outputId: 11a665e8-40eb-4b0e-dc2e-a544a25fc57e
:tags: [raises-exception]

jax.grad(f)(x)
```

+++ {"id": "y9DAibV4Nwpo"}

For an example of using `pure_callback` with `jax.custom_jvp`, see *Example: `pure_callback` with `custom_jvp`* below.

+++ {"id": "LrvdAloMZbIe"}

By design functions passed to `pure_callback` are treated as if they have no side-effects: one consequence of this is that if the output of the function is not used, the compiler may eliminate the callback entirely:

```{code-cell}
:id: mmFc_zawZrBq
:outputId: a4df7568-3f64-4b2f-9a2c-7adb2e0815e0

def print_something():
  print('printing something')
  return np.int32(0)

@jax.jit
def f1():
  return jax.pure_callback(print_something, np.int32(0))
f1();
```

```{code-cell}
:id: tTwE4kpmaNei

@jax.jit
def f2():
  jax.pure_callback(print_something, np.int32(0))
  return 1.0
f2();
```

+++ {"id": "qfyGYbw4Z5U3"}

In `f1`, the output of the callback is used in the return value of the function, so the callback is executed and we see the printed output.
In `f2` on the other hand, the output of the callback is unused, and so the compiler notices this and eliminates the function call. These are the correct semantics for a callback to a function with no side-effects.

+++ {"id": "JHcJybr7OEBM"}

### Exploring `jax.experimental.io_callback`

In contrast to {func}`jax.pure_callback`, {func}`jax.experimental.io_callback` is explicitly meant to be used with impure functions, i.e. functions that do have side-effects.

As an example, here is a callback to a global host-side numpy random generator. This is an impure operation because a side-effect of generating a random number in numpy is that the random state is updated (Please note that this is meant as a toy example of `io_callback` and not necessarily a recommended way of generating random numbers in JAX!).

```{code-cell}
:id: eAg5xIhrOiWV
:outputId: e3cfec21-d843-4852-a49d-69a69fba9fc1

from jax.experimental import io_callback
from functools import partial

global_rng = np.random.default_rng(0)

def host_side_random_like(x):
  """Generate a random array like x using the global_rng state"""
  # We have two side-effects here:
  # - printing the shape and dtype
  # - calling global_rng, thus updating its state
  print(f'generating {x.dtype}{list(x.shape)}')
  return global_rng.uniform(size=x.shape).astype(x.dtype)

@jax.jit
def numpy_random_like(x):
  return io_callback(host_side_random_like, x, x)

x = jnp.zeros(5)
numpy_random_like(x)
```

+++ {"id": "mAIF31MlXj33"}

The `io_callback` is compatible with `vmap` by default:

```{code-cell}
:id: NY3o5dG6Vg6u
:outputId: a67a8a98-214e-40ca-ad98-a930cd3db85e

jax.vmap(numpy_random_like)(x)
```

+++ {"id": "XXvSeeOXXquZ"}

Note, however, that this may execute the mapped callbacks in any order. So, for example, if you ran this on a GPU, the order of the mapped outputs might differ from run to run.

If it is important that the order of callbacks be preserved, you can set `ordered=True`, in which case attempting to `vmap` will raise an error:

```{code-cell}
:id: 3aNmRsDrX3-2
:outputId: a8ff4b77-f4cb-442f-8cfb-ea7251c66274
:tags: [raises-exception]

@jax.jit
def numpy_random_like_ordered(x):
  return io_callback(host_side_random_like, x, x, ordered=True)

jax.vmap(numpy_random_like_ordered)(x)
```

+++ {"id": "fD2FTHlUYAZH"}

On the other hand, `scan` and `while_loop` work with `io_callback` regardless of whether ordering is enforced:

```{code-cell}
:id: lMVzZlIEWL7F
:outputId: f9741c18-a30d-4d46-b706-8102849286b5

def body_fun(_, x):
  return _, numpy_random_like_ordered(x)
jax.lax.scan(body_fun, None, jnp.arange(5.0))[1]
```

+++ {"id": "w_sf8mCbbo8K"}

Like `pure_callback`, `io_callback` fails under automatic differentiation if it is passed a differentiated variable:

```{code-cell}
:id: Cn6_RG4JcKZm
:outputId: 336ae5d2-e35b-4fe5-cbfb-14a7aef28c07
:tags: [raises-exception]

jax.grad(numpy_random_like)(x)
```

+++ {"id": "plvfn9lWcKu4"}

However, if the callback is not dependent on a differentiated variable, it will execute:

```{code-cell}
:id: wxgfDmDfb5bx
:outputId: d8c0285c-cd04-4b4d-d15a-1b07f778882d

@jax.jit
def f(x):
  io_callback(lambda: print('hello'), None)
  return x

jax.grad(f)(1.0);
```

+++ {"id": "STLI40EZcVIY"}

Unlike `pure_callback`, the compiler will not remove the callback execution in this case, even though the output of the callback is unused in the subsequent computation.

+++ {"id": "pkkM1ZmqclV-"}

### Exploring `debug.callback`

Both `pure_callback` and `io_callback` enforce some assumptions about the purity of the function they're calling, and limit in various ways what JAX transforms and compilation machinery may do. `debug.callback` essentially assumes *nothing* about the callback function, such that the action of the callback reflects exactly what JAX is doing during the course of a program. Further, `debug.callback` *cannot* return any value to the program.

```{code-cell}
:id: 74TdWyu9eqBa
:outputId: d8551dab-2e61-492e-9ac3-dc3db51b2c18

from jax import debug

def log_value(x):
  # This could be an actual logging call; we'll use
  # print() for demonstration
  print("log:", x)

@jax.jit
def f(x):
  debug.callback(log_value, x)
  return x

f(1.0);
```

+++ {"id": "P848STlsfzmW"}

The debug callback is compatible with `vmap`:

```{code-cell}
:id: 2sSNsPB-fGVI
:outputId: fff58575-d94c-48fb-b88a-c1c395595fd0

x = jnp.arange(5.0)
jax.vmap(f)(x);
```

+++ {"id": "VDMacqpXf3La"}

And is also compatible with `grad` and other autodiff transformations

```{code-cell}
:id: wkFRle-tfTDe
:outputId: 4e8a81d0-5012-4c51-d843-3fbdc498df31

jax.grad(f)(1.0);
```

+++ {"id": "w8t-SDZ3gRzE"}

This can make `debug.callback` more useful for general-purpose debugging than either `pure_callback` or `io_callback`.

+++ {"id": "dF7hoWGQUneJ"}

## Example: `pure_callback` with `custom_jvp`

One powerful way to take advantage of {func}`jax.pure_callback` is to combine it with {class}`jax.custom_jvp` (see [Custom derivative rules](https://jax.readthedocs.io/en/latest/notebooks/Custom_derivative_rules_for_Python_code.html) for more details on `custom_jvp`).
Suppose we want to create a JAX-compatible wrapper for a scipy or numpy function that is not yet available in the `jax.scipy` or `jax.numpy` wrappers.

Here, we'll consider creating a wrapper for the Bessel function of the first kind, implemented in `scipy.special.jv`.
We can start by defining a straightforward `pure_callback`:

```{code-cell}
:id: Ge4fNPZdVSJY

import jax
import jax.numpy as jnp
import scipy.special

def jv(v, z):
  v, z = jnp.asarray(v), jnp.asarray(z)

  # Require the order v to be integer type: this simplifies
  # the JVP rule below.
  assert jnp.issubdtype(v.dtype, jnp.integer)

  # Promote the input to inexact (float/complex).
  # Note that jnp.result_type() accounts for the enable_x64 flag.
  z = z.astype(jnp.result_type(float, z.dtype))

  # Wrap scipy function to return the expected dtype.
  _scipy_jv = lambda v, z: scipy.special.jv(v, z).astype(z.dtype)

  # Define the expected shape & dtype of output.
  result_shape_dtype = jax.ShapeDtypeStruct(
      shape=jnp.broadcast_shapes(v.shape, z.shape),
      dtype=z.dtype)

  # We use vectorize=True because scipy.special.jv handles broadcasted inputs.
  return jax.pure_callback(_scipy_jv, result_shape_dtype, v, z, vectorized=True)
```

+++ {"id": "vyjQj-0QVuoN"}

This lets us call into `scipy.special.jv` from transformed JAX code, including when transformed by `jit` and `vmap`:

```{code-cell}
:id: f4e46670f4e4

from functools import partial
j1 = partial(jv, 1)
z = jnp.arange(5.0)
```

```{code-cell}
:id: 6svImqFHWBwj
:outputId: bc8c778a-6c10-443b-9be2-c0f28e2ac1a9

print(j1(z))
```

+++ {"id": "d48eb4f2d48e"}

Here is the same result with `jit`:

```{code-cell}
:id: txvRqR9DWGdC
:outputId: d25f3476-23b1-48e4-dda1-3c06d32c3b87

print(jax.jit(j1)(z))
```

+++ {"id": "d861a472d861"}

And here is the same result again with `vmap`:

```{code-cell}
:id: BS-Ve5u_WU0C
:outputId: 08cecd1f-6953-4853-e9db-25a03eb5b000

print(jax.vmap(j1)(z))
```

+++ {"id": "SCH2ii_dWXP6"}

However, if we call `jax.grad`, we see an error because there is no autodiff rule defined for this function:

```{code-cell}
:id: q3qh_4DrWxdQ
:outputId: c46b0bfa-96f3-4629-b9af-a4d4f3ccb870
:tags: [raises-exception]

jax.grad(j1)(z)
```

+++ {"id": "PtYeJ_xUW09v"}

Let's define a custom gradient rule for this. Looking at the definition of the [Bessel Function of the First Kind](https://en.wikipedia.org/?title=Bessel_function_of_the_first_kind), we find that there is a relatively straightforward recurrence relationship for the derivative with respect to the argument `z`:

$$
d J_\nu(z) = \left\{
\begin{eqnarray}
-J_1(z),\ &\nu=0\\
[J_{\nu - 1}(z) - J_{\nu + 1}(z)]/2,\ &\nu\ne 0
\end{eqnarray}\right.
$$

The gradient with respect to $\nu$ is more complicated, but since we've restricted the `v` argument to integer types we don't need to worry about its gradient for the sake of this example.

We can use `jax.custom_jvp` to define this automatic differentiation rule for our callback function:

```{code-cell}
:id: BOVQnt05XvLs

jv = jax.custom_jvp(jv)

@jv.defjvp
def _jv_jvp(primals, tangents):
  v, z = primals
  _, z_dot = tangents  # Note: v_dot is always 0 because v is integer.
  jv_minus_1, jv_plus_1 = jv(v - 1, z), jv(v + 1, z)
  djv_dz = jnp.where(v == 0, -jv_plus_1, 0.5 * (jv_minus_1 - jv_plus_1))
  return jv(v, z), z_dot * djv_dz
```

+++ {"id": "W1SxcvQSX44c"}

Now computing the gradient of our function will work correctly:

```{code-cell}
:id: sCGceBs-X8nL
:outputId: 71c5589f-f996-44a0-f09a-ca8bb40c167a

j1 = partial(jv, 1)
print(jax.grad(j1)(2.0))
```

+++ {"id": "gWQ4phN5YB26"}

Further, since we've defined our gradient in terms of `jv` itself, JAX's architecture means that we get second-order and higher derivatives for free:

```{code-cell}
:id: QTe5mRAvYQBh
:outputId: d58ecff3-9419-422a-fd0e-14a7d9cf2cc3

jax.hessian(j1)(2.0)
```

+++ {"id": "QEXGxU4uYZii"}

Keep in mind that although this all works correctly with JAX, each call to our callback-based `jv` function will result in passing the input data from the device to the host, and passing the output of `scipy.special.jv` from the host back to the device.
When running on accelerators like GPU or TPU, this data movement and host synchronization can lead to significant overhead each time `jv` is called.
However, if you are running JAX on a single CPU (where the "host" and "device" are on the same hardware), JAX will generally do this data transfer in a fast, zero-copy fashion, making this pattern is a relatively straightforward way extend JAX's capabilities.

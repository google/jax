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

```{code-cell}
:tags: [remove-cell]

# This ensures that code cell tracebacks appearing below will be concise.
%xmode minimal
```

(external-callbacks)=
# External callbacks

<!--* freshness: { reviewed: '2024-05-16' } *-->

This tutorial outlines how you can use various callback functions, which allow JAX runtimes to execute Python code on the host. Examples of JAX callbacks are {func}`jax.pure_callback`, {func}`jax.experimental.io_callback` and {func}`jax.debug.callback`. You can use them even while running under JAX transformations, including {func}`~jax.jit`, {func}`~jax.vmap`, {func}`~jax.grad`.

## Why callbacks?

A callback routine is a way to perform **host-side** execution of code at runtime.
As a simple example, suppose you'd like to print the *value* of some variable during the course of a computation.
Using a simple Python {func}`print` statement, it looks like this:

```{code-cell}
import jax

@jax.jit
def f(x):
  y = x + 1
  print("intermediate value: {}".format(y))
  return y * 2

result = f(2)
```

What is printed is not the runtime value, but the trace-time abstract value (if you're not familiar with *tracing* in JAX, a good primer can be found in {ref}`key-concepts-tracing`.

To print the value at runtime, you need a callback, for example {func}`jax.debug.print` (you can learn more about debugging in {ref}`debugging`):

```{code-cell}
@jax.jit
def f(x):
  y = x + 1
  jax.debug.print("intermediate value: {}", y)
  return y * 2

result = f(2)
```

This works by passing the runtime value of `y` as a CPU {class}`jax.Array` back to the host process, where the host can print it.

(external-callbacks-flavors-of-callback)=
## Flavors of callback

In earlier versions of JAX, there was only one kind of callback available, implemented in {func}`jax.experimental.host_callback`. The `host_callback` routines had some deficiencies, and are now deprecated in favor of several callbacks designed for different situations:

- {func}`jax.pure_callback`: appropriate for pure functions: i.e. functions with no side effects.
- {func}`jax.experimental.io_callback`: appropriate for impure functions: e.g. functions which read or write data to disk.
- {func}`jax.debug.callback`: appropriate for functions that should reflect the execution behavior of the compiler.

(The {func}`jax.debug.print` function you used previously is a wrapper around {func}`jax.debug.callback`).

From the user perspective, these three flavors of callback are mainly distinguished by what transformations and compiler optimizations they allow.

|callback function | supports return value | `jit` | `vmap` | `grad` | `scan`/`while_loop` | guaranteed execution |
|-------------------------------------|----|----|----|----|----|----|
|{func}`jax.pure_callback`            | ✅ | ✅ | ✅ | ❌¹ | ✅ | ❌ |
|{func}`jax.experimental.io_callback` | ✅ | ✅ | ✅/❌² | ❌ | ✅³ | ✅ |
|{func}`jax.debug.callback`           | ❌ | ✅ | ✅ | ✅ | ✅ | ❌ |

¹ `jax.pure_callback` can be used with `custom_jvp` to make it compatible with autodiff

² `jax.experimental.io_callback` is compatible with `vmap` only if `ordered=False`.

³ Note that `vmap` of `scan`/`while_loop` of `io_callback` has complicated semantics, and its behavior may change in future releases.

### Exploring `pure_callback`

{func}`jax.pure_callback` is generally the callback function you should reach for when you want host-side execution of a pure function: i.e. a function that has no side-effects (such as printing values, reading data from disk, updating a global state, etc.).

The function you pass to {func}`jax.pure_callback` need not actually be pure, but it will be assumed pure by JAX's transformations and higher-order functions, which means that it may be silently elided or called multiple times.

```{code-cell}
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

Because `pure_callback` can be elided or duplicated, it is compatible out-of-the-box with transformations like `jit` and `vmap`, as well as higher-order primitives like `scan` and `while_loop`:"

```{code-cell}
jax.jit(f)(x)
```

```{code-cell}
jax.vmap(f)(x)
```

```{code-cell}
def body_fun(_, x):
  return _, f(x)
jax.lax.scan(body_fun, None, jnp.arange(5.0))[1]
```

However, because there is no way for JAX to introspect the content of the callback, `pure_callback` has undefined autodiff semantics:

```{code-cell}
:tags: [raises-exception]

jax.grad(f)(x)
```

For an example of using `pure_callback` with {func}`jax.custom_jvp`, see *Example: `pure_callback` with `custom_jvp`* below.



By design functions passed to `pure_callback` are treated as if they have no side-effects: one consequence of this is that if the output of the function is not used, the compiler may eliminate the callback entirely:

```{code-cell}
def print_something():
  print('printing something')
  return np.int32(0)

@jax.jit
def f1():
  return jax.pure_callback(print_something, np.int32(0))
f1();
```

```{code-cell}
@jax.jit
def f2():
  jax.pure_callback(print_something, np.int32(0))
  return 1.0
f2();
```

In `f1`, the output of the callback is used in the return value of the function, so the callback is executed and we see the printed output.
In `f2` on the other hand, the output of the callback is unused, and so the compiler notices this and eliminates the function call. These are the correct semantics for a callback to a function with no side-effects.

### Exploring `io_callback`

In contrast to {func}`jax.pure_callback`, {func}`jax.experimental.io_callback` is explicitly meant to be used with impure functions, i.e. functions that do have side-effects.

As an example, here is a callback to a global host-side numpy random generator. This is an impure operation because a side-effect of generating a random number in numpy is that the random state is updated (Please note that this is meant as a toy example of `io_callback` and not necessarily a recommended way of generating random numbers in JAX!).

```{code-cell}
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

The `io_callback` is compatible with `vmap` by default:

```{code-cell}
jax.vmap(numpy_random_like)(x)
```

Note, however, that this may execute the mapped callbacks in any order. So, for example, if you ran this on a GPU, the order of the mapped outputs might differ from run to run.

If it is important that the order of callbacks be preserved, you can set `ordered=True`, in which case attempting to `vmap` will raise an error:

```{code-cell}
:tags: [raises-exception]

@jax.jit
def numpy_random_like_ordered(x):
  return io_callback(host_side_random_like, x, x, ordered=True)

jax.vmap(numpy_random_like_ordered)(x)
```

On the other hand, `scan` and `while_loop` work with `io_callback` regardless of whether ordering is enforced:

```{code-cell}
def body_fun(_, x):
  return _, numpy_random_like_ordered(x)
jax.lax.scan(body_fun, None, jnp.arange(5.0))[1]
```

Like `pure_callback`, `io_callback` fails under automatic differentiation if it is passed a differentiated variable:

```{code-cell}
:tags: [raises-exception]

jax.grad(numpy_random_like)(x)
```

However, if the callback is not dependent on a differentiated variable, it will execute:

```{code-cell}
@jax.jit
def f(x):
  io_callback(lambda: print('hello'), None)
  return x

jax.grad(f)(1.0);
```

Unlike `pure_callback`, the compiler will not remove the callback execution in this case, even though the output of the callback is unused in the subsequent computation.


(external-callbacks-exploring-debug-callback)=
### Exploring `debug.callback`

Both `pure_callback` and `io_callback` enforce some assumptions about the purity of the function they're calling, and limit in various ways what JAX transforms and compilation machinery may do. `debug.callback` essentially assumes *nothing* about the callback function, such that the action of the callback reflects exactly what JAX is doing during the course of a program. Further, `debug.callback` *cannot* return any value to the program.

```{code-cell}
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

The debug callback is compatible with `vmap`:

```{code-cell}
x = jnp.arange(5.0)
jax.vmap(f)(x);
```

And is also compatible with `grad` and other autodiff transformations

```{code-cell}
jax.grad(f)(1.0);
```

This can make `debug.callback` more useful for general-purpose debugging than either `pure_callback` or `io_callback`.

## Example: `pure_callback` with `custom_jvp`

One powerful way to take advantage of {func}`jax.pure_callback` is to combine it with {class}`jax.custom_jvp`. (Refer to {ref}`advanced-autodiff-custom-derivative-rules` for more details on {func}`jax.custom_jvp`).

Suppose you want to create a JAX-compatible wrapper for a scipy or numpy function that is not yet available in the {mod}`jax.scipy` or {mod}`jax.numpy` wrappers.

Here, we'll consider creating a wrapper for the Bessel function of the first kind, available in {mod}`scipy.special.jv`.
You can start by defining a straightforward {func}`~jax.pure_callback`:

```{code-cell}
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

  # You use vectorize=True because scipy.special.jv handles broadcasted inputs.
  return jax.pure_callback(_scipy_jv, result_shape_dtype, v, z, vectorized=True)
```

This lets us call into {func}`scipy.special.jv` from transformed JAX code, including when transformed by {func}`~jax.jit` and {func}`~jax.vmap`:

```{code-cell}
from functools import partial
j1 = partial(jv, 1)
z = jnp.arange(5.0)
```

```{code-cell}
print(j1(z))
```

Here is the same result with {func}`~jax.jit`:

```{code-cell}
print(jax.jit(j1)(z))
```

And here is the same result again with {func}`~jax.vmap`:

```{code-cell}
print(jax.vmap(j1)(z))
```

However, if you call {func}`~jax.grad`, you will get an error because there is no autodiff rule defined for this function:

```{code-cell}
:tags: [raises-exception]

jax.grad(j1)(z)
```

Let's define a custom gradient rule for this. Looking at the definition of the [Bessel Function of the First Kind](https://en.wikipedia.org/?title=Bessel_function_of_the_first_kind), you find that there is a relatively straightforward recurrence relationship for the derivative with respect to the argument `z`:

$$
d J_\nu(z) = \left\{
\begin{eqnarray}
-J_1(z),\ &\nu=0\\
[J_{\nu - 1}(z) - J_{\nu + 1}(z)]/2,\ &\nu\ne 0
\end{eqnarray}\right.
$$

The gradient with respect to $\nu$ is more complicated, but since we've restricted the `v` argument to integer types you don't need to worry about its gradient for the sake of this example.

You can use {func}`jax.custom_jvp` to define this automatic differentiation rule for your callback function:

```{code-cell}
jv = jax.custom_jvp(jv)

@jv.defjvp
def _jv_jvp(primals, tangents):
  v, z = primals
  _, z_dot = tangents  # Note: v_dot is always 0 because v is integer.
  jv_minus_1, jv_plus_1 = jv(v - 1, z), jv(v + 1, z)
  djv_dz = jnp.where(v == 0, -jv_plus_1, 0.5 * (jv_minus_1 - jv_plus_1))
  return jv(v, z), z_dot * djv_dz
```

Now computing the gradient of your function will work correctly:

```{code-cell}
j1 = partial(jv, 1)
print(jax.grad(j1)(2.0))
```

Further, since we've defined your gradient in terms of `jv` itself, JAX's architecture means that you get second-order and higher derivatives for free:

```{code-cell}
jax.hessian(j1)(2.0)
```

Keep in mind that although this all works correctly with JAX, each call to your callback-based `jv` function will result in passing the input data from the device to the host, and passing the output of {func}`scipy.special.jv` from the host back to the device.

When running on accelerators like GPU or TPU, this data movement and host synchronization can lead to significant overhead each time `jv` is called.

However, if you are running JAX on a single CPU (where the "host" and "device" are on the same hardware), JAX will generally do this data transfer in a fast, zero-copy fashion, making this pattern a relatively straightforward way to extend JAX's capabilities.

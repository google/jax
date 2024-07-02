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

(key-concepts)=
# Key Concepts

<!--* freshness: { reviewed: '2024-05-03' } *-->

This section briefly introduces some key concepts of the JAX package.

(key-concepts-jax-arrays)=
## JAX arrays ({class}`jax.Array`)

The default array implementation in JAX is {class}`jax.Array`. In many ways it is similar to
the {class}`numpy.ndarray` type that you may be familar with from the NumPy package, but it
has some important differences.

### Array creation

We typically don't call the {class}`jax.Array` constructor directly, but rather create arrays via JAX API functions.
For example, {mod}`jax.numpy` provides familar NumPy-style array construction functionality
such as {func}`jax.numpy.zeros`, {func}`jax.numpy.linspace`, {func}`jax.numpy.arange`, etc.

```{code-cell}
import jax
import jax.numpy as jnp

x = jnp.arange(5)
isinstance(x, jax.Array)
```

If you use Python type annotations in your code, {class}`jax.Array` is the appropriate
annotation for jax array objects (see {mod}`jax.typing` for more discussion).

### Array devices and sharding

JAX Array objects have a `devices` method that lets you inspect where the contents of the array are stored. In the simplest cases, this will be a single CPU device:

```{code-cell}
x.devices()
```

In general, an array may be *sharded* across multiple devices, in a manner that can be inspected via the `sharding` attribute:

```{code-cell}
x.sharding
```

Here the array is on a single device, but in general a JAX array can be
sharded across multiple devices, or even multiple hosts.
To read more about sharded arrays and parallel computation, refer to {ref}`sharded-computation`

(key-concepts-transformations)=
## Transformations
Along with functions to operate on arrays, JAX includes a number of
{term}`transformations <transformation>` which operate on JAX functions. These include

- {func}`jax.jit`: Just-in-time (JIT) compilation; see {ref}`jit-compilation`
- {func}`jax.vmap`: Vectorizing transform; see {ref}`automatic-vectorization`
- {func}`jax.grad`: Gradient transform; see {ref}`automatic-differentiation`

as well as several others. Transformations accept a function as an argument, and return a
new transformed function. For example, here's how you might JIT-compile a simple SELU function:

```{code-cell}
def selu(x, alpha=1.67, lambda_=1.05):
  return lambda_ * jnp.where(x > 0, x, alpha * jnp.exp(x) - alpha)

selu_jit = jax.jit(selu)
print(selu_jit(1.0))
```

Often you'll see transformations applied using Python's decorator syntax for convenience:

```{code-cell}
@jax.jit
def selu(x, alpha=1.67, lambda_=1.05):
  return lambda_ * jnp.where(x > 0, x, alpha * jnp.exp(x) - alpha)
```

Transformations like {func}`~jax.jit`, {func}`~jax.vmap`, {func}`~jax.grad`, and others are
key to using JAX effectively, and we'll cover them in detail in later sections.

(key-concepts-tracing)=
## Tracing

The magic behind transformations is the notion of a {term}`Tracer`.
Tracers are abstract stand-ins for array objects, and are passed to JAX functions in order
to extract the sequence of operations that the function encodes.

You can see this by printing any array value within transformed JAX code; for example:

```{code-cell}
@jax.jit
def f(x):
  print(x)
  return x + 1

x = jnp.arange(5)
result = f(x)
```

The value printed is not the array `x`, but a {class}`~jax.core.Tracer` instance that
represents essential attributes of `x`, such as its `shape` and `dtype`. By executing
the function with traced values, JAX can determine the sequence of operations encoded
by the function before those operations are actually executed: transformations like
{func}`~jax.jit`, {func}`~jax.vmap`, and {func}`~jax.grad` can then map this sequence
of input operations to a transformed sequence of operations.

(key-concepts-jaxprs)=
## Jaxprs

JAX has its own intermediate representation for sequences of operations, known as a {term}`jaxpr`.
A jaxpr (short for *JAX exPRession*) is a simple representation of a functional program, comprising a sequence of {term}`primitive` operations.

For example, consider the `selu` function we defined above:

```{code-cell}
def selu(x, alpha=1.67, lambda_=1.05):
  return lambda_ * jnp.where(x > 0, x, alpha * jnp.exp(x) - alpha)
```

We can use the {func}`jax.make_jaxpr` utility to convert this function into a jaxpr
given a particular input:

```{code-cell}
x = jnp.arange(5.0)
jax.make_jaxpr(selu)(x)
```

Comparing this to the Python function definition, we see that it encodes the precise
sequence of operations that the function represents. We'll go into more depth about
jaxprs later in {ref}`jax-internals-jaxpr`.

(key-concepts-pytrees)=
## Pytrees

JAX functions and transformations fundamentally operate on arrays, but in practice it is
convenient to write code that work with collections of arrays: for example, a neural
network might organize its parameters in a dictionary of arrays with meaningful keys.
Rather than handle such structures on a case-by-case basis, JAX relies on the {term}`pytree`
abstraction to treat such collections in a uniform matter.

Here are some examples of objects that can be treated as pytrees:

```{code-cell}
# (nested) list of parameters
params = [1, 2, (jnp.arange(3), jnp.ones(2))]

print(jax.tree.structure(params))
print(jax.tree.leaves(params))
```

```{code-cell}
# Dictionary of parameters
params = {'n': 5, 'W': jnp.ones((2, 2)), 'b': jnp.zeros(2)}

print(jax.tree.structure(params))
print(jax.tree.leaves(params))
```

```{code-cell}
# Named tuple of parameters
from typing import NamedTuple

class Params(NamedTuple):
  a: int
  b: float

params = Params(1, 5.0)
print(jax.tree.structure(params))
print(jax.tree.leaves(params))
```

JAX has a number of general-purpose utilities for working with PyTrees; for example
the functions {func}`jax.tree.map` can be used to map a function to every leaf in a
tree, and {func}`jax.tree.reduce` can be used to apply a reduction across the leaves
in a tree.

You can learn more in the {ref}`working-with-pytrees` tutorial.

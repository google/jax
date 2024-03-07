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
  language: python
  name: python3
---

+++ {"id": "M-hPMKlwXjMr"}

# Writing custom Jaxpr interpreters in JAX

<!--* freshness: { reviewed: '2024-04-08' } *-->

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/google/jax/blob/main/docs/notebooks/Writing_custom_interpreters_in_Jax.ipynb) [![Open in Kaggle](https://kaggle.com/static/images/open-in-kaggle.svg)](https://kaggle.com/kernels/welcome?src=https://github.com/google/jax/blob/main/docs/notebooks/Writing_custom_interpreters_in_Jax.ipynb)

+++ {"id": "r-3vMiKRYXPJ"}

JAX offers several composable function transformations (`jit`, `grad`, `vmap`,
etc.) that enable writing concise, accelerated code. 

Here we show how to add your own function transformations to the system, by writing a custom Jaxpr interpreter. And we'll get composability with all the other transformations for free.

**This example uses internal JAX APIs, which may break at any time. Anything not in [the API Documentation](https://jax.readthedocs.io/en/latest/jax.html) should be assumed internal.**

```{code-cell} ipython3
:id: s27RDKvKXFL8

import numpy as np
import jax
import jax.numpy as jnp
from jax import jit, grad, vmap
from jax import random
```

+++ {"id": "jb_8mEsJboVM"}

## What is JAX doing?

+++ {"id": "KxR2WK0Ubs0R"}

JAX provides a NumPy-like API for numerical computing which can be used as is, but JAX's true power comes from composable function transformations. Take the `jit` function transformation, which takes in a function and returns a semantically identical function but is lazily compiled by XLA for accelerators.

```{code-cell} ipython3
:id: HmlMcICOcSXR

x = random.normal(random.key(0), (5000, 5000))
def f(w, b, x):
  return jnp.tanh(jnp.dot(x, w) + b)
fast_f = jit(f)
```

+++ {"id": "gA8V51wZdsjh"}

When we call `fast_f`, what happens? JAX traces the function and constructs an XLA computation graph. The graph is then JIT-compiled and executed. Other transformations work similarly in that they first trace the function and handle the output trace in some way. To learn more about Jax's tracing machinery, you can refer to the ["How it works"](https://github.com/google/jax#how-it-works) section in the README.

+++ {"id": "2Th1vYLVaFBz"}

## Jaxpr tracer

A tracer of special importance in Jax is the Jaxpr tracer, which records ops into a Jaxpr (Jax expression). A Jaxpr is a data structure that can be evaluated like a mini functional programming language and 
thus Jaxprs are a useful intermediate representation
for function transformation.

+++ {"id": "pH7s63lpaHJO"}

To get a first look at Jaxprs, consider the `make_jaxpr` transformation. `make_jaxpr` is essentially a "pretty-printing" transformation:
it transforms a function into one that, given example arguments, produces a Jaxpr representation of its computation.
`make_jaxpr` is useful for debugging and introspection.
Let's use it to look at how some example Jaxprs are structured.

```{code-cell} ipython3
:id: RSxEiWi-EeYW

def examine_jaxpr(closed_jaxpr):
  jaxpr = closed_jaxpr.jaxpr
  print("invars:", jaxpr.invars)
  print("outvars:", jaxpr.outvars)
  print("constvars:", jaxpr.constvars)
  for eqn in jaxpr.eqns:
    print("equation:", eqn.invars, eqn.primitive, eqn.outvars, eqn.params)
  print()
  print("jaxpr:", jaxpr)

def foo(x):
  return x + 1
print("foo")
print("=====")
examine_jaxpr(jax.make_jaxpr(foo)(5))

print()

def bar(w, b, x):
  return jnp.dot(w, x) + b + jnp.ones(5), x
print("bar")
print("=====")
examine_jaxpr(jax.make_jaxpr(bar)(jnp.ones((5, 10)), jnp.ones(5), jnp.ones(10)))
```

+++ {"id": "k-HxK9iagnH6"}

* `jaxpr.invars` - the `invars` of a Jaxpr are a list of the input variables to Jaxpr, analogous to arguments in Python functions.
* `jaxpr.outvars` - the `outvars` of a Jaxpr are the variables that are returned by the Jaxpr. Every Jaxpr has multiple outputs.
* `jaxpr.constvars` - the `constvars` are a list of variables that are also inputs to the Jaxpr, but correspond to constants from the trace (we'll go over these in more detail later).
* `jaxpr.eqns` - a list of equations, which are essentially let-bindings. Each equation is a list of input variables, a list of output variables, and a *primitive*, which is used to evaluate inputs to produce outputs. Each equation also has a `params`, a dictionary of parameters.

Altogether, a Jaxpr encapsulates a simple program that can be evaluated with inputs to produce an output. We'll go over how exactly to do this later. The important thing to note now is that a Jaxpr is a data structure that can be manipulated and evaluated in whatever way we want.

+++ {"id": "NwY7TurYn6sr"}

### Why are Jaxprs useful?

+++ {"id": "UEy6RorCgdYt"}

Jaxprs are simple program representations that are easy to transform. And because Jax lets us stage out Jaxprs from Python functions, it gives us a way to transform numerical programs written in Python.

+++ {"id": "qizTKpbno_ua"}

## Your first interpreter: `invert`

+++ {"id": "OIto-KX4pD7j"}

Let's try to implement a simple function "inverter", which takes in the output of the original function and returns the inputs that produced those outputs. For now, let's focus on simple, unary functions which are composed of other invertible unary functions.

Goal:
```python
def f(x):
  return jnp.exp(jnp.tanh(x))
f_inv = inverse(f)
assert jnp.allclose(f_inv(f(1.0)), 1.0)
```

The way we'll implement this is by (1) tracing `f` into a Jaxpr, then (2) interpreting the Jaxpr *backwards*. While interpreting the Jaxpr backwards, for each equation we'll look up the primitive's inverse in a table and apply it.

### 1. Tracing a function

Let's use `make_jaxpr` to trace a function into a Jaxpr.

```{code-cell} ipython3
:id: BHkg_3P1pXJj

# Importing Jax functions useful for tracing/interpreting.
import numpy as np
from functools import wraps

from jax import core
from jax import lax
from jax._src.util import safe_map
```

+++ {"id": "CpTml2PTrzZ4"}

`jax.make_jaxpr` returns a *closed* Jaxpr, which is a Jaxpr that has been bundled with
the constants (`literals`) from the trace.

```{code-cell} ipython3
:id: Tc1REN5aq_fH

def f(x):
  return jnp.exp(jnp.tanh(x))

closed_jaxpr = jax.make_jaxpr(f)(jnp.ones(5))
print(closed_jaxpr.jaxpr)
print(closed_jaxpr.literals)
```

+++ {"id": "WmZ3BcmZsbfR"}

### 2. Evaluating a Jaxpr


Before we write a custom Jaxpr interpreter, let's first implement the "default" interpreter, `eval_jaxpr`, which evaluates the Jaxpr as-is, computing the same values that the original, un-transformed Python function would. 

To do this, we first create an environment to store the values for each of the variables, and update the environment with each equation we evaluate in the Jaxpr.

```{code-cell} ipython3
:id: ACMxjIHStHwD

def eval_jaxpr(jaxpr, consts, *args):
  # Mapping from variable -> value
  env = {}
  
  def read(var):
    # Literals are values baked into the Jaxpr
    if type(var) is core.Literal:
      return var.val
    return env[var]

  def write(var, val):
    env[var] = val

  # Bind args and consts to environment
  safe_map(write, jaxpr.invars, args)
  safe_map(write, jaxpr.constvars, consts)

  # Loop through equations and evaluate primitives using `bind`
  for eqn in jaxpr.eqns:
    # Read inputs to equation from environment
    invals = safe_map(read, eqn.invars)  
    # `bind` is how a primitive is called
    outvals = eqn.primitive.bind(*invals, **eqn.params)
    # Primitives may return multiple outputs or not
    if not eqn.primitive.multiple_results: 
      outvals = [outvals]
    # Write the results of the primitive into the environment
    safe_map(write, eqn.outvars, outvals) 
  # Read the final result of the Jaxpr from the environment
  return safe_map(read, jaxpr.outvars) 
```

```{code-cell} ipython3
:id: mGHPc3NruCFV

closed_jaxpr = jax.make_jaxpr(f)(jnp.ones(5))
eval_jaxpr(closed_jaxpr.jaxpr, closed_jaxpr.literals, jnp.ones(5))
```

+++ {"id": "XhZhzbVBvAiT"}

Notice that `eval_jaxpr` will always return a flat list even if the original function does not.

Furthermore, this interpreter does not handle higher-order primitives (like `jit` and `pmap`), which we will not cover in this guide. You can refer to `core.eval_jaxpr` ([link](https://github.com/google/jax/blob/main/jax/core.py)) to see the edge cases that this interpreter does not cover.

+++ {"id": "0vb2ZoGrCMM4"}

### Custom `inverse` Jaxpr interpreter

An `inverse` interpreter doesn't look too different from `eval_jaxpr`. We'll first set up the registry which will map primitives to their inverses. We'll then write a custom interpreter that looks up primitives in the registry.

It turns out that this interpreter will also look similar to the "transpose" interpreter used in reverse-mode autodifferentiation [found here](https://github.com/google/jax/blob/main/jax/interpreters/ad.py#L164-L234).

```{code-cell} ipython3
:id: gSMIT2z1vUpO

inverse_registry = {}
```

+++ {"id": "JgrpMgDyCrC7"}

We'll now register inverses for some of the primitives. By convention, primitives in Jax end in `_p` and a lot of the popular ones live in `lax`.

```{code-cell} ipython3
:id: fUerorGkCqhw

inverse_registry[lax.exp_p] = jnp.log
inverse_registry[lax.tanh_p] = jnp.arctanh
```

+++ {"id": "mDtH_lYDC5WK"}

`inverse` will first trace the function, then custom-interpret the Jaxpr. Let's set up a simple skeleton.

```{code-cell} ipython3
:id: jGNfV6JJC1B3

def inverse(fun):
  @wraps(fun)
  def wrapped(*args, **kwargs):
    # Since we assume unary functions, we won't worry about flattening and
    # unflattening arguments.
    closed_jaxpr = jax.make_jaxpr(fun)(*args, **kwargs)
    out = inverse_jaxpr(closed_jaxpr.jaxpr, closed_jaxpr.literals, *args)
    return out[0]
  return wrapped
```

+++ {"id": "g6v6wV7SDM7g"}

Now we just need to define `inverse_jaxpr`, which will walk through the Jaxpr backward and invert primitives when it can.

```{code-cell} ipython3
:id: uUAd-L-BDKT5

def inverse_jaxpr(jaxpr, consts, *args):
  env = {}
  
  def read(var):
    if type(var) is core.Literal:
      return var.val
    return env[var]

  def write(var, val):
    env[var] = val
  # Args now correspond to Jaxpr outvars
  safe_map(write, jaxpr.outvars, args)
  safe_map(write, jaxpr.constvars, consts)

  # Looping backward
  for eqn in jaxpr.eqns[::-1]:
    #  outvars are now invars 
    invals = safe_map(read, eqn.outvars)
    if eqn.primitive not in inverse_registry:
      raise NotImplementedError(
          f"{eqn.primitive} does not have registered inverse.")
    # Assuming a unary function 
    outval = inverse_registry[eqn.primitive](*invals)
    safe_map(write, eqn.invars, [outval])
  return safe_map(read, jaxpr.invars)
```

+++ {"id": "M8i3wGbVERhA"}

That's it!

```{code-cell} ipython3
:id: cjEKWso-D5Bu

def f(x):
  return jnp.exp(jnp.tanh(x))

f_inv = inverse(f)
assert jnp.allclose(f_inv(f(1.0)), 1.0)
```

+++ {"id": "Ny7Oo4WLHdXt"}

Importantly, you can trace through a Jaxpr interpreter.

```{code-cell} ipython3
:id: j6ov_rveHmTb

jax.make_jaxpr(inverse(f))(f(1.))
```

+++ {"id": "yfWVBsKwH0j6"}

That's all it takes to add a new transformation to a system, and you get composition with all the others for free! For example, we can use `jit`, `vmap`, and `grad` with `inverse`!

```{code-cell} ipython3
:id: 3tjNk21CH4yZ

jit(vmap(grad(inverse(f))))((jnp.arange(5) + 1.) / 5.)
```

+++ {"id": "APtG-u_6E4tK"}

## Exercises for the reader

* Handle primitives with multiple arguments where inputs are partially known, for example `lax.add_p`, `lax.mul_p`.
* Handle `xla_call` and `xla_pmap` primitives, which will not work with both `eval_jaxpr` and `inverse_jaxpr` as written.

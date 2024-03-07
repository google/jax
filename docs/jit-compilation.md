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

(jit-compilation)=
# Just-in-time compilation

<!--* freshness: { reviewed: '2024-05-03' } *-->

In this section, we will further explore how JAX works, and how we can make it performant.
We will discuss the {func}`jax.jit` transformation, which will perform *Just In Time* (JIT)
compilation of a JAX Python function so it can be executed efficiently in XLA.

## How JAX transformations work

In the previous section, we discussed that JAX allows us to transform Python functions.
JAX accomplishes this by reducing each function into a sequence of {term}`primitive` operations, each
representing one fundamental unit of computation.

One way to see the sequence of primitives behind a function is using {func}`jax.make_jaxpr`:

```{code-cell}
import jax
import jax.numpy as jnp

global_list = []

def log2(x):
  global_list.append(x)
  ln_x = jnp.log(x)
  ln_2 = jnp.log(2.0)
  return ln_x / ln_2

print(jax.make_jaxpr(log2)(3.0))
```

The {ref}`understanding-jaxprs` section of the documentation provides more information on the meaning of the above output.

Importantly, notice that the jaxpr does not capture the side-effect present in the function: there is nothing in it corresponding to `global_list.append(x)`.
This is a feature, not a bug: JAX transformations are designed to understand side-effect-free (a.k.a. functionally pure) code.
If *pure function* and *side-effect* are unfamiliar terms, this is explained in a little more detail in [🔪 JAX - The Sharp Bits 🔪: Pure Functions](https://jax.readthedocs.io/en/latest/notebooks/Common_Gotchas_in_JAX.html#pure-functions).

Impure functions are dangerous because under JAX transformations they are likely not to behave as intended; they might fail silently, or produce surprising downstream errors like leaked Tracers.
Moreover, JAX often can't detect when side effects are present.
(If you want debug printing, use {func}`jax.debug.print`. To express general side-effects at the cost of performance, see {func}`jax.experimental.io_callback`.
To check for tracer leaks at the cost of performance, use with {func}`jax.check_tracer_leaks`).

When tracing, JAX wraps each argument by a *tracer* object. These tracers then record all JAX operations performed on them during the function call (which happens in regular Python). Then, JAX uses the tracer records to reconstruct the entire function. The output of that reconstruction is the jaxpr. Since the tracers do not record the Python side-effects, they do not appear in the jaxpr. However, the side-effects still happen during the trace itself.

Note: the Python `print()` function is not pure: the text output is a side-effect of the function. Therefore, any `print()` calls will only happen during tracing, and will not appear in the jaxpr:

```{code-cell}
def log2_with_print(x):
  print("printed x:", x)
  ln_x = jnp.log(x)
  ln_2 = jnp.log(2.0)
  return ln_x / ln_2

print(jax.make_jaxpr(log2_with_print)(3.))
```

See how the printed `x` is a `Traced` object? That's the JAX internals at work.

The fact that the Python code runs at least once is strictly an implementation detail, and so shouldn't be relied upon. However, it's useful to understand as you can use it when debugging to print out intermediate values of a computation.

A key thing to understand is that a jaxpr captures the function as executed on the parameters given to it.
For example, if we have a Python conditional, the jaxpr will only know about the branch we take:

```{code-cell}
def log2_if_rank_2(x):
  if x.ndim == 2:
    ln_x = jnp.log(x)
    ln_2 = jnp.log(2.0)
    return ln_x / ln_2
  else:
    return x

print(jax.make_jaxpr(log2_if_rank_2)(jax.numpy.array([1, 2, 3])))
```

## JIT compiling a function

As explained before, JAX enables operations to execute on CPU/GPU/TPU using the same code.
Let's look at an example of computing a *Scaled Exponential Linear Unit*
([SELU](https://proceedings.neurips.cc/paper/6698-self-normalizing-neural-networks.pdf)), an
operation commonly used in deep learning:

```{code-cell}
import jax
import jax.numpy as jnp

def selu(x, alpha=1.67, lambda_=1.05):
  return lambda_ * jnp.where(x > 0, x, alpha * jnp.exp(x) - alpha)

x = jnp.arange(1000000)
%timeit selu(x).block_until_ready()
```

The code above is sending one operation at a time to the accelerator. This limits the ability of the XLA compiler to optimize our functions.

Naturally, what we want to do is give the XLA compiler as much code as possible, so it can fully optimize it. For this purpose, JAX provides the {func}`jax.jit` transformation, which will JIT compile a JAX-compatible function. The example below shows how to use JIT to speed up the previous function.

```{code-cell}
selu_jit = jax.jit(selu)

# Pre-compile the function before timing...
selu_jit(x).block_until_ready()

%timeit selu_jit(x).block_until_ready()
```

Here's what just happened:

1) We defined `selu_jit` as the compiled version of `selu`.

2) We called `selu_jit` once on `x`. This is where JAX does its tracing -- it needs to have some inputs to wrap in tracers, after all. The jaxpr is then compiled using XLA into very efficient code optimized for your GPU or TPU. Finally, the compiled code is executed to satisfy the call. Subsequent calls to `selu_jit` will use the compiled code directly, skipping the python implementation entirely.
(If we didn't include the warm-up call separately, everything would still work, but then the compilation time would be included in the benchmark. It would still be faster, because we run many loops in the benchmark, but it wouldn't be a fair comparison.)

3) We timed the execution speed of the compiled version. (Note the use of {func}`~jax.block_until_ready`, which is required due to JAX's {ref}`async-dispatch`).

## Why can't we just JIT everything?

After going through the example above, you might be wondering whether we should simply apply {func}`jax.jit` to every function. To understand why this is not the case, and when we should/shouldn't apply `jit`, let's first check some cases where JIT doesn't work.

```{code-cell}
:tags: [raises-exception]

# Condition on value of x.

def f(x):
  if x > 0:
    return x
  else:
    return 2 * x

jax.jit(f)(10)  # Raises an error
```

```{code-cell}
:tags: [raises-exception]

# While loop conditioned on x and n.

def g(x, n):
  i = 0
  while i < n:
    i += 1
  return x + i

jax.jit(g)(10, 20)  # Raises an error
```

The problem in both cases is that we tried to condition the trace-time flow of the program using runtime values.
Traced values within JIT, like `x` and `n` here, can only affect control flow via their static attributes: such as
`shape` or `dtype`, and not via their values.
For more detail on the interaction between Python control flow and JAX, see [🔪 JAX - The Sharp Bits 🔪: Control Flow](https://jax.readthedocs.io/en/latest/notebooks/Common_Gotchas_in_JAX.html#control-flow).

One way to deal with this problem is to rewrite the code to avoid conditionals on value. Another is to use special {ref}`lax-control-flow` like {func}`jax.lax.cond`. However, sometimes that is not possible or practical.
In that case, you can consider JIT-compiling only part of the function.
For example, if the most computationally expensive part of the function is inside the loop, we can JIT-compile just that inner part (though make sure to check the next section on caching to avoid shooting yourself in the foot):

```{code-cell}
# While loop conditioned on x and n with a jitted body.

@jax.jit
def loop_body(prev_i):
  return prev_i + 1

def g_inner_jitted(x, n):
  i = 0
  while i < n:
    i = loop_body(i)
  return x + i

g_inner_jitted(10, 20)
```

## Marking arguments as static

If we really need to JIT-compile a function that has a condition on the value of an input, we can tell JAX to help itself to a less abstract tracer for a particular input by specifying `static_argnums` or `static_argnames`.
The cost of this is that the resulting jaxpr and compiled artifact depends on the particular value passed, and so JAX will have to re-compile the function for every new value of the specified static input.
It is only a good strategy if the function is guaranteed to see a limited set of static values.

```{code-cell}
f_jit_correct = jax.jit(f, static_argnums=0)
print(f_jit_correct(10))
```

```{code-cell}
g_jit_correct = jax.jit(g, static_argnames=['n'])
print(g_jit_correct(10, 20))
```

To specify such arguments when using `jit` as a decorator, a common pattern is to use python's {func}`functools.partial`:

```{code-cell}
from functools import partial

@partial(jax.jit, static_argnames=['n'])
def g_jit_decorated(x, n):
  i = 0
  while i < n:
    i += 1
  return x + i

print(g_jit_decorated(10, 20))
```

## JIT and caching

With the compilation overhead of the first JIT call, understanding how and when {func}`jax.jit` caches previous compilations is key to using it effectively.

Suppose we define `f = jax.jit(g)`. When we first invoke `f`, it will get compiled, and the resulting XLA code will get cached. Subsequent calls of `f` will reuse the cached code.
This is how `jax.jit` makes up for the up-front cost of compilation.

If we specify `static_argnums`, then the cached code will be used only for the same values of arguments labelled as static. If any of them change, recompilation occurs.
If there are many values, then your program might spend more time compiling than it would have executing ops one-by-one.

Avoid calling {func}`jax.jit` on temporary functions defined inside loops or other Python scopes.
For most cases, JAX will be able to use the compiled, cached function in subsequent calls to {func}`jax.jit`.
However, because the cache relies on the hash of the function, it becomes problematic when equivalent functions are redefined.
This will cause unnecessary compilation each time in the loop:

```{code-cell}
from functools import partial

def unjitted_loop_body(prev_i):
  return prev_i + 1

def g_inner_jitted_partial(x, n):
  i = 0
  while i < n:
    # Don't do this! each time the partial returns
    # a function with different hash
    i = jax.jit(partial(unjitted_loop_body))(i)
  return x + i

def g_inner_jitted_lambda(x, n):
  i = 0
  while i < n:
    # Don't do this!, lambda will also return
    # a function with a different hash
    i = jax.jit(lambda x: unjitted_loop_body(x))(i)
  return x + i

def g_inner_jitted_normal(x, n):
  i = 0
  while i < n:
    # this is OK, since JAX can find the
    # cached, compiled function
    i = jax.jit(unjitted_loop_body)(i)
  return x + i

print("jit called in a loop with partials:")
%timeit g_inner_jitted_partial(10, 20).block_until_ready()

print("jit called in a loop with lambdas:")
%timeit g_inner_jitted_lambda(10, 20).block_until_ready()

print("jit called in a loop with caching:")
%timeit g_inner_jitted_normal(10, 20).block_until_ready()
```

# Sequencing side-effects in JAX
_sharadmv@_
_May 9 2022_

## Overview

When we write JAX code, we can usually pretend we're writing single-threaded, eagerly-executed Python
even though underneath the hood, JAX and its runtime may execute it asynchronously in the background.
As long as we write pure (side-effect-free) code, these performance optimizations are usually invisible to us and don't interfere with our single-threaded mental model.
Asynchronous execution is great -- we get performant, parallel code without
having to think about it at all!

However, in the presence of side-effects, the illusion begins to break down and
the cracks in our mental model start to show. Specifically, these differences
show up when we think about the *order* in which side-effects happen.

In this design note, we explore the interaction between JAX's execution model,
and the ordering of side-effects. We also provide a way of enforcing a
"single-threaded" ordering of effects.

## Background

When we write the following Python code
```python
def f():
  print("hello")
  return 2
def g():
  print("world")
  return 3
f()
g()
```
we expect `"hello"` to be printed before `"world"`. This might seem obvious
but consider the following JAX code:
```python
@partial(jax.jit, device=<device 0>)
def f():
  return 2

@partial(jax.jit, device=<device 1>)
def g():
  return 3
f()
g()
```
In many cases, JAX will execute `f` and `g` *in parallel*, dispatching
the computations onto different threads -- `g` might actually be executed
before `f`. Parallel execution is a nice performance optimization, especially if copying
to and from a device is expensive (see the [asynchronous dispatch note](https://jax.readthedocs.io/en/latest/async_dispatch.html) for more details).
In practice, however, we often don't need to
think about asynchronous dispatch because we're writing pure functions and only
care about the inputs and outputs of functions -- we'll naturally block on future
values.

However, now imagine that we have a `jax.print` function that works inside of
JIT-ted JAX functions (`host_callback.id_print` is an example of this). Let's
return to the previous example except with prints in the mix.
```python
@partial(jax.jit, device=<device 0>)
def f():
  jax.print("hello")
  return 2

@partial(jax.jit, device=<device 1>)
def g():
  jax.print("world")
  return 3
f()
g()
```
Thanks to asynchronous dispatch, we could actually see `"world"` being printed
before `"hello"`. The reordering of the print side-effects breaks the illusion
of a single-threaded execution model.

Another example of where side-effects can "reveal" out-of-order execution is
when we compile JAX programs. Consider the following JAX code:
```python
@jax.jit
def f(x):
  jax.print("hello")
  jax.print("world")
  return x
```
Even though in Python, we've written the `"hello"` print before the `"world"` print,
a compiler like XLA is free to reorder them because there's no explicit data-dependence between the prints.

## Motivation

We'd like to support "ordered" effects. When we say ordered, we mean that the effects
occur in the same order as we would if we were executing a single-threaded Python program.
This is our main desideratum. In the presence of explicit parallelism like `pmap` or
user threads, we don't need to maintain this behavior but at least if the user is not
explicitly requesting parallelism, we'd like to preserve a single-threaded ordering.

Before we dive in more, let's first step back and ask ourselves if it is okay
if we reorder effects in the name of performance, and conversely, do we need to
enforce an ordering on effects at all? In some cases, we don't need ordering.
Maybe some side-effects shouldn't adversely affect the
performance of a JAX program. However, for other side-effects, we may
want to enforce a single-threaded program order so users don't get counterintuitive
behavior. Consider a logging effect.
```python
@jax.jit
def f(x, y):
  log_value(x)
  log_value(y)
f(1, 2)
```
If `log` is mutating a global list, we might expect that we add `x` before adding `y`.
For a more strict effect, we may want the option to order the effects.

## Enforcing ordered effects

The main tool we have to enforce the ordering of computations is *data-dependence*.
Simply put, if a function `g` has an input that is the output of a function `f`,
`f` must be executed before `g`.

However, we may have side effects like prints that have no inputs at all
so naively we couldn't sequence them. We thus use *tokens* as a means of injecting
artificial data-dependence into a computation.

What is a token? A token is just a dummy value that can be threaded in and out of a computation.
By threading the same token in and out and several computations, we enforce that they have to happen
in a certain order. Let's take the previous print example and see what it would look like with tokens
in the mix:
```python
@jax.jit
def f(token, x):
  token = jax.print(token, "hello")
  token = jax.print(token, "world")
  return token, x
```
If we rewrite `jax.print` to take in and return a token, we have now sequenced
the two prints since the input to the second print depends on the output of the first print.
The actual value of `token` can be anything really, but we'll see in practice
that the tokens are invisible to users.

## Runtime tokens vs. compiler tokens

Here we will actually start talking about implementation details. In practice, we'll need
two separate types of tokens to sequence effects: one for each of the aforementioned sources
of reordering. We'll need *runtime tokens* to sequence asynchronously dispatched
side-effecting computations and we'll need *compiler tokens* to sequence effects within computations.

In practice, our computation will be rewritten to look like this:
```python
@jax.jit
def f(runtime_token, x):
  compiler_token = new_compiler_token()
  compiler_token = jax.print(compiler_token, "hello")
  compiler_token = jax.print(compiler_token, "world")
  return runtime_token, x
```
Notice how the runtime tokens are only used at the JIT boundary and the compiler tokens
are only within the compiled code. Compiler tokens are created during
"lowering" (we convert Python code to a lower level representation like HLO or StableHLO)
but runtime tokens need to be managed in Python since they're being threaded in and out
of JIT-ted functions.

Furthermore, notice that the runtime tokens are "disconnected"
from the compiler tokens meaning there's no data dependency between them. This could
potentially be dangerous as if we will lose the data dependence between the bodies
of two dispatched function calls. However, if we assume "strict execution" -- i.e.
a dispatched function will only start execution when all of its inputs are ready
and all of it outputs will become ready at the same time -- we are safe to create a
fresh compiler token and return a non-output-dependent runtime token.


## Managing runtime tokens

To manage runtime tokens on behalf of the user, we'll need to hook into JAX's dispatch machinery.
Whenever we call a JIT-ted function, we eventually bottom out in a function that looks like
this:
```python
def _execute(compiled_computation, *args):
  outputs = compiled_computation.execute(*args)
  return outputs
```
At this point we need to "inject" the runtime tokens into the computation
and "extract" them from the computation's outputs:
```python
def _execute(compiled_computation, *args):
  runtime_token = get_runtime_token() # Grab global token
  runtime_token, *outputs = compiled_computation.execute(runtime_token, *args)
  update_runtime_token(runtime_token) # Update global token
  return outputs
```

What is `runtime_token` exactly? Well we need to be able to pass it into a `compiled_computation`,
which means it needs to be some sort of array (for now, since there's no shared token representation
inside and outside compiled JAX code). In practice we can use a `(0,)`-shaped array to minimize overheads.


We also need to think about the multiple device use case, e.g. the first example where
we first call a JIT-ted function on device 0 and then one on device 1.
In that case, we need to also *copy* the runtime token returned from the first computation (which lives on device 0)
to device 1 so we can pass it into the second computation. If two subsequent computations share the same device,
this copy is not necessary.

## Adding compiler tokens

When we lower Python code to HLO or StableHLO we need to create a token at the start of the computation and
ensure they are available when we have side-effecting computations that need to be ordered. The side-effecting
computations will take the token as input and return it as an output.

The implementation of this token threading involves upgrading the JAX lowering machinery to do
this bookkeeping automatically.
The main challenges involve dealing with higher-order primitives like call primitives
and control-flow primitives. We won't go into details on how to handle those in this design note.

## Blocking on output tokens

Adding support for runtime and compiler tokens for side-effecting computations is important for sequencing
but there's also another subtle use-case for tokens, which is blocking on side-effecting computations.
Even if we don't want a side-effecting computation to be *ordered* we may still want to wait on its
completion. Currently we have `jax.block_until_ready`, which waits until a future value has its
result ready. However, with side-effecting computations, we may have functions that don't have a return
value but are still executing a side-effect. Take the simple example here:
```python
@jax.jit
def f():
  jax.print("hello world")
  return
f() # Executed asynchronously
```
This compiled computation takes no explicit inputs and has no explicit outputs. If it was an ordered print effect,
we could block on the returned runtime token, However,
when this is an unordered computation we don't do any token threading. How do we wait for `f()` to
finish executing when we have no output value to call `block_until_ready` on? Well, we could apply our same
token strategy except we only return runtime tokens and don't take them as inputs. This will give us
a value to block on that will only be ready once `f()` is done being executed. We'll call these tokens
*output tokens*. We end up with a function that looks like this:
```python
@jax.jit
def f():
  jax.print("hello world")
  return new_runtime_token()
f() # Executed asynchronously
```

Underneath the hood, we'll manage the output tokens in the same way we manage the runtime tokens but
provide a method for users to block on the current set of output tokens. Unlike runtime tokens,
output tokens need to be *device-specific*.
Consider a single device use-case:

```python
@jax.jit
def f():
  jax.print("hello")

@jax.jit
def g():
  jax.print("world")

f()
g()
```
Since `f()` and `g()` are executed on the same device, blocking on `g()`'s output token
effectively blocks on `f()` since (as of now!), the JAX runtime does not interleave computations
executed on the same device. We'll have to revise this entire design if that changes, of course.

However, consider the two device use-case:
```python
@partial(jax.jit, device=<device 0>)
def f():
  jax.print("hello")

@partial(jax.jit, device=<device 1>)
def g():
  jax.print("world")

f()
g()
```
Here we don't want to explicitly sequence `f()` and `g()` but want to wait for both of them to finish.
We'll need one output token for `f()` and one for `g()` and we'll block on both of those tokens:
```python
@partial(jax.jit, device=<device 0>)
def f():
  jax.print("hello")
  return new_runtime_token()

@partial(jax.jit, device=<device 1>)
def g():
  jax.print("world")
  return new_runtime_token()

t0 = f()
t1 = g()
block_until_ready((t0, t1))
```
We'll thus need a per-device output token so we can avoid sequencing computations on different
devices while offering the ability to block on side-effecting computations. We end up with the following
(approximate) change to the JAX dispatch machinery:

```python
def _execute(compiled_computation, *args):
  output_token, *outputs = compiled_computation.execute(runtime_token, *args)
  update_output_token(output_token, compiled_computation.device)
  return outputs
```
We'll also need to expose a function to that blocks on the output token:
```python
def effects_barrier():
  output_token.block_until_ready()
```

Note that blocking on output tokens may not be fairly common since most JAX computations will return
a value to block on. However, output tokens are helpful for testing and profiling, and are good to
support so that we have a consistent and cohesive effect system.

## Some more details

* All of the aforementioned token management infrastructure will be *thread-local*. This means
  that each user thread will have their own independent stream of runtime tokens. Sequencing
  is only promised at a user thread level.
* In practice, we have one runtime token per effect. Different instances of that effect will be
  sequenced. This is to avoid sequencing effectul computations that may not have any relation to each
  other. Technically this goes against our original goal though of enforcing a single-threaded Python
  program ordering, but this is a tradeoff that could be modulated by having both "effect"-specific tokens
  and "global" tokens.

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

(pseudorandom-numbers)=
# Pseudorandom numbers

<!--* freshness: { reviewed: '2024-05-03' } *-->

In this section we focus on {mod}`jax.random` and pseudo random number generation (PRNG); that is, the process of algorithmically generating sequences of numbers whose properties approximate the properties of sequences of random numbers sampled from an appropriate distribution.

PRNG-generated sequences are not truly random because they are actually determined by their initial value, which is typically referred to as the `seed`, and each step of random sampling is a deterministic function of some `state` that is carried over from a sample to the next.

Pseudo random number generation is an essential component of any machine learning or scientific computing framework. Generally, JAX strives to be compatible with NumPy, but pseudo random number generation is a notable exception.

To better understand the difference between the approaches taken by JAX and NumPy when it comes to random number generation we will discuss both approaches in this section.

## Random numbers in NumPy

Pseudo random number generation is natively supported in NumPy by the {mod}`numpy.random` module.
In NumPy, pseudo random number generation is based on a global `state`, which can be set to a deterministic initial condition using {func}`numpy.random.seed`.

```{code-cell}
import numpy as np
np.random.seed(0)
```

You can inspect the content of the state using the following command.

```{code-cell}
def print_truncated_random_state():
  """To avoid spamming the outputs, print only part of the state."""
  full_random_state = np.random.get_state()
  print(str(full_random_state)[:460], '...')

print_truncated_random_state()
```

The `state` is updated by each call to a random function:

```{code-cell}
np.random.seed(0)
print_truncated_random_state()
```

```{code-cell}
_ = np.random.uniform()
print_truncated_random_state()
```

NumPy allows you to sample both individual numbers, or entire vectors of numbers in a single function call. For instance, you may sample a vector of 3 scalars from a uniform distribution by doing:

```{code-cell}
np.random.seed(0)
print(np.random.uniform(size=3))
```

NumPy provides a *sequential equivalent guarantee*, meaning that sampling N numbers in a row individually or sampling a vector of N numbers results in the same pseudo-random sequences:

```{code-cell}
np.random.seed(0)
print("individually:", np.stack([np.random.uniform() for _ in range(3)]))

np.random.seed(0)
print("all at once: ", np.random.uniform(size=3))
```

## Random numbers in JAX

JAX's random number generation differs from NumPy's in important ways, because NumPy's
PRNG design makes it hard to simultaneously guarantee a number of desirable properties.
Specifically, in JAX we want PRNG generation to be:

1. reproducible,
2. parallelizable,
3. vectorisable.

We will discuss why in the following. First, we will focus on the implications of a PRNG design based on a global state. Consider the code:

```{code-cell}
import numpy as np

np.random.seed(0)

def bar(): return np.random.uniform()
def baz(): return np.random.uniform()

def foo(): return bar() + 2 * baz()

print(foo())
```

The function `foo` sums two scalars sampled from a uniform distribution.

The output of this code can only satisfy requirement #1 if we assume a predictable order of execution for `bar()` and `baz()`.
This is not a problem in NumPy, which always evaluates code in the order defined by the Python interpreter.
In JAX, however, this is more problematic: for efficient execution, we want the JIT compiler to be free to reorder, elide, and fuse various operations in the function we define.
Further, when executing in multi-device environments, execution efficiency would be hampered by the need for each process to synchronize a global state.

### Explicit random state

To avoid this issue, JAX avoids implicit global random state, and instead tracks state explicitly via a random `key`:

```{code-cell}
from jax import random

key = random.key(42)
print(key)
```

```{note}
This section uses the new-style typed PRNG keys produced by {func}`jax.random.key`, rather than the
old-style raw PRNG keys produced by {func}`jax.random.PRNGKey`. For details, see {ref}`jep-9263`.
```

A key is an array with a special dtype corresponding to the particular PRNG implementation being used; in the default implementation each key is backed by a pair of `uint32` values.

The key is effectively a stand-in for NumPy's hidden state object, but we pass it explicitly to {func}`jax.random` functions.
Importantly, random functions consume the key, but do not modify it: feeding the same key object to a random function will always result in the same sample being generated.

```{code-cell}
print(random.normal(key))
print(random.normal(key))
```

Re-using the same key, even with different {mod}`~jax.random` APIs, can result in correlated outputs, which is generally undesirable. 

**The rule of thumb is: never reuse keys (unless you want identical outputs).**

In order to generate different and independent samples, you must {func}`~jax.random.split` the key explicitly before passing it to a random function:

```{code-cell}
for i in range(3):
  new_key, subkey = random.split(key)
  del key  # The old key is consumed by split() -- we must never use it again.

  val = random.normal(subkey)
  del subkey  # The subkey is consumed by normal().

  print(f"draw {i}: {val}")
  key = new_key  # new_key is safe to use in the next iteration.
```

(Calling `del` here is not required, but we do so to emphasize that the key should not be reused once consumed.)

{func}`jax.random.split` is a deterministic function that converts one `key` into several independent (in the pseudorandomness sense) keys.
We keep one of the outputs as the `new_key`, and can safely use the unique extra key (called `subkey`) as input into a random function, and then discard it forever.
If you wanted to get another sample from the normal distribution, you would split `key` again, and so on: the crucial point is that you never use the same key twice.

It doesn't matter which part of the output of `split(key)` we call `key`, and which we call `subkey`.
They are all independent keys with equal status.
The key/subkey naming convention is a typical usage pattern that helps track how keys are consumed:
subkeys are destined for immediate consumption by random functions, while the key is retained to generate more randomness later.

Usually, the above example would be written concisely as

```{code-cell}
key, subkey = random.split(key)
```

which discards the old key automatically.
It's worth noting that {func}`~jax.random.split` can create as many keys as you need, not just 2:

```{code-cell}
key, *forty_two_subkeys = random.split(key, num=43)
```

### Lack of sequential equivalence

Another difference between NumPy's and JAX's random modules relates to the sequential equivalence guarantee mentioned above.

As in NumPy, JAX's random module also allows sampling of vectors of numbers.
However, JAX does not provide a sequential equivalence guarantee, because doing so would interfere with the vectorization on SIMD hardware (requirement #3 above).

In the example below, sampling 3 values out of a normal distribution individually using three subkeys gives a different result to using giving a single key and specifying `shape=(3,)`:

```{code-cell}
key = random.key(42)
subkeys = random.split(key, 3)
sequence = np.stack([random.normal(subkey) for subkey in subkeys])
print("individually:", sequence)

key = random.key(42)
print("all at once: ", random.normal(key, shape=(3,)))
```

The lack of sequential equivalence gives us freedom to write code more efficiently; for example,
instead of generating `sequence` above via a sequential loop, we can use {func}`jax.vmap` to
compute the same result in a vectorized manner:

```{code-cell}
import jax
print("vectorized:", jax.vmap(random.normal)(subkeys))
```

## Next Steps

For more information on JAX random numbers, refer to the documentation of the {mod}`jax.random`
module. If you're interested in the details of the design of JAX's random number generator,
see {ref}`prng-design-jep`.

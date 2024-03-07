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

+++ {"id": "hjM_sV_AepYf"}

# 🔪 JAX - The Sharp Bits 🔪

<!--* freshness: { reviewed: '2024-06-03' } *-->

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/google/jax/blob/main/docs/notebooks/Common_Gotchas_in_JAX.ipynb) [![Open in Kaggle](https://kaggle.com/static/images/open-in-kaggle.svg)](https://kaggle.com/kernels/welcome?src=https://github.com/google/jax/blob/main/docs/notebooks/Common_Gotchas_in_JAX.ipynb)

+++ {"id": "4k5PVzEo2uJO"}

*levskaya@ mattjj@*

When walking about the countryside of Italy, the people will not hesitate to tell you that __JAX__ has [_"una anima di pura programmazione funzionale"_](https://www.sscardapane.it/iaml-backup/jax-intro/).

__JAX__ is a language for __expressing__ and __composing__ __transformations__ of numerical programs. __JAX__ is also able to __compile__ numerical programs for CPU or accelerators (GPU/TPU).
JAX works great for many numerical and scientific programs, but __only if they are written with certain constraints__ that we describe below.

```{code-cell} ipython3
:id: GoK_PCxPeYcy

import numpy as np
from jax import grad, jit
from jax import lax
from jax import random
import jax
import jax.numpy as jnp
```

+++ {"id": "gX8CZU1g2agP"}

## 🔪 Pure functions

+++ {"id": "2oHigBkW2dPT"}

JAX transformation and compilation are designed to work only on Python functions that are functionally pure: all the input data is passed through the function parameters, all the results are output through the function results. A pure function will always return the same result if invoked with the same inputs.

Here are some examples of functions that are not functionally pure for which JAX behaves differently than the Python interpreter. Note that these behaviors are not guaranteed by the JAX system; the proper way to use JAX is to use it only on functionally pure Python functions.

```{code-cell} ipython3
:id: A6R-pdcm4u3v
:outputId: 25dcb191-14d4-4620-bcb2-00492d2f24e1

def impure_print_side_effect(x):
  print("Executing function")  # This is a side-effect
  return x

# The side-effects appear during the first run
print ("First call: ", jit(impure_print_side_effect)(4.))

# Subsequent runs with parameters of same type and shape may not show the side-effect
# This is because JAX now invokes a cached compilation of the function
print ("Second call: ", jit(impure_print_side_effect)(5.))

# JAX re-runs the Python function when the type or shape of the argument changes
print ("Third call, different type: ", jit(impure_print_side_effect)(jnp.array([5.])))
```

```{code-cell} ipython3
:id: -N8GhitI2bhD
:outputId: fd3624c9-197d-42cb-d97f-c5e0ef885467

g = 0.
def impure_uses_globals(x):
  return x + g

# JAX captures the value of the global during the first run
print ("First call: ", jit(impure_uses_globals)(4.))
g = 10.  # Update the global

# Subsequent runs may silently use the cached value of the globals
print ("Second call: ", jit(impure_uses_globals)(5.))

# JAX re-runs the Python function when the type or shape of the argument changes
# This will end up reading the latest value of the global
print ("Third call, different type: ", jit(impure_uses_globals)(jnp.array([4.])))
```

```{code-cell} ipython3
:id: RTB6iFgu4DL6
:outputId: 16697bcd-3623-49b1-aabb-c54614aeadea

g = 0.
def impure_saves_global(x):
  global g
  g = x
  return x

# JAX runs once the transformed function with special Traced values for arguments
print ("First call: ", jit(impure_saves_global)(4.))
print ("Saved global: ", g)  # Saved global has an internal JAX value
```

+++ {"id": "Mlc2pQlp6v-9"}

A Python function can be functionally pure even if it actually uses stateful objects internally, as long as it does not read or write external state:

```{code-cell} ipython3
:id: TP-Mqf_862C0
:outputId: 78d55886-54de-483c-e7c4-bafd1d2c7219

def pure_uses_internal_state(x):
  state = dict(even=0, odd=0)
  for i in range(10):
    state['even' if i % 2 == 0 else 'odd'] += x
  return state['even'] + state['odd']

print(jit(pure_uses_internal_state)(5.))
```

+++ {"id": "cDpQ5u63Ba_H"}

It is not recommended to use iterators in any JAX function you want to `jit` or in any control-flow primitive. The reason is that an iterator is a python object which introduces state to retrieve the next element. Therefore, it is incompatible with JAX functional programming model. In the code below, there are some examples of incorrect attempts to use iterators with JAX. Most of them return an error, but some give unexpected results.

```{code-cell} ipython3
:id: w99WXa6bBa_H
:outputId: 52d885fd-0239-4a08-f5ce-0c38cc008903

import jax.numpy as jnp
import jax.lax as lax
from jax import make_jaxpr

# lax.fori_loop
array = jnp.arange(10)
print(lax.fori_loop(0, 10, lambda i,x: x+array[i], 0)) # expected result 45
iterator = iter(range(10))
print(lax.fori_loop(0, 10, lambda i,x: x+next(iterator), 0)) # unexpected result 0

# lax.scan
def func11(arr, extra):
    ones = jnp.ones(arr.shape)
    def body(carry, aelems):
        ae1, ae2 = aelems
        return (carry + ae1 * ae2 + extra, carry)
    return lax.scan(body, 0., (arr, ones))
make_jaxpr(func11)(jnp.arange(16), 5.)
# make_jaxpr(func11)(iter(range(16)), 5.) # throws error

# lax.cond
array_operand = jnp.array([0.])
lax.cond(True, lambda x: x+1, lambda x: x-1, array_operand)
iter_operand = iter(range(10))
# lax.cond(True, lambda x: next(x)+1, lambda x: next(x)-1, iter_operand) # throws error
```

+++ {"id": "oBdKtkVW8Lha"}

## 🔪 In-Place Updates

+++ {"id": "JffAqnEW4JEb"}

In Numpy you're used to doing this:

```{code-cell} ipython3
:id: om4xV7_84N9j
:outputId: 88b0074a-4440-41f6-caa7-031ac2d1a96f

numpy_array = np.zeros((3,3), dtype=np.float32)
print("original array:")
print(numpy_array)

# In place, mutating update
numpy_array[1, :] = 1.0
print("updated array:")
print(numpy_array)
```

+++ {"id": "go3L4x3w4-9p"}

If we try to update a JAX device array in-place, however, we get an __error__!  (☉_☉)

```{code-cell} ipython3
:id: iOscaa_GecEK
:outputId: 26fdb703-a476-4b7f-97ba-d28997ef750c

%xmode Minimal
```

```{code-cell} ipython3
:id: 2AxeCufq4wAp
:outputId: fa4a87ad-1a84-471a-a3c5-a1396c432c85
:tags: [raises-exception]

jax_array = jnp.zeros((3,3), dtype=jnp.float32)

# In place update of JAX's array will yield an error!
jax_array[1, :] = 1.0
```

+++ {"id": "7mo76sS25Wco"}

Allowing mutation of variables in-place makes program analysis and transformation difficult. JAX requires that programs are pure functions.

Instead, JAX offers a _functional_ array update using the [`.at` property on JAX arrays](https://jax.readthedocs.io/en/latest/_autosummary/jax.numpy.ndarray.at.html#jax.numpy.ndarray.at).

+++ {"id": "hfloZ1QXCS_J"}

️⚠️ inside `jit`'d code and `lax.while_loop` or `lax.fori_loop` the __size__ of slices can't be functions of argument _values_ but only functions of argument _shapes_ -- the slice start indices have no such restriction.  See the below __Control Flow__ Section for more information on this limitation.

+++ {"id": "X2Xjjvd-l8NL"}

### Array updates: `x.at[idx].set(y)`

+++ {"id": "SHLY52KQEiuX"}

For example, the update above can be written as:

```{code-cell} ipython3
:id: PBGI-HIeCP_s
:outputId: de13f19a-2066-4df1-d503-764c34585529

updated_array = jax_array.at[1, :].set(1.0)
print("updated array:\n", updated_array)
```

+++ {"id": "zUANAw9sCmgu"}

JAX's array update functions, unlike their NumPy versions, operate out-of-place. That is, the updated array is returned as a new array and the original array is not modified by the update.

```{code-cell} ipython3
:id: dbB0UmMhCe8f
:outputId: 55d46fa1-d0de-4c43-996c-f3bbc87b7175

print("original array unchanged:\n", jax_array)
```

+++ {"id": "eM6MyndXL2NY"}

However, inside __jit__-compiled code, if the __input value__ `x` of `x.at[idx].set(y)` is not reused, the compiler will optimize the array update to occur _in-place_.

+++ {"id": "7to-sF8EmC_y"}

### Array updates with other operations

+++ {"id": "ZY5l3tAdDmsJ"}

Indexed array updates are not limited simply to overwriting values. For example, we can perform indexed addition as follows:

```{code-cell} ipython3
:id: tsw2svao8FUp
:outputId: 3c62a3b1-c12d-46f0-da74-791ec4b61e0b

print("original array:")
jax_array = jnp.ones((5, 6))
print(jax_array)

new_jax_array = jax_array.at[::2, 3:].add(7.)
print("new array post-addition:")
print(new_jax_array)
```

+++ {"id": "sTjJ3WuaDyqU"}

For more details on indexed array updates, see the [documentation for the `.at` property](https://jax.readthedocs.io/en/latest/_autosummary/jax.numpy.ndarray.at.html#jax.numpy.ndarray.at).

+++ {"id": "oZ_jE2WAypdL"}

## 🔪 Out-of-Bounds Indexing

+++ {"id": "btRFwEVzypdN"}

In Numpy, you are used to errors being thrown when you index an array outside of its bounds, like this:

```{code-cell} ipython3
:id: 5_ZM-BJUypdO
:outputId: c9c41ae8-2653-4219-e6dc-09b03faa3b95
:tags: [raises-exception]

np.arange(10)[11]
```

+++ {"id": "eoXrGARWypdR"}

However, raising an error from code running on an accelerator can be difficult or impossible. Therefore, JAX must choose some non-error behavior for out of bounds indexing (akin to how invalid floating point arithmetic results in `NaN`). When the indexing operation is an array index update (e.g. `index_add` or `scatter`-like primitives), updates at out-of-bounds indices will be skipped; when the operation is an array index retrieval (e.g. NumPy indexing or `gather`-like primitives) the index is clamped to the bounds of the array since __something__ must be returned. For example, the last value of the array will be returned from this indexing operation:

```{code-cell} ipython3
:id: cusaAD0NypdR
:outputId: af1708aa-b50b-4da8-f022-7f2fa67030a8

jnp.arange(10)[11]
```

+++ {"id": "NAcXJNAcDi_v"}

If you would like finer-grained control over the behavior for out-of-bound indices, you can use the optional parameters of [`ndarray.at`](https://jax.readthedocs.io/en/latest/_autosummary/jax.numpy.ndarray.at.html); for example:

```{code-cell} ipython3
:id: -0-MaFddO-xy
:outputId: 746c4b2b-a90e-4ec9-de56-ed6682d451e5

jnp.arange(10.0).at[11].get()
```

```{code-cell} ipython3
:id: g5JEJtIUPBXi
:outputId: 4a0f6854-1165-47f2-e1ac-5a21fa2b8516

jnp.arange(10.0).at[11].get(mode='fill', fill_value=jnp.nan)
```

+++ {"id": "J8uO8yevBa_M"}

Note that due to this behavior for index retrieval, functions like `jnp.nanargmin` and `jnp.nanargmax` return -1 for slices consisting of NaNs whereas Numpy would throw an error.

Note also that, as the two behaviors described above are not inverses of each other, reverse-mode automatic differentiation (which turns index updates into index retrievals and vice versa) [will not preserve the semantics of out of bounds indexing](https://github.com/google/jax/issues/5760). Thus it may be a good idea to think of out-of-bounds indexing in JAX as a case of [undefined behavior](https://en.wikipedia.org/wiki/Undefined_behavior).

+++ {"id": "LwB07Kx5sgHu"}

## 🔪 Non-array inputs: NumPy vs. JAX

NumPy is generally happy accepting Python lists or tuples as inputs to its API functions:

```{code-cell} ipython3
:id: sErQES14sjCG
:outputId: 601485ff-4cda-48c5-f76c-2789073c4591

np.sum([1, 2, 3])
```

+++ {"id": "ZJ1Wt1bTtrSA"}

JAX departs from this, generally returning a helpful error:

```{code-cell} ipython3
:id: DFEGcENSsmEc
:outputId: 08535679-6c1f-4dd9-a414-d8b59310d1ee
:tags: [raises-exception]

jnp.sum([1, 2, 3])
```

+++ {"id": "QPliLUZztxJt"}

This is a deliberate design choice, because passing lists or tuples to traced functions can lead to silent performance degradation that might otherwise be difficult to detect.

For example, consider the following permissive version of `jnp.sum` that allows list inputs:

```{code-cell} ipython3
:id: jhe-L_TwsvKd
:outputId: ab2ee183-d9ec-45cc-d6be-5009347e1bc5

def permissive_sum(x):
  return jnp.sum(jnp.array(x))

x = list(range(10))
permissive_sum(x)
```

+++ {"id": "m0XZLP7nuYdE"}

The output is what we would expect, but this hides potential performance issues under the hood. In JAX's tracing and JIT compilation model, each element in a Python list or tuple is treated as a separate JAX variable, and individually processed and pushed to device. This can be seen in the jaxpr for the ``permissive_sum`` function above:

```{code-cell} ipython3
:id: k81u6DQ7vAjQ
:outputId: 869fc3b9-feda-4aa9-d2e5-5b5107de102d

make_jaxpr(permissive_sum)(x)
```

+++ {"id": "C0_dpCfpvCts"}

Each entry of the list is handled as a separate input, resulting in a tracing & compilation overhead that grows linearly with the size of the list. To prevent surprises like this, JAX avoids implicit conversions of lists and tuples to arrays.

If you would like to pass a tuple or list to a JAX function, you can do so by first explicitly converting it to an array:

```{code-cell} ipython3
:id: nFf_DydixG8v
:outputId: e31b43b3-05f7-4300-fdd2-40e3896f6f8f

jnp.sum(jnp.array(x))
```

+++ {"id": "MUycRNh6e50W"}

## 🔪 Random Numbers

+++ {"id": "O8vvaVt3MRG2"}

> _If all scientific papers whose results are in doubt because of bad
> `rand()`s were to disappear from library shelves, there would be a
> gap on each shelf about as big as your fist._ - Numerical Recipes

+++ {"id": "Qikt9pPW9L5K"}

### RNGs and State
You're used to _stateful_ pseudorandom number generators (PRNGs) from numpy and other libraries, which helpfully hide a lot of details under the hood to give you a ready fountain of pseudorandomness:

```{code-cell} ipython3
:id: rr9FeP41fynt
:outputId: df0ceb15-96ec-4a78-e327-c77f7ea3a745

print(np.random.random())
print(np.random.random())
print(np.random.random())
```

+++ {"id": "ORMVVGZJgSVi"}

Underneath the hood, numpy uses the [Mersenne Twister](https://en.wikipedia.org/wiki/Mersenne_Twister) PRNG to power its pseudorandom functions.  The PRNG has a period of $2^{19937}-1$ and at any point can be described by __624 32-bit unsigned ints__ and a __position__ indicating how much of this  "entropy" has been used up.

```{code-cell} ipython3
:id: 7Pyp2ajzfPO2

np.random.seed(0)
rng_state = np.random.get_state()
# print(rng_state)
# --> ('MT19937', array([0, 1, 1812433255, 1900727105, 1208447044,
#       2481403966, 4042607538,  337614300, ... 614 more numbers...,
#       3048484911, 1796872496], dtype=uint32), 624, 0, 0.0)
```

+++ {"id": "aJIxHVXCiM6m"}

This pseudorandom state vector is automagically updated behind the scenes every time a random number is needed, "consuming" 2 of the uint32s in the Mersenne twister state vector:

```{code-cell} ipython3
:id: GAHaDCYafpAF

_ = np.random.uniform()
rng_state = np.random.get_state()
#print(rng_state)
# --> ('MT19937', array([2443250962, 1093594115, 1878467924,
#       ..., 2648828502, 1678096082], dtype=uint32), 2, 0, 0.0)

# Let's exhaust the entropy in this PRNG statevector
for i in range(311):
  _ = np.random.uniform()
rng_state = np.random.get_state()
#print(rng_state)
# --> ('MT19937', array([2443250962, 1093594115, 1878467924,
#       ..., 2648828502, 1678096082], dtype=uint32), 624, 0, 0.0)

# Next call iterates the RNG state for a new batch of fake "entropy".
_ = np.random.uniform()
rng_state = np.random.get_state()
# print(rng_state)
# --> ('MT19937', array([1499117434, 2949980591, 2242547484,
#      4162027047, 3277342478], dtype=uint32), 2, 0, 0.0)
```

+++ {"id": "N_mWnleNogps"}

The problem with magic PRNG state is that it's hard to reason about how it's being used and updated across different threads, processes, and devices, and it's _very easy_ to screw up when the details of entropy production and consumption are hidden from the end user.

The Mersenne Twister PRNG is also known to have a [number](https://cs.stackexchange.com/a/53475) of problems, it has a large 2.5kB state size, which leads to problematic [initialization issues](https://dl.acm.org/citation.cfm?id=1276928).  It [fails](http://www.pcg-random.org/pdf/toms-oneill-pcg-family-v1.02.pdf) modern BigCrush tests, and is generally slow.

+++ {"id": "Uvq7nV-j4vKK"}

### JAX PRNG

+++ {"id": "COjzGBpO4tzL"}

JAX instead implements an _explicit_ PRNG where entropy production and consumption are handled by explicitly passing and iterating PRNG state.  JAX uses a modern [Threefry counter-based PRNG](https://github.com/google/jax/blob/main/docs/jep/263-prng.md) that's __splittable__.  That is, its design allows us to __fork__ the PRNG state into new PRNGs for use with parallel stochastic generation.

The random state is described by a special array element that we call a __key__:

```{code-cell} ipython3
:id: yPHE7KTWgAWs
:outputId: ae8af0ee-f19e-474e-81b6-45e894eb2fc3

from jax import random
key = random.key(0)
key
```

+++ {"id": "XjYyWYNfq0hW"}

JAX's random functions produce pseudorandom numbers from the PRNG state, but __do not__ change the state!

Reusing the same state will cause __sadness__ and __monotony__, depriving the end user of __lifegiving chaos__:

```{code-cell} ipython3
:id: 7zUdQMynoE5e
:outputId: 23a86b72-dfb9-410a-8e68-22b48dc10805

print(random.normal(key, shape=(1,)))
print(key)
# No no no!
print(random.normal(key, shape=(1,)))
print(key)
```

+++ {"id": "hQN9van8rJgd"}

Instead, we __split__ the PRNG to get usable __subkeys__ every time we need a new pseudorandom number:

```{code-cell} ipython3
:id: ASj0_rSzqgGh
:outputId: 2f13f249-85d1-47bb-d503-823eca6961aa

print("old key", key)
key, subkey = random.split(key)
normal_pseudorandom = random.normal(subkey, shape=(1,))
print("    \---SPLIT --> new key   ", key)
print("             \--> new subkey", subkey, "--> normal", normal_pseudorandom)
```

+++ {"id": "tqtFVE4MthO3"}

We propagate the __key__ and make new __subkeys__ whenever we need a new random number:

```{code-cell} ipython3
:id: jbC34XLor2Ek
:outputId: 4059a2e2-0205-40bc-ad55-17709d538871

print("old key", key)
key, subkey = random.split(key)
normal_pseudorandom = random.normal(subkey, shape=(1,))
print("    \---SPLIT --> new key   ", key)
print("             \--> new subkey", subkey, "--> normal", normal_pseudorandom)
```

+++ {"id": "0KLYUluz3lN3"}

We can generate more than one __subkey__ at a time:

```{code-cell} ipython3
:id: lEi08PJ4tfkX
:outputId: 1f280560-155d-4c04-98e8-c41d72ee5b01

key, *subkeys = random.split(key, 4)
for subkey in subkeys:
  print(random.normal(subkey, shape=(1,)))
```

+++ {"id": "rg4CpMZ8c3ri"}

## 🔪 Control Flow

+++ {"id": "izLTvT24dAq0"}

### ✔ python control_flow + autodiff ✔

If you just want to apply `grad` to your python functions, you can use regular python control-flow constructs with no problems, as if you were using [Autograd](https://github.com/hips/autograd) (or Pytorch or TF Eager).

```{code-cell} ipython3
:id: aAx0T3F8lLtu
:outputId: 383b7bfa-1634-4d23-8497-49cb9452ca52

def f(x):
  if x < 3:
    return 3. * x ** 2
  else:
    return -4 * x

print(grad(f)(2.))  # ok!
print(grad(f)(4.))  # ok!
```

+++ {"id": "hIfPT7WMmZ2H"}

### python control flow + JIT

Using control flow with `jit` is more complicated, and by default it has more constraints.

This works:

```{code-cell} ipython3
:id: OZ_BJX0CplNC
:outputId: 60c902a2-eba1-49d7-c8c8-2f68616d660c

@jit
def f(x):
  for i in range(3):
    x = 2 * x
  return x

print(f(3))
```

+++ {"id": "22RzeJ4QqAuX"}

So does this:

```{code-cell} ipython3
:id: pinVnmRWp6w6
:outputId: 25e06cf2-474f-4782-af7c-4f5514b64422

@jit
def g(x):
  y = 0.
  for i in range(x.shape[0]):
    y = y + x[i]
  return y

print(g(jnp.array([1., 2., 3.])))
```

+++ {"id": "TStltU2dqf8A"}

But this doesn't, at least by default:

```{code-cell} ipython3
:id: 9z38AIKclRNM
:outputId: 38dd2075-92fc-4b81-fee0-b9dff8da1fac
:tags: [raises-exception]

@jit
def f(x):
  if x < 3:
    return 3. * x ** 2
  else:
    return -4 * x

# This will fail!
f(2)
```

+++ {"id": "pIbr4TVPqtDN"}

__What gives!?__

When we `jit`-compile a function, we usually want to compile a version of the function that works for many different argument values, so that we can cache and reuse the compiled code. That way we don't have to re-compile on each function evaluation.

For example, if we evaluate an `@jit` function on the array `jnp.array([1., 2., 3.], jnp.float32)`, we might want to compile code that we can reuse to evaluate the function on `jnp.array([4., 5., 6.], jnp.float32)` to save on compile time.

To get a view of your Python code that is valid for many different argument values, JAX traces it on _abstract values_ that represent sets of possible inputs. There are [multiple different levels of abstraction](https://github.com/google/jax/blob/main/jax/_src/abstract_arrays.py), and different transformations use different abstraction levels.

By default, `jit` traces your code on the `ShapedArray` abstraction level, where each abstract value represents the set of all array values with a fixed shape and dtype. For example, if we trace using the abstract value `ShapedArray((3,), jnp.float32)`, we get a view of the function that can be reused for any concrete value in the corresponding set of arrays. That means we can save on compile time.

But there's a tradeoff here: if we trace a Python function on a `ShapedArray((), jnp.float32)` that isn't committed to a specific concrete value, when we hit a line like `if x < 3`, the expression `x < 3` evaluates to an abstract `ShapedArray((), jnp.bool_)` that represents the set `{True, False}`. When Python attempts to coerce that to a concrete `True` or `False`, we get an error: we don't know which branch to take, and can't continue tracing! The tradeoff is that with higher levels of abstraction we gain a more general view of the Python code (and thus save on re-compilations), but we require more constraints on the Python code to complete the trace.

The good news is that you can control this tradeoff yourself. By having `jit` trace on more refined abstract values, you can relax the traceability constraints. For example, using the `static_argnums` argument to `jit`, we can specify to trace on concrete values of some arguments. Here's that example function again:

```{code-cell} ipython3
:id: -Tzp0H7Bt1Sn
:outputId: f7f664cb-2cd0-4fd7-c685-4ec6ba1c4b7a

def f(x):
  if x < 3:
    return 3. * x ** 2
  else:
    return -4 * x

f = jit(f, static_argnums=(0,))

print(f(2.))
```

+++ {"id": "MHm1hIQAvBVs"}

Here's another example, this time involving a loop:

```{code-cell} ipython3
:id: iwY86_JKvD6b
:outputId: 48f9b51f-bd32-466f-eac1-cd23444ce937

def f(x, n):
  y = 0.
  for i in range(n):
    y = y + x[i]
  return y

f = jit(f, static_argnums=(1,))

f(jnp.array([2., 3., 4.]), 2)
```

+++ {"id": "nSPTOX8DvOeO"}

In effect, the loop gets statically unrolled.  JAX can also trace at _higher_ levels of abstraction, like `Unshaped`, but that's not currently the default for any transformation

+++ {"id": "wWdg8LTYwCW3"}

️⚠️ **functions with argument-__value__ dependent shapes**

These control-flow issues also come up in a more subtle way: numerical functions we want to __jit__ can't specialize the shapes of internal arrays on argument _values_ (specializing on argument __shapes__ is ok).  As a trivial example, let's make a function whose output happens to depend on the input variable `length`.

```{code-cell} ipython3
:id: Tqe9uLmUI_Gv
:outputId: 989be121-dfce-4bb3-c78e-a10829c5f883

def example_fun(length, val):
  return jnp.ones((length,)) * val
# un-jit'd works fine
print(example_fun(5, 4))
```

```{code-cell} ipython3
:id: fOlR54XRgHpd
:outputId: cf31d798-a4ce-4069-8e3e-8f9631ff4b71
:tags: [raises-exception]

bad_example_jit = jit(example_fun)
# this will fail:
bad_example_jit(10, 4)
```

```{code-cell} ipython3
:id: kH0lOD4GgFyI
:outputId: d009fcf5-c9f9-4ce6-fc60-22dc2cf21ade

# static_argnums tells JAX to recompile on changes at these argument positions:
good_example_jit = jit(example_fun, static_argnums=(0,))
# first compile
print(good_example_jit(10, 4))
# recompiles
print(good_example_jit(5, 4))
```

+++ {"id": "MStx_r2oKxpp"}

`static_argnums` can be handy if `length` in our example rarely changes, but it would be disastrous if it changed a lot!

Lastly, if your function has global side-effects, JAX's tracer can cause weird things to happen. A common gotcha is trying to print arrays inside __jit__'d functions:

```{code-cell} ipython3
:id: m2ABpRd8K094
:outputId: 4f7ebe17-ade4-4e18-bd8c-4b24087c33c3

@jit
def f(x):
  print(x)
  y = 2 * x
  print(y)
  return y
f(2)
```

+++ {"id": "uCDcWG4MnVn-"}

### Structured control flow primitives

There are more options for control flow in JAX. Say you want to avoid re-compilations but still want to use control flow that's traceable, and that avoids un-rolling large loops. Then you can use these 4 structured control flow primitives:

 - `lax.cond`  _differentiable_
 - `lax.while_loop` __fwd-mode-differentiable__
 - `lax.fori_loop` __fwd-mode-differentiable__ in general; __fwd and rev-mode differentiable__ if endpoints are static.
 - `lax.scan` _differentiable_

+++ {"id": "Sd9xrLMXeK3A"}

#### `cond`
python equivalent:

```python
def cond(pred, true_fun, false_fun, operand):
  if pred:
    return true_fun(operand)
  else:
    return false_fun(operand)
```

```{code-cell} ipython3
:id: SGxz9JOWeiyH
:outputId: 942a8d0e-5ff6-4702-c499-b3941f529ca3

from jax import lax

operand = jnp.array([0.])
lax.cond(True, lambda x: x+1, lambda x: x-1, operand)
# --> array([1.], dtype=float32)
lax.cond(False, lambda x: x+1, lambda x: x-1, operand)
# --> array([-1.], dtype=float32)
```

+++ {"id": "lIYdn1woOS1n"}

`jax.lax` provides two other functions that allow branching on dynamic predicates:

- [`lax.select`](https://jax.readthedocs.io/en/latest/_autosummary/jax.lax.select.html) is
  like a batched version of `lax.cond`, with the choices expressed as pre-computed arrays
  rather than as functions.
- [`lax.switch`](https://jax.readthedocs.io/en/latest/_autosummary/jax.lax.switch.html) is
  like `lax.cond`, but allows switching between any number of callable choices.

In addition, `jax.numpy` provides several numpy-style interfaces to these functions:

- [`jnp.where`](https://jax.readthedocs.io/en/latest/_autosummary/jax.numpy.where.html) with
  three arguments is the numpy-style wrapper of `lax.select`.
- [`jnp.piecewise`](https://jax.readthedocs.io/en/latest/_autosummary/jax.numpy.piecewise.html)
  is a numpy-style wrapper of `lax.switch`, but switches on a list of boolean conditions rather than a single scalar index.
- [`jnp.select`](https://jax.readthedocs.io/en/latest/_autosummary/jax.numpy.select.html) has
  an API similar to `jnp.piecewise`, but the choices are given as pre-computed arrays rather
  than as functions. It is implemented in terms of multiple calls to `lax.select`.

+++ {"id": "xkOFAw24eOMg"}

#### `while_loop`

python equivalent:
```
def while_loop(cond_fun, body_fun, init_val):
  val = init_val
  while cond_fun(val):
    val = body_fun(val)
  return val
```

```{code-cell} ipython3
:id: jM-D39a-c436
:outputId: 552fe42f-4d32-4e25-c8c2-b951160a3f4e

init_val = 0
cond_fun = lambda x: x<10
body_fun = lambda x: x+1
lax.while_loop(cond_fun, body_fun, init_val)
# --> array(10, dtype=int32)
```

+++ {"id": "apo3n3HAeQY_"}

#### `fori_loop`
python equivalent:
```
def fori_loop(start, stop, body_fun, init_val):
  val = init_val
  for i in range(start, stop):
    val = body_fun(i, val)
  return val
```

```{code-cell} ipython3
:id: dt3tUpOmeR8u
:outputId: 7819ca7c-1433-4d85-b542-f6159b0e8380

init_val = 0
start = 0
stop = 10
body_fun = lambda i,x: x+i
lax.fori_loop(start, stop, body_fun, init_val)
# --> array(45, dtype=int32)
```

+++ {"id": "SipXS5qiqk8e"}

#### Summary

$$
\begin{array} {r|rr}
\hline \
\textrm{construct}
& \textrm{jit}
& \textrm{grad} \\
\hline \
\textrm{if} & ❌ & ✔ \\
\textrm{for} & ✔* & ✔\\
\textrm{while} & ✔* & ✔\\
\textrm{lax.cond} & ✔ & ✔\\
\textrm{lax.while_loop} & ✔ & \textrm{fwd}\\
\textrm{lax.fori_loop} & ✔ & \textrm{fwd}\\
\textrm{lax.scan} & ✔ & ✔\\
\hline
\end{array}
$$

<center>

$\ast$ = argument-<b>value</b>-independent loop condition - unrolls the loop

</center>

+++ {"id": "OxLsZUyRt_kF"}

## 🔪 Dynamic Shapes

+++ {"id": "1tKXcAMduDR1"}

JAX code used within transforms like `jax.jit`, `jax.vmap`, `jax.grad`, etc. requires all output arrays and intermediate arrays to have static shape: that is, the shape cannot depend on values within other arrays.

For example, if you were implementing your own version of `jnp.nansum`, you might start with something like this:

```{code-cell} ipython3
:id: 9GIwgvfLujiD

def nansum(x):
  mask = ~jnp.isnan(x)  # boolean mask selecting non-nan values
  x_without_nans = x[mask]
  return x_without_nans.sum()
```

+++ {"id": "43S7wYAiupGe"}

Outside JIT and other transforms, this works as expected:

```{code-cell} ipython3
:id: ITYoNQEZur4s
:outputId: a9a03d25-9c54-43b6-d35e-aea6c448d680

x = jnp.array([1, 2, jnp.nan, 3, 4])
print(nansum(x))
```

+++ {"id": "guup5n8xvGI-"}

If you attempt to apply `jax.jit` or another transform to this function, it will error:

```{code-cell} ipython3
:id: nms9KjQEvNTz
:outputId: d8ae982f-111d-45b6-99f8-37715e2eaab3
:tags: [raises-exception]

jax.jit(nansum)(x)
```

+++ {"id": "r2aGyHDkvauu"}

The problem is that the size of `x_without_nans` is dependent on the values within `x`, which is another way of saying its size is *dynamic*.
Often in JAX it is possible to work-around the need for dynamically-sized arrays via other means.
For example, here it is possible to use the three-argument form of  `jnp.where` to replace the NaN values with zeros, thus computing the same result while avoiding dynamic shapes:

```{code-cell} ipython3
:id: Zbuj7Dg1wnSg
:outputId: 81a5e356-cd28-4709-b307-07c6254c82de

@jax.jit
def nansum_2(x):
  mask = ~jnp.isnan(x)  # boolean mask selecting non-nan values
  return jnp.where(mask, x, 0).sum()

print(nansum_2(x))
```

+++ {"id": "uGH-jqK7wxTl"}

Similar tricks can be played in other situations where dynamically-shaped arrays occur.

+++ {"id": "DKTMw6tRZyK2"}

## 🔪 NaNs

+++ {"id": "ncS0NI4jZrwy"}

### Debugging NaNs

If you want to trace where NaNs are occurring in your functions or gradients, you can turn on the NaN-checker by:

* setting the `JAX_DEBUG_NANS=True` environment variable;

* adding `jax.config.update("jax_debug_nans", True)` near the top of your main file;

* adding `jax.config.parse_flags_with_absl()` to your main file, then set the option using a command-line flag like `--jax_debug_nans=True`;

This will cause computations to error-out immediately on production of a NaN. Switching this option on adds a nan check to every floating point type value produced by XLA. That means values are pulled back to the host and checked as ndarrays for every primitive operation not under an `@jit`. For code under an `@jit`, the output of every `@jit` function is checked and if a nan is present it will re-run the function in de-optimized op-by-op mode, effectively removing one level of `@jit` at a time.

There could be tricky situations that arise, like nans that only occur under a `@jit` but don't get produced in de-optimized mode. In that case you'll see a warning message print out but your code will continue to execute.

If the nans are being produced in the backward pass of a gradient evaluation, when an exception is raised several frames up in the stack trace you will be in the backward_pass function, which is essentially a simple jaxpr interpreter that walks the sequence of primitive operations in reverse. In the example below, we started an ipython repl with the command line `env JAX_DEBUG_NANS=True ipython`, then ran this:

+++ {"id": "p6ZtDHPbBa_W"}

```
In [1]: import jax.numpy as jnp

In [2]: jnp.divide(0., 0.)
---------------------------------------------------------------------------
FloatingPointError                        Traceback (most recent call last)
<ipython-input-2-f2e2c413b437> in <module>()
----> 1 jnp.divide(0., 0.)

.../jax/jax/numpy/lax_numpy.pyc in divide(x1, x2)
    343     return floor_divide(x1, x2)
    344   else:
--> 345     return true_divide(x1, x2)
    346
    347

.../jax/jax/numpy/lax_numpy.pyc in true_divide(x1, x2)
    332   x1, x2 = _promote_shapes(x1, x2)
    333   return lax.div(lax.convert_element_type(x1, result_dtype),
--> 334                  lax.convert_element_type(x2, result_dtype))
    335
    336

.../jax/jax/lax.pyc in div(x, y)
    244 def div(x, y):
    245   r"""Elementwise division: :math:`x \over y`."""
--> 246   return div_p.bind(x, y)
    247
    248 def rem(x, y):

... stack trace ...

.../jax/jax/interpreters/xla.pyc in handle_result(device_buffer)
    103         py_val = device_buffer.to_py()
    104         if np.any(np.isnan(py_val)):
--> 105           raise FloatingPointError("invalid value")
    106         else:
    107           return Array(device_buffer, *result_shape)

FloatingPointError: invalid value
```

+++ {"id": "_NCnVt_GBa_W"}

The nan generated was caught. By running `%debug`, we can get a post-mortem debugger. This also works with functions under `@jit`, as the example below shows.

+++ {"id": "pf8RF6eiBa_W"}

```
In [4]: from jax import jit

In [5]: @jit
   ...: def f(x, y):
   ...:     a = x * y
   ...:     b = (x + y) / (x - y)
   ...:     c = a + 2
   ...:     return a + b * c
   ...:

In [6]: x = jnp.array([2., 0.])

In [7]: y = jnp.array([3., 0.])

In [8]: f(x, y)
Invalid value encountered in the output of a jit function. Calling the de-optimized version.
---------------------------------------------------------------------------
FloatingPointError                        Traceback (most recent call last)
<ipython-input-8-811b7ddb3300> in <module>()
----> 1 f(x, y)

 ... stack trace ...

<ipython-input-5-619b39acbaac> in f(x, y)
      2 def f(x, y):
      3     a = x * y
----> 4     b = (x + y) / (x - y)
      5     c = a + 2
      6     return a + b * c

.../jax/jax/numpy/lax_numpy.pyc in divide(x1, x2)
    343     return floor_divide(x1, x2)
    344   else:
--> 345     return true_divide(x1, x2)
    346
    347

.../jax/jax/numpy/lax_numpy.pyc in true_divide(x1, x2)
    332   x1, x2 = _promote_shapes(x1, x2)
    333   return lax.div(lax.convert_element_type(x1, result_dtype),
--> 334                  lax.convert_element_type(x2, result_dtype))
    335
    336

.../jax/jax/lax.pyc in div(x, y)
    244 def div(x, y):
    245   r"""Elementwise division: :math:`x \over y`."""
--> 246   return div_p.bind(x, y)
    247
    248 def rem(x, y):

 ... stack trace ...
```

+++ {"id": "6ur2yArDBa_W"}

When this code sees a nan in the output of an `@jit` function, it calls into the de-optimized code, so we still get a clear stack trace. And we can run a post-mortem debugger with `%debug` to inspect all the values to figure out the error.

⚠️ You shouldn't have the NaN-checker on if you're not debugging, as it can introduce lots of device-host round-trips and performance regressions!

⚠️ The NaN-checker doesn't work with `pmap`. To debug nans in `pmap` code, one thing to try is replacing `pmap` with `vmap`.

+++ {"id": "YTktlwTTMgFl"}

## 🔪 Double (64bit) precision

At the moment, JAX by default enforces single-precision numbers to mitigate the Numpy API's tendency to aggressively promote operands to `double`.  This is the desired behavior for many machine-learning applications, but it may catch you by surprise!

```{code-cell} ipython3
:id: CNNGtzM3NDkO
:outputId: b422bb23-a784-44dc-f8c9-57f3b6c861b8

x = random.uniform(random.key(0), (1000,), dtype=jnp.float64)
x.dtype
```

+++ {"id": "VcvqzobxNPbd"}

To use double-precision numbers, you need to set the `jax_enable_x64` configuration variable __at startup__.

There are a few ways to do this:

1. You can enable 64-bit mode by setting the environment variable `JAX_ENABLE_X64=True`.

2. You can manually set the `jax_enable_x64` configuration flag at startup:

   ```python
   # again, this only works on startup!
   import jax
   jax.config.update("jax_enable_x64", True)
   ```

3. You can parse command-line flags with `absl.app.run(main)`

   ```python
   import jax
   jax.config.config_with_absl()
   ```

4. If you want JAX to run absl parsing for you, i.e. you don't want to do `absl.app.run(main)`, you can instead use

   ```python
   import jax
   if __name__ == '__main__':
     # calls jax.config.config_with_absl() *and* runs absl parsing
     jax.config.parse_flags_with_absl()
   ```

Note that #2-#4 work for _any_ of JAX's configuration options.

We can then confirm that `x64` mode is enabled:

```{code-cell} ipython3
:id: HqGbBa9Rr-2g
:outputId: 5aa72952-08cc-4569-9b51-a10311ae9e81

import jax.numpy as jnp
from jax import random
x = random.uniform(random.key(0), (1000,), dtype=jnp.float64)
x.dtype # --> dtype('float64')
```

+++ {"id": "6Cks2_gKsXaW"}

### Caveats
⚠️ XLA doesn't support 64-bit convolutions on all backends!

+++ {"id": "WAHjmL0E2XwO"}

## 🔪 Miscellaneous Divergences from NumPy

While `jax.numpy` makes every attempt to replicate the behavior of numpy's API, there do exist corner cases where the behaviors differ.
Many such cases are discussed in detail in the sections above; here we list several other known places where the APIs diverge.

- For binary operations, JAX's type promotion rules differ somewhat from those used by NumPy. See [Type Promotion Semantics](https://jax.readthedocs.io/en/latest/type_promotion.html) for more details.
- When performing unsafe type casts (i.e. casts in which the target dtype cannot represent the input value), JAX's behavior may be backend dependent, and in general may diverge from NumPy's behavior. Numpy allows control over the result in these scenarios via the `casting` argument (see [`np.ndarray.astype`](https://numpy.org/devdocs/reference/generated/numpy.ndarray.astype.html)); JAX does not provide any such configuration, instead directly inheriting the behavior of [XLA:ConvertElementType](https://www.tensorflow.org/xla/operation_semantics#convertelementtype).

  Here is an example of an unsafe cast with differing results between NumPy and JAX:
  ```python
  >>> np.arange(254.0, 258.0).astype('uint8')
  array([254, 255,   0,   1], dtype=uint8)

  >>> jnp.arange(254.0, 258.0).astype('uint8')
  Array([254, 255, 255, 255], dtype=uint8)

  ```
  This sort of mismatch would typically arise when casting extreme values from floating to integer types or vice versa.


## Fin.

If something's not covered here that has caused you weeping and gnashing of teeth, please let us know and we'll extend these introductory _advisos_!

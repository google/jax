(jep-9263)=
# JEP 9263: Typed keys & pluggable RNGs

*Jake VanderPlas, Roy Frostig*

*August 2023*

## Overview
Going forward, RNG keys in JAX will be more type-safe and customizable.
Rather than representing a single PRNG key by a length-2  `uint32` array,
it will be represented as a scalar array with a special RNG dtype that
satisfies `jnp.issubdtype(key.dtype, jax.dtypes.prng_key)`.

For now, old-style RNG keys can still be created with
{func}`jax.random.PRNGKey`:
```python
>>> key = jax.random.PRNGKey(0)
>>> key
Array([0, 0], dtype=uint32)
>>> key.shape
(2,)
>>> key.dtype
dtype('uint32')
```
Starting now, new-style RNG keys can be created with
{func}`jax.random.key`:
```
>>> key = jax.random.key(0)
>>> key
Array((), dtype=key<fry>) overlaying:
[0 0]
>>> key.shape
()
>>> key.dtype
key<fry>
```
This (scalar-shaped) array behaves the same as any other JAX array, except
that its element type is a key (and associated metadata). We can make
non-scalar key arrays as well, for example by applying {func}`jax.vmap` to
{func}`jax.random.key`:
```python
>>> key_arr = jax.vmap(jax.random.key)(jnp.arange(4))
>>> key_arr
Array((4,), dtype=key<fry>) overlaying:
[[0 0]
 [0 1]
 [0 2]
 [0 3]]
>>> key_arr.shape
(4,)
```
Aside from switching to a new constructor, most PRNG-related code should
continue to work as expected. You can continue to use keys in
{mod}`jax.random` APIs as before; for example:
```python
# split
new_key, subkey = jax.random.split(key)

# random number generation
data = jax.random.uniform(key, shape=(5,))
```
However, not all numerical operations work on key arrays. They now
intentionally raise errors:
```python
>>> key = key + 1
ValueError: dtype=key<fry> is not a valid dtype for JAX type promotion.
```
If for some reason you need to recover the underlying buffer
(the old-style key), you can do so with {func}`jax.random.key_data`:
```python
>>> jax.random.key_data(key)
Array([0, 0], dtype=uint32)
```
For old-style keys, {func}`~jax.random.key_data` is an identity operation.

## What does this mean for users?
For JAX users, this change does not require any code changes now, but we hope
that you will find the upgrade worthwhile and switch to using typed keys. To
try this out, replace uses of jax.random.PRNGKey() with jax.random.key(). This
may introduce breakages in your code that fall into one of a few categories:

- If your code performs unsafe/unsupported operations on keys (such as indexing,
  arithmetic, transposition, etc; see Type Safety section below), this change
  will catch it. You can update your code to avoid such unsupported operations,
  or use {func}`jax.random.key_data` and {func}`jax.random.wrap_key_data`
  to manipulate raw key buffers in an unsafe way.
- If your code includes explicit logic about `key.shape`, you may need to update
  this logic to account for the fact that the trailing key buffer dimension is
  no longer an explicit part of the shape.
- If your code includes explicit logic about `key.dtype`, you will need to
  upgrade it to use the new public APIs for reasoning about RNG dtypes, such as
  `dtypes.issubdtype(dtype, dtypes.prng_key)`.
- If you call a JAX-based library which does not yet handle typed PRNG keys, you
  can use `raw_key = jax.random.key_data(key)` for now to recover the raw buffer,
  but please keep a TODO to remove this once the downstream library supports
  typed RNG keys.

At some point in the future, we plan to deprecate {func}`jax.random.PRNGKey` and
require the use of {func}`jax.random.key`.

### Detecting new-style typed keys
To check whether an object is a new-style typed PRNG key, you can use
`jax.dtypes.issubdtype` or `jax.numpy.issubdtype`:
```python
>>> typed_key = jax.random.key(0)
>>> jax.dtypes.issubdtype(typed_key.dtype, jax.dtypes.prng_key)
True
>>> raw_key = jax.random.PRNGKey(0)
>>> jax.dtypes.issubdtype(raw_key.dtype, jax.dtypes.prng_key)
False
```

### Type annotations for PRNG Keys
The recommended type annotation for both old and new-style PRNG keys is `jax.Array`.
A PRNG key is distinguished from other arrays based on its `dtype`, and it is not
currently possible to specify dtypes of JAX arrays within a type annotation.
Previously it was possible to use `jax.random.KeyArray` or `jax.random.PRNGKeyArray`
as type annotations, but these have always been aliased to `Any` under type checking,
and so `jax.Array` has much more specificity.

*Note: `jax.random.KeyArray` and `jax.random.PRNGKeyArray` were deprecated in JAX
version 0.4.16, and removed in JAX version 0.4.24*.

### Notes for JAX library authors
If you maintain a JAX-based library, your users are also JAX users. Know that JAX
will continue to support "raw" old-style keys in {mod}`jax.random` for now, so
callers may expect them to remain accepted everywhere. If you prefer to require
new-style typed keys in your library, then you may want to enforce them with a
check along the following lines:
```python
from jax import dtypes

def ensure_typed_key_array(key: Array) -> Array:
  if dtypes.issubdtype(key.dtype, dtypes.prng_key):
    return key
  else:
    raise TypeError("New-style typed JAX PRNG keys required")
```

## Motivation
Two major motivating factors for this change are customizability and safety.

### Customizing PRNG implementations
JAX currently operates with a single, globally configured PRNG algorithm. A
PRNG key is a vector of unsigned 32-bit integers, which jax.random APIs consume
to produce pseudorandom streams. Any higher-rank uint32 array is interpreted as
an array of such key buffers, where the trailing dimension represents keys.

The drawbacks of this design became clearer as we introduced alternative PRNG
implementations, which must be selected by setting a global or local
configuration flag. Different PRNG implementations have different size key
buffers, and different algorithms for generating random bits. Determining this
behavior with a global flag is error-prone, especially when there is more than
one key implementation in use process-wide.

Our new approach is to carry the implementation as part of the PRNG key type,
i.e. with the element type of the key array. Using the new key API, here is an
example of generating pseudorandom values under the default threefry2x32
implementation (which is implemented in pure Python and compiled with JAX), and
under the non-default rbg implementation (which corresponds to a single XLA
random-bit generation operation):
```python
>>> key = jax.random.key(0, impl='threefry2x32')  # this is the default impl
>>> key
Array((), dtype=key<fry>) overlaying:
[0 0]
>>> jax.random.uniform(key, shape=(3,))
Array([0.9653214 , 0.31468165, 0.63302994], dtype=float32)

>>> key = jax.random.key(0, impl='rbg')
>>> key
Array((), dtype=key<rbg>) overlaying:
[0 0 0 0]
>>> jax.random.uniform(key, shape=(3,))
Array([0.39904642, 0.8805201 , 0.73571277], dtype=float32)
```

### Safe PRNG key use
PRNG keys are really only meant to support a few operations in principle,
namely key derivation (e.g. splitting) and random number generation. The PRNG
is designed to generate independent pseudorandom numbers, provided keys are
properly split and that every key is consumed once.

Code that manipulates or consumes key data in other ways often indicates an
accidental bug, and representing key arrays as raw uint32 buffers has allowed
for easy misuse along these lines. Here are a few example misuses that we've
encountered in the wild:

#### Key buffer indexing
Access to the underlying integer buffers makes it easy to try and derive keys
in non-standard ways, sometimes with unexpectedly bad consequences:

```{code-block} python
:class: red-background
# Incorrect
key = random.PRNGKey(999)
new_key = random.PRNGKey(key[1])  # identical to the original key!
```
```{code-block} python
:class: green-background
# Correct
key = random.PRNGKey(999)
key, new_key = random.split(key)
```
If this key were a new-style typed key made with `random.key(999)`, indexing
into the key buffer would error instead.

#### Key arithmetic

Key arithmetic is a similarly treacherous way to derive keys from other keys.
Deriving keys in a way that avoids {func}`jax.random.split` or
{func}`jax.random.fold_in` by manipulating key data directly produces a batch
of keys that—depending on the PRNG implementation—might then generate
correlated random numbers within the batch:
```{code-block} python
:class: red-background
# Incorrect
key = random.PRNGKey(0)
batched_keys = key + jnp.arange(10, dtype=key.dtype)[:, None]
```
```{code-block} python
:class: green-background
# Correct
key = random.PRNGKey(0)
batched_keys = random.split(key, 10)
```
New-style typed keys created with `random.key(0)` address this by disallowing
arithmetic operations on keys.

#### Inadvertent transposing of key buffers

With "raw" old-style key arrays, it's easy to accidentally swap batch (leading)
dimensions and key buffer (trailing) dimensions. Again this possibly results in
keys that produce correlated pseudorandomness. A pattern that we've seen over
time boils down to this:
```{code-block} python
:class: red-background
# Incorrect
keys = random.split(random.PRNGKey(0))
data = jax.vmap(random.uniform, in_axes=1)(keys)
```
```{code-block} python
:class: green-background
# Correct
keys = random.split(random.PRNGKey(0))
data = jax.vmap(random.uniform, in_axes=0)(keys)
```
The bug here is subtle. By mapping over `in_axes=1`, this code makes new keys by
combining a single element from each key buffer in the batch. The resulting
keys are different from one another, but are effectively "derived" in a
non-standard way. Again, the PRNG is not designed or tested to produce
independent random streams from such a key batch.

New-style typed keys created with `random.key(0)` address this by hiding the
buffer representation of individual keys, instead treating keys as opaque
elements of a key array. Key arrays have no trailing "buffer" dimension to
index, transpose, or map over.

#### Key reuse
Unlike state-based PRNG APIs like {mod}`numpy.random`, JAX's functional PRNG
does not implicitly update a key when it has been used.
```{code-block} python
:class: red-background
# Incorrect
key = random.PRNGKey(0)
x = random.uniform(key, (100,))
y = random.uniform(key, (100,))  # Identical values!
```
```{code-block} python
:class: green-background
# Correct
key = random.PRNGKey(0)
key1, key2 = random.split(random.key(0))
x = random.uniform(key1, (100,))
y = random.uniform(key2, (100,))
```
We're actively working on tools to detect and prevent unintended key reuse.
This is still work in progress, but it relies on typed key arrays. Upgrading
to typed keys now sets us up to introduce these safety features as we build
them out.

## Design of typed PRNG keys
Typed PRNG keys are implemented as an instance of extended dtypes within JAX,
of which the new PRNG dtypes are a sub-dtype.

### Extended dtypes
From the user perspective, an extended dtype dt has the following user-visible
properties:

- `jax.dtypes.issubdtype(dt, jax.dtypes.extended)` returns `True`: this is the
  public API that should be used to detect whether a dtype is an extended dtype.
- It has a class-level attribute `dt.type`, which returns a typeclass in the
  hierarchy of `numpy.generic`. This is analogous to how `np.dtype('int32').type`
  returns `numpy.int32`, which is not a dtype but rather a scalar type, and a
  subclass of `numpy.generic`.
- Unlike numpy scalar types, we do not allow instantiation of `dt.type` scalar
  objects: this is in accordance with JAX's decision to represent scalar values
  as zero-dimensional arrays.

From a non-public implementation perspective, an extended dtype has the
following properties:

- Its type is a subclass of the private base class `jax._src.dtypes.ExtendedDtype`,
  the non-public base class used for extended dtypes. An instance of
  `ExtendedDtype` is analogous to an instance of `np.dtype`, like
  `np.dtype('int32')`.
- It has a private `_rules` attribute which allows the dtype to define how it
  behaves under particular operations. For example,
  `jax.lax.full(shape, fill_value, dtype)` will delegate to
  `dtype._rules.full(shape, fill_value, dtype)` when `dtype` is an extended dtype.

Why introduce extended dtypes in generality, beyond PRNGs? We reuse this same
extended dtype mechanism elsewhere internally. For example, the
`jax._src.core.bint` object, a bounded integer type used for experimental work
on dynamic shapes, is another extended dtype. In recent JAX versions it satisfies
the properties above (See [jax/_src/core.py#L1789-L1802](https://github.com/google/jax/blob/jax-v0.4.14/jax/_src/core.py#L1789-L1802)).

### PRNG dtypes
PRNG dtypes are defined as a particular case of extended dtypes. Specifically,
this change introduces a new public scalar type class jax.dtypes.prng_key,
which has the following property:
```python
>>> jax.dtypes.issubdtype(jax.dtypes.prng_key, jax.dtypes.extended)
True
```
PRNG key arrays then have a dtype with the following properties:
```python
>>> key = jax.random.key(0)
>>> jax.dtypes.issubdtype(key.dtype, jax.dtypes.extended)
True
>>> jax.dtypes.issubdtype(key.dtype, jax.dtypes.prng_key)
True
```
And in addition to `key.dtype._rules` as outlined for extended dtypes in
general, PRNG dtypes define `key.dtype._impl`, which contains the metadata
that defines the PRNG implementation. The PRNG implementation is currently
defined by the non-public `jax._src.prng.PRNGImpl` class. For now, `PRNGImpl`
isn't meant to be a public API, but we might revisit this soon to allow for
fully custom PRNG implementations.

## Progress
Following is a non-comprehensive list of key Pull Requests implementing the
above design. The main tracking issue is {jax-issue}`9263`.

- Implement pluggable PRNG via `PRNGImpl`: {jax-issue}`#6899` 
- Implement `PRNGKeyArray`, without dtype: {jax-issue}`#11952`
- Add a “custom element” dtype property to `PRNGKeyArray` with `_rules`
  attribute: {jax-issue}`#12167`
- Rename “custom element type” to “opaque dtype”: {jax-issue}`#12170`
- Refactor `bint` to use the opaque dtype infrastructure: {jax-issue}`#12707`
- Add `jax.random.key` to create typed keys directly: {jax-issue}`#16086`
- Add `impl` argument to `key` and `PRNGKey`: {jax-issue}`#16589`
- Rename “opaque dtype” to  “extended dtype” & define `jax.dtypes.extended`:
  {jax-issue}`#16824`
- Introduce `jax.dtypes.prng_key` and unify PRNG dtype with Extended dtype:
  {jax-issue}`#16781`
- Add a `jax_legacy_prng_key` flag to support warning or erroring when using
  legacy (raw) PRNG keys: {jax-issue}`#17225`

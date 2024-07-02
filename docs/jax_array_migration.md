(jax-array-migration)=
# jax.Array migration

<!--* freshness: { reviewed: '2023-03-17' } *-->

**yashkatariya@**

## TL;DR

JAX switched its default array implementation to the new `jax.Array` as of version 0.4.1.
This guide explains the reasoning behind this, the impact it might have on your code,
and how to (temporarily) switch back to the old behavior.

### What’s going on?

`jax.Array` is a unified array type that subsumes `DeviceArray`, `ShardedDeviceArray`,
and `GlobalDeviceArray` types in JAX. The `jax.Array` type helps make parallelism a
core feature of JAX, simplifies and unifies JAX internals, and allows us to
unify jit and pjit. If your code doesn't mention `DeviceArray` vs
`ShardedDeviceArray` vs `GlobalDeviceArray`, no changes are needed. But code that
depends on details of these separate classes may need to be tweaked to work with
the unified jax.Array

After the migration is complete `jax.Array` will be the only type of array in
JAX.

This doc explains how to migrate existing codebases to `jax.Array`. For more information on using `jax.Array` and JAX parallelism APIs, see the [Distributed arrays and automatic parallelization](https://jax.readthedocs.io/en/latest/notebooks/Distributed_arrays_and_automatic_parallelization.html) tutorial.


### How to enable jax.Array?

You can enable `jax.Array` by:

*   setting the shell environment variable `JAX_ARRAY` to something true-like
    (e.g., `1`);
*   setting the boolean flag `jax_array` to something true-like if your code
    parses flags with absl;
*   using this statement at the top of your main file:

    ```
    import jax
    jax.config.update('jax_array', True)
    ```

### How do I know if jax.Array broke my code?

The easiest way to tell if `jax.Array` is responsible for any problems is to
disable `jax.Array` and see if the issues go away.

### How can I disable jax.Array for now?

Through **March 15, 2023** it will be possible to disable jax.Array by:

*   setting the shell environment variable `JAX_ARRAY` to something falsey
    (e.g., `0`);
*   setting the boolean flag `jax_array` to something falsey if your code parses
    flags with absl;
*   using this statement at the top of your main file:

    ```
    import jax
    jax.config.update('jax_array', False)
    ```

## Why create jax.Array?

Currently JAX has three types; `DeviceArray`, `ShardedDeviceArray` and
`GlobalDeviceArray`. `jax.Array` merges these three types and cleans up JAX’s
internals while adding new parallelism features.

We also introduce a new `Sharding` abstraction that describes how a logical
Array is physically sharded out across one or more devices, such as TPUs or
GPUs. The change also upgrades, simplifies and merges the parallelism features
of `pjit` into `jit`. Functions decorated with `jit` will be able to operate
over sharded arrays without copying data onto a single device.

Features you get with `jax.Array`:

*   C++ `pjit` dispatch path
*   Op-by-op parallelism (even if the array distributed across multiple devices
    across multiple hosts)
*   Simpler batch data parallelism with `pjit`/`jit`.
*   Ways to create `Sharding`s that are not necessarily consisting of a mesh and
    partition spec. Can fully utilize the flexibility of OpSharding if you want
    or any other Sharding that you want.
*   and many more

Example:

```
import jax
import jax.numpy as jnp
from jax.sharding import PartitionSpec as P
import numpy as np
x = jnp.arange(8)

# Let's say there are 8 devices in jax.devices()
mesh = jax.sharding.Mesh(np.array(jax.devices()).reshape(4, 2), ('x', 'y'))
sharding = jax.sharding.NamedSharding(mesh, P('x'))

sharded_x = jax.device_put(x, sharding)

# `matmul_sharded_x` and `sin_sharded_x` are sharded. `jit` is able to operate over a
# sharded array without copying data to a single device.
matmul_sharded_x = sharded_x @ sharded_x.T
sin_sharded_x = jnp.sin(sharded_x)

# Even jnp.copy preserves the sharding on the output.
copy_sharded_x = jnp.copy(sharded_x)

# double_out is also sharded
double_out = jax.jit(lambda x: x * 2)(sharded_x)
```

## What issues can arise when jax.Array is switched on?

### New public type named jax.Array

All `isinstance(..., jnp.DeviceArray)` or `isinstance(.., jax.xla.DeviceArray)`
and other variants of `DeviceArray` should be switched to using `isinstance(...,
jax.Array)`.

Since `jax.Array` can represent DA, SDA and GDA, you can differentiate those 3
types in `jax.Array` via:

*   `x.is_fully_addressable and len(x.sharding.device_set) == 1` -- this means
    that `jax.Array` is like a DA
*   `x.is_fully_addressable and (len(x.sharding.device_set) > 1` -- this means
    that `jax.Array` is like a SDA
*   `not x.is_fully_addressable` -- this means that `jax.Array` is like a GDA
    and spans across multiple processes

For `ShardedDeviceArray`, you can move `isinstance(...,
pxla.ShardedDeviceArray)` to `isinstance(..., jax.Array) and
x.is_fully_addressable and len(x.sharding.device_set) > 1`.

In general it is not possible to differentiate a `ShardedDeviceArray` on 1
device from any other kind of single-device Array.

### GDA’s API name changes

GDA’s `local_shards` and `local_data` have been deprecated.

Please use `addressable_shards` and `addressable_data` which are compatible with
`jax.Array` and `GDA`.

### Creating jax.Array

All JAX functions will output `jax.Array` when the `jax_array` flag is True. If
you were using `GlobalDeviceArray.from_callback` or `make_sharded_device_array`
or `make_device_array` functions to explicitly create the respective JAX data
types, you will need to switch them to use {func}`jax.make_array_from_callback`
or {func}`jax.make_array_from_single_device_arrays`.

**For GDA:**

`GlobalDeviceArray.from_callback(shape, mesh, pspec, callback)` can become
`jax.make_array_from_callback(shape, jax.sharding.NamedSharding(mesh, pspec), callback)`
in a 1:1 switch.

If you were using the raw GDA constructor to create GDAs, then do this:

`GlobalDeviceArray(shape, mesh, pspec, buffers)` can become 
`jax.make_array_from_single_device_arrays(shape, jax.sharding.NamedSharding(mesh, pspec), buffers)`

**For SDA:**

`make_sharded_device_array(aval, sharding_spec, device_buffers, indices)` can
become `jax.make_array_from_single_device_arrays(shape, sharding, device_buffers)`.

To decide what the sharding should be, it depends on why you were creating the
SDAs:

If it was created to give as an input to `pmap`, then sharding can be:
`jax.sharding.PmapSharding(devices, sharding_spec)`.

If it was created to give as an input
to `pjit`, then sharding can be `jax.sharding.NamedSharding(mesh, pspec)`.

### Breaking change for pjit after switching to jax.Array for host local inputs

**If you are exclusively using GDA arguments to pjit, you can skip this section!
🎉**

With `jax.Array` enabled, all inputs to `pjit` must be globally shaped. This is
a breaking change from the previous behavior where `pjit` would concatenate
process-local arguments into a global value; this concatenation no longer
occurs.

Why are we making this breaking change? Each array now says explicitly how its
local shards fit into a global whole, rather than leaving it implicit. The more
explicit representation also unlocks additional flexibility, for example the use
of non-contiguous meshes with `pjit` which can improve efficiency on some TPU
models.

Running **multi-process pjit computation** and passing host-local inputs when
`jax.Array` is enabled can lead to an error similar to this:

Example:

Mesh = `{'x': 2, 'y': 2, 'z': 2}` and host local input shape == `(4,)` and
pspec = `P(('x', 'y', 'z'))`

Since `pjit` doesn’t lift host local shapes to global shapes with `jax.Array`,
you get the following error:

Note: You will only see this error if your host local shape is smaller than the
shape of the mesh.

```
ValueError: One of pjit arguments was given the sharding of
NamedSharding(mesh={'x': 2, 'y': 2, 'chips': 2}, partition_spec=PartitionSpec(('x', 'y', 'chips'),)),
which implies that the global size of its dimension 0 should be divisible by 8,
but it is equal to 4
```

The error makes sense because you can't shard dimension 0, 8 ways when the value
on dimension `0` is `4`.

How can you migrate if you still pass host local inputs to `pjit`? We are
providing transitional APIs to help you migrate:

Note: You don't need these utilities if you run your pjitted computation on a
single process.

```
from jax.experimental import multihost_utils

global_inps = multihost_utils.host_local_array_to_global_array(
    local_inputs, mesh, in_pspecs)

global_outputs = pjit(f, in_shardings=in_pspecs,
                      out_shardings=out_pspecs)(global_inps)

local_outs = multihost_utils.global_array_to_host_local_array(
    global_outputs, mesh, out_pspecs)
```

`host_local_array_to_global_array` is a type cast that looks at a value with
only local shards and changes its local shape to the shape that `pjit` would
have previously assumed if that value was passed before the change.

Passing in fully replicated inputs i.e. same shape on each process with
`P(None)` as `in_axis_resources` is still supported. In this case you do not
have to use `host_local_array_to_global_array` because the shape is already
global.

```
key = jax.random.PRNGKey(1)

# As you can see, using host_local_array_to_global_array is not required since in_axis_resources says
# that the input is fully replicated via P(None)
pjit(f, in_shardings=None, out_shardings=None)(key)

# Mixing inputs
global_inp = multihost_utils.host_local_array_to_global_array(
    local_inp, mesh, P('data'))
global_out = pjit(f, in_shardings=(P(None), P('data')),
                  out_shardings=...)(key, global_inp)
```

### FROM_GDA and jax.Array

If you were using `FROM_GDA` in `in_axis_resources` argument to `pjit`, then
with `jax.Array` there is no need to pass anything to `in_axis_resources` as
`jax.Array` will follow **computation follows sharding** semantics.

For example:

```
pjit(f, in_shardings=FROM_GDA, out_shardings=...) can be replaced by pjit(f, out_shardings=...)
```

If you have PartitionSpecs mixed in with `FROM_GDA` for inputs like numpy
arrays, etc, then use `host_local_array_to_global_array` to convert them to
`jax.Array`.

For example:

If you had this:

```
pjitted_f = pjit(
    f, in_shardings=(FROM_GDA, P('x'), FROM_GDA, P(None)),
    out_shardings=...)
pjitted_f(gda1, np_array1, gda2, np_array2)
```

then you can replace it with:

```

pjitted_f = pjit(f, out_shardings=...)

array2, array3 = multihost_utils.host_local_array_to_global_array(
    (np_array1, np_array2), mesh, (P('x'), P(None)))

pjitted_f(array1, array2, array3, array4)
```

### live_buffers replaced with live_arrays

`live_buffers` attribute on jax `Device`
has been deprecated. Please use `jax.live_arrays()` instead which is compatible
with `jax.Array`.


### Handling of host local inputs to pjit like batch, etc

If you are passing host local inputs to `pjit` in a **multi-process
environment**, then please use
`multihost_utils.host_local_array_to_global_array` to convert the batch to a
global `jax.Array` and then pass that to `pjit`.

The most common example of such a host local input is a **batch of input data**.

This will work for any host local input (not just a batch of input data).

```
from jax.experimental import multihost_utils

batch = multihost_utils.host_local_array_to_global_array(
    batch, mesh, batch_partition_spec)
```

See the pjit section above for more details about this change and more examples.

### RecursionError: Recursively calling jit

This happens when some part of your code has `jax.Array` disabled and then you
enable it only for some other part. For example, if you use some third\_party
code which has `jax.Array` disabled and you get a `DeviceArray` from that
library and then you enable `jax.Array` in your library and pass that
`DeviceArray` to JAX functions, it will lead to a RecursionError.

This error should go away when `jax.Array` is enabled by default so that all
libraries return `jax.Array` unless they explicitly disable it.

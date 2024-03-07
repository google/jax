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

(sharded-computation)=
# Introduction to sharded computation

<!--* freshness: { reviewed: '2024-05-10' } *-->

This tutorial serves as an introduction to device parallelism for Single-Program Multi-Data (SPMD) code in JAX. SPMD is a parallelism technique where the same computation, such as the forward pass of a neural network, can be run on different input data (for example, different inputs in a batch) in parallel on different devices, such as several GPUs or Google TPUs.

The tutorial covers three modes of parallel computation:

- _Automatic parallelism via {func}`jax.jit`_: The compiler chooses the optimal computation strategy (a.k.a. "the compiler takes the wheel").
- _Semi-automated parallelism_ using {func}`jax.jit` and {func}`jax.lax.with_sharding_constraint`
- _Fully manual parallelism with manual control using {func}`jax.experimental.shard_map.shard_map`_: `shard_map` enables per-device code and explicit communication collectives

Using these schools of thought for SPMD, you can transform a function written for one device into a function that can run in parallel on multiple devices.

If you are running these examples in a Google Colab notebook, make sure that your hardware accelerator is the latest Google TPU by checking your notebook settings: **Runtime** > **Change runtime type** > **Hardware accelerator** > **TPU v2** (which provides eight devices to work with).

```{code-cell}
:outputId: 18905ae4-7b5e-4bb9-acb4-d8ab914cb456

import jax
jax.devices()
```

## Key concept: Data sharding

Key to all of the distributed computation approaches below is the concept of *data sharding*, which describes how data is laid out on the available devices.

How can JAX can understand how the data is laid out across devices? JAX's datatype, the {class}`jax.Array` immutable array data structure, represents arrays with physical storage spanning one or multiple devices, and helps make parallelism a core feature of JAX.  The {class}`jax.Array` object is designed with distributed data and computation in mind. Every `jax.Array` has an associated {mod}`jax.sharding.Sharding` object, which describes which shard of the global data is required by each global device. When you create a {class}`jax.Array` from scratch, you also need to create its `Sharding`.

In the simplest cases, arrays are sharded on a single device, as demonstrated below:

```{code-cell}
:outputId: 39fdbb79-d5c0-4ea6-8b20-88b2c502a27a

import jax.numpy as jnp
arr = jnp.arange(32.0).reshape(4, 8)
arr.devices()
```

```{code-cell}
:outputId: 536f773a-7ef4-4526-c58b-ab4d486bf5a1

arr.sharding
```

For a more visual representation of the storage layout, the {mod}`jax.debug` module provides some helpers to visualize the sharding of an array. For example, {func}`jax.debug.visualize_array_sharding` displays how the array is stored in memory of a single device:

```{code-cell}
:outputId: 74a793e9-b13b-4d07-d8ec-7e25c547036d

jax.debug.visualize_array_sharding(arr)
```

To create an array with a non-trivial sharding, you can define a {mod}`jax.sharding` specification for the array and pass this to {func}`jax.device_put`.

Here, define a {class}`~jax.sharding.NamedSharding`, which specifies an N-dimensional grid of devices with named axes, where {class}`jax.sharding.Mesh` allows for precise device placement:

```{code-cell}
:outputId: 0b397dba-3ddc-4aca-f002-2beab7e6b8a5

# Pardon the boilerplate; constructing a sharding will become easier in future!
from jax.sharding import Mesh
from jax.sharding import PartitionSpec
from jax.sharding import NamedSharding
from jax.experimental import mesh_utils

P = jax.sharding.PartitionSpec
devices = mesh_utils.create_device_mesh((2, 4))
mesh = jax.sharding.Mesh(devices, ('x', 'y'))
sharding = jax.sharding.NamedSharding(mesh, P('x', 'y'))
print(sharding)
```

Passing this `Sharding` object to {func}`jax.device_put`, you can obtain a sharded array:

```{code-cell}
:outputId: c8ceedba-05ca-4156-e6e4-1e98bb664a66

arr_sharded = jax.device_put(arr, sharding)

print(arr_sharded)
jax.debug.visualize_array_sharding(arr_sharded)
```

The device numbers here are not in numerical order, because the mesh reflects the underlying toroidal topology of the device.

## 1. Automatic parallelism via `jit`

Once you have sharded data, the easiest way to do parallel computation is to simply pass the data to a {func}`jax.jit`-compiled function! In JAX, you need to only specify how you want the input and output of your code to be partitioned, and the compiler will figure out how to: 1) partition everything inside; and 2) compile inter-device communications.

The XLA compiler behind `jit` includes heuristics for optimizing computations across multiple devices.
In the simplest of cases, those heuristics boil down to *computation follows data*.

To demonstrate how auto-parallelization works in JAX, below is an example that uses a {func}`jax.jit`-decorated staged-out function: it's a simple element-wise function, where the computation for each shard will be performed on the device associated with that shard, and the output is sharded in the same way:

```{code-cell}
:outputId: de46f86a-6907-49c8-f36c-ed835e78bc3d

@jax.jit
def f_elementwise(x):
  return 2 * jnp.sin(x) + 1

result = f_elementwise(arr_sharded)

print("shardings match:", result.sharding == arr_sharded.sharding)
```

As computations get more complex, the compiler makes decisions about how to best propagate the sharding of the data.

Here, you sum along the leading axis of `x`, and visualize how the result values are stored across multiple devices (with {func}`jax.debug.visualize_array_sharding`):

```{code-cell}
:outputId: 90c3b997-3653-4a7b-c8ff-12a270f11d02

@jax.jit
def f_contract(x):
  return x.sum(axis=0)

result = f_contract(arr_sharded)
jax.debug.visualize_array_sharding(result)
print(result)
```

The result is partially replicated: that is, the first two elements of the array are replicated on devices `0` and `6`, the second on `1` and `7`, and so on.

## 2. Semi-automated sharding with constraints

If you'd like to have some control over the sharding used within a particular computation, JAX offers the {func}`~jax.lax.with_sharding_constraint` function. You can use {func}`jax.lax.with_sharding_constraint` (in place of (func}`jax.device_put()`) together with {func}`jax.jit` for more control over how the compiler constraints how the intermediate values and outputs are distributed.

For example, suppose that within `f_contract` above, you'd prefer the output not to be partially-replicated, but rather to be fully sharded across the eight devices:

```{code-cell}
:outputId: 8468f5c6-76ca-4367-c9f2-93c723687cfd

@jax.jit
def f_contract_2(x):
  out = x.sum(axis=0)
  # mesh = jax.create_mesh((8,), 'x')
  devices = mesh_utils.create_device_mesh(8)
  mesh = jax.sharding.Mesh(devices, 'x')
  sharding = jax.sharding.NamedSharding(mesh, P('x'))
  return jax.lax.with_sharding_constraint(out, sharding)

result = f_contract_2(arr_sharded)
jax.debug.visualize_array_sharding(result)
print(result)
```

This gives you a function with the particular output sharding you'd like.

## 3. Manual parallelism with `shard_map`

In the automatic parallelism methods explored above, you can write a function as if you're operating on the full dataset, and `jit` will split that computation across multiple devices. By contrast, with {func}`jax.experimental.shard_map.shard_map` you write the function that will handle a single shard of data, and `shard_map` will construct the full function.

`shard_map` works by mapping a function across a particular *mesh* of devices (`shard_map` maps over shards). In the example below:

- As before, {class}`jax.sharding.Mesh` allows for precise device placement, with the axis names parameter for logical and physical axis names.
- The `in_specs` argument determines the shard sizes. The `out_specs` argument identifies how the blocks are assembled back together.

**Note:** {func}`jax.experimental.shard_map.shard_map` code can work inside {func}`jax.jit` if you need it.

```{code-cell}
:outputId: 435c32f3-557a-4676-c11b-17e6bab8c1e2

from jax.experimental.shard_map import shard_map
P = jax.sharding.PartitionSpec
mesh = jax.sharding.Mesh(jax.devices(), 'x')

f_elementwise_sharded = shard_map(
    f_elementwise,
    mesh=mesh,
    in_specs=P('x'),
    out_specs=P('x'))

arr = jnp.arange(32)
f_elementwise_sharded(arr)
```

The function you write only "sees" a single batch of the data, which you can check by printing the device local shape:

```{code-cell}
:outputId: 99a3dc6e-154a-4ef6-8eaa-3dd0b68fb1da

x = jnp.arange(32)
print(f"global shape: {x.shape=}")

def f(x):
  print(f"device local shape: {x.shape=}")
  return x * 2

y = shard_map(f, mesh=mesh, in_specs=P('x'), out_specs=P('x'))(x)
```

Because each of your functions only "sees" the device-local part of the data, it means that aggregation-like functions require some extra thought.

For example, here's what a `shard_map` of a {func}`jax.numpy.sum` looks like:

```{code-cell}
:outputId: 1e9a45f5-5418-4246-c75b-f9bc6dcbbe72

def f(x):
  return jnp.sum(x, keepdims=True)

shard_map(f, mesh=mesh, in_specs=P('x'), out_specs=P('x'))(x)
```

Your function `f` operates separately on each shard, and the resulting summation reflects this.

If you want to sum across shards, you need to explicitly request it using collective operations like {func}`jax.lax.psum`:

```{code-cell}
:outputId: 4fd29e80-4fee-42b7-ff80-29f9887ab38d

def f(x):
  sum_in_shard = x.sum()
  return jax.lax.psum(sum_in_shard, 'x')

shard_map(f, mesh=mesh, in_specs=P('x'), out_specs=P())(x)
```

Because the output no longer has a sharded dimension, set `out_specs=P()` (recall that the `out_specs` argument identifies how the blocks are assembled back together in `shard_map`).

## Comparing the three approaches

With these concepts fresh in our mind, let's compare the three approaches for a simple neural network layer.

Start by defining your canonical function like this:

```{code-cell}
:id: 1TdhfTsoiqS1

@jax.jit
def layer(x, weights, bias):
  return jax.nn.sigmoid(x @ weights + bias)
```

```{code-cell}
:outputId: f3007fe4-f6f3-454e-e7c5-3638de484c0a

import numpy as np
rng = np.random.default_rng(0)

x = rng.normal(size=(32,))
weights = rng.normal(size=(32, 4))
bias = rng.normal(size=(4,))

layer(x, weights, bias)
```

You can automatically run this in a distributed manner using {func}`jax.jit` and passing appropriately sharded data.

If you shard the leading axis of both `x` and `weights` in the same way, then the matrix multiplication will automatically happen in parallel:

```{code-cell}
:outputId: 80be899e-8dbc-4bfc-acd2-0f3d554a0aa5

P = jax.sharding.PartitionSpec
mesh = jax.sharding.Mesh(jax.devices(), 'x')
sharding = jax.sharding.NamedSharding(mesh, P('x'))

x_sharded = jax.device_put(x, sharding)
weights_sharded = jax.device_put(weights, sharding)

layer(x_sharded, weights_sharded, bias)
```

Alternatively, you can use {func}`jax.lax.with_sharding_constraint` in the function to automatically distribute unsharded inputs:

```{code-cell}
:outputId: bb63e8da-ff4f-4e95-f083-10584882daf4

@jax.jit
def layer_auto(x, weights, bias):
  x = jax.lax.with_sharding_constraint(x, sharding)
  weights = jax.lax.with_sharding_constraint(weights, sharding)
  return layer(x, weights, bias)

layer_auto(x, weights, bias)  # pass in unsharded inputs
```

Finally, you can do the same thing with `shard_map`, using {func}`jax.lax.psum` to indicate the cross-shard collective required for the matrix product:

```{code-cell}
:outputId: 568d1c85-39a7-4dba-f09a-0e4f7c2ea918

from functools import partial

@jax.jit
@partial(shard_map, mesh=mesh,
         in_specs=(P('x'), P('x', None), P(None)),
         out_specs=P(None))
def layer_sharded(x, weights, bias):
  return jax.nn.sigmoid(jax.lax.psum(x @ weights, 'x') + bias)

layer_sharded(x, weights, bias)
```

## Next steps

This tutorial serves as a brief introduction of sharded and parallel computation in JAX.

To learn about each SPMD method in-depth, check out these docs:
- {doc}`../notebooks/Distributed_arrays_and_automatic_parallelization`
- {doc}`../notebooks/shard_map`

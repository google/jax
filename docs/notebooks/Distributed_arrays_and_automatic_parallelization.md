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

+++ {"id": "PxHrg4Cjuapm"}

# Distributed arrays and automatic parallelization

<!--* freshness: { reviewed: '2024-04-16' } *-->

+++ {"id": "pFtQjv4SzHRj"}

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/google/jax/blob/main/docs/notebooks/Distributed_arrays_and_automatic_parallelization.ipynb) [![Open in Kaggle](https://kaggle.com/static/images/open-in-kaggle.svg)](https://kaggle.com/kernels/welcome?src=https://github.com/google/jax/blob/main/docs/notebooks/Distributed_arrays_and_automatic_parallelization.ipynb)

This tutorial discusses parallelism via `jax.Array`, the unified array object model available in JAX v0.4.1 and newer.

```{code-cell}
:id: FNxScTfq3vGF

import os

import functools
from typing import Optional

import numpy as np

import jax
import jax.numpy as jnp
```

+++ {"id": "eyHMwyEfQJcz"}

⚠️ WARNING: The notebook requires 8 devices to run.

```{code-cell}
:id: IZMLqOUV3vGG

if len(jax.local_devices()) < 8:
  raise Exception("Notebook requires 8 devices to run")
```

+++ {"id": "3f37ca93"}

## Intro and a quick example

By reading this tutorial notebook, you'll learn about `jax.Array`, a unified
datatype for representing arrays, even with physical storage spanning multiple
devices. You'll also learn about how using `jax.Array`s together with `jax.jit`
can provide automatic compiler-based parallelization.

Before we think step by step, here's a quick example.
First, we'll create a `jax.Array` sharded across multiple devices:

```{code-cell}
:id: Gf2lO4ii3vGG

from jax.experimental import mesh_utils
from jax.sharding import PositionalSharding
```

```{code-cell}
:id: q-XBTEoy3vGG

# Create a Sharding object to distribute a value across devices:
sharding = PositionalSharding(mesh_utils.create_device_mesh((8,)))
```

```{code-cell}
:id: vI39znW93vGH
:outputId: 3b518df8-5c29-4848-acc3-e41df939f30b

# Create an array of random values:
x = jax.random.normal(jax.random.key(0), (8192, 8192))
# and use jax.device_put to distribute it across devices:
y = jax.device_put(x, sharding.reshape(4, 2))
jax.debug.visualize_array_sharding(y)
```

+++ {"id": "jZ0ZY9Um9Jg4"}

Next, we'll apply a computation to it and visualize how the result values are
stored across multiple devices too:

```{code-cell}
:id: -qCnHZl83vGI
:outputId: 9da9c29e-ce88-4425-e1ec-e93e5bcf3106

z = jnp.sin(y)
jax.debug.visualize_array_sharding(z)
```

+++ {"id": "5qccVQoE9tEi"}

The evaluation of the `jnp.sin` application was automatically parallelized
across the devices on which the input values (and output values) are stored:

```{code-cell}
:id: _VTzN0r03vGI
:outputId: c9208010-984b-442b-d105-c8c6a3a010e6

# `x` is present on a single device
%timeit -n 5 -r 5 jnp.sin(x).block_until_ready()
```

```{code-cell}
:id: QuzhU1g63vGI
:outputId: d48fc76e-79a7-47b9-d392-b18a1c33c798

# `y` is sharded across 8 devices.
%timeit -n 5 -r 5 jnp.sin(y).block_until_ready()
```

+++ {"id": "xWknFQbQ-bzV"}

Now let's look at each of these pieces in more detail!


## `Sharding` describes how array values are laid out in memory across devices

+++ {"id": "W6HsXauGxL6w"}

### Sharding basics, and the `PositionalSharding` subclass

+++ {"id": "NWDyp_EjVHkg"}

To parallelize computation across multiple devices, we first must lay out input data across multiple devices.

In JAX, `Sharding` objects describe distributed memory layouts. They can be used with `jax.device_put` to produce a value with distributed layout.

For example, here's a value with a single-device `Sharding`:

```{code-cell}
:id: VmoX4SUp3vGJ

import jax
x = jax.random.normal(jax.random.key(0), (8192, 8192))
```

```{code-cell}
:id: vNRabO2J3vGJ
:outputId: 73db7b6e-c2e7-467d-a0ef-c35e29e582dd

jax.debug.visualize_array_sharding(x)
```

+++ {"id": "HhCjhK0zXIqX"}

Here, we're using the `jax.debug.visualize_array_sharding` function to show where the value `x` is stored in memory. All of `x` is stored on a single device, so the visualization is pretty boring!

But we can shard `x` across multiple devices by using `jax.device_put` and a `Sharding` object. First, we make a `numpy.ndarray` of `Devices` using `mesh_utils.create_device_mesh`, which takes hardware topology into account for the `Device` order:

```{code-cell}
:id: VUIEIzRp3vGK

from jax.experimental import mesh_utils
devices = mesh_utils.create_device_mesh((8,))
```

+++ {"id": "lbOKFWmBX1iv"}

Then, we create a `PositionalSharding` and use it with `device_put`:

```{code-cell}
:id: jwrWfZeB3vGK
:outputId: e6f126bd-f6bd-48c7-c130-6f02757e3342

from jax.sharding import PositionalSharding

sharding = PositionalSharding(devices)

x = jax.device_put(x, sharding.reshape(8, 1))
jax.debug.visualize_array_sharding(x)
```

+++ {"id": "TUu69IWXZdTm"}

Here `sharding` is a `PositionalSharding` which acts like an array with sets of devices as elements:

```{code-cell}
:id: zxWB82Kz3vGK
:outputId: 11384a6b-fabc-4c4c-bcad-a3be51eb0465

sharding
```

+++ {"id": "uRLpOcmNj_Vt"}

The device numbers here are not in numerical order, because the mesh reflects the underlying toroidal topology of the device.

By writing `PositionalSharding(ndarray_of_devices)`, we fix the device order and the initial shape. Then we can reshape it:

```{code-cell}
:id: PLsnpSzc3vGL
:outputId: 9f4db733-cafe-46ae-c057-dc31046a6f66

sharding.reshape(8, 1)
```

```{code-cell}
:id: iqKdI4LO3vGL
:outputId: 6aa10fc2-cec4-4401-a0df-343e71646e0a

sharding.reshape(4, 2)
```

+++ {"id": "KBu6WLfhm7ra"}

To use `device_put` with a data array `x`, we can reshape the `sharding` into a shape that is _congruent_ with `x.shape`, meaning a shape with the same length as `x.shape` and where each element evenly divides the corresponding element of `x.shape`:
```python
def is_congruent(x_shape: Sequence[int], sharding_shape: Sequence[int]) -> bool:
  return (len(x_shape) == len(sharding_shape) and
          all(d1 % d2 == 0 for d1, d2 in zip(x_shape, sharding_shape)))
```

For example, we can reshape `sharding` to have shape `(4, 2)`, then use it in a `device_put`:

```{code-cell}
:id: SELr4xNi3vGL
:outputId: b2f4acec-0cd3-4829-ca16-cae2e0e8ca60

sharding = sharding.reshape(4, 2)
print(sharding)
```

```{code-cell}
:id: 8IVIsqfX3vGL
:outputId: 033d0e02-a643-4f4c-9d24-9cd8465bc69a

y = jax.device_put(x, sharding)
jax.debug.visualize_array_sharding(y)
```

+++ {"id": "tyg9F-UIsU__"}

Here `y` represents the same _value_ as `x`, but its shards (i.e. slices) are stored in different devices' memories.

Different `PositionalSharding` shapes result in different distributed layouts (i.e. shardings) of the result:

```{code-cell}
:id: cCjt6QCz3vGM
:outputId: 4ad8a611-596d-424f-b6c5-fc00f1adc306

sharding = sharding.reshape(1, 8)
print(sharding)
```

```{code-cell}
:id: yTK4Nz3u3vGM
:outputId: e445c6bc-4fe3-4e9d-cc9e-d82858f58312

y = jax.device_put(x, sharding)
jax.debug.visualize_array_sharding(y)
```

+++ {"id": "0PuamOvXubcf"}

In some cases, we don't just want to store each slice of `x` in a single device's memory; we might want to _replicate_ some slices, meaning storing copies of a slice's values in multiple devices' memories.

With `PositionalSharding`, we can express replication by calling the reducer method `replicate`:

```{code-cell}
:id: _jr6XYKx3vGM
:outputId: 59c8b9a4-b8af-493a-ba8d-da5931e88f93

sharding = sharding.reshape(4, 2)
print(sharding.replicate(axis=0, keepdims=True))
```

```{code-cell}
:id: S5vzjFuH3vGN
:outputId: b6ce2675-7261-4e57-fa8c-b4e87abf7e52

y = jax.device_put(x, sharding.replicate(axis=0, keepdims=True))
jax.debug.visualize_array_sharding(y)
```

+++ {"id": "FzeP0kpTvJv-"}

Here the visualization shows that `x` is sharded two ways along its second dimension (and not sharded along the first dimension), and each of those shards is replicated four ways (i.e. stored in four device memories).

The `replicate` method is analogous to the familiar NumPy array reduction methods like `.sum()` and `.prod()`. It operates along an axis performing a set union. So if `sharding` has shape `(4, 2)`, then `sharding.replicate(0, keepdims=True)` has shape `(1, 2)`, and `sharding.replicate(1, keepdims=True)` has shape `(4, 1)`. Unlike analogous NumPy methods, `keepdims=True` is actually the default, so reduced-over axes aren't squeezed:

```{code-cell}
:id: DR7VV-6e3vGN
:outputId: f879fc2c-5723-4199-b306-295bc1b3681e

print(sharding.replicate(0).shape)
print(sharding.replicate(1).shape)
```

```{code-cell}
:id: agUtVUVx3vGN
:outputId: 0e9789ef-ce52-4ed6-8bd5-c876b95f66e6

y = jax.device_put(x, sharding.replicate(1))
jax.debug.visualize_array_sharding(y)
```

+++ {"id": "D31t5POXxHHJ"}

### `NamedSharding` gives a way to express shardings with names

+++ {"id": "ayMKWeTmxl-X"}

So far we've worked with `PositionalSharding`, but there are alternative ways to express shardings. In fact, `Sharding` is an interface, and any class that implements that interface can be used with functions like `device_put`.

Another convenient way to express sharding is with the `NamedSharding`:

```{code-cell}
:id: zpB1JxyK3vGN
:outputId: 46d5da37-840c-49d8-8380-a162811bae8a

from jax.sharding import Mesh
from jax.sharding import PartitionSpec
from jax.sharding import NamedSharding
from jax.experimental import mesh_utils

P = PartitionSpec

devices = mesh_utils.create_device_mesh((4, 2))
mesh = Mesh(devices, axis_names=('a', 'b'))
y = jax.device_put(x, NamedSharding(mesh, P('a', 'b')))
jax.debug.visualize_array_sharding(y)
```

+++ {"id": "OW_Cc92G1-nr"}

We can define a helper function to make things simpler:

```{code-cell}
:id: 8g0Md2Gd3vGO

devices = mesh_utils.create_device_mesh((4, 2))
default_mesh = Mesh(devices, axis_names=('a', 'b'))

def mesh_sharding(
    pspec: PartitionSpec, mesh: Optional[Mesh] = None,
  ) -> NamedSharding:
  if mesh is None:
    mesh = default_mesh
  return NamedSharding(mesh, pspec)
```

```{code-cell}
:id: zp3MfS4Y3vGO
:outputId: 2c2f7201-c2c1-49e5-f8a5-0730c124d89a

y = jax.device_put(x, mesh_sharding(P('a', 'b')))
jax.debug.visualize_array_sharding(y)
```

+++ {"id": "xZ88riVm1mv5"}

Here, we use `P('a', 'b')` to express that the first and second axes of `x` should be sharded over the device mesh axes `'a'` and `'b'`, respectively. We can easily switch to `P('b', 'a')` to shard the axes of `x` over different devices:

```{code-cell}
:id: FigK5Zsa3vGO
:outputId: eca784e8-33fe-4e9b-a41d-21e9ee781a35

y = jax.device_put(x, mesh_sharding(P('b', 'a')))
jax.debug.visualize_array_sharding(y)
```

```{code-cell}
:id: hI-HD0xN3vGO
:outputId: c3e7dc3c-4048-448a-ef0b-50683532fcdc

# This `None` means that `x` is not sharded on its second dimension,
# and since the Mesh axis name 'b' is not mentioned, shards are
# replicated across it.
y = jax.device_put(x, mesh_sharding(P('a', None)))
jax.debug.visualize_array_sharding(y)
```

+++ {"id": "AqcAsNUgXCZz"}

Here, because `P('a', None)` doesn't mention the `Mesh` axis name `'b'`, we get replication over the axis `'b'`. The `None` here is just acting as a placeholder to line up against the second axis of the value `x`, without expressing sharding over any mesh axis. (As a shorthand, trailing `None`s can be omitted, so that `P('a', None)` means the same thing as `P('a')`. But it doesn't hurt to be explicit!)

To shard only over the second axis of `x`, we can use a `None` placeholder in the `PartitionSpec`:

```{code-cell}
:id: EXBExMQC3vGP
:outputId: fe1c8d7e-3345-4438-b9d2-780e7854b4eb

y = jax.device_put(x, mesh_sharding(P(None, 'b')))
jax.debug.visualize_array_sharding(y)
```

```{code-cell}
:id: PjUpG8uz3vGP
:outputId: 64d8224d-15d9-4ad4-d613-f7f85b1dc1af

y = jax.device_put(x, mesh_sharding(P(None, 'a')))
jax.debug.visualize_array_sharding(y)
```

+++ {"id": "--AZgW1P3HFT"}

For a fixed mesh, we can even partition one logical axis of `x` over multiple device mesh axes:

```{code-cell}
:id: fVcPbDUA3vGP
:outputId: 7f524ba5-a6d8-4490-cda9-685ad11416f9

y = jax.device_put(x, mesh_sharding(P(('a', 'b'), None)))
jax.debug.visualize_array_sharding(y)
```

+++ {"id": "c1tTFudr3Ae7"}

Using `NamedSharding` makes it easy to define a device mesh once and give its axes names, then just refer to those names in `PartitionSpec`s for each `device_put` as needed.

+++ {"id": "rhWzHgGf4mkg"}

## Computation follows data sharding and is automatically parallelized

+++ {"id": "JukoaRhl4tXJ"}

With sharded input data, the compiler can give us parallel computation. In particular, functions decorated with `jax.jit` can operate over sharded arrays without copying data onto a single device. Instead, computation follows sharding: based on the sharding of the input data, the compiler decides shardings for intermediates and output values, and parallelizes their evaluation, even inserting communication operations as necessary.

For example, the simplest computation is an elementwise one:

```{code-cell}
:id: _EmQwggc3vGQ

from jax.experimental import mesh_utils
from jax.sharding import PositionalSharding
sharding = PositionalSharding(mesh_utils.create_device_mesh((8,)))
```

```{code-cell}
:id: LnT0vWjc3vGQ
:outputId: 8089effc-aa4c-49e3-dd19-7064881dbad0

x = jax.device_put(x, sharding.reshape(4, 2))
print('input sharding:')
jax.debug.visualize_array_sharding(x)

y = jnp.sin(x)
print('output sharding:')
jax.debug.visualize_array_sharding(y)
```

+++ {"id": "7tY2gVRfazaT"}

Here for the elementwise operation `jnp.sin` the compiler chose the output sharding to be the same as the input. Moreover, the compiler automatically parallelized the computation, so that each device computed its output shard from its input shard in parallel.

In other words, even though we wrote the `jnp.sin` computation as if a single machine were to execute it, the compiler splits up the computation for us and executes it on multiple devices.

We can do the same for more than just elementwise operations too. Consider a matrix multiplication with sharded inputs:

```{code-cell}
:id: Dq043GkP3vGQ
:outputId: 350219a8-1e4a-4404-fe14-50f97ea3e7ba

y = jax.device_put(x, sharding.reshape(4, 2).replicate(1))
z = jax.device_put(x, sharding.reshape(4, 2).replicate(0))
print('lhs sharding:')
jax.debug.visualize_array_sharding(y)
print('rhs sharding:')
jax.debug.visualize_array_sharding(z)

w = jnp.dot(y, z)
print('out sharding:')
jax.debug.visualize_array_sharding(w)
```

+++ {"id": "_EPNaWzgazft"}

Here the compiler chose the output sharding so that it could maximally parallelize the computation: without needing communication, each device already has the input shards it needs to compute its output shard.

How can we be sure it's actually running in parallel? We can do a simple timing experiment:

```{code-cell}
:id: QjQ5u8qh3vGQ
:outputId: bd29edcd-b87c-486e-c568-906f06ae16be

x_single = jax.device_put(x, jax.devices()[0])
jax.debug.visualize_array_sharding(x_single)
```

```{code-cell}
:id: 8tn8lOj73vGR
:outputId: 5809b3c8-7333-4cd3-db97-a7aede943dce

np.allclose(jnp.dot(x_single, x_single),
            jnp.dot(y, z))
```

```{code-cell}
:id: D7PpZwhR3vGR
:outputId: 4f0bd43d-0b32-4089-d3da-c8f1449e3526

%timeit -n 5 -r 5 jnp.dot(x_single, x_single).block_until_ready()
```

```{code-cell}
:id: rgo_yVHF3vGR
:outputId: 97f19052-f1c9-4d30-f453-07b3a7208aa9

%timeit -n 5 -r 5 jnp.dot(y, z).block_until_ready()
```

+++ {"id": "gglQIMXJnnJw"}

Even copying a sharded `Array` produces a result with the sharding of the input:

```{code-cell}
:id: f1Zw-2lH3vGR
:outputId: a796bed4-07b0-497d-8fd8-31a22ab9762e

w_copy = jnp.copy(w)
jax.debug.visualize_array_sharding(w_copy)
```

+++ {"id": "3qfPjJdhgerc"}

So computation follows data placement: when we explicitly shard data with `jax.device_put`, and apply functions to that data, the compiler attempts to parallelize the computation and decide the output sharding. This policy for sharded data is a generalization of [JAX's policy of following explicit device placement](https://jax.readthedocs.io/en/latest/faq.html#controlling-data-and-computation-placement-on-devices).

+++ {"id": "QRB95LaWuT80"}

### When explicit shardings disagree, JAX errors

But what if two arguments to a computation are explicitly placed on different sets of devices, or with incompatible device orders?
In these ambiguous cases, an error is raised:

```{code-cell}
:id: 1vAkZAOY3vGR

import textwrap
from termcolor import colored

def print_exception(e):
  name = colored(f'{type(e).__name__}', 'red')
  print(textwrap.fill(f'{name}: {str(e)}'))
```

```{code-cell}
:id: DHh0N3vn3vGS
:outputId: e7741882-0ebf-4237-e5d1-e48c9b9c178f

sharding1 = PositionalSharding(jax.devices()[:4])
sharding2 = PositionalSharding(jax.devices()[4:])

y = jax.device_put(x, sharding1.reshape(2, 2))
z = jax.device_put(x, sharding2.reshape(2, 2))
try: y + z
except ValueError as e: print_exception(e)
```

```{code-cell}
:id: Im7DkoOl3vGS
:outputId: 3adfe1cb-db52-4a9d-e98e-62c6455c3100

devices = jax.devices()
permuted_devices = [devices[i] for i in [0, 1, 2, 3, 6, 7, 4, 5]]

sharding1 = PositionalSharding(devices)
sharding2 = PositionalSharding(permuted_devices)

y = jax.device_put(x, sharding1.reshape(4, 2))
z = jax.device_put(x, sharding2.reshape(4, 2))
try: y + z
except ValueError as e: print_exception(e)
```

+++ {"id": "6ZYcK8eXrn0p"}

We say arrays that have been explicitly placed or sharded with `jax.device_put` are _committed_ to their device(s), and so won't be automatically moved. See the [device placement FAQ](https://jax.readthedocs.io/en/latest/faq.html#controlling-data-and-computation-placement-on-devices) for more information.

When arrays are _not_ explicitly placed or sharded with `jax.device_put`, they are placed _uncommitted_ on the default device.
Unlike committed arrays, uncommitted arrays can be moved and resharded automatically: that is, uncommitted arrays can be arguments to a computation even if other arguments are explicitly placed on different devices.

For example, the output of `jnp.zeros`, `jnp.arange`, and `jnp.array` are uncommitted:

```{code-cell}
:id: _QvtKL8r3vGS
:outputId: e0078805-bdfd-436e-f94f-7cd256d2574f

y = jax.device_put(x, sharding1.reshape(4, 2))
y + jnp.ones_like(y)
y + jnp.arange(y.size).reshape(y.shape)
print('no error!')
```

+++ {"id": "dqMKl79NaIWF"}

## Constraining shardings of intermediates in `jit`ted code

+++ {"id": "g4LrDDcJwkHc"}

While the compiler will attempt to decide how a function's intermediate values and outputs should be sharded, we can also give it hints using `jax.lax.with_sharding_constraint`. Using `jax.lax.with_sharding_constraint` is much like `jax.device_put`, except we use it inside staged-out (i.e. `jit`-decorated) functions:

```{code-cell}
:id: jniSFm5V3vGT

sharding = PositionalSharding(mesh_utils.create_device_mesh((8,)))
```

```{code-cell}
:id: Q1wuDp-L3vGT

x = jax.random.normal(jax.random.key(0), (8192, 8192))
x = jax.device_put(x, sharding.reshape(4, 2))
```

```{code-cell}
:id: rqEDj0wB3vGT

@jax.jit
def f(x):
  x = x + 1
  y = jax.lax.with_sharding_constraint(x, sharding.reshape(2, 4))
  return y
```

```{code-cell}
:id: zYFS-n4r3vGT
:outputId: d23a7938-cb7d-44b4-b9c7-83edf1d1145e

jax.debug.visualize_array_sharding(x)
y = f(x)
jax.debug.visualize_array_sharding(y)
```

```{code-cell}
:id: 8g_2Y8wp3vGT

@jax.jit
def f(x):
  x = x + 1
  y = jax.lax.with_sharding_constraint(x, sharding.replicate())
  return y
```

```{code-cell}
:id: AiRFtVsR3vGT
:outputId: f3e28a70-46cf-46fb-c801-82f0ddb447e4

jax.debug.visualize_array_sharding(x)
y = f(x)
jax.debug.visualize_array_sharding(y)
```

+++ {"id": "_Y1P5wLTzJSz"}

By adding `with_sharding_constraint`, we've constrained the sharding of the output. In addition to respecting the annotation on a particular intermediate, the compiler will use annotations to decide shardings for other values.

It's often a good practice to annotate the outputs of computations, for example based on how the values are ultimately consumed.

+++ {"id": "QUkXWG-baMUs"}

## Examples: neural networks

+++ {"id": "g7y0OJBSGoSW"}

**⚠️ WARNING: The following is meant to be a simple demonstration of automatic sharding propagation with `jax.Array`, but it may not reflect best practices for real examples.** For instance, real examples may require more use of `with_sharding_constraint`.

+++ {"id": "3ii_UPkG3gzP"}

We can use `jax.device_put` and `jax.jit`'s computation-follows-sharding features to parallelize computation in neural networks. Here are some simple examples, based on this basic neural network:

```{code-cell}
:id: mEKF3zIF3vGU

import jax
import jax.numpy as jnp
```

```{code-cell}
:id: Mocs3oGe3vGU

def predict(params, inputs):
  for W, b in params:
    outputs = jnp.dot(inputs, W) + b
    inputs = jnp.maximum(outputs, 0)
  return outputs

def loss(params, batch):
  inputs, targets = batch
  predictions = predict(params, inputs)
  return jnp.mean(jnp.sum((predictions - targets)**2, axis=-1))
```

```{code-cell}
:id: glBB8tzW3vGU

loss_jit = jax.jit(loss)
gradfun = jax.jit(jax.grad(loss))
```

```{code-cell}
:id: R0x62AIa3vGU

def init_layer(key, n_in, n_out):
    k1, k2 = jax.random.split(key)
    W = jax.random.normal(k1, (n_in, n_out)) / jnp.sqrt(n_in)
    b = jax.random.normal(k2, (n_out,))
    return W, b

def init_model(key, layer_sizes, batch_size):
    key, *keys = jax.random.split(key, len(layer_sizes))
    params = list(map(init_layer, keys, layer_sizes[:-1], layer_sizes[1:]))

    key, *keys = jax.random.split(key, 3)
    inputs = jax.random.normal(keys[0], (batch_size, layer_sizes[0]))
    targets = jax.random.normal(keys[1], (batch_size, layer_sizes[-1]))

    return params, (inputs, targets)

layer_sizes = [784, 8192, 8192, 8192, 10]
batch_size = 8192

params, batch = init_model(jax.random.key(0), layer_sizes, batch_size)
```

+++ {"id": "sJv_h0AS2drh"}

### 8-way batch data parallelism

```{code-cell}
:id: _Q5NbdOn3vGV

sharding = PositionalSharding(jax.devices()).reshape(8, 1)
```

```{code-cell}
:id: 3KC6ieEe3vGV

batch = jax.device_put(batch, sharding)
params = jax.device_put(params, sharding.replicate())
```

```{code-cell}
:id: MUb-QE2b3vGV
:outputId: 1f831ea5-5a30-49ad-8195-977ff7ed476a

loss_jit(params, batch)
```

```{code-cell}
:id: HUkw0u413vGV
:outputId: dfa2599c-9440-4657-9035-0dc3bbf625e1

step_size = 1e-5

for _ in range(30):
  grads = gradfun(params, batch)
  params = [(W - step_size * dW, b - step_size * db)
            for (W, b), (dW, db) in zip(params, grads)]

print(loss_jit(params, batch))
```

```{code-cell}
:id: paCw6Zaj3vGV
:outputId: 8ab1c32c-f2b1-465c-df71-f5a599e7f19e

%timeit -n 5 -r 5 gradfun(params, batch)[0][0].block_until_ready()
```

```{code-cell}
:id: BF86UWpg3vGV

batch_single = jax.device_put(batch, jax.devices()[0])
params_single = jax.device_put(params, jax.devices()[0])
```

```{code-cell}
:id: Z1wgUKXk3vGV
:outputId: 74df8892-c349-41dc-cb1b-e0843ec5c994

%timeit -n 5 -r 5 gradfun(params_single, batch_single)[0][0].block_until_ready()
```

+++ {"id": "3AjeeB7B4NP6"}

### 4-way batch data parallelism and 2-way model tensor parallelism

```{code-cell}
:id: N5-zzgW03vGW

sharding = sharding.reshape(4, 2)
```

```{code-cell}
:id: sgIWCjJK3vGW
:outputId: b2fdc556-05cc-4e68-fa04-48643d194dee

batch = jax.device_put(batch, sharding.replicate(1))
jax.debug.visualize_array_sharding(batch[0])
jax.debug.visualize_array_sharding(batch[1])
```

```{code-cell}
:id: BqCjYCgg3vGW

(W1, b1), (W2, b2), (W3, b3), (W4, b4) = params

W1 = jax.device_put(W1, sharding.replicate())
b1 = jax.device_put(b1, sharding.replicate())

W2 = jax.device_put(W2, sharding.replicate(0))
b2 = jax.device_put(b2, sharding.replicate(0))

W3 = jax.device_put(W3, sharding.replicate(0).T)
b3 = jax.device_put(b3, sharding.replicate())

W4 = jax.device_put(W4, sharding.replicate())
b4 = jax.device_put(b4, sharding.replicate())

params = (W1, b1), (W2, b2), (W3, b3), (W4, b4)
```

```{code-cell}
:id: _lSJ63sh3vGW
:outputId: 5b37aa8b-3226-4805-8282-876e8d06edda

jax.debug.visualize_array_sharding(W2)
```

```{code-cell}
:id: fxkfWYkk3vGW
:outputId: 8a1063c3-540b-47c1-d990-a6845da861f7

jax.debug.visualize_array_sharding(W3)
```

```{code-cell}
:id: uPCVs-_k3vGW
:outputId: de01cdfc-36cb-4823-c692-22c692ef4220

print(loss_jit(params, batch))
```

```{code-cell}
:id: L9JebLK_3vGW

step_size = 1e-5

for _ in range(30):
    grads = gradfun(params, batch)
    params = [(W - step_size * dW, b - step_size * db)
              for (W, b), (dW, db) in zip(params, grads)]
```

```{code-cell}
:id: c9Sbl69e3vGX
:outputId: 8272c5fa-e59f-4953-c2d5-658c42a28712

print(loss_jit(params, batch))
```

```{code-cell}
:id: lkAF0dAb3vGX
:outputId: acf0df31-c5e1-4683-b73f-b0cd1b0929f8

(W1, b1), (W2, b2), (W3, b3), (W4, b4) = params
jax.debug.visualize_array_sharding(W2)
jax.debug.visualize_array_sharding(W3)
```

```{code-cell}
:id: I1Npor3i3vGX
:outputId: 4099f6dd-7b46-4123-c1cb-5173c3d3278e

%timeit -n 10 -r 10 gradfun(params, batch)[0][0].block_until_ready()
```

+++ {"id": "3diqi5VRBy6S"}

## Sharp bits

+++ {"id": "OTfoXNnxFYDJ"}

### Generating random numbers

JAX comes with a functional, deterministic [random number generator](https://jax.readthedocs.io/en/latest/jep/263-prng.html). It underlies the various sampling functions in the [`jax.random` module](https://jax.readthedocs.io/en/latest/jax.random.html), such as `jax.random.uniform`.

JAX's random numbers are produced by a counter-based PRNG, so in principle, random number generation should be a pure map over counter values. A pure map is a trivially partitionable operation in principle. It should require no cross-device communication, nor any redundant computation across devices.

However, the existing stable RNG implementation is not automatically partitionable, for historical reasons.

+++ {"id": "ht_zYFVXNrjN"}

Consider the following example, where a function draws random uniform numbers and adds them to the input, elementwise:

```{code-cell}
:id: kwS-aQE_3vGX

@jax.jit
def f(key, x):
  numbers = jax.random.uniform(key, x.shape)
  return x + numbers

key = jax.random.key(42)
x_sharding = jax.sharding.PositionalSharding(jax.devices())
x = jax.device_put(jnp.arange(24), x_sharding)
```

+++ {"id": "ZgSA9x9NLMaP"}

On a partitioned input, the function `f` produces output that is also partitioned:

```{code-cell}
:id: Oi97rpLz3vGY
:outputId: 204a7e8d-dc88-4b77-b7e3-0e72f306c5d3

jax.debug.visualize_array_sharding(f(key, x))
```

+++ {"id": "WnjlWDUYLkp6"}

But if we inspect the compiled computation for `f` on this partitioned input, we see that it does involve some communication:

```{code-cell}
:id: 64wIZuSJ3vGY
:outputId: 1054fe99-0476-44ec-9693-b0d8f98bf6a8

f_exe = f.lower(key, x).compile()
print('Communicating?', 'collective-permute' in f_exe.as_text())
```

+++ {"id": "AXp9i8fbL8DD"}

One way to work around this is to configure JAX with the experimental upgrade flag `jax_threefry_partitionable`. With the flag on, the "collective permute" operation is now gone from the compiled computation:

```{code-cell}
:id: 1I7bqxA63vGY
:outputId: ec4c579d-f446-4b48-ceda-785c09ba299b

jax.config.update('jax_threefry_partitionable', True)
f_exe = f.lower(key, x).compile()
print('Communicating?', 'collective-permute' in f_exe.as_text())
```

+++ {"id": "WV8ZccM5SXOU"}

The output is still partitioned:

```{code-cell}
:id: zHPJzdn23vGY
:outputId: a8904d20-4d04-4f59-8eae-281e47d29246

jax.debug.visualize_array_sharding(f(key, x))
```

+++ {"id": "kaK--hPmSPpV"}

One caveat to the `jax_threefry_partitionable` option, however, is that _the random values produced may be different than without the flag set_, even though they were generated by the same random key:

```{code-cell}
:id: nBUHBBal3vGY
:outputId: f194c213-0688-4b7a-ffb8-c4453b82b1f1

jax.config.update('jax_threefry_partitionable', False)
print('Stable:')
print(f(key, x))
print()

jax.config.update('jax_threefry_partitionable', True)
print('Partitionable:')
print(f(key, x))
```

+++ {"id": "8BDPqgOrTMfK"}

In `jax_threefry_partitionable` mode, the JAX PRNG remains deterministic, but its implementation is new (and under development). The random values generated for a given key will be the same at a given JAX version (or a given commit on the `main` branch), but may vary across releases.

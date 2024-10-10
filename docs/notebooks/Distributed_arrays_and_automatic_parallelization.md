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

+++ {"id": "dnbZdHiXD9Xa"}

(distributed-arrays-and-automatic-parallelization)=
# Automatic and semi-automated parallelization with `jax.jit`

<!--* freshness: { reviewed: '2024-10-01' } *-->


In this tutorial, you will learn about automatic and semi-automated parallelism for Single-Program Multi-Data (SPMD) code in JAX indepth. For automatic parallelism, you use {func}`jax.jit`, and for semi-automated parallelism - {func}`jax.jit` and {func}`jax.lax.with_sharding_constraint`. You'll also learn about how using {class}`jax.Array`s together with {func}`jax.jit` can provide automatic compiler-based parallelization.

{class}`jax.Array` is a unified datatype for representing arrays or a unified array object model. With SPMD, you can transform a function, such as the forward pass of a neural network, written for one device into a function that can run in parallel on multiple devices, such as several GPUs or Google TPUs.

**Note:** If you are new to {class}`jax.Array` and JAX parallelism, check out {ref}`sharded-computation`, which also goes through the basics {func}`jax.experimental.shard_map.shard_map` for manual parallelism with manual control.

**Warning:** To run the code in this tutorial, you need 8 devices, such as the TPU v2-8 available on Google Colab.

+++ {"id": "sjbPo1b9gpUz"}

## Setup and first example

```{code-cell}
:id: mcO9pfbkD9Xv

from typing import Optional

import numpy as np

import jax
import jax.numpy as jnp
```

+++ {"id": "axjoeT9hD9X1"}

**Warning:** ⚠️ To run the code, you need 8 devices, such as the TPU v2-8 available on Google Colab.

```{code-cell}
:id: n2JIbznnD9X2

if len(jax.local_devices()) < 8:
  raise Exception("Notebook requires 8 devices to run")
```

+++ {"id": "9GCx28cQhc3n"}

Before exploring and explaining each step required to distribute a value across several devices and perform a computation, run this lightly-annotated and simple end-to-end example:

```{code-cell}
:id: Gf2lO4ii3vGG

from jax.experimental import mesh_utils
from jax.sharding import Mesh, PartitionSpec as P, NamedSharding
```

+++ {"id": "b7NA0V7zjg54"}

Create a {mod}`jax.sharding.Sharding` object to distribute a value across devices:

```{code-cell}
:id: q-XBTEoy3vGG

mesh = Mesh(devices=mesh_utils.create_device_mesh((4, 2)),
            axis_names=('x', 'y'))
```

+++ {"id": "UVthGlBgjmty"}

Create an array of random values with {func}`jax.random.normal` and use {func}`jax.device_put` to distribute it across devices:

```{code-cell}
---
colab:
  base_uri: https://localhost:8080/
  height: 217
id: vI39znW93vGH
outputId: b85613de-1d51-43c7-dac1-ff50bc600436
---
x = jax.random.normal(jax.random.key(0), (8192, 8192))
y = jax.device_put(x, NamedSharding(mesh, P('x', 'y')))
jax.debug.visualize_array_sharding(y)
```

+++ {"id": "jZ0ZY9Um9Jg4"}

Next, apply a computation to it and visualize how the result values are stored across multiple devices too using {func}`jax.debug.visualize_array_sharding`:

```{code-cell}
---
colab:
  base_uri: https://localhost:8080/
  height: 217
id: -qCnHZl83vGI
outputId: e608a945-1af7-473e-c2d4-ba2bb6604ea2
---
z = jnp.sin(y)
jax.debug.visualize_array_sharding(z)
```

+++ {"id": "5qccVQoE9tEi"}

The evaluation of the {func}`jnp.sin` application was automatically parallelized across the devices on which the input values (and output values) are stored:

```{code-cell}
---
colab:
  base_uri: https://localhost:8080/
id: _VTzN0r03vGI
outputId: 81451598-d0f2-4682-cba4-e698ae12a400
---
# `x` is present on a single device
%timeit -n 5 -r 5 jnp.sin(x).block_until_ready()
```

```{code-cell}
---
colab:
  base_uri: https://localhost:8080/
id: QuzhU1g63vGI
outputId: e5a88656-316f-4db7-de0c-79253a2e32c7
---
# `y` is sharded across 8 devices.
%timeit -n 5 -r 5 jnp.sin(y).block_until_ready()
```

+++ {"id": "kpyZYSqph8iV"}

Notice the time difference.

Let's review each of these pieces in more detail!

+++ {"id": "xWknFQbQ-bzV"}


## JAX `Sharding` and `NamedSharding` subclass

+++ {"id": "NWDyp_EjVHkg"}

To parallelize computation across multiple devices, you need to first lay out input data across multiple devices.

As covered in {ref}`sharded-computation`'s {ref}`key-concept-data-sharding` section, every {class}`jax.Array` has an associated {mod}`jax.sharding.Sharding` object. `Sharding` objects describe distributed memory layouts. That is, `Sharding` describes how `jax.Array` values are laid out in memory across devices. They can be used with {func}`jax.device_put` to produce a value with distributed layout.

For example, here's a value with a single-device `Sharding`:

```{code-cell}
:id: VmoX4SUp3vGJ

import jax
x = jax.random.normal(jax.random.key(0), (8192, 8192))
```

```{code-cell}
---
colab:
  base_uri: https://localhost:8080/
  height: 200
id: vNRabO2J3vGJ
outputId: a49ebf66-9cf1-4137-e58f-c96e70874c92
---
jax.debug.visualize_array_sharding(x)
```

+++ {"id": "HhCjhK0zXIqX"}

Here, you called the {func}`jax.debug.visualize_array_sharding` function to show where the value `x` is stored in memory. All of `x` is stored on a single device, so the visualization is pretty boring!

But you can shard `x` across multiple devices by using {func}`jax.device_put` and a `Sharding` object. First, create a `numpy.ndarray` of `Devices` using {func}`jax.experimental.mesh_utils.create_device_mesh`, which takes hardware topology into account for the `Device` order:

```{code-cell}
---
colab:
  base_uri: https://localhost:8080/
  height: 217
id: zpB1JxyK3vGN
outputId: 86b612a3-da86-4b9a-c797-90d293c320a8
---
from jax.sharding import Mesh, PartitionSpec, NamedSharding
from jax.experimental import mesh_utils

P = PartitionSpec

devices = mesh_utils.create_device_mesh((4, 2))
mesh = Mesh(devices, axis_names=('a', 'b'))
y = jax.device_put(x, NamedSharding(mesh, P('a', 'b')))
jax.debug.visualize_array_sharding(y)
```

+++ {"id": "OW_Cc92G1-nr"}

Define a helper function to make things simpler:

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
---
colab:
  base_uri: https://localhost:8080/
  height: 217
id: zp3MfS4Y3vGO
outputId: 0bf995da-8a36-4e2c-d679-abf32ee3974c
---
y = jax.device_put(x, mesh_sharding(P('a', 'b')))
jax.debug.visualize_array_sharding(y)
```

+++ {"id": "xZ88riVm1mv5"}

Here, `P('a', 'b')` is used to express that the first and second axes of `x` should be sharded over the device mesh axes `'a'` and `'b'`, respectively. You can easily switch to `P('b', 'a')` to shard the axes of `x` over different devices:

```{code-cell}
---
colab:
  base_uri: https://localhost:8080/
  height: 217
id: FigK5Zsa3vGO
outputId: 6e3b82a4-7d0e-44f8-a929-dfa00b201bc7
---
y = jax.device_put(x, mesh_sharding(P('b', 'a')))
jax.debug.visualize_array_sharding(y)
```

```{code-cell}
---
colab:
  base_uri: https://localhost:8080/
  height: 217
id: hI-HD0xN3vGO
outputId: ab2bbd16-806b-43e8-d065-8b58237d1dc0
---
# This `None` means that `x` is not sharded on its second dimension,
# and since the Mesh axis name 'b' is not mentioned, shards are
# replicated across it.
y = jax.device_put(x, mesh_sharding(P('a', None)))
jax.debug.visualize_array_sharding(y)
```

+++ {"id": "AqcAsNUgXCZz"}

Here, because `P('a', None)` doesn't mention the `Mesh` axis name `'b'`, you get replication over the axis `'b'`. The `None` here is just acting as a placeholder to line up against the second axis of the value `x`, without expressing sharding over any mesh axis. (As a shorthand, trailing `None`s can be omitted, so that `P('a', None)` means the same thing as `P('a')`. But it doesn't hurt to be explicit!)

To shard only over the second axis of `x`, you can use a `None` placeholder in the {mod}`jax.sharding.PartitionSpec`:

```{code-cell}
---
colab:
  base_uri: https://localhost:8080/
  height: 200
id: EXBExMQC3vGP
outputId: 7a27b292-c829-489d-8be8-d56ba71f552e
---
y = jax.device_put(x, mesh_sharding(P(None, 'b')))
jax.debug.visualize_array_sharding(y)
```

```{code-cell}
---
colab:
  base_uri: https://localhost:8080/
  height: 200
id: PjUpG8uz3vGP
outputId: f788993e-4072-41ed-d65f-ae114bd6b94f
---
y = jax.device_put(x, mesh_sharding(P(None, 'a')))
jax.debug.visualize_array_sharding(y)
```

+++ {"id": "--AZgW1P3HFT"}

For a fixed mesh, we can even partition one logical axis of `x` over multiple device mesh axes:

```{code-cell}
---
colab:
  base_uri: https://localhost:8080/
  height: 284
id: fVcPbDUA3vGP
outputId: 64656a33-cdd9-4910-bde7-5be981faa5db
---
y = jax.device_put(x, mesh_sharding(P(('a', 'b'), None)))
jax.debug.visualize_array_sharding(y)
```

+++ {"id": "c1tTFudr3Ae7"}

Using {mod}`jax.sharding.NamedSharding` makes it easy to define a device mesh once and give its axes names, then just refer to those names in {mod}`jax.sharding.PartitionSpec`s for each {func}`jax.device_put` as needed.

+++ {"id": "rhWzHgGf4mkg"}

## Computation follows data sharding and is automatically parallelized

+++ {"id": "JukoaRhl4tXJ"}

With sharded input data, the compiler can give us parallel computation. In particular, functions decorated with {func}`jax.jit` can operate over sharded arrays without copying data onto a single device. Instead, computation follows sharding: based on the sharding of the input data, the compiler decides shardings for intermediates and output values, and parallelizes their evaluation, even inserting communication operations as necessary.

For example, the simplest computation is an elementwise one:

```{code-cell}
:id: _EmQwggc3vGQ

devices = mesh_utils.create_device_mesh((4, 2))
mesh = Mesh(devices, axis_names=('a', 'b'))
```

```{code-cell}
---
colab:
  base_uri: https://localhost:8080/
  height: 453
id: LnT0vWjc3vGQ
outputId: 392b5afb-8fe3-4ab8-e745-2b75136925a7
---
x = jax.device_put(x, NamedSharding(mesh, P('a', 'b')))
print('input sharding:')
jax.debug.visualize_array_sharding(x)

y = jnp.sin(x)
print('output sharding:')
jax.debug.visualize_array_sharding(y)
```

+++ {"id": "7tY2gVRfazaT"}

Here for the elementwise operation {func}`jnp.sin` the compiler chose the output sharding to be the same as the input. Moreover, the compiler automatically parallelized the computation, so that each device computed its output shard from its input shard in parallel.

In other words, even though you wrote the {func}`jnp.sin` computation as if a single machine were to execute it, the compiler splits up the computation for us and executes it on multiple devices.

You can do the same for more than just elementwise operations too. Consider a matrix multiplication with sharded inputs:

```{code-cell}
---
colab:
  base_uri: https://localhost:8080/
  height: 655
id: Dq043GkP3vGQ
outputId: 48ce38ed-0064-4ea8-86c8-d27f09b3c15c
---
y = jax.device_put(x, NamedSharding(mesh, P('a', None)))
z = jax.device_put(x, NamedSharding(mesh, P(None, 'b')))
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

To be sure it's actually running in parallel, you can do a simple timing experiment.

```{code-cell}
---
colab:
  base_uri: https://localhost:8080/
  height: 200
id: QjQ5u8qh3vGQ
outputId: 63e7f30a-6b95-4c1c-e787-9d2e1c4d3e35
---
x_single = jax.device_put(x, jax.devices()[0])
jax.debug.visualize_array_sharding(x_single)
```

```{code-cell}
---
colab:
  base_uri: https://localhost:8080/
id: 8tn8lOj73vGR
outputId: a85ab204-054d-43ff-b7f6-3a1999c858f3
---
np.allclose(jnp.dot(x_single, x_single),
            jnp.dot(y, z))
```

```{code-cell}
---
colab:
  base_uri: https://localhost:8080/
id: D7PpZwhR3vGR
outputId: e3e14a6a-5652-4d00-ec76-a26552e63421
---
%timeit -n 5 -r 5 jnp.dot(x_single, x_single).block_until_ready()
```

```{code-cell}
---
colab:
  base_uri: https://localhost:8080/
id: rgo_yVHF3vGR
outputId: 2043bea9-2ca4-4a0b-cd9a-b1a6a92e0afb
---
%timeit -n 5 -r 5 jnp.dot(y, z).block_until_ready()
```

+++ {"id": "gglQIMXJnnJw"}

Even copying a sharded `Array` produces a result with the sharding of the input:

```{code-cell}
---
colab:
  base_uri: https://localhost:8080/
  height: 217
id: f1Zw-2lH3vGR
outputId: f6755237-1eec-4fce-a06c-e3bde520f95f
---
w_copy = jnp.copy(w)
jax.debug.visualize_array_sharding(w_copy)
```

+++ {"id": "3qfPjJdhgerc"}

The _computation follows data placement_: when you explicitly shard data with {func}`jax.device_put`, and apply functions to that data, the compiler attempts to parallelize the computation and decide the output sharding. This policy for sharded data is a generalization of [JAX's policy of following explicit device placement](https://jax.readthedocs.io/en/latest/faq.html#controlling-data-and-computation-placement-on-devices).

+++ {"id": "QRB95LaWuT80"}

### When explicit shardings disagree, JAX errors

But what if two arguments to a computation are explicitly placed on different sets of devices, or with incompatible device orders?
In these ambiguous cases, an error is raised:

```{code-cell}
:id: 1vAkZAOY3vGR

import textwrap
from termcolor import colored

def print_exception(e):
  name = colored(f'{type(e).__name__}', 'red', force_color=True)
  print(textwrap.fill(f'{name}: {str(e)}'))
```

```{code-cell}
---
colab:
  base_uri: https://localhost:8080/
id: DHh0N3vn3vGS
outputId: 08e9a479-0062-4227-d4f8-f18218c9a76e
---
sharding1 = NamedSharding(Mesh(jax.devices()[:4], 'x'), P('x'))
sharding2 = NamedSharding(Mesh(jax.devices()[4:], 'x'), P('x'))

y = jax.device_put(x, sharding1)
z = jax.device_put(x, sharding2)
try: y + z
except ValueError as e: print_exception(e)
```

```{code-cell}
---
colab:
  base_uri: https://localhost:8080/
id: Im7DkoOl3vGS
outputId: b72459b3-15dc-4c22-9b90-53c8bca92f37
---
devices = jax.devices()
permuted_devices = [devices[i] for i in [0, 1, 2, 3, 6, 7, 4, 5]]

sharding1 = NamedSharding(Mesh(devices, 'x'), P('x'))
sharding2 = NamedSharding(Mesh(permuted_devices, 'x'), P('x'))

y = jax.device_put(x, sharding1)
z = jax.device_put(x, sharding2)
try: y + z
except ValueError as e: print_exception(e)
```

+++ {"id": "6ZYcK8eXrn0p"}

Arrays that have been explicitly placed or sharded with {func}`jax.device_put` are _committed_ to their device(s), and therefore they won't be automatically moved. Refer to the [device placement FAQ](https://jax.readthedocs.io/en/latest/faq.html#controlling-data-and-computation-placement-on-devices) for more information.

When arrays are _not_ explicitly placed or sharded with {func}`jax.device_put`, they are placed _uncommitted_ on the default device.
Unlike committed arrays, uncommitted arrays can be moved and resharded automatically: that is, uncommitted arrays can be arguments to a computation even if other arguments are explicitly placed on different devices.

For example, the output of {func}`jnp.zeros`, {func}`jnp.arange`, and {func}`jnp.array` are uncommitted:

```{code-cell}
---
colab:
  base_uri: https://localhost:8080/
id: _QvtKL8r3vGS
outputId: e876716d-c945-4f12-dc2c-72e387f784e7
---
y = jax.device_put(x, sharding1)
y + jnp.ones_like(y)
y + jnp.arange(y.size).reshape(y.shape)
print('no error!')
```

+++ {"id": "dqMKl79NaIWF"}

## Semi-automated parallelization: Constraining shardings of intermediates in `jit`ted code

+++ {"id": "g4LrDDcJwkHc"}

While the compiler will attempt to decide how a function's intermediate values and outputs should be sharded, you can also give it hints using {func}`jax.lax.with_sharding_constraint`. Using {func}`jax.lax.with_sharding_constraint` is much like {func}`jax.device_put`, except you use it inside staged-out (i.e. `jit`-decorated) functions:

```{code-cell}
:id: jniSFm5V3vGT

mesh = Mesh(mesh_utils.create_device_mesh((4, 2)), ('x', 'y'))
```

```{code-cell}
:id: Q1wuDp-L3vGT

x = jax.random.normal(jax.random.key(0), (8192, 8192))
x = jax.device_put(x, NamedSharding(mesh, P('x', 'y')))
```

```{code-cell}
:id: rqEDj0wB3vGT

@jax.jit
def f(x):
  x = x + 1
  y = jax.lax.with_sharding_constraint(x, NamedSharding(mesh, P('y', 'x')))
  return y
```

```{code-cell}
---
colab:
  base_uri: https://localhost:8080/
  height: 417
id: zYFS-n4r3vGT
outputId: 1d44785c-10c9-4534-a9fd-b5777744e27f
---
jax.debug.visualize_array_sharding(x)
y = f(x)
jax.debug.visualize_array_sharding(y)
```

```{code-cell}
:id: 8g_2Y8wp3vGT

@jax.jit
def f(x):
  x = x + 1
  y = jax.lax.with_sharding_constraint(x, NamedSharding(mesh, P()))
  return y
```

```{code-cell}
---
colab:
  base_uri: https://localhost:8080/
  height: 400
id: AiRFtVsR3vGT
outputId: ebc5bec2-2694-401e-a9a9-0d65ed1891d5
---
jax.debug.visualize_array_sharding(x)
y = f(x)
jax.debug.visualize_array_sharding(y)
```

+++ {"id": "_Y1P5wLTzJSz"}

By adding {func}`jax.lax.with_sharding_constraint`, you constrained the sharding of the output. In addition to respecting the annotation on a particular intermediate, the compiler will use annotations to decide shardings for other values.

It's often a good practice to annotate the outputs of computations, for example based on how the values are ultimately consumed.

+++ {"id": "QUkXWG-baMUs"}

## Examples: neural networks

+++ {"id": "g7y0OJBSGoSW"}

**⚠️ WARNING: The following is meant to be a simple demonstration of automatic sharding propagation with `jax.Array`, but it may not reflect best practices for real examples.** For instance, real examples may require more use of `with_sharding_constraint`.

+++ {"id": "3ii_UPkG3gzP"}

You can use {func}`jax.device_put` and {func}`jax.jit`'s computation-follows-sharding features to parallelize computation in neural networks. Here are some simple examples, based on this basic neural network:

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
:id: mJLqRPpSDX0i

mesh = Mesh(mesh_utils.create_device_mesh((8,)), 'batch')
```

```{code-cell}
:id: _Q5NbdOn3vGV

sharding = NamedSharding(mesh, P('batch'))
replicated_sharding = NamedSharding(mesh, P())
```

```{code-cell}
:id: 3KC6ieEe3vGV

batch = jax.device_put(batch, sharding)
params = jax.device_put(params, replicated_sharding)
```

```{code-cell}
---
colab:
  base_uri: https://localhost:8080/
id: MUb-QE2b3vGV
outputId: e04ed664-12bc-47a5-f1a1-fc8123d726f7
---
loss_jit(params, batch)
```

```{code-cell}
---
colab:
  base_uri: https://localhost:8080/
id: HUkw0u413vGV
outputId: ddda3db4-91c4-436e-eb9b-bb6ad465267f
---
step_size = 1e-5

for _ in range(30):
  grads = gradfun(params, batch)
  params = [(W - step_size * dW, b - step_size * db)
            for (W, b), (dW, db) in zip(params, grads)]

print(loss_jit(params, batch))
```

```{code-cell}
---
colab:
  base_uri: https://localhost:8080/
id: paCw6Zaj3vGV
outputId: 05320478-a0cc-42f6-dffe-f46b3e2c2d62
---
%timeit -n 5 -r 5 gradfun(params, batch)[0][0].block_until_ready()
```

```{code-cell}
:id: BF86UWpg3vGV

batch_single = jax.device_put(batch, jax.devices()[0])
params_single = jax.device_put(params, jax.devices()[0])
```

```{code-cell}
---
colab:
  base_uri: https://localhost:8080/
id: Z1wgUKXk3vGV
outputId: 12ecfa42-5a2e-413b-ae71-4a2e9ffb83e4
---
%timeit -n 5 -r 5 gradfun(params_single, batch_single)[0][0].block_until_ready()
```

+++ {"id": "3AjeeB7B4NP6"}

### 4-way batch data parallelism and 2-way model tensor parallelism

```{code-cell}
:id: k1hxOfgRDwo0

mesh = Mesh(mesh_utils.create_device_mesh((4, 2)), ('batch', 'model'))
```

```{code-cell}
---
colab:
  base_uri: https://localhost:8080/
  height: 417
id: sgIWCjJK3vGW
outputId: 764ef948-126c-4ef4-ef5b-66e3e6a01670
---
batch = jax.device_put(batch, NamedSharding(mesh, P('batch', None)))
jax.debug.visualize_array_sharding(batch[0])
jax.debug.visualize_array_sharding(batch[1])
```

```{code-cell}
:id: q9PQP-0eEAO6

replicated_sharding = NamedSharding(mesh, P())
```

```{code-cell}
:id: BqCjYCgg3vGW

(W1, b1), (W2, b2), (W3, b3), (W4, b4) = params

W1 = jax.device_put(W1, replicated_sharding)
b1 = jax.device_put(b1, replicated_sharding)

W2 = jax.device_put(W2, NamedSharding(mesh, P(None, 'model')))
b2 = jax.device_put(b2, NamedSharding(mesh, P('model')))

W3 = jax.device_put(W3, NamedSharding(mesh, P('model', None)))
b3 = jax.device_put(b3, replicated_sharding)

W4 = jax.device_put(W4, replicated_sharding)
b4 = jax.device_put(b4, replicated_sharding)

params = (W1, b1), (W2, b2), (W3, b3), (W4, b4)
```

```{code-cell}
---
colab:
  base_uri: https://localhost:8080/
  height: 200
id: _lSJ63sh3vGW
outputId: bbcc4885-6e41-4df8-ee20-e6f91e452a72
---
jax.debug.visualize_array_sharding(W2)
```

```{code-cell}
---
colab:
  base_uri: https://localhost:8080/
  height: 217
id: fxkfWYkk3vGW
outputId: bd52faa4-3f03-4ecb-e645-c850af0b934c
---
jax.debug.visualize_array_sharding(W3)
```

```{code-cell}
---
colab:
  base_uri: https://localhost:8080/
id: uPCVs-_k3vGW
outputId: fccc6e94-bd25-4ca6-dfcc-f6f498057b30
---
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
---
colab:
  base_uri: https://localhost:8080/
id: c9Sbl69e3vGX
outputId: a62b0f60-0bfe-4710-93cf-aab4f9e15d6d
---
print(loss_jit(params, batch))
```

```{code-cell}
---
colab:
  base_uri: https://localhost:8080/
  height: 400
id: lkAF0dAb3vGX
outputId: 79a49eda-74cc-404f-8ddd-ca613c0b1924
---
(W1, b1), (W2, b2), (W3, b3), (W4, b4) = params
jax.debug.visualize_array_sharding(W2)
jax.debug.visualize_array_sharding(W3)
```

```{code-cell}
---
colab:
  base_uri: https://localhost:8080/
id: I1Npor3i3vGX
outputId: 3d15965f-4596-46ed-e632-32cd15d26ffd
---
%timeit -n 10 -r 10 gradfun(params, batch)[0][0].block_until_ready()
```

+++ {"id": "3diqi5VRBy6S"}

## Sharp bits

+++ {"id": "OTfoXNnxFYDJ"}

### PRNG implementation in JAX is not automatically partitionable

JAX comes with a functional, deterministic [random number generator](https://jax.readthedocs.io/en/latest/jep/263-prng.html). It underlies the various sampling functions in the {mod}`jax.random`, such as {func}`jax.random.uniform`.

JAX's random numbers are produced by a counter-based PRNG, so in principle, random number generation should be a pure map over counter values. A pure map is a trivially partitionable operation in principle. It should require no cross-device communication, nor any redundant computation across devices.

However, the existing stable PRNG implementation in JAX is not automatically partitionable, for historical reasons.

Consider the following example, where a function draws random uniform numbers and adds them to the input, elementwise:

+++ {"id": "ht_zYFVXNrjN"}

Consider the following example, where a function draws random uniform numbers and adds them to the input, elementwise:

```{code-cell}
:id: kwS-aQE_3vGX

@jax.jit
def f(key, x):
  numbers = jax.random.uniform(key, x.shape)
  return x + numbers

key = jax.random.key(42)
mesh = Mesh(jax.devices(), 'x')
x_sharding = NamedSharding(mesh, P('x'))
x = jax.device_put(jnp.arange(24), x_sharding)
```

+++ {"id": "ZgSA9x9NLMaP"}

On a partitioned input, the function `f` produces output that is also partitioned:

```{code-cell}
---
colab:
  base_uri: https://localhost:8080/
  height: 50
id: Oi97rpLz3vGY
outputId: 185de801-6f64-44c9-a85b-e5d5ac1e9bc9
---
jax.debug.visualize_array_sharding(f(key, x))
```

+++ {"id": "WnjlWDUYLkp6"}

But if you inspect the compiled computation for `f` on this partitioned input, notice that it does involve some communication:

```{code-cell}
---
colab:
  base_uri: https://localhost:8080/
id: 64wIZuSJ3vGY
outputId: 341460da-bd02-413e-f2c2-36b98ec1f8a6
---
f_exe = f.lower(key, x).compile()
print('Communicating?', 'collective-permute' in f_exe.as_text())
```

+++ {"id": "AXp9i8fbL8DD"}

One way to work around this is to configure JAX with the experimental upgrade flag `'jax_threefry_partitionable'` in {func}`jax.config.update`. With the flag on, the "collective permute" operation is now gone from the compiled computation:

```{code-cell}
---
colab:
  base_uri: https://localhost:8080/
id: 1I7bqxA63vGY
outputId: 6f982597-b3fc-4095-803a-2a9928d55186
---
jax.config.update('jax_threefry_partitionable', True)
f_exe = f.lower(key, x).compile()
print('Communicating?', 'collective-permute' in f_exe.as_text())
```

+++ {"id": "WV8ZccM5SXOU"}

The output is still partitioned:

```{code-cell}
---
colab:
  base_uri: https://localhost:8080/
  height: 50
id: zHPJzdn23vGY
outputId: 3399084f-d6a4-4646-9286-e6c338f06809
---
jax.debug.visualize_array_sharding(f(key, x))
```

+++ {"id": "kaK--hPmSPpV"}

One caveat to the `'jax_threefry_partitionable'` option, however, is that _the random values produced may be different than without the flag set_, even though they were generated by the same random key:

```{code-cell}
---
colab:
  base_uri: https://localhost:8080/
id: nBUHBBal3vGY
outputId: d11a93c6-c7fb-4a81-fba2-3d94c5ac0e3d
---
jax.config.update('jax_threefry_partitionable', False)
print('Stable:')
print(f(key, x))
print()

jax.config.update('jax_threefry_partitionable', True)
print('Partitionable:')
print(f(key, x))
```

+++ {"id": "8BDPqgOrTMfK"}

In `'jax_threefry_partitionable'` mode, the JAX PRNG remains deterministic, but its implementation is new (and under development). The random values generated for a given key will be the same at a given JAX version (or a given commit on the `main` branch), but may vary across releases.

+++ {"id": "K5cowA5lj5_e"}

## Next steps

This tutorial serves as a brief introduction of sharded and parallel computation in JAX.

To learn about each SPMD method in-depth, check out these docs:
- {ref}`distributed-arrays-and-automatic-parallelization`
- {doc}`../notebooks/shard_map`

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

+++ {"id": "18AF5Ab4p6VL"}

# Training a Simple Neural Network, with PyTorch Data Loading

<!--* freshness: { reviewed: '2024-05-03' } *-->

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/google/jax/blob/main/docs/notebooks/Neural_Network_and_Data_Loading.ipynb) [![Open in Kaggle](https://kaggle.com/static/images/open-in-kaggle.svg)](https://kaggle.com/kernels/welcome?src=https://github.com/google/jax/blob/main/docs/notebooks/Neural_Network_and_Data_Loading.ipynb)

**Copyright 2018 The JAX Authors.**

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License.
You may obtain a copy of the License at

https://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

+++ {"id": "B_XlLLpcWjkA"}

![JAX](https://raw.githubusercontent.com/google/jax/main/images/jax_logo_250px.png)

Let's combine everything we showed in the [quickstart](https://colab.research.google.com/github/google/jax/blob/main/docs/quickstart.html) to train a simple neural network. We will first specify and train a simple MLP on MNIST using JAX for the computation. We will use PyTorch's data loading API to load images and labels (because it's pretty great, and the world doesn't need yet another data loading library).

Of course, you can use JAX with any API that is compatible with NumPy to make specifying the model a bit more plug-and-play. Here, just for explanatory purposes, we won't use any neural network libraries or special APIs for building our model.

```{code-cell} ipython3
:id: OksHydJDtbbI

import jax.numpy as jnp
from jax import grad, jit, vmap
from jax import random
```

+++ {"id": "MTVcKi-ZYB3R"}

## Hyperparameters
Let's get a few bookkeeping items out of the way.

```{code-cell} ipython3
:id: -fmWA06xYE7d

# A helper function to randomly initialize weights and biases
# for a dense neural network layer
def random_layer_params(m, n, key, scale=1e-2):
  w_key, b_key = random.split(key)
  return scale * random.normal(w_key, (n, m)), scale * random.normal(b_key, (n,))

# Initialize all layers for a fully-connected neural network with sizes "sizes"
def init_network_params(sizes, key):
  keys = random.split(key, len(sizes))
  return [random_layer_params(m, n, k) for m, n, k in zip(sizes[:-1], sizes[1:], keys)]

layer_sizes = [784, 512, 512, 10]
step_size = 0.01
num_epochs = 8
batch_size = 128
n_targets = 10
params = init_network_params(layer_sizes, random.key(0))
```

+++ {"id": "BtoNk_yxWtIw"}

## Auto-batching predictions

Let us first define our prediction function. Note that we're defining this for a _single_ image example. We're going to use JAX's `vmap` function to automatically handle mini-batches, with no performance penalty.

```{code-cell} ipython3
:id: 7APc6tD7TiuZ

from jax.scipy.special import logsumexp

def relu(x):
  return jnp.maximum(0, x)

def predict(params, image):
  # per-example predictions
  activations = image
  for w, b in params[:-1]:
    outputs = jnp.dot(w, activations) + b
    activations = relu(outputs)
  
  final_w, final_b = params[-1]
  logits = jnp.dot(final_w, activations) + final_b
  return logits - logsumexp(logits)
```

+++ {"id": "dRW_TvCTWgaP"}

Let's check that our prediction function only works on single images.

```{code-cell} ipython3
:id: 4sW2A5mnXHc5
:outputId: 9d3b29e8-fab3-4ecb-9f63-bc8c092f9006

# This works on single examples
random_flattened_image = random.normal(random.key(1), (28 * 28,))
preds = predict(params, random_flattened_image)
print(preds.shape)
```

```{code-cell} ipython3
:id: PpyQxuedXfhp
:outputId: d5d20211-b6da-44e9-f71e-946f2a9d0fc4

# Doesn't work with a batch
random_flattened_images = random.normal(random.key(1), (10, 28 * 28))
try:
  preds = predict(params, random_flattened_images)
except TypeError:
  print('Invalid shapes!')
```

```{code-cell} ipython3
:id: oJOOncKMXbwK
:outputId: 31285fab-7667-4871-fcba-28e86adc3fc6

# Let's upgrade it to handle batches using `vmap`

# Make a batched version of the `predict` function
batched_predict = vmap(predict, in_axes=(None, 0))

# `batched_predict` has the same call signature as `predict`
batched_preds = batched_predict(params, random_flattened_images)
print(batched_preds.shape)
```

+++ {"id": "elsG6nX03BvW"}

At this point, we have all the ingredients we need to define our neural network and train it. We've built an auto-batched version of `predict`, which we should be able to use in a loss function. We should be able to use `grad` to take the derivative of the loss with respect to the neural network parameters. Last, we should be able to use `jit` to speed up everything.

+++ {"id": "NwDuFqc9X7ER"}

## Utility and loss functions

```{code-cell} ipython3
:id: 6lTI6I4lWdh5

def one_hot(x, k, dtype=jnp.float32):
  """Create a one-hot encoding of x of size k."""
  return jnp.array(x[:, None] == jnp.arange(k), dtype)
  
def accuracy(params, images, targets):
  target_class = jnp.argmax(targets, axis=1)
  predicted_class = jnp.argmax(batched_predict(params, images), axis=1)
  return jnp.mean(predicted_class == target_class)

def loss(params, images, targets):
  preds = batched_predict(params, images)
  return -jnp.mean(preds * targets)

@jit
def update(params, x, y):
  grads = grad(loss)(params, x, y)
  return [(w - step_size * dw, b - step_size * db)
          for (w, b), (dw, db) in zip(params, grads)]
```

+++ {"id": "umJJGZCC2oKl"}

## Data Loading with PyTorch

JAX is laser-focused on program transformations and accelerator-backed NumPy, so we don't include data loading or munging in the JAX library. There are already a lot of great data loaders out there, so let's just use them instead of reinventing anything. We'll grab PyTorch's data loader, and make a tiny shim to make it work with NumPy arrays.

```{code-cell} ipython3
:id: gEvWt8_u2pqG
:outputId: 2c83a679-9ce5-4c67-bccb-9ea835a8eaf6

!pip install torch torchvision
```

```{code-cell} ipython3
:cellView: both
:id: 94PjXZ8y3dVF

import numpy as np
from jax.tree_util import tree_map
from torch.utils import data
from torchvision.datasets import MNIST

def numpy_collate(batch):
  return tree_map(np.asarray, data.default_collate(batch))

class NumpyLoader(data.DataLoader):
  def __init__(self, dataset, batch_size=1,
                shuffle=False, sampler=None,
                batch_sampler=None, num_workers=0,
                pin_memory=False, drop_last=False,
                timeout=0, worker_init_fn=None):
    super(self.__class__, self).__init__(dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        sampler=sampler,
        batch_sampler=batch_sampler,
        num_workers=num_workers,
        collate_fn=numpy_collate,
        pin_memory=pin_memory,
        drop_last=drop_last,
        timeout=timeout,
        worker_init_fn=worker_init_fn)

class FlattenAndCast(object):
  def __call__(self, pic):
    return np.ravel(np.array(pic, dtype=jnp.float32))
```

```{code-cell} ipython3
:id: l314jsfP4TN4

# Define our dataset, using torch datasets
mnist_dataset = MNIST('/tmp/mnist/', download=True, transform=FlattenAndCast())
training_generator = NumpyLoader(mnist_dataset, batch_size=batch_size, num_workers=0)
```

```{code-cell} ipython3
:id: FTNo4beUvb6t
:outputId: 65a9087c-c326-49e5-cbfc-e0839212fa31

# Get the full train dataset (for checking accuracy while training)
train_images = np.array(mnist_dataset.train_data).reshape(len(mnist_dataset.train_data), -1)
train_labels = one_hot(np.array(mnist_dataset.train_labels), n_targets)

# Get full test dataset
mnist_dataset_test = MNIST('/tmp/mnist/', download=True, train=False)
test_images = jnp.array(mnist_dataset_test.test_data.numpy().reshape(len(mnist_dataset_test.test_data), -1), dtype=jnp.float32)
test_labels = one_hot(np.array(mnist_dataset_test.test_labels), n_targets)
```

+++ {"id": "xxPd6Qw3Z98v"}

## Training Loop

```{code-cell} ipython3
:id: X2DnZo3iYj18
:outputId: 0eba3ca2-24a1-4cba-aaf4-3ac61d0c650e

import time

for epoch in range(num_epochs):
  start_time = time.time()
  for x, y in training_generator:
    y = one_hot(y, n_targets)
    params = update(params, x, y)
  epoch_time = time.time() - start_time

  train_acc = accuracy(params, train_images, train_labels)
  test_acc = accuracy(params, test_images, test_labels)
  print("Epoch {} in {:0.2f} sec".format(epoch, epoch_time))
  print("Training set accuracy {}".format(train_acc))
  print("Test set accuracy {}".format(test_acc))
```

+++ {"id": "xC1CMcVNYwxm"}

We've now used the whole of the JAX API: `grad` for derivatives, `jit` for speedups and `vmap` for auto-vectorization.
We used NumPy to specify all of our computation, and borrowed the great data loaders from PyTorch, and ran the whole thing on the GPU.

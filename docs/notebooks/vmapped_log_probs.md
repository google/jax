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

+++ {"id": "6umP1IKf4Dg6"}

# Autobatching for Bayesian Inference

<!--* freshness: { reviewed: '2024-04-08' } *-->

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/google/jax/blob/main/docs/notebooks/vmapped_log_probs.ipynb) [![Open in Kaggle](https://kaggle.com/static/images/open-in-kaggle.svg)](https://kaggle.com/kernels/welcome?src=https://github.com/google/jax/blob/main/docs/notebooks/vmapped_log_probs.ipynb)

This notebook demonstrates a simple Bayesian inference example where autobatching makes user code easier to write, easier to read, and less likely to include bugs.

Inspired by a notebook by @davmre.

```{code-cell} ipython3
:id: 8RZDkfbV3zdR

import functools
import itertools
import re
import sys
import time

from matplotlib.pyplot import *

import jax

from jax import lax
import jax.numpy as jnp
import jax.scipy as jsp
from jax import random

import numpy as np
import scipy as sp
```

+++ {"id": "p2VcZS1d34C6"}

## Generate a fake binary classification dataset

```{code-cell} ipython3
:id: pq41hMvn4c_i

np.random.seed(10009)

num_features = 10
num_points = 100

true_beta = np.random.randn(num_features).astype(jnp.float32)
all_x = np.random.randn(num_points, num_features).astype(jnp.float32)
y = (np.random.rand(num_points) < sp.special.expit(all_x.dot(true_beta))).astype(jnp.int32)
```

```{code-cell} ipython3
:id: O0nVumAw7IlT
:outputId: 751a3290-a81b-4538-9183-16cd685fbaf9

y
```

+++ {"id": "DZRVvhpn5aB1"}

## Write the log-joint function for the model

We'll write a non-batched version, a manually batched version, and an autobatched version.

+++ {"id": "C_mDXInL7nsP"}

### Non-batched

```{code-cell} ipython3
:id: ZHyL2sJh5ajG

def log_joint(beta):
    result = 0.
    # Note that no `axis` parameter is provided to `jnp.sum`.
    result = result + jnp.sum(jsp.stats.norm.logpdf(beta, loc=0., scale=1.))
    result = result + jnp.sum(-jnp.log(1 + jnp.exp(-(2*y-1) * jnp.dot(all_x, beta))))
    return result
```

```{code-cell} ipython3
:id: e51qW0ro6J7C
:outputId: 2ec6bbbd-12ee-45bc-af76-5111c53e4d5a

log_joint(np.random.randn(num_features))
```

```{code-cell} ipython3
:id: fglQXK1Y6wnm
:outputId: 2b934336-08ad-4776-9a58-aa575bf601eb

# This doesn't work, because we didn't write `log_prob()` to handle batching.
try:
  batch_size = 10
  batched_test_beta = np.random.randn(batch_size, num_features)

  log_joint(np.random.randn(batch_size, num_features))
except ValueError as e:
  print("Caught expected exception " + str(e))
```

+++ {"id": "_lQ8MnKq7sLU"}

### Manually batched

```{code-cell} ipython3
:id: 2g5-4bQE7gRA

def batched_log_joint(beta):
    result = 0.
    # Here (and below) `sum` needs an `axis` parameter. At best, forgetting to set axis
    # or setting it incorrectly yields an error; at worst, it silently changes the
    # semantics of the model.
    result = result + jnp.sum(jsp.stats.norm.logpdf(beta, loc=0., scale=1.),
                           axis=-1)
    # Note the multiple transposes. Getting this right is not rocket science,
    # but it's also not totally mindless. (I didn't get it right on the first
    # try.)
    result = result + jnp.sum(-jnp.log(1 + jnp.exp(-(2*y-1) * jnp.dot(all_x, beta.T).T)),
                           axis=-1)
    return result
```

```{code-cell} ipython3
:id: KdDMr-Gy85CO
:outputId: db746654-68e9-43b8-ce3b-6e5682e22eb5

batch_size = 10
batched_test_beta = np.random.randn(batch_size, num_features)

batched_log_joint(batched_test_beta)
```

+++ {"id": "-uuGlHQ_85kd"}

### Autobatched with vmap

It just works.

```{code-cell} ipython3
:id: SU20bouH8-Za
:outputId: ee450298-982f-4b9a-bed9-a6f9b8f63d92

vmap_batched_log_joint = jax.vmap(log_joint)
vmap_batched_log_joint(batched_test_beta)
```

+++ {"id": "L1KNBo9y_yZJ"}

## Self-contained variational inference example

A little code is copied from above.

+++ {"id": "lQTPaaQMJh8Y"}

### Set up the (batched) log-joint function

```{code-cell} ipython3
:id: AITXbaofA3Pm

@jax.jit
def log_joint(beta):
    result = 0.
    # Note that no `axis` parameter is provided to `jnp.sum`.
    result = result + jnp.sum(jsp.stats.norm.logpdf(beta, loc=0., scale=10.))
    result = result + jnp.sum(-jnp.log(1 + jnp.exp(-(2*y-1) * jnp.dot(all_x, beta))))
    return result

batched_log_joint = jax.jit(jax.vmap(log_joint))
```

+++ {"id": "UmmFMQ8LJk6a"}

### Define the ELBO and its gradient

```{code-cell} ipython3
:id: MJtnskL6BKwV

def elbo(beta_loc, beta_log_scale, epsilon):
    beta_sample = beta_loc + jnp.exp(beta_log_scale) * epsilon
    return jnp.mean(batched_log_joint(beta_sample), 0) + jnp.sum(beta_log_scale - 0.5 * np.log(2*np.pi))
 
elbo = jax.jit(elbo)
elbo_val_and_grad = jax.jit(jax.value_and_grad(elbo, argnums=(0, 1)))
```

+++ {"id": "oQC7xKYnJrp5"}

### Optimize the ELBO using SGD

```{code-cell} ipython3
:id: 9JrD5nNgH715
:outputId: 80bf62d8-821a-45c4-885c-528b2e449e97

def normal_sample(key, shape):
    """Convenience function for quasi-stateful RNG."""
    new_key, sub_key = random.split(key)
    return new_key, random.normal(sub_key, shape)

normal_sample = jax.jit(normal_sample, static_argnums=(1,))

key = random.key(10003)

beta_loc = jnp.zeros(num_features, jnp.float32)
beta_log_scale = jnp.zeros(num_features, jnp.float32)

step_size = 0.01
batch_size = 128
epsilon_shape = (batch_size, num_features)
for i in range(1000):
    key, epsilon = normal_sample(key, epsilon_shape)
    elbo_val, (beta_loc_grad, beta_log_scale_grad) = elbo_val_and_grad(
        beta_loc, beta_log_scale, epsilon)
    beta_loc += step_size * beta_loc_grad
    beta_log_scale += step_size * beta_log_scale_grad
    if i % 10 == 0:
        print('{}\t{}'.format(i, elbo_val))
```

+++ {"id": "b3ZAe5fJJ2KM"}

### Display the results

Coverage isn't quite as good as we might like, but it's not bad, and nobody said variational inference was exact.

```{code-cell} ipython3
:id: zt1NBLoVHtOG
:outputId: fb159795-e6e7-497c-e501-9933ec761af4

figure(figsize=(7, 7))
plot(true_beta, beta_loc, '.', label='Approximated Posterior Means')
plot(true_beta, beta_loc + 2*jnp.exp(beta_log_scale), 'r.', label='Approximated Posterior $2\sigma$ Error Bars')
plot(true_beta, beta_loc - 2*jnp.exp(beta_log_scale), 'r.')
plot_scale = 3
plot([-plot_scale, plot_scale], [-plot_scale, plot_scale], 'k')
xlabel('True beta')
ylabel('Estimated beta')
legend(loc='best')
```

```{code-cell} ipython3
:id: _bXdOlvUEJl0


```

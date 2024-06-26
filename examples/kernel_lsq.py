# Copyright 2018 The JAX Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


from functools import partial

import numpy.random as npr

import jax.numpy as jnp
from jax.example_libraries import optimizers
from jax import grad, jit, make_jaxpr, vmap, lax


def gram(kernel, xs):
  """Compute a Gram matrix from a kernel and an array of data points.

  Args:
    kernel: callable, maps pairs of data points to scalars.
    xs: array of data points, stacked along the leading dimension.

  Returns:
    A 2d array `a` such that `a[i, j] = kernel(xs[i], xs[j])`.
  """
  return vmap(lambda x: vmap(lambda y: kernel(x, y))(xs))(xs)


def minimize(f, x, num_steps=10000, step_size=0.000001, mass=0.9):
  opt_init, opt_update, get_params = optimizers.momentum(step_size, mass)

  @jit
  def update(i, opt_state):
    x = get_params(opt_state)
    return opt_update(i, grad(f)(x), opt_state)

  opt_state = opt_init(x)
  for i in range(num_steps):
    opt_state = update(i, opt_state)
  return get_params(opt_state)


def train(kernel, xs, ys, regularization=0.01):
  gram_ = jit(partial(gram, kernel))
  gram_mat = gram_(xs)
  n = xs.shape[0]

  def objective(v):
    risk = .5 * jnp.sum((jnp.dot(gram_mat, v) - ys) ** 2.0)
    reg = regularization * jnp.sum(v ** 2.0)
    return risk + reg

  v = minimize(objective, jnp.zeros(n))

  def predict(x):
    prods = vmap(lambda x_: kernel(x, x_))(xs)
    return jnp.sum(v * prods)

  return jit(vmap(predict))


if __name__ == "__main__":
  n = 100
  d = 20

  # linear kernel

  linear_kernel = lambda x, y: jnp.dot(x, y, precision=lax.Precision.HIGH)
  truth = npr.randn(d)
  xs = npr.randn(n, d)
  ys = jnp.dot(xs, truth)

  predict = train(linear_kernel, xs, ys)

  print('MSE:', jnp.sum((predict(xs) - ys) ** 2.))

  def gram_jaxpr(kernel):
    return make_jaxpr(partial(gram, kernel))(xs)

  rbf_kernel = lambda x, y: jnp.exp(-jnp.sum((x - y) ** 2))

  print()
  print('jaxpr of gram(linear_kernel):')
  print(gram_jaxpr(linear_kernel))
  print()
  print('jaxpr of gram(rbf_kernel):')
  print(gram_jaxpr(rbf_kernel))

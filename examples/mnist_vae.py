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

"""A basic variational autoencoder (VAE) on binarized MNIST using Numpy and JAX.

This file uses the stax network definition library and the optimizers
optimization library.
"""


import os
import time

import matplotlib.pyplot as plt

import jax
import jax.numpy as jnp
from jax import jit, grad, lax, random
from jax.example_libraries import optimizers
from jax.example_libraries import stax
from jax.example_libraries.stax import Dense, FanOut, Relu, Softplus
from examples import datasets


def gaussian_kl(mu, sigmasq):
  """KL divergence from a diagonal Gaussian to the standard Gaussian."""
  return -0.5 * jnp.sum(1. + jnp.log(sigmasq) - mu**2. - sigmasq)

def gaussian_sample(rng, mu, sigmasq):
  """Sample a diagonal Gaussian."""
  return mu + jnp.sqrt(sigmasq) * random.normal(rng, mu.shape)

def bernoulli_logpdf(logits, x):
  """Bernoulli log pdf of data x given logits."""
  return -jnp.sum(jnp.logaddexp(0., jnp.where(x, -1., 1.) * logits))

def elbo(rng, params, images):
  """Monte Carlo estimate of the negative evidence lower bound."""
  enc_params, dec_params = params
  mu_z, sigmasq_z = encode(enc_params, images)
  logits_x = decode(dec_params, gaussian_sample(rng, mu_z, sigmasq_z))
  return bernoulli_logpdf(logits_x, images) - gaussian_kl(mu_z, sigmasq_z)

def image_sample(rng, params, nrow, ncol):
  """Sample images from the generative model."""
  _, dec_params = params
  code_rng, img_rng = random.split(rng)
  logits = decode(dec_params, random.normal(code_rng, (nrow * ncol, 10)))
  sampled_images = random.bernoulli(img_rng, jnp.logaddexp(0., logits))
  return image_grid(nrow, ncol, sampled_images, (28, 28))

def image_grid(nrow, ncol, imagevecs, imshape):
  """Reshape a stack of image vectors into an image grid for plotting."""
  images = iter(imagevecs.reshape((-1,) + imshape))
  return jnp.vstack([jnp.hstack([next(images).T for _ in range(ncol)][::-1])
                    for _ in range(nrow)]).T


encoder_init, encode = stax.serial(
    Dense(512), Relu,
    Dense(512), Relu,
    FanOut(2),
    stax.parallel(Dense(10), stax.serial(Dense(10), Softplus)),
)

decoder_init, decode = stax.serial(
    Dense(512), Relu,
    Dense(512), Relu,
    Dense(28 * 28),
)


if __name__ == "__main__":
  step_size = 0.001
  num_epochs = 100
  batch_size = 32
  nrow, ncol = 10, 10  # sampled image grid size

  test_rng = random.key(1)  # fixed prng key for evaluation
  imfile = os.path.join(os.getenv("TMPDIR", "/tmp/"), "mnist_vae_{:03d}.png")

  train_images, _, test_images, _ = datasets.mnist(permute_train=True)
  num_complete_batches, leftover = divmod(train_images.shape[0], batch_size)
  num_batches = num_complete_batches + bool(leftover)

  enc_init_rng, dec_init_rng = random.split(random.key(2))
  _, init_encoder_params = encoder_init(enc_init_rng, (batch_size, 28 * 28))
  _, init_decoder_params = decoder_init(dec_init_rng, (batch_size, 10))
  init_params = init_encoder_params, init_decoder_params

  opt_init, opt_update, get_params = optimizers.momentum(step_size, mass=0.9)

  train_images = jax.device_put(train_images)
  test_images = jax.device_put(test_images)

  def binarize_batch(rng, i, images):
    i = i % num_batches
    batch = lax.dynamic_slice_in_dim(images, i * batch_size, batch_size)
    return random.bernoulli(rng, batch)

  @jit
  def run_epoch(rng, opt_state, images):
    def body_fun(i, opt_state):
      elbo_rng, data_rng = random.split(random.fold_in(rng, i))
      batch = binarize_batch(data_rng, i, images)
      loss = lambda params: -elbo(elbo_rng, params, batch) / batch_size
      g = grad(loss)(get_params(opt_state))
      return opt_update(i, g, opt_state)
    return lax.fori_loop(0, num_batches, body_fun, opt_state)

  @jit
  def evaluate(opt_state, images):
    params = get_params(opt_state)
    elbo_rng, data_rng, image_rng = random.split(test_rng, 3)
    binarized_test = random.bernoulli(data_rng, images)
    test_elbo = elbo(elbo_rng, params, binarized_test) / images.shape[0]
    sampled_images = image_sample(image_rng, params, nrow, ncol)
    return test_elbo, sampled_images

  opt_state = opt_init(init_params)
  for epoch in range(num_epochs):
    tic = time.time()
    opt_state = run_epoch(random.key(epoch), opt_state, train_images)
    test_elbo, sampled_images = evaluate(opt_state, test_images)
    print(f"{epoch: 3d} {test_elbo} ({time.time() - tic:.3f} sec)")
    plt.imsave(imfile.format(epoch), sampled_images, cmap=plt.cm.gray)

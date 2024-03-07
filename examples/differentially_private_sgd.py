# Copyright 2019 The JAX Authors.
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

"""JAX efficiently trains a differentially private conv net on MNIST.

This script contains a JAX implementation of Differentially Private Stochastic
Gradient Descent (https://arxiv.org/abs/1607.00133). DPSGD requires clipping
the per-example parameter gradients, which is non-trivial to implement
efficiently for convolutional neural networks.  The JAX XLA compiler shines in
this setting by optimizing the minibatch-vectorized computation for
convolutional architectures. Train time takes a few seconds per epoch on a
commodity GPU.

This code depends on the `dp-accounting` package
  Install instructions:
    $ pip install dp-accounting

The results match those in the reference TensorFlow baseline implementation:
  https://github.com/tensorflow/privacy/tree/master/tutorials

Example invocations:
  # this non-private baseline should get ~99% acc
  python -m examples.differentially_private_sgd \
    --dpsgd=False \
    --learning_rate=.1 \
    --epochs=20 \

   this private baseline should get ~95% acc
  python -m examples.differentially_private_sgd \
   --dpsgd=True \
   --noise_multiplier=1.3 \
   --l2_norm_clip=1.5 \
   --epochs=15 \
   --learning_rate=.25 \

  # this private baseline should get ~96.6% acc
  python -m examples.differentially_private_sgd \
   --dpsgd=True \
   --noise_multiplier=1.1 \
   --l2_norm_clip=1.0 \
   --epochs=60 \
   --learning_rate=.15 \

  # this private baseline should get ~97% acc
  python -m examples.differentially_private_sgd \
   --dpsgd=True \
   --noise_multiplier=0.7 \
   --l2_norm_clip=1.5 \
   --epochs=45 \
   --learning_rate=.25 \
"""

import itertools
import time
import warnings

from absl import app
from absl import flags

from jax import grad
from jax import jit
from jax import random
from jax import vmap
from jax.example_libraries import optimizers
from jax.example_libraries import stax
from jax.tree_util import tree_flatten, tree_unflatten
import jax.numpy as jnp
from examples import datasets
import numpy.random as npr

# https://github.com/google/differential-privacy
from dp_accounting import dp_event
from dp_accounting import rdp


_DPSGD = flags.DEFINE_boolean(
    'dpsgd', True, 'If True, train with DP-SGD. If False, '
    'train with vanilla SGD.')
_LEARNING_RATE = flags.DEFINE_float('learning_rate', .15, 'Learning rate for training')
_NOISE_MULTIPLIER = flags.DEFINE_float('noise_multiplier', 1.1,
                   'Ratio of the standard deviation to the clipping norm')
_L2_NORM_CLIP = flags.DEFINE_float('l2_norm_clip', 1.0, 'Clipping norm')
_BATCH_SIZE = flags.DEFINE_integer('batch_size', 256, 'Batch size')
_EPOCHS = flags.DEFINE_integer('epochs', 60, 'Number of epochs')
_SEED = flags.DEFINE_integer('seed', 0, 'Seed for jax PRNG')
_MICROBATCHES = flags.DEFINE_integer(
    'microbatches', None, 'Number of microbatches '
    '(must evenly divide batch_size)')
_MODEL_DIR = flags.DEFINE_string('model_dir', None, 'Model directory')


init_random_params, predict = stax.serial(
    stax.Conv(16, (8, 8), padding='SAME', strides=(2, 2)),
    stax.Relu,
    stax.MaxPool((2, 2), (1, 1)),
    stax.Conv(32, (4, 4), padding='VALID', strides=(2, 2)),
    stax.Relu,
    stax.MaxPool((2, 2), (1, 1)),
    stax.Flatten,
    stax.Dense(32),
    stax.Relu,
    stax.Dense(10),
)


def loss(params, batch):
  inputs, targets = batch
  logits = predict(params, inputs)
  logits = stax.logsoftmax(logits)  # log normalize
  return -jnp.mean(jnp.sum(logits * targets, axis=1))  # cross entropy loss


def accuracy(params, batch):
  inputs, targets = batch
  target_class = jnp.argmax(targets, axis=1)
  predicted_class = jnp.argmax(predict(params, inputs), axis=1)
  return jnp.mean(predicted_class == target_class)


def clipped_grad(params, l2_norm_clip, single_example_batch):
  """Evaluate gradient for a single-example batch and clip its grad norm."""
  grads = grad(loss)(params, single_example_batch)
  nonempty_grads, tree_def = tree_flatten(grads)
  total_grad_norm = jnp.linalg.norm(
      jnp.array([jnp.linalg.norm(neg.ravel()) for neg in nonempty_grads]))
  divisor = jnp.maximum(total_grad_norm / l2_norm_clip, 1.)
  normalized_nonempty_grads = [g / divisor for g in nonempty_grads]
  return tree_unflatten(tree_def, normalized_nonempty_grads)


def private_grad(params, batch, rng, l2_norm_clip, noise_multiplier,
                 batch_size):
  """Return differentially private gradients for params, evaluated on batch."""
  clipped_grads = vmap(clipped_grad, (None, None, 0))(params, l2_norm_clip, batch)
  clipped_grads_flat, grads_treedef = tree_flatten(clipped_grads)
  aggregated_clipped_grads = [g.sum(0) for g in clipped_grads_flat]
  rngs = random.split(rng, len(aggregated_clipped_grads))
  noised_aggregated_clipped_grads = [
      g + l2_norm_clip * noise_multiplier * random.normal(r, g.shape)
      for r, g in zip(rngs, aggregated_clipped_grads)]
  normalized_noised_aggregated_clipped_grads = [
      g / batch_size for g in noised_aggregated_clipped_grads]
  return tree_unflatten(grads_treedef, normalized_noised_aggregated_clipped_grads)


def shape_as_image(images, labels, dummy_dim=False):
  target_shape = (-1, 1, 28, 28, 1) if dummy_dim else (-1, 28, 28, 1)
  return jnp.reshape(images, target_shape), labels


def compute_epsilon(steps, num_examples=60000, target_delta=1e-5):
  if num_examples * target_delta > 1.:
    warnings.warn('Your delta might be too high.')
  q = _BATCH_SIZE.value / float(num_examples)
  orders = list(jnp.linspace(1.1, 10.9, 99)) + list(range(11, 64))
  accountant = rdp.rdp_privacy_accountant.RdpAccountant(orders)
  accountant.compose(
      dp_event.PoissonSampledDpEvent(
          q, dp_event.GaussianDpEvent(_NOISE_MULTIPLIER.value)), steps)
  return accountant.get_epsilon(target_delta)


def main(_):

  if _MICROBATCHES.value:
    raise NotImplementedError(
        'Microbatches < batch size not currently supported'
    )

  train_images, train_labels, test_images, test_labels = datasets.mnist()
  num_train = train_images.shape[0]
  num_complete_batches, leftover = divmod(num_train, _BATCH_SIZE.value)
  num_batches = num_complete_batches + bool(leftover)
  key = random.key(_SEED.value)

  def data_stream():
    rng = npr.RandomState(_SEED.value)
    while True:
      perm = rng.permutation(num_train)
      for i in range(num_batches):
        batch_idx = perm[i * _BATCH_SIZE.value:(i + 1) * _BATCH_SIZE.value]
        yield train_images[batch_idx], train_labels[batch_idx]

  batches = data_stream()

  opt_init, opt_update, get_params = optimizers.sgd(_LEARNING_RATE.value)

  @jit
  def update(_, i, opt_state, batch):
    params = get_params(opt_state)
    return opt_update(i, grad(loss)(params, batch), opt_state)

  @jit
  def private_update(rng, i, opt_state, batch):
    params = get_params(opt_state)
    rng = random.fold_in(rng, i)  # get new key for new random numbers
    return opt_update(
        i,
        private_grad(params, batch, rng, _L2_NORM_CLIP.value,
                     _NOISE_MULTIPLIER.value, _BATCH_SIZE.value), opt_state)

  _, init_params = init_random_params(key, (-1, 28, 28, 1))
  opt_state = opt_init(init_params)
  itercount = itertools.count()

  steps_per_epoch = 60000 // _BATCH_SIZE.value
  print('\nStarting training...')
  for epoch in range(1, _EPOCHS.value + 1):
    start_time = time.time()
    for _ in range(num_batches):
      if _DPSGD.value:
        opt_state = \
            private_update(
                key, next(itercount), opt_state,
                shape_as_image(*next(batches), dummy_dim=True))
      else:
        opt_state = update(
            key, next(itercount), opt_state, shape_as_image(*next(batches)))
    epoch_time = time.time() - start_time
    print(f'Epoch {epoch} in {epoch_time:0.2f} sec')

    # evaluate test accuracy
    params = get_params(opt_state)
    test_acc = accuracy(params, shape_as_image(test_images, test_labels))
    test_loss = loss(params, shape_as_image(test_images, test_labels))
    print('Test set loss, accuracy (%): ({:.2f}, {:.2f})'.format(
        test_loss, 100 * test_acc))

    # determine privacy loss so far
    if _DPSGD.value:
      delta = 1e-5
      num_examples = 60000
      eps = compute_epsilon(epoch * steps_per_epoch, num_examples, delta)
      print(
          f'For delta={delta:.0e}, the current epsilon is: {eps:.2f}')
    else:
      print('Trained with vanilla non-private SGD optimizer')


if __name__ == '__main__':
  app.run(main)

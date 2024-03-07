# Copyright 2020 The JAX Authors.
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
from absl import app
from absl import flags

import os
import time

from flax import linen as nn
from flax.training import train_state

import jax
from jax import numpy as jnp

import optax

import numpy as np
import tensorflow as tf
import tensorflowjs as tfjs

from . import input_pipeline


_NUM_EPOCHS = flags.DEFINE_integer(
    "num_epochs", 5, "Number of epochs to train for."
)
_NUM_CLASSES = flags.DEFINE_integer(
    "num_classes", 100, "Number of classification classes."
)

flags.register_validator("num_classes",
                         lambda value: value >= 1 and value <= 100,
                         message="--num_classes must be in range [1, 100]")


# The code below is an adaptation for Flax from the work published here:
# https://blog.tensorflow.org/2018/07/train-model-in-tfkeras-with-colab-and-run-in-browser-tensorflowjs.html

class QuickDraw(nn.Module):
  @nn.compact
  def __call__(self, x):
    x = nn.Conv(features=16, kernel_size=(3, 3), padding='SAME')(x)
    x = nn.relu(x)
    x = nn.max_pool(x, window_shape=(2, 2), strides=(2, 2))

    x = nn.Conv(features=32, kernel_size=(3, 3), padding='SAME')(x)
    x = nn.relu(x)
    x = nn.max_pool(x, window_shape=(2, 2), strides=(2, 2))

    x = nn.Conv(features=64, kernel_size=(3, 3), padding='SAME')(x)
    x = nn.relu(x)
    x = nn.max_pool(x, window_shape=(2, 2), strides=(2, 2))

    x = x.reshape((x.shape[0], -1)) # flatten
    x = nn.Dense(features=128)(x)
    x = nn.relu(x)

    x = nn.Dense(features=_NUM_CLASSES.value)(x)

    return x


@jax.jit
def apply_model(state, inputs, labels):
  """Computes gradients, loss and accuracy for a single batch."""
  def loss_fn(params):
    logits = state.apply_fn({'params': params}, inputs)
    one_hot = jax.nn.one_hot(labels, _NUM_CLASSES.value)
    loss = jnp.mean(optax.softmax_cross_entropy(logits=logits, labels=one_hot))
    return loss, logits
  grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
  (loss, logits), grads = grad_fn(state.params)
  accuracy = jnp.mean(jnp.argmax(logits, -1) == labels)
  return grads, loss, accuracy


@jax.jit
def update_model(state, grads):
  return state.apply_gradients(grads=grads)


def run_epoch(state, dataset, train=True):
  epoch_loss = []
  epoch_accuracy = []

  for inputs, labels in dataset:
    grads, loss, accuracy = apply_model(state, inputs, labels)
    if train:
      state = update_model(state, grads)
    epoch_loss.append(loss)
    epoch_accuracy.append(accuracy)
  loss = np.mean(epoch_loss)
  accuracy = np.mean(epoch_accuracy)
  return state, loss, accuracy


def create_train_state(rng):
  quick_draw = QuickDraw()
  params = quick_draw.init(rng, jnp.ones((1, 28, 28, 1)))['params']
  tx = optax.adam(learning_rate=0.001, b1=0.9, b2=0.999)
  return train_state.TrainState.create(
    apply_fn=quick_draw.apply, params=params, tx=tx)


def train(state, train_ds, test_ds):
  for epoch in range(1, _NUM_EPOCHS.value+1):
    start_time = time.time()

    state, train_loss, train_accuracy = run_epoch(state, train_ds)
    _, test_loss, test_accuracy = run_epoch(state, test_ds, train=False)

    print(f"Training set accuracy {train_accuracy}")
    print(f"Training set loss {train_loss}")
    print(f"Test set accuracy {test_accuracy}")
    print(f"Test set loss {test_loss}")

    epoch_time = time.time() - start_time
    print(f"Epoch {epoch} in {epoch_time:0.2f} sec")

  return state


def main(argv):
  if len(argv) > 1:
    raise app.UsageError('Too many command-line arguments.')

  base_model_path = "/tmp/jax2tf/tf_js_quickdraw"
  dataset_path = os.path.join(base_model_path, "data")
  classes = input_pipeline.download_dataset(dataset_path, _NUM_CLASSES.value)
  assert len(classes) == _NUM_CLASSES.value, "Incorrect number of classes"
  print(f"Classes are: {classes}")
  print("Loading dataset into memory...")
  train_ds, test_ds = input_pipeline.get_datasets(dataset_path, classes)
  print(f"Starting training for {_NUM_EPOCHS.value} epochs...")

  state = create_train_state(jax.random.PRNGKey(0))
  state = train(state, train_ds, test_ds)

  tfjs.converters.convert_jax(
    apply_fn=state.apply_fn,
    params={'params': state.params},
    input_signatures=[tf.TensorSpec([1, 28, 28, 1])],
    model_dir=os.path.join(base_model_path, 'tfjs_models'))

if __name__ == "__main__":
  app.run(main)

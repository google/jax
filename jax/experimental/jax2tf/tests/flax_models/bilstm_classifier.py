# Copyright 2022 The JAX Authors.
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
"""Bi-LSTM text classification model from Flax example SST2, see:

https://github.com/google/flax/tree/main/examples/sst2
"""

from __future__ import annotations

from collections.abc import Callable
import functools
from typing import Any

from flax import linen as nn
import jax
from jax import numpy as jnp
from jax._src.typing import Array


def sequence_mask(lengths: Array, max_length: int) -> Array:
  """Computes a boolean mask over sequence positions for each given length.

  Example:
  ```
  sequence_mask([1, 2], 3)
  [[True, False, False],
   [True, True, False]]
  ```

  Args:
    lengths: The length of each sequence. <int>[batch_size]
    max_length: The width of the boolean mask. Must be >= max(lengths).

  Returns:
    A mask with shape: <bool>[batch_size, max_length] indicating which
    positions are valid for each sequence.
  """
  return jnp.arange(max_length)[None] < lengths[:, None]


@jax.vmap
def flip_sequences(inputs: Array, lengths: Array) -> Array:
  """Flips a sequence of inputs along the time dimension.

  This function can be used to prepare inputs for the reverse direction of a
  bidirectional LSTM. It solves the issue that, when naively flipping multiple
  padded sequences stored in a matrix, the first elements would be padding
  values for those sequences that were padded. This function keeps the padding
  at the end, while flipping the rest of the elements.

  Example:
  ```python
  inputs = [[1, 0, 0],
            [2, 3, 0]
            [4, 5, 6]]
  lengths = [1, 2, 3]
  flip_sequences(inputs, lengths) = [[1, 0, 0],
                                     [3, 2, 0],
                                     [6, 5, 4]]
  ```

  Args:
    inputs: An array of input IDs <int>[batch_size, seq_length].
    lengths: The length of each sequence <int>[batch_size].

  Returns:
    An ndarray with the flipped inputs.
  """
  # Note: since this function is vmapped, the code below is effectively for
  # a single example.
  max_length = inputs.shape[0]
  return jnp.flip(jnp.roll(inputs, max_length - lengths, axis=0), axis=0)


class WordDropout(nn.Module):
  """Applies word dropout to a batch of input IDs.

  This is basically the same as `nn.Dropout`, but allows specifying the
  value of dropped out items.
  """
  dropout_rate: float
  unk_idx: int
  deterministic: bool | None = None

  @nn.compact
  def __call__(self, inputs: Array, deterministic: bool | None = None):
    deterministic = nn.module.merge_param(
        'deterministic', self.deterministic, deterministic)
    if deterministic or self.dropout_rate == 0.:
      return inputs
    rng = self.make_rng('dropout')
    mask = jax.random.bernoulli(rng, p=self.dropout_rate, shape=inputs.shape)
    return jnp.where(mask, jnp.array([self.unk_idx]), inputs)


class Embedder(nn.Module):
  """Embeds batches of token IDs into feature space.

  Attributes:
    vocab_size: The size of the vocabulary (i.e., the number of embeddings).
    embedding_size: The dimensionality of the embeddings.
    embedding_init: The initializer used to initialize the embeddings.
    frozen: Freezes the embeddings table, keeping it fixed at initial values.
    dropout_rate: Percentage of units to drop after embedding the inputs.
    word_dropout_rate: Percentage of input words to replace with unk_idx.
    unk_idx: The index (integer) to use to replace inputs for word dropout.
  """
  vocab_size: int
  embedding_size: int
  embedding_init: Callable[...,
                           Array] = nn.initializers.normal(stddev=0.1)
  frozen: bool = False
  dropout_rate: float = 0.
  word_dropout_rate: float = 0.
  unk_idx: int | None = None
  deterministic: bool | None = None
  dtype: jnp.dtype = jnp.dtype('float32')

  def setup(self):
    self.embedding = self.param(
        'embedding',
        self.embedding_init,
        (self.vocab_size,
         self.embedding_size),
        self.dtype)
    self.dropout_layer = nn.Dropout(rate=self.dropout_rate)
    self.word_dropout_layer = WordDropout(
        dropout_rate=self.word_dropout_rate,
        unk_idx=self.unk_idx)

  def __call__(self, inputs: Array,
               deterministic: bool | None = None) -> Array:
    """Embeds the input sequences and applies word dropout and dropout.

    Args:
      inputs: Batch of input token ID sequences <int64>[batch_size, seq_length].
      deterministic: Disables dropout when set to True.

    Returns:
      The embedded inputs, shape: <float32>[batch_size, seq_length,
      embedding_size].
    """
    deterministic = nn.module.merge_param(
        'deterministic', self.deterministic, deterministic)
    inputs = self.word_dropout_layer(inputs, deterministic=deterministic)
    embedded_inputs = self.embedding[inputs]

    # Keep the embeddings fixed at initial (e.g. pretrained) values.
    if self.frozen:
      embedded_inputs = jax.lax.stop_gradient(embedded_inputs)

    return self.dropout_layer(embedded_inputs, deterministic=deterministic)


class SimpleLSTM(nn.Module):
  """A simple unidirectional LSTM."""

  @functools.partial(
      nn.transforms.scan,
      variable_broadcast='params',
      in_axes=1, out_axes=1,
      split_rngs={'params': False})
  @nn.compact
  def __call__(self, carry, x):
    return nn.OptimizedLSTMCell(features=carry[0].shape[-1])(carry, x)

  @staticmethod
  def initialize_carry(batch_dims, hidden_size):
    # Use fixed random key since default state init fn is just zeros.
    return nn.OptimizedLSTMCell(hidden_size, parent=None).initialize_carry(
        jax.random.PRNGKey(0), (batch_dims, 1)
    )


class SimpleBiLSTM(nn.Module):
  """A simple bi-directional LSTM."""
  hidden_size: int

  def setup(self):
    self.forward_lstm = SimpleLSTM()
    self.backward_lstm = SimpleLSTM()

  def __call__(self, embedded_inputs, lengths):
    batch_size = embedded_inputs.shape[0]

    # Forward LSTM.
    initial_state = SimpleLSTM.initialize_carry((batch_size,), self.hidden_size)
    _, forward_outputs = self.forward_lstm(initial_state, embedded_inputs)

    # Backward LSTM.
    reversed_inputs = flip_sequences(embedded_inputs, lengths)
    initial_state = SimpleLSTM.initialize_carry((batch_size,), self.hidden_size)
    _, backward_outputs = self.backward_lstm(initial_state, reversed_inputs)
    backward_outputs = flip_sequences(backward_outputs, lengths)

    # Concatenate the forward and backward representations.
    outputs = jnp.concatenate([forward_outputs, backward_outputs], -1)
    return outputs


class MLP(nn.Module):
  """A simple Multilayer perceptron with 1 hidden layer.

  Attributes:
    hidden_size: The size of the hidden layer.
    output_size: The size of the output.
    activation: The activation function to apply to the hidden layer.
    dropout_rate: The dropout rate applied to the hidden layer.
    output_bias: If False, do not use a bias term in the last layer.
    deterministic: Disables dropout if set to True.
  """
  hidden_size: int
  output_size: int
  activation: Callable[..., Any] = nn.tanh
  dropout_rate: float = 0.0
  output_bias: bool = False
  deterministic: bool | None = None

  def setup(self):
    self.intermediate_layer = nn.Dense(self.hidden_size)
    self.output_layer = nn.Dense(self.output_size, use_bias=self.output_bias)
    self.dropout_layer = nn.Dropout(rate=self.dropout_rate)

  def __call__(self, inputs: Array, deterministic: bool | None = None):
    """Applies the MLP to the last dimension of the inputs.

    Args:
      inputs: <float32>[batch_size, ..., input_features].
      deterministic: Disables dropout when set to True.

    Returns:
      The MLP output <float32>[batch_size, ..., output_size]
    """
    deterministic = nn.module.merge_param(
        'deterministic', self.deterministic, deterministic)
    hidden = self.intermediate_layer(inputs)
    hidden = self.activation(hidden)
    hidden = self.dropout_layer(hidden, deterministic=deterministic)
    output = self.output_layer(hidden)
    return output


class KeysOnlyMlpAttention(nn.Module):
  """Computes MLP-based attention scores based on keys alone, without a query.

  Attention scores are computed by feeding the keys through an MLP. This
  results in a single scalar per key, and for each sequence the attention
  scores are normalized using a softmax so that they sum to 1. Invalid key
  positions are ignored as indicated by the mask. This is also called
  "Bahdanau attention" and was originally proposed in:
  ```
  Bahdanau et al., 2015. Neural Machine Translation by Jointly Learning to
  Align and Translate. ICLR. https://arxiv.org/abs/1409.0473
  ```

  Attributes:
    hidden_size: The hidden size of the MLP that computes the attention score.
  """
  hidden_size: int

  @nn.compact
  def __call__(self, keys: Array, mask: Array) -> Array:
    """Applies model  to the input keys and mask.

    Args:
      keys: The inputs for which to compute an attention score. Shape:
        <float32>[batch_size, seq_length, embeddings_size].
      mask: A mask that determinines which values in `keys` are valid. Only
        values for which the mask is True will get non-zero attention scores.
        <bool>[batch_size, seq_length].

    Returns:
      The normalized attention scores. <float32>[batch_size, seq_length].
    """
    hidden = nn.Dense(self.hidden_size, name='keys', use_bias=False)(keys)
    energy = nn.tanh(hidden)
    scores = nn.Dense(1, name='energy', use_bias=False)(energy)
    scores = scores.squeeze(-1)  # New shape: <float32>[batch_size, seq_len].
    scores = jnp.where(mask, scores, -jnp.inf)  # Using exp(-inf) = 0 below.
    scores = nn.softmax(scores, axis=-1)

    # Captures the scores if 'intermediates' is mutable, otherwise does nothing.
    self.sow('intermediates', 'attention', scores)

    return scores


class AttentionClassifier(nn.Module):
  """A classifier that uses attention to summarize the inputs.

  Attributes:
    hidden_size: The hidden size of the MLP classifier.
    output_size: The number of output classes for the classifier.
    dropout_rate: The dropout rate applied over the encoded_inputs, the summary
      of the inputs, and inside the MLP. Applied when `deterministic` is False.
    deterministic: Disables dropout if True.
  """
  hidden_size: int
  output_size: int
  dropout_rate: float = 0.
  deterministic: bool | None = None

  def setup(self):
    self.dropout_layer = nn.Dropout(rate=self.dropout_rate)
    self.keys_only_mlp_attention = KeysOnlyMlpAttention(
        hidden_size=self.hidden_size)
    self.mlp = MLP(
        hidden_size=self.hidden_size,
        output_size=self.output_size,
        output_bias=False,
        dropout_rate=self.dropout_rate)

  def __call__(self, encoded_inputs: Array, lengths: Array,
               deterministic: bool | None = None) -> Array:
    """Applies model to the encoded inputs.

    Args:
      encoded_inputs: The inputs (e.g., sentences) that have already been
        encoded by some encoder, e.g., an LSTM. <float32>[batch_size,
        seq_length, encoded_inputs_size].
      lengths: The lengths of the inputs. <int64>[batch_size].
      deterministic: Disables dropout when set to True.

    Returns:
      An array of logits <float32>[batch_size, output_size].
    """
    deterministic = nn.module.merge_param(
        'deterministic', self.deterministic, deterministic)
    encoded_inputs = self.dropout_layer(
        encoded_inputs, deterministic=deterministic)

    # Compute attention. attention.shape: <float32>[batch_size, seq_len].
    mask = sequence_mask(lengths, encoded_inputs.shape[1])
    attention = self.keys_only_mlp_attention(encoded_inputs, mask)

    # Summarize the inputs by taking their weighted sum using attention scores.
    context = jnp.expand_dims(attention, 1) @ encoded_inputs
    context = context.squeeze(1)  # <float32>[batch_size, encoded_inputs_size]
    context = self.dropout_layer(context, deterministic=deterministic)

    # Make the final prediction from the context vector (the summarized inputs).
    logits = self.mlp(context, deterministic=deterministic)
    return logits


class TextClassifier(nn.Module):
  """A Text Classification model."""

  embedding_size: int
  hidden_size: int
  vocab_size: int
  output_size: int

  dropout_rate: float
  word_dropout_rate: float
  unk_idx: int = 1
  deterministic: bool | None = None

  def setup(self):
    self.embedder = Embedder(
        vocab_size=self.vocab_size,
        embedding_size=self.embedding_size,
        dropout_rate=self.dropout_rate,
        word_dropout_rate=self.word_dropout_rate,
        unk_idx=self.unk_idx)
    self.encoder = SimpleBiLSTM(hidden_size=self.hidden_size)
    self.classifier = AttentionClassifier(
        hidden_size=self.hidden_size,
        output_size=self.output_size,
        dropout_rate=self.dropout_rate)

  def embed_token_ids(self, token_ids: Array,
                      deterministic: bool | None = None) -> Array:
    deterministic = nn.module.merge_param(
        'deterministic', self.deterministic, deterministic)
    return self.embedder(token_ids, deterministic=deterministic)

  def logits_from_embedded_inputs(
      self, embedded_inputs: Array, lengths: Array,
      deterministic: bool | None = None) -> Array:
    deterministic = nn.module.merge_param(
        'deterministic', self.deterministic, deterministic)
    encoded_inputs = self.encoder(embedded_inputs, lengths)
    return self.classifier(
        encoded_inputs, lengths, deterministic=deterministic)

  def __call__(self, token_ids: Array, lengths: Array,
               deterministic: bool | None = None) -> Array:
    """Embeds the token IDs, encodes them, and classifies with attention."""
    embedded_inputs = self.embed_token_ids(
        token_ids, deterministic=deterministic)
    logits = self.logits_from_embedded_inputs(
        embedded_inputs, lengths, deterministic=deterministic)
    return logits

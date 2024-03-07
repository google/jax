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
"""Transformer-based language model from Flax nlp_seq example, see:

https://github.com/google/flax/tree/main/examples/lm1b
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from flax import linen as nn
from flax import struct
import jax.numpy as jnp
import numpy as np


@struct.dataclass
class TransformerConfig:
  """Global hyperparameters used to minimize obnoxious kwarg plumbing."""
  vocab_size: int
  output_vocab_size: int
  dtype: Any = jnp.float32
  emb_dim: int = 512
  num_heads: int = 8
  num_layers: int = 6
  qkv_dim: int = 512
  mlp_dim: int = 2048
  max_len: int = 2048
  dropout_rate: float = 0.3
  attention_dropout_rate: float = 0.3
  kernel_init: Callable = nn.initializers.xavier_uniform()
  bias_init: Callable = nn.initializers.normal(stddev=1e-6)
  posemb_init: Callable | None = None


def sinusoidal_init(max_len=2048):
  """1D Sinusoidal Position Embedding Initializer.

  Args:
      max_len: maximum possible length for the input

  Returns:
      output: init function returning `(1, max_len, d_feature)`
  """

  def init(key, shape, dtype=np.float32):
    """Sinusoidal init."""
    del key, dtype
    d_feature = shape[-1]
    pe = np.zeros((max_len, d_feature), dtype=np.float32)
    position = np.arange(0, max_len)[:, np.newaxis]
    div_term = np.exp(
        np.arange(0, d_feature, 2) * -(np.log(10000.0) / d_feature))
    pe[:, 0::2] = np.sin(position * div_term)
    pe[:, 1::2] = np.cos(position * div_term)
    pe = pe[np.newaxis, :, :]  # [1, max_len, d_feature]
    return jnp.array(pe)

  return init


class AddPositionEmbs(nn.Module):
  """Adds (optionally learned) positional embeddings to the inputs.

  Attributes:
    config: TransformerConfig dataclass containing hyperparameters.
  """
  config: TransformerConfig

  @nn.compact
  def __call__(self, inputs):
    """Applies AddPositionEmbs module.

    By default this layer uses a fixed sinusoidal embedding table. If a
    learned position embedding is desired, pass an initializer to
    posemb_init in the configuration.

    Args:
      inputs: input data.

    Returns:
      output: `(bs, timesteps, in_dim)`
    """
    config = self.config
    # inputs.shape is (batch_size, seq_len, emb_dim)
    assert inputs.ndim == 3, ('Number of dimensions should be 3,'
                              ' but it is: %d' % inputs.ndim)
    length = inputs.shape[1]
    pos_emb_shape = (1, config.max_len, inputs.shape[-1])
    if config.posemb_init is None:
      # Use a fixed (non-learned) sinusoidal position embedding.
      pos_embedding = sinusoidal_init(max_len=config.max_len)(None,
                                                              pos_emb_shape,
                                                              None)
    else:
      pos_embedding = self.param('pos_embedding', config.posemb_init,
                                 pos_emb_shape)
    pe = pos_embedding[:, :length, :]
    return inputs + pe


class MlpBlock(nn.Module):
  """Transformer MLP / feed-forward block.

  Attributes:
    config: TransformerConfig dataclass containing hyperparameters.
    out_dim: optionally specify out dimension.
  """
  config: TransformerConfig
  out_dim: int | None = None

  @nn.compact
  def __call__(self, inputs, deterministic=True):
    """Applies Transformer MlpBlock module."""
    config = self.config
    actual_out_dim = (inputs.shape[-1] if self.out_dim is None
                      else self.out_dim)
    x = nn.Dense(
        config.mlp_dim,
        dtype=config.dtype,
        kernel_init=config.kernel_init,
        bias_init=config.bias_init)(
            inputs)
    x = nn.elu(x)
    x = nn.Dropout(rate=config.dropout_rate)(x, deterministic=deterministic)
    output = nn.Dense(
        actual_out_dim,
        dtype=config.dtype,
        kernel_init=config.kernel_init,
        bias_init=config.bias_init)(
            x)
    output = nn.Dropout(rate=config.dropout_rate)(
        output, deterministic=deterministic)
    return output


class Encoder1DBlock(nn.Module):
  """Transformer encoder layer.

  Attributes:
    config: TransformerConfig dataclass containing hyperparameters.
  """
  config: TransformerConfig

  @nn.compact
  def __call__(self, inputs, deterministic):
    """Applies Encoder1DBlock module.

    Args:
      inputs: input data.
      deterministic: if true dropout is applied otherwise not.

    Returns:
      output after transformer encoder block.
    """
    config = self.config

    # Attention block.
    assert inputs.ndim == 3
    x = nn.LayerNorm(dtype=config.dtype)(inputs)
    x = nn.SelfAttention(
        num_heads=config.num_heads,
        dtype=config.dtype,
        qkv_features=config.qkv_dim,
        kernel_init=config.kernel_init,
        bias_init=config.bias_init,
        use_bias=False,
        broadcast_dropout=False,
        dropout_rate=config.attention_dropout_rate,
        deterministic=deterministic)(
            x)

    x = nn.Dropout(rate=config.dropout_rate)(x, deterministic=deterministic)
    x = x + inputs

    # MLP block.
    y = nn.LayerNorm(dtype=config.dtype)(x)
    y = MlpBlock(config=config)(y, deterministic=deterministic)
    return x + y


class Transformer(nn.Module):
  """Transformer Model for sequence tagging."""

  config: TransformerConfig

  @nn.compact
  def __call__(self, inputs, *, train):
    """Applies Transformer model on the inputs.

    Args:
      inputs: input data
      train: if it is training.

    Returns:
      output of a transformer encoder.

    """
    assert inputs.ndim == 2  # (batch, len)

    config = self.config

    x = inputs.astype('int32')
    x = nn.Embed(
        num_embeddings=config.vocab_size, features=config.emb_dim,
        name='embed')(
            x)
    x = nn.Dropout(rate=config.dropout_rate)(x, deterministic=not train)
    x = AddPositionEmbs(config)(x)

    for _ in range(config.num_layers):
      x = Encoder1DBlock(config)(x, deterministic=not train)

    x = nn.LayerNorm(dtype=config.dtype)(x)
    logits = nn.Dense(
        config.output_vocab_size,
        kernel_init=config.kernel_init,
        bias_init=config.bias_init)(
            x)
    return logits

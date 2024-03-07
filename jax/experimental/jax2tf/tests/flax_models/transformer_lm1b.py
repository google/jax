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
"""Transformer-based language model from Flax lm1b example, see:

https://github.com/google/flax/tree/main/examples/lm1b
"""

# pylint: disable=attribute-defined-outside-init
# See issue #620.
# pytype: disable=wrong-arg-count
# pytype: disable=wrong-keyword-args
# pytype: disable=attribute-error

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from flax import linen as nn
from flax import struct
from jax import lax
import jax.numpy as jnp
import numpy as np

@struct.dataclass
class TransformerConfig:
  """Global hyperparameters used to minimize obnoxious kwarg plumbing."""
  vocab_size: int
  output_vocab_size: int
  share_embeddings: bool = False
  logits_via_embedding: bool = False
  dtype: Any = jnp.float32
  emb_dim: int = 512
  num_heads: int = 8
  num_layers: int = 6
  qkv_dim: int = 512
  mlp_dim: int = 2048
  max_len: int = 2048
  dropout_rate: float = 0.1
  attention_dropout_rate: float = 0.1
  deterministic: bool = False
  decode: bool = False
  kernel_init: Callable = nn.initializers.xavier_uniform()
  bias_init: Callable = nn.initializers.normal(stddev=1e-6)
  posemb_init: Callable | None = None


def shift_right(x, axis=1):
  """Shift the input to the right by padding and slicing on axis."""
  pad_widths = [(0, 0)] * len(x.shape)
  pad_widths[axis] = (1, 0)
  padded = jnp.pad(
      x, pad_widths, mode='constant', constant_values=x.dtype.type(0))
  return lax.dynamic_slice_in_dim(padded, 0, padded.shape[axis] - 1, axis)


def shift_inputs(x, segment_ids=None, axis=1):
  """Shift inputs and replace EOS by 0 for packed inputs."""
  shifted = shift_right(x, axis=axis)
  # For packed targets, the first shifted token of a new sequence is made
  # 0, rather than being the EOS token for the last sequence.
  if segment_ids is not None:
    shifted *= (segment_ids == shift_right(segment_ids, axis=axis))
  return shifted


def sinusoidal_init(max_len=2048,
                    min_scale=1.0,
                    max_scale=10000.0):
  """1D Sinusoidal Position Embedding Initializer.

  Args:
      max_len: maximum possible length for the input.
      min_scale: float: minimum frequency-scale in sine grating.
      max_scale: float: maximum frequency-scale in sine grating.

  Returns:
      output: init function returning `(1, max_len, d_feature)`
  """

  def init(key, shape, dtype=np.float32):
    """Sinusoidal init."""
    del key, dtype
    d_feature = shape[-1]
    pe = np.zeros((max_len, d_feature), dtype=np.float32)
    position = np.arange(0, max_len)[:, np.newaxis]
    scale_factor = -np.log(max_scale / min_scale) / (d_feature // 2 - 1)
    div_term = min_scale * np.exp(np.arange(0, d_feature // 2) * scale_factor)
    pe[:, :d_feature // 2] = np.sin(position * div_term)
    pe[:, d_feature // 2: 2 * (d_feature // 2)] = np.cos(position * div_term)
    pe = pe[np.newaxis, :, :]  # [1, max_len, d_feature]
    return jnp.array(pe)

  return init


class AddPositionEmbs(nn.Module):
  """Adds (optionally learned) positional embeddings to the inputs.

  Args:
    config: TransformerConfig dataclass containing hyperparameters.
    decode: whether to run in single-position autoregressive mode.
  """
  config: TransformerConfig
  decode: bool = False

  @nn.compact
  def __call__(self,
               inputs,
               inputs_positions=None):
    """Applies AddPositionEmbs module.

    By default this layer uses a fixed sinusoidal embedding table. If a
    learned position embedding is desired, pass an initializer to
    posemb_init in the configuration.

    Args:
      inputs: input data.
      inputs_positions: input position indices for packed sequences.

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

    # We use a cache position index for tracking decoding position.
    if self.decode:
      is_initialized = self.has_variable('cache', 'cache_index')
      cache_index = self.variable('cache', 'cache_index',
                                  lambda: jnp.array(0, dtype=jnp.uint32))
      if is_initialized:
        i = cache_index.value
        cache_index.value = i + 1
        _, _, df = pos_embedding.shape
        pe = lax.dynamic_slice(pos_embedding,
                               jnp.array((0, i, 0)),
                               (1, 1, df))
    if inputs_positions is None:
      # normal unpacked case:
      return inputs + pe
    else:
      # for packed data we need to use known position indices:
      return inputs + jnp.take(pe[0], inputs_positions, axis=0)


class MlpBlock(nn.Module):
  """Transformer MLP / feed-forward block.

  Args:
    config: TransformerConfig dataclass containing hyperparameters.
    out_dim: optionally specify out dimension.
  """
  config: TransformerConfig
  out_dim: int | None = None

  @nn.compact
  def __call__(self, inputs):
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
    x = nn.relu(x)
    x = nn.Dropout(rate=config.dropout_rate)(
        x, deterministic=config.deterministic)
    output = nn.Dense(
        actual_out_dim,
        dtype=config.dtype,
        kernel_init=config.kernel_init,
        bias_init=config.bias_init)(
            x)
    output = nn.Dropout(rate=config.dropout_rate)(
        output, deterministic=config.deterministic)
    return output


class EncoderDecoder1DBlock(nn.Module):
  """Transformer encoder-decoder layer.

  Args:
    config: TransformerConfig dataclass containing hyperparameters.
  """
  config: TransformerConfig

  @nn.compact
  def __call__(self,
               inputs,
               decoder_mask=None,
               encoder_decoder_mask=None):
    """Applies EncoderDecoder1DBlock module.

    Args:
      inputs: input data for decoder
      decoder_mask: decoder self-attention mask.
      encoder_decoder_mask: encoder-decoder attention mask.

    Returns:
      output after transformer encoder-decoder block.
    """
    config = self.config

    # Decoder block.
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
        deterministic=config.deterministic,
        decode=config.decode)(x, decoder_mask)
    x = nn.Dropout(rate=config.dropout_rate)(
        x, deterministic=config.deterministic)
    x = x + inputs

    # MLP block.
    z = nn.LayerNorm(dtype=config.dtype)(x)
    z = MlpBlock(config=config)(z)

    return x + z


class Decoder(nn.Module):
  """Transformer Model Decoder for sequence to sequence translation.

  Args:
    config: TransformerConfig dataclass containing hyperparameters.
    shared_embedding: a shared embedding layer to use.
  """
  config: TransformerConfig
  shared_embedding: Any = None

  @nn.compact
  def __call__(self,
               inputs,
               inputs_positions=None,
               inputs_segmentation=None,
               decoder_mask=None,
               encoder_decoder_mask=None):
    """Applies Transformer model on the inputs.

    Args:
      encoded: encoded input data from encoder.
      inputs: input data.
      inputs_positions: input subsequence positions for packed examples.
      inputs_segmentation: input segmentation info for packed examples.
      decoder_mask: decoder self-attention mask.
      encoder_decoder_mask: encoder-decoder attention mask.

    Returns:
      output of a transformer decoder.
    """
    config = self.config
    assert inputs.ndim == 2  # (batch, len)

    # Target Embedding
    if self.shared_embedding is None:
      output_embed = nn.Embed(
          num_embeddings=config.output_vocab_size,
          features=config.emb_dim,
          embedding_init=nn.initializers.normal(stddev=1.0))
    else:
      output_embed = self.shared_embedding

    y = inputs.astype('int32')
    if not config.decode:
      y = shift_inputs(y, segment_ids=inputs_segmentation)
    y = output_embed(y)
    y = AddPositionEmbs(
        config=config, decode=config.decode, name='posembed_output')(
            y, inputs_positions=inputs_positions)
    y = nn.Dropout(rate=config.dropout_rate)(
        y, deterministic=config.deterministic)

    y = y.astype(config.dtype)

    # Target-Input Decoder
    for lyr in range(config.num_layers):
      y = EncoderDecoder1DBlock(
          config=config, name=f'encoderdecoderblock_{lyr}')(
              y,
              decoder_mask=decoder_mask,
              encoder_decoder_mask=encoder_decoder_mask)
    y = nn.LayerNorm(dtype=config.dtype, name='encoderdecoder_norm')(y)

    # Decoded Logits
    if config.logits_via_embedding:
      # Use the transpose of embedding matrix for logit transform.
      logits = output_embed.attend(y.astype(jnp.float32))
      # Correctly normalize pre-softmax logits for this shared case.
      logits = logits / jnp.sqrt(y.shape[-1])
    else:
      logits = nn.Dense(
          config.output_vocab_size,
          dtype=config.dtype,
          kernel_init=config.kernel_init,
          bias_init=config.bias_init,
          name='logitdense')(
              y)
    return logits


class TransformerLM(nn.Module):
  """Transformer pure decoder stack for language modelling.

  Args:
    config: TransformerConfig dataclass containing hyperparameters.
  """
  config: TransformerConfig

  @nn.compact
  def __call__(self,
               inputs,
               inputs_positions=None,
               inputs_segmentation=None):
    """Applies TransformerLM on the inputs.

    Args:
      inputs: target data.
      inputs_positions: input subsequence positions for packed examples.
      inputs_segmentation: input segmentation info for packed examples.

    Returns:
      logits array from transformer decoder.
    """
    config = self.config

    # Make padding attention masks.
    if config.decode:
      # for fast autoregressive decoding we use no decoder mask
      decoder_mask = None
    else:
      decoder_mask = nn.combine_masks(
          nn.make_attention_mask(inputs > 0, inputs > 0, dtype=config.dtype),
          nn.make_causal_mask(inputs, dtype=config.dtype))

    # Add segmentation block-diagonal attention masks if using segmented data.
    if inputs_segmentation is not None:
      decoder_mask = nn.combine_masks(
          decoder_mask,
          nn.make_attention_mask(
              inputs_segmentation,
              inputs_segmentation,
              jnp.equal,
              dtype=config.dtype))

    logits = Decoder(
        config=config, shared_embedding=None, name='decoder')(
            inputs,
            inputs_positions=inputs_positions,
            inputs_segmentation=inputs_segmentation,
            decoder_mask=decoder_mask,
            encoder_decoder_mask=None)
    return logits.astype(self.config.dtype)

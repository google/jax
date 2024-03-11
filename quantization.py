from functools import partial

import jax
import jax.numpy as jnp

from jax._src.state.discharge import run_state, run_state2

# quantization helpers

FAKE_E4M3 = jnp.float16
FAKE_E5M2 = jnp.bfloat16
E4M3_MAX = 448
E5M2_MAX = 57344

def get_fp8_max(fake_dtype):
  if fake_dtype == FAKE_E4M3:
    return E4M3_MAX
  elif fake_dtype == FAKE_E5M2:
    return E5M2_MAX
  else:
    raise ValueError('Only FAKE_E4M3 and FAKE_E5M2 supported')

def quantize(x, quantized_dtype, scale):
  dtype_max = get_fp8_max(quantized_dtype)
  scaled_x = jnp.clip(x / scale, -dtype_max, dtype_max)
  return scaled_x.astype(quantized_dtype)

def dequantize(x, wide_dtype, scale):
  return x.astype(wide_dtype) * scale

def quantize_dequantize(x, quantized_dtype, scale):
  orig_dtype = x.dtype
  qx = quantize(x, quantized_dtype, scale)
  return dequantize(qx, orig_dtype, scale)

def compute_new_scale(x, quantized_dtype, scale):
  dtype_max = get_fp8_max(quantized_dtype)
  amax = jnp.max(jnp.abs(x)).astype(scale.dtype)
  # Ensure scale != 0 and avoid divide-by-zero.
  amax = jnp.maximum(amax, 2**-10)
  return 1.1 * amax / dtype_max


def qdq(x, dtype, scale_ref):
  qx = quantize_dequantize(x, dtype, scale_ref[()])
  scale_ref[()] = compute_new_scale(x, dtype, scale_ref[()])
  return qx

@jax.custom_gradient
def out_qdq(out, out_scale_ref, out_grad_scale_ref):
  bwd = lambda g: qdq(g, FAKE_E5M2, out_grad_scale_ref)
  return qdq(out, FAKE_E4M3, out_scale_ref), bwd


def predict(params, scale_refs, inputs):
  for (W, b), (in_scale_ref, out_scale_ref, out_grad_scale_ref) in zip(params, scale_refs):
    inputs = qdq(inputs, FAKE_E4M3, in_scale_ref)
    outputs = jnp.dot(inputs, W) + b
    inputs = out_qdq(jax.nn.relu(outputs), out_scale_ref, out_grad_scale_ref)
  return outputs

def loss(params, scale_refs, batch):
  inputs, targets = batch
  predictions = predict(params, scale_refs, inputs)
  return jnp.mean(jnp.sum((predictions - targets)**2, axis=-1))


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

layer_sizes = [16, 10]
batch_size = 32
params, batch = init_model(jax.random.PRNGKey(0), layer_sizes, batch_size)

init_scales = [(jnp.float32(32),) * 3 for _ in range(len(params))]


loss_val, scales = run_state2(
    lambda scale_refs: loss(params, scale_refs, batch))(init_scales)


# [ ] pjit discharge rule
# [ ] make should_discharge not optional
# [ ] fix scan should_discharge, with outer refs (need a test case)

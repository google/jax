import os
os.environ['XLA_FLAGS'] = '--xla_force_host_platform_device_count=8'

import pdb, sys, traceback
def info(type, value, tb):
    traceback.print_exception(type, value, tb)
    pdb.pm()
sys.excepthook = info


import numpy as np

import jax
import jax.numpy as jnp
jax.config.update('jax_platforms', 'cpu')
jax.config.update('jax_traceback_filtering', 'off')

# TODO a verbose mode can say what was the _same_, but let's focus on what was
# the _difference_

# TODO only print for things that are not inline=True?

@jax.jit
def f(x, y):
  return
  return jnp.sin(x) + y['hi']

f(1, {'hi': np.arange(3)})
f(1, {'hi': np.arange(3)})

f(1, {'hi': np.arange(4)})
f(1, y={'hi': np.arange(4)})
f(x=1, y={'hi': np.arange(4)})
# # f(1, {'hi': np.arange(4), 'bye': np.arange(5)})


# @jax.jit
# def f(x):
#   return jnp.sin(x)

# x = jax.device_put(jnp.arange(3), jax.devices()[0])
# f(x)  # I've never seen this function before...
# print()


# x = jax.device_put(jnp.arange(3), jax.devices()[1])
# f(x)  # I've never seen this function before...

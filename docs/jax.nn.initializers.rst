``jax.nn.initializers`` module
==============================

.. currentmodule:: jax.nn.initializers

.. automodule:: jax.nn.initializers


Initializers
------------

This module provides common neural network layer initializers,
consistent with definitions used in Keras and Sonnet.

An initializer is a function that takes three arguments:
``(key, shape, dtype)`` and returns an array with dimensions ``shape`` and
data type ``dtype``. Argument ``key`` is a PRNG key (e.g. from
:func:`jax.random.key`), used to generate random numbers to initialize the array.

.. autosummary::
  :toctree: _autosummary

    constant
    delta_orthogonal
    glorot_normal
    glorot_uniform
    he_normal
    he_uniform
    lecun_normal
    lecun_uniform
    normal
    ones
    orthogonal
    truncated_normal
    uniform
    variance_scaling
    zeros

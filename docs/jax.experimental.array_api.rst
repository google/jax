``jax.experimental.array_api`` module
=====================================

.. note::
  The ``jax.experimental.array_api`` module is deprecated as of JAX v0.4.32, and
  importing ``jax.experimental.array_api`` is no longer necessary. {mod}`jax.numpy`
  implements the array API standard directly by default. See :ref:`python-array-api`
  for details.

This module includes experimental JAX support for the `Python array API standard`_.
Support for this is currently experimental and not fully complete.

Example Usage::

  >>> from jax.experimental import array_api as xp

  >>> xp.__array_api_version__
  '2023.12'

  >>> arr = xp.arange(1000)

  >>> arr.sum()
  Array(499500, dtype=int32)

The ``xp`` namespace is the array API compliant analog of :mod:`jax.numpy`,
and implements most of the API listed in the standard.

.. _Python array API standard: https://data-apis.org/array-api/

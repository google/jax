``jax.export`` module
=====================

.. automodule:: jax.export

:mod:`jax.export` is a library for exporting and serializing JAX functions
for persistent archival.

See the :ref:`export` documentation.

Classes
-------

.. autosummary::
  :toctree: _autosummary

  Exported
  DisabledSafetyCheck

Functions
---------

.. autosummary::
  :toctree: _autosummary

  export
  deserialize
  minimum_supported_calling_convention_version
  maximum_supported_calling_convention_version
  default_export_platform

Functions related to shape polymorphism
---------------------------------------

.. autosummary::
  :toctree: _autosummary

  symbolic_shape
  symbolic_args_specs
  is_symbolic_dim
  SymbolicScope

Constants
---------

.. data:: jax.export.minimum_supported_serialization_version

   The minimum supported serialization version; see :ref:`export-calling-convention-version`.

.. data:: jax.export.maximum_supported_serialization_version

   The maximum supported serialization version; see :ref:`export-calling-convention-version`.

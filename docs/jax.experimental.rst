.. currentmodule:: jax.experimental

``jax.experimental`` module
===========================

``jax.experimental.optix`` has been moved into its own Python package
(https://github.com/deepmind/optax).

``jax.experimental.ann`` has been moved into ``jax.lax``.

Experimental Modules
--------------------

.. toctree::
    :maxdepth: 1

    jax.experimental.array_api
    jax.experimental.checkify
    jax.experimental.host_callback
    jax.experimental.maps
    jax.experimental.pjit
    jax.experimental.sparse
    jax.experimental.jet
    jax.experimental.custom_partitioning
    jax.experimental.multihost_utils
    jax.experimental.compilation_cache
    jax.experimental.key_reuse
    jax.experimental.mesh_utils
    jax.experimental.serialize_executable
    jax.experimental.shard_map
    jax.experimental.pallas

Experimental APIs
-----------------

.. autosummary::
   :toctree: _autosummary

   enable_x64
   disable_x64

   jax.experimental.checkify.checkify
   jax.experimental.checkify.check
   jax.experimental.checkify.check_error

.. _pallas:

Pallas: a JAX kernel language
=============================
Pallas is an extension to JAX that enables writing custom kernels for GPU and TPU.
It aims to provide fine-grained control over the generated code, combined with
the high-level ergonomics of JAX tracing and the `jax.numpy` API.

This section contains tutorials, guides and examples for using Pallas.
See also the :class:`jax.experimental.pallas` module API documentation.

.. warning::
  Pallas is experimental and is changing frequently.
  See the :ref:`pallas-changelog` for the recent changes.

  You can expect to encounter errors and unimplemented cases, e.g., when
  lowering of high-level JAX concepts that would require emulation,
  or simply because Pallas is still under development.

.. toctree::
   :caption: Guides
   :maxdepth: 2

   quickstart
   design
   grid_blockspec


.. toctree::
   :caption: Platform Features
   :maxdepth: 2

   tpu/index

.. toctree::
   :maxdepth: 1

   CHANGELOG

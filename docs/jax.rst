.. currentmodule:: jax

Public API: ``jax`` package
===========================

Subpackages
-----------

.. toctree::
   :maxdepth: 1

   jax.numpy
   jax.scipy
   jax.lax
   jax.random
   jax.sharding
   jax.debug
   jax.dlpack
   jax.distributed
   jax.dtypes
   jax.flatten_util
   jax.image
   jax.nn
   jax.ops
   jax.profiler
   jax.stages
   jax.tree
   jax.tree_util
   jax.typing
   jax.export
   jax.extend
   jax.example_libraries
   jax.experimental

.. toctree::
   :hidden:

   jax.lib

Configuration
-------------

.. autosummary::
   :toctree: _autosummary

   config
   check_tracer_leaks
   checking_leaks
   debug_nans
   debug_infs
   default_device
   default_matmul_precision
   default_prng_impl
   enable_checks
   enable_custom_prng
   enable_custom_vjp_by_custom_transpose
   log_compiles
   numpy_rank_promotion
   transfer_guard

.. _jax-jit:

Just-in-time compilation (:code:`jit`)
--------------------------------------

.. autosummary::
  :toctree: _autosummary

    jit
    disable_jit
    ensure_compile_time_eval
    make_jaxpr
    eval_shape
    ShapeDtypeStruct
    device_put
    device_put_replicated
    device_put_sharded
    device_get
    default_backend
    named_call
    named_scope
    block_until_ready

.. _jax-grad:

Automatic differentiation
-------------------------

.. autosummary::
  :toctree: _autosummary

    grad
    value_and_grad
    jacobian
    jacfwd
    jacrev
    hessian
    jvp
    linearize
    linear_transpose
    vjp
    custom_gradient
    closure_convert
    checkpoint
    remat

``custom_jvp``
~~~~~~~~~~~~~~

.. autosummary::
  :toctree: _autosummary

  custom_jvp
  custom_jvp.defjvp
  custom_jvp.defjvps

``custom_vjp``
~~~~~~~~~~~~~~

.. autosummary::
  :toctree: _autosummary

  custom_vjp
  custom_vjp.defvjp

jax.Array (:code:`jax.Array`)
-----------------------------

.. autosummary::
  :toctree: _autosummary

    Array
    make_array_from_callback
    make_array_from_single_device_arrays
    make_array_from_process_local_data

Array properties and methods
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autosummary::
  :toctree: _autosummary

    Array.addressable_shards
    Array.all
    Array.any
    Array.argmax
    Array.argmin
    Array.argpartition
    Array.argsort
    Array.astype
    Array.at
    Array.choose
    Array.clip
    Array.compress
    Array.conj
    Array.conjugate
    Array.copy
    Array.copy_to_host_async
    Array.cumprod
    Array.cumsum
    Array.device
    Array.diagonal
    Array.dot
    Array.dtype
    Array.flat
    Array.flatten
    Array.global_shards
    Array.imag
    Array.is_fully_addressable
    Array.is_fully_replicated
    Array.item
    Array.itemsize
    Array.max
    Array.mean
    Array.min
    Array.nbytes
    Array.ndim
    Array.nonzero
    Array.prod
    Array.ptp
    Array.ravel
    Array.real
    Array.repeat
    Array.reshape
    Array.round
    Array.searchsorted
    Array.shape
    Array.sharding
    Array.size
    Array.sort
    Array.squeeze
    Array.std
    Array.sum
    Array.swapaxes
    Array.take
    Array.to_device
    Array.trace
    Array.transpose
    Array.var
    Array.view
    Array.T
    Array.mT

Vectorization (:code:`vmap`)
----------------------------

.. autosummary::
  :toctree: _autosummary

    vmap
    numpy.vectorize

Parallelization (:code:`pmap`)
------------------------------

.. autosummary::
  :toctree: _autosummary

    pmap
    devices
    local_devices
    process_index
    device_count
    local_device_count
    process_count
    process_indices

Callbacks
---------

.. autosummary::
  :toctree: _autosummary

    pure_callback
    experimental.io_callback
    debug.callback
    debug.print

Miscellaneous
-------------

.. autosummary::
  :toctree: _autosummary

    Device
    print_environment_info
    live_arrays
    clear_caches

``jax.lax`` module
==================

.. automodule:: jax.lax

:mod:`jax.lax` is a library of primitives operations that underpins libraries
such as :mod:`jax.numpy`. Transformation rules, such as JVP and batching rules,
are typically defined as transformations on :mod:`jax.lax` primitives.

Many of the primitives are thin wrappers around equivalent XLA operations,
described by the `XLA operation semantics
<https://www.tensorflow.org/xla/operation_semantics>`_ documentation. In a few
cases JAX diverges from XLA, usually to ensure that the set of operations is
closed under the operation of JVP and transpose rules.

Where possible, prefer to use libraries such as :mod:`jax.numpy` instead of
using :mod:`jax.lax` directly. The :mod:`jax.numpy` API follows NumPy, and is
therefore more stable and less likely to change than the :mod:`jax.lax` API.

Operators
---------

.. autosummary::
  :toctree: _autosummary

    abs
    acos
    acosh
    add
    after_all
    approx_max_k
    approx_min_k
    argmax
    argmin
    asin
    asinh
    atan
    atan2
    atanh
    batch_matmul
    bessel_i0e
    bessel_i1e
    betainc
    bitcast_convert_type
    bitwise_and
    bitwise_not
    bitwise_or
    bitwise_xor
    population_count
    broadcast
    broadcast_in_dim
    broadcast_shapes
    broadcast_to_rank
    broadcasted_iota
    cbrt
    ceil
    clamp
    clz
    collapse
    complex
    concatenate
    conj
    conv
    convert_element_type
    conv_dimension_numbers
    conv_general_dilated
    conv_general_dilated_local
    conv_general_dilated_patches
    conv_transpose
    conv_with_general_padding
    cos
    cosh
    cumlogsumexp
    cummax
    cummin
    cumprod
    cumsum
    digamma
    div
    dot
    dot_general
    dynamic_index_in_dim
    dynamic_slice
    dynamic_slice_in_dim
    dynamic_update_index_in_dim
    dynamic_update_slice
    dynamic_update_slice_in_dim
    eq
    erf
    erfc
    erf_inv
    exp
    expand_dims
    expm1
    fft
    floor
    full
    full_like
    gather
    ge
    gt
    igamma
    igammac
    imag
    index_in_dim
    index_take
    integer_pow
    iota
    is_finite
    le
    lgamma
    log
    log1p
    logistic
    lt
    max
    min
    mul
    ne
    neg
    nextafter
    pad
    polygamma
    population_count
    pow
    random_gamma_grad
    real
    reciprocal
    reduce
    reduce_precision
    reduce_window
    rem
    reshape
    rev
    rng_bit_generator
    rng_uniform
    round
    rsqrt
    scatter
    scatter_add
    scatter_apply
    scatter_max
    scatter_min
    scatter_mul
    shift_left
    shift_right_arithmetic
    shift_right_logical
    sign
    sin
    sinh
    slice
    slice_in_dim
    sort
    sort_key_val
    sqrt
    square
    squeeze
    sub
    tan
    tanh
    top_k
    transpose
    zeros_like_array
    zeta

.. _lax-control-flow:

Control flow operators
----------------------

.. autosummary::
  :toctree: _autosummary

    associative_scan
    cond
    fori_loop
    map
    scan
    select
    select_n
    switch
    while_loop

Custom gradient operators
-------------------------

.. autosummary::
  :toctree: _autosummary

    stop_gradient
    custom_linear_solve
    custom_root

.. _jax-parallel-operators:

Parallel operators
------------------

.. autosummary::
  :toctree: _autosummary

    all_gather
    all_to_all
    pdot
    psum
    psum_scatter
    pmax
    pmin
    pmean
    ppermute
    pshuffle
    pswapaxes
    axis_index

Sharding-related operators
--------------------------
.. autosummary::
  :toctree: _autosummary

    with_sharding_constraint

Linear algebra operators (jax.lax.linalg)
-----------------------------------------

.. automodule:: jax.lax.linalg

.. autosummary::
  :toctree: _autosummary

    cholesky
    eig
    eigh
    hessenberg
    lu
    householder_product
    qdwh
    qr
    schur
    svd
    triangular_solve
    tridiagonal
    tridiagonal_solve

Argument classes
----------------

.. currentmodule:: jax.lax

.. autoclass:: ConvDimensionNumbers
.. autoclass:: ConvGeneralDilatedDimensionNumbers
.. autoclass:: GatherDimensionNumbers
.. autoclass:: GatherScatterMode
.. autoclass:: Precision
.. autoclass:: PrecisionLike
.. autoclass:: RoundingMethod
.. autoclass:: ScatterDimensionNumbers

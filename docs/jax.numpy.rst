``jax.numpy`` module
====================

.. currentmodule:: jax.numpy

.. automodule:: jax.numpy

Implements the NumPy API, using the primitives in :mod:`jax.lax`.

While JAX tries to follow the NumPy API as closely as possible, sometimes JAX
cannot follow NumPy exactly.

* Notably, since JAX arrays are immutable, NumPy APIs that mutate arrays
  in-place cannot be implemented in JAX. However, often JAX is able to provide
  an alternative API that is purely functional. For example, instead of in-place
  array updates (:code:`x[i] = y`), JAX provides an alternative pure indexed
  update function :code:`x.at[i].set(y)` (see :attr:`ndarray.at`).

* Relatedly, some NumPy functions often return views of arrays when possible
  (examples are :func:`transpose` and :func:`reshape`). JAX versions of such
  functions will return copies instead, although such are often optimized
  away by XLA when sequences of operations are compiled using :func:`jax.jit`.

* NumPy is very aggressive at promoting values to :code:`float64` type. JAX
  sometimes is less aggressive about type promotion (See :ref:`type-promotion`).

* Some NumPy routines have data-dependent output shapes (examples include
  :func:`unique` and :func:`nonzero`). Because the XLA compiler requires array
  shapes to be known at compile time, such operations are not compatible with
  JIT. For this reason, JAX adds an optional ``size`` argument to such functions
  which may be specified statically in order to use them with JIT.

Nearly all applicable NumPy functions are implemented in the ``jax.numpy``
namespace; they are listed below.

.. Generate the list below as follows:
   >>> import jax.numpy, numpy
   >>> fns = set(dir(numpy)) & set(dir(jax.numpy))
   >>> print('\n'.join('    ' + x for x in fns if callable(getattr(jax.numpy, x))))  # doctest: +SKIP

   # Finally, sort the list using sort(1), which is different than Python's
   # sorted() function.

.. autosummary::
  :toctree: _autosummary

    ndarray.at
    abs
    absolute
    acos
    acosh
    add
    all
    allclose
    amax
    amin
    angle
    any
    append
    apply_along_axis
    apply_over_axes
    arange
    arccos
    arccosh
    arcsin
    arcsinh
    arctan
    arctan2
    arctanh
    argmax
    argmin
    argpartition
    argsort
    argwhere
    around
    array
    array_equal
    array_equiv
    array_repr
    array_split
    array_str
    asarray
    asin
    asinh
    astype
    atan
    atanh
    atan2
    atleast_1d
    atleast_2d
    atleast_3d
    average
    bartlett
    bincount
    bitwise_and
    bitwise_count
    bitwise_invert
    bitwise_left_shift
    bitwise_not
    bitwise_or
    bitwise_right_shift
    bitwise_xor
    blackman
    block
    bool_
    broadcast_arrays
    broadcast_shapes
    broadcast_to
    c_
    can_cast
    cbrt
    cdouble
    ceil
    character
    choose
    clip
    column_stack
    complex_
    complex128
    complex64
    complexfloating
    ComplexWarning
    compress
    concat
    concatenate
    conj
    conjugate
    convolve
    copy
    copysign
    corrcoef
    correlate
    cos
    cosh
    count_nonzero
    cov
    cross
    csingle
    cumprod
    cumsum
    cumulative_sum
    deg2rad
    degrees
    delete
    diag
    diag_indices
    diag_indices_from
    diagflat
    diagonal
    diff
    digitize
    divide
    divmod
    dot
    double
    dsplit
    dstack
    dtype
    ediff1d
    einsum
    einsum_path
    empty
    empty_like
    equal
    exp
    exp2
    expand_dims
    expm1
    extract
    eye
    fabs
    fill_diagonal
    finfo
    fix
    flatnonzero
    flexible
    flip
    fliplr
    flipud
    float_
    float_power
    float16
    float32
    float64
    floating
    floor
    floor_divide
    fmax
    fmin
    fmod
    frexp
    frombuffer
    fromfile
    fromfunction
    fromiter
    frompyfunc
    fromstring
    from_dlpack
    full
    full_like
    gcd
    generic
    geomspace
    get_printoptions
    gradient
    greater
    greater_equal
    hamming
    hanning
    heaviside
    histogram
    histogram_bin_edges
    histogram2d
    histogramdd
    hsplit
    hstack
    hypot
    i0
    identity
    iinfo
    imag
    index_exp
    indices
    inexact
    inner
    insert
    int_
    int16
    int32
    int64
    int8
    integer
    interp
    intersect1d
    invert
    isclose
    iscomplex
    iscomplexobj
    isdtype
    isfinite
    isin
    isinf
    isnan
    isneginf
    isposinf
    isreal
    isrealobj
    isscalar
    issubdtype
    iterable
    ix_
    kaiser
    kron
    lcm
    ldexp
    left_shift
    less
    less_equal
    lexsort
    linspace
    load
    log
    log10
    log1p
    log2
    logaddexp
    logaddexp2
    logical_and
    logical_not
    logical_or
    logical_xor
    logspace
    mask_indices
    matmul
    matrix_transpose
    max
    maximum
    mean
    median
    meshgrid
    mgrid
    min
    minimum
    mod
    modf
    moveaxis
    multiply
    nan_to_num
    nanargmax
    nanargmin
    nancumprod
    nancumsum
    nanmax
    nanmean
    nanmedian
    nanmin
    nanpercentile
    nanprod
    nanquantile
    nanstd
    nansum
    nanvar
    ndarray
    ndim
    negative
    nextafter
    nonzero
    not_equal
    number
    object_
    ogrid
    ones
    ones_like
    outer
    packbits
    pad
    partition
    percentile
    permute_dims
    piecewise
    place
    poly
    polyadd
    polyder
    polydiv
    polyfit
    polyint
    polymul
    polysub
    polyval
    positive
    pow
    power
    printoptions
    prod
    promote_types
    ptp
    put
    quantile
    r_
    rad2deg
    radians
    ravel
    ravel_multi_index
    real
    reciprocal
    remainder
    repeat
    reshape
    resize
    result_type
    right_shift
    rint
    roll
    rollaxis
    roots
    rot90
    round
    round_
    s_
    save
    savez
    searchsorted
    select
    set_printoptions
    setdiff1d
    setxor1d
    shape
    sign
    signbit
    signedinteger
    sin
    sinc
    single
    sinh
    size
    sort
    sort_complex
    split
    sqrt
    square
    squeeze
    stack
    std
    subtract
    sum
    swapaxes
    take
    take_along_axis
    tan
    tanh
    tensordot
    tile
    trace
    trapezoid
    transpose
    tri
    tril
    tril_indices
    tril_indices_from
    trim_zeros
    triu
    triu_indices
    triu_indices_from
    true_divide
    trunc
    ufunc
    uint
    uint16
    uint32
    uint64
    uint8
    union1d
    unique
    unique_all
    unique_counts
    unique_inverse
    unique_values
    unpackbits
    unravel_index
    unstack
    unsignedinteger
    unwrap
    vander
    var
    vdot
    vecdot
    vectorize
    vsplit
    vstack
    where
    zeros
    zeros_like

jax.numpy.fft
-------------

.. automodule:: jax.numpy.fft

.. autosummary::
  :toctree: _autosummary

  fft
  fft2
  fftfreq
  fftn
  fftshift
  hfft
  ifft
  ifft2
  ifftn
  ifftshift
  ihfft
  irfft
  irfft2
  irfftn
  rfft
  rfft2
  rfftfreq
  rfftn

jax.numpy.linalg
----------------

.. automodule:: jax.numpy.linalg

.. autosummary::
  :toctree: _autosummary

  cholesky
  cond
  cross
  det
  diagonal
  eig
  eigh
  eigvals
  eigvalsh
  inv
  lstsq
  matmul
  matrix_norm
  matrix_power
  matrix_rank
  matrix_transpose
  multi_dot
  norm
  outer
  pinv
  qr
  slogdet
  solve
  svd
  svdvals
  tensordot
  tensorinv
  tensorsolve
  trace
  vector_norm
  vecdot

JAX Array
---------
The JAX :class:`~jax.Array` (along with its alias, :class:`jax.numpy.ndarray`) is
the core array object in JAX: you can think of it as JAX's equivalent of a
:class:`numpy.ndarray`. Like :class:`numpy.ndarray`, most users will not need to
instantiate :class:`~jax.Array` objects manually, but rather will create them via
:mod:`jax.numpy` functions like :func:`~jax.numpy.array`, :func:`~jax.numpy.arange`,
:func:`~jax.numpy.linspace`, and others listed above.

Copying and Serialization
~~~~~~~~~~~~~~~~~~~~~~~~~
JAX :class:`~jax.Array` objects are designed to work seamlessly with Python
standard library tools where appropriate.

With the built-in :mod:`copy` module, when :func:`copy.copy` or :func:`copy.deepcopy`
encounder an :class:`~jax.Array`, it is equivalent to calling the
:meth:`~jax.Array.copy` method, which will create a copy of
the buffer on the same device as the original array. This will work correctly within
traced/JIT-compiled code, though copy operations may be elided by the compiler
in this context.

When the built-in :mod:`pickle` module encounters an :class:`~jax.Array`,
it will be serialized via a compact bit representation in a similar manner to pickled
:class:`numpy.ndarray` objects. When unpickled, the result will be a new
:class:`~jax.Array` object *on the default device.*
This is because in general, pickling and unpickling may take place in different runtime
environments, and there is no general way to map the device IDs of one runtime
to the device IDs of another. If :mod:`pickle` is used in traced/JIT-compiled code,
it will result in a :class:`~jax.errors.ConcretizationTypeError`.

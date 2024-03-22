# Change log

Best viewed [here](https://jax.readthedocs.io/en/latest/changelog.html).

<!--
Remember to align the itemized text with the first line of an item within a list.
-->

## jax 0.4.26

* Deprecations & Removals
  * {func}`jax.tree_map` is deprecated; use `jax.tree.map` instead, or for backward
    compatibility with older JAX versions, use {func}`jax.tree_util.tree_map`.
  * {func}`jax.clear_backends` is deprecated as it does not necessarily do what
    its name suggests and can lead to unexpected consequences, e.g., it will not
    destroy existing backends and release corresponding owned resources. Use
    {func}`jax.clear_caches` if you only want to clean up compilation caches.
    For backward compatibility or you really need to switch/reinitialize the
    default backend, use {func}`jax.extend.backend.clear_backends`.
  * The `jax.experimental.maps` module and `jax.experimental.maps.xmap` are
    deprecated. Use `jax.experimental.shard_map` or `jax.vmap` with the
    `spmd_axis_name` argument for expressing SPMD device-parallel computations.
  * The `jax.experimental.host_callback` module is deprecated.
    Use instead the [new JAX external callbacks](https://jax.readthedocs.io/en/latest/notebooks/external_callbacks.html).
  * Passing arguments to {func}`jax.numpy.array_equal` and {func}`jax.numpy.array_equiv`
    that cannot be converted to a JAX array now results in an exception.
  * The deprecated flag `jax_parallel_functions_output_gda` has been removed.
    This flag was long deprecated and did nothing; its use was a no-op.
  * The previously-deprecated imports `jax.interpreters.ad.config` and
    `jax.interpreters.ad.source_info_util` have now been removed. Use `jax.config`
    and `jax.extend.source_info_util` instead.

## jaxlib 0.4.26

## jax 0.4.25 (Feb 26, 2024)

* New Features
  * Added [CUDA Array
    Interface](https://numba.readthedocs.io/en/stable/cuda/cuda_array_interface.html)
    import support (requires jaxlib 0.4.24).
  * JAX arrays now support NumPy-style scalar boolean indexing, e.g. `x[True]` or `x[False]`.
  * Added {mod}`jax.tree` module, with a more convenient interface for referencing functions
    in {mod}`jax.tree_util`.
  * {func}`jax.tree.transpose` (i.e. {func}`jax.tree_util.tree_transpose`) now accepts
    `inner_treedef=None`, in which case the inner treedef will be automatically inferred.

* Changes
  * Pallas now uses XLA instead of the Triton Python APIs to compile Triton
    kernels. You can revert to the old behavior by setting the
    `JAX_TRITON_COMPILE_VIA_XLA` environment variable to `"0"`.
  * Several deprecated APIs in {mod}`jax.interpreters.xla` that were removed in v0.4.24
    have been re-added in v0.4.25, including `backend_specific_translations`,
    `translations`, `register_translation`, `xla_destructure`, `TranslationRule`,
    `TranslationContext`, and `XLAOp`. These are still considered deprecated, and
    will be removed again in the future when better replacements are available.
    Refer to {jax-issue}`#19816` for discussion.

* Deprecations & Removals
  * {func}`jax.numpy.linalg.solve` now shows a deprecation warning for batched 1D
    solves with `b.ndim > 1`. In the future these will be treated as batched 2D
    solves.
  * Conversion of a non-scalar array to a Python scalar now raises an error, regardless
    of the size of the array. Previously a deprecation warning was raised in the case of
    non-scalar arrays of size 1. This follows a similar deprecation in NumPy.
  * The previously deprecated configuration APIs have been removed
    following a standard 3 months deprecation cycle (see {ref}`api-compatibility`).
    These include
    * the `jax.config.config` object and
    * the `define_*_state` and `DEFINE_*` methods of {data}`jax.config`.
  * Importing the `jax.config` submodule via `import jax.config` is deprecated.
    To configure JAX use `import jax` and then reference the config object
    via `jax.config`.
  * The minimum jaxlib version is now 0.4.20.

## jaxlib 0.4.25 (Feb 26, 2024)

## jax 0.4.24 (Feb 6, 2024)

* Changes

  * JAX lowering to StableHLO does not depend on physical devices anymore.
    If your primitive wraps custom_paritioning or JAX callbacks in the lowering
    rule i.e. function passed to `rule` parameter of `mlir.register_lowering` then add your
    primitive to `jax._src.dispatch.prim_requires_devices_during_lowering` set.
    This is needed because custom_partitioning and JAX callbacks need physical
    devices to create `Sharding`s during lowering.
    This is a temporary state until we can create `Sharding`s without physical
    devices.
  * {func}`jax.numpy.argsort` and {func}`jax.numpy.sort` now support the `stable`
    and `descending` arguments.
  * Several changes to the handling of shape polymorphism (used in
    {mod}`jax.experimental.jax2tf` and {mod}`jax.experimental.export`):
    * cleaner pretty-printing of symbolic expressions ({jax-issue}`#19227`)
    * added the ability to specify symbolic constraints on the dimension variables.
      This makes shape polymorphism more expressive, and gives a way to workaround
      limitations in the reasoning about inequalities.
      See https://github.com/google/jax/blob/main/jax/experimental/jax2tf/README.md#user-specified-symbolic-constraints.
    * with the addition of symbolic constraints ({jax-issue}`#19235`) we now
      consider dimension variables from different scopes to be different, even
      if they have the same name. Symbolic expressions from different scopes
      cannot interact, e.g., in arithmetic operations.
      Scopes are introduced by {func}`jax.experimental.jax2tf.convert`,
      {func}`jax.experimental.export.symbolic_shape`, {func}`jax.experimental.export.symbolic_args_specs`.
      The scope of a symbolic expression `e` can be read with `e.scope` and passed in
      to the above functions to direct them to construct symbolic expressions in
      a given scope.
      See https://github.com/google/jax/blob/main/jax/experimental/jax2tf/README.md#user-specified-symbolic-constraints.
    * simplified and faster equality comparisons, where we consider two symbolic dimensions
      to be equal if the normalized form of their difference reduces to 0
      ({jax-issue}`#19231`; note that this may result in user-visible behavior
        changes)
    * improved the error messages for inconclusive inequality comparisons
      ({jax-issue}`#19235`).
    * the `core.non_negative_dim` API (introduced recently)
      was deprecated and `core.max_dim` and `core.min_dim` were introduced
      ({jax-issue}`#18953`) to express `max` and `min` for symbolic dimensions.
      You can use `core.max_dim(d, 0)` instead of `core.non_negative_dim(d)`.
    * the `shape_poly.is_poly_dim` is deprecated in favor of `export.is_symbolic_dim`
      ({jax-issue}`#19282`).
    * the `export.args_specs` is deprecated in favor of `export.symbolic_args_specs
      ({jax-issue}`#19283`).
    * the `shape_poly.PolyShape` and `jax2tf.PolyShape` are deprecated, use
      strings for polymorphic shapes specifications ({jax-issue}`#19284`).
    * JAX default native serialization version is now 9. This is relevant
      for {mod}`jax.experimental.jax2tf` and {mod}`jax.experimental.export`.
      See [description of version numbers](https://github.com/google/jax/blob/main/jax/experimental/jax2tf/README.md#native-serialization-versions).
  * Refactored the API for `jax.experimental.export`. Instead of
    `from jax.experimental.export import export` you should use now
    `from jax.experimental import export`. The old way of importing will
    continue to work for a deprecation period of 3 months.
  * Added {func}`jax.scipy.stats.sem`.
  * {func}`jax.numpy.unique` with `return_inverse = True` returns inverse indices
    reshaped to the dimension of the input, following a similar change to
    {func}`numpy.unique` in NumPy 2.0.
  * {func}`jax.numpy.sign` now returns `x / abs(x)` for nonzero complex inputs. This is
    consistent with the behavior of {func}`numpy.sign` in NumPy version 2.0.
  * {func}`jax.scipy.special.logsumexp` with `return_sign=True` now uses the NumPy 2.0
    convention for the complex sign, `x / abs(x)`. This is consistent with the behavior
    of {func}`scipy.special.logsumexp` in SciPy v1.13.
  * JAX now supports the bool DLPack type for both import and export.
    Previously bool values could not be imported and were exported as integers.

* Deprecations & Removals
  * A number of previously deprecated functions have been removed, following a
    standard 3+ month deprecation cycle (see {ref}`api-compatibility`).
    This includes:
    * From {mod}`jax.core`: `TracerArrayConversionError`,
      `TracerIntegerConversionError`, `UnexpectedTracerError`,
      `as_hashable_function`, `collections`, `dtypes`, `lu`, `map`,
      `namedtuple`, `partial`, `pp`, `ref`, `safe_zip`, `safe_map`,
      `source_info_util`, `total_ordering`, `traceback_util`, `tuple_delete`,
      `tuple_insert`, and `zip`.
    * From {mod}`jax.lax`: `dtypes`, `itertools`, `naryop`, `naryop_dtype_rule`,
      `standard_abstract_eval`, `standard_naryop`, `standard_primitive`,
      `standard_unop`, `unop`, and `unop_dtype_rule`.
    * The `jax.linear_util` submodule and all its contents.
    * The `jax.prng` submodule and all its contents.
    * From {mod}`jax.random`: `PRNGKeyArray`, `KeyArray`, `default_prng_impl`,
      `threefry_2x32`, `threefry2x32_key`, `threefry2x32_p`, `rbg_key`, and
      `unsafe_rbg_key`.
    * From {mod}`jax.tree_util`: `register_keypaths`, `AttributeKeyPathEntry`, and
      `GetItemKeyPathEntry`.
    * from {mod}`jax.interpreters.xla`: `backend_specific_translations`, `translations`,
      `register_translation`, `xla_destructure`, `TranslationRule`, `TranslationContext`,
      `axis_groups`, `ShapedArray`, `ConcreteArray`, `AxisEnv`, `backend_compile`,
      and `XLAOp`.
    * from {mod}`jax.numpy`: `NINF`, `NZERO`, `PZERO`, `row_stack`, `issubsctype`,
      `trapz`, and `in1d`.
    * from {mod}`jax.scipy.linalg`: `tril` and `triu`.
  * The previously-deprecated method `PRNGKeyArray.unsafe_raw_array` has been
    removed. Use {func}`jax.random.key_data` instead.
  * `bool(empty_array)` now raises an error rather than returning `False`. This
    previously raised a deprecation warning, and follows a similar change in NumPy.
  * Support for the mhlo MLIR dialect has been deprecated. JAX no longer uses
    the mhlo dialect, in favor of stablehlo. APIs that refer to "mhlo" will be
    removed in the future. Use the "stablehlo" dialect instead.
  * {mod}`jax.random`: passing batched keys directly to random number generation functions,
    such as {func}`~jax.random.bits`, {func}`~jax.random.gamma`, and others, is deprecated
    and will emit a `FutureWarning`.  Use `jax.vmap` for explicit batching.
  * {func}`jax.lax.tie_in` is deprecated: it has been a no-op since JAX v0.2.0.

## jaxlib 0.4.24 (Feb 6, 2024)

* Changes

  * JAX now supports CUDA 12.3 and CUDA 11.8. Support for CUDA 12.2 has been
    dropped.
  * `cost_analysis` now works with cross-compiled `Compiled` objects (i.e. when
    using `.lower().compile()` with a topology object, e.g., to compile for
    Cloud TPU from a non-TPU computer).
  * Added [CUDA Array
    Interface](https://numba.readthedocs.io/en/stable/cuda/cuda_array_interface.html)
    import support (requires jax 0.4.25).

## jax 0.4.23 (Dec 13, 2023)

## jaxlib 0.4.23 (Dec 13, 2023)

* Fixed a bug that caused verbose logging from the GPU compiler during
  compilation.

## jax 0.4.22 (Dec 13, 2023)

* Deprecations
  * The `device_buffer` and `device_buffers` properties of JAX arrays are deprecated.
    Explicit buffers have been replaced by the more flexible array sharding interface,
    but the previous outputs can be recovered this way:
    * `arr.device_buffer` becomes `arr.addressable_data(0)`
    * `arr.device_buffers` becomes `[x.data for x in arr.addressable_shards]`

## jaxlib 0.4.22 (Dec 13, 2023)

## jax 0.4.21 (Dec 4 2023)

* New Features
  * Added {obj}`jax.nn.squareplus`.

* Changes
  * The minimum jaxlib version is now 0.4.19.
  * Released wheels are built now with clang instead of gcc.
  * Enforce that the device backend has not been initialized prior to calling `jax.distributed.initialize()`.
  * Automate arguments to `jax.distributed.initialize()` in cloud TPU environments.

* Deprecations
  * The previously-deprecated `sym_pos` argument has been removed from
    {func}`jax.scipy.linalg.solve`. Use `assume_a='pos'` instead.
  * Passing `None` to {func}`jax.array` or {func}`jax.asarray`, either directly or
    within a list or tuple, is deprecated and now raises a {obj}`FutureWarning`.
    It currently is converted to NaN, and in the future will raise a {obj}`TypeError`.
  * Passing the `condition`, `x`, and `y` parameters to `jax.numpy.where` by
    keyword arguments has been deprecated, to match `numpy.where`.
  * Passing arguments to {func}`jax.numpy.array_equal` and {func}`jax.numpy.array_equiv`
    that cannot be converted to a JAX array is deprecated and now raises a
    {obj}`DeprecationWaning`. Currently the functions return False, in the future this
    will raise an exception.
  * The `device()` method of JAX arrays is deprecated. Depending on the context, it may
    be replaced with one of the following:
    - {meth}`jax.Array.devices` returns the set of all devices used by the array.
    - {attr}`jax.Array.sharding` gives the sharding configuration used by the array.

## jaxlib 0.4.21 (Dec 4 2023)

* Changes
  * In preparation for adding distributed CPU support, JAX now treats CPU
    devices identically to GPU and TPU devices, that is:

    * `jax.devices()` includes all devices present in a distributed job, even
      those not local to the current process. `jax.local_devices()` still only
      includes devices local to the current process, so if the change to
      `jax.devices()` breaks you, you most likely want to use
      `jax.local_devices()` instead.
    * CPU devices now receive a globally unique ID number within a distributed
      job; previously CPU devices would receive a process-local ID number.
    * The `process_index` of each CPU device will now match any GPU or TPU
      devices within the same process; previously the `process_index` of a CPU
      device was always 0.

  * On NVIDIA GPU, JAX now prefers a Jacobi SVD solver for matrices up to
    1024x1024. The Jacobi solver appears faster than the non-Jacobi version.

* Bug fixes
  * Fixed error/hang when an array with non-finite values is passed to a
    non-symmetric eigendecomposition (#18226). Arrays with non-finite values now
    produce arrays full of NaNs as outputs.

## jax 0.4.20 (Nov 2, 2023)

## jaxlib 0.4.20 (Nov 2, 2023)

* Bug fixes
  * Fixed some type confusion between E4M3 and E5M2 float8 types.

## jax 0.4.19 (Oct 19, 2023)

* New Features
  * Added {obj}`jax.typing.DTypeLike`, which can be used to annotate objects that
    are convertible to JAX dtypes.
  * Added `jax.numpy.fill_diagonal`.

* Changes
  * JAX now requires SciPy 1.9 or newer.

* Bug fixes
  * Only process 0 in a multicontroller distributed JAX program will write
    persistent compilation cache entries. This fixes write contention if the
    cache is placed on a network filesystem such as GCS.
  * The version check for cusolver and cufft no longer considers the patch
    versions when determining if the installed version of these libraries is at
    least as new as the versions against which JAX was built.

## jaxlib 0.4.19 (Oct 19, 2023)

* Changes
  * jaxlib will now always prefer pip-installed NVIDIA CUDA libraries
    (nvidia-... packages) over any other CUDA installation if they are
    installed, including installations named in `LD_LIBRARY_PATH`. If this
    causes problems and the intent is to use a system-installed CUDA, the fix is
    to remove the pip installed CUDA library packages.

## jax 0.4.18 (Oct 6, 2023)

## jaxlib 0.4.18 (Oct 6, 2023)

* Changes
  * CUDA jaxlibs now depend on the user to install a compatible NCCL version.
    If using the recommended `cuda12_pip` installation, NCCL should be installed
    automatically. Currently, NCCL 2.16 or newer is required.
  * We now provide Linux aarch64 wheels, both with and without NVIDIA GPU
    support.
  * {meth}`jax.Array.item` now supports optional index arguments.

* Deprecations
  * A number of internal utilities and inadvertent exports in {mod}`jax.lax` have
    been deprecated, and will be removed in a future release.
    * `jax.lax.dtypes`: use `jax.dtypes` instead.
    * `jax.lax.itertools`: use `itertools` instead.
    * `naryop`, `naryop_dtype_rule`, `standard_abstract_eval`, `standard_naryop`,
      `standard_primitive`, `standard_unop`, `unop`, and `unop_dtype_rule` are
      internal utilities, now deprecated without replacement.

* Bug fixes
  * Fixed Cloud TPU regression where compilation would OOM due to smem.

## jax 0.4.17 (Oct 3, 2023)

* New features
  * Added new {func}`jax.numpy.bitwise_count` function, matching the API of the similar
    function recently added to NumPy.
* Deprecations
  * Removed the deprecated module `jax.abstract_arrays` and all its contents.
  * Named key constructors in {mod}`jax.random` are deprecated. Pass the `impl` argument
    to {func}`jax.random.PRNGKey` or {func}`jax.random.key` instead:
    * `random.threefry2x32_key(seed)` becomes `random.PRNGKey(seed, impl='threefry2x32')`
    * `random.rbg_key(seed)` becomes `random.PRNGKey(seed, impl='rbg')`
    * `random.unsafe_rbg_key(seed)` becomes `random.PRNGKey(seed, impl='unsafe_rbg')`
* Changes:
  * CUDA: JAX now verifies that the CUDA libraries it finds are at least as new
    as the CUDA libraries that JAX was built against. If older libraries are
    found, JAX raises an exception since that is preferable to mysterious
    failures and crashes.
  * Removed the "No GPU/TPU" found warning. Instead warn if, on Linux, an
    NVIDIA GPU or a Google TPU are found but not used and `--jax_platforms` was
    not specified.
  * {func}`jax.scipy.stats.mode` now returns a 0 count if the mode is taken
    across a size-0 axis, matching the behavior of `scipy.stats.mode` in SciPy
    1.11.
  * Most `jax.numpy` functions and attributes now have fully-defined type stubs.
    Previously many of these were treated as `Any` by static type checkers like
    `mypy` and `pytype`.

## jaxlib 0.4.17 (Oct 3, 2023)

* Changes:
  * Python 3.12 wheels were added in this release.
  * The CUDA 12 wheels now require CUDA 12.2 or newer and cuDNN 8.9.4 or newer.

* Bug fixes:
  * Fixed log spam from ABSL when the JAX CPU backend was initialized.

## jax 0.4.16 (Sept 18, 2023)

* Changes
  * Added {class}`jax.numpy.ufunc`, as well as {func}`jax.numpy.frompyfunc`, which can convert
    any scalar-valued function into a {func}`numpy.ufunc`-like object, with methods such as
    {meth}`~jax.numpy.ufunc.outer`, {meth}`~jax.numpy.ufunc.reduce`,
    {meth}`~jax.numpy.ufunc.accumulate`, {meth}`~jax.numpy.ufunc.at`, and
    {meth}`~jax.numpy.ufunc.reduceat` ({jax-issue}`#17054`).
  * Added {func}`jax.scipy.integrate.trapezoid`.
  * When not running under IPython: when an exception is raised, JAX now filters out the
    entirety of its internal frames from tracebacks. (Without the "unfiltered stack trace"
    that previously appeared.) This should produce much friendlier-looking tracebacks. See
    [here](https://github.com/google/jax/pull/16949) for an example.
    This behavior can be changed by setting `JAX_TRACEBACK_FILTERING=remove_frames` (for two
    separate unfiltered/filtered tracebacks, which was the old behavior) or
    `JAX_TRACEBACK_FILTERING=off` (for one unfiltered traceback).
  * jax2tf default serialization version is now 7, which introduces new shape
    [safety assertions](https://github.com/google/jax/blob/main/jax/experimental/jax2tf/README.md#errors-in-presence-of-shape-polymorphism).
  * Devices passed to `jax.sharding.Mesh` should be hashable. This specifically
    applies to mock devices or user created devices. `jax.devices()` are
    already hashable.
  * The host_callback mechanism now uses custom calls on GPUs by default.
    You can still use `--jax_host_callback_use_outfeed` to revert to the
    previous outfeed implementation mechanism.

* Breaking changes:
  * jax2tf now uses native serialization by default. See
    the [jax2tf documentation](https://github.com/google/jax/blob/main/jax/experimental/jax2tf/README.md)
    for details and for mechanisms to override the default.
  * The option `--jax_coordination_service` has been removed. It is now always
    `True`.
  * `jax.jaxpr_util` has been removed from the public JAX namespace.
  * `JAX_USE_PJRT_C_API_ON_TPU` no longer has an effect (i.e. it always defaults to true).
  * The backwards compatibility flag `--jax_host_callback_ad_transforms`
    introduced in December 2021, has been removed.

* Deprecations:
  * Several `jax.numpy` APIs have been deprecated following
    [NumPy NEP-52](https://numpy.org/neps/nep-0052-python-api-cleanup.html):
    * `jax.numpy.NINF` has been deprecated. Use `-jax.numpy.inf` instead.
    * `jax.numpy.PZERO` has been deprecated. Use `0.0` instead.
    * `jax.numpy.NZERO` has been deprecated. Use `-0.0` instead.
    * `jax.numpy.issubsctype(x, t)` has been deprecated. Use `jax.numpy.issubdtype(x.dtype, t)`.
    * `jax.numpy.row_stack` has been deprecated. Use `jax.numpy.vstack` instead.
    * `jax.numpy.in1d` has been deprecated. Use `jax.numpy.isin` instead.
    * `jax.numpy.trapz` has been deprecated. Use `jax.scipy.integrate.trapezoid` instead.
  * `jax.scipy.linalg.tril` and `jax.scipy.linalg.triu` have been deprecated,
    following SciPy. Use `jax.numpy.tril` and `jax.numpy.triu` instead.
  * `jax.lax.prod` has been removed after being deprecated in JAX v0.4.11.
    Use the built-in `math.prod` instead.
  * A number of exports from `jax.interpreters.xla` related to defining
    HLO lowering rules for custom JAX primitives have been deprecated. Custom
    primitives should be defined using the StableHLO lowering utilities in
    `jax.interpreters.mlir` instead.
  * The following previously-deprecated functions have been removed after a
    three-month deprecation period:
    * `jax.abstract_arrays.ShapedArray`: use `jax.core.ShapedArray`.
    * `jax.abstract_arrays.raise_to_shaped`: use `jax.core.raise_to_shaped`.
    * `jax.numpy.alltrue`: use `jax.numpy.all`.
    * `jax.numpy.sometrue`: use `jax.numpy.any`.
    * `jax.numpy.product`: use `jax.numpy.prod`.
    * `jax.numpy.cumproduct`: use `jax.numpy.cumprod`.

* Deprecations/removals:
  * The internal submodule `jax.prng` is now deprecated. Its contents are available at
    {mod}`jax.extend.random`.
  * The internal submodule path `jax.linear_util` has been deprecated. Use
    {mod}`jax.extend.linear_util` instead (Part of {ref}`jax-extend-jep`)
  * `jax.random.PRNGKeyArray` and `jax.random.KeyArray` are deprecated.  Use {class}`jax.Array`
    for type annotations, and `jax.dtypes.issubdtype(arr.dtype, jax.dtypes.prng_key)` for
    runtime detection of typed prng keys.
  * The method `PRNGKeyArray.unsafe_raw_array` is deprecated. Use
    {func}`jax.random.key_data` instead.
  * `jax.experimental.pjit.with_sharding_constraint` is deprecated. Use
    `jax.lax.with_sharding_constraint` instead.
  * The internal utilities `jax.core.is_opaque_dtype` and `jax.core.has_opaque_dtype`
    have been removed. Opaque dtypes have been renamed to Extended dtypes; use
    `jnp.issubdtype(dtype, jax.dtypes.extended)` instead (available since jax v0.4.14).
  * The utility `jax.interpreters.xla.register_collective_primitive` has been
    removed. This utility did nothing useful in recent JAX releases and calls
    to it can be safely removed.
  * The internal submodule path `jax.linear_util` has been deprecated. Use
    {mod}`jax.extend.linear_util` instead (Part of {ref}`jax-extend-jep`)

## jaxlib 0.4.16 (Sept 18, 2023)

* Changes:
  * Sparse CSR matrix multiplications via the experimental jax sparse APIs
    no longer uses a deterministic algorithm on NVIDIA GPUs. This change was
    made to improve compatibility with CUDA 12.2.1.

* Bug fixes:
  * Fixed a crash on Windows due to a fatal LLVM error related to out-of-order
    sections and IMAGE_REL_AMD64_ADDR32NB relocations
    (https://github.com/openxla/xla/commit/cb732a921f0c4184995cbed82394931011d12bd4).

## jax 0.4.14 (July 27, 2023)

* Changes
  * `jax.jit` takes `donate_argnames` as an argument. It's semantics are similar
    to `static_argnames`.
    If neither donate_argnums nor donate_argnames is provided, no
    arguments are donated. If donate_argnums is not provided but
    donate_argnames is, or vice versa, JAX uses
    `inspect.signature(fun)` to find any positional arguments that
    correspond to donate_argnames (or vice versa). If both donate_argnums and donate_argnames are provided, inspect.signature is not used, and only actual
    parameters listed in either donate_argnums or donate_argnames will
    be donated.
  * {func}`jax.random.gamma` has been re-factored to a more efficient algorithm
    with more robust endpoint behavior ({jax-issue}`#16779`). This means that the
    sequence of values returned for a given `key` will change between JAX v0.4.13
    and v0.4.14 for `gamma` and related samplers (including {func}`jax.random.ball`,
    {func}`jax.random.beta`, {func}`jax.random.chisquare`, {func}`jax.random.dirichlet`,
    {func}`jax.random.generalized_normal`, {func}`jax.random.loggamma`, {func}`jax.random.t`).

* Deletions
  * `in_axis_resources` and `out_axis_resources` have been deleted from pjit since
    it has been more than 3 months since their deprecation. Please use
    `in_shardings` and `out_shardings` as the replacement.
    This is a safe and trivial name replacement. It does not change any of the
    current pjit semantics and doesn't break any code.
    You can still pass in `PartitionSpecs` to in_shardings and out_shardings.


* Deprecations
  * Python 3.8 support has been dropped as per
    https://jax.readthedocs.io/en/latest/deprecation.html
  * JAX now requires NumPy 1.22 or newer as per
    https://jax.readthedocs.io/en/latest/deprecation.html
  * Passing optional arguments to {func}`jax.numpy.ndarray.at` by position is
    no longer supported, after being deprecated in JAX version 0.4.7.
    For example, instead of `x.at[i].get(True)`, use `x.at[i].get(indices_are_sorted=True)`
  * The following `jax.Array` methods have been removed, after being deprecated
    in JAX v0.4.5:
    * `jax.Array.broadcast`: use {func}`jax.lax.broadcast` instead.
    * `jax.Array.broadcast_in_dim`: use {func}`jax.lax.broadcast_in_dim` instead.
    * `jax.Array.split`: use {func}`jax.numpy.split` instead.
  * The following APIs have been removed after previous deprecation:
    * `jax.ad`: use {mod}`jax.interpreters.ad`.
    * `jax.curry`: use ``curry = lambda f: partial(partial, f)``.
    * `jax.partial_eval`: use {mod}`jax.interpreters.partial_eval`.
    * `jax.pxla`: use {mod}`jax.interpreters.pxla`.
    * `jax.xla`: use {mod}`jax.interpreters.xla`.
    * `jax.ShapedArray`: use {class}`jax.core.ShapedArray`.
    * `jax.interpreters.pxla.device_put`: use {func}`jax.device_put`.
    * `jax.interpreters.pxla.make_sharded_device_array`: use {func}`jax.make_array_from_single_device_arrays`.
    * `jax.interpreters.pxla.ShardedDeviceArray`: use {class}`jax.Array`.
    * `jax.numpy.DeviceArray`: use {class}`jax.Array`.
    * `jax.stages.Compiled.compiler_ir`: use {func}`jax.stages.Compiled.as_text`.

* Breaking changes
  * JAX now requires ml_dtypes version 0.2.0 or newer.
  * To fix a corner case, calls to {func}`jax.lax.cond` with five
    arguments will always resolve to the "common operands" `cond`
    behavior (as documented) if the second and third arguments are
    callable, even if other operands are callable as well. See
    [#16413](https://github.com/google/jax/issues/16413).
  * The deprecated config options `jax_array` and `jax_jit_pjit_api_merge`,
    which did nothing, have been removed. These options have been true by
    default for many releases.

* New features
  * JAX now supports a configuration flag --jax_serialization_version
    and a JAX_SERIALIZATION_VERSION environment variable to control the
    serialization version ({jax-issue}`#16746`).
  * jax2tf in presence of shape polymorphism now generates code that checks
    certain shape constraints, if the serialization version is at least 7.
    See https://github.com/google/jax/blob/main/jax/experimental/jax2tf/README.md#errors-in-presence-of-shape-polymorphism.

## jaxlib 0.4.14 (July 27, 2023)

* Deprecations
  * Python 3.8 support has been dropped as per
      https://jax.readthedocs.io/en/latest/deprecation.html

## jax 0.4.13 (June 22, 2023)

* Changes
  * `jax.jit` now allows `None` to be passed to `in_shardings` and
    `out_shardings`. The semantics are as follows:
      * For in_shardings, JAX will mark is as replicated but this behavior
        can change in the future.
      * For out_shardings, we will rely on the XLA GSPMD partitioner to
        determine the output shardings.
  * `jax.experimental.pjit.pjit` also allows `None` to be passed to
    `in_shardings` and `out_shardings`. The semantics are as follows:
    * If the mesh context manager is *not* provided, JAX has the freedom to
      choose whatever sharding it wants.
      * For in_shardings, JAX will mark is as replicated but this behavior
        can change in the future.
      * For out_shardings, we will rely on the XLA GSPMD partitioner to
        determine the output shardings.
    * If the mesh context manager is provided, None will imply that the value
      will be replicated on all devices of the mesh.
  * Executable.cost_analysis() works on Cloud TPU
  * Added a warning if a non-allowlisted `jaxlib` plugin is in use.
  * Added `jax.tree_util.tree_leaves_with_path`.
  * `None` is not a valid input to
    `jax.experimental.multihost_utils.host_local_array_to_global_array` or
    `jax.experimental.multihost_utils.global_array_to_host_local_array`.
    Please use `jax.sharding.PartitionSpec()` if you wanted to replicate your
    input.

* Bug fixes
  * Fixed incorrect wheel name in CUDA 12 releases (#16362); the correct wheel
    is named `cudnn89` instead of `cudnn88`.

* Deprecations
  * The `native_serialization_strict_checks` parameter to
    {func}`jax.experimental.jax2tf.convert` is deprecated in favor of the
    new `native_serializaation_disabled_checks` ({jax-issue}`#16347`).

## jaxlib 0.4.13 (June 22, 2023)

* Changes
  * Added Windows CPU-only wheels to the `jaxlib` Pypi release.

* Bug fixes
  * `__cuda_array_interface__` was broken in previous jaxlib versions and is now
    fixed ({jax-issue}`16440`).
  * Concurrent CUDA kernel tracing is now enabled by default on NVIDIA GPUs.

## jax 0.4.12 (June 8, 2023)

* Changes
  * Added {class}`scipy.spatial.transform.Rotation` and {class}`scipy.spatial.transform.Slerp`

* Deprecations
  * `jax.abstract_arrays` and its contents are now deprecated. See related
    functionality in :mod:`jax.core`.
  * `jax.numpy.alltrue`: use `jax.numpy.all`. This follows the deprecation
    of `numpy.alltrue` in NumPy version 1.25.0.
  * `jax.numpy.sometrue`: use `jax.numpy.any`. This follows the deprecation
    of `numpy.sometrue` in NumPy version 1.25.0.
  * `jax.numpy.product`: use `jax.numpy.prod`. This follows the deprecation
    of `numpy.product` in NumPy version 1.25.0.
  * `jax.numpy.cumproduct`: use `jax.numpy.cumprod`. This follows the deprecation
    of `numpy.cumproduct` in NumPy version 1.25.0.
  * `jax.sharding.OpShardingSharding` has been removed since it has been 3
    months since it was deprecated.

## jaxlib 0.4.12 (June 8, 2023)

* Changes
  * Includes PTX/SASS for Hopper (SM version 9.0+) GPUs. Previous
    versions of jaxlib should work on Hopper but would have a long
    JIT-compilation delay the first time a JAX operation was executed.

* Bug fixes
  * Fixes incorrect source line information in JAX-generated Python tracebacks
    under Python 3.11.
  * Fixes crash when printing local variables of frames in JAX-generated Python
    tracebacks (#16027).

## jax 0.4.11 (May 31, 2023)

* Deprecations
  * The following APIs have been removed after a 3 month deprecation period, in
    accordance with the {ref}`api-compatibility` policy:
    * `jax.experimental.PartitionSpec`: use `jax.sharding.PartitionSpec`.
    * `jax.experimental.maps.Mesh`: use `jax.sharding.Mesh`
    * `jax.experimental.pjit.NamedSharding`: use `jax.sharding.NamedSharding`.
    * `jax.experimental.pjit.PartitionSpec`: use `jax.sharding.PartitionSpec`.
    * `jax.experimental.pjit.FROM_GDA`. Instead pass sharded `jax.Array` objects
      as input and remove the optional `in_shardings` argument to `pjit`.
    * `jax.interpreters.pxla.PartitionSpec`: use `jax.sharding.PartitionSpec`.
    * `jax.interpreters.pxla.Mesh`: use `jax.sharding.Mesh`
    * `jax.interpreters.xla.Buffer`: use `jax.Array`.
    * `jax.interpreters.xla.Device`: use `jax.Device`.
    * `jax.interpreters.xla.DeviceArray`: use `jax.Array`.
    * `jax.interpreters.xla.device_put`: use `jax.device_put`.
    * `jax.interpreters.xla.xla_call_p`: use `jax.experimental.pjit.pjit_p`.
    * `axis_resources` argument of `with_sharding_constraint` is removed. Please
      use `shardings` instead.


## jaxlib 0.4.11 (May 31, 2023)

* Changes
  * Added `memory_stats()` method to `Device`s. If supported, this returns a
    dict of string stat names with int values, e.g. `"bytes_in_use"`, or None if
    the platform doesn't support memory statistics. The exact stats returned may
    vary across platforms. Currently only implemented on Cloud TPU.
  * Readded support for the Python buffer protocol (`memoryview`) on CPU
    devices.

## jax 0.4.10 (May 11, 2023)

## jaxlib 0.4.10 (May 11, 2023)

* Changes
  * Fixed `'apple-m1' is not a recognized processor for this target (ignoring
    processor)` issue that prevented previous release from running on Mac M1.

## jax 0.4.9 (May 9, 2023)

* Changes
  * The flags experimental_cpp_jit, experimental_cpp_pjit and
    experimental_cpp_pmap have been removed.
    They are now always on.
  * Accuracy of singular value decomposition (SVD) on TPU has been improved
    (requires jaxlib 0.4.9).

* Deprecations
  * `jax.experimental.gda_serialization` is deprecated and has been renamed to
    `jax.experimental.array_serialization`.
    Please change your imports to use `jax.experimental.array_serialization`.
  * The `in_axis_resources` and `out_axis_resources` arguments of pjit have been
    deprecated. Please use `in_shardings` and `out_shardings` respectively.
  * The function `jax.numpy.msort` has been removed. It has been deprecated since
    JAX v0.4.1. Use `jnp.sort(a, axis=0)` instead.
  * `in_parts` and `out_parts` arguments have been removed from `jax.xla_computation`
    since they were only used with sharded_jit and sharded_jit is long gone.
  * `instantiate_const_outputs` argument has been removed from `jax.xla_computation`
    since it has been unused for a very long time.

## jaxlib 0.4.9 (May 9, 2023)

## jax 0.4.8 (March 29, 2023)

* Breaking changes
  * A major component of the Cloud TPU runtime has been upgraded. This enables
    the following new features on Cloud TPU:
    * {func}`jax.debug.print`, {func}`jax.debug.callback`, and
      {func}`jax.debug.breakpoint()` now work on Cloud TPU
    * Automatic TPU memory defragmentation

    {func}`jax.experimental.host_callback` is no longer supported on Cloud TPU
    with the new runtime component. Please file an issue on the [JAX issue
    tracker](https://github.com/google/jax/issues) if the new `jax.debug` APIs
    are insufficient for your use case.

    The old runtime component will be available for at least the next three
    months by setting the environment variable
    `JAX_USE_PJRT_C_API_ON_TPU=false`. If you find you need to disable the new
    runtime for any reason, please let us know on the [JAX issue
    tracker](https://github.com/google/jax/issues).

* Changes
  * The minimum jaxlib version has been bumped from 0.4.6 to 0.4.7.

* Deprecations
  * CUDA 11.4 support has been dropped. JAX GPU wheels only support
    CUDA 11.8 and CUDA 12. Older CUDA versions may work if jaxlib is built
    from source.
  * `global_arg_shapes` argument of pmap only worked with sharded_jit and has
    been removed from pmap. Please migrate to pjit and remove global_arg_shapes
    from pmap.

## jax 0.4.7 (March 27, 2023)

* Changes
  * As per https://jax.readthedocs.io/en/latest/jax_array_migration.html#jax-array-migration
    `jax.config.jax_array` cannot be disabled anymore.
  * `jax.config.jax_jit_pjit_api_merge` cannot be disabled anymore.
  * {func}`jax.experimental.jax2tf.convert` now supports the `native_serialization`
    parameter to use JAX's native lowering to StableHLO to obtain a
    StableHLO module for the entire JAX function instead of lowering each JAX
    primitive to a TensorFlow op. This simplifies the internals and increases
    the confidence that what you serialize matches the JAX native semantics.
    See [documentation](https://github.com/google/jax/blob/main/jax/experimental/jax2tf/README.md).
    As part of this change the config flag `--jax2tf_default_experimental_native_lowering`
    has been renamed to `--jax2tf_native_serialization`.
  * JAX now depends on `ml_dtypes`, which contains definitions of NumPy types
    like bfloat16. These definitions were previously internal to JAX, but have
    been split into a separate package to facilitate sharing them with other
    projects.
  * JAX now requires NumPy 1.21 or newer and SciPy 1.7 or newer.

* Deprecations
  * The type `jax.numpy.DeviceArray` is deprecated. Use `jax.Array` instead,
    for which it is an alias.
  * The type `jax.interpreters.pxla.ShardedDeviceArray` is deprecated. Use
    `jax.Array` instead.
  * Passing additional arguments to {func}`jax.numpy.ndarray.at` by position is deprecated.
    For example, instead of `x.at[i].get(True)`, use `x.at[i].get(indices_are_sorted=True)`
  * `jax.interpreters.xla.device_put` is deprecated. Please use `jax.device_put`.
  * `jax.interpreters.pxla.device_put` is deprecated. Please use `jax.device_put`.
  * `jax.experimental.pjit.FROM_GDA` is deprecated. Please pass in sharded
    jax.Arrays as input and remove the `in_shardings` argument to pjit since
    it is optional.

## jaxlib 0.4.7 (March 27, 2023)

Changes:
  * jaxlib now depends on `ml_dtypes`, which contains definitions of NumPy types
    like bfloat16. These definitions were previously internal to JAX, but have
    been split into a separate package to facilitate sharing them with other
    projects.

## jax 0.4.6 (Mar 9, 2023)

* Changes
  * `jax.tree_util` now contain a set of APIs that allow user to define keys for their
    custom pytree node. This includes:
    * `tree_flatten_with_path` that flattens a tree and return not only each leaf but
      also their key paths.
    * `tree_map_with_path` that can map a function that takes the key path as an argument.
    * `register_pytree_with_keys` to register how the key path and leaves should looks
      like in a custom pytree node.
    * `keystr` that pretty-prints a key path.

  * {func}`jax2tf.call_tf` has a new parameter `output_shape_dtype` (default `None`)
    that can be used to declare the output shape and type of the result. This enables
    {func}`jax2tf.call_tf` to work in the presence of shape polymorphism. ({jax-issue}`#14734`).

* Deprecations
  * The old key-path APIs in `jax.tree_util` are deprecated and will be removed 3 months
    from Mar 10 2023:
    * `register_keypaths`: use {func}`jax.tree_util.register_pytree_with_keys` instead.
    * `AttributeKeyPathEntry` : use `GetAttrKey` instead.
    * `GetitemKeyPathEntry` : use `SequenceKey` or `DictKey` instead.

## jaxlib 0.4.6 (Mar 9, 2023)

## jax 0.4.5 (Mar 2, 2023)

* Deprecations
  * `jax.sharding.OpShardingSharding` has been renamed to `jax.sharding.GSPMDSharding`.
    `jax.sharding.OpShardingSharding` will be removed in 3 months from Feb 17, 2023.
  * The following `jax.Array` methods are deprecated and will be removed 3 months from
    Feb 23 2023:
    * `jax.Array.broadcast`: use {func}`jax.lax.broadcast` instead.
    * `jax.Array.broadcast_in_dim`: use {func}`jax.lax.broadcast_in_dim` instead.
    * `jax.Array.split`: use {func}`jax.numpy.split` instead.

## jax 0.4.4 (Feb 16, 2023)

* Changes
  * The implementation of `jit` and `pjit` has been merged. Merging jit and pjit
    changes the internals of JAX without affecting the public API of JAX.
    Before, `jit` was a final style primitive. Final style means that the creation
    of jaxpr was delayed as much as possible and transformations were stacked
    on top of each other. With the `jit`-`pjit` implementation merge, `jit`
    becomes an initial style primitive which means that we trace to jaxpr
    as early as possible. For more information see
    [this section in autodidax](https://jax.readthedocs.io/en/latest/autodidax.html#on-the-fly-final-style-and-staged-initial-style-processing).
    Moving to initial style should simplify JAX's internals and make
    development of features like dynamic shapes, etc easier.
    You can disable it only via the environment variable i.e.
    `os.environ['JAX_JIT_PJIT_API_MERGE'] = '0'`.
    The merge must be disabled via an environment variable since it affects JAX
    at import time so it needs to be disabled before jax is imported.
  * `axis_resources` argument of `with_sharding_constraint` is deprecated.
    Please use `shardings` instead. There is no change needed if you were using
    `axis_resources` as an arg. If you were using it as a kwarg, then please
    use `shardings` instead. `axis_resources` will be removed after 3 months
    from Feb 13, 2023.
  * added the {mod}`jax.typing` module, with tools for type annotations of JAX
    functions.
  * The following names have been deprecated:
    * `jax.xla.Device` and `jax.interpreters.xla.Device`: use `jax.Device`.
    * `jax.experimental.maps.Mesh`. Use `jax.sharding.Mesh`
    instead.
    * `jax.experimental.pjit.NamedSharding`: use `jax.sharding.NamedSharding`.
    * `jax.experimental.pjit.PartitionSpec`: use `jax.sharding.PartitionSpec`.
    * `jax.interpreters.pxla.Mesh`: use `jax.sharding.Mesh`.
    * `jax.interpreters.pxla.PartitionSpec`: use `jax.sharding.PartitionSpec`.
* Breaking Changes
  * the `initial` argument to reduction functions like :func:`jax.numpy.sum`
    is now required to be a scalar, consistent with the corresponding NumPy API.
    The previous behavior of broadcasting the output against non-scalar `initial`
    values was an unintentional implementation detail ({jax-issue}`#14446`).

## jaxlib 0.4.4 (Feb 16, 2023)
  * Breaking changes
    * Support for NVIDIA Kepler series GPUs has been removed from the default
      `jaxlib` builds. If Kepler support is needed, it is still possible to
      build `jaxlib` from source with Kepler support (via the
      `--cuda_compute_capabilities=sm_35` option to `build.py`), however note
      that CUDA 12 has completely dropped support for Kepler GPUs.

## jax 0.4.3 (Feb 8, 2023)
  * Breaking changes
    * Deleted {func}`jax.scipy.linalg.polar_unitary`, which was a deprecated JAX
      extension to the scipy API. Use {func}`jax.scipy.linalg.polar` instead.

  * Changes
    * Added {func}`jax.scipy.stats.rankdata`.

## jaxlib 0.4.3 (Feb 8, 2023)
  * `jax.Array` now has the non-blocking `is_ready()` method, which returns `True`
    if the array is ready (see also {func}`jax.block_until_ready`).

## jax 0.4.2 (Jan 24, 2023)

* Breaking changes
  * Deleted `jax.experimental.callback`
  * Operations with dimensions in presence of jax2tf shape polymorphism have
    been generalized to work in more scenarios, by converting the symbolic
    dimension to JAX arrays. Operations involving symbolic dimensions and
    `np.ndarray` now can raise errors when the result is used as a shape value
    ({jax-issue}`#14106`).
  * jaxpr objects now raise an error on attribute setting in order to avoid
    problematic mutations ({jax-issue}`14102`)

* Changes
  * {func}`jax2tf.call_tf` has a new parameter `has_side_effects` (default `True`)
    that can be used to declare whether an instance can be removed or replicated
    by JAX optimizations such as dead-code elimination ({jax-issue}`#13980`).
  * Added more support for floordiv and mod for jax2tf shape polymorphism. Previously,
    certain division operations resulted in errors in presence of symbolic dimensions
    ({jax-issue}`#14108`).

## jaxlib 0.4.2 (Jan 24, 2023)

* Changes
  * Set JAX_USE_PJRT_C_API_ON_TPU=1 to enable new Cloud TPU runtime, featuring
    automatic device memory defragmentation.

## jax 0.4.1 (Dec 13, 2022)

* Changes
  * Support for Python 3.7 has been dropped, in accordance with JAX's
    {ref}`version-support-policy`.
  * We introduce `jax.Array` which is a unified array type that subsumes
    `DeviceArray`, `ShardedDeviceArray`, and `GlobalDeviceArray` types in JAX.
    The `jax.Array` type helps make parallelism a core feature of JAX,
    simplifies and unifies JAX internals, and allows us to unify `jit` and
    `pjit`.  `jax.Array` has been enabled by default in JAX 0.4 and makes some
    breaking change to the `pjit` API.  The [jax.Array migration
    guide](https://jax.readthedocs.io/en/latest/jax_array_migration.html) can
    help you migrate your codebase to `jax.Array`. You can also look at the
    [Distributed arrays and automatic parallelization](https://jax.readthedocs.io/en/latest/notebooks/Distributed_arrays_and_automatic_parallelization.html)
    tutorial to understand the new concepts.
  * `PartitionSpec` and `Mesh` are now out of experimental. The new API endpoints
    are `jax.sharding.PartitionSpec` and `jax.sharding.Mesh`.
    `jax.experimental.maps.Mesh` and `jax.experimental.PartitionSpec` are
    deprecated and will be removed in 3 months.
  * `with_sharding_constraint`s new public endpoint is
    `jax.lax.with_sharding_constraint`.
  * If using ABSL flags together with `jax.config`, the ABSL flag values are no
    longer read or written after the JAX configuration options are initially
    populated from the ABSL flags. This change improves performance of reading
    `jax.config` options, which are used pervasively in JAX.
  * The jax2tf.call_tf function now uses for TF lowering the first TF
    device of the same platform as used by the embedding JAX computation.
    Before, it was using the 0th device for the JAX-default backend.
  * A number of `jax.numpy` functions now have their arguments marked as
    positional-only, matching NumPy.
  * `jnp.msort` is now deprecated, following the deprecation of `np.msort` in numpy 1.24.
    It will be removed in a future release, in accordance with the {ref}`api-compatibility`
    policy. It can be replaced with `jnp.sort(a, axis=0)`.

## jaxlib 0.4.1 (Dec 13, 2022)

* Changes
  * Support for Python 3.7 has been dropped, in accordance with JAX's
    {ref}`version-support-policy`.
  * The behavior of `XLA_PYTHON_CLIENT_MEM_FRACTION=.XX` has been changed to allocate XX% of
    the total GPU memory instead of the previous behavior of using currently available GPU memory
    to calculate preallocation. Please refer to
    [GPU memory allocation](https://jax.readthedocs.io/en/latest/gpu_memory_allocation.html) for
    more details.
  * The deprecated method `.block_host_until_ready()` has been removed. Use
    `.block_until_ready()` instead.

## jax 0.4.0 (Dec 12, 2022)

* The release was yanked.

## jaxlib 0.4.0 (Dec 12, 2022)

* The release was yanked.

## jax 0.3.25 (Nov 15, 2022)
* Changes
  * {func}`jax.numpy.linalg.pinv` now supports the `hermitian` option.
  * {func}`jax.scipy.linalg.hessenberg` is now supported on CPU only. Requires
    jaxlib > 0.3.24.
  * New functions {func}`jax.lax.linalg.hessenberg`,
    {func}`jax.lax.linalg.tridiagonal`, and
    {func}`jax.lax.linalg.householder_product` were added. Householder reduction
    is currently CPU-only and tridiagonal reductions are supported on CPU and
    GPU only.
  * The gradients of `svd` and `jax.numpy.linalg.pinv` are now computed more
    economically for non-square matrices.
* Breaking Changes
  * Deleted the `jax_experimental_name_stack` config option.
  * Convert a string `axis_names` arguments to the
    {class}`jax.experimental.maps.Mesh` constructor into a singleton tuple
    instead of unpacking the string into a sequence of character axis names.

## jaxlib 0.3.25 (Nov 15, 2022)
* Changes
  * Added support for tridiagonal reductions on CPU and GPU.
  * Added support for upper Hessenberg reductions on CPU.
* Bugs
  * Fixed a bug that meant that frames in tracebacks captured by JAX were
    incorrectly mapped to source lines under Python 3.10+

## jax 0.3.24 (Nov 4, 2022)
* Changes
  * JAX should be faster to import. We now import scipy lazily, which accounted
    for a significant fraction of JAX's import time.
  * Setting the env var `JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS=$N` can be
    used to limit the number of cache entries written to the persistent cache.
    By default, computations that take 1 second or more to compile will be
    cached.
    * Added {func}`jax.scipy.stats.mode`.
  * The default device order used by `pmap` on TPU if no order is specified now
    matches `jax.devices()` for single-process jobs. Previously the
    two orderings differed, which could lead to unnecessary copies or
    out-of-memory errors. Requiring the orderings to agree simplifies matters.
* Breaking Changes
    * {func}`jax.numpy.gradient` now behaves like most other functions in {mod}`jax.numpy`,
      and forbids passing lists or tuples in place of arrays ({jax-issue}`#12958`)
    * Functions in {mod}`jax.numpy.linalg` and {mod}`jax.numpy.fft` now uniformly
      require inputs to be array-like: i.e. lists and tuples cannot be used in place
      of arrays. Part of {jax-issue}`#7737`.
* Deprecations
  * `jax.sharding.MeshPspecSharding` has been renamed to `jax.sharding.NamedSharding`.
    `jax.sharding.MeshPspecSharding` name will be removed in 3 months.

## jaxlib 0.3.24 (Nov 4, 2022)
* Changes
  * Buffer donation now works on CPU. This may break code that marked buffers
    for donation on CPU but relied on donation not being implemented.

## jax 0.3.23 (Oct 12, 2022)
* Changes
  * Update Colab TPU driver version for new jaxlib release.

## jax 0.3.22 (Oct 11, 2022)
* Changes
  * Add `JAX_PLATFORMS=tpu,cpu` as default setting in TPU initialization,
  so JAX will raise an error if TPU cannot be initialized instead of falling
  back to CPU. Set `JAX_PLATFORMS=''` to override this behavior and automatically
  choose an available backend (the original default), or set `JAX_PLATFORMS=cpu`
  to always use CPU regardless of if the TPU is available.
* Deprecations
  * Several test utilities deprecated in JAX v0.3.8 are now removed from
    {mod}`jax.test_util`.

## jaxlib 0.3.22 (Oct 11, 2022)

## jax 0.3.21 (Sep 30, 2022)
* [GitHub commits](https://github.com/google/jax/compare/jax-v0.3.20...jax-v0.3.21).
* Changes
  * The persistent compilation cache will now warn instead of raising an
    exception on error ({jax-issue}`#12582`), so program execution can continue
    if something goes wrong with the cache. Set
    `JAX_RAISE_PERSISTENT_CACHE_ERRORS=true` to revert this behavior.

## jax 0.3.20 (Sep 28, 2022)
* Bug fixes:
  * Adds missing `.pyi` files that were missing from the previous release ({jax-issue}`#12536`).
  * Fixes an incompatibility between `jax` 0.3.19 and the libtpu version it pinned ({jax-issue}`#12550`). Requires jaxlib 0.3.20.
  * Fix incorrect `pip` url in `setup.py` comment ({jax-issue}`#12528`).

## jaxlib 0.3.20 (Sep 28, 2022)
* [GitHub commits](https://github.com/google/jax/compare/jaxlib-v0.3.15...jaxlib-v0.3.20).
* Bug fixes
  * Fixes support for limiting the visible CUDA devices via
   `jax_cuda_visible_devices` in distributed jobs. This functionality is needed for
   the JAX/SLURM integration on GPU ({jax-issue}`#12533`).

## jax 0.3.19 (Sep 27, 2022)
* [GitHub commits](https://github.com/google/jax/compare/jax-v0.3.18...jax-v0.3.19).
* Fixes required jaxlib version.

## jax 0.3.18 (Sep 26, 2022)
* [GitHub commits](https://github.com/google/jax/compare/jax-v0.3.17...jax-v0.3.18).
* Changes
  * Ahead-of-time lowering and compilation functionality (tracked in
    {jax-issue}`#7733`) is stable and public. See [the
    overview](https://jax.readthedocs.io/en/latest/aot.html) and the API docs
    for {mod}`jax.stages`.
  * Introduced {class}`jax.Array`, intended to be used for both `isinstance` checks
    and type annotations for array types in JAX. Notice that this included some subtle
    changes to how `isinstance` works for {class}`jax.numpy.ndarray` for jax-internal
    objects, as {class}`jax.numpy.ndarray` is now a simple alias of {class}`jax.Array`.
* Breaking changes
  * `jax._src` is no longer imported into the public `jax` namespace.
    This may break users that were using JAX internals.
  * `jax.soft_pmap` has been deleted. Please use `pjit` or `xmap` instead.
    `jax.soft_pmap` is undocumented. If it were documented, a deprecation period
    would have been provided.

## jax 0.3.17 (Aug 31, 2022)
* [GitHub commits](https://github.com/google/jax/compare/jax-v0.3.16...jax-v0.3.17).
* Bugs
  * Fix corner case issue in gradient of `lax.pow` with an exponent of zero
    ({jax-issue}`12041`)
* Breaking changes
  * {func}`jax.checkpoint`, also known as {func}`jax.remat`, no longer supports
    the `concrete` option, following the previous version's deprecation; see
    [JEP 11830](https://jax.readthedocs.io/en/latest/jep/11830-new-remat-checkpoint.html).
* Changes
  * Added {func}`jax.pure_callback` that enables calling back to pure Python functions from compiled functions (e.g. functions decorated with `jax.jit` or `jax.pmap`).
* Deprecations:
  * The deprecated `DeviceArray.tile()` method has been removed. Use {func}`jax.numpy.tile`
    ({jax-issue}`#11944`).
  * `DeviceArray.to_py()` has been deprecated. Use `np.asarray(x)` instead.

## jax 0.3.16
* [GitHub commits](https://github.com/google/jax/compare/jax-v0.3.15...main).
* Breaking changes
  * Support for NumPy 1.19 has been dropped, per the
    [deprecation policy](https://jax.readthedocs.io/en/latest/deprecation.html).
    Please upgrade to NumPy 1.20 or newer.
* Changes
  * Added {mod}`jax.debug` that includes utilities for runtime value debugging such at {func}`jax.debug.print` and {func}`jax.debug.breakpoint`.
  * Added new documentation for [runtime value debugging](debugging/index)
* Deprecations
  * {func}`jax.mask` {func}`jax.shapecheck` APIs have been removed.
    See {jax-issue}`#11557`.
  * {mod}`jax.experimental.loops` has been removed. See {jax-issue}`#10278`
    for an alternative API.
  * {func}`jax.tree_util.tree_multimap` has been removed. It has been deprecated since
    JAX release 0.3.5, and {func}`jax.tree_util.tree_map` is a direct replacement.
  * Removed `jax.experimental.stax`; it has long been a deprecated alias of
    {mod}`jax.example_libraries.stax`.
  * Removed `jax.experimental.optimizers`; it has long been a deprecated alias of
    {mod}`jax.example_libraries.optimizers`.
  * {func}`jax.checkpoint`, also known as {func}`jax.remat`, has a new
    implementation switched on by default, meaning the old implementation is
    deprecated; see [JEP 11830](https://jax.readthedocs.io/en/latest/jep/11830-new-remat-checkpoint.html).

## jax 0.3.15 (July 22, 2022)
* [GitHub commits](https://github.com/google/jax/compare/jax-v0.3.14...jax-v0.3.15).
* Changes
  * `JaxTestCase` and `JaxTestLoader` have been removed from `jax.test_util`. These
    classes have been deprecated since v0.3.1 ({jax-issue}`#11248`).
  * Added {class}`jax.scipy.gaussian_kde` ({jax-issue}`#11237`).
  * Binary operations between JAX arrays and built-in collections (`dict`, `list`, `set`, `tuple`)
    now raise a `TypeError` in all cases. Previously some cases (particularly equality and inequality)
    would return boolean scalars inconsistent with similar operations in NumPy ({jax-issue}`#11234`).
  * Several {mod}`jax.tree_util` routines accessed as top-level JAX package imports are now
    deprecated, and will be removed in a future JAX release in accordance with the
    {ref}`api-compatibility` policy:
    * {func}`jax.treedef_is_leaf` is deprecated in favor of {func}`jax.tree_util.treedef_is_leaf`
    * {func}`jax.tree_flatten` is deprecated in favor of {func}`jax.tree_util.tree_flatten`
    * {func}`jax.tree_leaves` is deprecated in favor of {func}`jax.tree_util.tree_leaves`
    * {func}`jax.tree_structure` is deprecated in favor of {func}`jax.tree_util.tree_structure`
    * {func}`jax.tree_transpose` is deprecated in favor of {func}`jax.tree_util.tree_transpose`
    * {func}`jax.tree_unflatten` is deprecated in favor of {func}`jax.tree_util.tree_unflatten`
  * The `sym_pos` argument of {func}`jax.scipy.linalg.solve` is deprecated in favor of `assume_a='pos'`,
    following a similar deprecation in {func}`scipy.linalg.solve`.

## jaxlib 0.3.15 (July 22, 2022)
* [GitHub commits](https://github.com/google/jax/compare/jaxlib-v0.3.14...jaxlib-v0.3.15).

## jax 0.3.14 (June 27, 2022)
* [GitHub commits](https://github.com/google/jax/compare/jax-v0.3.13...jax-v0.3.14).
* Breaking changes
  * {func}`jax.experimental.compilation_cache.initialize_cache` does not support
    `max_cache_size_  bytes` anymore and will not get that as an input.
  * `JAX_PLATFORMS` now raises an exception when platform initialization fails.
* Changes
  * Fixed compatibility problems with NumPy 1.23.
  * {func}`jax.numpy.linalg.slogdet` now accepts an optional `method` argument
    that allows selection between an LU-decomposition based implementation and
    an implementation based on QR decomposition.
  * {func}`jax.numpy.linalg.qr` now supports `mode="raw"`.
  * `pickle`, `copy.copy`, and `copy.deepcopy` now have more complete support when
    used on jax arrays ({jax-issue}`#10659`). In particular:
    - `pickle` and `deepcopy` previously returned `np.ndarray` objects when used
      on a `DeviceArray`; now `DeviceArray` objects are returned. For `deepcopy`,
      the copied array is on the same device as the original. For `pickle` the
      deserialized array will be on the default device.
    - Within function transformations (i.e. traced code), `deepcopy` and `copy`
      previously were no-ops. Now they use the same mechanism as `DeviceArray.copy()`.
    - Calling `pickle` on a traced array now results in an explicit
      `ConcretizationTypeError`.
  * The implementation of singular value decomposition (SVD) and
    symmetric/Hermitian eigendecomposition should be significantly faster on
    TPU, especially for matrices above 1000x1000 or so. Both now use a spectral
    divide-and-conquer algorithm for eigendecomposition (QDWH-eig).
  * {func}`jax.numpy.ldexp` no longer silently promotes all inputs to float64,
    instead it promotes to float32 for integer inputs of size int32 or smaller
    ({jax-issue}`#10921`).
  * Add a `create_perfetto_link` option to {func}`jax.profiler.start_trace` and
    {func}`jax.profiler.start_trace`. When used, the profiler will generate a
    link to the Perfetto UI to view the trace.
  * Changed the semantics of {func}`jax.profiler.start_server(...)` to store the
    keepalive globally, rather than requiring the user to keep a reference to
    it.
  * Added {func}`jax.random.generalized_normal`.
  * Added {func}`jax.random.ball`.
  * Added {func}`jax.default_device`.
  * Added a `python -m jax.collect_profile` script to manually capture program
    traces as an alternative to the Tensorboard UI.
  * Added a `jax.named_scope` context manager that adds profiler metadata to
    Python programs (similar to `jax.named_call`).
  * In scatter-update operations (i.e. :attr:`jax.numpy.ndarray.at`), unsafe implicit
    dtype casts are deprecated, and now result in a `FutureWarning`.
    In a future release, this will become an error. An example of an unsafe implicit
    cast is `jnp.zeros(4, dtype=int).at[0].set(1.5)`, in which `1.5` previously was
    silently truncated to `1`.
  * {func}`jax.experimental.compilation_cache.initialize_cache` now supports gcs
    bucket path as input.
  * Added {func}`jax.scipy.stats.gennorm`.
  * {func}`jax.numpy.roots` is now better behaved when `strip_zeros=False` when
    coefficients have leading zeros ({jax-issue}`#11215`).

## jaxlib 0.3.14 (June 27, 2022)
* [GitHub commits](https://github.com/google/jax/compare/jaxlib-v0.3.10...jaxlib-v0.3.14).
  * x86-64 Mac wheels now require Mac OS 10.14 (Mojave) or newer. Mac OS 10.14
    was released in 2018, so this should not be a very onerous requirement.
  * The bundled version of NCCL was updated to 2.12.12, fixing some deadlocks.
  * The Python flatbuffers package is no longer a dependency of jaxlib.

## jax 0.3.13 (May 16, 2022)
* [GitHub commits](https://github.com/google/jax/compare/jax-v0.3.12...jax-v0.3.13).

## jax 0.3.12 (May 15, 2022)
* [GitHub commits](https://github.com/google/jax/compare/jax-v0.3.11...jax-v0.3.12).
* Changes
  * Fixes [#10717](https://github.com/google/jax/issues/10717).

## jax 0.3.11 (May 15, 2022)
* [GitHub commits](https://github.com/google/jax/compare/jax-v0.3.10...jax-v0.3.11).
* Changes
  * {func}`jax.lax.eigh` now accepts an optional `sort_eigenvalues` argument
    that allows users to opt out of eigenvalue sorting on TPU.
* Deprecations
  * Non-array arguments to functions in {mod}`jax.lax.linalg` are now marked
    keyword-only. As a backward-compatibility step passing keyword-only
    arguments positionally yields a warning, but in a future JAX release passing
    keyword-only arguments positionally will fail.
    However, most users should prefer to use {mod}`jax.numpy.linalg` instead.
  * {func}`jax.scipy.linalg.polar_unitary`, which was a JAX extension to the
    scipy API, is deprecated. Use {func}`jax.scipy.linalg.polar` instead.

## jax 0.3.10 (May 3, 2022)
* [GitHub commits](https://github.com/google/jax/compare/jax-v0.3.9...jax-v0.3.10).

## jaxlib 0.3.10 (May 3, 2022)
* [GitHub commits](https://github.com/google/jax/compare/jaxlib-v0.3.7...jaxlib-v0.3.10).
* Changes
  * [TF commit](https://github.com/tensorflow/tensorflow/commit/207d50d253e11c3a3430a700af478a1d524a779a)
    fixes an issue in the MHLO canonicalizer that caused constant folding to
    take a long time or crash for certain programs.

## jax 0.3.9 (May 2, 2022)
* [GitHub commits](https://github.com/google/jax/compare/jax-v0.3.8...jax-v0.3.9).
* Changes
  * Added support for fully asynchronous checkpointing for GlobalDeviceArray.

## jax 0.3.8 (April 29 2022)
* [GitHub commits](https://github.com/google/jax/compare/jax-v0.3.7...jax-v0.3.8).
* Changes
  * {func}`jax.numpy.linalg.svd` on TPUs uses a qdwh-svd solver.
  * {func}`jax.numpy.linalg.cond` on TPUs now accepts complex input.
  * {func}`jax.numpy.linalg.pinv` on TPUs now accepts complex input.
  * {func}`jax.numpy.linalg.matrix_rank` on TPUs now accepts complex input.
  * {func}`jax.scipy.cluster.vq.vq` has been added.
  * `jax.experimental.maps.mesh` has been deleted.
    Please use `jax.experimental.maps.Mesh`. Please see https://jax.readthedocs.io/en/latest/_autosummary/jax.experimental.maps.Mesh.html#jax.experimental.maps.Mesh
    for more information.
  * {func}`jax.scipy.linalg.qr` now returns a length-1 tuple rather than the raw array when
    `mode='r'`, in order to match the behavior of `scipy.linalg.qr` ({jax-issue}`#10452`)
  * {func}`jax.numpy.take_along_axis` now takes an optional `mode` parameter
    that specifies the behavior of out-of-bounds indexing. By default,
    invalid values (e.g., NaN) will be returned for out-of-bounds indices. In
    previous versions of JAX, invalid indices were clamped into range. The
    previous behavior can be restored by passing `mode="clip"`.
  * {func}`jax.numpy.take` now defaults to `mode="fill"`, which returns
    invalid values (e.g., NaN) for out-of-bounds indices.
  * Scatter operations, such as `x.at[...].set(...)`, now have `"drop"` semantics.
    This has no effect on the scatter operation itself, but it means that when
    differentiated the gradient of a scatter will yield zero cotangents for
    out-of-bounds indices. Previously out-of-bounds indices were clamped into
    range for the gradient, which was not mathematically correct.
  * {func}`jax.numpy.take_along_axis` now raises a `TypeError` if its indices
    are not of an integer type, matching the behavior of
    {func}`numpy.take_along_axis`. Previously non-integer indices were silently
    cast to integers.
  * {func}`jax.numpy.ravel_multi_index` now raises a `TypeError` if its `dims` argument
    is not of an integer type, matching the behavior of
    {func}`numpy.ravel_multi_index`. Previously non-integer `dims` was silently
    cast to integers.
  * {func}`jax.numpy.split` now raises a `TypeError` if its `axis` argument
    is not of an integer type, matching the behavior of
    {func}`numpy.split`. Previously non-integer `axis` was silently
    cast to integers.
  * {func}`jax.numpy.indices` now raises a `TypeError` if its dimensions
    are not of an integer type, matching the behavior of
    {func}`numpy.indices`. Previously non-integer dimensions were silently
    cast to integers.
  * {func}`jax.numpy.diag` now raises a `TypeError` if its `k` argument
    is not of an integer type, matching the behavior of
    {func}`numpy.diag`. Previously non-integer `k` was silently
    cast to integers.
  * Added {func}`jax.random.orthogonal`.
* Deprecations
  * Many functions and objects available in {mod}`jax.test_util` are now deprecated and will raise a
    warning on import. This includes `cases_from_list`, `check_close`, `check_eq`, `device_under_test`,
    `format_shape_dtype_string`, `rand_uniform`, `skip_on_devices`, `with_config`, `xla_bridge`, and
    `_default_tolerance` ({jax-issue}`#10389`). These, along with previously-deprecated `JaxTestCase`,
    `JaxTestLoader`, and `BufferDonationTestCase`, will be removed in a future JAX release.
    Most of these utilities can be replaced by calls to standard python & numpy testing utilities found
    in e.g.  {mod}`unittest`, {mod}`absl.testing`, {mod}`numpy.testing`, etc. JAX-specific functionality
    such as device checking can be replaced through the use of public APIs such as {func}`jax.devices`.
    Many of the deprecated utilities will still exist in {mod}`jax._src.test_util`, but these are not
    public APIs and as such may be changed or removed without notice in future releases.

## jax 0.3.7 (April 15, 2022)
* [GitHub
  commits](https://github.com/google/jax/compare/jax-v0.3.6...jax-v0.3.7).
* Changes:
  * Fixed a performance problem if the indices passed to
    {func}`jax.numpy.take_along_axis` were broadcasted ({jax-issue}`#10281`).
  * {func}`jax.scipy.special.expit` and {func}`jax.scipy.special.logit` now
    require their arguments to be scalars or JAX arrays. They also now promote
    integer arguments to floating point.
  * The `DeviceArray.tile()` method is deprecated, because numpy arrays do not have a
    `tile()` method. As a replacement for this, use {func}`jax.numpy.tile`
    ({jax-issue}`#10266`).

## jaxlib 0.3.7 (April 15, 2022)
* Changes:
  * Linux wheels are now built conforming to the `manylinux2014` standard, instead
    of `manylinux2010`.

## jax 0.3.6 (April 12, 2022)
* [GitHub
  commits](https://github.com/google/jax/compare/jax-v0.3.5...jax-v0.3.6).
* Changes:
  * Upgraded libtpu wheel to a version that fixes a hang when initializing a TPU
    pod. Fixes [#10218](https://github.com/google/jax/issues/10218).
* Deprecations:
  * {mod}`jax.experimental.loops` is being deprecated. See {jax-issue}`#10278`
    for an alternative API.

## jax 0.3.5 (April 7, 2022)
* [GitHub
  commits](https://github.com/google/jax/compare/jax-v0.3.4...jax-v0.3.5).
* Changes:
  * added {func}`jax.random.loggamma` & improved behavior of {func}`jax.random.beta`
    and {func}`jax.random.dirichlet` for small parameter values ({jax-issue}`#9906`).
  * the private `lax_numpy` submodule is no longer exposed in the `jax.numpy` namespace ({jax-issue}`#10029`).
  * added array creation routines {func}`jax.numpy.frombuffer`, {func}`jax.numpy.fromfunction`,
    and {func}`jax.numpy.fromstring` ({jax-issue}`#10049`).
  * `DeviceArray.copy()` now returns a `DeviceArray` rather than a `np.ndarray` ({jax-issue}`#10069`)
  * added {func}`jax.scipy.linalg.rsf2csf`
  * `jax.experimental.sharded_jit` has been deprecated and will be removed soon.
* Deprecations:
  * {func}`jax.nn.normalize` is being deprecated. Use {func}`jax.nn.standardize` instead ({jax-issue}`#9899`).
  * {func}`jax.tree_util.tree_multimap` is deprecated. Use {func}`jax.tree_util.tree_map` instead ({jax-issue}`#5746`).
  * `jax.experimental.sharded_jit` is deprecated. Use `pjit` instead.

## jaxlib 0.3.5 (April 7, 2022)
* Bug fixes
  * Fixed a bug where double-precision complex-to-real IRFFTs would mutate their
    input buffers on GPU ({jax-issue}`#9946`).
  * Fixed incorrect constant-folding of complex scatters ({jax-issue}`#10159`)

## jax 0.3.4 (March 18, 2022)
* [GitHub
  commits](https://github.com/google/jax/compare/jax-v0.3.3...jax-v0.3.4).


## jax 0.3.3 (March 17, 2022)
* [GitHub
  commits](https://github.com/google/jax/compare/jax-v0.3.2...jax-v0.3.3).


## jax 0.3.2 (March 16, 2022)
* [GitHub
  commits](https://github.com/google/jax/compare/jax-v0.3.1...jax-v0.3.2).
* Changes:
  * The functions `jax.ops.index_update`, `jax.ops.index_add`, which were
    deprecated in 0.2.22, have been removed. Please use
    [the `.at` property on JAX arrays](https://jax.readthedocs.io/en/latest/_autosummary/jax.numpy.ndarray.at.html)
    instead, e.g., `x.at[idx].set(y)`.
  * Moved `jax.experimental.ann.approx_*_k` into `jax.lax`. These functions are
    optimized alternatives to `jax.lax.top_k`.
  * {func}`jax.numpy.broadcast_arrays` and {func}`jax.numpy.broadcast_to` now require scalar
    or array-like inputs, and will fail if they are passed lists (part of {jax-issue}`#7737`).
  * The standard jax[tpu] install can now be used with Cloud TPU v4 VMs.
  * `pjit` now works on CPU (in addition to previous TPU and GPU support).


## jaxlib 0.3.2 (March 16, 2022)
* Changes
  * ``XlaComputation.as_hlo_text()`` now supports printing large constants by
    passing boolean flag ``print_large_constants=True``.
* Deprecations:
  * The ``.block_host_until_ready()`` method on JAX arrays has been deprecated.
    Use ``.block_until_ready()`` instead.

## jax 0.3.1 (Feb 18, 2022)
* [GitHub
  commits](https://github.com/google/jax/compare/jax-v0.3.0...jax-v0.3.1).

* Changes:
  * `jax.test_util.JaxTestCase` and `jax.test_util.JaxTestLoader` are now deprecated.
    The suggested replacement is to use `parametrized.TestCase` directly. For tests that
    rely on custom asserts such as `JaxTestCase.assertAllClose()`, the suggested replacement
    is to use standard numpy testing utilities such as {func}`numpy.testing.assert_allclose()`,
    which work directly with JAX arrays ({jax-issue}`#9620`).
  * `jax.test_util.JaxTestCase` now sets `jax_numpy_rank_promotion='raise'` by default
    ({jax-issue}`#9562`). To recover the previous behavior, use the new
    `jax.test_util.with_config` decorator:
    ```python
    @jtu.with_config(jax_numpy_rank_promotion='allow')
    class MyTestCase(jtu.JaxTestCase):
      ...
    ```
  * Added {func}`jax.scipy.linalg.schur`, {func}`jax.scipy.linalg.sqrtm`,
    {func}`jax.scipy.signal.csd`, {func}`jax.scipy.signal.stft`,
    {func}`jax.scipy.signal.welch`.


## jax 0.3.0 (Feb 10, 2022)
* [GitHub
  commits](https://github.com/google/jax/compare/jax-v0.2.28...jax-v0.3.0).

* Changes
  * jax version has been bumped to 0.3.0. Please see the [design doc](https://jax.readthedocs.io/en/latest/design_notes/jax_versioning.html)
    for the explanation.

## jaxlib 0.3.0 (Feb 10, 2022)
* Changes
  * Bazel 5.0.0 is now required to build jaxlib.
  * jaxlib version has been bumped to 0.3.0. Please see the [design doc](https://jax.readthedocs.io/en/latest/design_notes/jax_versioning.html)
    for the explanation.

## jax 0.2.28 (Feb 1, 2022)
* [GitHub
  commits](https://github.com/google/jax/compare/jax-v0.2.27...jax-v0.2.28).
  * `jax.jit(f).lower(...).compiler_ir()` now defaults to the MHLO dialect if no
    `dialect=` is passed.
  * The `jax.jit(f).lower(...).compiler_ir(dialect='mhlo')` now returns an MLIR
    `ir.Module` object instead of its string representation.

## jaxlib 0.1.76 (Jan 27, 2022)

* New features
  * Includes precompiled SASS for NVidia compute capability 8.0 GPUS
    (e.g. A100). Removes precompiled SASS for compute capability 6.1 so as not
    to increase the number of compute capabilities: GPUs with compute capability
    6.1 can use the 6.0 SASS.
  * With jaxlib 0.1.76, JAX uses the MHLO MLIR dialect as its primary target compiler IR
    by default.
* Breaking changes
  * Support for NumPy 1.18 has been dropped, per the
    [deprecation policy](https://jax.readthedocs.io/en/latest/deprecation.html).
    Please upgrade to a supported NumPy version.
* Bug fixes
  * Fixed a bug where apparently identical pytreedef objects constructed by different routes
    do not compare as equal (#9066).
  * The JAX jit cache requires two static arguments to have identical types for a cache hit (#9311).

## jax 0.2.27 (Jan 18 2022)
* [GitHub commits](https://github.com/google/jax/compare/jax-v0.2.26...jax-v0.2.27).

* Breaking changes:
  * Support for NumPy 1.18 has been dropped, per the
    [deprecation policy](https://jax.readthedocs.io/en/latest/deprecation.html).
    Please upgrade to a supported NumPy version.
  * The host_callback primitives have been simplified to drop the
    special autodiff handling for hcb.id_tap and id_print.
    From now on, only the primals are tapped. The old behavior can be
    obtained (for a limited time) by setting the ``JAX_HOST_CALLBACK_AD_TRANSFORMS``
    environment variable, or the ```--flax_host_callback_ad_transforms``` flag.
    Additionally, added documentation for how to implement the old behavior
    using JAX custom AD APIs ({jax-issue}`#8678`).
  * Sorting now matches the behavior of NumPy for ``0.0`` and ``NaN`` regardless of the
    bit representation. In particular, ``0.0`` and ``-0.0`` are now treated as equivalent,
    where previously ``-0.0`` was treated as less than ``0.0``. Additionally all ``NaN``
    representations are now treated as equivalent and sorted to the end of the array.
    Previously negative ``NaN`` values were sorted to the front of the array, and ``NaN``
    values with different internal bit representations were not treated as equivalent, and
    were sorted according to those bit patterns ({jax-issue}`#9178`).
  * {func}`jax.numpy.unique` now treats ``NaN`` values in the same way as `np.unique` in
    NumPy versions 1.21 and newer: at most one ``NaN`` value will appear in the uniquified
    output ({jax-issue}`9184`).

* Bug fixes:
  * host_callback now supports ad_checkpoint.checkpoint ({jax-issue}`#8907`).

* New features:
  * add `jax.block_until_ready` ({jax-issue}`#8941)
  * Added a new debugging flag/environment variable `JAX_DUMP_IR_TO=/path`.
    If set, JAX dumps the MHLO/HLO IR it generates for each computation to a
    file under the given path.
  * Added `jax.ensure_compile_time_eval` to the public api ({jax-issue}`#7987`).
  * jax2tf now supports a flag jax2tf_associative_scan_reductions to change
    the lowering for associative reductions, e.g., jnp.cumsum, to behave
    like JAX on CPU and GPU (to use an associative scan). See the jax2tf README
    for more details ({jax-issue}`#9189`).


## jaxlib 0.1.75 (Dec 8, 2021)
* New features:
  * Support for python 3.10.

## jax 0.2.26 (Dec 8, 2021)
* [GitHub
  commits](https://github.com/google/jax/compare/jax-v0.2.25...jax-v0.2.26).

* Bug fixes:
  * Out-of-bounds indices to `jax.ops.segment_sum` will now be handled with
    `FILL_OR_DROP` semantics, as documented. This primarily affects the
    reverse-mode derivative, where gradients corresponding to out-of-bounds
    indices will now be returned as 0. (#8634).
  * jax2tf will force the converted code to use XLA for the code fragments
    under jax.jit, e.g., most jax.numpy functions ({jax-issue}`#7839`).

## jaxlib 0.1.74 (Nov 17, 2021)
* Enabled peer-to-peer copies between GPUs. Previously, GPU copies were bounced via
  the host, which is usually slower.
* Added experimental MLIR Python bindings for use by JAX.

## jax 0.2.25 (Nov 10, 2021)
* [GitHub
  commits](https://github.com/google/jax/compare/jax-v0.2.24...jax-v0.2.25).

* New features:
  * (Experimental) `jax.distributed.initialize` exposes multi-host GPU backend.
  * `jax.random.permutation` supports new `independent` keyword argument
    ({jax-issue}`#8430`)
* Breaking changes
  * Moved `jax.experimental.stax` to `jax.example_libraries.stax`
  * Moved `jax.experimental.optimizers` to `jax.example_libraries.optimizers`
* New features:
  * Added `jax.lax.linalg.qdwh`.

## jax 0.2.24 (Oct 19, 2021)
* [GitHub
  commits](https://github.com/google/jax/compare/jax-v0.2.22...jax-v0.2.24).

* New features:
  * `jax.random.choice` and `jax.random.permutation` now support
    multidimensional arrays and an optional `axis` argument ({jax-issue}`#8158`)
* Breaking changes:
  * `jax.numpy.take` and `jax.numpy.take_along_axis` now require array-like inputs
    (see {jax-issue}`#7737`)

## jaxlib 0.1.73 (Oct 18, 2021)

* Multiple cuDNN versions are now supported for jaxlib GPU `cuda11` wheels.
  * cuDNN 8.2 or newer. We recommend using the cuDNN 8.2 wheel if your cuDNN
    installation is new enough, since it supports additional functionality.
  * cuDNN 8.0.5 or newer.

* Breaking changes:
  * The install commands for GPU jaxlib are as follows:

    ```bash
    pip install --upgrade pip

    # Installs the wheel compatible with CUDA 11 and cuDNN 8.2 or newer.
    pip install --upgrade "jax[cuda]" -f https://storage.googleapis.com/jax-releases/jax_releases.html

    # Installs the wheel compatible with Cuda 11 and cudnn 8.2 or newer.
    pip install jax[cuda11_cudnn82] -f https://storage.googleapis.com/jax-releases/jax_releases.html

    # Installs the wheel compatible with Cuda 11 and cudnn 8.0.5 or newer.
    pip install jax[cuda11_cudnn805] -f https://storage.googleapis.com/jax-releases/jax_releases.html
    ```

## jax 0.2.22 (Oct 12, 2021)
* [GitHub
  commits](https://github.com/google/jax/compare/jax-v0.2.21...jax-v0.2.22).
* Breaking Changes
  * Static arguments to `jax.pmap` must now be hashable.

    Unhashable static arguments have long been disallowed on `jax.jit`, but they
    were still permitted on `jax.pmap`; `jax.pmap` compared unhashable static
    arguments using object identity.

    This behavior is a footgun, since comparing arguments using
    object identity leads to recompilation each time the object identity
    changes. Instead, we now ban unhashable arguments: if a user of `jax.pmap`
    wants to compare static arguments by object identity, they can define
    `__hash__` and `__eq__` methods on their objects that do that, or wrap their
    objects in an object that has those operations with object identity
    semantics. Another option is to use `functools.partial` to encapsulate the
    unhashable static arguments into the function object.
  * `jax.util.partial` was an accidental export that has now been removed. Use
    `functools.partial` from the Python standard library instead.
* Deprecations
  * The functions `jax.ops.index_update`, `jax.ops.index_add` etc. are
    deprecated and will be removed in a future JAX release. Please use
    [the `.at` property on JAX arrays](https://jax.readthedocs.io/en/latest/_autosummary/jax.numpy.ndarray.at.html)
    instead, e.g., `x.at[idx].set(y)`. For now, these functions produce a
    `DeprecationWarning`.
* New features:
  * An optimized C++ code-path improving the dispatch time for `pmap` is now the
    default when using jaxlib 0.1.72 or newer. The feature can be disabled using
    the `--experimental_cpp_pmap` flag (or `JAX_CPP_PMAP` environment variable).
  * `jax.numpy.unique` now supports an optional `fill_value` argument ({jax-issue}`#8121`)

## jaxlib 0.1.72 (Oct 12, 2021)
  * Breaking changes:
    * Support for CUDA 10.2 and CUDA 10.1 has been dropped. Jaxlib now supports
      CUDA 11.1+.
  * Bug fixes:
    * Fixes https://github.com/google/jax/issues/7461, which caused wrong
      outputs on all platforms due to incorrect buffer aliasing inside the XLA
      compiler.

## jax 0.2.21 (Sept 23, 2021)
* [GitHub
  commits](https://github.com/google/jax/compare/jax-v0.2.20...jax-v0.2.21).
* Breaking Changes
  * `jax.api` has been removed. Functions that were available as `jax.api.*`
    were aliases for functions in `jax.*`; please use the functions in
    `jax.*` instead.
  * `jax.partial`, and `jax.lax.partial` were accidental exports that have now
    been removed. Use `functools.partial` from the Python standard library
    instead.
  * Boolean scalar indices now raise a `TypeError`; previously this silently
    returned wrong results ({jax-issue}`#7925`).
  * Many more `jax.numpy` functions now require array-like inputs, and will error
    if passed a list ({jax-issue}`#7747` {jax-issue}`#7802` {jax-issue}`#7907`).
    See {jax-issue}`#7737` for a discussion of the rationale behind this change.
  * When inside a transformation such as `jax.jit`, `jax.numpy.array` always
    stages the array it produces into the traced computation. Previously
    `jax.numpy.array` would sometimes produce a on-device array, even under
    a `jax.jit` decorator. This change may break code that used JAX arrays to
    perform shape or index computations that must be known statically; the
    workaround is to perform such computations using classic NumPy arrays
    instead.
  * `jnp.ndarray` is now a true base-class for JAX arrays. In particular, this
    means that for a standard numpy array `x`, `isinstance(x, jnp.ndarray)` will
    now return `False` ({jax-issue}`7927`).
* New features:
  * Added {func}`jax.numpy.insert` implementation ({jax-issue}`#7936`).

## jax 0.2.20 (Sept 2, 2021)
* [GitHub
  commits](https://github.com/google/jax/compare/jax-v0.2.19...jax-v0.2.20).
* Breaking Changes
  * `jnp.poly*` functions now require array-like inputs ({jax-issue}`#7732`)
  * `jnp.unique` and other set-like operations now require array-like inputs
    ({jax-issue}`#7662`)

## jaxlib 0.1.71 (Sep 1, 2021)
* Breaking changes:
  * Support for CUDA 11.0 and CUDA 10.1 has been dropped. Jaxlib now supports
    CUDA 10.2 and CUDA 11.1+.

## jax 0.2.19 (Aug 12, 2021)
* [GitHub
  commits](https://github.com/google/jax/compare/jax-v0.2.18...jax-v0.2.19).
* Breaking changes:
  * Support for NumPy 1.17 has been dropped, per the
    [deprecation policy](https://jax.readthedocs.io/en/latest/deprecation.html).
    Please upgrade to a supported NumPy version.
  * The `jit` decorator has been added around the implementation of a number of
    operators on JAX arrays. This speeds up dispatch times for common
    operators such as `+`.

    This change should largely be transparent to most users. However, there is
    one known behavioral change, which is that large integer constants may now
    produce an error when passed directly to a JAX operator
    (e.g., `x + 2**40`). The workaround is to cast the constant to an
    explicit type (e.g., `np.float64(2**40)`).
* New features:
  * Improved the support for shape polymorphism in jax2tf for operations that
    need to use a dimension size in array computation, e.g., `jnp.mean`.
    ({jax-issue}`#7317`)
* Bug fixes:
  * Some leaked trace errors from the previous release ({jax-issue}`#7613`)

## jaxlib 0.1.70 (Aug 9, 2021)
* Breaking changes:
  * Support for Python 3.6 has been dropped, per the
    [deprecation policy](https://jax.readthedocs.io/en/latest/deprecation.html).
    Please upgrade to a supported Python version.
  * Support for NumPy 1.17 has been dropped, per the
    [deprecation policy](https://jax.readthedocs.io/en/latest/deprecation.html).
    Please upgrade to a supported NumPy version.

  * The host_callback mechanism now uses one thread per local device for
    making the calls to the Python callbacks. Previously there was a single
    thread for all devices. This means that the callbacks may now be called
    interleaved. The callbacks corresponding to one device will still be
    called in sequence.

## jax 0.2.18 (July 21 2021)
* [GitHub commits](https://github.com/google/jax/compare/jax-v0.2.17...jax-v0.2.18).

* Breaking changes:
  * Support for Python 3.6 has been dropped, per the
    [deprecation policy](https://jax.readthedocs.io/en/latest/deprecation.html).
    Please upgrade to a supported Python version.
  * The minimum jaxlib version is now 0.1.69.
  * The `backend` argument to {py:func}`jax.dlpack.from_dlpack` has been
    removed.

* New features:
  * Added a polar decomposition ({py:func}`jax.scipy.linalg.polar`).

* Bug fixes:
  * Tightened the checks for lax.argmin and lax.argmax to ensure they are
    not used with an invalid `axis` value, or with an empty reduction dimension.
    ({jax-issue}`#7196`)


## jaxlib 0.1.69 (July 9 2021)
* Fix bugs in TFRT CPU backend that results in incorrect results.

## jax 0.2.17 (July 9 2021)
* [GitHub commits](https://github.com/google/jax/compare/jax-v0.2.16...jax-v0.2.17).
* Bug fixes:
  * Default to the older "stream_executor" CPU runtime for jaxlib <= 0.1.68
    to work around #7229, which caused wrong outputs on CPU due to a concurrency
    problem.
* New features:
  * New SciPy function {py:func}`jax.scipy.special.sph_harm`.
  * Reverse-mode autodiff functions ({func}`jax.grad`,
    {func}`jax.value_and_grad`, {func}`jax.vjp`, and
    {func}`jax.linear_transpose`) support a parameter that indicates which named
    axes should be summed over in the backward pass if they were broadcasted
    over in the forward pass. This enables use of these APIs in a
    non-per-example way inside maps (initially only
    {func}`jax.experimental.maps.xmap`) ({jax-issue}`#6950`).


## jax 0.2.16 (June 23 2021)
* [GitHub commits](https://github.com/google/jax/compare/jax-v0.2.15...jax-v0.2.16).

## jax 0.2.15 (June 23 2021)
* [GitHub commits](https://github.com/google/jax/compare/jax-v0.2.14...jax-v0.2.15).
* New features:
  * [#7042](https://github.com/google/jax/pull/7042) Turned on TFRT CPU backend
    with significant dispatch performance improvements on CPU.
  * The {func}`jax2tf.convert` supports inequalities and min/max for booleans
    ({jax-issue}`#6956`).
  * New SciPy function {py:func}`jax.scipy.special.lpmn_values`.

* Breaking changes:
  * Support for NumPy 1.16 has been dropped, per the
    [deprecation policy](https://jax.readthedocs.io/en/latest/deprecation.html).

* Bug fixes:
  * Fixed bug that prevented round-tripping from JAX to TF and back:
    `jax2tf.call_tf(jax2tf.convert)` ({jax-issue}`#6947`).

## jaxlib 0.1.68 (June 23 2021)
* Bug fixes:
  * Fixed bug in TFRT CPU backend that gets nans when transfer TPU buffer to
    CPU.

## jax 0.2.14 (June 10 2021)
* [GitHub commits](https://github.com/google/jax/compare/jax-v0.2.13...jax-v0.2.14).
* New features:
  * The {func}`jax2tf.convert` now has support for `pjit` and `sharded_jit`.
  * A new configuration option JAX_TRACEBACK_FILTERING controls how JAX filters
    tracebacks.
  * A new traceback filtering mode using `__tracebackhide__` is now enabled by
    default in sufficiently recent versions of IPython.
  * The {func}`jax2tf.convert` supports shape polymorphism even when the
    unknown dimensions are used in arithmetic operations, e.g., `jnp.reshape(-1)`
    ({jax-issue}`#6827`).
  * The {func}`jax2tf.convert` generates custom attributes with location information
   in TF ops. The code that XLA generates after jax2tf
   has the same location information as JAX/XLA.
  * New SciPy function {py:func}`jax.scipy.special.lpmn`.

* Bug fixes:
  * The {func}`jax2tf.convert` now ensures that it uses the same typing rules
    for Python scalars and for choosing 32-bit vs. 64-bit computations
    as JAX ({jax-issue}`#6883`).
  * The {func}`jax2tf.convert` now scopes the `enable_xla` conversion parameter
    properly to apply only during the just-in-time conversion
    ({jax-issue}`#6720`).
  * The {func}`jax2tf.convert` now converts `lax.dot_general` using the
    `XlaDot` TensorFlow op, for better fidelity w.r.t. JAX numerical precision
    ({jax-issue}`#6717`).
  * The {func}`jax2tf.convert` now has support for inequality comparisons and
    min/max for complex numbers ({jax-issue}`#6892`).

## jaxlib 0.1.67 (May 17 2021)

## jaxlib 0.1.66 (May 11 2021)

* New features:
  * CUDA 11.1 wheels are now supported on all CUDA 11 versions 11.1 or higher.

    NVidia now promises compatibility between CUDA minor releases starting with
    CUDA 11.1. This means that JAX can release a single CUDA 11.1 wheel that
    is compatible with CUDA 11.2 and 11.3.

    There is no longer a separate jaxlib release for CUDA 11.2 (or higher); use
    the CUDA 11.1 wheel for those versions (cuda111).
  * Jaxlib now bundles `libdevice.10.bc` in CUDA wheels. There should be no need
    to point JAX to a CUDA installation to find this file.
  * Added automatic support for static keyword arguments to the {func}`jit`
    implementation.
  * Added support for pretransformation exception traces.
  * Initial support for pruning unused arguments from {func}`jit` -transformed
    computations.
    Pruning is still a work in progress.
  * Improved the string representation of {class}`PyTreeDef` objects.
  * Added support for XLA's variadic ReduceWindow.
* Bug fixes:
  * Fixed a bug in the remote cloud TPU support when large numbers of arguments
    are passed to a computation.
  * Fix a bug that meant that JAX garbage collection was not triggered by
    {func}`jit` transformed functions.

## jax 0.2.13 (May 3 2021)
* [GitHub commits](https://github.com/google/jax/compare/jax-v0.2.12...jax-v0.2.13).
* New features:
  * When combined with jaxlib 0.1.66, {func}`jax.jit` now supports static
    keyword arguments. A new `static_argnames` option has been added to specify
    keyword arguments as static.
  * {func}`jax.nonzero` has a new optional `size` argument that allows it to
    be used within `jit` ({jax-issue}`#6501`)
  * {func}`jax.numpy.unique` now supports the `axis` argument ({jax-issue}`#6532`).
  * {func}`jax.experimental.host_callback.call` now supports `pjit.pjit` ({jax-issue}`#6569`).
  * Added {func}`jax.scipy.linalg.eigh_tridiagonal` that computes the
    eigenvalues of a tridiagonal matrix. Only eigenvalues are supported at
    present.
  * The order of the filtered and unfiltered stack traces in exceptions has been
    changed. The traceback attached to an exception thrown from JAX-transformed
    code is now filtered, with an `UnfilteredStackTrace` exception
    containing the original trace as the `__cause__` of the filtered exception.
    Filtered stack traces now also work with Python 3.6.
  * If an exception is thrown by code that has been transformed by reverse-mode
    automatic differentiation, JAX now attempts to attach as a `__cause__` of
    the exception a `JaxStackTraceBeforeTransformation` object that contains the
    stack trace that created the original operation in the forward pass.
    Requires jaxlib 0.1.66.

* Breaking changes:
  * The following function names have changed. There are still aliases, so this
    should not break existing code, but the aliases will eventually be removed
    so please change your code.
    * `host_id` --> {func}`~jax.process_index`
    * `host_count` --> {func}`~jax.process_count`
    * `host_ids` --> `range(jax.process_count())`
  * Similarly, the argument to {func}`~jax.local_devices` has been renamed from
    `host_id` to `process_index`.
  * Arguments to {func}`jax.jit` other than the function are now marked as
    keyword-only. This change is to prevent accidental breakage when arguments
    are added to `jit`.
* Bug fixes:
  * The {func}`jax2tf.convert` now works in presence of gradients for functions
    with integer inputs ({jax-issue}`#6360`).
  * Fixed assertion failure in {func}`jax2tf.call_tf` when used with captured
    `tf.Variable` ({jax-issue}`#6572`).

## jaxlib 0.1.65 (April 7 2021)

## jax 0.2.12 (April 1 2021)
* [GitHub commits](https://github.com/google/jax/compare/jax-v0.2.11...v0.2.12).
* New features
  * New profiling APIs: {func}`jax.profiler.start_trace`,
    {func}`jax.profiler.stop_trace`, and {func}`jax.profiler.trace`
  * {func}`jax.lax.reduce` is now differentiable.
* Breaking changes:
  * The minimum jaxlib version is now 0.1.64.
  * Some profiler APIs names have been changed. There are still aliases, so this
    should not break existing code, but the aliases will eventually be removed
    so please change your code.
    * `TraceContext` --> {func}`~jax.profiler.TraceAnnotation`
    * `StepTraceContext` --> {func}`~jax.profiler.StepTraceAnnotation`
    * `trace_function` --> {func}`~jax.profiler.annotate_function`
  * Omnistaging can no longer be disabled. See [omnistaging](https://github.com/google/jax/blob/main/docs/design_notes/omnistaging.md)
    for more information.
  * Python integers larger than the maximum `int64` value will now lead to an overflow
    in all cases, rather than being silently converted to `uint64` in some cases ({jax-issue}`#6047`).
  * Outside X64 mode, Python integers outside the range representable by `int32` will now lead to an
    `OverflowError` rather than having their value silently truncated.
* Bug fixes:
  * `host_callback` now supports empty arrays in arguments and results ({jax-issue}`#6262`).
  * {func}`jax.random.randint` clips rather than wraps of out-of-bounds limits, and can now generate
    integers in the full range of the specified dtype ({jax-issue}`#5868`)

## jax 0.2.11 (March 23 2021)

* [GitHub
  commits](https://github.com/google/jax/compare/jax-v0.2.10...jax-v0.2.11).
* New features:
  * [#6112](https://github.com/google/jax/pull/6112) added context managers:
    `jax.enable_checks`, `jax.check_tracer_leaks`, `jax.debug_nans`,
    `jax.debug_infs`, `jax.log_compiles`.
  * [#6085](https://github.com/google/jax/pull/6085) added `jnp.delete`

* Bug fixes:
  * [#6136](https://github.com/google/jax/pull/6136) generalized
    `jax.flatten_util.ravel_pytree` to handle integer dtypes.
  * [#6129](https://github.com/google/jax/issues/6129) fixed a bug with handling
    some constants like `enum.IntEnums`
  * [#6145](https://github.com/google/jax/pull/6145) fixed batching issues with
    incomplete beta functions
  * [#6014](https://github.com/google/jax/pull/6014) fixed H2D transfers during
    tracing
  * [#6165](https://github.com/google/jax/pull/6165) avoids OverflowErrors when
    converting some large Python integers to floats
* Breaking changes:
  * The minimum jaxlib version is now 0.1.62.


## jaxlib 0.1.64 (March 18 2021)

## jaxlib 0.1.63 (March 17 2021)

## jax 0.2.10 (March 5 2021)

* [GitHub commits](https://github.com/google/jax/compare/jax-v0.2.9...jax-v0.2.10).
* New features:
  * {func}`jax.scipy.stats.chi2` is now available as a distribution with logpdf and pdf methods.
  * {func}`jax.scipy.stats.betabinom` is now available as a distribution with logpmf and pmf methods.
  * Added {func}`jax.experimental.jax2tf.call_tf` to call TensorFlow functions
    from JAX ({jax-issue}`#5627`)
    and [README](https://github.com/google/jax/blob/main/jax/experimental/jax2tf/README.md#calling-tensorflow-functions-from-jax)).
  * Extended the batching rule for `lax.pad` to support batching of the padding values.
* Bug fixes:
  * {func}`jax.numpy.take` properly handles negative indices ({jax-issue}`#5768`)
* Breaking changes:
  * JAX's promotion rules were adjusted to make promotion more consistent and
    invariant to JIT. In particular, binary operations can now result in weakly-typed
    values when appropriate. The main user-visible effect of the change is that
    some operations result in outputs of different precision than before; for
    example the expression `jnp.bfloat16(1) + 0.1 * jnp.arange(10)`
    previously returned a `float64` array, and now returns a `bfloat16` array.
    JAX's type promotion behavior is described at {ref}`type-promotion`.
  * {func}`jax.numpy.linspace` now computes the floor of integer values, i.e.,
    rounding towards -inf rather than 0. This change was made to match NumPy
    1.20.0.
  * {func}`jax.numpy.i0` no longer accepts complex numbers. Previously the
    function computed the absolute value of complex arguments. This change was
    made to match the semantics of NumPy 1.20.0.
  * Several {mod}`jax.numpy` functions no longer accept tuples or lists in place
    of array arguments: {func}`jax.numpy.pad`, :func`jax.numpy.ravel`,
    {func}`jax.numpy.repeat`, {func}`jax.numpy.reshape`.
    In general, {mod}`jax.numpy` functions should be used with scalars or array arguments.

## jaxlib 0.1.62 (March 9 2021)

* New features:
  * jaxlib wheels are now built to require AVX instructions on x86-64 machines
    by default. If you want to use JAX on a machine that doesn't support AVX,
    you can build a jaxlib from source using the `--target_cpu_features` flag
    to `build.py`. `--target_cpu_features` also replaces
    `--enable_march_native`.

## jaxlib 0.1.61 (February 12 2021)

## jaxlib 0.1.60 (February 3 2021)

* Bug fixes:
  * Fixed a memory leak when converting CPU DeviceArrays to NumPy arrays. The
    memory leak was present in jaxlib releases 0.1.58 and 0.1.59.
  * `bool`, `int8`, and `uint8` are now considered safe to cast to
    `bfloat16` NumPy extension type.

## jax 0.2.9 (January 26 2021)

* [GitHub commits](https://github.com/google/jax/compare/jax-v0.2.8...jax-v0.2.9).
* New features:
  * Extend the {mod}`jax.experimental.loops` module with support for pytrees. Improved
    error checking and error messages.
  * Add {func}`jax.experimental.enable_x64` and {func}`jax.experimental.disable_x64`.
    These are context managers which allow X64 mode to be temporarily enabled/disabled
    within a session.
* Breaking changes:
  * {func}`jax.ops.segment_sum` now drops segment IDs that are out of range rather
    than wrapping them into the segment ID space. This was done for performance
    reasons.

## jaxlib 0.1.59 (January 15 2021)

## jax 0.2.8 (January 12 2021)

* [GitHub commits](https://github.com/google/jax/compare/jax-v0.2.7...jax-v0.2.8).
* New features:
  * Add {func}`jax.closure_convert` for use with higher-order custom
    derivative functions. ({jax-issue}`#5244`)
  * Add {func}`jax.experimental.host_callback.call` to call a custom Python
    function on the host and return a result to the device computation.
    ({jax-issue}`#5243`)
* Bug fixes:
  * `jax.numpy.arccosh` now returns the same branch as `numpy.arccosh` for
    complex inputs ({jax-issue}`#5156`)
  * `host_callback.id_tap` now works for `jax.pmap` also. There is an
    optional parameter for `id_tap` and `id_print` to request that the
    device from which the value is tapped be passed as a keyword argument
    to the tap function ({jax-issue}`#5182`).
* Breaking changes:
  * `jax.numpy.pad` now takes keyword arguments. Positional argument `constant_values`
    has been removed. In addition, passing unsupported keyword arguments raises an error.
  * Changes for {func}`jax.experimental.host_callback.id_tap` ({jax-issue}`#5243`):
    * Removed support for `kwargs` for {func}`jax.experimental.host_callback.id_tap`.
      (This support has been deprecated for a few months.)
    * Changed the printing of tuples for {func}`jax.experimental.host_callback.id_print`
      to use '(' instead of '['.
    * Changed the {func}`jax.experimental.host_callback.id_print` in presence of JVP
      to print a pair of primal and tangent. Previously, there were two separate
      print operations for the primals and the tangent.
    * `host_callback.outfeed_receiver` has been removed (it is not necessary,
      and was deprecated a few months ago).
* New features:
  * New flag for debugging `inf`, analogous to that for `NaN` ({jax-issue}`#5224`).

## jax 0.2.7 (Dec 4 2020)

* [GitHub commits](https://github.com/google/jax/compare/jax-v0.2.6...jax-v0.2.7).
* New features:
  * Add `jax.device_put_replicated`
  * Add multi-host support to `jax.experimental.sharded_jit`
  * Add support for differentiating eigenvalues computed by `jax.numpy.linalg.eig`
  * Add support for building on Windows platforms
  * Add support for general in_axes and out_axes in `jax.pmap`
  * Add complex support for `jax.numpy.linalg.slogdet`
* Bug fixes:
  * Fix higher-than-second order derivatives of `jax.numpy.sinc` at zero
  * Fix some hard-to-hit bugs around symbolic zeros in transpose rules
* Breaking changes:
  * `jax.experimental.optix` has been deleted, in favor of the standalone
    `optax` Python package.
  * indexing of JAX arrays with non-tuple sequences now raises a `TypeError`. This type of indexing
    has been deprecated in Numpy since v1.16, and in JAX since v0.2.4.
    See {jax-issue}`#4564`.

## jax 0.2.6 (Nov 18 2020)

* [GitHub commits](https://github.com/google/jax/compare/jax-v0.2.5...jax-v0.2.6).
* New Features:
  * Add support for shape-polymorphic tracing for the jax.experimental.jax2tf converter.
    See [README.md](https://github.com/google/jax/blob/main/jax/experimental/jax2tf/README.md).
* Breaking change cleanup

  * Raise an error on non-hashable static arguments for jax.jit and
    xla_computation.  See [cb48f42](https://github.com/google/jax/commit/cb48f42).
  * Improve consistency of type promotion behavior ({jax-issue}`#4744`):
    * Adding a complex Python scalar to a JAX floating point number respects the precision of
      the JAX float. For example, `jnp.float32(1) + 1j` now returns `complex64`, where previously
      it returned `complex128`.
    * Results of type promotion with 3 or more terms involving uint64, a signed int, and a third type
      are now independent of the order of arguments. For example:
      `jnp.result_type(jnp.uint64, jnp.int64, jnp.float16)` and
      `jnp.result_type(jnp.float16, jnp.uint64, jnp.int64)` both return `float16`, where previously
      the first returned `float64` and the second returned `float16`.
  * The contents of the (undocumented) `jax.lax_linalg` linear algebra module
    are now exposed publicly as `jax.lax.linalg`.
  * `jax.random.PRNGKey` now produces the same results in and out of JIT compilation
    ({jax-issue}`#4877`).
    This required changing the result for a given seed in a few particular cases:
    * With `jax_enable_x64=False`, negative seeds passed as Python integers now return a different result
      outside JIT mode. For example, `jax.random.PRNGKey(-1)` previously returned
      `[4294967295, 4294967295]`, and now returns `[0, 4294967295]`. This matches the behavior in JIT.
    * Seeds outside the range representable by `int64` outside JIT now result in an `OverflowError`
      rather than a `TypeError`. This matches the behavior in JIT.

    To recover the keys returned previously for negative integers with `jax_enable_x64=False`
    outside JIT, you can use:

    ```
    key = random.PRNGKey(-1).at[0].set(0xFFFFFFFF)
    ```
  * DeviceArray now raises `RuntimeError` instead of `ValueError` when trying
    to access its value while it has been deleted.

## jaxlib 0.1.58 (January 12ish 2021)

* Fixed a bug that meant JAX sometimes return platform-specific types (e.g.,
  `np.cint`) instead of standard types (e.g., `np.int32`). (#4903)
* Fixed a crash when constant-folding certain int16 operations. (#4971)
* Added an `is_leaf` predicate to {func}`pytree.flatten`.

## jaxlib 0.1.57 (November 12 2020)

* Fixed manylinux2010 compliance issues in GPU wheels.
* Switched the CPU FFT implementation from Eigen to PocketFFT.
* Fixed a bug where the hash of bfloat16 values was not correctly initialized
  and could change (#4651).
* Add support for retaining ownership when passing arrays to DLPack (#4636).
* Fixed a bug for batched triangular solves with sizes greater than 128 but not
  a multiple of 128.
* Fixed a bug when performing concurrent FFTs on multiple GPUs (#3518).
* Fixed a bug in profiler where tools are missing (#4427).
* Dropped support for CUDA 10.0.

## jax 0.2.5 (October 27 2020)

* [GitHub commits](https://github.com/google/jax/compare/jax-v0.2.4...jax-v0.2.5).
* Improvements:
  * Ensure that `check_jaxpr` does not perform FLOPS.  See {jax-issue}`#4650`.
  * Expanded the set of JAX primitives converted by jax2tf.
    See [primitives_with_limited_support.md](https://github.com/google/jax/blob/main/jax/experimental/jax2tf/primitives_with_limited_support.md).

## jax 0.2.4 (October 19 2020)

* [GitHub commits](https://github.com/google/jax/compare/jax-v0.2.3...jax-v0.2.4).
* Improvements:
  * Add support for `remat` to jax.experimental.host_callback.  See {jax-issue}`#4608`.
* Deprecations

  * Indexing with non-tuple sequences is now deprecated, following a similar deprecation in Numpy.
    In a future release, this will result in a TypeError. See {jax-issue}`#4564`.

## jaxlib 0.1.56 (October 14, 2020)

## jax 0.2.3 (October 14 2020)

* [GitHub commits](https://github.com/google/jax/compare/jax-v0.2.2...jax-v0.2.3).
* The reason for another release so soon is we need to temporarily roll back a
  new jit fastpath while we look into a performance degradation

## jax 0.2.2 (October 13 2020)

* [GitHub commits](https://github.com/google/jax/compare/jax-v0.2.1...jax-v0.2.2).

## jax 0.2.1 (October 6 2020)

* [GitHub commits](https://github.com/google/jax/compare/jax-v0.2.0...jax-v0.2.1).
* Improvements:
  * As a benefit of omnistaging, the host_callback functions are executed (in program
    order) even if the result of the {py:func}`jax.experimental.host_callback.id_print`/
    {py:func}`jax.experimental.host_callback.id_tap` is not used in the computation.

## jax (0.2.0) (September 23 2020)

* [GitHub commits](https://github.com/google/jax/compare/jax-v0.1.77...jax-v0.2.0).
* Improvements:
  * Omnistaging on by default. See {jax-issue}`#3370` and
    [omnistaging](https://github.com/google/jax/blob/main/docs/design_notes/omnistaging.md)

## jax (0.1.77) (September 15 2020)

* Breaking changes:
  * New simplified interface for {py:func}`jax.experimental.host_callback.id_tap` (#4101)

## jaxlib 0.1.55 (September 8, 2020)

* Update XLA:
  * Fix bug in DLPackManagedTensorToBuffer (#4196)

## jax 0.1.76 (September 8, 2020)

* [GitHub commits](https://github.com/google/jax/compare/jax-v0.1.75...jax-v0.1.76).

## jax 0.1.75 (July 30, 2020)

* [GitHub commits](https://github.com/google/jax/compare/jax-v0.1.74...jax-v0.1.75).
* Bug Fixes:
  * make jnp.abs() work for unsigned inputs (#3914)
* Improvements:
  * "Omnistaging" behavior added behind a flag, disabled by default (#3370)

## jax 0.1.74 (July 29, 2020)

* [GitHub commits](https://github.com/google/jax/compare/jax-v0.1.73...jax-v0.1.74).
* New Features:
  * BFGS (#3101)
  * TPU support for half-precision arithmetic (#3878)
* Bug Fixes:
  * Prevent some accidental dtype warnings (#3874)
  * Fix a multi-threading bug in custom derivatives (#3845, #3869)
* Improvements:
  * Faster searchsorted implementation (#3873)
  * Better test coverage for jax.numpy sorting algorithms (#3836)

## jaxlib 0.1.52 (July 22, 2020)

* Update XLA.

## jax 0.1.73 (July 22, 2020)

* [GitHub commits](https://github.com/google/jax/compare/jax-v0.1.72...jax-v0.1.73).
* The minimum jaxlib version is now 0.1.51.
* New Features:
  * jax.image.resize. (#3703)
  * hfft and ihfft (#3664)
  * jax.numpy.intersect1d (#3726)
  * jax.numpy.lexsort (#3812)
  * `lax.scan` and the `scan` primitive support an `unroll`
    parameter for loop unrolling when lowering to XLA
    ({jax-issue}`#3738`).
* Bug Fixes:
  * Fix reduction repeated axis error (#3618)
  * Fix shape rule for lax.pad for input dimensions of size 0. (#3608)
  * make psum transpose handle zero cotangents (#3653)
  * Fix shape error when taking JVP of reduce-prod over size 0 axis. (#3729)
  * Support differentiation through jax.lax.all_to_all (#3733)
  * address nan issue in jax.scipy.special.zeta (#3777)
* Improvements:
  * Many improvements to jax2tf
  * Reimplement argmin/argmax using a single pass variadic reduction. (#3611)
  * Enable XLA SPMD partitioning by default. (#3151)
  * Add support for 0d transpose convolution (#3643)
  * Make LU gradient work for low-rank matrices (#3610)
  * support multiple_results and custom JVPs in jet (#3657)
  * Generalize reduce-window padding to support (lo, hi) pairs. (#3728)
  * Implement complex convolutions on CPU and GPU. (#3735)
  * Make jnp.take work for empty slices of empty arrays. (#3751)
  * Relax dimension ordering rules for dot_general. (#3778)
  * Enable buffer donation for GPU. (#3800)
  * Add support for base dilation and window dilation to reduce window op… (#3803)

## jaxlib 0.1.51 (July 2, 2020)

* Update XLA.
* Add new runtime support for host_callback.

## jax 0.1.72 (June 28, 2020)

* [GitHub commits](https://github.com/google/jax/compare/jax-v0.1.71...jax-v0.1.72).
* Bug fixes:
  * Fix an odeint bug introduced in the previous release, see
    {jax-issue}`#3587`.

## jax 0.1.71 (June 25, 2020)

* [GitHub commits](https://github.com/google/jax/compare/jax-v0.1.70...jax-v0.1.71).
* The minimum jaxlib version is now 0.1.48.
* Bug fixes:
  * Allow `jax.experimental.ode.odeint` dynamics functions to close over
    values with respect to which we're differentiating
    {jax-issue}`#3562`.

## jaxlib 0.1.50 (June 25, 2020)

* Add support for CUDA 11.0.
* Drop support for CUDA 9.2 (we only maintain support for the last four CUDA
  versions.)
* Update XLA.

## jaxlib 0.1.49 (June 19, 2020)

* Bug fixes:
  * Fix build issue that could result in slow compiles
    (<https://github.com/tensorflow/tensorflow/commit/f805153a25b00d12072bd728e91bb1621bfcf1b1>)

## jaxlib 0.1.48 (June 12, 2020)

* New features:
  * Adds support for fast traceback collection.
  * Adds preliminary support for on-device heap profiling.
  * Implements `np.nextafter` for `bfloat16` types.
  * Complex128 support for FFTs on CPU and GPU.
* Bug fixes:
  * Improved float64 `tanh` accuracy on GPU.
  * float64 scatters on GPU are much faster.
  * Complex matrix multiplication on CPU should be much faster.
  * Stable sorts on CPU should actually be stable now.
  * Concurrency bug fix in CPU backend.

## jax 0.1.70 (June 8, 2020)

* [GitHub commits](https://github.com/google/jax/compare/jax-v0.1.69...jax-v0.1.70).
* New features:
  * `lax.switch` introduces indexed conditionals with multiple
    branches, together with a generalization of the `cond`
    primitive
    {jax-issue}`#3318`.

## jax 0.1.69 (June 3, 2020)

* [GitHub commits](https://github.com/google/jax/compare/jax-v0.1.68...jax-v0.1.69).

## jax 0.1.68 (May 21, 2020)

* [GitHub commits](https://github.com/google/jax/compare/jax-v0.1.67...jax-v0.1.68).
* New features:
  * {func}`lax.cond` supports a single-operand form, taken as the argument
    to both branches
    {jax-issue}`#2993`.
* Notable changes:
  * The format of the `transforms` keyword for the {func}`jax.experimental.host_callback.id_tap`
    primitive has changed {jax-issue}`#3132`.

## jax 0.1.67 (May 12, 2020)

* [GitHub commits](https://github.com/google/jax/compare/jax-v0.1.66...jax-v0.1.67).
* New features:
  * Support for reduction over subsets of a pmapped axis using `axis_index_groups`
    {jax-issue}`#2382`.
  * Experimental support for printing and calling host-side Python function from
    compiled code. See [id_print and id_tap](https://jax.readthedocs.io/en/latest/jax.experimental.host_callback.html)
    ({jax-issue}`#3006`).
* Notable changes:
  * The visibility of names exported from {mod}`jax.numpy` has been
    tightened. This may break code that was making use of names that were
    previously exported accidentally.

## jaxlib 0.1.47 (May 8, 2020)

* Fixes crash for outfeed.

## jax 0.1.66 (May 5, 2020)

* [GitHub commits](https://github.com/google/jax/compare/jax-v0.1.65...jax-v0.1.66).
* New features:
  * Support for `in_axes=None` on {func}`pmap`
    {jax-issue}`#2896`.

## jaxlib 0.1.46 (May 5, 2020)

* Fixes crash for linear algebra functions on Mac OS X (#432).
* Fixes an illegal instruction crash caused by using AVX512 instructions when
  an operating system or hypervisor disabled them (#2906).

## jax 0.1.65 (April 30, 2020)

* [GitHub commits](https://github.com/google/jax/compare/jax-v0.1.64...jax-v0.1.65).
* New features:
  * Differentiation of determinants of singular matrices
    {jax-issue}`#2809`.
* Bug fixes:
  * Fix {func}`odeint` differentiation with respect to time of ODEs with
    time-dependent dynamics {jax-issue}`#2817`,
    also add ODE CI testing.
  * Fix {func}`lax_linalg.qr` differentiation
    {jax-issue}`#2867`.

## jaxlib 0.1.45 (April 21, 2020)

* Fixes segfault: {jax-issue}`#2755`
* Plumb is_stable option on Sort HLO through to Python.

## jax 0.1.64 (April 21, 2020)

* [GitHub commits](https://github.com/google/jax/compare/jax-v0.1.63...jax-v0.1.64).
* New features:
  * Add syntactic sugar for functional indexed updates
    {jax-issue}`#2684`.
  * Add {func}`jax.numpy.linalg.multi_dot` {jax-issue}`#2726`.
  * Add {func}`jax.numpy.unique` {jax-issue}`#2760`.
  * Add {func}`jax.numpy.rint` {jax-issue}`#2724`.
  * Add {func}`jax.numpy.rint` {jax-issue}`#2724`.
  * Add more primitive rules for {func}`jax.experimental.jet`.
* Bug fixes:
  * Fix {func}`logaddexp` and {func}`logaddexp2` differentiation at zero {jax-issue}`#2107`.
  * Improve memory usage in reverse-mode autodiff without {func}`jit`
    {jax-issue}`#2719`.
* Better errors:
  * Improves error message for reverse-mode differentiation of {func}`lax.while_loop`
    {jax-issue}`#2129`.

## jaxlib 0.1.44 (April 16, 2020)

* Fixes a bug where if multiple GPUs of different models were present, JAX
  would only compile programs suitable for the first GPU.
* Bugfix for `batch_group_count` convolutions.
* Added precompiled SASS for more GPU versions to avoid startup PTX compilation
  hang.

## jax 0.1.63 (April 12, 2020)

* [GitHub commits](https://github.com/google/jax/compare/jax-v0.1.62...jax-v0.1.63).
* Added `jax.custom_jvp` and `jax.custom_vjp` from {jax-issue}`#2026`, see the [tutorial notebook](https://jax.readthedocs.io/en/latest/notebooks/Custom_derivative_rules_for_Python_code.html). Deprecated `jax.custom_transforms` and removed it from the docs (though it still works).
* Add `scipy.sparse.linalg.cg` {jax-issue}`#2566`.
* Changed how Tracers are printed to show more useful information for debugging {jax-issue}`#2591`.
* Made `jax.numpy.isclose` handle `nan` and `inf` correctly {jax-issue}`#2501`.
* Added several new rules for `jax.experimental.jet` {jax-issue}`#2537`.
* Fixed `jax.experimental.stax.BatchNorm` when `scale`/`center` isn't provided.
* Fix some missing cases of broadcasting in `jax.numpy.einsum` {jax-issue}`#2512`.
* Implement `jax.numpy.cumsum` and `jax.numpy.cumprod` in terms of a parallel prefix scan {jax-issue}`#2596` and make `reduce_prod` differentiable to arbitray order {jax-issue}`#2597`.
* Add `batch_group_count` to `conv_general_dilated` {jax-issue}`#2635`.
* Add docstring for `test_util.check_grads` {jax-issue}`#2656`.
* Add `callback_transform` {jax-issue}`#2665`.
* Implement `rollaxis`, `convolve`/`correlate` 1d & 2d, `copysign`,
  `trunc`, `roots`, and `quantile`/`percentile` interpolation options.

## jaxlib 0.1.43 (March 31, 2020)

* Fixed a performance regression for Resnet-50 on GPU.

## jax 0.1.62 (March 21, 2020)

* [GitHub commits](https://github.com/google/jax/compare/jax-v0.1.61...jax-v0.1.62).
* JAX has dropped support for Python 3.5. Please upgrade to Python 3.6 or newer.
* Removed the internal function `lax._safe_mul`, which implemented the
  convention `0. * nan == 0.`. This change means some programs when
  differentiated will produce nans when they previously produced correct
  values, though it ensures nans rather than silently incorrect results are
  produced for other programs. See #2447 and #1052 for details.
* Added an `all_gather` parallel convenience function.
* More type annotations in core code.

## jaxlib 0.1.42 (March 19, 2020)

* jaxlib 0.1.41 broke cloud TPU support due to an API incompatibility. This
  release fixes it again.
* JAX has dropped support for Python 3.5. Please upgrade to Python 3.6 or newer.

## jax 0.1.61 (March 17, 2020)

* [GitHub commits](https://github.com/google/jax/compare/jax-v0.1.60...jax-v0.1.61).
* Fixes Python 3.5 support. This will be the last JAX or jaxlib release that
  supports Python 3.5.

## jax 0.1.60 (March 17, 2020)

* [GitHub commits](https://github.com/google/jax/compare/jax-v0.1.59...jax-v0.1.60).
* New features:
  * {py:func}`jax.pmap` has `static_broadcast_argnums` argument which allows
    the user to specify arguments that should be treated as compile-time
    constants and should be broadcasted to all devices. It works analogously to
    `static_argnums` in {py:func}`jax.jit`.
  * Improved error messages for when tracers are mistakenly saved in global state.
  * Added {py:func}`jax.nn.one_hot` utility function.
  * Added {mod}`jax.experimental.jet` for exponentially faster
    higher-order automatic differentiation.
  * Added more correctness checking to arguments of {py:func}`jax.lax.broadcast_in_dim`.
* The minimum jaxlib version is now 0.1.41.

## jaxlib 0.1.40 (March 4, 2020)

* Adds experimental support in Jaxlib for TensorFlow profiler, which allows
  tracing of CPU and GPU computations from TensorBoard.
* Includes prototype support for multihost GPU computations that communicate via
  NCCL.
* Improves performance of NCCL collectives on GPU.
* Adds TopK, CustomCallWithoutLayout, CustomCallWithLayout, IGammaGradA and
  RandomGamma implementations.
* Supports device assignments known at XLA compilation time.

## jax 0.1.59 (February 11, 2020)

* [GitHub commits](https://github.com/google/jax/compare/jax-v0.1.58...jax-v0.1.59).
* Breaking changes

  * The minimum jaxlib version is now 0.1.38.
  * Simplified {py:class}`Jaxpr` by removing the `Jaxpr.freevars` and
    `Jaxpr.bound_subjaxprs`. The call primitives (`xla_call`, `xla_pmap`,
    `sharded_call`, and `remat_call`) get a new parameter `call_jaxpr` with a
    fully-closed (no `constvars`) jaxpr. Also, added a new field `call_primitive`
    to primitives.
* New features:
  * Reverse-mode automatic differentiation (e.g. `grad`) of `lax.cond`, making it
    now differentiable in both modes ({jax-issue}`#2091`)
  * JAX now supports DLPack, which allows sharing CPU and GPU arrays in a
    zero-copy way with other libraries, such as PyTorch.
  * JAX GPU DeviceArrays now support `__cuda_array_interface__`, which is another
    zero-copy protocol for sharing GPU arrays with other libraries such as CuPy
    and Numba.
  * JAX CPU device buffers now implement the Python buffer protocol, which allows
    zero-copy buffer sharing between JAX and NumPy.
  * Added JAX_SKIP_SLOW_TESTS environment variable to skip tests known as slow.

## jaxlib 0.1.39 (February 11, 2020)

* Updates XLA.

## jaxlib 0.1.38 (January 29, 2020)

* CUDA 9.0 is no longer supported.
* CUDA 10.2 wheels are now built by default.

## jax 0.1.58 (January 28, 2020)

* [GitHub commits](https://github.com/google/jax/compare/46014da21...jax-v0.1.58).
* Breaking changes

  * JAX has dropped Python 2 support, because Python 2 reached its end of life on
    January 1, 2020. Please update to Python 3.5 or newer.
* New features

  >   > * Forward-mode automatic differentiation (`jvp`) of while loop
  >   ({jax-issue}`#1980`)
  > * New NumPy and SciPy functions:
  >
  >   * {py:func}`jax.numpy.fft.fft2`
  >   * {py:func}`jax.numpy.fft.ifft2`
  >   * {py:func}`jax.numpy.fft.rfft`
  >   * {py:func}`jax.numpy.fft.irfft`
  >   * {py:func}`jax.numpy.fft.rfft2`
  >   * {py:func}`jax.numpy.fft.irfft2`
  >   * {py:func}`jax.numpy.fft.rfftn`
  >   * {py:func}`jax.numpy.fft.irfftn`
  >   * {py:func}`jax.numpy.fft.fftfreq`
  >   * {py:func}`jax.numpy.fft.rfftfreq`
  >   * {py:func}`jax.numpy.linalg.matrix_rank`
  >   * {py:func}`jax.numpy.linalg.matrix_power`
  >   * {py:func}`jax.scipy.special.betainc`
  > * Batched Cholesky decomposition on GPU now uses a more efficient batched
  >   kernel.

### Notable bug fixes

* With the Python 3 upgrade, JAX no longer depends on `fastcache`, which should
  help with installation.

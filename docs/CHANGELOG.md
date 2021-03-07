# Change Log

<!--
Remember to align the itemized text with the first line of an item within a list.

PLEASE REMEMBER TO CHANGE THE '..master' WITH AN ACTUAL TAG in GITHUB LINK.
-->

## jax 0.2.11 (unreleased)

* [GitHub commits](https://github.com/google/jax/compare/jax-v0.2.10...master).
* New features:
* Bug fixes:
* Breaking changes:

## jax 0.2.10 (March 5 2021)

* [GitHub commits](https://github.com/google/jax/compare/jax-v0.2.9...jax-v0.2.10).
* New features:
  * {func}`jax.scipy.stats.chi2` is now available as a distribution with logpdf and pdf methods.
  * {func}`jax.scipy.stats.betabinom` is now available as a distribution with logpmf and pmf methods.
  * Added {func}`jax.experimental.jax2tf.call_tf` to call TensorFlow functions
    from JAX ({jax-issue}`#5627`)
    and [README](https://github.com/google/jax/blob/master/jax/experimental/jax2tf/README.md#calling-tensorflow-functions-from-jax)).
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
  * Reversed the order of {func}`jax.lax.linalg.eigh` and underlying primitives to be
    consistent with documentation and `numpy` and `scipy` wrappers.

## jaxlib 0.1.62 (Unreleased)

* New features:
  * jaxlib wheels are now built to require AVX instructions on x86-64 machines
    by default. If you want to use JAX on a machine that doesn't support AVX,
    you can build a jaxlib from source using the `--target_cpu_features` flag
    to `build.py`. `--target_cpu_features` also replaces
    `--enable_march_native`.

## jaxlib 0.1.61 (February 12 2021)

## jaxlib 0.1.60 (Febuary 3 2021)

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
  * `host_callback.id_tap` now works for `jax.pmap` also. There is a
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
  * New flag for debugging `inf`, analagous to that for `NaN` ({jax-issue}`#5224`).

## jax 0.2.7 (Dec 4 2020)

* [GitHub commits](https://github.com/google/jax/compare/jax-v0.2.6...jax-v0.2.7).
* New features:
  * Add `jax.device_put_replicated`
  * Add multi-host support to `jax.experimental.sharded_jit`
  * Add support for differentiating eigenvaleus computed by `jax.numpy.linalg.eig`
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
    See [README.md](https://github.com/google/jax/blob/master/jax/experimental/jax2tf/README.md).
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
    See [primitives_with_limited_support.md](https://github.com/google/jax/blob/master/jax/experimental/jax2tf/primitives_with_limited_support.md).

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
  * Omnistaging on by default. See {jax-issue}`#3370`.

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
  * TPU suppot for half-precision arithmetic (#3878)
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
* Bugfixes:
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



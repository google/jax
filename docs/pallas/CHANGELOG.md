(pallas-changelog)=

# Pallas Changelog

<!--* freshness: { reviewed: '2024-07-11' } *-->

This is the list of changes specific to {class}`jax.experimental.pallas`.
For the overall JAX change log see [here](https://jax.readthedocs.io/en/latest/changelog.html).

<!--
Remember to align the itemized text with the first line of an item within a list.
-->

## Released with jax 0.4.32

* Changes
  * The kernel function is not allowed to close over constants. Instead, all the needed arrays
    must be passed as inputs, with proper block specs ({jax-issue}`#22746`).

* Deprecations

* New functionality:
  * Improved error messages for mistakes in the signature of the index map functions,
    to include the name and source location of the index map.

##  Released with jax 0.4.31 (July 29, 2024)

* Changes
  * {class}`jax.experimental.pallas.BlockSpec` now expects `block_shape` to
    be passed *before* `index_map`. The old argument order is deprecated and
    will be removed in a future release.
  * {class}`jax.experimental.pallas.GridSpec` does not have anymore the `in_specs_tree`,
    and the `out_specs_tree` fields, and the `in_specs` and `out_specs` tree now
    store the values as pytrees of BlockSpec. Previously, `in_specs` and
    `out_specs` were flattened ({jax-issue}`#22552`).
  * The method `compute_index` of {class}`jax.experimental.pallas.GridSpec` has
    been removed because it is private. Similarly, the `get_grid_mapping` and
    `unzip_dynamic_bounds` have been removed from `BlockSpec` ({jax-issue}`#22593`).
  * Fixed the interpreter mode to work with BlockSpec that involve padding
    ({jax-issue}`#22275`).
    Padding in interpreter mode will be with NaN, to help debug out-of-bounds
    errors, but this behavior is not present when running in custom kernel mode,
    and should not be depended on.
  * Previously it was possible to import many APIs that are meant to be
    private, as `jax.experimental.pallas.pallas`. This is not possible anymore.


* Deprecations


* New Functionality
  * Added documentation for BlockSpec: {ref}`pallas_grids_and_blockspecs`.
  * Improved error messages for the {func}`jax.experimental.pallas.pallas_call`
    API.
  * Added lowering rules for TPU for `lax.shift_right_arithmetic` ({jax-issue}`#22279`)
    and `lax.erf_inv` ({jax-issue}`#22310`).
  * Added initial support for shape polymorphism for the Pallas TPU custom kernels\
    ({jax-issue}`#22084`).
  * Added TPU support for checkify. ({jax-issue}`#22480`)
  * Added clearer error messages when the block sizes do not match the TPU
    requirements. Previously, the errors were coming from the Mosaic backend
    and did not have useful Python stack traces.
  * Added support for TPU lowering with 1D blocks, and relaxed the requirements
    for the block sizes with at least 2 dimensions: the last 2 dimensions must
    be divisible by 8 and 128 respectively, unless they span the entire
    corresponding array dimension. Previously, block dimensions that spanned the
    entire array were allowed only if the block dimensions in the last two
    dimensions were smaller than 8 and 128 respectively.

## Released with JAX 0.4.30 (June 18, 2024)

* New Functionality
  * Added checkify support for {func}`jax.experimental.pallas.pallas_call` in
    interpret mode ({jax-issue}`#21862`).
  * Improved support for PRNG keys for TPU kernels ({jax-issue}`#21773`).





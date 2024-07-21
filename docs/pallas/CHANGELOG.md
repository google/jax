(pallas-changelog)=

# Pallas Changelog

<!--* freshness: { reviewed: '2024-07-11' } *-->

This is the list of changes specific to {class}`jax.experimental.pallas`.
For the overall JAX change log see [here](https://jax.readthedocs.io/en/latest/changelog.html).

<!--
Remember to align the itemized text with the first line of an item within a list.
-->

## Released with JAX 0.4.31

* Changes
  * {class}`jax.experimental.pallas.BlockSpec` now expects `block_shape` to
    be passed *before* `index_map`. The old argument order is deprecated and
    will be removed in a future release.
  * Fixed the interpreter mode to work with BlockSpec that involve padding
    ({jax-issue}`#22275`).
    Padding in interpreter mode will be with NaN, to help debug out-of-bounds
    errors, but this behavior is not present when running in custom kernel mode,
    and should not be depended on.


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

## Released with JAX 0.4.30 (June 18, 2024)

* New Functionality
  * Added checkify support for {func}`jax.experimental.pallas.pallas_call` in
    interpret mode ({jax-issue}`#21862`).
  * Improved support for PRNG keys for TPU kernels ({jax-issue}`#21773`).





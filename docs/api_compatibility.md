(api-compatibility)=

# API compatibility

<!--* freshness: { reviewed: '2023-07-18' } *-->

JAX is constantly evolving, and we want to be able to make improvements to its
APIs. That said, we want to minimize churn for the JAX user community, and we
try to make breaking changes rarely.

JAX follows a 3 month deprecation policy. When an incompatible change is made
to an API, we will make our best effort to obey the following procedure:
* the change will be announced in `CHANGELOG.md` and in the doc string for the
  deprecated API, and the old API will issue a `DeprecationWarning`.
* three months after the `jax` release that deprecated an API, we may remove the
  deprecated API at any time. Note that three months is a *lower* bound, and is
  intentionally chosen to be faster than that of many more mature projects. In
  practice, deprecations may take considerably longer, particularly if there are
  many users of a feature. If a three month deprecation period becomes
  problematic, please raise this with us.

We reserve the right to change this policy at any time.

## What is covered?

Only public JAX APIs are covered, which includes the following modules:

* `jax`
* `jax.dlpack`
* `jax.image`
* `jax.lax`
* `jax.nn`
* `jax.numpy`
* `jax.ops`
* `jax.profiler`
* `jax.random` (see [details below](#numerics-and-randomness))
* `jax.scipy`
* `jax.tree`
* `jax.tree_util`
* `jax.test_util`

Not everything in these modules is intended to be public, and over time, we
are working to separate public and private APIs. Public APIs are documented
in the JAX documentation.
Additionally, our goal is that all non-public APIs should have names
prefixed with underscores, although we do not entirely comply with this yet.

## What is not covered?

*  anything prefixed with an underscore.
* `jax._src`
* `jax.core`
* `jax.lib`
* `jax.interpreters`
* `jax.experimental`
* `jax.example_libraries`
* `jax.extend` (see [details](https://jax.readthedocs.io/en/latest/jax.extend.html))

This list is not exhaustive.

## Numerics and randomness

The *exact* values of numerical operations are not guaranteed to be
stable across JAX releases. In fact, exact numerics are not
necessarily stable at a given JAX version, across accelerator
platforms, within or without `jax.jit`, and more.

For a fixed PRNG key input, the outputs of pseudorandom functions in
`jax.random` may vary across JAX versions. The compatibility policy
applies only to the output *distribution*. For example, the expression
`jax.random.gumbel(jax.random.key(72))` may return a different value
across JAX releases, but `jax.random.gumbel` will remain a
pseudorandom generator for the Gumbel distribution.

We try to make such changes to pseudorandom values infrequently. When
they happen, the changes are announced in the changelog, but do not
follow a deprecation cycle. In some situations, JAX might expose a
transient configuration flag that reverts the new behavior, to help
users diagnose and update affected code. Such flags will last a
deprecation window's amount of time.

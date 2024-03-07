# JEP 18137: Scope of JAX NumPy & SciPy Wrappers

*Jake VanderPlas*

*October 2023*

Until now, the intended scope of {mod}`jax.numpy` and {mod}`jax.scipy` has been relatively
ill-defined. This document proposes a well-defined scope for these packages to better guide
and evaluate future contributions, and to motivate the removal of some out-of-scope code.

## Background
From the beginning, JAX has aimed to provide a NumPy-like API for executing code in XLA,
and a big part of the project’s development has been building out the {mod}`jax.numpy` and
{mod}`jax.scipy` namespaces as JAX-based implementations of NumPy and SciPy APIs. There has always
been an implicit understanding that some parts of {mod}`numpy` and {mod}`scipy` are out-of-scope
for JAX, but this scope has not been well defined. This can lead to confusion and frustration
for contributors, because there’s no clear answer to whether potential {mod}`jax.numpy` and
{mod}`jax.scipy` contributions will be accepted into JAX.

## Why Limit the Scope?
To avoid leaving this unsaid, we should be explicit: it is a fact that any code included
in a project like JAX incurs a small but nonzero ongoing maintenance burden for the developers.
The success of a project over time directly relates to the ability of maintainers to continue
this maintenance for the sum of all the project’s parts: documenting functionality, responding
to questions, fixing bugs, etc. For long-term success and sustainability of any software tool,
it’s vital that maintainers carefully weigh whether any particular contribution will be a net
positive for the project given its goals and resources.

## Evaluation Rubric
This document proposes a rubric of six axes along which the scope of any particular {mod}`numpy`
or {mod}`scipy` API can be judged for inclusion into JAX. An API which is strong along all axes
is an excellent candidate for inclusion in the JAX package; a strong weakness along *any* of
the six axes is a good argument against inclusion in JAX.

### Axis 1: XLA alignment
The first axis we consider is the degree to which the proposed API aligns with native XLA
operations. For example, {func}`jax.numpy.exp` is a function that more-or-less directly mirrors
`jax.lax.exp`. A large number of functions in  {mod}`numpy`, {mod}`scipy.special`,  {mod}`numpy.linalg`,
`scipy.linalg`, and others meet this criteria: such functions pass the XLA-alignment check
when considering their inclusion into JAX.

On the other end, there are functions like  {func}`numpy.unique`, which do not directly correspond
to any XLA operation, and in some cases are fundamentally incompatible with JAX’s current
computational model, which requires statically-shaped arrays (e.g. `unique` returns a
value-dependent dynamic array shape). Such functions do not pass the XLA alignment check
when considering their inclusion into JAX.

We also consider as part of this axis the need for pure function semantics. For example,
 {mod}`numpy.random` is built on an implicitly-updated state-based RNG, which is fundamentally
incompatible with JAX’s computational model built on XLA.

### Axis 2: Array API Alignment
The second axis we consider focuses on the
[Python Array API Standard](https://data-apis.org/array-api/2022.12/): this is in some
senses a community-driven outline of which array operations are central to array-oriented
programming across a wide range of user communities. If an API in  {mod}`numpy` or {mod}`scipy` is
listed within the Array API standard, it is a strong signal that JAX should include it.
Using the example from above, the Array API standard includes several variants of
 {func}`numpy.unique` (`unique_all`, `unique_counts`, `unique_inverse`, `unique_values`) which
suggests that, despite the function not being precisely aligned with XLA, it is important
enough to the Python user community that JAX should perhaps implement it.

### Axis 3: Existence of Downstream Implementations
For functionality that does not align with Axis 1 or 2, an important consideration for
inclusion into JAX is whether there exist well-supported downstream packages that supply
the functionality in question. A good example of this is {mod}`scipy.optimize`: while JAX does
include a minimal set of wrappers of {mod}`scipy.optimize` functionality, a much more complete
treatment exists in the [JAXopt](https://jaxopt.github.io/) package, which is actively
maintained by JAX collaborators. In cases like this, we should lean toward pointing users
and contributors to these specialized packages rather than re-implementing such APIs in
JAX itself.

### Axis 4: Complexity & Robustness of Implementation
For functionality that does not align with XLA, one consideration is the degree of
complexity of the proposed implementation. This aligns to some degree with Axis 1,
but nevertheless is important to call out. A number of functions have been contributed
to JAX which have relatively complex implementations which are difficult to validate
and introduce outsized maintenance burdens; an example is {func}`jax.scipy.special.bessel_jn`:
as of the writing of this JEP, its current implementation is a non-straightforward
iterative approximation that has
[convergence issues in some domains](https://github.com/google/jax/issues/12402#issuecomment-1384828637),
and [proposed fixes](https://github.com/google/jax/pull/17038/files) introduce further
complexity. Had we more carefully weighed the complexity and robustness of the
implementation when accepting the contribution, we may have chosen not to accept this
contribution to the package.

### Axis 5: Functional vs. Object-Oriented APIs
JAX works best with functional APIs rather than object-oriented APIs. Object-oriented
APIs can often hide impure semantics, making them often difficult to implement well.
NumPy and SciPy generally stick to functional APIs, but sometimes provide object-oriented
convenience wrappers.

Examples of this are  {class}`numpy.polynomial.Polynomial`, which wraps lower-level operations
like  {func}`numpy.polyadd`,  {func}`numpy.polydiv`, etc. In general, when there are both functional
and object-oriented APIs available, JAX should avoid providing wrappers for the
object-oriented APIs and instead provide wrappers for the functional APIs.

In cases where only the object-oriented APIs exist, JAX should avoid providing wrappers
unless the case is strong along other axes.

### Axis 6: General “Importance” to JAX Users & Stakeholders
The decision to include a NumPy/SciPy API in JAX should also take into account the
importance of the algorithm to the general user community. It is admittedly difficult
to quantify who is a “stakeholder” and how this importance should be measured; but we
include this to make clear that any decision about what to include in JAX’s NumPy and
SciPy wrappers will involve some amount of discretion that cannot be easily quantified.

For existing APIs, searches for usage in github may be useful in establishing importance
or lack thereof; as an example, we might return to {func}`jax.scipy.special.bessel_jn`
discussed above: a search shows that this function has only a 
[handful of uses](https://github.com/search?q=jax+AND+%22bessel_jn%28%22+NOT+MathJax&type=code)
on github, probably partly to do with the previously mentioned accuracy issues.

## Evaluation: what’s in scope?
In this section, we’ll attempt to evaluate the NumPy and SciPy APIs, including some
examples from the current JAX API, in light of the above rubric. This will not be a
comprehensive listing of all existing functions and classes, but rather a more general
discussion by submodule and topic, with relevant examples.

### NumPy APIs

#### ✅ `numpy` namespace 
We consider the functions in the main  {mod}`numpy` namespace to be essentially all in-scope
for JAX, due to its general alignment with XLA (Axis 1) and the Python Array API
(Axis 2), as well as its general importance to the JAX user community (Axis 6).
Some functions are perhaps borderline (functions like {func}`numpy.intersect1d`,
{func}`np.setdiff1d`, {func}`np.union1d` arguably fail parts of the rubric) but for
simplicity we declare that all array functions in the main numpy namespace are in-scope
for JAX.

#### ✅ `numpy.linalg` & `numpy.fft`
The {mod}`numpy.linalg` and {mod}`numpy.fft` submodules contain many functions that
broadly align with functionality provided by XLA.  Others have complicated device-specific
lowerings, but represent a case where importance to stakeholders (Axis 6) outweighs complexity.
For this reason, we deem both of these submodules in-scope for JAX.

#### ❌ `numpy.random`
{mod}`numpy.random` is out-of-scope for JAX, because state-based RNGs are fundamentally
incompatible with JAX’s computation model. We instead focus on {mod}`jax.random`,
which offers similar functionality using a counter-based PRNG.

#### ❌ `numpy.ma` & `numpy.polynomial`
The {mod}`numpy.ma` and {mod}`numpy.polynomial` submodules are mostly concerned with
providing object-oriented interfaces to computations that can be expressed via other
functional means (Axis 5); for this reason, we deem them out-of-scope for JAX.

#### ❌ `numpy.testing`
NumPy’s testing functionality only really makes sense for host-side computation,
and so we don’t include any wrappers for it in JAX. That said, JAX arrays are
compatible with {mod}`numpy.testing`, and JAX makes frequent use of it throughout
the JAX test suite.

### SciPy APIs
SciPy has no functions in the top-level namespace, but includes a number of
submodules. We consider each below, leaving out modules which have been deprecated.

#### ❌ `scipy.cluster`
The {mod}`scipy.cluster` module includes tools for hierarchical clustering, k-means,
and related algorithms. These are weak along several axes, and would be better
served by a downstream package. One function already exists within JAX
({func}`jax.scipy.cluster.vq.vq`) but has
[no obvious usage](https://github.com/search?q=%22jax.scipy.cluster%22+AND+vq&type=code&p=5)
on github: this suggests that clustering is not broadly important to JAX users.

*Recommendation: deprecate and remove {func}`jax.scipy.cluster.vq`.*

#### ❌ `scipy.constants`
The {mod}`scipy.constants` module includes mathematical and physical constants.
These constants can be used directly with JAX, and so there is no reason to
re-implement this in JAX.

#### ❌ `scipy.datasets`
The {mod}`scipy.datasets` module includes tools to fetch and load datasets.
These fetched datasets can be used directly with JAX, and so there is no
reason to re-implement this in JAX.

#### ✅ `scipy.fft`
The {mod}`scipy.fft` module contains functions that broadly align with functionality
provided by XLA, and fare well along other axes as well. For this reason,
we deem them in-scope for JAX.

#### ❌ `scipy.integrate`
The {mod}`scipy.integrate` module contains functions for numerical integration. The
more sophisticated of these (`quad`, `dblquad`, `ode`) are out-of-scope for JAX by
axes 1 & 4, since they tend to be loopy algorithms based on dynamic numbers of
evaluations. {func}`jax.experimental.ode.odeint` is related, but rather limited and not
under any active development.

JAX does currently include {func}`jax.scipy.integrate.trapezoid`, but this is only because
{func}`numpy.trapz` was recently deprecated in favor of this. For any particular input,
its implementation could be replaced with one line of {mod}`jax.numpy` expressions, so
it’s not a particularly useful API to provide.

Based on Axes 1, 2, 4, and 6, {mod}`scipy.integrate` should be considered out-of-scope for JAX.

*Recommendation: remove {func}`jax.scipy.integrate.trapezoid`, which was added in JAX 0.4.14.*

#### ❌ `scipy.interpolate`
The {mod}`scipy.interpolate` module provides both low-level and object-oriented routines
for interpolating in one or more dimensions. These APIs rate poorly along a number
of the axes above: they are class-based rather than low-level, and none but the
simplest methods can be expressed efficiently in terms of XLA operations.

JAX does currently have wrappers for {class}`scipy.interpolate.RegularGridInterpolator`.
Were we considering this contribution today, we would probably reject it by the
above criteria. But this code has been fairly stable so there is not much downside
to continuing to maintain it.

Going forward, we should consider other members of {mod}`scipy.interpolate` to be
out-of-scope for JAX.

#### ❌ `scipy.io`
The {mod}`scipy.io` submodule has to do with file input/output. There is no reason
to re-implement this in JAX.

#### ✅ `scipy.linalg`
The {mod}`scipy.linalg` submodule contains functions that broadly align with functionality
provided by XLA, and fast linear algebra is broadly important to the JAX user community.
For this reason, we deem it in-scope for JAX.

#### ❌ `scipy.ndimage`
The {mod}`scipy.ndimage` submodule contains a set of tools for working on image data. Many
of these overlap with tools in {mod}`scipy.signal` (e.g. convolutions and filtering). JAX
currently provides one {mod}`scipy.ndimage` API, in {func}`jax.scipy.ndimage.map_coordinates`.
Additionally, JAX provides some image-related tools in the `jax.image` module. The
deepmind ecosystem includes [dm-pix](https://github.com/google-deepmind/dm_pix), a
more full-featured set of tools for image manipulation in JAX. Given all these factors,
I’d suggest that {mod}`scipy.ndimage` should be considered out-of-scope for JAX core; we can
point interested users and contributors to dm-pix. We can consider moving `map_coordinates`
to `dm-pix` or to another appropriate package.

#### ❌ `scipy.odr`
The {mod}`scipy.odr` module provides an object-oriented wrapper around `ODRPACK` for
performing orthogonal distance regressions. It is not clear that this could be cleanly
expressed using existing JAX primitives, and so we deem it out of scope for JAX itself.

#### ❌ `scipy.optimize`
The {mod}`scipy.optimize` module provides high-level and low-level interfaces for optimization.
Such functionality is important to a lot of JAX users, and very early on JAX created
{mod}`jax.scipy.optimize` wrappers. However, developers of these routines soon realized that
the {mod}`scipy.optimize` API was too constraining, and different teams began working on the
[JAXopt](https://jaxopt.github.io/) package and the
[Optimistix](https://github.com/patrick-kidger/optimistix) package, each of which contain
a much more comprehensive and better-tested set of optimization routines in JAX.

Because of these well-supported external packages, we now consider {mod}`scipy.optimize`
to be out-of-scope for JAX.

*Recommendation: deprecate {mod}`jax.scipy.optimize` and/or make it a lightweight wrapper
around an optional JAXopt or Optimistix dependency.*

#### 🟡 `scipy.signal`
The {mod}`scipy.signal` module is mixed: some functions are squarely in-scope for JAX
(e.g. `correlate` and `convolve`, which are more user-friendly wrappers of
`lax.conv_general_dilated`), while many others are squarely out-of-scope (domain-specific
tools with no viable lowering path to XLA). Potential contributions to {mod}`jax.scipy.signal`
will have to be weighed on a case-by-case basis.

#### 🟡 `scipy.sparse`
The {mod}`scipy.sparse` submodule mainly contains data structures for storing and operating
on sparse matrices and arrays in a variety of formats. Additionally, {mod}`scipy.sparse.linalg`
contains a number of matrix-free solvers, suitable for use with sparse matrices,
dense matrices, and linear operators.

The {mod}`scipy.sparse` array and matrix data structures are out-of-scope for JAX, because
they do not align with JAX’s computational model (e.g. many operations depend on
dynamically-sized buffers). JAX has developed the `jax.experimental.sparse` module
as an alternative set of data structures that are more in-line with JAX’s computational
constraints. For these reasons, we consider the data structures in {mod}`scipy.sparse` to
be out-of-scope for JAX.

On the other hand, {mod}`scipy.sparse.linalg` has proven to be an interesting area, and
{mod}`jax.scipy.sparse.linalg` includes the `bicgstab`, `cg`, and `gmres` solvers. These
are useful to the JAX user community (Axis 6) but aside from this do not fare well
along other axes. They would be very suitable for moving into a downstream library;
one potential option may be [Lineax](https://github.com/google/lineax), which features
a number of linear solvers built on JAX.

*Recommendation: explore moving sparse solvers into Lineax, and otherwise treat
`scipy.sparse`` as out-of-scope for JAX.*

#### ❌ `scipy.spatial`
The {mod}`scipy.spatial` module contains mainly object-oriented interfaces to spatial/distance
computations and nearest neighbor searches. It is mostly out-of-scope for JAX

The {mod}`scipy.spatial.transform` submodule provides tools for manipulating three-dimensional
spatial rotations. It is a relatively complicated object-oriented interface, and could
perhaps be better served by a downstream project. JAX currently contains partial
implementations of {class}`~jax.scipy.spatial.transform.Rotation` and
{class}`~jax.scipy.spatial.transform.Slerp` within {mod}`jax.scipy.spatial.transform`;
these are object-oriented wrappers of otherwise basic
functions, which introduce a very large API surface and have very few users. It is our
judgment that they are out-of-scope for JAX itself, with users better-served by a
hypothetical downstream project.

The {mod}`scipy.spatial.distance` submodule contains a useful collection of distance metrics,
and it might be tempting to provide JAX wrappers for these. That said, with jit and vmap
it would be straightforward for a user to define efficient versions of most of these from
scratch if needed, so adding them to JAX is not particularly beneficial.

*Recommendation: consider deprecating and removing the {class}`Rotation` and `Slerp` APIs, and
consider {mod}`scipy.spatial` as a whole out-of-scope for future contributions.*

#### ✅ `scipy.special`
The {mod}`scipy.special` module includes implementations of a number of more specialized
functions. In many cases, these functions are squarely in scope: for example, functions
like `gammaln`, `betainc`, `digamma`, and many others correspond directly to available
XLA primitives, and are clearly in-scope by Axis 1 and others.

Other functions require more complicated implementations; one example mentioned above
is `bessel_jn`. Despite not aligning on Axes 1 and 2, these functions tend to be very
strong along Axis 6: {mod}`scipy.special` provides fundamental functions necessary for
computation in a variety of domains, so even functions with complicated implementations
should lean toward in-scope, so long as the implementations are well-designed and robust.

There are a few existing function wrappers that we should take a closer look at; for example:
- {func}`jax.scipy.special.lpmn`: this generates legendre polynomials via a complicated fori_loop,
  in a way that does not match the scipy API (e.g. for `scipy`, `z` must be a scalar, while for
  JAX, `z` must be a 1D array). The function has few discoverable uses making it a weak
  candidate along Axes 1, 2, 4, and 6.
- {func}`jax.scipy.special.lpmn_values`: this has similar weaknesses to `lmpn` above.
- {func}`jax.scipy.special.sph_harm`: this is built on lpmn, and similarly has an API that diverges
  from the corresponding `scipy` function.
- {func}`jax.scipy.special.bessel_jn`: as discussed under Axis 4 above, this has weaknesses in
  terms of robustness of implementation and little usage. We might consider replacing it
  with a new, more robust implementation (e.g. {jax-issue}`#17038`). 

*Recommendation: refactor and improve robustness & test coverage for `bessel_jn`. Consider deprecating  `lpmn`, `lpmn_values`, and `sph_harm` if they cannot be modified to more closely match the `scipy` APIs.*

#### ✅ `scipy.stats`
The {mod}`scipy.stats` module contains a wide range of statistical functions, including discrete
and continuous distributions, summary statistics, and hypothesis testing. JAX currently wraps
a number of these in {mod}`jax.scipy.stats`, primarily including 20 or so statistical distributions,
along with a few other functions (`mode`, `rankdata`, `gaussian_kde`). In general these are
well-aligned with JAX: distributions usually are expressible in terms of efficient XLA operations,
and the APIs are clean and functional.

We don’t currently have any wrappers for hypothesis testing functions, probably because
these are less useful to the primary user-base of JAX.

Regarding distributions, in some cases, `tensorflow_probability` provides similar functionality,
and in the future we might consider whether to deprecate the scipy.stats distributions in favor
of that implementation.

*Recommendation: going forward, we should treat statistical distributions and summary statistics as in-scope, and consider hypothesis tests and related functionality generally out-of-scope.*

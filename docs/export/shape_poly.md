(shape_poly)=

# Shape polymorphism

When JAX is used in JIT mode, a function will be traced, lowered to StableHLO, and compiled for each
combination of input types and shapes. After exporting a function and
deserializing it on another system we don't have the Python sources available anymore,
so we cannot re-trace and re-lower it. **Shape polymorphism** is a feature of JAX export
to allow some exported functions to be used for a whole family of input shapes.
These functions are traced and lowered once, during exporting, and `Exported`
object contains the information needed to be able to compile and execute the function
on many concrete input shapes. We do this by specifying shapes that contain
dimension variables (symbolic shapes) when exporting, as in the
following example:

```python
>>> import jax
>>> from jax import export
>>> from jax import numpy as jnp
>>> def f(x):  # f: f32[a, b]
...   return jnp.concatenate([x, x], axis=1)

>>> # We construct symbolic dimension variables.
>>> a, b = export.symbolic_shape("a, b")

>>> # We can use the symbolic dimensions to construct shapes.
>>> x_shape = (a, b)
>>> x_shape
(a, b)

>>> # Then we export with symbolic shapes:
>>> exp: export.Exported = export.export(jax.jit(f))(
...     jax.ShapeDtypeStruct(x_shape, jnp.int32))
>>> exp.in_avals
(ShapedArray(int32[a,b]),)
>>> exp.out_avals
(ShapedArray(int32[a,2*b]),)

>>> # We can later call with concrete shapes (with a=3 and b=4), without re-tracing `f`.
>>> res = exp.call(np.ones((3, 4), dtype=np.int32))
>>> res.shape
(3, 8)

```

Note that such functions are still re-compiled on demand for
each concrete input shapes they are invoked on. Only the
tracing and the lowering are saved.

The {func}`jax.export.symbolic_shape` is used in the above
example to parse a string representation of a symbolic shape
into dimension expressions objects (of type `_DimExpr`) that are usable in place of integer
constants to construct shapes. The dimension expression objects
overload most integer operators, so you can use them as
you'd use integer constants in most cases.
See {ref}`computing-with-dimension-variables` for more details.

Additionally, we provide the {func}`jax.export.symbolic_args_specs` that
can be used to construct pytrees of `jax.ShapeDtypeStruct` objects based
on a polymorphic shape specification:

```python
>>> def f1(x, y): # x: f32[a, 1], y : f32[a, 4]
...  return x + y

>>> # Assuming you have some actual args with concrete shapes
>>> x = np.ones((3, 1), dtype=np.int32)
>>> y = np.ones((3, 4), dtype=np.int32)
>>> args_specs = export.symbolic_args_specs((x, y), "a, ...")
>>> exp = export.export(jax.jit(f1))(* args_specs)
>>> exp.in_avals
(ShapedArray(int32[a,1]), ShapedArray(int32[a,4]))

```

Note how the polymorphic shape specification `"a, ..."` contains
the placeholder `...` to be filled from the concrete shapes of
the concrete shapes of the arguments `(x, y)`.
The placeholder `...` stands for 0 or more dimensions, while the
placeholder `_` stands for one dimension.
The {func}`jax.export.symbolic_args_specs` supports pytrees of arguments,
which are used to fill-in the dtypes and any placeholders.
The function will construct a pytree of
argument specifications ({class}`jax.ShapeDtypeStruct`)
matching the structure of the arguments passed to it.
The polymorphic shapes specification can be a
pytree prefix in cases where one specification should apply
to multiple arguments, as in the above example.
See [how optional parameters are matched to arguments](https://jax.readthedocs.io/en/latest/pytrees.html#applying-optional-parameters-to-pytrees).


A few examples of shape specifications:

  * `("(b, _, _)", None)` can be used for a function with two arguments, the first
    being a 3D array with a batch leading dimension that should be symbolic.
    The other dimensions for the
    first argument and the shape of the second argument are specialized based on the actual
    arguments. Note that the same specification would work if the first
    argument is a pytree of 3D arrays, all with the same leading dimension
    but possibly with different trailing dimensions.
    The value `None` for the second arugment means that the argument
    is not symbolic. Equivalently, one can use `...`.

  * `("(batch, ...)", "(batch,)")` specifies that the two arguments
    have matching leading dimensions, the first argument has rank at
    least 1, and the second has rank 1.

## Correctness of shape polymorphism

We want to trust that the exported program produces the same results as the
original JAX program when compiled and executed for any applicable concrete shapes.
More precisely:

For any JAX function `f` and any argument specification `arg_spec` containing a
symbolic shape, and any concrete argument `arg` whose shape matches `arg_spec`:

 * If the JAX native execution succeeds on the concrete argument: `res = f(arg)`,
 * and if the exporting succeeds with symbolic shapes: `exp = export.export(f)(arg_spec)`,
 * then compiling and running the export will succeed with the same result: `res == exp.call(arg)`

It is crucial to understand that `f(arg)` has the freedom to re-invoke
the JAX tracing machinery,
and in fact it does so for each distinct concrete `arg` shape,
while the execution of `exp.call(arg)` cannot use JAX tracing anymore
(this execution may happen in an environment where the source code
of `f` is not available).

Ensuring this form of correctness is hard, and in the hardest cases
exporting fails. The rest of this chapter describes how to handle these failures.

(computing-with-dimension-variables)=

## Computing with dimension variables

JAX keeps track of the shapes of all intermediate results. When those shapes depend
on dimension variables JAX computes them as symbolic dimension expressions
involving dimension variables.
Dimension variables stand for integer values greater or equal to 1.
The symbolic expressions can represent the result
of applying arithmetic operators (add, sub, mul, floordiv, mod,
including the NumPy variants `np.sum`, `np.prod`, etc.) **on dimension
expressions and integers** (`int`, `np.int`, or anything convertible by `operator.index`).
These symbolic dimensions can then be used in shape-parameters of JAX primitives
and APIs, e.g., in `jnp.reshape`, `jnp.arange`, slicing indices, etc.

For example, in the following code to flatten a 2D array, the computation
`x.shape[0] * x.shape[1]` computes the symbolic dimension `4 * b` as the
new shape:

```python
>>> f = lambda x: jnp.reshape(x, (x.shape[0] * x.shape[1],))
>>> arg_spec = jax.ShapeDtypeStruct(export.symbolic_shape("b, 4"), jnp.int32)
>>> exp = export.export(jax.jit(f))(arg_spec)
>>> exp.out_avals
(ShapedArray(int32[4*b]),)

```

It is possible to convert dimension expressions explicitly
to JAX arrays, with `jnp.array(x.shape[0])` or even `jnp.array(x.shape)`.
The result of these operations can be used as regular JAX arrays,
bug cannot be used anymore as dimensions in shapes.

```python
>>> exp = export.export(jax.jit(lambda x: jnp.array(x.shape[0]) + x))(
...     jax.ShapeDtypeStruct(export.symbolic_shape("b"), np.int32))
>>> exp.call(jnp.arange(3, dtype=np.int32))
Array([3, 4, 5], dtype=int32)

>>> exp = export.export(jax.jit(lambda x: x.reshape(jnp.array(x.shape[0]) + 2)))(
...     jax.ShapeDtypeStruct(export.symbolic_shape("b"), np.int32))  # doctest: +IGNORE_EXCEPTION_DETAIL
Traceback (most recent call last):
TypeError: Shapes must be 1D sequences of concrete values of integer type, got [Traced<ShapedArray(int32[], weak_type=True)>with<DynamicJaxprTrace(level=1/0)>].

```

When a symbolic dimension is used in arithmetic operations with **non-integers**,
e.g., `float`, `np.float`, `np.ndarray`, or JAX arrays, it is automatically
converted to a JAX array using `jnp.array`.
For example, in the function below all occurrences of `x.shape[0]`
are converted implicitly to `jnp.array(x.shape[0])` because
they are involved in operations with non-integer scalars or with
JAX arrays:

```python
>>> exp = export.export(jax.jit(
...     lambda x: (5. + x.shape[0],
...                x.shape[0] - np.arange(5, dtype=jnp.int32),
...                x + x.shape[0] + jnp.sin(x.shape[0]))))(
...     jax.ShapeDtypeStruct(export.symbolic_shape("b"), jnp.int32))
>>> exp.out_avals
(ShapedArray(float32[], weak_type=True),
 ShapedArray(int32[5]),
 ShapedArray(float32[b], weak_type=True))

>>> exp.call(jnp.ones((3,), jnp.int32))
 (Array(8., dtype=float32, weak_type=True),
  Array([ 3, 2, 1, 0, -1], dtype=int32),
  Array([4.14112, 4.14112, 4.14112], dtype=float32, weak_type=True))

```

Another typical example is when computing averages
(observe how `x.shape[0]` is automatically turned into a JAX array):

```python
>>> exp = export.export(jax.jit(
...     lambda x: jnp.sum(x, axis=0) / x.shape[0]))(
...     jax.ShapeDtypeStruct(export.symbolic_shape("b, c"), jnp.int32))
>>> exp.call(jnp.arange(12, dtype=jnp.int32).reshape((3, 4)))
Array([4., 5., 6., 7.], dtype=float32)

```

### Errors in presence of shape polymorphism

Most JAX code assumes that the shapes of JAX arrays are tuples of integers,
but with shape polymorphism some dimensions may be symbolic expressions.
This can lead to a number of errors. For example, we can have the usual
JAX shape check errors:

```python
>>> v, = export.symbolic_shape("v,")
>>> export.export(jax.jit(lambda x, y: x + y))(
...     jax.ShapeDtypeStruct((v,), dtype=np.int32),
...     jax.ShapeDtypeStruct((4,), dtype=np.int32))
Traceback (most recent call last):
TypeError: add got incompatible shapes for broadcasting: (v,), (4,).

>>> export.export(jax.jit(lambda x: jnp.matmul(x, x)))(
...     jax.ShapeDtypeStruct((v, 4), dtype=np.int32))
Traceback (most recent call last):
TypeError: dot_general requires contracting dimensions to have the same shape, got (4,) and (v,).

```

We can fix the above matmul example by specifying that the
argument has shape `(v, v)`.

### Comparison of symbolic dimensions is partially supported

Inside JAX there are a number of equality and inequality comparisons
involving shapes, e.g., for doing shape checking or even for choosing
the implementation for some primitives. Comparisons are supported
as follows:

  * equality is supported with a caveat: if the two symbolic dimensions denote the same
    value under all valuations for dimension variables, then equality evaluates to `True`,
    e.g., for `b + b == 2*b`; otherwise the equality evaluates to `False`.
    See [below](#caveat-for-equality-comparisons)
    for a discussion of important consequences of this behavior.
  * disequality is always the negation of equality.
  * inequality is partially supported, in a similar way as partial equality.
    However, in this
    case we take into consideration that dimension variables range over strictly positive
    integers. E.g., `b >= 1`, `b >= 0`, `2 * a + b >= 3` are `True`, while `b >= 2`,
    `a >= b`, `a - b >= 0` are inconclusive and result in an exception.

In cases where a comparison operation cannot be resolve to a boolean,
we raise {class}`InconclusiveDimensionOperation`. E.g.,

```python
import jax
>>> export.export(jax.jit(lambda x: 0 if x.shape[0] + 1 >= x.shape[1] else 1))(
...     jax.ShapeDtypeStruct(export.symbolic_shape("a, b"), dtype=np.int32))  # doctest: +IGNORE_EXCEPTION_DETAIL
Traceback (most recent call last):
jax._src.export.shape_poly.InconclusiveDimensionOperation: Symbolic dimension comparison 'a + 1' >= 'b' is inconclusive.
This error arises for comparison operations with shapes that
are non-constant, and the result of the operation cannot be represented as
a boolean value for all values of the symbolic dimensions involved.

```

If you do get a `InconclusiveDimensionOperation`, you can try
several strategies:

 * If your code uses the built-in `max` or `min`, or the
   `np.max` or `np.min` then you can replace those with
   `core.max_dim` and `core.min_dim`, which have the effect
   of delaying the inequality comparison to the compilation
   time, when shapes become known.
 * Try to rewrite conditionals using `core.max_dim` and
   `core.min_dim`, e.g., instead of `d if d > 0 else 0`
   you can write `core.max_dim(d, 0)`.
 * Try to rewrite the code to be less dependent on the fact
   that dimensions should be integers, and rely on the fact
   that symbolic dimensions duck-type as integers for most
   arithmetic operations. E.g., instead of `int(d) + 5` write
   `d + 5`.
 * Specify symbolic constraints, as explained below.

#### User-specified symbolic constraints

By default, JAX assumes that all dimension variables range
over values greater-or-equal to 1, and it tries to derive
other simple inequalities from that, e.g.:

  * `a + 2 >= 3`,
  * `a * 2 >= 1`,
  * `a + b + c >= 3`,
  * `a // 4 >= 0`, `a**2 >= 1`, and so on.

You can avoid some inequality comparison failures if you
change the symbolic shape specifications to add **implicit** constraints
for dimension sizes. E.g.,

  * You can use `2*b` for a dimension to constrain it to be even and greater or equal
    to 2.
  * You can use `b + 15` for a dimension to constrain it to
    be at least 16. E.g., the following code would fail without
    the `+ 15` part, because JAX will want to verify that slice sizes
    are at most as large as the axis size.

```python
>>> _ = export.export(jax.jit(lambda x: x[0:16]))(
...    jax.ShapeDtypeStruct(export.symbolic_shape("b + 15"), dtype=np.int32))

```

Such implicit symbolic constraints are used for deciding comparisons and are
checked at compile time, as explained [below](#shape-assertion-errors).

You can also specify **explicit** symbolic constraints:

```python
>>> # Introduce dimension variable with constraints.
>>> a, b = export.symbolic_shape("a, b",
...                              constraints=("a >= b", "b >= 16"))
>>> _ = export.export(jax.jit(lambda x: x[:x.shape[1], :16]))(
...    jax.ShapeDtypeStruct((a, b), dtype=np.int32))

```

The constraints form a conjunction together with the implicit
constraints. You can specify `>=`, `<=`, and `==` constraints.
At the moment, JAX has limited support for reasoning with
symbolic constraints:

  * You get the most from constraints of the form
    of a variable being greater-or-equal or
    less-or-equal to a constant.
    For example, from the constraints that
    `a >= 16` and `b >= 8` we can infer
    that `a + 2*b >= 32`.
  * You get limited power when the constraint involves
    more complex expressions, e.g., from `a >= b + 8` we
    can infer that `a - b >= 8` but not that `a >= 9`.
    We may improve somewhat this area in the future.
  * Equality constraints are treated as normalization rules.
    E.g., `floordiv(a, b) = c` works by replacing all
    occurences of the left-hand-side with the right-hand-side.
    You can only have equality constraints where the left-hand-side
    is a multiplication of factors, e.g, `a * b`, or `4 * a`, or
    `floordiv(a, b)`. Thus, the left-hand-side cannot contain
    addition or subtraction at the top-level.

The symbolic constraints can also help to work around the
limitations in the JAX reasoning mechanisms.
For example, in the code below JAX will attempt to prove that
the slice size `x.shape[0] % 3`, which is the symbolic expression
`mod(b, 3)`, is less or equal to the axis size, which is `b`.
This happens to be true for all strictly positive values of
`b`, but it is not something JAX's symbolic comparison rules
can prove. Hence the following code raises an error:

```python
from jax import lax
>>> b, = export.symbolic_shape("b")
>>> f = lambda x: lax.slice_in_dim(x, 0, x.shape[0] % 3)
>>> export.export(jax.jit(f))(
...     jax.ShapeDtypeStruct((b,), dtype=np.int32))  # doctest: +IGNORE_EXCEPTION_DETAIL
Traceback (most recent call last):
jax._src.export.shape_poly.InconclusiveDimensionOperation: Symbolic dimension comparison 'b' >= 'mod(b, 3)' is inconclusive.
This error arises for comparison operations with shapes that
are non-constant, and the result of the operation cannot be represented as
a boolean value for all values of the symbolic dimensions involved.

```

One option here would be to restrict the code to work only on
axis sizes that are multiple of `3` (by replacing
`b` with `3*b` in the shape). Then, JAX would be able
to simplify the modulo operation `mod(3*b, 3)` to `0`.
Another option is to add a symbolic constraint
with the exact inconclusive inequality that JAX
is attempting to prove:

```python
>>> b, = export.symbolic_shape("b",
...                            constraints=["b >= mod(b, 3)"])
>>> f = lambda x: lax.slice_in_dim(x, 0, x.shape[0] % 3)
>>> _ = export.export(jax.jit(f))(
...     jax.ShapeDtypeStruct((b,), dtype=np.int32))

```

Just like the implicit constraints, the explicit
symbolic constraints are checked at compile time,
using the same mechanism as explained [below](#shape-assertion-errors).

#### Symbolic dimension scopes

The symbolic constraints are stored in αn
{class}`jax.export.SymbolicScope` object, which is created implicitly
for each call to {func}`jax.export.symbolic_shapes`. You must be careful
to not mix symbolic expressions that use different scopes.
For example,
the following code will fail because `a1` and `a2`
use different scopes (created by different invocations of
{func}`jax.export.symbolic_shape`):

```python
>>> a1, = export.symbolic_shape("a,")
>>> a2, = export.symbolic_shape("a,", constraints=("a >= 8",))

>>> a1 + a2  # doctest: +IGNORE_EXCEPTION_DETAIL
Traceback (most recent call last):
ValueError: Invalid mixing of symbolic scopes for linear combination.
Expected  scope 4776451856 created at <doctest shape_poly.md[31]>:1:6 (<module>)
and found for 'a' (unknown) scope 4776979920 created at <doctest shape_poly.md[32]>:1:6 (<module>) with constraints:
  a >= 8
```

The symbolic expressions that originate from a single call
to {func}`jax.export.symbolic_shape` share a scope and
can be mixed up in arithmetic operations. The result would
also share the same scope.

You can re-use scopes:

```python
>>> a, = export.symbolic_shape("a,", constraints=("a >= 8",))
>>> b, = export.symbolic_shape("b,", scope=a.scope)  # Reuse the scope of `a`

>>> a + b  # Allowed
b + a

```

You can also create scopes explicitly:

```python
>>> my_scope = export.SymbolicScope()
>>> c, = export.symbolic_shape("c", scope=my_scope)
>>> d, = export.symbolic_shape("d", scope=my_scope)
>>> c + d  # Allowed
d + c

```

JAX tracing uses caches keyed partially by shapes, and
symbolic shapes that are printed identically will be considered
distinct if they use different scopes.

### Caveat for equality comparisons

The equality comparison returns `False` for `b + 1 == b` or `b == 0`
(in which case it is certain that the dimensions are different for all values
of the dimension variables),
but also for `b == 1` and for `a == b`. This is unsound, and we
ought to raise `core.InconclusiveDimensionOperation` because under
some valuations the result should be `True` and under other
valuations it should be `False`. We choose to make equality total
thus allowing unsoundness because otherwise we may get spurious errors
in presence of hash collisions
when hashing dimension expressions or objects that include
them (shapes, `core.AbstractValue`, `core.Jaxpr`).
Besides the hashing errors, a partial semantics of equality
leads to errors for the following expressions `b == a or b == b` or `b in [a, b]`
even though the error is avoided if we change the order of the comparisons.

Code of the form `if x.shape[0] != 1: raise NiceErrorMessage` is sound even
with this treatment of equality, but code of the form `if x.shape[0] != 1: return 1`
is unsound.

### Dimension variables must be solvable from the input shapes

Currently, the only way to pass the values of dimension variables
when an exported object is invoked is indirectly through the shapes
of the array arguments. E.g., the value of `b` can be inferred at the
call site from the shape of the first argument of type `f32[b]`.
This works well for most use cases, and
it mirrors the calling convention of JIT functions.

Sometimes you may want to export a function parameterized
by an integer values that determines some shapes in the program.
For example, we may
want to export the function `my_top_k` defined below,
parameterized by the
value of `k`, which determined the shape of the result.
The following attempt will lead to an error since the dimension
variable `k` cannot be derived from the shape of the input `x: i32[4, 10]`:

```python
>>> def my_top_k(k, x):  # x: i32[4, 10], k <= 10
...   return lax.top_k(x, k)[0]  # : i32[4, 3]
>>> x = np.arange(40, dtype=np.int32).reshape((4, 10))

>>> # Export with static `k=3`. Since `k` appears in shapes it must be in `static_argnums`.
>>> exp_static_k = export.export(jax.jit(my_top_k, static_argnums=0))(3, x)
>>> exp_static_k.in_avals[0]
ShapedArray(int32[4,10])

>>> exp_static_k.out_avals[0]
ShapedArray(int32[4,3])

>>> # When calling the exported function we pass only the non-static arguments
>>> exp_static_k.call(x)
Array([[ 9,  8,  7],
       [19, 18, 17],
       [29, 28, 27],
       [39, 38, 37]], dtype=int32)

>>> # Now attempt to export with symbolic `k` so that we choose `k` after export.
>>> k, = export.symbolic_shape("k", constraints=["k <= 10"])
>>> export.export(jax.jit(my_top_k, static_argnums=0))(k, x)  # doctest: +IGNORE_EXCEPTION_DETAIL
Traceback (most recent call last):
KeyError: "Encountered dimension variable 'k' that is not appearing in the shapes of the function arguments

```

In the future, we may add an additional mechanism to pass the values of
dimension variables, besides implicitly through the input shapes.
Meanwhile, the workaround for the above use case is to replace the
function parameter `k` with an array of shape `(0, k)`, so that
`k` can be derived from the input shape of an array.
The first dimension is 0 to ensure that the whole array is empty
and there is no performance penalty when we call the exported function.

```python
>>> def my_top_k_with_dimensions(dimensions, x):  # dimensions: i32[0, k], x: i32[4, 10]
...   return my_top_k(dimensions.shape[1], x)
>>> exp = export.export(jax.jit(my_top_k_with_dimensions))(
...     jax.ShapeDtypeStruct((0, k), dtype=np.int32),
...     x)
>>> exp.in_avals
(ShapedArray(int32[0,k]), ShapedArray(int32[4,10]))

>>> exp.out_avals[0]
ShapedArray(int32[4,k])

>>> # When we invoke `exp` we must construct and pass an array of shape (0, k)
>>> exp.call(np.zeros((0, 3), dtype=np.int32), x)
Array([[ 9,  8,  7],
       [19, 18, 17],
       [29, 28, 27],
       [39, 38, 37]], dtype=int32)

```

Another situation when you may get an error is when some dimension
variables do appear in the input shapes, but in a non-linear
expression that JAX cannot currently solve:

```python
>>> a, = export.symbolic_shape("a")
>>> export.export(jax.jit(lambda x: x.shape[0]))(
...    jax.ShapeDtypeStruct((a * a,), dtype=np.int32))  # doctest: +IGNORE_EXCEPTION_DETAIL
Traceback (most recent call last):
ValueError: Cannot solve for values of dimension variables {'a'}.
We can only solve linear uni-variate constraints.
Using the following polymorphic shapes specifications: args[0].shape = (a^2,).
Unprocessed specifications: 'a^2' for dimension size args[0].shape[0].

```

### Shape assertion errors

JAX assumes that dimension variables range over strictly positive integers,
and this assumption is checked when the code is compiled for concrete
input shapes.

For example, given the symbolic input shape `(b, b, 2*d)`,
JAX will generate code to check the following assertions when
invoked with actual argument `arg`:

  * `arg.shape[0] >= 1`
  * `arg.shape[1] == arg.shape[0]`
  * `arg.shape[2] % 2 == 0`
  * `arg.shape[2] // 2 >= 1`

For example, here is the error we get when we call the exported
on an argument of shape `(3, 3, 5)`:

```python
>>> def f(x):  # x: f32[b, b, 2*d]
...   return x
>>> exp = export.export(jax.jit(f))(
...     jax.ShapeDtypeStruct(export.symbolic_shape("b, b, 2*d"), dtype=np.int32))   
>>> exp.call(np.ones((3, 3, 5), dtype=np.int32))  # doctest: +IGNORE_EXCEPTION_DETAIL
Traceback (most recent call last):
ValueError: Input shapes do not match the polymorphic shapes specification.
Division had remainder 1 when computing the value of 'd'.
Using the following polymorphic shapes specifications:
  args[0].shape = (b, b, 2*d).
Obtained dimension variables: 'b' = 3 from specification 'b' for dimension args[0].shape[0] (= 3), .
Please see https://jax.readthedocs.io/en/latest/export/shape_poly.html#shape-assertion-errors for more details.

```

These errors arise in a pre-processing step before the
compilation.

### Division of symbolic dimensions is partially supported

JAX will attempt to simplify division and modulo operations,
e.g., `(a * b + a) // (b + 1) == a` and `6*a + 4 % 3 == 1`.
In particular, JAX will handle the cases when either (a) there
is no remainder, or (b) the divisor is a constant
in which case there may be a constant remainder.

For example, the code below results in a division error when trying to
compute the inferred dimension for a `reshape` operation:

```python
>>> b, = export.symbolic_shape("b")
>>> export.export(jax.jit(lambda x: x.reshape((2, -1))))(
...     jax.ShapeDtypeStruct((b,), dtype=np.int32))
Traceback (most recent call last):
jax._src.core.InconclusiveDimensionOperation: Cannot divide evenly the sizes of shapes (b,) and (2, -1).
The remainder mod(b, - 2) should be 0.

```

Note that the following will succeed:

```python
>>> b, = export.symbolic_shape("b")
>>> # We specify that the first dimension is a multiple of 4
>>> exp = export.export(jax.jit(lambda x: x.reshape((2, -1))))(
...     jax.ShapeDtypeStruct((4*b,), dtype=np.int32))
>>> exp.out_avals
(ShapedArray(int32[2,2*b]),)

>>> # We specify that some other dimension is even
>>> exp = export.export(jax.jit(lambda x: x.reshape((2, -1))))(
...     jax.ShapeDtypeStruct((b, 5, 6), dtype=np.int32))
>>> exp.out_avals
(ShapedArray(int32[2,15*b]),)

```

(shape_poly_debugging)=
## Debugging

First, see the {ref}`export_debugging` documentation.
Additionally, you can debug the shape refinement, which is
invoked at compilation time for modules that have dimension variables or multi-platform
support.

If there is an error during shape refinement, you can set the `JAX_DUMP_IR_TO`
environment variable to see a dump of the HLO module before
shape refinement (named `..._before_refine_polymorphic_shapes.mlir`).
This module should already have static input shapes.

To enable the logging of all stages of shape refinement you can set the
environment variable `TF_CPP_VMODULE=refine_polymorphic_shapes=3` in OSS
(inside Google, you pass `--vmodule=refine_polymorphic_shapes=3`):

```shell
# Log from python
JAX_DUMP_IR_TO=/tmp/export.dumps/ TF_CPP_VMODULE=refine_polymorphic_shapes=3 python tests/shape_poly_test.py ShapePolyTest.test_simple_unary -v=3
```

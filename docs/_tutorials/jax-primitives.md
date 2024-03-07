---
jupytext:
  formats: md:myst
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.16.1
kernelspec:
  display_name: Python 3
  language: python
  name: python3
---

(jax-internals-jax-primitives)=
# JAX Internals: primitives

<!--* freshness: { reviewed: '2024-05-03' } *-->

## Introduction to JAX primitives

A JAX primitive is the basic computational unit of a JAX program. This document explains the interface that a JAX primitive must support to allow JAX to perform all its transformations (this is not a how-to guide).

For example, the multiply-add operation can be implemented in terms of the low-level `jax.lax.*` primitives (which are like XLA operator wrappers) or `jax.core.Primitive("multiply_add")`, as demonstrated further below. 

And JAX is able to take sequences of such primitive operations, and transform them via its composable transformations of Python functions, such as {func}`jax.jit`, {func}`jax.grad` and {func}`jax.vmap`. JAX implements these transforms in a *JAX-traceable* way. This means that when a Python function is executed, the only operations it applies to the data are either:

- **Inspections of data attributes:** Data information, such as shape or type; or 
- **JAX primitives:** These are the JAX special operations covered in this tutorial.

JAX primitives know how to operate on both concrete data values and abstract JAX values. *A JAX-traceable function* can be invoked by JAX with abstract arguments. For example, a JAX abstract value — `ShapedArray(float32[2,2])` — captures the type and the shape of values, but not the concrete data values. 

The JAX-transformed functions must themselves be JAX-traceable functions *to make sure that these transformations are composable*, for example like `jax.jit(jax.jacfwd(jax.grad(f)))`.

JAX provides pre-defined primitives corresponding to most XLA operations, including add, matmul, sin, cos, and indexing.

In addition, JAX offers an implementation of NumPy functions in terms of JAX primitives. This means that *Python programs using JAX’s implementation of NumPy are JAX-traceable and, therefore, transformable*. Other libraries can be made JAX-traceable by implementing them in terms of JAX primitives.

Furthermore, the set of JAX primitives is extensible, so instead of reimplementing a function in terms of pre-defined JAX primitives, you can define a new primitive that encapsulates the behavior of the function.

Consider the following example: you want to add to JAX support for a multiply-add function with three arguments, defined mathematically as `multiply_add(x, y, z) = x * y + z`. This function operates on 3 identically-shaped tensors of floating point values and performs the operations pointwise. You can do this by:

- {ref}`using-existing-jax-primitives`; or
- {ref}`defining-new-jax-primitives`

(using-existing-jax-primitives)=
## Using existing JAX primitives

The easiest way to define new functions is to write them in terms of JAX primitives, or in terms of other functions that are themselves written using JAX primitives, for example, those defined in the {func}`jax.lax` module:

```{code-cell}
from jax import lax
from jax._src import api

def multiply_add_lax(x, y, z):
  """Implementation of multiply-add using the `jax.lax` primitives."""
  return lax.add(lax.mul(x, y), z)


def square_add_lax(a, b):
  """A square-add function using the newly defined multiply-add."""
  return multiply_add_lax(a, a, b)

print("square_add_lax = ", square_add_lax(2., 10.))
# Differentiate w.r.t. the first argument
print("grad(square_add_lax) = ", api.grad(square_add_lax, argnums=0)(2.0, 10.))
```

To understand how JAX is internally using the primitives, add some helpers for tracing function calls:

```{code-cell}
#@title Helper functions (execute this cell)
import functools
import traceback

_indentation = 0
def _trace(msg=None):
    """Print a message at current indentation."""
    if msg is not None:
        print("  " * _indentation + msg)

def _trace_indent(msg=None):
    """Print a message and then indent the rest."""
    global _indentation
    _trace(msg)
    _indentation = 1 + _indentation

def _trace_unindent(msg=None):
    """Unindent then print a message."""
    global _indentation
    _indentation = _indentation - 1
    _trace(msg)

def trace(name):
  """A decorator for functions to trace arguments and results."""

  def trace_func(func):  # pylint: disable=missing-docstring
    def pp(v):
        """Print certain values more succinctly"""
        vtype = str(type(v))
        if "jax._src.xla_bridge._JaxComputationBuilder" in vtype:
            return "<JaxComputationBuilder>"
        elif "jaxlib.xla_extension.XlaOp" in vtype:
            return "<XlaOp at 0x{:x}>".format(id(v))
        elif ("partial_eval.JaxprTracer" in vtype or
              "batching.BatchTracer" in vtype or
              "ad.JVPTracer" in vtype):
            return "Traced<{}>".format(v.aval)
        elif isinstance(v, tuple):
            return "({})".format(pp_values(v))
        else:
            return str(v)
    def pp_values(args):
        return ", ".join([pp(arg) for arg in args])
    
    @functools.wraps(func)
    def func_wrapper(*args):
      _trace_indent("call {}({})".format(name, pp_values(args)))
      res = func(*args)
      _trace_unindent("|<- {} = {}".format(name, pp(res)))
      return res

    return func_wrapper

  return trace_func

class expectNotImplementedError(object):
  """Context manager to check for NotImplementedError."""
  def __enter__(self): pass
  def __exit__(self, type, value, tb):
    global _indentation
    _indentation = 0
    if type is NotImplementedError:
      print("\nFound expected exception:")
      traceback.print_exc(limit=3)
      return True
    elif type is None:  # No exception
      assert False, "Expected NotImplementedError"
    else:
      return False
```

Instead of using {func}`jax.lax` primitives directly, you can use other functions 
that are already written in terms of those primitives, such as those in `jax.numpy`:

```{code-cell}
import jax.numpy as jnp
import numpy as np

@trace("multiply_add_numpy")
def multiply_add_numpy(x, y, z):
    return jnp.add(jnp.multiply(x, y), z)

@trace("square_add_numpy")
def square_add_numpy(a, b):
    return multiply_add_numpy(a, a, b)

print("\nNormal evaluation:")  
print("square_add_numpy = ", square_add_numpy(2., 10.))
print("\nGradient evaluation:")
print("grad(square_add_numpy) = ", api.grad(square_add_numpy)(2.0, 10.))
```

Notice that in the process of computing {func}`jax.grad`, JAX invokes `square_add_numpy` and `multiply_add_numpy` with special arguments `ConcreteArray(...)` (described further below in this colab). It is important to remember that a JAX-traceable function must be able to operate not only on concrete arguments but also on special abstract arguments that JAX may use to abstract the function execution.

The JAX traceability property is satisfied as long as the function is written in terms of JAX primitives.

(defining-new-jax-primitives)=
## Defining new JAX primitives

The right way to add support for multiply-add is in terms of existing JAX primitives, as shown above. However, to demonstrate how JAX primitives work, pretend that you want to add a new primitive to JAX for the multiply-add functionality.

```{code-cell}
from jax import core

multiply_add_p = core.Primitive("multiply_add")  # Create the primitive

@trace("multiply_add_prim")
def multiply_add_prim(x, y, z):
  """The JAX-traceable way to use the JAX primitive.
  
  Note that the traced arguments must be passed as positional arguments
  to `bind`. 
  """
  return multiply_add_p.bind(x, y, z)

@trace("square_add_prim")
def square_add_prim(a, b):
  """A square-add function implemented using the new JAX-primitive."""
  return multiply_add_prim(a, a, b)
```

If you try to call the newly defined functions, you'll get an error, because you haven't yet told JAX anything about the semantics of the new primitive.

```{code-cell}
:tags: [raises-exception]

with expectNotImplementedError():
  square_add_prim(2., 10.)
```

### Primal evaluation rules

```{code-cell}
@trace("multiply_add_impl")
def multiply_add_impl(x, y, z):
  """Concrete implementation of the primitive.

  This function does not need to be JAX traceable.

  Args:
    x, y, z: The concrete arguments of the primitive. Will only be called with 
      concrete values.

  Returns:
    the concrete result of the primitive.
  """
  # Note: you can use the ordinary (non-JAX) NumPy, which is not JAX-traceable.
  return np.add(np.multiply(x, y), z)

# Now, register the primal implementation with JAX:
multiply_add_p.def_impl(multiply_add_impl)
```

```{code-cell}
assert square_add_prim(2., 10.) == 14.
```

### What happens when you use `jit`

Now, if you try to use `jit`, you'll get a `NotImplementedError`:

```{code-cell}
:tags: [raises-exception]

with expectNotImplementedError():
  api.jit(square_add_prim)(2., 10.)
```

#### Abstract evaluation rules

To JIT the function, and for other transformations as well, JAX first evaluates it abstractly using only the shape and type of the arguments. This abstract evaluation serves multiple purposes:

  * Gets the sequence of JAX primitives that are used in the computation. This sequence will be compiled. 
  * Computes the shape and type of all vectors and operations used in the computation. 

For example, the abstraction of a vector with 3 elements may be `ShapedArray(float32[3])`, or `ConcreteArray([1., 2., 3.])`.  In the latter case, JAX uses the actual concrete value wrapped as an abstract value.

```{code-cell}
from jax import core

@trace("multiply_add_abstract_eval")
def multiply_add_abstract_eval(xs, ys, zs):
  """Abstract evaluation of the primitive.

  This function does not need to be JAX traceable. It will be invoked with
  abstractions of the actual arguments

  Args:
    xs, ys, zs: Abstractions of the arguments.

  Result:
    a ShapedArray for the result of the primitive.
  """
  assert xs.shape == ys.shape
  assert xs.shape == zs.shape
  return core.ShapedArray(xs.shape, xs.dtype)

# Now, register the abstract evaluation with JAX:
multiply_add_p.def_abstract_eval(multiply_add_abstract_eval)
```

If you re-attempt to apply `jit`, you can inspect how the abstract evaluation proceeds, but you'll get another error about missing the actual XLA compilation rule:

```{code-cell}
:tags: [raises-exception]

with expectNotImplementedError():
  api.jit(square_add_prim)(2., 10.)
```

#### XLA Compilation rules

JAX compilation works by compiling each primitive into a graph of XLA operations.

This is the biggest hurdle to adding new functionality to JAX, because the  set of XLA operations is limited, and JAX already has pre-defined primitives for most of them. However, XLA includes a `CustomCall` operation that can be used to encapsulate arbitrary functionality defined using C++.

```{code-cell}
from jax._src.lib.mlir.dialects import hlo

@trace("multiply_add_lowering")
def multiply_add_lowering(ctx, xc, yc, zc):
  """The compilation to XLA of the primitive.

  Given an mlir.ir.Value for each argument, return the mlir.ir.Values for
  the results of the function.

  Does not need to be a JAX-traceable function.
  """
  return [hlo.AddOp(hlo.MulOp(xc, yc), zc).result]

# Now, register the lowering rule with JAX.
# For GPU, refer to the https://jax.readthedocs.io/en/latest/Custom_Operation_for_GPUs.html
from jax.interpreters import mlir

mlir.register_lowering(multiply_add_p, multiply_add_lowering, platform='cpu')
```

You will now succeed to apply `jax.jit`. Notice below that JAX first evaluates the function abstractly, which triggers the `multiply_add_abstract_eval` function, and  then compiles the set of primitives it has encountered, including `multiply_add`. At this point JAX invokes `multiply_add_xla_translation`.

```{code-cell}
assert api.jit(lambda x, y: square_add_prim(x, y))(2., 10.) == 14.
```

Below is another use of `jit`, where you compile only with respect to the first argument. Notice how the second argument to `square_add_prim` is concrete, which leads in the third argument to `multiply_add_abstract_eval` being `ConcreteArray`. Notice that `multiply_add_abstract_eval` may be used with both `ShapedArray` and `ConcreteArray`.

```{code-cell}
assert api.jit(lambda x, y: square_add_prim(x, y), 
               static_argnums=1)(2., 10.) == 14.
```

### Forward differentiation

JAX implements forward differentiation in the form of a Jacobian-Vector Product (JVP) (you can learn more about it in {ref}`advanced-autodiff`).

If you attempt to compute the `jvp` function, you'll get an error because you have not yet told JAX how to differentiate the `multiply_add` primitive.

```{code-cell}
:tags: [raises-exception]

# The second argument is set to `(2., 10.)` values where you
# evaluate the Jacobian, and the third argument `(1., 1.)`
# contains the values of the tangents for the arguments.
with expectNotImplementedError():
  api.jvp(square_add_prim, (2., 10.), (1., 1.))
```

```{code-cell}
from jax.interpreters import ad

@trace("multiply_add_value_and_jvp")
def multiply_add_value_and_jvp(arg_values, arg_tangents):
  """Evaluates the primal output and the tangents (Jacobian-vector product).

  Given values of the arguments and perturbation of the arguments (tangents), 
  compute the output of the primitive and the perturbation of the output.

  This method must be JAX-traceable. JAX may invoke it with abstract values 
  for the arguments and tangents.

  Args:
    arg_values: A tuple of arguments
    arg_tangents: A tuple with the tangents of the arguments. The tuple has 
      the same length as the arg_values. Some of the tangents may also be the 
      special value `ad.Zero` to specify a zero tangent

  Returns:
     A pair of the primal output and the tangent.
  """
  x, y, z = arg_values
  xt, yt, zt = arg_tangents
  _trace("Primal evaluation:")
  # Now, you have a JAX-traceable computation of the output. 
  # Normally, you can use the multiply add (`ma`) primitive itself to compute the primal output. 
  primal_out = multiply_add_prim(x, y, z)

  _trace("Tangent evaluation:")
  # You must use a JAX-traceable way to compute the tangent. It turns out that 
  # the output tangent can be computed as (xt * y + x * yt + zt),
  # which you can implement in a JAX-traceable way using the same "multiply_add_prim" primitive.

  # You do need to deal specially with `Zero`. Here, you just turn it into a 
  # proper tensor of 0s (of the same shape as 'x'). 
  # An alternative would be to check for `Zero` and perform algebraic 
  # simplification of the output tangent computation.
  def make_zero(tan):
    return lax.zeros_like_array(x) if type(tan) is ad.Zero else tan  

  output_tangent = multiply_add_prim(make_zero(xt), y, multiply_add_prim(x, make_zero(yt), make_zero(zt)))
  return (primal_out, output_tangent)

# Register the forward differentiation rule with JAX:
ad.primitive_jvps[multiply_add_p] = multiply_add_value_and_jvp
```

```{code-cell}
# Tangent is: xt*y + x*yt + zt = 1.*2. + 2.*1. + 1. = 5.
assert api.jvp(square_add_prim, (2., 10.), (1., 1.)) == (14., 5.)
```

#### JIT of forward differentiation

You can apply `jit` to the forward differentiation function:

```{code-cell}
assert api.jit(lambda arg_values, arg_tangents: 
                   api.jvp(square_add_prim, arg_values, arg_tangents))(
         (2., 10.), (1., 1.)) == (14., 5.)
```

Notice that first, you evaluate `multiply_add_value_and_jvp` abstractly, which in turn evaluates abstractly both the primal and the tangent evaluation (a total of 3 invocations of the `ma` primitive). Then, you compile the 3 occurrences of the primitive.


### Reverse differentiation

If you attempt now to use reverse differentiation, you'll notice that JAX starts by using the `multiply_add_value_and_jvp` to compute the forward differentiation for abstract values, but then runs into a `NotImplementedError`.

When computing the reverse differentiation, JAX first performs an abstract evaluation of the forward differentiation code `multiply_add_value_and_jvp` to obtain a  trace of primitives that compute the output tangent.

- Observe that JAX performs this abstract evaluation with concrete values for the differentiation point, and abstract values for the tangents.
- Notice that JAX uses the special abstract tangent value `Zero` for the tangent corresponding to the third argument of `ma`. This reflects the fact that you do not differentiate w.r.t. the second argument to `square_add_prim`, which flows to the third argument to `multiply_add_prim`.
- Notice also that during the abstract evaluation of the tangent you pass the value `0.0` as the tangent for the third argument. This is because of the use of the `make_zero` function in the definition of `multiply_add_value_and_jvp`.

```{code-cell}
:tags: [raises-exception]

# This is reverse differentiation w.r.t. the first argument of `square_add_prim`
with expectNotImplementedError():
  api.grad(square_add_prim)(2., 10.)
```

The above error is because there is a missing piece for JAX to be able to use the forward differentiation code to compute reverse differentiation.


#### Transposition

As previously explained, when computing reverse differentiation, JAX obtains a trace of primitives that compute the tangent using forward differentiation. Then, **JAX interprets this trace abstractly backwards** and for each primitive it applies a **transposition rule**.

To understand what is going on, consider a simpler example of the function `f(x, y) = x * y + y`. Assume, you need to differentiate at the point `(2., 4.)`. JAX will produce the following JVP tangent calculation of `ft` from the tangents of the input `xt` and `yt`:

```python
   a = xt * 4.
   b = 2. * yt
   c = a + b
   ft = c + yt
```

By construction, the tangent calculation is always linear in the input tangents. The only non-linear operator that may arise in the tangent calculation is multiplication, but then one of the operands is constant.

JAX will produce the reverse differentiation computation by processing the JVP computation backwards. For each operation in the tangent computation, it accumulates the cotangents of the variables used by the operation, using the cotangent of the result of the operation:

```python
  # Initialize cotangents of inputs and intermediate variables:
  xct = yct = act = bct = cct = 0.
  # Initialize cotangent of the output:
  fct = 1.
  # Process `ft = c + yt`:
  cct += fct
  yct += fct
  # Process `c = a + b`:
  act += cct
  bct += cct
  # Process `b = 2. * yt`:
  yct += 2. * bct
  # Process `a = xt * 4.`:
  xct += act * 4.
```

One can verify that this computation produces `xct = 4.` and `yct = 3.`, which 
are the partial derivatives of the function `f`. 

JAX knows for each primitive that may appear in a JVP calculation how to transpose it. Conceptually, if the primitive `p(x, y, z)` is linear in the arguments `y` and `z` for a constant value of `x`, e.g., `p(x, y, z) = y*cy + z*cz`, then the transposition of the primitive is:

```python
p_transpose(out_ct, x, _, _) = (None, out_ct*cy, out_ct*cz)
```

Notice that `p_transpose` takes the cotangent of the output of the primitive and a value corresponding to each argument of the primitive. For the linear arguments, the transposition gets an undefined `_` value, and for the other arguments it gets the actual constants. The transposition returns a cotangent value for each argument of the primitive, with the value `None` returned  for the constant arguments.

In particular:

```python
 add_transpose(out_ct, _, _) = (out_ct, out_ct)
 mult_transpose(out_ct, x, _) = (None, x * out_ct)
 mult_transpose(out_ct, _, y) = (out_ct * y, None)
```

```{code-cell}
@trace("multiply_add_transpose")
def multiply_add_transpose(ct, x, y, z):
  """Evaluates the transpose of a linear primitive.

  This method is only used when computing the backward gradient following 
  `value_and_jvp`, and is only needed for primitives that are used in the JVP 
  calculation for some other primitive. You need a transposition for `multiply_add_prim`, 
  because you have used `multiply_add_prim` in the computation of the `output_tangent` in 
  `multiply_add_value_and_jvp`.

  In this case, multiply_add is not a linear primitive. However, it is used linearly 
  w.r.t. tangents in `multiply_add_value_and_jvp`:
       `output_tangent(xt, yt, zt) = multiply_add_prim(xt, y, multiply_add_prim(x, yt, zt))`.

  Always one of the first two multiplicative arguments is a constant.

  Args:
      ct: The cotangent of the output of the primitive.
      x, y, z: The values of the arguments. The arguments that are used linearly
        get an ad.UndefinedPrimal value. The other arguments get a constant
        value.

  Returns:
      A tuple with the cotangent of the inputs, with the value None
      corresponding to the constant arguments.
  """
  if not ad.is_undefined_primal(x):
    # This use of multiply_add is with a constant "x".
    assert ad.is_undefined_primal(y)
    ct_y = ad.Zero(y.aval) if type(ct) is ad.Zero else multiply_add_prim(x, ct, lax.zeros_like_array(x))
    res = None, ct_y, ct
  else:
    # This use of multiply_add is with a constant "y".
    assert ad.is_undefined_primal(x)
    ct_x = ad.Zero(x.aval) if type(ct) is ad.Zero else multiply_add_prim(ct, y, lax.zeros_like_array(y))
    res = ct_x, None, ct
  return res

ad.primitive_transposes[multiply_add_p] = multiply_add_transpose
```

Now you can complete the run of the `grad`:

```{code-cell}
assert api.grad(square_add_prim)(2., 10.) == 4.
```

Notice the two calls to `multiply_add_transpose`. They correspond to the two uses of `multiply_add_prim` in the computation of the `output_tangent` in `multiply_add_value_and_jvp`. The first call to transpose corresponds to the last use of `multiply_add_prim`: `multiply_add_prim(xt, y, ...)` where `y` is the constant `2.0`.


#### JIT of reverse differentiation 

Notice that the abstract evaluation of the `multiply_add_value_and_jvp` is using only abstract values. Meanwhile, in the absence of JIT, you used `ConcreteArray`.

```{code-cell}
assert api.jit(api.grad(square_add_prim))(2., 10.) == 4.
```

### Batching

The batching transformation takes a point-wise computation and turns it into a computation on vectors. If you try it right now, you will get a `NotImplementedError`:

```{code-cell}
:tags: [raises-exception]

# The arguments are two vectors instead of two scalars.
with expectNotImplementedError():
  api.vmap(square_add_prim, in_axes=0, out_axes=0)(np.array([2., 3.]),
                                               np.array([10., 20.]))
```

You need to instruct JAX how to evaluate the batched version of the primitive. In this particular case, the `multiply_add_prim` already operates pointwise for any dimension of input vectors, so the batched version can use the same `multiply_add_prim` implementation.

```{code-cell}
from jax.interpreters import batching

@trace("multiply_add_batch")
def multiply_add_batch(vector_arg_values, batch_axes):
  """Computes the batched version of the primitive.
  
  This must be a JAX-traceable function.
  
  Since the `multiply_add primitive` already operates point-wise on arbitrary
  dimension tensors, to batch it you can use the primitive itself. This works as
  long as both the inputs have the same dimensions and are batched along the
  same axes. The result is batched along the axis that the inputs are batched.

  Args:
    vector_arg_values: A tuple of two arguments, each being a tensor of matching
      shape.
    batch_axes: The axes that are being batched. See vmap documentation.

  Returns:
    A tuple of the result, and the result axis that was batched. 
  """
  assert batch_axes[0] == batch_axes[1]
  assert batch_axes[0] == batch_axes[2]
  _trace("Using multiply_add to compute the batch:")
  res = multiply_add_prim(*vector_arg_values)
  return res, batch_axes[0]


batching.primitive_batchers[multiply_add_p] = multiply_add_batch
```

```{code-cell}
assert np.allclose(api.vmap(square_add_prim, in_axes=0, out_axes=0)(
  np.array([2., 3.]),
  np.array([10., 20.])),
  [14., 29.])
```

#### JIT of batching

Below is an example of applying JIT to batching:

```{code-cell}
assert np.allclose(api.jit(api.vmap(square_add_prim, in_axes=0, out_axes=0))
                    (np.array([2., 3.]),
                     np.array([10., 20.])),
                    [14., 29.])
```

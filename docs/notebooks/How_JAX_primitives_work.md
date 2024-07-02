---
jupytext:
  formats: ipynb,md:myst
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.16.1
kernelspec:
  display_name: Python 3
  name: python3
---

+++ {"id": "vfxqky4PCUnh"}

# How JAX primitives work

<!--* freshness: { reviewed: '2024-04-08' } *-->

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/google/jax/blob/main/docs/notebooks/How_JAX_primitives_work.ipynb) [![Open in Kaggle](https://kaggle.com/static/images/open-in-kaggle.svg)](https://kaggle.com/kernels/welcome?src=https://github.com/google/jax/blob/main/docs/notebooks/How_JAX_primitives_work.ipynb)

*necula@google.com*, October 2019.

JAX implements certain transformations of Python functions, e.g., `jit`, `grad`,
`vmap`, or `pmap`. The Python functions to be transformed must be JAX-traceable, 
which means that as the Python function executes
the only operations it applies to the data are either inspections of data
attributes such as shape or type, or special operations called JAX primitives.
In particular, a JAX-traceable function is sometimes invoked by JAX with
abstract arguments. An example of a JAX abstract value is `ShapedArray(float32[2,2])`, 
which captures the type and the shape of values, but not the concrete data values.
JAX primitives know how to operate on both concrete data
values and on the JAX abstract values.


The JAX-transformed functions must themselves be JAX-traceable functions,
to ensure that these transformations
can be composed, e.g., `jit(jacfwd(grad(f)))`.

There are pre-defined JAX primitives corresponding to most XLA operations, 
e.g., add, matmul, sin, cos, indexing.
JAX comes with an implementation of numpy functions in terms of JAX primitives, which means that Python programs
using JAX’s implementation of numpy are JAX-traceable and therefore transformable.
Other libraries can be made JAX-traceable by implementing them in terms of JAX primitives.

The set of JAX primitives is extensible. Instead of reimplementing a function in terms of pre-defined JAX primitives,
one can define a new primitive that encapsulates the behavior of the function.

**The goal of this document is to explain the interface that a JAX primitive must support in order to allow JAX to perform all its transformations.**

Consider that we want to add to JAX support for a multiply-add function with three arguments, defined mathematically
as "multiply_add(x, y, z) = x * y + z". 
This function operates on 3 identically-shaped tensors of floating point 
values and performs the operations pointwise.

+++ {"id": "HIJYIHNTD1yI"}

## Using existing primitives

The easiest way to define new functions is to write them in terms of JAX primitives, or in terms of other
functions that are themselves written using JAX primitives, e.g., those 
defined in the `jax.lax` module:

```{code-cell} ipython3
:id: tbOF0LB0EMne
:outputId: 3fb1c8a7-7a4c-4a3a-f7ff-37b7dc740528

from jax import lax
from jax._src import api

def multiply_add_lax(x, y, z):
  """Implementation of multiply-add using the jax.lax primitives."""
  return lax.add(lax.mul(x, y), z)


def square_add_lax(a, b):
  """A square-add function using the newly defined multiply-add."""
  return multiply_add_lax(a, a, b)

print("square_add_lax = ", square_add_lax(2., 10.))
# Differentiate w.r.t. the first argument
print("grad(square_add_lax) = ", api.grad(square_add_lax, argnums=0)(2.0, 10.))
```

+++ {"id": "Cgv60Wm3E_D5"}

In order to understand how JAX is internally using the primitives,
we add some helpers for tracing function calls.

```{code-cell} ipython3
:cellView: form
:id: mQRQGEGiE53K

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

+++ {"id": "Qf4eLrLCFYDl"}

Instead of using `jax.lax` primitives directly, we can use other functions 
that are already written in terms of those primitives, such as those in `jax.numpy`:

```{code-cell} ipython3
:id: QhKorz6cFRJb
:outputId: aba3cef3-6bcc-4eb3-c7b3-34e405f2f82a

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

+++ {"id": "Sg-D8EdeFn4a"}

Notice that in the process of computing `grad`, JAX invokes `square_add_numpy` and
`multiply_add_numpy` with special arguments `ConcreteArray(...)` (described further 
below in this colab). 
It is important to remember that a JAX-traceable function must be able to 
operate not only on concrete arguments but also on special abstract arguments
that JAX may use to abstract the function execution.

The JAX traceability property is satisfied as long as the function is written 
in terms of JAX primitives.

+++ {"id": "WxrQO7-XGLcg"}

## Defining new JAX primitives

The right way to add support for multiply-add is in terms of existing
JAX primitives, as shown above. However, in order to demonstrate how JAX
primitives work let us pretend that we want to add a new primitive to 
JAX for the multiply-add functionality.

```{code-cell} ipython3
:id: cPqAH1XOGTN4

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

+++ {"id": "LMzs5PAKGr-4"}

If we try to call the newly defined functions we get an error, because
we have not yet told JAX anything about the semantics of the new primitive.

```{code-cell} ipython3
:id: _X3PAYxhGpWd
:outputId: 90ea2c6a-9ef3-40ea-e9a3-3ab1cfc59fc8

with expectNotImplementedError():
  square_add_prim(2., 10.)
```

+++ {"id": "elha0FdgHSEF"}

### Primal evaluation rules

```{code-cell} ipython3
:id: FT34FFAGHARU
:outputId: 4c54f1c2-8a50-4788-90e1-06aee412c43b

@trace("multiply_add_impl")
def multiply_add_impl(x, y, z):
  """Concrete implementation of the primitive.

  This function does not need to be JAX traceable.
  Args:
    x, y, z: the concrete arguments of the primitive. Will only be called with 
      concrete values.
  Returns:
    the concrete result of the primitive.
  """
  # Note that we can use the original numpy, which is not JAX traceable
  return np.add(np.multiply(x, y), z)

# Now we register the primal implementation with JAX
multiply_add_p.def_impl(multiply_add_impl)
```

```{code-cell} ipython3
:id: G5bstKaeNAVV
:outputId: deb94d5b-dfea-4e6f-9ec2-70b416c996c5

assert square_add_prim(2., 10.) == 14.
```

+++ {"id": "upBf-uAuHhPJ"}

### JIT

If we now try to use `jit` we get a `NotImplementedError`:

```{code-cell} ipython3
:id: QG-LULjiHk4b
:outputId: d4ef4406-8dae-4c96-97ca-b662340474ee

with expectNotImplementedError():
  api.jit(square_add_prim)(2., 10.)
```

+++ {"id": "rHS1bAGHH44E"}

#### Abstract evaluation rules
In order to JIT the function, and for other transformations as well, 
JAX first evaluates it abstractly using only the 
shape and type of the arguments. This abstract evaluation serves multiple
purposes:

  * Gets the sequence of JAX primitives that are used in the computation. This 
  sequence will be compiled. 
  * Computes the shape and type of all vectors and operations used in the computation. 


For example, the abstraction of a vector with 3 elements may be `ShapedArray(float32[3])`, or `ConcreteArray([1., 2., 3.])`. 
In the latter case, JAX uses the actual concrete value wrapped as an abstract value.

```{code-cell} ipython3
:id: ctQmEeckIbdo
:outputId: e751d0cc-460e-4ffd-df2e-fdabf9cffdc2

from jax import core
@trace("multiply_add_abstract_eval")
def multiply_add_abstract_eval(xs, ys, zs):
  """Abstract evaluation of the primitive.

  This function does not need to be JAX traceable. It will be invoked with
  abstractions of the actual arguments. 
  Args:
    xs, ys, zs: abstractions of the arguments.
  Result:
    a ShapedArray for the result of the primitive.
  """
  assert xs.shape == ys.shape
  assert xs.shape == zs.shape
  return core.ShapedArray(xs.shape, xs.dtype)

# Now we register the abstract evaluation with JAX
multiply_add_p.def_abstract_eval(multiply_add_abstract_eval)
```

+++ {"id": "RPN88X6YI43A"}

If we re-attempt to JIT, we see how the abstract evaluation proceeds, but
we get another error, about missing the actual XLA compilation rule:

```{code-cell} ipython3
:id: eOcNR92SI2h-
:outputId: 356ef229-3703-4696-cc3d-7c05de405fb0

with expectNotImplementedError():
  api.jit(square_add_prim)(2., 10.)
```

+++ {"id": "9IOV1R-fJMHp"}

#### XLA Compilation rules

JAX compilation works by compiling each primitive into a graph of XLA operations.

This is the biggest hurdle to adding new functionality to JAX, because the 
set of XLA operations is limited, and JAX already has pre-defined primitives
for most of them. However, XLA includes a `CustomCall` operation that can be used to encapsulate arbitrary functionality defined using C++.

```{code-cell} ipython3
:id: FYQWSSjKJaWP

from jax._src.lib.mlir.dialects import hlo
@trace("multiply_add_lowering")
def multiply_add_lowering(ctx, xc, yc, zc):
  """The compilation to XLA of the primitive.

  Given an mlir.ir.Value for each argument, return the mlir.ir.Values for
  the results of the function.

  Does not need to be a JAX-traceable function.
  """
  return [hlo.AddOp(hlo.MulOp(xc, yc), zc).result]

# Now we register the lowering rule with JAX
# For GPU see the [Custom operations for GPUs](https://jax.readthedocs.io/en/latest/Custom_Operation_for_GPUs.html)
# TODO: TPU?
from jax.interpreters import mlir
mlir.register_lowering(multiply_add_p, multiply_add_lowering, platform='cpu')
```

+++ {"id": "K98LX-VaJkFu"}

Now we succeed to JIT. Notice below that JAX first evaluates the function
abstractly, which triggers the `multiply_add_abstract_eval` function, and 
then compiles the set of primitives it has encountered, including `multiply_add`.
At this point JAX invokes `multiply_add_xla_translation`.

```{code-cell} ipython3
:id: rj3TLsolJgEc
:outputId: e384bee4-1e9c-4344-f49c-d3b5ec08eb32

assert api.jit(lambda x, y: square_add_prim(x, y))(2., 10.) == 14.
```

+++ {"id": "Omrez-2_KFfo"}

Below is another use of `jit` where we compile only
with respect to the first argument. Notice how the second argument to `square_add_prim` is concrete, which leads
in the third argument to `multiply_add_abstract_eval` being 
`ConcreteArray`. We see that `multiply_add_abstract_eval` may be used with
both `ShapedArray` and `ConcreteArray`.

```{code-cell} ipython3
:id: mPfTwIBoKOEK
:outputId: b293b9b6-a2f9-48f5-f7eb-d4f99c3d905b

assert api.jit(lambda x, y: square_add_prim(x, y), 
               static_argnums=1)(2., 10.) == 14.
```

+++ {"id": "_Ya3B5l4J1VA"}

### Forward differentiation

JAX implements forward differentiation in the form of
a Jacobian-vector product (see the [JAX autodiff cookbook](https://jax.readthedocs.io/en/latest/notebooks/autodiff_cookbook.html#Jacobian-Matrix-and-Matrix-Jacobian-products)).

If we attempt now to compute the `jvp` function we get an
error because we have not yet told JAX how to differentiate
the `multiply_add` primitive.

```{code-cell} ipython3
:id: OxDx6NQnKwMI
:outputId: ce659ef3-c03c-4856-f252-49ec4b6eb964

# The second argument `(2., 10.)` are the argument values
# where we evaluate the Jacobian, and the third `(1., 1.)`
# are the values of the tangents for the arguments.
with expectNotImplementedError():
  api.jvp(square_add_prim, (2., 10.), (1., 1.))
```

```{code-cell} ipython3
:id: zxG24C1JMIMM

from jax.interpreters import ad


@trace("multiply_add_value_and_jvp")
def multiply_add_value_and_jvp(arg_values, arg_tangents):
  """Evaluates the primal output and the tangents (Jacobian-vector product).

  Given values of the arguments and perturbation of the arguments (tangents), 
  compute the output of the primitive and the perturbation of the output.

  This method must be JAX-traceable. JAX may invoke it with abstract values 
  for the arguments and tangents.

  Args:
    arg_values: a tuple of arguments
    arg_tangents: a tuple with the tangents of the arguments. The tuple has 
      the same length as the arg_values. Some of the tangents may also be the 
      special value ad.Zero to specify a zero tangent.
  Returns:
     a pair of the primal output and the tangent.
  """
  x, y, z = arg_values
  xt, yt, zt = arg_tangents
  _trace("Primal evaluation:")
  # Now we have a JAX-traceable computation of the output. 
  # Normally, we can use the ma primitive itself to compute the primal output. 
  primal_out = multiply_add_prim(x, y, z)
  
  _trace("Tangent evaluation:")
  # We must use a JAX-traceable way to compute the tangent. It turns out that 
  # the output tangent can be computed as (xt * y + x * yt + zt),
  # which we can implement in a JAX-traceable way using the same "multiply_add_prim" primitive.
  
  # We do need to deal specially with Zero. Here we just turn it into a 
  # proper tensor of 0s (of the same shape as 'x'). 
  # An alternative would be to check for Zero and perform algebraic 
  # simplification of the output tangent computation.
  def make_zero(tan):
    return lax.zeros_like_array(x) if type(tan) is ad.Zero else tan  
  
  output_tangent = multiply_add_prim(make_zero(xt), y, multiply_add_prim(x, make_zero(yt), make_zero(zt)))
  return (primal_out, output_tangent)

# Register the forward differentiation rule with JAX 
ad.primitive_jvps[multiply_add_p] = multiply_add_value_and_jvp
```

```{code-cell} ipython3
:id: ma3KBkiAMfW1
:outputId: f34cbbc6-20d9-48ca-9a9a-b5d91a972cdd

# Tangent is: xt*y + x*yt + zt = 1.*2. + 2.*1. + 1. = 5.
assert api.jvp(square_add_prim, (2., 10.), (1., 1.)) == (14., 5.)
```

+++ {"id": "69QsEcu-lP4u"}

TO EXPLAIN: 

  * Why is JAX using ConcreteArray in square_add_prim? There is no abstract evaluation going on here.
  * Not sure how to explain that multiply_add_prim is invoked with ConcreteValue, yet
  we do not call the multiply_add_abstract_eval.
  * I think it would be useful to show the jaxpr here

+++ {"id": "Sb6e3ZAHOPHv"}

#### JIT of forward differentiation

We can apply JIT to the forward differentiation function:

```{code-cell} ipython3
:id: hg-hzVu-N-hv
:outputId: 38d32067-e152-4046-ad80-7f95a31ba628

assert api.jit(lambda arg_values, arg_tangents: 
                   api.jvp(square_add_prim, arg_values, arg_tangents))(
         (2., 10.), (1., 1.)) == (14., 5.)
```

+++ {"id": "jlZt1_v2mU88"}

Notice that first we evaluate `multiply_add_value_and_jvp` abstractly, which in turn
evaluates abstractly both the primal and the tangent evaluation (a total of 
3 invocations of the `ma` primitive). Then we compile the 3 occurrences
of the primitive.

+++ {"id": "555yt6ZIOePB"}

### Reverse differentiation

If we attempt now to use reverse differentiation we
see that JAX starts by using the `multiply_add_value_and_jvp` to 
compute the forward differentiation for abstract values, but then runs
into a `NotImplementedError`. 

When computing the reverse differentiation JAX first does abstract evaluation
of the forward differentiation code `multiply_add_value_and_jvp` to obtain a 
trace of primitives that compute the output tangent. 
Observe that JAX performs this abstract evaluation with concrete values
for the differentiation point, and abstract values for the tangents. 
Observe also that JAX uses the special abstract tangent value `Zero` for
the tangent corresponding to the 3rd argument of `ma`. This reflects the 
fact that we do not differentiate w.r.t. the 2nd argument to `square_add_prim`,
which flows to the 3rd argument to `multiply_add_prim`.

Observe also that during the abstract evaluation of the tangent we pass the 
value 0.0 as the tangent for the 3rd argument. This is due to the use
of the `make_zero` function in the definition of `multiply_add_value_and_jvp`.

```{code-cell} ipython3
:id: 8eAVnexaOjBn
:outputId: e4ee89cf-ab4a-4505-9817-fa978a2865ab

# This is reverse differentiation w.r.t. the first argument of square_add_prim
with expectNotImplementedError():
  api.grad(square_add_prim)(2., 10.)
```

+++ {"id": "fSHLUMDN26AY"}

The above error is because there is a missing piece for JAX to be able
to use the forward differentiation code to compute reverse differentiation.

+++ {"id": "3ibDbGF-PjK9"}

#### Transposition


As explained above, when computing reverse differentiation JAX obtains
a trace of primitives that compute the tangent using forward differentiation.
Then, **JAX interprets this trace abstractly backwards** and for each 
primitive it applies a **transposition** rule.

To understand what is going on, consider for now a simpler example of the function "f(x, y) = x * y + y". Assume we need to differentiate at the point `(2., 4.)`. JAX will produce the following JVP tangent calculation of `ft` from the tangents of the input `xt` and `yt`:
```
   a = xt * 4.
   b = 2. * yt
   c = a + b
   ft = c + yt
```

By construction, the tangent calculation is always linear in the input tangents. 
The only non-linear operator that may arise in the tangent calculation is multiplication,
but then one of the operands is constant.

JAX will produce the reverse differentiation computation by processing the
JVP computation backwards. For each operation in the tangent computation,
it accumulates the cotangents
of the variables used by the operation, using the cotangent of the result
of the operation:
```
  # Initialize cotangents of inputs and intermediate vars
  xct = yct = act = bct = cct = 0.
  # Initialize cotangent of the output
  fct = 1.
  # Process "ft = c + yt"
  cct += fct
  yct += fct
  # Process "c = a + b"
  act += cct
  bct += cct
  # Process "b = 2. * yt"
  yct += 2. * bct
  # Process "a = xt * 4."
  xct += act * 4.
```

One can verify that this computation produces `xct = 4.` and `yct = 3.`, which 
are the partial derivatives of the function `f`. 

JAX knows for each primitive that may appear in a JVP calculation how to transpose it. Conceptually, if the primitive `p(x, y, z)` is linear in the arguments `y` and `z` for a constant value of `x`, e.g., `p(x, y, z) = y*cy + z*cz`, then the transposition of the primitive is:
```
p_transpose(out_ct, x, _, _) = (None, out_ct*cy, out_ct*cz)
```

Notice that `p_transpose` takes the cotangent of the output of the primitive and a value corresponding to each argument of the primitive. For the linear arguments, the transposition gets an undefined `_` value, and for the other
arguments it gets the actual constants. The transposition returns a cotangent value for each argument of the primitive, with the value `None` returned 
for the constant arguments.

In particular, 
```
 add_transpose(out_ct, _, _) = (out_ct, out_ct)
 mult_transpose(out_ct, x, _) = (None, x * out_ct)
 mult_transpose(out_ct, _, y) = (out_ct * y, None)
```

```{code-cell} ipython3
:id: JaHxFdkRO42r

@trace("multiply_add_transpose")
def multiply_add_transpose(ct, x, y, z):
  """Evaluates the transpose of a linear primitive.

  This method is only used when computing the backward gradient following 
  value_and_jvp, and is only needed for primitives that are used in the JVP 
  calculation for some other primitive. We need transposition for multiply_add_prim, 
  because we have used multiply_add_prim in the computation of the output_tangent in 
  multiply_add_value_and_jvp.

  In our case, multiply_add is not a linear primitive. However, it is used linearly 
  w.r.t. tangents in multiply_add_value_and_jvp:
       output_tangent(xt, yt, zt) = multiply_add_prim(xt, y, multiply_add_prim(x, yt, zt))
  
  Always one of the first two multiplicative arguments is a constant.

  Args:
      ct: the cotangent of the output of the primitive.
      x, y, z: values of the arguments. The arguments that are used linearly
        get an ad.UndefinedPrimal value. The other arguments get a constant
        value.
  Returns:
      a tuple with the cotangent of the inputs, with the value None
      corresponding to the constant arguments.
  """
  if not ad.is_undefined_primal(x):
    # This use of multiply_add is with a constant "x"
    assert ad.is_undefined_primal(y)
    ct_y = ad.Zero(y.aval) if type(ct) is ad.Zero else multiply_add_prim(x, ct, lax.zeros_like_array(x))
    res = None, ct_y, ct
  else:
    # This use of multiply_add is with a constant "y"
    assert ad.is_undefined_primal(x)
    ct_x = ad.Zero(x.aval) if type(ct) is ad.Zero else multiply_add_prim(ct, y, lax.zeros_like_array(y))
    res = ct_x, None, ct
  return res


ad.primitive_transposes[multiply_add_p] = multiply_add_transpose
```

+++ {"id": "PpChox-Jp7wb"}

Now we can complete the run of the `grad`:

```{code-cell} ipython3
:id: PogPKS4MPevd
:outputId: d33328d4-3e87-45b5-9b31-21ad624b67af

assert api.grad(square_add_prim)(2., 10.) == 4.
```

+++ {"id": "8M1xLCXW4fK7"}

Notice the two calls to `multiply_add_transpose`. They correspond to the two
uses of `multiply_add_prim` in the computation of the `output_tangent` in `multiply_add_value_and_jvp`. The first call to transpose corresponds to the 
last use of `multiply_add_prim`: `multiply_add_prim(xt, y, ...)` where `y` is the constant 2.0.

+++ {"id": "EIJs6FYmPg6c"}

#### JIT of reverse differentiation 

Notice that the abstract evaluation of the `multiply_add_value_and_jvp` is using only
abstract values, while in the absence of JIT we used `ConcreteArray`.

```{code-cell} ipython3
:id: FZ-JGbWZPq2-
:outputId: e42b5222-9c3e-4853-e13a-874f6605d178

assert api.jit(api.grad(square_add_prim))(2., 10.) == 4.
```

+++ {"id": "-3lqPkdQPvl5"}

### Batching

The batching transformation takes a point-wise computation and turns it
into a computation on vectors. If we try it right now, we get a `NotImplementedError`:

```{code-cell} ipython3
:id: hFvBR3I9Pzh3
:outputId: 434608bc-281f-4d3b-83bd-eaaf3b51b1cd

# The arguments are two vectors instead of two scalars
with expectNotImplementedError():
  api.vmap(square_add_prim, in_axes=0, out_axes=0)(np.array([2., 3.]),
                                               np.array([10., 20.]))
```

+++ {"id": "gILasMiP6elR"}

We need to tell JAX how to evaluate the batched version of the primitive. In this particular case, the `multiply_add_prim` already operates pointwise for any dimension of input vectors. So the batched version can use the same `multiply_add_prim` implementation.

```{code-cell} ipython3
:id: KQfeqRIrP7zg

from jax.interpreters import batching


@trace("multiply_add_batch")
def multiply_add_batch(vector_arg_values, batch_axes):
  """Computes the batched version of the primitive.
  
  This must be a JAX-traceable function.
  
  Since the multiply_add primitive already operates pointwise on arbitrary
  dimension tensors, to batch it we can use the primitive itself. This works as
  long as both the inputs have the same dimensions and are batched along the
  same axes. The result is batched along the axis that the inputs are batched.
  
  Args:
    vector_arg_values: a tuple of two arguments, each being a tensor of matching
      shape.
    batch_axes: the axes that are being batched. See vmap documentation.
  Returns:
    a tuple of the result, and the result axis that was batched. 
  """
  assert batch_axes[0] == batch_axes[1]
  assert batch_axes[0] == batch_axes[2]
  _trace("Using multiply_add to compute the batch:")
  res = multiply_add_prim(*vector_arg_values)
  return res, batch_axes[0]


batching.primitive_batchers[multiply_add_p] = multiply_add_batch
```

```{code-cell} ipython3
:id: VwxNk869P_YG
:outputId: 9d22c921-5803-4d33-9e88-b6e439ba9738

assert np.allclose(api.vmap(square_add_prim, in_axes=0, out_axes=0)(
  np.array([2., 3.]),
  np.array([10., 20.])),
  [14., 29.])
```

+++ {"id": "NmqLlV1TQDCC"}

#### JIT of batching

```{code-cell} ipython3
:id: xqEdXVUgQCTt
:outputId: 9c22fd9c-919c-491d-bbeb-32c241b808fa

assert np.allclose(api.jit(api.vmap(square_add_prim, in_axes=0, out_axes=0))
                    (np.array([2., 3.]),
                     np.array([10., 20.])),
                    [14., 29.])
```

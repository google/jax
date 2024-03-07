# JAX and TensorFlow interoperation (jax2tf/call_tf)

<!-- Next line must match the copybara config. -->
<!-- Link to internal documentation. -->

This package provides support for JAX native serialization and for interoperation
between JAX and TensorFlow.
There are two interoperation directions:

- `jax2tf.convert`: for calling JAX functions in a TensorFlow context, e.g.,
for eager or graph TensorFlow execution,
or for serializing as a TensorFlow SavedModel; and
- `jax2tf.call_tf`: for calling TensorFlow functions in a JAX context, e.g.,
to call a TensorFlow library or to reload a TensorFlow SavedModel and call
its functions in JAX.

These APIs can be combined, e.g., to reload in JAX a program that
has been serialized from JAX to a TensorFlow SavedModel, or to save to
TensorFlow SavedModel a JAX program that uses a TensorFlow library.

Tip: As of version 0.4.14 (July 2023) the default mode of JAX-TensorFlow
interoperation is by way of **native serialization** in which the target
function is lowered to StableHLO
using standard native JAX or TensorFlow APIs, and then the StableHLO module
is invoked from the other framework.
The native serialization mode has several advantages:

   * supports virtually all operations supported by native execution, e.g.,
     `xmap`, `shard_map`, `pmap`, parallel collective operations, and all
     primitives at all data types.
   * uses standard native JAX code paths for lowering, and thus it is easier
     to trust that the semantics and performance stays faithful to the native
     semantics, across platforms.
   * the metadata associated with the operations, e.g., source location, is
     identical to what native execution uses.
   * includes safety checking that the serialized code is executed on
     the platform for which it was serialized.

At the moment when using JAX native serialization the whole
JAX compilation unit is wrapped with a single thin TensorFlow op,
called [`XlaCallModule`](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/compiler/tf2xla/ops/xla_ops.cc#L1318),
that carries the serialized version of the StableHLO obtained from JAX. This
op is supported only on TensorFlow platforms that include the XLA compiler, and
it compiles and then invokes the embedded StableHLO.
The reasons we wrap the StableHLO in a TensorFlow op are:

  * it allows saving the serialization in a tf.SavedModel, for use with
    multiple mature tools for TensorFlow,
  * it allows composing the JAX program with TensorFlow pre-processing,
    post-processing, and host callback functions,
  * the `XlaCallModule` contains the code that must be executed
    to deserialize, compile, and execute the JAX program, e.g., to
    handle properly backward compatibility and to
    do the just-in-time preprocessing needed for shape polymorphism.
  * the semantics of JAX program is still preserved faithfully because it
    is entirely captured by the StableHLO serialization.

For backwards compatibility purposes, and for special uses,
the JAX-TensorFlow interoperation APIs can be used also
in a **graph serialization** mode (the only mode available before version 0.4.7,
and the default mode before JAX version 0.4.15),
without going through StableHLO.

  * For calling JAX functions from TensorFlow,
    it is possible to request that the JAX function be lowered with one TensorFlow
    op for each JAX primitive.
    This can be achieved by setting `native_serialization=False`.
    This enables the following:

       * TensorFlow eager mode execution, e.g., for debugging,
       * producing a `tf.Graph` for consumption by tooling that understands
         TensorFlow ops but does not yet work with StableHLO,
         e.g., TFLite and TensorFlow.js.
       * using the more mature support for dynamic shapes in TensorFlow.
         [StableHLO does have support for dynamic
         shapes](https://github.com/openxla/stablehlo/blob/main/rfcs/20230704-dynamism-101.md),
         and in the near future we expect it will support shape polymorphism
         to the same extent as graph serialization.

    Even in the graph serialization mode the resulting TensorFlow graph
    is pretty much 1:1 with the StableHLO module
    that would be obtained through native serialization.

  * For calling TensorFlow functions from JAX, if the resulting JAX program
    is executed in op-by-op mode (i.e., not under `jax.jit` or `jax.pmap`
    and not inside `lax.cond` or `lax.scan`)
    then the target TensorFlow function is executed in eager mode. This can
    be useful if the target TensorFlow function is not lowerable to HLO, e.g.,
    is using strings.

To disable native serialization, you can do the following, in decreasing
priority order:

  * set `native_serialization=False`, or
  * use the configuration flag `--jax2tf_default_native_serialization=false`, or
  * use the environment variable `JAX2TF_DEFAULT_NATIVE_SERIALIZATION=false`.

We describe below some general concepts and capabilities, first for
`jax2tf.convert` and [later](#calling-tensorflow-functions-from-jax)
for `jax2tf.call_tf`.
For more involved examples, please see examples involving:

   * SavedModel for archival ([examples below](#usage-saved-model)), including
     saving [batch-polymorphic functions](#shape-polymorphic-conversion),
   * TensorFlow Lite ([examples](https://github.com/google/jax/blob/main/jax/experimental/jax2tf/examples/tflite/mnist/README.md)),
   * TensorFlow.js ([examples](https://github.com/google/jax/blob/main/jax/experimental/jax2tf/examples/tf_js/quickdraw/README.md)),
   * TFX ([examples](https://github.com/tensorflow/tfx/blob/master/tfx/examples/penguin/README.md#instructions-for-using-flax)),
   * TensorFlow Hub and Keras ([examples](https://github.com/google/jax/blob/main/jax/experimental/jax2tf/examples/README.md)).

[TOC]

## Usage: basic functions.

As a rule of thumb, if you can `jax.jit` your function then you should be able
to use `jax2tf.convert`:

```python
from jax.experimental import jax2tf
from jax import numpy as jnp

import numpy as np
import tensorflow as tf

def f_jax(x):
  return jnp.sin(jnp.cos(x))

# jax2tf.convert is a higher-order function that returns a wrapped function with
# the same signature as your input function but accepting TensorFlow tensors (or
# variables) as input.
f_tf = jax2tf.convert(f_jax)

# For example you execute f_tf eagerly with valid TensorFlow inputs:
f_tf(np.random.random(...))

# Additionally you can use tools like `tf.function` to improve the execution
# time of your function, or to stage it out to a SavedModel:
f_tf_graph = tf.function(f_tf, autograph=False)
```

Note that when using the default native serialization, the target JAX function
must be jittable (see [JAX - The Sharp Bits](https://jax.readthedocs.io/en/latest/notebooks/Common_Gotchas_in_JAX.html)).
In the native serialization mode, under TensorFlow eager
the whole JAX function executes as one op.

The Autograph feature of `tf.function` cannot be expected to work on
functions lowered from JAX as above, so it is recommended to
set `autograph=False` in order to speed up the execution
and to avoid warnings and outright errors.


## Usage: saved model

You can serialize JAX program into a TensorFlow SavedModel, for use
with tooling that understands SavedModel. Both in native and non-native
serialization you can count on 6 months of backwards compatibility (you
can load a function serialized today with tooling that will be built
up to 6 months in the future), and 3 weeks of limited forwards compatibility
(you can load a function serialized today with tooling that was built
up to 3 weeks in the past, provided the model that not use any
new features).

Since jax2tf provides a regular TensorFlow function using it with SavedModel
is trivial:

```python
# You can save the model just like you would with any other TensorFlow function:
my_model = tf.Module()
# Save a function that can take scalar inputs.
my_model.f = tf.function(jax2tf.convert(f_jax), autograph=False,
                         input_signature=[tf.TensorSpec([], tf.float32)])
tf.saved_model.save(my_model, '/some/directory',
                    options=tf.saved_model.SaveOptions(experimental_custom_gradients=True))

# Restoring (note: the restored model does *not* require JAX to run, just XLA).
restored_model = tf.saved_model.load('/some/directory')
```

An important point is that in the above code snippet **everything after the
jax2tf invocation is standard TensorFlow code.
In particular, the saving of the model is not directly part
of the jax2tf API, and the user has full control over how to create the SavedModel**.

For example, just like for regular TensorFlow functions, it is possible to include in the
SavedModel multiple versions of a function for different input shapes, by
"warming up" the function on different input shapes:

```python
my_model.f = tf.function(jax2tf.convert(f_jax), autograph=False)
my_model.f(tf.ones([1, 28, 28]))  # a batch size of 1
my_model.f(tf.ones([16, 28, 28]))  # a batch size of 16
tf.saved_model.save(my_model, '/some/directory',
                    options=tf.saved_model.SaveOptions(experimental_custom_gradients=True))
```

### Saved model with parameters

Some special care is needed to ensure that the model parameters are not embedded
as constants in the graph and are instead saved separately as variables.
This is useful for two reasons:
the parameters could be very large and exceed the 2GB limits of the
GraphDef part of the SavedModel, or you may want to fine-tune the
model and change the value of the parameters.

For example, consider the following function:

```python
def model_jax(inputs):
  return param0 + param1 * inputs
```

If you just lower and save the model directly, the values of
`param0` and `param1` will be embedded in the computation graph. In fact, the
value of `param1` is needed for the gradient computation and
will be embedded twice: once in the computation
graph for the forward computation and once for the backward computation,
unless you turn off the staging of gradients or their saving as discussed
further below (e.g., `with_gradient=False`). Note also that if one
views the above function as an ML model parameterized by `param0` and `param1`
then the gradient function will be w.r.t. the inputs, while you probably
want gradients w.r.t. the parameters.

A better way to deal with parameters (or any large constants) is to
pass them as parameters to the function to be lowered:

```python
def model_jax(params, inputs):
  return params[0] + params[1] * inputs

# Wrap the parameter constants as tf.Variables; this will signal to the model
# saving code to save those constants as variables, separate from the
# computation graph.
params_vars = tf.nest.map_structure(tf.Variable, params)

# Build the prediction function by closing over the `params_vars`. If you
# instead were to close over `params` your SavedModel would have no variables
# and the parameters will be included in the function graph.
prediction_tf = lambda inputs: jax2tf.convert(model_jax)(params_vars, inputs)

my_model = tf.Module()
# Tell the model saver what are the variables.
my_model._variables = tf.nest.flatten(params_vars)
my_model.f = tf.function(prediction_tf, jit_compile=True, autograph=False)
tf.saved_model.save(my_model)
```

This strategy will avoid any copies of the large parameters in the computation
graph (they will be saved in a `variables` area of the model, which is not
subject to the 2GB limitation).

For examples of how to save a Flax model as a SavedModel see the
[examples directory](https://github.com/google/jax/blob/main/jax/experimental/jax2tf/examples/README.md).

### Saved model and differentiation

The code lowered from JAX supports differentiation from TensorFlow. In order to
ensure that the result of TensorFlow differentiation is identical to the
one that JAX differentiation would produce, we will
annotate the lowered primal function with a ``tf.custom_gradient`` that,
upon TensorFlow differentiation, will lazily
call into JAX to compute the ``jax.vjp`` of the lowered primal function, followed by
jax2tf lowering of the gradient function.
This ensures that ultimately it is JAX that performs the
differentiation, thus respecting any custom gradients that may be present
in the original function.

The `jax2tf.convert` function has an option ``with_gradient=False`` to skip the
custom gradients and wrap instead the lowered function with
``tf.raw_ops.PreventGradient`` to generate an error in case a gradient
computation is attempted.

SavedModels enables saving custom derivative rules by using the `experimental_custom_gradients` option:

```python
options = tf.saved_model.SaveOptions(experimental_custom_gradients=True)
tf.saved_model.save(model, path, options=options)
```

If you use `with_gradient=True` and forget to use the `experimental_custom_gradients=True` parameter
to `tf.saved_model.save` when you later load the saved model you will see a warning:

```
WARNING:absl:Importing a function (__inference_converted_fun_25) with ops with unsaved custom gradients. Will likely fail if a gradient is requested.
```

and if you do attempt to take a gradient of the loaded model you may get an error:

```
TypeError: An op outside of the function building code is being passed
a "Graph" tensor. It is possible to have Graph tensors
leak out of the function building context by including a
tf.init_scope in your function building code.
For example, the following function will fail:
  @tf.function
  def has_init_scope():
    my_constant = tf.constant(1.)
    with tf.init_scope():
      added = my_constant * 2
The graph tensor has name: args_0:0
```

(We are working with the TF team to give a more explicit error in this case.)

### Saved model for non-differentiable JAX functions

Note that if the JAX function is not reverse-mode differentiable, e.g., uses `lax.while_loop` then
attempting to save its conversion to a SavedModel will fail with:

```
ValueError: Error when tracing gradients for SavedModel
```

You have two options, either pass `with_gradient=False` to `jax2tf.convert`, or
set `tf.saved_model.SaveOptions(experimental_custom_gradients=False)`. In either case,
you will not be able to compute the gradients of the function loaded from the SavedModel.

## Support for partitioning

jax2tf supports JAX functions that use `jax.pjit` and `jax.jit` with sharded
arguments and results, for single-host meshes.
The lowering is actually similar as for a `jax.jit`, except that the
arguments and results will be wrapped with
`tensorflow.python.compiler.xla.experimental.xla_sharding.XlaSharding` TensorFlow ops.

In the default native serialization mode, if the target JAX function
includes sharding operations, e.g., from nested `jax.pjit`, then
there should be a top-level `jax.pjit`. E.g.,

```python
# The following is correct
with mesh:
   jax2tf.convert(pjit.pjit(f_jax, in_shardings=...))(...)

# The following will lead to errors because pjit is not at top-level.
def wrapped_pjit(x):
   ...pjit.pjit(f_jax, in_shardings=...))...

with mesh:
  jax2tf.convert(wrapped_pjit)
```

A limitation of `XlaSharding` is that it cannot be used in TensorFlow eager
mode. Therefore, `jax2tf` will give an error when lowering a function that
requires sharded (not replicated) arguments or results and the lowered
function is used outside a `tf.function` context (see b/255511660).

Another limitation is that today only TPUs have integrated with XLA SPMD
support in serving, while CPUs and GPUs don't have e2e XLA SPMD support yet in
TensorFlow. Executing a jax2tf converted `tf.function` with `XlaSharding` ops on
CPUs and GPUs will simply ignore all the `XlaSharding` ops.

Note that when saving a model, the parameters to the model are wrapped with
`tf.Variable` before calling the lowered function (see [above](#saved_model_with_parameters)),
therefore outside of the `XlaSharding` wrapper.

## Shape-polymorphic conversion

**The shape polymorphism support is work in progress.
Please report any bugs you encounter.**

We described above how to include in the SavedModel several specializations
of a lowered function for a few specific input shapes. `jax2tf` can
also produce a shape-polymorphic TensorFlow graph that is usable with inputs
of any shape matching
certain constraints. This is useful, e.g., to allow a single SavedModel
to be used for multiple batch sizes.

The standard TensorFlow technique for producing a shape-polymorphic graph is
to warm the `tf.function` on partially-specified (shape-polymorphic) inputs, e.g.,
`tf.TensorSpec([None, 28, 28], tf.float32)` for a function that processes a
batch (of unspecified batch size) of 28x28 images.
For jax2tf it is **additionally** necessary to specify an additional `polymorphic_shapes` parameter
for the `jax2tf.convert` function:

```python
f_tf = tf.function(jax2tf.convert(f_jax,
                                  polymorphic_shapes=["(b, 28, 28)"]),
                   autograph=False)
f_tf.get_concrete_function(tf.TensorSpec([None, 28, 28], tf.float32))
```

The `polymorphic_shapes` parameter, in the form of a pytree of strings corresponding
to the pytree of positional
arguments, introduces one or more dimension variables, e.g., `b`, to stand for shape
dimensions that are assumed to be unknown at JAX tracing time.
Dimension variables are assumed to range
over all integers that are greater or equal to 1.
In this particular example, we can
also abbreviate `polymorphic_shapes=["(b, _, _)"]`,
because the `_` placeholders take their value
from the corresponding dimension of the `tf.TensorSpec` (which must be known).
As a further shortcut for a series of `_` at the end of a shape specification you can
use `...`: `polymorphic_shapes=["(b, ...)"]`.

In the example above, the `polymorphic_shapes` specification does
not convey more information than the partial `tf.TensorSpec`,
except that it gives a name to the unknown dimension, which improves
error messages. The real need for named shape
variables arises when there are
multiple unknown dimensions and there is a relationship between them.
For example,
if the function to be lowered is also polymorphic on the size of each
image while requiring the images to be square,
we would add a dimension variable `d` to stand for
the unknown image size:

```python
f_tf = tf.function(jax2tf.convert(f_jax, polymorphic_shapes=["(b, d, d)"]), autograph=False)
f_tf.get_concrete_function(tf.TensorSpec([None, None, None], tf.float32))
```

The JAX tracing mechanism performs shape checking using the same strict rules as
when the shapes are fully known. For example, given the `"(b, d, d)"`
specification for the argument `x` of a function, JAX will know that a conditional
`x.shape[-2] == x.shape[-1]` is `True`, and will also know that `x` and `jnp.sin(x)` have the
same shape of a batch of square matrices that can be passed to `jnp.matmul`.

### Correctness of shape-polymorphic tracing

We want to trust that the lowered program produces the same results as the
original JAX program. More precisely:

For any function `f_jax` and any input signature `abs_sig` containing partially
known `tf.TensorSpec`, and any concrete input `x` whose shape matches `abs_sig`:

 * If the conversion to TensorFlow succeeds: `f_tf = tf.function(jax2tf.convert(f_jax, polymorphic_shapes)).get_concrete_function(abs_sig)`
 * and if the TensorFlow execution succeeds with result `y`: `f_tf(x) = y`
 * then the JAX execution would produce the same result: `f_jax(x) = y`,

It is crucial to understand that `f_jax(x)` has the freedom to re-invoke the JAX tracing machinery,
and in fact it does so for each distinct concrete input shape, while the generation of `f_tf`
uses JAX tracing only once, and invoking `f_tf(x)` does not use JAX tracing anymore. In fact,
the latter invocation may happen after the `f_tf` has been serialized
to a SavedModel and reloaded in an environment where `f_jax` and the JAX
tracing machinery are not available anymore.

### Coverage of shape-polymorphic tracing

Besides correctness, a secondary goal is to be able to lower many shape-polymorphic programs,
but at the very
least batch-size-polymorphic programs, so that one SavedModel can be used for any batch sizes.
For example, we want to ensure that any function written using `jax.vmap` at the top level can be
lowered with the batch dimension polymorphic and the remaining dimensions concrete.

It is reasonable to expect that there will be JAX programs for which there is a
shape-polymorphic TensorFlow graph, but which will give an error when lowering with jax2tf.
In general, you should expect that shape polymorphism can handle those programs for which
all the intermediate shapes can be expressed as simple expressions in the dimension variables
appearing in the input shapes. In particular, this does not apply to programs whose
intermediate shapes depend on the data.

### Details

In order to be able to use shape polymorphism effectively with jax2tf, it
is worth considering what happens under the hood. When the lowered function
is invoked with a `TensorSpec`, `jax2tf` will use the `polymorphic_shapes` parameter
to  obtain a shape abstraction for the inputs. The dimension sizes from the
`TensorSpec` are used to fill in the `_` and `...` placeholders from `polymorphic_shapes`.
Normally, the shape abstraction contains the dimension sizes, but in the
presence of shape polymorphism, some dimensions may be dimension variables.

The `polymorphic_shapes` parameter must be either `None`,
or a pytree of shape specifiers corresponding to the pytree of arguments.
(A value `None` for `polymorphic_shapes` is equivalent to a list of `None`.
See [how optional parameters are matched to arguments](https://jax.readthedocs.io/en/latest/pytrees.html#applying-optional-parameters-to-pytrees).)
A shape specifier is combined with a `TensorSpec` as follows:

  * A shape specifier of `None` means that the shape is given
    by the actual argument `TensorSpec`, which must be fully known.
  * Otherwise, the specifier must be a comma-separated string of dimension specifiers: `(dim_1, ..., dim_n)`, denoting
    an n-dimensional array. The `TensorSpec` must also be of rank ``n``.
    An `...` at the end of the shape specifier is expanded to a list of `_` or appropriate length.
    The corresponding dimensions from the shape specifier and the `TensorSpec` are matched:

       * the dimension specifier of `_` means that the size of the dimension is given by
         the actual `TensorSpec`, which must have a known size in the corresponding dimension.
       * a dimension specifier can also be a lowercase identifier, denoting a dimension-size
         variable ranging over strictly positive integers.
         The abstract value of the dimension is going to be set to this variable.
         The corresponding dimension in `TensorSpec` can be `None` or can be a
         constant.
       * All occurrences of a dimension variable in any dimension
         for any argument are assumed to be equal.

Note that `polymorphic_shapes` controls the shape abstraction used by JAX when tracing
the function. The `TensorSpec`
gives the shape abstraction that TensorFlow will associate with the produced
graph, and can be more specific.

A few examples of shape specifications and uses:

  * `polymorphic_shapes=["(b, _, _)", None]` can be used for a function with two arguments, the first
    having a batch leading dimension that should be polymorphic. The other dimensions for the
    first argument and the shape of the second argument are specialized based on the actual
    `TensorSpec`, which must be known. The lowered function can be used, e.g.,
    with `TensorSpec`s `[None, 28, 28]` and `[28, 16]` for the first and second argument
    respectively. An alternative `TensorSpec` pair can be `[1, 28, 28]` and `[28, 16]`,
    in which case the JAX tracing is done for the same polymorphic shape given by
    `polymorphic_shapes=["(b, 28, 28)", "(28, 16)"]`.

  * `polymorphic_shapes=["(batch, _)", "(batch,)"]`: the leading dimensions of the two arguments
     must match, and are assumed to be greater than 1.
     The second dimension of the first argument is taken from the
     actual `TensorSpec`. This can be used with a `TensorSpec` pair `[None, 16]`
     and `[None]`. It can also be used with a pair of shapes `[8, 16]` and `[8]`.

### Computing with dimension variables

JAX keeps track of the shape of all intermediate results. When those shapes depend
on dimension variables JAX computes them as symbolic expressions
involving dimension variables. The symbolic expressions can represent the result
of applying arithmetic operators (add, sub, mul, floordiv, mod,
including the NumPy variants `np.sum`, `np.prod`, etc.) **on dimension
variables and integers** (`int`, `np.int`, or anything convertible by `operator.index`).
These symbolic dimensions can then be used in shape-parameters of JAX primitives
and APIs, e.g., in `jnp.reshape`, `jnp.arange`, slicing indices, etc.

For example, in the following code to flatten a 2D array, the computation
`x.shape[0] * x.shape[1]` computes the symbolic dimension `4 * b` as the
new shape:

```python
jax2tf.convert(lambda x: jnp.reshape(x, (x.shape[0] * x.shape[1],)),
                polymorphic_shapes=["(b, 4)"])(np.ones((3, 4)))
```

When a symbolic dimension is used in **arithmetic operations with non-integers**,
e.g., `float`, `np.float`, `np.ndarray`, or JAX arrays, it is automatically
converted to a JAX array using `jnp.array`.
For example, in the function below all occurrences of `x.shape[0]`
are converted implicitly to `jnp.array(x.shape[0])` because
they are involved in operations with non-integer scalars or with
JAX arrays:

```python
jax2tf.convert(lambda x: (x + x.shape[0] + jnp.sin(x.shape[0]),
                          5. + x.shape[0],
                          x.shape[0] - np.ones((5,), dtype=np.int32)),
               polymorphic_shapes=["b"])(np.ones(3))
```

Another typical example is when computing averages:

```python
jax2tf.convert(lambda x: jnp.sum(x, axis=0) / x.shape[0],
               polymorphic_shapes=["(v, _)"])(np.ones((3, 4)))
```

It is also possible to convert dimension polynomials explicitly
to JAX arrays, with `jnp.array(x.shape[0])` or even `jnp.array(x.shape)`.
The result of these operations
cannot be used anymore as dimension parameters and will raise a JAX error.

### Errors in presence of shape polymorphism

Most JAX code assumes that the shapes of JAX arrays are tuples of integers,
but with shape polymorphism some dimensions may be symbolic expressions.
This can lead to a number of errors. For example, the program:

```python
four_ones = np.ones((4,))
jax2tf.convert(lambda x, y: x + y,
               polymorphic_shapes=["(v,)", "(4,)"])(four_ones, four_ones)
```

with result in the error `'add got incompatible shapes for broadcasting: (v,), (4,)'`
because the shape abstraction that JAX tracing uses is given by the
`polymorphic_shapes`, even though the
actual arguments are more specific and would actually work.

Also,
```python
jax2tf.convert(lambda x: jnp.matmul(x, x),
               polymorphic_shapes=["(v, 4)"])(np.ones((4, 4)))
```

will result in the error `dot_general requires contracting dimensions to have the same shape, got [4] and [v]`. What is
happening here is that in the process of type checking the `matmul` operation, JAX
will want to ensure the size of the two axes is the same (`v == 4`).
Note that `v` can stand for any integer greater than 0, so the value of the
equality expression can be true or false. Since it is not always true
that `v == 4`, the shape checking rules fail with the above error.
Since the lowered function works only for square matrices, the correct
`polymorphic_shapes` is `["(v, v)"]`.

As explained above, if the dimension polynomials are used in operations with
non-integers, the result will be a JAX array that cannot be used as a shape
parameter. For example, if we modify the reshape example slightly,
to use `np.array([x.shape[1]])` instead of `x.shape[1]`:

```python
jax2tf.convert(lambda x: jnp.reshape(x, (x.shape[0] * np.array([x.shape[1]]),)),
                polymorphic_shapes=["(b, 4)"])(np.ones((3, 4)))
```

we get an error `Shapes must be 1D sequences of concrete values of integer type, got Traced<...>`.
If you get this error on JAX code that works for static shapes, it means that one operation
that computes shape parameters is using non-integer arguments, e.g., `np.ndarray`, that get
implicitly converted to JAX arrays.
The solution is to avoid `np.array`, `float`, or JAX arrays in operations whose
results are used as shapes, e.g., instead of `np.arange(n) * x.shape[0]` write
`[i * x.shape[0] for i in range(n)]`.

### Dimension variables must be solvable from the input shapes

JAX will generate code to derive the values of the dimension variables
from the input shapes. This works only if the symbolic dimensions in the input shapes are linear.
For example, the following `polymorphic_shapes` will result in errors:

```python
polymorphic_shapes = ["a * a"]  # Not a linear polynomial
polymorphic_shapes = ["a + b"]  # Too few equations to derive both `a` and `b`
```

The error message for the last specification above would be:

```
Cannot solve for values of dimension variables {'a', 'b'}. "
We can only solve linear uni-variate constraints. "
Using the following polymorphic shapes specifications: args[0].shape = (a + b,).
Unprocessed specifications: 'a + b' for dimension size args[0].shape[0]. "
Please see https://github.com/google/jax/blob/main/jax/experimental/jax2tf/README.md#dimension-variables-must-be-solvable-from-the-input-shapes for more details.
```

### Shape assertion errors

JAX assumes that dimension variables range over strictly positive integers.
Starting with serialization version 7 these assumptions are
checked against the shapes of the actual arguments
when the lowered code is invoked.
For example, given the `polymorphic_shapes="(b, b, 2*d)"`
specification, we will generate code to check the following constraints when
invoked with actual argument `arg`:

  * `arg.shape[0] >= 1`
  * `arg.shape[1] == arg.shape[0]`
  * `arg.shape[2] % 2 == 0`
  * `arg.shape[2] // 2 >= 1`

An example error for the third constraint above, e.g., when invoked with
shape `(3, 3, 5)`, would be:

```
Input shapes do not match the polymorphic shapes specification.
Division had remainder 1 when computing the value of 'd'.
Using the following polymorphic shapes specifications: args[0].shape = (b, b, 2*d).
Obtained dimension variables: 'b' = 3 from specification 'b' for dimension args[0].shape[0] (= 3).
Please see https://github.com/google/jax/blob/main/jax/experimental/jax2tf/README.md#shape-assertion-errors for more details.
```

When using native serialization these are checked by the `tf.XlaCallModule`
op (starting with serialization
[version 7](https://github.com/search?q=repo%3Agoogle%2Fjax+path%3Aconfig.py+jax_serialization_version&type=code)),
and you will get `tf.errors.InvalidArgument` errors.
You can disable this checking by including `DisabledSafetyCheck.shape_assertions()`
in the `disabled_checks` parameter to `jax2tf.convert`, or by setting
the environment variable
`TF_XLA_FLAGS=--tf_xla_call_module_disabled_checks=shape_assertions`.
When using graph serialization these are checked using `tf.debugging.assert`,
which will also result in `tf.errors.InvalidArgument`.
Note that due to limitations in TensorFlow, these errors are suppressed when using
`jit_compile=True` and when running on TPU.

### Comparison of symbolic dimensions is partially supported

Inside JAX there are a number of equality and inequality comparisons
involving shapes, e.g., for doing shape checking or even for choosing
the implementation for some primitives. Comparisons are supported
as follows:

  * equality is supported with a caveat: if the two symbolic dimensions denote the same
    value under all valuations for dimension variables, then equality evaluates to `True`,
    e.g., for `b + b == 2*b`; otherwise the equality evaluates to `False`. See below
    for a discussion of important consequences of this behavior.
  * disequality is always the negation of equality.
  * inequality is partially supported, in a similar way as partial equality.
    However, in this
    case we take into consideration that dimension variables range over strictly positive
    integers. E.g., `b >= 1`, `b >= 0`, `2 * a + b >= 3` are `True`, while `b >= 2`,
    `a >= b`, `a - b >= 0` are inconclusive and result in an exception.

For example, the following code raises the exception
`core.InconclusiveDimensionOperation` with the message
`Dimension polynomial comparison 'a + 1' >= 'b' is inconclusive`.

```python
jax2tf.convert(lambda x: 0 if x.shape[0] + 1 >= x.shape[1] else 1,
                polymorphic_shapes=["(a, b)"])(np.ones((3, 4)))
```

If you do get an `core.InconclusiveDimensionOperation`, you can try
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
change the symbolic shape specifications to add implicit constraints
for dimension sizes. E.g.,

  * You can use `2*b` for a dimension to constrain it to be even (and `>= 2`).
  * You can use `b + 15` for a dimension to constrain it to
    be at least 16. E.g., the following code would fail without
    the `+ 15` part, because JAX will want to verify that slice sizes
    are at most as large as the axis size.

```python
jax2tf.convert(lambda x: x[0:16],
               polymorphic_shapes="b + 15, ...")
```

Such implicit symbolic constraints are used for reasoning, and are
checked at compile time, as explained [above](#shape-assertion-errors).

Starting with JAX version 0.4.24 you can also specify explicit
symbolic constraints:

```python
jax2tf.convert(lambda x: x[:x.shape[1], :16],
               polymorphic_shapes="(a, b)",
               polymorphic_constraints=("a >= b", "b >= 16"))
```

The constraints form a conjunction together with the implicit
constraints. You can specify `>=`, `<=`, and `==` constraints.
At the moment, JAX has limited support for reasoning with
symbolic constraints:

  * You get most from constraints of the form
    of a variable being greater-or-equal or
    less-or-equal to a constant.
    For example, from the constraints that
    `a >= 16` and `b >= 8` we can infer
    that `a + 2*b >= 32`.
  * You get limited power when the constraint involves
    more complex expressions, e.g., from `a >= b + 8` we
    can infer that `a - b >= 8` but not that `a >= 9`.
    We plan to improve somewhat this area in the future.
  * Equality constraints are treated as normalization rules.
    E.g., `floordiv(a, b) = c` works by replacing all
    occurences of the left-hand-side with the right-hand-side.
    You can only have equality constraints where the left-hand-side
    is a multiplication of factors, e.g, `a * b`, or `4 * a`, or
    `floordiv(a, b)`. Thus, the left-hand-side cannot contain
    addition or subtraction at the top-level.

The symbolic constraints can also help to work around the
limitations in the JAX reasoning mechanisms. For example, the following
code would not be able to prove that the slice size fits
into the axis size (such examples come up when using
striding):

```python
jax2tf.convert(lambda x: x[: 4*(x.shape[0] // 4)],
               polymorphic_shapes=("b, ...",))
```

You will likely see an error that the comparison
`b >= 4*floordiv(b, 4)` is inconclusive, even though
the inequality always holds when `b >= 1`. One option
here would be to restrict the code to work only on
axis sizes that are multiple of `4` (by replacing
`b` with `4*b` in the shape specification);
another option is to add a symbolic constraint
with the exact inconclusive inequality:

```python
jax2tf.convert(lambda x: x[: 4*(x.shape[0] // 4)],
               polymorphic_shapes=("b, ...",),
               polymorphic_constraints=("b >= 4*floordiv(b, 4)",))
```

An example where an equality constraint would be useful
is in the following code:

```python
jax2tf.convert(lambda x, y: x + y[:y.shape[0] // 2],
               polymorphic_shapes=("a", "b"))(x, y)
```

The above code would raise a `TypeError` because JAX cannot verify that
`x` and `y[:x.shape[0]]` have the same shape:

```python
TypeError: add got incompatible shapes for broadcasting: (a,), (floordiv(b, 2),)
```

You can fix this by adding a constraint:
```python
jax2tf.convert(lambda x, y: x + y[:y.shape[0] // 2],
               polymorphic_shapes=("a", "b"),
               polymorphic_constraints=("floordiv(b, 2) == a",))(x, y)
```

Just like the implicit constraints, the explicit
symbolic constraints are checked at compile time,
using the same mechanism as explained [above](#shape-assertion-errors).

The symbolic constraints are stored in αn
`export.SymbolicScope` object, which is created implicitly
for each call to `jax2tf.convert`. You must be careful
to not mix symbolic expressions that use different scopes.
For example,
the following code will fail because `a1` and `a2`
use different scopes (created by `export.symbolic_shape`):

````python
a1, = export.symbolic_shape("a,")
a2, = export.symbolic_shape("a,", constraints=("a >= 8",))

a1 + a2
````

The symbolic expressions that originate from a single call
to `export.symbolic_shape` share a scope and
can be mixed up in arithmetic operations. The result would
also share the same scope.

You can re-use scopes:

```python
a, = export.symbolic_shape("a,", constraints=("a >= 8",))
b, = export.symbolic_shape("b,", scope=a1.scope)

a + b  # Allowed
```

JAX tracing uses caches keyed partially by shapes, and
symbolic shapes that are printed identically will be considered
distinct if they use different scopes.

#### Caveat for equality comparisons

The equality comparison returns `False` for `b + 1 == b` or `b == 0`
(in which case it is certain that the dimensions are different for all valuations),
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

We attempted to retain soundness and hashability by creating both hashable and unhashable
kinds of symbolic dimensions [PR #14200](https://github.com/google/jax/pull/14200),
but it turned out to be very hard to diagnose hashing failures in user programs because
often hashing is implicit when using sets or memo tables.

Code of the form `if x.shape[0] != 1: raise NiceErrorMessage` is sound even
with this treatment of equality, but code of the form `if x.shape[0] != 1: return 1`
is unsound.

### Division of symbolic dimensions is partially supported

JAX will attempt to simplify division and modulo operations,
e.g., `(a * b + a) // (b + 1) == a` and `6*a + 4 % 3 == 1`.
In particular, JAX will handle the cases when either (a) there
is no remainder, or (b) the divisor is a constant
in which case there may be a constant remainder.
For example, the code below results in a division error when trying to
compute the inferred dimension for a `reshape` operation:

```python
jax2tf.convert(lambda x: jnp.reshape(x, (2, -1)),
               polymorphic_shapes=["(b, ...)"])(np.ones((4, 5, 7)))
```

In this case you will see the error `Cannot divide evenly the sizes of shapes (b, 5, 7) and (2, -1)`,
with a further `Details: Cannot divide '35*b' by '-2'`.
The polynomial `35*b` represents the total size of the input tensor.

Note that the following will succeed:

```python
## The resulting symbolic shape is (2, 15 b).
jax2tf.convert(lambda x: jnp.reshape(x, (2, -1)),
               polymorphic_shapes=["(b, ...)"])(np.ones((4, 5, 6)))

## The resulting symbolic shape is (6 b2, b1).
jax2tf.convert(lambda x: jnp.reshape(x, (-1, x.shape[0])),
               polymorphic_shapes=["(b1, b2, ...)"])(np.ones((4, 5, 6)))
```

You may also encounter division errors when working with strides, such as
when computing the padding in a strided convolution.

When JAX cannot simplify the result of symbolic dimension division it
will construct symbolic expressions of the form `floordiv(E, N)` and
`mod(E, N)` and it will use a number of heuristics to evaluate comparisons
involving these. If you encounter `InconclusiveDimensionOperation` exceptions
you can specify that a dimension variable
is a multiple of the divisor,
e.g., `b` in the above example of dividing `35*b` by `-2` may
be known to be a multiple of `2`. You can specify that by replacing
`b` with `2*b` in the polymorphic shape specification:

```python
jax2tf.convert(lambda x: jnp.reshape(x, (2, -1)),
               polymorphic_shapes=["(2*b, ...)"])(np.ones((4, 5, 7)))
```

## Native serialization versions

We use a serialization version number to help evolve the serialization
mechanism while allowing serialized artifacts to be used by consumers built
at different code versions.

If consumers use the `tf.XlaCallModule` op, e.g. when using the TensorFlow
SavedModel, then they support a range of serialization versions.
See [tf.XlaCallModule code](https://github.com/search?q=repo%3Atensorflow%2Ftensorflow+path%3Axla_call_module+%22int+kVersionMaximumSupported%22&type=code).
There is also an API to get the maximum version number supported by your
installed version of TensorFlow:

```
from tensorflow.compiler.tf2xla.python import xla as tfxla
tfxla.call_module_maximum_supported_version()
```

For **backward compatibility**, we want to allow a freshly built consumer
to load artifacts that have been serialized in the past 6 months
(by a serializer using the latest version supported at the time). Thus,
the minimum supported version number should match the maximum supported
version number from 6 months in the past.

The serialization version used by JAX is determined by the
`--jax_serialization_version` flag, or if missing, the
`JAX_SERIALIZATION_VERSION` environment variable. The default value is
specified in the [`config.py` file](https://github.com/search?q=repo%3Agoogle%2Fjax+path%3Aconfig.py+JAX_SERIALIZATION_VERSION&type=code).

For **forward compatibility**, we want freshly serialized artifacts to be
loadable by consumers that have been built in the last 1 month.
Thus, we bump the default serialization version
number about 1 month after the `tf.XlaCallModule` is upgraded to a
given version number.

You can use `--jax_serialization_version` to adjust the serialization version
to your deployed consumer. We reserve the right to remove support for
generating or consuming old serialization versions older than 6 months.

## Serialization version numbers

We list here a history of the serialization version numbers:

  * Version 1 used MHLO & CHLO to serialize the code, not supported anymore.
  * Version 2 supports StableHLO & CHLO. Used from October 2022. Not supported
    anymore.
  * Version 3 supports platform checking and multiple platforms.
    Used from February 2023. Not supported anymore.
  * Version 4 supports StableHLO with compatibility guarantees.
    This is the earliest version at the time of the JAX native serialization
    launch.
    Used in JAX from March 15, 2023 (cl/516885716). Starting with
    March 28th, 2023 we stopped using `dim_args_spec` (cl/520033493).
    The support for this version was dropped on
    October 17th, 2023 (cl/573858283).
  * Version 5 adds support for `call_tf_graph`. This is currently used
    for some specialized use cases. Used in JAX from May 3rd, 2023
    (cl/529106145).
  * Version 6 adds support for the `disabled_checks` attribute. This version
    mandates a non-empty `platforms` attribute. Supported by XlaCallModule
    since June 7th, 2023 and available in JAX since
    June 13th, 2023 (JAX 0.4.13).
  * Version 7 adds support for `stablehlo.shape_assertion` operations and
    for `shape_assertions` specified in `disabled_checks`.
    See [Errors in presence of shape polymorphism](https://github.com/google/jax/blob/main/jax/experimental/jax2tf/README.md#errors-in-presence-of-shape-polymorphism). Supported by XlaCallModule
    since July 12th, 2023 (cl/547482522),
    available in JAX serialization since July 20th, 2023 (JAX 0.4.14),
    and the default since August 12th, 2023 (JAX 0.4.15).
  * Version 8 adds support for the `jax.uses_shape_polymorphism` module
    attribute and enables the shape refinement pass only when the
    attribute is present. Supported by XlaCallModule since July 21st, 2023
    (cl/549973693), available in JAX since July 26th, 2023 (JAX 0.4.14),
    and the default since October 21st, 2023 (JAX 0.4.20).
  * Version 9 adds support for effects.
    See the docstring for `export.Exported` for the precise calling convention.
    In this serialization version we also tag the platform index and the
    dimension variables arguments with `jax.global_constant` attributes.
    Supported by XlaCallModule since October 27th, 2023,
    available in JAX since October 20th, 2023 (JAX 0.4.20),
    and the default since February 1st, 2024 (JAX 0.4.24).
    This is the only supported version as of 27th of March, 2024.

## Known issues

`jax2tf` has been in use since 2020 and the vast majority of users encounter
no problems. However, there are a few rare corner cases
in which the different conventions of JAX and TensorFlow result in a breakage.
We try to give an exhaustive list below, specifying whether the limitations
apply to the native serialization or non-native.

### Different 64-bit precision in JAX and TensorFlow

Applies to both native and non-native serialization.

JAX behaves somewhat differently than TensorFlow in the handling
of 32-bit vs. 64-bit values. However, the `jax2tf` lowered function
always behaves like the JAX function.

JAX interprets the type of Python scalars differently based on
`JAX_ENABLE_X64` flag. (See
[JAX - The Sharp Bits: Double (64bit) precision](https://jax.readthedocs.io/en/latest/notebooks/Common_Gotchas_in_JAX.html#double-64bit-precision).)
In the default configuration, the
flag is unset, and JAX interprets Python constants as 32-bit,
e.g., the type of `3.14` is `float32`. This is also what
TensorFlow always does. JAX goes further, it forces
all explicitly-specified 64-bit values to be interpreted as
32-bit:

```python
# with JAX_ENABLE_X64=0
jnp.sin(3.14)  # Has type float32
tf.math.sin(3.14)  # Has type float32

jnp.sin(np.float64(3.14))  # Also has type float32
tf.math.sin(np.float64(3.14))  # Has type float64

# The jax2tf.convert function behaves like the JAX function.
jax2tf.convert(jnp.sin)(3.14)  # Has type float32
jax2tf.convert(jnp.sin)(np.float64(3.14))  # Has type float32

# The following will still compute `sin` in float32 (with a tf.cast on the argument).
tf.function(jax2tf.convert(jnp.sin), autograph=False)(tf.Variable(3.14, dtype=tf.float64))
```

When the `JAX_ENABLE_X64` flas is set, JAX uses 64-bit types
for Python scalars and respects the explicit 64-bit types:

```python
# with JAX_ENABLE_X64=1
jnp.sin(3.14)  # Has type float64
tf.math.sin(3.14)  # Has type float32

# The jax2tf.convert function behaves like the JAX function.
jax2tf.convert(jnp.sin)(3.14)  # Has type float64

# The following will compute `sin` in float64.
tf.function(jax2tf.convert(jnp.sin), autograph=False)(tf.Variable(3.14, dtype=tf.float64))

# The following will compute `sin` in float32.
tf.function(jax2tf.convert(jnp.sin), autograph=False)(tf.Variable(3.14))
```

This is achieved by inserting `tf.cast` operations
on the input arguments inside the lowered function,
if necessary.

If you want to create a `tf.Variable` or `tf.TensorSpec` with the
same dtype, you should use `jax2tf.dtype_of_val`:

```python
# The following two calls will lower jax_fun at the same dtypes
# independently of the value of JAX_ENABLE_X64.
jax2tf.convert(jax_fun)(3.14)
jax2tf.convert(jax_fun)(tf.Variable(3.14, dtype=jax2tf.dtype_of_val(3.14)))
```

### Functions whose arguments and results are nested Python data structures

Applies to both native and non-native serialization.

`jax2tf` can lower functions with arguments and results that are nested
collections (tuples, lists, dictionaries) of numeric values or JAX arrays
([pytrees](https://jax.readthedocs.io/en/latest/pytrees.html)). The
resulting TensorFlow function will take the same kind of arguments except the
leaves can be numeric values or TensorFlow tensors (`tf.Tensor`, `tf.TensorSpec`, `tf.Variable`).

As long as the arguments use only standard Python containers (tuple, list, dictionaries),
both JAX and TensorFlow can flatten and unflatten them and you can use the lowered
function in TensorFlow without limitations.

However, if your JAX function takes a custom container, you can register it with
the JAX `tree_util` module so that JAX will know how to operate with it, and you
can still lower the function to use it in TensorFlow
eager and with `tf.function`, but you won't be able to save it to a SavedModel, nor
will you be able to compute gradients with TensorFlow
(code from `jax2tf_test.test_custom_pytree_readme`):

```python
class CustomPair:
  def __init__(self, a, b):
    self.a = a
    self.b = b

# Register it with the JAX tree_util module
jax.tree_util.register_pytree_node(CustomPair,
                                   lambda x: ((x.a, x.b), None),
                                   lambda _, ab: CustomPair(*ab))
def f_jax(pair: CustomPair):
  return 2. * pair.a + 3. * pair.b

x = CustomPair(4., 5.)
res_jax = f_jax(x)
# TF execution works as long as JAX can flatten the arguments
res_tf = jax2tf.convert(f_jax)(x)
self.assertAllClose(res_jax, res_tf.numpy())
res_tf_2 = tf.function(jax2tf.convert(f_jax), autograph=False, jit_compile=True)(x)
```

If you want to save the function in a SavedModel or compute gradients,
you should construct a wrapper:

```python
 # wrapped TF function to use only standard containers
def f_tf_wrapped(a, b):
  return f_tf(CustomPair(a, b))

# Try to put into SavedModel
my_model = tf.Module()
# Save a function that can take scalar inputs.
my_model.f = tf.function(f_tf_wrapped, autograph=False,
                         input_signature=[tf.TensorSpec([], tf.float32),
                                          tf.TensorSpec([], tf.float32)])
model_dir = os.path.join(absltest.get_default_test_tmpdir(), str(id(my_model)))
tf.saved_model.save(my_model, model_dir,
                    options=tf.saved_model.SaveOptions(experimental_custom_gradients=True))

# Restoring (note: the restored model does *not* require JAX to run, just XLA).
restored_model = tf.saved_model.load(model_dir)
def restored_f(pair: CustomPair):
  return restored_model.f(pair.a, pair.b)

res_tf_3 = restored_f(x)
self.assertAllClose(res_jax, res_tf_3)
grad_jax = jax.grad(f_jax)(x)

x_v = [tf.Variable(x.a), tf.Variable(x.b)]
with tf.GradientTape() as tape:
  res = f_tf_wrapped(*x_v)
  grad_tf = tape.gradient(res, x_v)

self.assertAllClose(grad_jax.a, grad_tf[0])
self.assertAllClose(grad_jax.b, grad_tf[1])
```

### Lowering gradients for functions with integer arguments or unused arguments

Applies to both native and non-native serialization.

When JAX differentiates functions with integer or boolean arguments, the gradients will
be zero-vectors with a special `float0` type (see PR 4039](https://github.com/google/jax/pull/4039)).
This type is translated to `int32` when lowering to TF.
For example,

```python
x = np.int16(2)
def f_jax(x):  # x: int16
  return x * 2.

jax.grad(f_jax, allow_int=True)(x)
# returns a special `float0`: array((b'',), dtype=[('float0', 'V')])

jax2tf.convert(jax.grad(f_jax, allow_int=True))(x)
# returns a tf.Tensor(0, shape=(), dtype=int32)
```

Note that this is different from how TensorFlow handles gradients
for integer or boolean arguments: sometimes the gradient is `None`,
sometimes it is a zero with the same dtype as the argument, and
sometimes it is a one with the same dtype as the argument (e.g.,
for the identity function).

```python
def f_tf(x):  # x: int16
  return tf.cast(x, tf.float32) * 2.

xv = tf.Variable(x)
with tf.GradientTape(persistent=True) as tape:
  print(tape.gradient(f_tf(xv), xv))
  # returns None
  print(tape.gradient(f_tf(xv), xv,
                      unconnected_gradients=tf.UnconnectedGradients.ZERO))
  # returns 0 with the same shape and dtype as x
```

When differentiating functions with unused arguments, TF by default
returns the value `None` for the corresponding gradients. The
`tape.gradient` function takes the option `tf.UnconnectedGradients.ZERO`
to ask that gradients for unused arguments be zero.

Functions lowered with `jax2tf.convert` behave the same way under
`tf.UnconnectedGradients.ZERO`, but by default, they will return
`None` only for gradients corresponding to integer arguments.

```python
# x1 and x3 are not used. x3 has integer type.
def fn(x0, x1, x2, x3):
  return x0 * 0. + x2 * 2.

xs = [tf.Variable(x) for x in [10., 11., 12., 13]]
with tf.GradientTape(persistent=True) as tape:
 res = fn(*xs)

g_tf_native = tape.gradient(res, xs)
# Returns: 0., None, 2., None

g_tf_native_0 = tape.gradient(res, xs,
                              unconnected_gradients=tf.UnconnectedGradients.ZERO)
# Returns: 0., 0., 2., 0

# Now with jax2tf.convert
with tf.GradientTape() as tape:
  res = jax2tf.convert(fn, with_gradient=True)(*xs)

g_jax2tf = tape.gradient(res, xs)
# Returns: 0., 0., 2., None
# Note that the gradient for x1 is 0.

g_jax2tf_0 = tape.gradient(res, xs,
                            unconnected_gradients=tf.UnconnectedGradients.ZERO)
# Returns: 0., 0., 2., 0
# In this case we get the same result as for TF native.
```

### Errors due to tf.Module magic conversion during attribute assignment

Applies to both native and non-native serialization.

`tf.Module` will automatically wrap the standard Python container data types into
trackable classes during attribute assignment.
Python Dict/List/Tuple are changed to _DictWrapper/_ListWrapper/_TupleWrapper
classes.
In most situation, these Wrapper classes work exactly as the standard
Python data types. However, the low-level pytree data structures are different
and this can lead to errors.

In such cases, the user can use this workaround:

```python
import tensorflow as tf
input_data = #Any data object

m = tf.Module()
flat, tree_def = jax.tree_util.tree_flatten(input_data)
m.input_data = {"flat": flat, "tree_def": tree_def}
```

Later the user can use `tree_unflatten` for the reverse process:

```python
input_data = jax.tree_util.tree_unflatten(m.input_data['tree_def'], m.input_data['flat'])
```

### Large saved_model.pb due too many PRNG operations

Applies to both native and non-native serialization.

The default `threefry2x32` PRNG is implemented in JAX with dozens
of additions and bitwise operations. This means that a single PRNG
operation in JAX will result in dozens of TF ops after jax2tf.
If the number of RPNG operations
is large, the generated TF graph will be very large.

To reduce the TF graph size and the compilation time
one can use the `unsafe_rbg` PRNG implementation by
setting `jax.config.update('jax_default_prng_impl', 'unsafe_rbg')`.
The `unsafe_rbg` implementation will be lowered to a TF op and several
casts and reshapes, thus significantly reducing the number of TF ops
per PRNG operation. The "unsafe" part is that it doesn't guarantee
determinism across JAX/XLA versions, and the quality of random
streams it generates from different keys is less well understood.
Nevertheless, this should be fine for most inference/serving cases.
See more details in the [JAX PRNG documentation](https://jax.readthedocs.io/en/latest/jax.random.html?highlight=unsafe_rbg#advanced-rng-configuration).

### SavedModel supports only first-order gradients

Applies to both native and non-native serialization.

The `jax2tf`-lowered function supports higher-order gradients, but when the
function is saved in a SavedModel, only the first-order gradient is saved.
This is primarily a limitation of the SavedModel support for custom gradients.

### Native serialization supports only select dialects

Applies to native serialization only.

JAX native serialization checks that the code to be serialized contains
operations only from MLIR dialects that are known to have stability guarantees,
e.g., StableHLO, and the "builtin" dialect. As an exception, it also accepts
operations from the MHLO dialect, but they are converted to corresponding
StableHLO operations upon serialization.

### Native serialization supports only select custom calls

Applies to native serialization only.

JAX natively uses custom calls for lowering of certain primitives.
The most common example is for the implementation of PRNG on GPUs,
where we get better performance with a custom call (`cu_threefry32`)
than if we use native StableHLO. Another class of examples are for
FFT and some linear algebra primitives (e.g., QR decomposition).

Unlike regular StableHLO ops, the compatibility guarantees for
custom calls are the burden of the teams maintaining the C++
code that backs the custom call. For this reason, we maintain
a list of allowed custom call targets. If you try to serialize
code that invokes other targets you will get an error.

If you want to disable this safety check for a specific custom call
with target `my_target`, you can add
`jax2tf.DisabledSafetyCheck.custom_call("my_target")` to the `disabled_checks`
parameter of the `jax2tf` function.

### XlaCallModule not supported by some TensorFlow tools

Applies to native serialization only.

JAX native serialization uses the `XlaCallModule` TensorFlow op to host
the StableHLO program obtained from JAX. This is a relatively
new TensorFlow op and may not be supported by some tools. In fact,
certain tools that need to do `tf.Graph` inspection and transformation
cannot work when the whole JAX program is a single TensorFlow op.

This is the case, for example, for the TFLite and TensorFlow.js converters.
There is work underway to enable more tools to consume StableHLO.

### Natively serialized JAX modules are platform specific

Applies to native serialization only.

When you use native serialization, JAX will record the platform for
which the module was serialized, and you will get an error if you
try to execute the `XlaCallModule` TensorFlow op on another platform.

Note that this error will only arise in native serialization; with
non-native serialization the lowering to TensorFlow ops is
platform independent, although it is only guaranteed to match the
JAX semantics and performance behavior for TPUs.

The error has the form:
```commandline
The current platform CPU is not among the platforms required by the module [CUDA]
```

where `CPU` is the TensorFlow platform where the op is being executed
and `CUDA` is the platform for which the module was serialized by JAX.
This probably means that JAX and TensorFlow may see different devices
as the default device (JAX defaults to GPU and TensorFlow to CPU
in the example error above).
You can check what devices TensorFlow uses:

```python
logging.info("All TF devices: %s", tf.config.list_logical_devices())
tf_device = (tf.config.list_logical_devices("TPU") +
             tf.config.list_logical_devices("GPU") +
             tf.config.list_logical_devices())[0]
assert jax.default_backend().upper() == tf_device.device_type
with tf.device(tf_device):
   ...
```

Users should pay attention to another case, which is that they must use
`jit_compile=True` in order to execute on TPU.

Because if `jit_compile=False`, TF "executes the function without XLA
compilation. Set this value to False when directly running a multi-device
function on TPUs (e.g. two TPU cores, one TPU core and its host CPU)" (see
[TF doc](https://www.tensorflow.org/api_docs/python/tf/function))

With `jit_compile=False` the converted TF program will be executed on CPU
instead of TPU and this will result in an error message

```
Node: 'XlaCallModule'
The current platform CPU is not among the platforms required by the module: [TPU]
	 [[{{node XlaCallModule}}]]
```

To work around this on `jit_compile=False`, you can wrap your function with a
new tf.function that explicitly assigns the TPU device, like this:

```python
f_tf = jax2tf.convert(jnp.sin)
x = np.float32(.5)

@tf.function(autograph=False, jit_compile=False)
def f_tf_wrapped(x):
  with tf.device('/device:TPU:0'):
    return f_tf(x)

with tf.device('/device:TPU:0'):
  self.assertAllClose(np.sin(x), f_tf_wrapped(x))
```

### Unsupported JAX features

Applies to non-native serialization only.

There is currently no support for `pmap`, `xmap`, `shard_map`,
nor for the collective operations, except in native serialization.

### Shape polymorphism with native serialization limitations for `lax.linalg.eigh`

Applies to native serialization only.

JAX lowers `lax.linalg.eigh` using custom calls, and needs to call helper
functions to determine the workspace size based on the non-batch dimensions.
Therefore, dynamic dimensions are supported only for the batch dimensions
(all but the last two dimensions).

Additionally, on GPU, JAX lowering uses the `cuSolver` library and chooses
`syevj` method (using Jacobi algorithm) for non-batch dimension size less or
equal to 32, and the `syevd` method (using QR algorithm) for larger dimensions.

In presence of shape polymorphism, JAX will always use `syevd`, because `syevj`
requires knowing the batch dimensions statically in order to compute
the workspace size. This means that the performance and the numerical behavior
may be slightly different for small matrices.

### Slow implementation of associative reductions for CPU

Applies to non-native serialization only.

Operations like ``jax.numpy.cumsum`` are lowered by JAX differently based
on the platform. For TPU, the lowering uses the [HLO ReduceWindow](https://www.tensorflow.org/xla/operation_semantics#reducewindow)
operation, which has an efficient implementation for the cases when the
reduction function is associative. For CPU and GPU, JAX uses an alternative
lowering using [associative scans](https://github.com/google/jax/blob/f08bb50bfa9f6cf2de1f3f78f76e1aee4a78735d/jax/_src/lax/control_flow.py#L2801).
jax2tf uses the TPU lowering (because it does not support backend-specific lowering)
and hence it can be slow in some cases on CPU and GPU.

We have filed a bug with the XLA:CPU compiler to improve ReduceWindow.
Meanwhile, if you run into this problem you can use the
``--jax2tf_associative_scan_reductions`` flag to get the special
associative scan lowering.
You can alternatively use the ``with jax.jax2tf_associative_scan_reductions(True)``
around the code that invokes the function returned by ``jax2tf.convert``.
Use this only if it improves the performance for your application.

Note that this lowering may not work as well as the default one in presence
of shape polymorphism.

### TensorFlow XLA ops

Applies to non-native serialization only.

For most JAX primitives there is a natural TensorFlow op that fits the needed semantics.
There are a few (listed in [no_xla_limitations.md](g3doc/no_xla_limitations.md)) JAX primitives
for which there is no single TensorFlow op with matching semantics.
This is not so surprising, because JAX primitives have been designed
to be compiled to [HLO ops](https://www.tensorflow.org/xla/operation_semantics),
while the corresponding TensorFlow ops are sometimes higher-level.
For the cases when there is no matching canonical TensorFlow op,
we use a set of special TensorFlow ops that are thin wrappers over HLO ops
(a subset of those registered in
[tf2xla/ops/xla_ops.cc](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/compiler/tf2xla/ops/xla_ops.cc)
and implemented in,
e.g.,
[tf2xla/kernels/xla_pad_op.cc](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/compiler/tf2xla/kernels/xla_pad_op.cc).)
We refer to these ops here as the XLA TensorFlow ops. Note that these are
still regular TF ops, e.g., they can be saved in a SavedModel.

There are several drawbacks of using XLA TensorFlow ops:

   * These ops will only be executable by a consumer that has XLA linked in.
   This should not be a problem for TPU execution, since that requires XLA anyway.
   * These ops are not yet recognized by tools that process
   tf.Graph, e.g., TensorFlow.js converter or the TensorFlow Lite converter.

As an experimental feature we implemented alternative conversions to avoid the XLA TensorFlow ops.
You can enable this with the `enable_xla=False` parameter to `jax2tf.convert`.
For more details see [no_xla_limitations.md](g3doc/no_xla_limitations.md).

### Different performance characteristics

Applies to non-native serialization only.

The lowered code may have slightly different performance characteristics than
the original JAX code.
We do expect that the performance characteristics of lowered code
should be the same as those of JAX when used with the XLA compiler (`tf.function(jit_compile=True)`).
This is because
during lowering we try to generate one TensorFlow op for one JAX primitive.
We expect that the lowering that XLA does is similar to that done by JAX
before conversion. (This is a hypothesis, we have not yet verified it extensively.)

There is one know case when the performance of the lowered code will be different.
JAX programs use a [stateless
deterministic PRNG](https://github.com/google/jax/blob/main/docs/design_notes/prng.md)
and it has an internal JAX primitive for it.
This primitive is at the moment lowered to a soup of tf.bitwise operations,
which has a clear performance penalty. We plan to look into using the
HLO [RNGBitGenerator](https://www.tensorflow.org/xla/operation_semantics#rngbitgenerator)
(exposed as a TFXLA op), which does implement
the same basic Threefry algorithm as JAX’s PRNG, although that would
result in different results than JAX’s PRNG.

In absence of TensorFlow XLA compilation,
if one were to write the same functionality in JAX idiomatic code vs.
native TensorFlow idiomatic code we could end up with very different compilation paths.
Take for example, the case of batch normalization.
In TensorFlow if one uses [tf.nn.batch_normalization](https://www.tensorflow.org/api_docs/python/tf/nn/batch_normalization),
a “high-level” TensorFlow op for batch
normalization is generated, and in the absence of XLA, on CPU or GPU,
a custom C++ “high-level” kernel implementing batch normalization is executed.
In JAX, there is no primitive for batch normalization, and instead the
operation is decomposed into low-level primitives (e.g., [flax.linen.BatchNorm](https://flax.readthedocs.io/en/latest/_autosummary/flax.linen.BatchNorm.html),
or haiku.BatchNorm).
Once those primitives are lowered to TensorFlow, and the resulting code is
run without XLA, the ensemble of the kernels executed will quite
possibly behave differently, performance-wise or even numerically,
than either the TensorFlow native or JAX native batch normalization.
A similar example is that of an LSTM cell.


### Unchecked assumption that the dimension variables take strictly positive values

Applies to non-native serialization only.

The shape polymorphic conversion is sound with the assumption that the dimension
variables take non-zero values. In the following example, the function to be lowered
has different behavior for empty shapes. The broken assumption is caught by jax2tf if
the lowered function is executed eagerly, but not if it is first traced to a
TensorFlow graph:

```python
def f_jax(x):
  return 0 if x.shape[0] == 0 else 1

x0 = np.array([], np.float32)
self.assertEqual(0, f_jax(x0))  # JAX sees that the x.shape[0] == 0

# jax2tf catches the broken assumption b >= 1 if the lowered function is executed
# eagerly.
# Raises: ValueError: Dimension variable b must have integer value >= 1. Found value 0 when solving b == 0
jax2tf.convert(f_jax, polymorphic_shapes=["b"])(x0)

# However, if we first trace to a TensorFlow graph, we may miss the broken assumption:
f_tf = tf.function(
        jax2tf.convert(f_jax, polymorphic_shapes=["b"]), autograph=False
       ).get_concrete_function(tf.TensorSpec([None], dtype=np.float32))
self.assertEqual(1, f_tf(x0))
```

Another possible source of unsoundness is that JAX assumes that all unknown
dimensions represented by the same dimension variable have equal size. As before,
this assumption is checked if the lowered function is executed eagerly, but
it may be missed if it is first traced to a TensorFlow graph:

```python
def f_jax(x):
  return 0 if x.shape[0] != x.shape[1] else 1

x45 = np.ones((4, 5), dtype=np.float32)
self.assertEqual(0, f_jax(x45))  # JAX seems that x.shape[0] != x.shape[1]

# jax2tf catches the broken assumption x.shape[0] == x.shape[1] if the lowered
# function is executed eagerly.
# Raises: ValueError: polymorphic shape ('b, b',) has dimension variable 'b' corresponding to multiple values {4, 5}, for argument shapes (TensorShape([4, 5]),)
jax2tf.convert(f_jax, polymorphic_shapes=["b, b"])(x45)

# However, if we first trace to a TensorFlow graph, we may miss the broken assumption.
f_tf = tf.function(
    jax2tf.convert(f_jax, polymorphic_shapes=["b, b"]),
    autograph=False).get_concrete_function(tf.TensorSpec([None, None], dtype=np.float32))
self.assertEqual(1, f_tf(x45))
```

### Incomplete TensorFlow data type coverage

Applies to non-native serialization only.

There are a number of cases when the TensorFlow ops that are used by the
`jax2tf` are not supported by TensorFlow for the same data types as in JAX.
There is an
[up-to-date list of unimplemented cases](https://github.com/google/jax/blob/main/jax/experimental/jax2tf/g3doc/primitives_with_limited_support.md).

If you try to lower and run in TensorFlow a program with partially supported primitives,
you may see TensorFlow errors that
a TensorFlow op is used with an unsupported data type, or that
there is no supported TensorFlow kernel for the op for the given
data type. The former case can happen even if you `jit_compile`
the TensorFlow program, and it is a priority to fit. The latter
case only appears in TensorFlow non-compiled mode; you can
avoid the problem if you use XLA to `jit_compile` (always recommended).

Our priority is to ensure numerical and performance accuracy for
the lowered program **when using XLA to compile the lowered program**.
It is always a good idea to use XLA on the lowered function.

Sometimes you cannot compile the entire TensorFlow function for your
model, because in addition to the function that is lowered from JAX,
it may include some pre-processing TensorFlow code that
is not compilable with XLA, e.g., string parsing. Even in those situations
you can instruct TensorFlow to compile only the portion that originates
from JAX:

```python
def entire_tf_fun(x):
  y = preprocess_tf_fun_not_compilable(x)
  # Compile the code that is lowered from JAX
  z = tf.function(jax2tf.convert(compute_jax_fn),
                  autograph=False, jit_compile=True)(y)
  return postprocess_tf_fun_not_compilable(z)
```

You won't be able to compile the `entire_tf_fun`, but you can still execute
it knowing that the jax2tf-lowered code is compiled. You can even save
the function to a SavedModel, knowing that upon restore the
jax2tf-lowered code will be compiled.

For a more elaborate example, see the test `test_tf_mix_jax_with_uncompilable`
in [savedmodel_test.py](https://github.com/google/jax/blob/main/jax/experimental/jax2tf/tests/savedmodel_test.py).

# Calling TensorFlow functions from JAX

The function ```call_tf``` allows JAX functions to call
TensorFlow functions. These functions can be called anywhere in a JAX
computation, including in staging contexts ``jax.jit``, ``jax.pmap``, ``jax.xmap``,
or inside JAX's control-flow primitives. In non-staging contexts,
the TensorFlow function is called in eager mode.
For now, only reverse-mode autodiff is supported for these functions
(no forward-mode autodiff, nor ``vmap``).

As a trivial example, consider computing ``sin(cos(1.))`` with ``sin`` done in JAX and ``cos`` in TF:

```python
from jax.experimental import jax2tf

# This is a TF function. It will be called with TensorFlow-compatible arguments,
# such as `numpy.ndarray`, `tf.Tensor` or `tf.Variable`, or a pytree thereof.
# It should return a similar result. This function will be called using
# TensorFlow eager mode if called from outside JAX staged contexts (`jit`,
# `pmap`, or control-flow primitives), and will be called using TensorFlow
# compiled mode otherwise. In the latter case, the function must be compilable
# with XLA (`tf.function(func, jit_compile=True)`)
def cos_tf(x):
  return tf.math.cos(x)

# Compute cos with TF and sin with JAX
def cos_tf_sin_jax(x):
  return jax.numpy.sin(jax2tf.call_tf(cos_tf)(x))

# Calls `cos_tf` in TF eager mode
x = np.float32(1.)
cos_tf_sin_jax(x)

# Compiles `cos_tf` using TF and embeds the XLA computation into the JAX
# XLA computation (containing `sin`). The XLA compiler may even be able to
# fuse through JAX-TF computations.
jax.jit(cos_tf_sin_jax)(x)

# Uses TF gradient for `cos_tf` and JAX gradient for `sin`
jax.grad(cos_tf_sin_jax)(x)
```

If you inspect the generated HLO for `cos_tf_sin_jax`, you will see that the
main JAX computation (`ENTRY xla_computation_cos_tf_sin_jax`) makes a call to
the `a_inference_cos_tf_68__` HLO function that was compiled by TF from `cos_tf`:

```
HloModule xla_computation_cos_tf_sin_jax.18

a_inference_cos_tf_68__.4 {
  arg0.5 = f32[] parameter(0), parameter_replication={false}
  reshape.6 = f32[] reshape(arg0.5)
  cosine.7 = f32[] cosine(reshape.6)
  reshape.8 = f32[] reshape(cosine.7)
  tuple.9 = (f32[]) tuple(reshape.8)
  ROOT get-tuple-element.10 = f32[] get-tuple-element(tuple.9), index=0
}

ENTRY xla_computation_cos_tf_sin_jax.18 {
  constant.2 = pred[] constant(false)
  constant.3 = pred[] constant(false)
  parameter.1 = f32[] parameter(0)
  call.11 = f32[] call(parameter.1), to_apply=a_inference_cos_tf_68__.4
  tuple.12 = (f32[]) tuple(call.11)
  get-tuple-element.13 = f32[] get-tuple-element(tuple.12), index=0
  tuple.14 = (f32[]) tuple(get-tuple-element.13)
  get-tuple-element.15 = f32[] get-tuple-element(tuple.14), index=0
  sine.16 = f32[] sine(get-tuple-element.15)
  ROOT tuple.17 = (f32[]) tuple(sine.16)
}
```

For a more elaborate example, including round-tripping from JAX
to TensorFlow and back through a SavedModel, with support for
custom gradients,
see the test `test_round_trip_custom_grad_saved_model`
in [call_tf_test.py](https://github.com/google/jax/blob/main/jax/experimental/jax2tf/tests/call_tf_test.py).

All the metadata inserted by TF during tracing and compilation, e.g.,
source location information and op names, is carried through to the
JAX XLA computation.

The TF custom gradients are respected, since it is TF that generates the
gradient computation.

`call_tf` works even with shape polymorphism, but in that case
the user must pass the `output_shape_dtype` parameter to `call_tf` to declare
the expected output shapes. This allows JAX tracing to know the shape and
dtype of the results so that it can continue tracing the rest of the program.
When `output_shape_dtype` is not given (the default case), `call_tf` will
form a `tf.Graph` for the called TF function and will use the inferred
type and shape. However, in presence of dynamic shape the inferred TF
type will contain `None` for the dynamic dimensions, which is not enough
information for JAX shape polymorphism.

For example:

```python
def fun_jax(x):
  y_shape = (x.shape[0] * 2, y.shape[1:])
  y = jax2tf.call_tf(
      lambda x: tf.concat([x, x], axis=0),
      output_shape_dype=jax.ShapeDtypeStruct(y_shape, x.dtype))(x)
  # JAX will know the y.shape
  return jnp.ones(y.shape, dtype=y.dtype) + y

jax2tf.convert(fun_jax, polymorphic_shapes=["b, ..."])(x)
```

An even simpler example for a function that returns the same shape as the input:

```python
def fun_jax(x):
  return jax2tf.call_tf(tf.math.sin,
                        output_shape_dtype=x)
  )(x)

jax2tf.convert(fun_jax, polymorphic_shapes=["b, ..."])(x)
```

If all the output shapes of the TF function are static, JAX does not need the
`output_shape_dtype` argument:

```python
def fun_tf(x):
  return tf.math.reduce_sum(tf.math.sin(x))

def fun_jax(x):
  return jax2tf.call_tf(fun_tf)(x)

# The following will not throw an error because the output shape of fun_tf is static.
jax2tf.convert(fun_jax, polymorphic_shapes=["b, ..."])(x)
```

The shape polymorphism support for `call_tf` does not yet work for native serialization.

### Limitations of call_tf

The TF function must be compilable (`tf.function(func, jit_compile=True)`)
and must have static output shapes
when used in a JAX staging context, e.g., `jax.jit`, `lax.scan`, `lax.cond`,
but may have unknown output shapes when used in a JAX op-by-op mode.
For example, the following
function uses strings operations that are not supported by XLA:

```python
def f_tf_non_compilable(x):
   return tf.strings.length(tf.strings.format("Hello {}!", [x]))

f_jax = jax2tf.call_tf(f_tf_non_compilable)
# Works in op-by-op mode
f_jax(np.float32(42.))

# Fails in jit mode
jax.jit(f_jax)(np.float(42.))
```

Yet another unsupported situation is when the TF function
is compilable but with dynamic output shapes:

```python
def f_tf_dynamic_shape(x):
  return x[x[0]:5]
x = np.array([1, 2], dtype=np.int32)

f_jax = jax2tf.call_tf(f_tf_dynamic_shape)
# Works in op-by-op mode
f_jax(x)

# Fails in jit mode
jax.jit(f_jax)(x)
```
Another similar example that will fail to compile:

```python
def f_tf_dynamic_output_shape(x):
  return tf.cond(x[0] >= 0, lambda: x, lambda: x[1:])

x = np.array([1, 2], dtype=np.int32)
```

``call_tf`` works best with pure TF functions that do not capture
``tf.Variable``s or tensors from the environment, and all such
context is passed in explicitly through arguments, and if variables
are modified, the resulting values are passed out through results.
There is a best-effort mechanism that can handle variable capture
and variable updates,
except in the case of a function that modifies ``tf.Variable``s
and is used in a JAX jitted context. Calling the ``inpure_func_tf``
will give an error:

```python
var1 = tf.Variable(1.)
def impure_func_tf(x):
  var1.write(11.)  # BAD: should not write to variables
  return x + var1

jax2tf.call_tf(impure_func_tf)(tf.constant(2.))  # Works in eager mode
jax.jit(jax2tf.call_tf(impure_func_tf))(tf.constant(2.))  # Fails in jit mode
```

The error can be avoided by passing the variable explicitly:

```python
def pure_func_tf(x, var1)
  new_var1 = 11.
  return x + new_var1, new_var1
```

This use case is likely to be revisited.

Note that when the TF function captures a variable from the context, the
TF function must be lowered for the same TF device that hosts the variable.
By default, the lowering will use the first TF device on the same platform
as the embedding JAX computation, e.g., "/device:TPU:0" if the embedding
JAX computation runs on TPU. This will fail if the computation captures
variables on some other devices. It is best to use ``call_tf``
with TF functions that do not capture variables.

In some rare cases your called TF function may contain ops with output
of statically known shape, but for which the shape inference is not implemented
completely and will appear to `call_tf` as if they have dynamically-shaped
outputs. In these cases you may get an error that
`call_tf cannot call functions whose output has dynamic shape`. Try using
the `output_shape_dtype` parameter to specify the expected output shape
(this essentially allows you to override the shape inference for the
purposes of `call_tf`.)

# Misc notes

<!-- Next line must match the copybara config. -->
<!-- Link to internal documentation. -->

## Debugging JAX native serialization

Inside Google, you can turn on logging by using the `--vmodule` argument to
specify the logging levels for different modules,
e.g., `--vmodule=jax_export=3`. You can set `TF_DUMP_GRAPH_PREFIX` to
a directory where modules should be dumped, or to `"-"` to dump the
modules to the log.
The following modules are useful for debugging JAX native serialization:

  * `jax_export=3` - will log the StableHLO module on serialization.
  * `jax2tf=3` - will log the parameters to `XlaCallModule` op on serialization.
  * `xla_call_module_loader=3` - will log the StableHLO module upon loading,
    after shape refinements, and on verification error. You can use level `4` to
    add location information, and level `5` to also print the module before and
    after each transformation.
  * `xla_call_module_op=3` - will log the HLO module generated after
    shape refinement and conversion from StableHLO.
  * `XlaCallModule` lowering has TensorFlow MLIR crash reproducer enabled, which
    can be instructed to generate a crash reproducer upon MLIR pass failures by
    setting an environment variable `MLIR_CRASH_REPRODUCER_DIRECTORY`.

For the two `xla` modules mentioned above, you can control logging in OSS
with environment variables, e.g.:

```
TF_CPP_MIN_LOG_LEVEL=0 TF_CPP_VMODULE=xla_call_module_loader=3 python ...
```

In addition, `TF_DUMP_GRAPH_PREFIX` controls where the dump will be stored, `-`
for stderr, `${SOME_DIR}` to store the dumps in the specified directory.

## TensorFlow versions supported

The ``jax2tf.convert`` and `call_tf` require fairly recent versions of TensorFlow.
As of today, the tests are run using `tf_nightly==2.14.0.dev20230720`.

## Running on GPU

To run jax2tf on GPU, both jaxlib and TensorFlow must be installed with support
for CUDA. One must be mindful to install a version of CUDA that is compatible
with both [jaxlib](https://github.com/google/jax/blob/main/README.md#pip-installation) and
[TensorFlow](https://www.tensorflow.org/install/source#tested_build_configurations).

## Updating the limitations documentation

The jax2tf tests are parameterized by a set of limitations
(see `tests/primitive_harness.py` and `tests/jax2tf_limitations.py`).
The limitations specify test harnesses that are known to fail, by
JAX primitive, data type, device type, and TensorFlow execution mode (`eager`,
`graph`, or `compiled`). These limitations are also used
to generate tables of limitations, e.g.,

   * [List of primitives not supported in JAX](https://github.com/google/jax/blob/main/jax/experimental/jax2tf/g3doc/jax_primitives_coverage.md),
     e.g., due to unimplemented cases in the XLA compiler, and
   * [List of primitives not supported in jax2tf](https://github.com/google/jax/blob/main/jax/experimental/jax2tf/g3doc/primitives_with_limited_support.md),
     e.g., due to unimplemented cases in TensorFlow. This list is incremental
     on top of the unsupported JAX primitives.

There are instructions for updating those documents at the end of each
document.

The set of limitations is an over-approximation, in the sense that if XLA
or TensorFlow improves and support more cases, no test will fail. Instead,
periodically, we check for unnecessary limitations. We do this by uncommenting
two assertions (in `tests/jax_primitives_coverage_test.py` and in
`tests/tf_test_util.py`) and running all the tests. With these assertions enabled
the tests will fail and point out unnecessary limitations. We remove limitations
until the tests pass. Then we re-generate the documentation.

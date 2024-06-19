# Exporting and serializing staged-out computations

The {ref}`ahead-of-time-lowering` APIs produce
objects that can be used for debugging or for compilation and
execution in the same process.
Sometimes you want to serialize a lowered JAX function for
compilation and execution in a separate process, perhaps
at a later time. This would allow you to:

  * compile and execute the function in another process or machine
    without requiring access to the JAX program,
    and without having to repeat the staging-out and lowering, e.g.,
    in an inference system.
  * trace and lower a function on a machine that does not have access
    to the accelerator for which you want to later compile and execute
    the function.
  * archive a snapshot of a JAX function, e.g., to be able to
    reproduce later your results. **Note:** check out the [compatibility
    guarantees](#compatibility-guarantees) for this use case.

Here is an example:

```python
>>> import re
>>> import numpy as np
>>> import jax
>>> from jax import export

>>> def f(x): return 2 * x * x


>>> exported: export.Exported = export.export(jax.jit(f))(
...    jax.ShapeDtypeStruct((), np.float32))

>>> # You can inspect the Exported object
>>> exported.fun_name
'f'

>>> exported.in_avals
(ShapedArray(float32[]),)

>>> print(re.search(r".*@main.*", exported.mlir_module()).group(0))
  func.func public @main(%arg0: tensor<f32> {mhlo.layout_mode = "default"} loc("x")) -> (tensor<f32> {jax.result_info = "", mhlo.layout_mode = "default"}) {

>>> # And you can serialize the Exported to a bytearray.
>>> serialized: bytearray = exported.serialize()

>>> # The serialized function can later be rehydrated and called from
>>> # another JAX computation, possibly in another process.
>>> rehydrated_exp: export.Exported = export.deserialize(serialized)
>>> rehydrated_exp.in_avals
(ShapedArray(float32[]),)

>>> def callee(y):
...  return 3. * rehydrated_exp.call(y * 4.)

>>> callee(1.)
Array(96., dtype=float32)

```

Serialization is broken down into two stages:
   1. exporting to produce an {class}`jax.export.Exported` object that contains
     the StableHLO for the lowered function along with the metadata necessary to
     call it from another JAX function. We have plans to add code to generate
     `Exported` objects from TensorFlow, and to use `Exported` objects from
     TensorFlow and PyTorch.
   2. the actual serialization to a byte array using the flatbuffers format. 
     See {ref}`jax2tf` for
     an alternative serialization to TensorFlow graph that can be used
     for interoperation with TensorFlow.

## Support for reverse-mode AD

Serialization can optionally support higher-order reverse-mode AD. This is done
by serializing the {func}`jax.vjp` of the primal function along with the primal function,
up to a user-specified order (default is 0, meaning that the rehydrated
function cannot be differentiated):

```python
>>> import jax
>>> from jax import export
>>> from typing import Callable

>>> def f(x): return 7 * x * x * x

>>> # Serialize 3 levels of VJP along with the primal function
>>> blob: bytearray = export.export(jax.jit(f))(1.).serialize(vjp_order=3)
>>> rehydrated_f: Callable = export.deserialize(blob).call

>>> rehydrated_f(0.1)  # 7 * 0.1^3
Array(0.007, dtype=float32)

>>> jax.grad(rehydrated_f)(0.1)  # 7*3 * 0.1^2
Array(0.21000001, dtype=float32)

>>> jax.grad(jax.grad(rehydrated_f))(0.1)  # 7*3*2 * 0.1
Array(4.2, dtype=float32)

>>> jax.grad(jax.grad(jax.grad(rehydrated_f)))(0.1)  # 7*3*2
Array(42., dtype=float32)

>>> jax.grad(jax.grad(jax.grad(jax.grad(rehydrated_f))))(0.1)  # doctest: +IGNORE_EXCEPTION_DETAIL
Traceback (most recent call last):
ValueError: No VJP is available

```

Note that the VJP function is computed lazily while serializing,
when the JAX program is still available.
This means that it respects all features of JAX VJP,
e.g., {func}`jax.custom_vjp` and {func}`jax.remat`.

Note that the rehydrated function does not support any other
transformations, e.g., forward-mode AD (jvp), or {func}`jax.vmap`.

## Compatibility guarantees

You should not use the raw StableHLO that is obtained from just lowering
(`jax.jit(f).lower(1.).compiler_ir()`)
for archival and for compilation in another process, for several reasons.

First, the compilation may use a different version of the compiler, supporting a
different version of StableHLO. The {class}`jax.export` module takes
care of this by using the
[portable-artifact feature of StableHLO](https://github.com/openxla/stablehlo/blob/main/docs/compatibility.md)
to deal with the possible evolution of the StableHLO opset.

### Compatibility guarantees for custom calls

Second, the raw StableHLO may contain custom calls referencing C++
functions.
JAX uses custom calls for lowering of a small number of primitives,
e.g., linear algebra primitives, sharding annotations, or Pallas kernels.
These do not fall under the compatibility guarantees for StableHLO.
The C++ implementations of these functions change rarely, but they can change.

`jax.export` makes the following export compatibility guarantees:
A JAX exported artifact can be compiled and executed by a compiler and
JAX runtime system that are:

  * **up to 6 months newer** than the version of JAX used for exporting
  (we say that JAX export offers **6 months backward compatibility**).
  This is useful if we want to archive the exported artifact to be compiled and executed later.
  * **up to 3 weeks older** than the version of JAX used for exporting
  (we say that JAX export offers **3 weeks forward compatibility**).
  This is useful if we want to compile and run an exported artifact with a
  consumer that was built and deployed before the export, e.g.,
  an inference system that is already deployed when the exporting is done.

(The particular compatibility window lengths are the same that JAX
[promised for jax2tf](https://github.com/google/jax/blob/main/jax/experimental/jax2tf/README.md#usage-saved-model),
and are based on [TensorFlow Compatibility](https://www.tensorflow.org/guide/versions#graph_and_checkpoint_compatibility_when_extending_tensorflow).
The terminology “backward compatibility” is from the perspective of the consumer,
e.g., the inference system.)

What **matters is when the exporting and consuming components were built**,
not the time when the exporting and the compilation happen.
For external JAX users, it is
[possible to run JAX and jaxlib at different versions](https://jax.readthedocs.io/en/latest/jep/9419-jax-versioning.html#how-are-jax-and-jaxlib-versioned);
what matters is when the jaxlib release was built.

To reduce chances of incompatibility, internal JAX users should:
  * **rebuild and redeploy consumer systems as frequently as possible**.

and external users should:
  * run the exporting and consumer systems with the same version of jaxlib, whenever possible, and
  * export for archival **with the latest released version of jaxlib**.

The compatibility guarantees do not apply if you bypass the `jax.export` APIs
to obtain the StableHLO code.

Only a subset of custom calls are guaranteed stable and have
compatibility guarantees ([see list](https://github.com/search?q=repo%3Agoogle%2Fjax%20_CUSTOM_CALL_TARGETS_GUARANTEED_STABLE&type=code)).
We continuously
add more custom call targets to the allowed list along with backwards
compatibility tests. If you try to serialize
code that invokes other custom call targets you will get an error
during exporting.

If you want to disable this safety check for a specific custom call,
e.g., with target `my_target`, you can add
`export.DisabledSafetyCheck.custom_call("my_target")` to the
`disabled_checks` parameter of the `export` method,
as in the following example:

```python
>>> import jax
>>> from jax import export
>>> from jax import lax
>>> from jax._src import core
>>> from jax._src.interpreters import mlir
>>> # Define a new primitive backed by a custom call
>>> new_prim = core.Primitive("new_prim")
>>> _ = new_prim.def_abstract_eval(lambda x: x)
>>> _ = mlir.register_lowering(new_prim, lambda ctx, o: mlir.custom_call("my_new_prim", operands=[o], result_types=[o.type]).results)
>>> print(jax.jit(new_prim.bind).lower(1.).compiler_ir())
module @jit_bind attributes {mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} {
  func.func public @main(%arg0: tensor<f32> {mhlo.layout_mode = "default"}) -> (tensor<f32> {jax.result_info = "", mhlo.layout_mode = "default"}) {
    %0 = stablehlo.custom_call @my_new_prim(%arg0) {api_version = 2 : i32} : (tensor<f32>) -> tensor<f32>
    return %0 : tensor<f32>
  }
}

>>> # If we try to export, we get an error
>>> export.export(jax.jit(new_prim.bind))(1.)  # doctest: +IGNORE_EXCEPTION_DETAIL
Traceback (most recent call last):
ValueError: Cannot serialize code with custom calls whose targets have no compatibility guarantees: my_new_bind

>>> # We can avoid the error if we pass a `DisabledSafetyCheck.custom_call`
>>> exp = export.export(
...    jax.jit(new_prim.bind),
...    disabled_checks=[export.DisabledSafetyCheck.custom_call("my_new_prim")])(1.)

```

## Cross-platform and multi-platform export

JAX lowering is platform specific for a small number of JAX primitives.
By default, the code is lowered and exported for the accelerator
present on the exporting machine:

```python
>>> from jax import export
>>> export.default_export_platform()
'cpu'

```

There is a safety check that will be raise an error when trying to compile
an `Exported` object on a machine that does not have the accelerator
for which the code was exported.

You can specify explicitly for what platforms the code should be exported.
This allows you to specify a different accelerator than you have
available at export time,
and it even allows you to specify multi-platform lexport to
obtain an `Exported` object that can be compiled and executed
on multiple platforms.


```python
>>> import jax
>>> from jax import export
>>> from jax import lax

>>> # You can specify the export platform, e.g., `tpu`, `cpu`, `cuda`, `rocm`
>>> # even if the current machine does not have that accelerator.
>>> exp = export.export(jax.jit(lax.cos), platforms=['tpu'])(1.)

>>> # But you will get an error if you try to compile `exp`
>>> # on a machine that does not have TPUs.
>>> exp.call(1.)  # doctest: +IGNORE_EXCEPTION_DETAIL
Traceback (most recent call last):
ValueError: Function 'cos' was lowered for platforms '('tpu',)' but it is used on '('cpu',)'.

>>> # We can avoid the error if we pass a `DisabledSafetyCheck.platform`
>>> # parameter to `export`, e.g., because you have reasons to believe
>>> # that the code lowered will run adequately on the current
>>> # compilation platform (which is the case for `cos` in this
>>> # example):
>>> exp_unsafe = export.export(jax.jit(lax.cos),
...    lowering_platforms=['tpu'],
...    disabled_checks=[export.DisabledSafetyCheck.platform()])(1.)

>>> exp_unsafe.call(1.)
Array(0.5403023, dtype=float32, weak_type=True)

# and similarly with multi-platform lowering
>>> exp_multi = export.export(jax.jit(lax.cos),
...    lowering_platforms=['tpu', 'cpu', 'cuda'])(1.)
>>> exp_multi.call(1.)
Array(0.5403023, dtype=float32, weak_type=True)

```

For multi-platform export, the StableHLO will contain multiple
lowerings but only for those primitives that require it, so the
resulting module size should be only marginally larger than the
size of a module with default export.
As an extreme case, when serializing a module without any
primitives with platform-specific lowering, you will get
the same StableHLO as for the single-plaform export.

```python
>>> import jax
>>> from jax import export
>>> from jax import lax
>>> # A largish function
>>> def f(x):
...   for i in range(1000):
...     x = jnp.cos(x)
...   return x

>>> exp_single = export.export(jax.jit(f))(1.)
>>> len(exp_single.mlir_module_serialized)  # doctest: +SKIP
9220

>>> exp_multi = export.export(jax.jit(f),
...                           lowering_platforms=["cpu", "tpu", "cuda"])(1.)
>>> len(exp_multi.mlir_module_serialized)  # doctest: +SKIP
9282

```

## Shape polymorphic export

When used in JIT mode, JAX will trace and lower a function separately
for each combination of input shapes. When exporting, it is possible
in some cases to use dimension variables for some input dimensions
in order to obtain an exported artifact that can be used with multiple
combinations of input shapes.

See the {ref}`shape_poly` documentation.

## Device polymorphic export

An exported artifact may contain sharding annotations for inputs,
outputs and for some intermediates, but these annotations do not refer
directly to the actual physical devices that existed at exporting time.
Instead, the sharding annotations refer to logical devices. This
means that you can compile and run the exported artifacts on different
physical devices that were used for exporting.

```python
>>> import jax
>>> from jax import export
>>> from jax.sharding import Mesh, NamedSharding
>>> from jax.sharding import PartitionSpec as P

>>> # Use the first 4 devices for exporting.
>>> export_devices = jax.local_devices()[:4]
>>> export_mesh = Mesh(export_devices, ("a",))
>>> def f(x):
...   return x.T

>>> arg = jnp.arange(8 * len(export_devices))
>>> exp = export.export(jax.jit(f, in_shardings=(NamedSharding(export_mesh, P("a")),)))(arg)

>>> # `exp` knows for how many devices it was exported.
>>> exp.nr_devices
4

>>> # and it knows the shardings for the inputs. These will be applied
>>> # when the exported is called.
>>> exp.in_shardings_hlo
({devices=[4]<=[4]},)

>>> res1 = exp.call(jax.device_put(arg,
...                                NamedSharding(export_mesh, P("a"))))

>>> # Check out the first 2 shards of the result
>>> [f"device={s.device} index={s.index}" for s in res1.addressable_shards[:2]]
['device=TFRT_CPU_0 index=(slice(0, 8, None),)',
 'device=TFRT_CPU_1 index=(slice(8, 16, None),)']

>>> # We can call `exp` with some other 4 devices and another
>>> # mesh with a different shape, as long as the number of devices is
>>> # the same.
>>> other_mesh = Mesh(np.array(jax.local_devices()[2:6]).reshape((2, 2)), ("b", "c"))
>>> res2 = exp.call(jax.device_put(arg,
...                                NamedSharding(other_mesh, P("b"))))

>>> # Check out the first 2 shards of the result. Notice that the output is
>>> # sharded similarly; this means that the input was resharded according to the
>>> # exp.in_shardings.
>>> [f"device={s.device} index={s.index}" for s in res2.addressable_shards[:2]]
['device=TFRT_CPU_2 index=(slice(0, 8, None),)',
 'device=TFRT_CPU_3 index=(slice(8, 16, None),)']

```

It is an error to try to invoke an exported artifact with a different number
of devices than it was exported for:

```python
>>> import jax
>>> from jax import export
>>> from jax.sharding import Mesh, NamedSharding
>>> from jax.sharding import PartitionSpec as P

>>> export_devices = jax.local_devices()
>>> export_mesh = Mesh(np.array(export_devices), ("a",))
>>> def f(x):
...   return x.T

>>> arg = jnp.arange(4 * len(export_devices))
>>> exp = export.export(jax.jit(f, in_shardings=(NamedSharding(export_mesh, P("a")),)))(arg)

>>> exp.call(arg)  # doctest: +IGNORE_EXCEPTION_DETAIL
Traceback (most recent call last):
ValueError: Exported module f was lowered for 8 devices and is called in a context with 1 devices. This is disallowed because: the module was lowered for more than 1 device.

```

There are helper functions to shard the inputs for calling an exported
artifacts using a new mesh constructed at the call site:

```python
>>> import jax
>>> from jax import export
>>> from jax.sharding import Mesh, NamedSharding
>>> from jax.sharding import PartitionSpec as P

>>> export_devices = jax.local_devices()
>>> export_mesh = Mesh(np.array(export_devices), ("a",))
>>> def f(x):
...   return x.T

>>> arg = jnp.arange(4 * len(export_devices))
>>> exp = export.export(jax.jit(f, in_shardings=(NamedSharding(export_mesh, P("a")),)))(arg)

>>> # Prepare the mesh for calling `exp`.
>>> calling_mesh = Mesh(np.array(export_devices[::-1]), ("b",))

>>> # Shard the arg according to what `exp` expects.
>>> sharded_arg = jax.device_put(arg, exp.in_shardings_jax(calling_mesh)[0])
>>> res = exp.call(sharded_arg)

```

As a special facility, if a function was exported for 1 device and if it contains no
sharding annotations, then it can be invoked on an argument of the same shape but sharded
on multiple devices, and the compiler will shard the function appropriately:

```python
```python
>>> import jax
>>> from jax import export
>>> from jax.sharding import Mesh, NamedSharding
>>> from jax.sharding import PartitionSpec as P

>>> def f(x):
...   return jnp.cos(x)

>>> arg = jnp.arange(4)
>>> exp = export.export(jax.jit(f))(arg)
>>> exp.in_avals
(ShapedArray(int32[4]),)

>>> exp.nr_devices
1

>>> # Prepare the mesh for calling `exp`.
>>> calling_mesh = Mesh(jax.local_devices()[:4], ("b",))

>>> # Shard the arg according to what `exp` expects.
>>> sharded_arg = jax.device_put(arg,
...                              NamedSharding(calling_mesh, P("b")))
>>> res = exp.call(sharded_arg)

```

## Calling convention versions

The JAX export support has evolved over time, e.g., to support
effects. In order to support compatibility (see [compatibility guarantees](#compatibility-guarantees))
we maintain a calling convention version for each `Exported`.
As of June 2024, all function exported with version 9
(the latest, see [all calling convention versions](#calling-convention-versions)):

```python
>>> from jax import export
>>> exp: export.Exported = export.export(jnp.cos)(1.)
>>> exp.calling_convention_version
9

```

At any given time, the export APIs may support a range
of calling convention versions. You can control which calling convention
version to use using the `--jax-export-calling-convention-version` flag
or the `JAX_EXPORT_CALLING_CONVENTION_VERSION` environment variable:

```python
>>> from jax import export
>>> (export.minimum_supported_calling_convention_version, export.maximum_supported_calling_convention_version)
(9, 9)

>>> from jax._src import config
>>> with config.jax_export_calling_convention_version(9):
...  exp = export.export(jnp.cos)(1.)
...  exp.calling_convention_version
9

```

We reserve the right to remove support for
generating or consuming calling convention versions older than 6 months.

### Module calling convention

The `Exported.mlir_module` has a `main` function that takes an optional first
platform index argument if the module supports multiple platforms
(`len(platforms) > 1`), followed by the token arguments corresponding
to the ordered effects, followed by the kept array
arguments (corresponding to `module_kept_var_idx` and `in_avals`).
The platform index is a i32 or i64 scalar encoding the index of the current
compilation platform into the `platforms` sequence.

Inner functions use a different calling convention: an optional
platform index argument, optional dimension variable arguments
(scalar tensors of type i32 or i64),
followed by optional token arguments (in presence of ordered effects),
followed by the regular array arguments.
The dimension arguments correspond to the dimension variables appearing in
the `args_avals`, in sorted order of their names.

Consider the lowering of a function with one array argument of type
`f32[w, 2 * h]`, where `w` and `h` are two dimension variables.
Assume that we use multi-platform lowering, and we have
one ordered effect. The `main` function will be as follows:

```
      func public main(
            platform_index: i32 {jax.global_constant="_platform_index"},
            token_in: token,
            arg: f32[?, ?]) {
         arg_w = hlo.get_dimension_size(arg, 0)
         dim1 = hlo.get_dimension_size(arg, 1)
         arg_h = hlo.floordiv(dim1, 2)
         call _check_shape_assertions(arg)  # See below
         token = new_token()
         token_out, res = call _wrapped_jax_export_main(platform_index,
                                                        arg_h,
                                                        arg_w,
                                                        token_in,
                                                        arg)
         return token_out, res
      }
```

The actual computation is in `_wrapped_jax_export_main`, taking also
the values of `h` and `w` dimension variables.

The signature of the `_wrapped_jax_export_main` is:

```
      func private _wrapped_jax_export_main(
          platform_index: i32 {jax.global_constant="_platform_index"},
          arg_h: i32 {jax.global_constant="h"},
          arg_w: i32 {jax.global_constant="w"},
          arg_token: stablehlo.token {jax.token=True},
          arg: f32[?, ?]) -> (stablehlo.token, ...)
```

Prior to calling convention version 9 the calling convention for effects was
different: the `main` function does not take or return a token. Instead
the function creates dummy tokens of type `i1[0]` and passes them to the
`_wrapped_jax_export_main`. The `_wrapped_jax_export_main`
takes dummy tokens of type `i1[0]` and will create internally real
tokens to pass to the inner functions. The inner functions use real
tokens (both before and after calling convention version 9)

Also starting with calling convention version 9, function arguments that contain
the platform index or the dimension variable values have a
`jax.global_constant` string attribute whose value is the name of the
global constant, either `_platform_index` or a dimension variable name.
The global constant name may be empty if it is not known.
Some global constant computations use inner functions, e.g., for
`floor_divide`. The arguments of such functions have a `jax.global_constant`
attribute for all attributes, meaning that the result of the function is
also a global constant.

Note that `main` contains a call to `_check_shape_assertions`.
JAX tracing assumes that `arg.shape[1]` is even, and that both `w` and `h`
have values >= 1. We must check these constraints when we invoke the
module. We use a special custom call `@shape_assertion` that takes
a boolean first operand, a string `error_message` attribute that may contain
format specifiers `{0}`, `{1}`, ..., and a variadic number of integer
scalar operands corresponding to the format specifiers.

```
       func private _check_shape_assertions(arg: f32[?, ?]) {
         # Check that w is >= 1
         arg_w = hlo.get_dimension_size(arg, 0)
         custom_call @shape_assertion(arg_w >= 1, arg_w,
            error_message="Dimension variable 'w' must have integer value >= 1. Found {0}")
         # Check that dim1 is even
         dim1 = hlo.get_dimension_size(arg, 1)
         custom_call @shape_assertion(dim1 % 2 == 0, dim1,
            error_message="Dimension variable 'h' must have integer value >= 1. Found non-zero remainder {0}")
         # Check that h >= 1
         arg_h = hlo.floordiv(dim1, 2)
         custom_call @shape_assertion(arg_h >= 1, arg_h,
            error_message=""Dimension variable 'h' must have integer value >= 1. Found {0}")
```

(export-calling-convention-version)=

### Calling convention versions

We list here a history of the calling convention version numbers:

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
    In this calling convention version we also tag the platform index and the
    dimension variables arguments with `jax.global_constant` attributes.
    Supported by XlaCallModule since October 27th, 2023,
    available in JAX since October 20th, 2023 (JAX 0.4.20),
    and the default since February 1st, 2024 (JAX 0.4.24).
    This is the only supported version as of 27th of March, 2024.


## Migration guide from jax.experimental.export

On June 14, 2024 we deprecated the `jax.experimental.export` APIs
in favor of `jax.export` APIs. There have been some minor changes:

  * `jax.experimental.export.export`:
    * The old function used to allow any Python callable, or the result of
      `jax.jit`. Now only the latter is accepted. You have to manually apply
      `jax.jit` to the function to export before calling `export`.
    * The old `lowering_parameters` kwarg is now named `platforms`
  * `jax.experimental.export.default_lowering_platform()` is now
    at {func}`jax.export.default_export_platform`.
  * `jax.experimental.export.call` is now a method of the {class}`jax.export.Exported` object.
    Instead of `export.call(exp)` you should use `exp.call`.
  * `jax.experimental.export.serialize` is now a method of the {class}`jax.export.Exported`
    object. Instead of `export.serialize(exp)` you should use `exp.serialize()`.
  * The configuration flag `--jax-serialization-version` is deprecated.
    Use `--jax-export-calling-convention-version`.
  * The value `jax.experimental.export.minimum_supported_serialization_version`
    is now at `jax.export.minimum_supported_calling_convention_version`.
  * The following fields of {class}`jax.export.Exported` have been renamed
     * `uses_shape_polymorphism` is now `uses_global_constants`
     * `mlir_module_serialization_version` is now `calling_convention_version`
     * `lowering_platforms` is now `platforms`.



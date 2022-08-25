# Copyright 2021 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Allows JAX to call TensorFlow functions with support for autodiff.

**Experimental: please give feedback, and expect changes.**

This module introduces the function :func:`call_tf` that allows JAX to call
TensorFlow functions.

For examples and details, see
https://github.com/google/jax/blob/main/jax/experimental/jax2tf/README.md#calling-tensorflow-functions-from-jax.

"""
import functools
import logging
from typing import Any, Callable, Optional, Sequence, Tuple

import jax
from jax import core
from jax import dlpack
from jax import dtypes
from jax import tree_util
from jax._src import util
from jax._src import ad_util
from jax.interpreters import mlir
from jax.interpreters import xla
from jax._src.lib.mlir import ir
from jax._src.lib.mlir.dialects import func as func_dialect
from jax._src.lib.mlir.dialects import mhlo
from jax._src.lib import xla_client
from jax.experimental.jax2tf import jax2tf as jax2tf_internal

import numpy as np
import jax.experimental.jax2tf._tf_deps as tf  # type: ignore[import]

map = util.safe_map
zip = util.safe_zip

TfConcreteFunction = Any

# The platforms for which to use DLPack to avoid copying (only works on GPU
# and CPU at the moment, and only for DeviceArray). For CPU we don't need
# DLPack, if we are careful.
_DLPACK_PLATFORMS = ("gpu",)

def call_tf(callable_tf: Callable) -> Callable:
  """Calls a TensorFlow function from JAX, with support for reverse autodiff.

  The ``callable_tf`` will be called with TensorFlow-compatible arguments (
  numpy.ndarray, ``tf.Tensor`` or ``tf.Variable``) or pytrees thereof. The
  function must return the same type of results.

  If ``call_tf`` appears in a JAX staging context (:func:`jax.jit`,
  or :func:`jax.pmap`, or :func:`jax.xmap`, or a control-flow primitive) then
  ``callable_tf`` will be compiled with ``tf.function(callable_tf, jit_compile=True)``
  and the resulting XLA computation will be embedded in JAX's XLA computation.

  If ``call_tf`` appears outside a JAX staging context, it will be called inline
  using TensorFlow eager mode.

  The ``call_tf`` supports JAX's reverse-mode autodiff, in which case the
  ``callable_tf`` will be differentiated using ``tf.GradientTape``. This means
  that the gradient will be TensorFlow-accurate, e.g., will respect the
  custom gradients that may be defined for the code in ``callable_tf``.

  For an example and more details see the
  `README <https://github.com/google/jax/blob/main/jax/experimental/jax2tf/README.md#calling-tensorflow-functions-from-jax>`_.

  Args:
    callable_tf: a TensorFlow Callable that can take a pytree of TensorFlow
      arguments.
  Returns: a JAX callable that can be invoked with JAX pytree arguments, in
    op-by-op mode or in a staged context. This callable can be used with
    JAX's reverse-mode autodiff (:func:`jax.grad`).
  """

  @jax.custom_vjp
  def make_call(*args_jax):
    """We wrap it all in `make_call` so that we can attach custom VJP."""

    args_flat_jax, args_treedef = tree_util.tree_flatten(args_jax)
    # Canonicalize the arguments; e.g., makes them x32 if JAX is in 32-bit mode
    def canonical_arg(v):
      v = v if getattr(v, "dtype", None) else np.asarray(v)
      dtype = dtypes.canonicalize_dtype(v.dtype)
      if dtype != v.dtype:
        v = v.astype(dtype)
      return v

    args_flat_jax = tuple(map(canonical_arg, args_flat_jax))
    def make_tensorspec(a_jax):
      a_tf_dtype = jax2tf_internal._to_tf_dtype(a_jax.dtype)
      if any(not core.is_constant_dim(d) for d in a_jax.shape):
        msg = ("call_tf cannot be applied to shape-polymorphic arguments. "
               f"Found argument shape: {a_jax.shape}. "
               "See https://github.com/google/jax/blob/main/jax/experimental/jax2tf/README.md#limitations-of-call_tf for a discussion.")
        raise ValueError(msg)

      return tf.TensorSpec(a_jax.shape, a_tf_dtype)
    args_flat_sig_tf = tuple(map(make_tensorspec, args_flat_jax))

    res_treedef = None  # We'll store here the result treedef
    # The function below will be called at least once, either in eager
    # or in graph mode.
    def callable_flat_tf(*args_tf_flat: TfVal) -> Sequence[TfVal]:
      args_tf = args_treedef.unflatten(args_tf_flat)
      res_tf = callable_tf(*args_tf)
      nonlocal res_treedef
      res_tf_flat, res_treedef_now = tree_util.tree_flatten(res_tf)
      assert res_treedef is None or res_treedef == res_treedef_now, f"Subsequent calls had different results. Previous {res_treedef} and now {res_treedef_now}"
      res_treedef = res_treedef_now
      return res_tf_flat

    # Prepare a tf.function ahead of time, to cache the concrete functions. This
    # won't be used in op-by-op execution mode.
    function_flat_tf = tf.function(callable_flat_tf, autograph=False, jit_compile=True)

    res_jax_flat = call_tf_p.bind(
        *args_flat_jax,
        # Carry the actual function such that op-by-op call can call in TF eager mode.
        callable_flat_tf=callable_flat_tf,
        function_flat_tf=function_flat_tf,
        args_flat_sig_tf=args_flat_sig_tf)
    return res_treedef.unflatten(res_jax_flat)

  # Define the fwd and bwd custom_vjp functions
  def make_call_vjp_fwd(*args_jax):
    # Return the primal arguments as the residual
    return make_call(*args_jax), args_jax

  def make_call_vjp_bwd(residual_jax, ct_res_jax):
    args_jax = residual_jax  # residual is the primal argument

    def tf_vjp_fun(args_tf, ct_res_tf):
      """Invoke TF gradient."""

      # TF does not like us to watch non-float vars
      def replace_non_float(arg):
        if np.issubdtype(arg.dtype.as_numpy_dtype, np.inexact):
          return arg
        else:
          # When watched, this will be ignored. When use in results it will
          # result in a floating 0. gradient, which JAX will ignore (and
          # replace it with a float0)
          return tf.zeros((), dtype=tf.float32)

      watched_args_tf = tf.nest_map_structure(replace_non_float, args_tf)
      with tf.GradientTape(persistent=True) as tape:
        tape.watch(watched_args_tf)
        res = callable_tf(*args_tf)

      tf.nest_assert_same_structure(res, ct_res_tf)
      dres_darg = tape.gradient(
          tf.nest_map_structure(replace_non_float, res),
          sources=watched_args_tf,
          output_gradients=ct_res_tf,
          unconnected_gradients=tf.UnconnectedGradients_ZERO)

      tf.nest_assert_same_structure(dres_darg, args_tf)
      return dres_darg

    # Use call_tf to call the VJP function
    ct_args_jax = call_tf(tf_vjp_fun)(args_jax, ct_res_jax)
    # We must make the float0s that JAX expects
    def fix_float0(arg_jax, ct_arg_jax):
      arg_dtype = dtypes.result_type(arg_jax)  # May be scalar
      ct_arg_dtype = core.primal_dtype_to_tangent_dtype(arg_dtype)
      if ct_arg_dtype != ct_arg_jax.dtype:
        return ad_util.zeros_like_aval(core.ShapedArray(np.shape(arg_jax),
                                                        ct_arg_dtype))
      return ct_arg_jax

    ct_args_jax_fixed = tree_util.tree_map(fix_float0, args_jax, ct_args_jax)
    return ct_args_jax_fixed

  make_call.defvjp(make_call_vjp_fwd, make_call_vjp_bwd)
  return util.wraps(callable_tf)(make_call)


call_tf_p = core.Primitive("call_tf")
call_tf_p.multiple_results = True

# The impl will be used in op-by-op mode and calls callable_tf in TF eager mode.
def _call_tf_impl(*args_jax_flat, callable_flat_tf, **_):
  # On GPU we use dlpack to avoid copies of data to the host.
  def _arg_jax_to_tf(arg_jax):
    if (isinstance(arg_jax, xla.DeviceArray) and
        arg_jax.device_buffer.client.platform in _DLPACK_PLATFORMS and
        arg_jax.dtype in dlpack.SUPPORTED_DTYPES):
      arg_dlpack = jax.dlpack.to_dlpack(arg_jax, take_ownership=False)
      return tf.experimental_dlpack_from_dlpack(arg_dlpack)
    # The following avoids copies to the host on CPU, always for DeviceArray
    # and even for ndarray if they are sufficiently aligned.
    # TODO(necula): on TPU this copies to the host!
    return tf.constant(np.asarray(arg_jax))

  args_tf_flat = tuple(map(_arg_jax_to_tf, args_jax_flat))
  with jax2tf_internal.inside_call_tf():
    # Call in TF eager mode
    res_tf_flat = callable_flat_tf(*args_tf_flat)

  def _res_tf_to_jax(res_tf: TfVal):
    res_tf, _ = jax2tf_internal._tfval_to_tensor_jax_dtype(res_tf)
    if isinstance(res_tf, tf.Tensor) and res_tf.dtype in dlpack.SUPPORTED_DTYPES:
      res_tf_platform = tf.DeviceSpec_from_string(res_tf.backing_device).device_type
      res_jax_platform = res_tf_platform.lower()
      if res_jax_platform in _DLPACK_PLATFORMS:
        res_dlpack = tf.experimental_dlpack_to_dlpack(res_tf)
        return jax.dlpack.from_dlpack(res_dlpack)

    return jax.device_put(np.asarray(res_tf))

  return list(map(_res_tf_to_jax, res_tf_flat))


call_tf_p.def_impl(_call_tf_impl)


def _call_tf_abstract_eval(*_,
                           function_flat_tf,
                           args_flat_sig_tf, **__):
  # See comments in _code_generator_and_avals of why we overkill and do a
  # full compilation only to get the abstract avals.
  _, result_avals = _code_generator_and_avals(function_flat_tf, args_flat_sig_tf,
                                              code_gen_optional=True)
  return tuple(result_avals)

call_tf_p.def_abstract_eval(_call_tf_abstract_eval)


def _call_tf_lowering(ctx, *args_op, function_flat_tf, args_flat_sig_tf, **_):
  # This will most likely hit the cache, because we used it for abstract_eval
  code_gen, _ = _code_generator_and_avals(function_flat_tf, args_flat_sig_tf,  # type: ignore
                                          code_gen_optional=False)
  assert code_gen is not None
  return code_gen(ctx.module_context, args_op)


@functools.lru_cache(maxsize=128)
def _code_generator_and_avals(
    function_flat_tf,
    args_flat_sig_tf,
    code_gen_optional=False
) -> Tuple[Optional[Callable[[mlir.ModuleContext, Sequence[ir.Value]],
                             Sequence[ir.Value]]],
           Sequence[core.ShapedArray]]:
  # Returns and caches a code generator (taking a builder and the
  # XlaOps for the arguments) and a sequence of result abstract shapes.

  # It turns out that both for abstract evaluation and for actual compilation
  # it is useful to actually generate the HLO. This is true because in some
  # cases just TF-level shape inference is not precise enough to recover the
  # output shapes (e.g., b/128924522), even in situations where XLA can compile
  # the code, from which we can get the shapes.

  # Due to bugs like b/193754660, the compilation may fail. To work around this
  # issue we pass the `code_gen_optional` when in an abstract evaluation context
  # in which case we fallback on TF shape inference. Luckily it seen that
  # it is never the case that we are under tf.function, and we call the
  # XLA translation rule for call_tf. The latter happens only for jax.jit, but
  # jax.jit under a tf.function must be under jax2tf.convert, which unrolls
  # the jit.

  # TODO(necula): It seems that we need concrete tensors for get_compiler_ir?
  # We know of one case when TF is sensitive to the values of the tensors that
  # affect shapes in the computation. In those cases, however, those tensors
  # are inlined in the computation, which we detect below.
  args_tf_flat = [
      tf.constant((0 if a.dtype != tf.bool else False),
                  shape=a.shape,
                  dtype=a.dtype) for a in args_flat_sig_tf]

  # TODO(necula): We should use the proper device, because in some cases we
  # generate different HLO for different devices.
  # One example is when the code refers to variables on one device. Or, for
  # sharding annotations (only supported on TPU).
  # For now we just use the default device, but ideally we should pass the
  # intended platform in. The problem is that we want to reuse and cache this
  # function across abstract_eval and XLA translation, but for abstract_eval
  # we do not know the platform.
  tf_device_name = f"/device:{jax.default_backend().upper()}:0"
  with jax2tf_internal.inside_call_tf():
    concrete_function_flat_tf = function_flat_tf.get_concrete_function(*args_flat_sig_tf)

  captured_inputs = []
  if concrete_function_flat_tf.captured_inputs:
    # The function uses either captured variables or tensors.
    msg = (
        "call_tf works best with a TensorFlow function that does not capture "
        "variables or tensors from the context. "
        "See https://github.com/google/jax/blob/main/jax/experimental/jax2tf/README.md#limitations-of-call_tf for a discussion. "
        f"The following captures were found {concrete_function_flat_tf.captured_inputs}")
    logging.warning(msg)
    for inp in concrete_function_flat_tf.captured_inputs:
      if inp.dtype == tf.resource:  # A variable; lookup by handle
        inp_vars = [v for v in concrete_function_flat_tf.variables if inp is v.handle]
        assert len(inp_vars) == 1, f"Found {inp_vars}"
        captured_inputs.append(inp_vars[0])
      else:
        captured_inputs.append(inp)

  with jax2tf_internal.inside_call_tf():
    # The above has traced the function and in fact has cached a ConcreteFunction
    # Grab it now, so that we don't have to construct `args_tf_flat` only to
    # get a cache hit.
    try:
      func_tf_hlo = function_flat_tf.experimental_get_compiler_ir(*args_tf_flat)(
            stage="hlo_serialized", device_name=tf_device_name)
    except Exception as e:
      # TODO(b/193754660): This is a workaround. Use a more robust mechanism
      # instead of relying on error message.
      # Check two different error messages, to ensure the code works internally
      # (with "out of scope") and also in OSS (with "An op outside ...").
      if type(e) is TypeError and ("out of scope" in str(e) or
                                   "An op outside of the function building code" in str(e)):
        # TODO(b/193754660): this may happen if we are in a function context
        # Try to salvage the situation if we are just doing abstract_eval, maybe
        # for jax2tf.convert. We can do that if all the output_shapes are known.
        def is_fully_known_shape(s):
          return s.rank is not None and all([d is not None for d in s])
        if code_gen_optional and (
            all([is_fully_known_shape(s)
                 for s in concrete_function_flat_tf.output_shapes])):
          result_avals = [
              # We convert to JAX type, and canonicalize to 32-bit if necessary
              core.ShapedArray(shape, jax2tf_internal._to_jax_dtype(dtype))
              for dtype, shape in zip(concrete_function_flat_tf.output_dtypes,
                                      concrete_function_flat_tf.output_shapes)]
          return None, result_avals
      msg = ("Error compiling TensorFlow function. call_tf can used " +
             "in a staged context (under jax.jit, lax.scan, etc.) only with " +
             "compileable functions with static output shapes. " +
             "See https://github.com/google/jax/blob/main/jax/experimental/jax2tf/README.md#limitations-of-call_tf for a discussion.")
      raise ValueError(msg) from e

  xla_comp = xla_client.XlaComputation(func_tf_hlo)
  # Check that the function does not have compile-time constant inputs that
  # have been inlined in the compiled code.
  xla_comp_parameter_shapes = xla_comp.program_shape().parameter_shapes()
  found_parameter_avals = [
      core.ShapedArray(found_xla_shape.dimensions(),
                       dtypes.canonicalize_dtype(found_xla_shape.numpy_dtype()))
      for found_xla_shape in xla_comp_parameter_shapes
  ]
  # Add the captured_inputs to args_flat_sig_tf
  expected_args_flat_sig_tf = list(args_flat_sig_tf) + list(captured_inputs)
  expected_parameter_avals = [
      core.ShapedArray(tuple(arg_sig.shape.as_list()),
                       dtypes.canonicalize_dtype(arg_sig.dtype.as_numpy_dtype))
      for arg_sig in expected_args_flat_sig_tf]
  if found_parameter_avals != expected_parameter_avals:
    msg = ("Compiled TensorFlow function has unexpected parameter types " +
           f"{found_parameter_avals}, while the expected types are " +
           f"{expected_parameter_avals}. Perhaps the TensorFlow function " +
           "has shape-influencing inputs, and thus needs to be recompiled " +
           "for each value of some inputs. " +
           "See https://github.com/google/jax/blob/main/jax/experimental/jax2tf/README.md#limitations-of-call_tf for a discussion.")
    raise ValueError(msg)

  # Canonicalize the results; e.g., makes them x32 if JAX is in 32-bit mode
  def canonical_res_aval(res_shape: xla.XlaShape) -> core.ShapedArray:
    if not res_shape.is_static():
      msg = ("Compiled TensorFlow function has dynamic output shape " +
             f"{res_shape}. call_tf can used " +
             "in a staged context (under jax.jit, lax.scan, etc.) only with " +
             "compileable functions with static output shapes. " +
             "See https://github.com/google/jax/blob/main/jax/experimental/jax2tf/README.md#limitations-of-call_tf for a discussion.")
      raise ValueError(msg)

    res_dtype = res_shape.numpy_dtype()
    jax_res_dtype = dtypes.canonicalize_dtype(res_dtype)
    return core.ShapedArray(res_shape.dimensions(), jax_res_dtype)

  result_shape = xla_comp.program_shape().result_shape()
  if not result_shape.is_tuple():
    # TF does not wrap singletons as tuples, but JAX expects tuples because
    # call_tf is a multiple_results primitive.
    result_shapes = (result_shape,)
  else:
    result_shapes = result_shape.tuple_shapes()  # type: ignore

  result_avals = tuple(map(canonical_res_aval, result_shapes))  # type: ignore

  def code_gen(ctx: mlir.ModuleContext, args_op: Sequence[ir.Value]
              ) -> Sequence[ir.Value]:
    captured_ops = tuple(mlir.ir_constant(np.asarray(inp),
                                          canonicalize_types=False)
                         for inp in captured_inputs)
    submodule = mlir.xla_computation_to_mhlo_module(xla_comp)
    symtab = ir.SymbolTable(submodule.operation)
    callee_result_types = symtab["main"].type.results
    fn = mlir.merge_mhlo_modules(ctx.module, f"call_tf_{function_flat_tf.name}",
                                 submodule)
    call = func_dialect.CallOp(callee_result_types,
                               ir.FlatSymbolRefAttr.get(fn),
                               tuple(args_op) + captured_ops)
    if result_shape.is_tuple():
      flat_results = [mhlo.GetTupleElementOp(call, mlir.i32_attr(i)).result
                      for i in range(len(result_shapes))]
    else:
      flat_results = call.results

    outputs = []
    for op, res_aval, res_shape in zip(flat_results, result_avals,
                                       result_shapes):
      if res_aval.dtype != res_shape.numpy_dtype():
        op = mhlo.ConvertOp(mlir.aval_to_ir_type(res_aval), op).result
      outputs.append(op)
    return outputs

  return code_gen, result_avals

mlir.register_lowering(call_tf_p, _call_tf_lowering)

TfVal = jax2tf_internal.TfVal
def _jax2tf_call_tf(*args: TfVal,
                    callable_flat_tf: Callable,
                    **_) -> TfVal:
  with jax2tf_internal.inside_call_tf():
    res_tf_flat = callable_flat_tf(*args)
  return res_tf_flat

jax2tf_internal.tf_impl[call_tf_p] = _jax2tf_call_tf

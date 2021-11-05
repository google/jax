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

from contextlib import contextmanager
from dataclasses import dataclass
from functools import partial
import itertools as it
from typing import Union, Optional, Callable, Dict

import numpy as np

import jax.numpy as jnp

from jax import core
from jax import linear_util as lu
from jax.api_util import flatten_fun
from jax.interpreters import partial_eval as pe
from jax.interpreters import pxla
from jax.interpreters import xla
from jax.tree_util import tree_flatten, tree_unflatten, register_pytree_node
from jax._src import source_info_util, traceback_util
from jax._src.lax import lax
from jax._src.util import as_hashable_function, unzip2

source_info_util.register_exclusion(__file__)
traceback_util.register_exclusion(__file__)


## Utils

def popattr(obj, attrname):
  val = getattr(obj, attrname)
  delattr(obj, attrname)
  return val

def setnewattr(obj, name, val):
  sentinel = object()
  assert getattr(obj, name, sentinel) is sentinel
  setattr(obj, name, val)

## Error value data type and functional assert.

@dataclass(frozen=True)
class Error:
  err: Union[bool, core.Tracer]
  code: Union[int, core.Tracer]
  msgs: Dict[int, str]

  def get(self) -> Optional[str]:
    assert np.shape(self.err) == np.shape(self.code)
    if np.size(self.err) == 1:
      if self.err:
        return self.msgs[int(self.code)]
    else:
      return '\n'.join(f'at mapped index {", ".join(map(str, idx))}: '  # type: ignore
                       f'{self.msgs[int(self.code[idx])]}'              # type: ignore
                       for idx, e in np.ndenumerate(self.err) if e) or None
    return None

register_pytree_node(Error,
                     lambda e: ((e.err, e.code), tuple(sorted(e.msgs.items()))),
                     lambda msgs, data: Error(*data, dict(msgs)))  # type: ignore

init_error = Error(False, 0, {})
next_code = it.count(1).__next__  # globally unique ids


Bool = Union[bool, core.Tracer]
Int = Union[int, core.Tracer]

def assert_func(error: Error, pred: Bool, msg: str) -> Error:
  code = next_code()
  out_err = error.err | jnp.logical_not(pred)
  out_code = lax.select(error.err, error.code, code)
  return Error(out_err, out_code, {code: msg, **error.msgs})

## Checkify transformation for plumbing functional error values.

class ErrorTrace(core.Trace):
  pure = lift = sublift = lambda self, val: ErrorTracer(self, val)

  def process_primitive(self, primitive, tracers, params):
    in_vals = [t.val for t in tracers]
    rule = error_checks.get(primitive)
    if rule:
      out, self.main.error = rule(self.main.error, *in_vals, **params)
    else:
      out = primitive.bind(*in_vals, **params)
    if primitive.multiple_results:
      return [ErrorTracer(self, x) for x in out]
    else:
      return ErrorTracer(self, out)

  def process_call(self, primitive, f, tracers, params):
    in_vals = [t.val for t in tracers]
    e = popattr(self.main, 'error')
    f, msgs = check_errors_subtrace(f, self.main, tuple(e.msgs.items()))
    params_ = dict(params, donated_invars=(False, False, *params['donated_invars']))
    err, code, *out_vals = primitive.bind(f, e.err, e.code, *in_vals, **params_)
    setnewattr(self.main, 'error', Error(err, code, msgs()))
    return [ErrorTracer(self, x) for x in out_vals]

  def process_map(self, primitive, f, tracers, params):
    in_vals = [t.val for t in tracers]
    e = popattr(self.main, 'error')
    f, msgs = check_errors_subtrace(f, self.main, tuple(e.msgs.items()))

    @as_hashable_function(closure=params['out_axes_thunk'])
    def new_out_axes_thunk():
      return (0, 0, *params['out_axes_thunk']())

    params_ = dict(params, in_axes=(None, None, *params['in_axes']),
                   out_axes_thunk=new_out_axes_thunk,
                   donated_invars=(False, False, *params['donated_invars']))
    errs, codes, *outs = primitive.bind(f, e.err, e.code, *in_vals, **params_)
    err, code = _reduce_any_error(errs, codes)
    setnewattr(self.main, 'error', Error(err, code, msgs()))
    return [ErrorTracer(self, x) for x in outs]

  def post_process_call(self, primitive, tracers, params):
    vals = [t.val for t in tracers]
    main = self.main
    e = popattr(self.main, 'error')
    err, code, main.msgs = e.err, e.code, e.msgs
    def todo(vals):
      trace = main.with_cur_sublevel()
      err, code, *vals = vals
      if hasattr(main, 'msgs'):  # TODO(mattjj): conspiracy w/ CheckifyEvaluator
        setnewattr(main, 'error', Error(err, code, popattr(main, 'msgs')))
        return [ErrorTracer(trace, x) for x in vals]
      else:
        return [ErrorTracer(trace, x) for x in vals]
    return (err, code, *vals), todo

  def post_process_map(self, primitive, tracers, params):
    vals = [t.val for t in tracers]
    main = self.main
    e = popattr(self.main, 'error')
    err, code, main.msgs = e.err, e.code, e.msgs
    def todo(vals):
      trace = main.with_cur_sublevel()
      err, code, *vals = vals
      if hasattr(main, 'msgs'):  # TODO(mattjj): conspiracy w/ CheckifyEvaluator
        setnewattr(main, 'error', Error(err, code, popattr(main, 'msgs')))
        return [ErrorTracer(trace, x) for x in vals]
      else:
        return [ErrorTracer(trace, x) for x in vals]
    def out_axes_transform(out_axes):
      return (0, 0, *out_axes)
    return (err, code, *vals), (todo, out_axes_transform)

def _reduce_any_error(errs, codes):
  errs_, codes_ = lax.sort_key_val(errs, codes, dimension=0)
  return errs_[-1], codes_[-1]

ErrorCheckRule = Callable
error_checks: Dict[core.Primitive, ErrorCheckRule] = {}

class ErrorTracer(core.Tracer):
  def __init__(self, trace, val):
    self._trace = trace
    self.val = val
    core.get_aval(val), val
  aval = property(lambda self: core.get_aval(self.val))
  full_lower = lambda self: self

def check_errors_flat(fun: lu.WrappedFun, *args):
  fun, msgs = check_errors_subtrace(fun)
  fun = check_errors_toplevel(fun)
  err, code, *out_vals = fun.call_wrapped(*args)
  return (err, code, out_vals), msgs()

@lu.transformation
def check_errors_toplevel(*args):
  error = init_error
  with core.new_main(ErrorTrace) as main:
    msgs = tuple(error.msgs.items())
    outs = yield (main, msgs, error.err, error.code, *args), {}
    del main
  yield outs

@lu.transformation_with_aux
def check_errors_subtrace(main, msgs, err, code, *args):
  setnewattr(main, 'error', Error(err, code, dict(msgs)))
  trace = main.with_cur_sublevel()
  in_tracers = [ErrorTracer(trace, x) for x in args]
  out = yield in_tracers, {}
  out_tracers = map(trace.full_raise, out)
  out_vals = [t.val for t in out_tracers]
  err, code, msgs = main.error.err, main.error.code, main.error.msgs
  del main.error
  yield (err, code, *out_vals), msgs

def checkify_jaxpr(jaxpr, error):
  f = lu.wrap_init(core.jaxpr_as_fun(jaxpr))
  f, msgs = check_errors_subtrace(f)
  f = check_errors_traceable(f, tuple(error.msgs.items()))
  err_aval = core.raise_to_shaped(core.get_aval(error.err))
  code_aval = core.raise_to_shaped(core.get_aval(error.code))
  avals_in = [err_aval, code_aval, *jaxpr.in_avals]
  jaxpr_out, _, literals_out = pe.trace_to_jaxpr_dynamic(f, avals_in)
  return core.ClosedJaxpr(jaxpr_out, literals_out), msgs()

# TODO dedup with check_errors_toplevel
@lu.transformation
def check_errors_traceable(msgs, err, code, *args):
  with core.new_main(ErrorTrace) as main:
    outs = yield (main, msgs, err, code, *args), {}
    del main
  yield outs

## checkify rules

from jax._src.lax import control_flow

def nan_error_check(prim, error, *in_vals, **params):
  out = prim.bind(*in_vals, **params)
  no_nans = jnp.logical_not(jnp.any(jnp.isnan(out)))
  summary = source_info_util.summarize(source_info_util.current())
  msg = f"nan generated by primitive {prim.name} at {summary}"
  return out, assert_func(error, no_nans, msg)

error_checks[lax.sin_p] = partial(nan_error_check, lax.sin_p)
error_checks[lax.cos_p] = partial(nan_error_check, lax.cos_p)

def gather_error_check(error, operand, start_indices, *,
                       dimension_numbers, slice_sizes, unique_indices,
                       indices_are_sorted, mode, fill_value):
  out = lax.gather_p.bind(
      operand, start_indices, dimension_numbers=dimension_numbers,
      slice_sizes=slice_sizes, unique_indices=unique_indices,
      indices_are_sorted=indices_are_sorted, mode=mode, fill_value=fill_value)

  # compare to OOB masking logic in lax._gather_translation_rule
  dnums = dimension_numbers
  operand_dims = np.array(operand.shape)

  upper_bound = operand_dims[np.array(dnums.start_index_map)]
  upper_bound -= np.array(slice_sizes)[np.array(dnums.start_index_map)]
  all_inbounds = jnp.all((start_indices >= 0) & (start_indices <= upper_bound))

  summary = source_info_util.summarize(source_info_util.current())
  msg = f"out-of-bounds indexing at {summary}"
  return out, assert_func(error, all_inbounds, msg)
error_checks[lax.gather_p] = gather_error_check

def cond_error_check(error, index, *ops, branches, linear):
  new_branches, msgs_ = unzip2(checkify_jaxpr(jxpr, error) for jxpr in branches)
  new_linear = (False, False, *linear)
  err, code, *outs = control_flow.cond_p.bind(
      index, error.err, error.code, *ops,
      branches=tuple(new_branches), linear=new_linear)
  new_msgs = {k:v for d in it.chain([error.msgs], msgs_) for k, v in d.items()}
  return outs, Error(err, code, new_msgs)
error_checks[control_flow.cond_p] = cond_error_check


## checkify api

def checkify(fun: Callable) -> Callable:
  @traceback_util.api_boundary
  def checked_fun(*args, **kwargs):
    args_flat, in_tree = tree_flatten((args, kwargs))
    f, out_tree = flatten_fun(lu.wrap_init(fun), in_tree)
    (err, code, out_flat), msgs = check_errors_flat(f, *args_flat)
    out = tree_unflatten(out_tree(), out_flat)
    return Error(err, code, msgs), out
  return checked_fun

### adding checkify to all operations

class CheckifyEvaluator(core.Trace):
  def pure(self, x): return x
  lift = sublift = pure

  def process_primitive(self, primitive, tracers, params):
    rule = error_checks.get(primitive)
    if rule:
      with core.eval_context():
        out, error = rule(init_error, *tracers, **params)
      if error.err:
        raise Exception(error.get())
      return out
    else:
      return primitive.impl(*tracers, **params)

  def process_call(self, primitive, f, tracers, params):
    f, msgs = check_errors_subtrace2(f)
    f = check_errors_toplevel(f)
    with core.eval_context():
      err, code, *out = primitive.impl(f, *tracers, **params)
      error = Error(err, code, msgs())
      if error.err.any():
        fun_info = pe.fun_sourceinfo(f.f)
        name = {xla.xla_call_p: 'jit', pxla.xla_pmap_p: 'pmap'
                }.get(primitive, primitive.name)
        raise Exception(f'inside {name}-decorated function {fun_info}:'
                        f'\n{error.get()}')
    return (err, code, *out)  # must return for the post_process_call...
  process_map = process_call

  def process_custom_jvp_call(self, primitive, fun, jvp, tracers):
    raise NotImplementedError

  def process_custom_vjp_call(self, primitive, fun, fwd, bwd, tracers, out_trees):
    raise NotImplementedError

# TODO(mattjj): this function is a hack, a conspiracy with post_process_call...
# The remove-post-process branch would remove the need for the hack.
@lu.transformation_with_aux
def check_errors_subtrace2(main, msgs, err, code, *args):
  setnewattr(main, 'error', Error(err, code, dict(msgs)))
  trace = main.with_cur_sublevel()
  in_tracers = [ErrorTracer(trace, x) for x in args]
  err, code, *out = yield in_tracers, {}
  out_tracers = map(trace.full_raise, out)
  out_vals = [t.val for t in out_tracers]
  yield (err, code, *out_vals), main.msgs

@contextmanager
def runtime_checks():
  with core.new_base_main(CheckifyEvaluator):
    yield

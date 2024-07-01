from typing import Callable
from functools import partial
import itertools as it

from .. import checkpoint
from .. import core
from .. import linear_util as lu
from ..interpreters import partial_eval as pe
from ..api_util import wraps, flatten_fun
from ..tree_util import tree_flatten, tree_unflatten
from ..traceback_util import api_boundary

# TODO: Support concrete?
def checkpoint_recursive(fun: Callable, threshold=1) -> Callable:
  if threshold < 1:
    raise ValueError(f"threshold has to be at least 1, got {threshold}")
  @wraps(fun)
  @api_boundary
  def fun_recursive(*args, **kwargs):
    args_flat, in_tree = tree_flatten((args, kwargs))
    flat_fun, out_tree = flatten_fun(lu.wrap_init(fun), in_tree)
    avals_flat = [core.raise_to_shaped(core.get_aval(arg)) for arg in args_flat]
    jaxpr, _, consts = pe.trace_to_jaxpr_dynamic(flat_fun, avals_flat)
    jaxpr_noconst = pe.convert_constvars_jaxpr(jaxpr)
    checkp_fun = _recursive_checkpoint_jaxpr(jaxpr_noconst, threshold=threshold)
    out_flat = checkp_fun(*consts, *args_flat)
    return tree_unflatten(out_tree(), out_flat)
  return fun_recursive

def _recursive_checkpoint_jaxpr(jaxpr, threshold):
  assert not jaxpr.constvars
  eqns = jaxpr.eqns
  if len(jaxpr.eqns) <= threshold:
    return partial(core.eval_jaxpr, jaxpr, ())
  split_point = (len(eqns) + 1) // 2
  eqns1, eqns2 = eqns[:split_point], eqns[split_point:]
  residuals = list(_jaxpr_free_vars(core.Jaxpr([], [], jaxpr.outvars, eqns2)))
  jaxpr1 = core.Jaxpr([], jaxpr.invars, residuals, eqns1)
  jaxpr2 = core.Jaxpr([], residuals, jaxpr.outvars, eqns2)
  f1 = checkpoint(_recursive_checkpoint_jaxpr(jaxpr1, threshold))
  f2 = _recursive_checkpoint_jaxpr(jaxpr2, threshold)
  return lambda *args: f2(*f1(*args))

def _jaxpr_free_vars(jaxpr):
  free = set(jaxpr.outvars)
  for eqn in jaxpr.eqns[::-1]:
    free -= set(eqn.outvars)
    free |= set(eqn.invars)
  free -= set(it.chain(jaxpr.invars, jaxpr.constvars))
  return free

from __future__ import annotations

import inspect
import types
import typing

import numpy as np

from jax._src import core
from jax._src.util import safe_zip, safe_map
import jax
import jax.numpy as jnp

zip, unsafe_zip = safe_zip, zip
map, unsafe_map = safe_map, map

jax.config.update('jax_dynamic_shapes', True)


if typing.TYPE_CHECKING:
  f32 = typing.Annotated
else:
  class dtype:
    def __init__(self, dtype):
      self.dtype = dtype

    def __getitem__(self, stuff):
      if type(stuff) is not tuple:
        stuff = (stuff,)
      return types.SimpleNamespace(shape=stuff, dtype=self.dtype)
  f32 = dtype(jnp.dtype('float32'))


def shapecheck(f):
  try: sig = inspect.signature(f)
  except: return
  def subst_vars(ty):
    new_shape = tuple(0 if type(d) is not int else d for d in ty.shape)
    return types.SimpleNamespace(shape=new_shape, dtype=ty.dtype)

  dummy_args = [np.zeros(ty.shape, ty.dtype)
                for param in sig.parameters.values()
                for ty in [subst_vars(eval(param.annotation))]]
  abstracted_axes_ = [eval(param.annotation).shape
                      for param in sig.parameters.values()]
  abstracted_axes = [{i:n for i, n in enumerate(ax) if type(n) is not int}
                     for ax in abstracted_axes_]

  jaxpr = jax.make_jaxpr(f, abstracted_axes=tuple(abstracted_axes))(*dummy_args)

  out_type, = jaxpr.out_avals

  num_dim_vars = len({d for ann in abstracted_axes for d in ann.values()})
  env = {jaxpr_var:ann[i]
         for v, ann in zip(jaxpr.jaxpr.invars[num_dim_vars:], abstracted_axes)
         for i, jaxpr_var in enumerate(v.aval.shape)
         if type(jaxpr_var) is core.Var}
  annotation_shape = eval(sig.return_annotation).shape
  v, = jaxpr.jaxpr.outvars
  computed_shape = tuple(env.get(d, d) for d in v.aval.shape)
  if computed_shape != annotation_shape: raise Exception
  return f

###

def axis_size_vars(*args):
  return args

m, k, n = axis_size_vars('m', 'k', 'n')

@shapecheck
def f(x: f32[3, n], y:f32[3, n]) -> f32[3]:
  z = jnp.dot(x, y.T)
  w = jnp.tanh(z)
  return w.sum(0)

# TODO [ ] handle let-bound dynamic shapes (ie output dim vars)
# TODO [ ] handle multiple outputs
# TODO [ ] make internal error message better (dont mention tracers in msg)
# TODO [ ] clean up
# TODO [ ] mapping to python variables, set trace
# TODO [ ] editor integration of some kind

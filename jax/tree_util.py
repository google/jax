# Copyright 2018 Google LLC
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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from collections import namedtuple
import itertools as it
from six.moves import reduce

from .util import unzip2, concatenate, partial, safe_map

map = safe_map


def tree_map(f, tree):
  node_type = get_node_type(type(tree))
  if node_type is not leaf:
    children, node_spec = node_type.to_iterable(tree)
    new_children = [tree_map(f, child) for child in children]
    return node_type.from_iterable(node_spec, new_children)
  else:
    return f(tree)


def tree_multimap(f, tree, *rest):
  node_type = get_node_type(type(tree))
  if node_type is not leaf:
    children, node_spec = node_type.to_iterable(tree)
    all_children = [children]
    for other_tree in rest:
      other_children, other_node_spec = node_type.to_iterable(other_tree)
      if other_node_spec != node_spec:
        raise TypeError('Mismatch: {} != {}'.format(other_node_spec, node_spec))
      all_children.append(other_children)

    new_children = [tree_multimap(f, *xs) for xs in zip(*all_children)]
    return node_type.from_iterable(node_spec, new_children)
  else:
    return f(tree, *rest)


def tree_reduce(f, tree):
  flat, _ = tree_flatten(tree)
  return reduce(f, flat)


def tree_all(tree):
  flat, _ = tree_flatten(tree)
  return all(flat)


def process_pytree(process_node, tree):
  return walk_pytree(process_node, lambda x: x, tree)


def walk_pytree(f_node, f_leaf, tree):
  node_type = get_node_type(type(tree))
  if node_type is not leaf:
    children, node_spec = node_type.to_iterable(tree)
    proc_children, child_specs = unzip2([walk_pytree(f_node, f_leaf, child)
                                         for child in children])
    tree_def = PyTreeDef(node_type, node_spec, child_specs)
    return f_node(proc_children), tree_def
  else:
    return f_leaf(tree), leaf


def build_tree(treedef, xs):
  if treedef is leaf:
    return xs
  else:
    # We use 'iter' for clearer error messages
    children = map(build_tree, iter(treedef.children), iter(xs))
    return treedef.node_type.from_iterable(treedef.node_data, children)


tree_flatten = partial(walk_pytree, concatenate, lambda x: [x])

def tree_unflatten(treedef, xs):
  return _tree_unflatten(iter(xs), treedef)

def _tree_unflatten(xs, treedef):
  if treedef is leaf:
    return next(xs)
  else:
    children = map(partial(_tree_unflatten, xs), treedef.children)
    return treedef.node_type.from_iterable(treedef.node_data, children)


def tree_transpose(outer_treedef, inner_treedef, pytree_to_transpose):
  flat, treedef = tree_flatten(pytree_to_transpose)
  expected_treedef = _nested_treedef(inner_treedef, outer_treedef)
  if treedef != expected_treedef:
    raise TypeError("Mismatch\n{}\n != \n{}".format(treedef, expected_treedef))

  inner_size = _num_leaves(inner_treedef)
  outer_size = _num_leaves(outer_treedef)
  flat = iter(flat)
  lol = [[next(flat) for _ in range(inner_size)] for __ in range(outer_size)]
  transposed_lol = zip(*lol)
  subtrees = map(partial(tree_unflatten, outer_treedef), transposed_lol)
  return tree_unflatten(inner_treedef, subtrees)

def _num_leaves(treedef):
  return 1 if treedef is leaf else sum(map(_num_leaves, treedef.children))

def _nested_treedef(inner, outer):
  # just used in tree_transpose error checking
  if outer is leaf:
    return inner
  else:
    children = map(partial(_nested_treedef, inner), outer.children)
    return PyTreeDef(outer.node_type, outer.node_data, tuple(children))


def tree_structure(tree):
  _, spec = process_pytree(lambda _: None, tree)
  return spec


def prune(treedef, tuple_tree):
  if treedef is leaf:
    return tuple_tree
  elif treedef.children:
    return tuple(map(prune, treedef.children, tuple_tree))
  else:
    return ()


class PyTreeDef(object):
  def __init__(self, node_type, node_data, children):
    self.node_type = node_type
    self.node_data = node_data
    self.children = children

  def __repr__(self):
    if self.node_data is None:
      data_repr = ""
    else:
      data_repr = "[{}]".format(self.node_data)

    return "PyTree({}{}, [{}])".format(self.node_type.name, data_repr,
                                     ','.join(map(repr, self.children)))

  def __hash__(self):
    return hash((self.node_type, self.node_data, tuple(self.children)))

  def __eq__(self, other):
    if other is leaf:
      return False
    else:
      return (self.node_type == other.node_type and
              self.node_data == other.node_data and
              self.children == other.children)

  def __ne__(self, other):
    return not self == other


class PyLeaf(object):
  include_subclasses = True
  def __repr__(self):
    return '*'

leaf = PyLeaf()

def dict_to_iterable(xs):
  keys, values = zip(*sorted(xs.items()))
  return values, keys

class NodeType(object):
  def __init__(self, name, to_iterable, from_iterable, include_subclasses):
    self.name = name
    self.to_iterable = to_iterable
    self.from_iterable = from_iterable
    self.include_subclasses = include_subclasses

node_types = {}

def get_node_type(py_type):
  for cls in py_type.__mro__:
    node_type = node_types.get(cls)
    if node_type is not None and (
        cls is py_type or node_type.include_subclasses):
      return node_type
  return leaf

def register_pytree_node(py_type, to_iterable, from_iterable,
                         include_subclasses=True):
  """Registers a Python type as a container-like data structure.

  Many JAX functions work equally well with individual arrays and
  nested trees of tuples, lists, dicts and other containers. This
  function registers a new container data type with the tree-handling
  machinery that enables that functionality. If multiple handlers
  are registered that apply to the same type, the most specific one
  is used.

  Args:
    py_type: A type object, such as a user-defined class.
    to_iterable: A function that takes an instance of that type and
      returns an iterable containing the instance's child objects
      and an object holding other state needed to rebuild the
      instance.
    from_iterable: A function that takes a state object returned by
      the to_iterable function, plus an iterable, and returns an
      instance of the py_type type or a subclass.
    include_subclasses: Whether the handlers being registered should
      also apply to subclasses of the provided type.
  """
  assert py_type not in node_types
  node_types[py_type] = NodeType(
    str(py_type), to_iterable, from_iterable, include_subclasses)

def register_pytree_leaf(py_type):
  assert py_type not in node_types
  node_types[py_type] = leaf

register_pytree_node(tuple, lambda xs: (tuple(xs), type(xs)),
                     lambda t, xs: t(xs) if t is tuple else t(*xs))
register_pytree_node(list, lambda xs: (tuple(xs), None), lambda _, xs: list(xs))
register_pytree_node(dict, dict_to_iterable, lambda keys, xs: dict(zip(keys, xs)))
register_pytree_node(type(None), lambda z: ((), None), lambda _, xs: None)

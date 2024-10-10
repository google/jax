# Copyright 2019 The JAX Authors.
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

import collections
import dataclasses
import functools
import pickle
import re
from typing import TypeVar

from absl.testing import absltest
from absl.testing import parameterized
import jax
from jax import flatten_util
from jax import tree_util
from jax._src import test_util as jtu
from jax._src.tree_util import flatten_one_level, prefix_errors
import jax.numpy as jnp


def _dummy_func(*args, **kwargs):
  return


ATuple = collections.namedtuple("ATuple", ("foo", "bar"))

class ANamedTupleSubclass(ATuple):
  pass

ATuple2 = collections.namedtuple("ATuple2", ("foo", "bar"))
tree_util.register_pytree_node(ATuple2, lambda o: ((o.foo,), o.bar),
                               lambda bar, foo: ATuple2(foo[0], bar))

BadFlattenNonTuple = collections.namedtuple("ATuple2", ("foo", "bar"))
tree_util.register_pytree_node(BadFlattenNonTuple, lambda o: "hello",
                               lambda bar, foo: ATuple2(foo[0], bar))

BadFlattenBadArityTuple = collections.namedtuple("ATuple2", ("foo", "bar"))
tree_util.register_pytree_node(BadFlattenBadArityTuple, lambda o: (2, 3, 4),
                               lambda bar, foo: ATuple2(foo[0], bar))

BadFlattenNonIterableLeaves = collections.namedtuple("ATuple2", ("foo", "bar"))
tree_util.register_pytree_node(BadFlattenNonIterableLeaves, lambda o: (7, 7),
                               lambda bar, foo: ATuple2(foo[0], bar))


class AnObject:

  def __init__(self, x, y, z):
    self.x = x
    self.y = y
    self.z = z

  def __eq__(self, other):
    return self.x == other.x and self.y == other.y and self.z == other.z

  def __hash__(self):
    return hash((self.x, self.y, self.z))

  def __repr__(self):
    return f"AnObject({self.x},{self.y},{self.z})"

tree_util.register_pytree_node(AnObject, lambda o: ((o.x, o.y), o.z),
                               lambda z, xy: AnObject(xy[0], xy[1], z))

class AnObject2(AnObject): pass

tree_util.register_pytree_with_keys(
    AnObject2,
    lambda o: ((("x", o.x), ("y", o.y)), o.z),  # flatten_with_keys
    lambda z, xy: AnObject2(xy[0], xy[1], z),    # unflatten (no key involved)
)

@tree_util.register_pytree_node_class
class Special:
  def __init__(self, x, y):
    self.x = x
    self.y = y

  def __repr__(self):
    return f"Special(x={self.x}, y={self.y})"

  def tree_flatten(self):
    return ((self.x, self.y), None)

  @classmethod
  def tree_unflatten(cls, aux_data, children):
    return cls(*children)

  def __eq__(self, other):
    return type(self) is type(other) and (self.x, self.y) == (other.x, other.y)


@tree_util.register_pytree_with_keys_class
class SpecialWithKeys(Special):
  def tree_flatten_with_keys(self):
    return (((tree_util.GetAttrKey('x'), self.x),
             (tree_util.GetAttrKey('y'), self.y)), None)


@tree_util.register_pytree_node_class
class FlatCache:
  def __init__(self, structured, *, leaves=None, treedef=None):
    if treedef is None:
      leaves, treedef = tree_util.tree_flatten(structured)
    self._structured = structured
    self.treedef = treedef
    self.leaves = leaves

  def __hash__(self):
    return hash(self.structured)

  def __eq__(self, other):
    return self.structured == other.structured

  def __repr__(self):
    return f"FlatCache({self.structured!r})"

  @property
  def structured(self):
    if self._structured is None:
      self._structured = tree_util.tree_unflatten(self.treedef, self.leaves)
    return self._structured

  def tree_flatten(self):
    return self.leaves, self.treedef

  @classmethod
  def tree_unflatten(cls, meta, data):
    if not tree_util.all_leaves(data):
      data, meta = tree_util.tree_flatten(tree_util.tree_unflatten(meta, data))
    return FlatCache(None, leaves=data, treedef=meta)

_T = TypeVar("_T")


# Inspired by Flax.
def pytree_node_dataclass(clz: _T, **kwargs) -> _T:
  data_clz = dataclasses.dataclass(**kwargs)(clz)  # type: ignore
  meta_fields = []
  data_fields = []
  for field_info in dataclasses.fields(data_clz):
    is_pytree_node = field_info.metadata.get("pytree_node", True)
    if is_pytree_node:
      data_fields.append(field_info.name)
    else:
      meta_fields.append(field_info.name)

  jax.tree_util.register_dataclass(
      data_clz, data_fields, meta_fields
  )

  return data_clz


@tree_util.register_static
class StaticInt(int):
  pass


@tree_util.register_static
class StaticTuple(tuple):
  pass


@tree_util.register_static
class StaticDict(dict):
  pass


@tree_util.register_static
@dataclasses.dataclass
class BlackBox:
  """Stores a value but pretends to be equal to every other black box."""

  value: int

  def __hash__(self):
    return 0

  def __eq__(self, other):
    return isinstance(other, BlackBox)


TREES = (
    (None,),
    ((None,),),
    ((),),
    (([()]),),
    ((1, 2),),
    (((1, "foo"), ["bar", (3, None, 7)]),),
    ([3],),
    ([3, ATuple(foo=(3, ATuple(foo=3, bar=None)), bar={"baz": 34})],),
    ([AnObject(3, None, [4, "foo"])],),
    ([AnObject2(3, None, [4, "foo"])],),
    (Special(2, 3.0),),
    ({"a": 1, "b": 2},),
    (StaticInt(1),),
    (StaticTuple((2, 3)),),
    (StaticDict(foo=4, bar=5),),
)


TREE_STRINGS = (
    "PyTreeDef(None)",
    "PyTreeDef((None,))",
    "PyTreeDef(())",
    "PyTreeDef([()])",
    "PyTreeDef((*, *))",
    "PyTreeDef(((*, *), [*, (*, None, *)]))",
    "PyTreeDef([*])",
    ("PyTreeDef([*, CustomNode(namedtuple[ATuple], [(*, "
     "CustomNode(namedtuple[ATuple], [*, None])), {'baz': *}])])"),
    "PyTreeDef([CustomNode(AnObject[[4, 'foo']], [*, None])])",
    "PyTreeDef([CustomNode(AnObject2[[4, 'foo']], [*, None])])",
    "PyTreeDef(CustomNode(Special[None], [*, *]))",
    "PyTreeDef({'a': *, 'b': *})",
    "PyTreeDef(CustomNode(StaticInt[1], []))",
    "PyTreeDef(CustomNode(StaticTuple[(2, 3)], []))",
    "PyTreeDef(CustomNode(StaticDict[{'foo': 4, 'bar': 5}], []))",
)

@pytree_node_dataclass
class ADataclass:
  x: tuple[int, int]
  y: int

@pytree_node_dataclass
class ADataclassWithMeta:
  x: tuple[int, int]
  y: int
  z: int = dataclasses.field(metadata={"pytree_node": False})

TREES += (
    (ADataclass(x=(1, 2), y=3),),
    (ADataclassWithMeta(x=(1, 2), y=3, z=4),),
)
TREE_STRINGS += (
    "PyTreeDef(CustomNode(ADataclass[()], [(*, *), *]))",
    "PyTreeDef(CustomNode(ADataclassWithMeta[(4,)], [(*, *), *]))",
)


TREES += (
    (collections.OrderedDict([("foo", 34), ("baz", 101), ("something", -42)]),),
    (
        collections.defaultdict(
            dict, [("foo", 34), ("baz", 101), ("something", -42)]
        ),
    ),
    (ANamedTupleSubclass(foo="hello", bar=3.5),),
    (FlatCache(None),),
    (FlatCache(1),),
    (FlatCache({"a": [1, 2]}),),
    (BlackBox(value=2),),
)


# pytest expects "tree_util_test.ATuple"
STRS = []
for tree_str in TREE_STRINGS:
  tree_str = re.escape(tree_str)
  tree_str = tree_str.replace("__main__", ".*")
  STRS.append(tree_str)
TREE_STRINGS = STRS

LEAVES = (
    ("foo",),
    (0.1,),
    (1,),
    (object(),),
)

# All except those decorated by register_pytree_node_class
TREES_WITH_KEYPATH = (
    (None,),
    ((None,),),
    ((),),
    (([()]),),
    ((1, 0),),
    (((1, "foo"), ["bar", (3, None, 7)]),),
    ([3],),
    ([3, ATuple(foo=(3, ATuple(foo=3, bar=None)), bar={"baz": 34})],),
    ([AnObject2(3, None, [4, "foo"])],),
    (SpecialWithKeys(2, 3.),),
    ({"a": 1, "b": 0},),
    (collections.OrderedDict([("foo", 34), ("baz", 101), ("something", -42)]),),
    (collections.defaultdict(dict,
                             [("foo", 34), ("baz", 101), ("something", -42)]),),
    (ANamedTupleSubclass(foo="hello", bar=3.5),),
    (StaticInt(1),),
    (StaticTuple((2, 3)),),
    (StaticDict(foo=4, bar=5),),
    (BlackBox(value=2),),
)


class TreeTest(jtu.JaxTestCase):

  @parameterized.parameters(*(TREES + LEAVES))
  def testRoundtrip(self, inputs):
    xs, tree = tree_util.tree_flatten(inputs)
    actual = tree_util.tree_unflatten(tree, xs)
    self.assertEqual(actual, inputs)

  @parameterized.parameters(*(TREES + LEAVES))
  def testRoundtripWithFlattenUpTo(self, inputs):
    _, tree = tree_util.tree_flatten(inputs)
    xs = tree.flatten_up_to(inputs)
    actual = tree_util.tree_unflatten(tree, xs)
    self.assertEqual(actual, inputs)

  @parameterized.parameters(
      (tree_util.Partial(_dummy_func),),
      (tree_util.Partial(_dummy_func, 1, 2),),
      (tree_util.Partial(_dummy_func, x="a"),),
      (tree_util.Partial(_dummy_func, 1, 2, 3, x=4, y=5),),
  )
  def testRoundtripPartial(self, inputs):
    xs, tree = tree_util.tree_flatten(inputs)
    actual = tree_util.tree_unflatten(tree, xs)
    # functools.partial does not support equality comparisons:
    # https://stackoverflow.com/a/32786109/809705
    self.assertEqual(actual.func, inputs.func)
    self.assertEqual(actual.args, inputs.args)
    self.assertEqual(actual.keywords, inputs.keywords)

  def testPartialDoesNotMergeWithOtherPartials(self):
    def f(a, b, c): pass
    g = functools.partial(f, 2)
    h = tree_util.Partial(g, 3)
    self.assertEqual(h.args, (3,))

  def testPartialFuncAttributeHasStableHash(self):
    # https://github.com/jax-ml/jax/issues/9429
    fun = functools.partial(print, 1)
    p1 = tree_util.Partial(fun, 2)
    p2 = tree_util.Partial(fun, 2)
    self.assertEqual(fun, p1.func)
    self.assertEqual(p1.func, fun)
    self.assertEqual(p1.func, p2.func)
    self.assertEqual(hash(p1.func), hash(p2.func))

  def testChildren(self):
    _, tree = tree_util.tree_flatten(((1, 2, 3), (4,)))
    _, c0 = tree_util.tree_flatten((0, 0, 0))
    _, c1 = tree_util.tree_flatten((7,))
    self.assertEqual([c0, c1], tree.children())

  def testTreedefTupleFromChildren(self):
    # https://github.com/jax-ml/jax/issues/7377
    tree = ((1, 2, (3, 4)), (5,))
    leaves, treedef1 = tree_util.tree_flatten(tree)
    treedef2 = tree_util.treedef_tuple(treedef1.children())
    self.assertEqual(treedef1.num_leaves, len(leaves))
    self.assertEqual(treedef1.num_leaves, treedef2.num_leaves)
    self.assertEqual(treedef1.num_nodes, treedef2.num_nodes)

  def testTreedefTupleComparesEqual(self):
    # https://github.com/jax-ml/jax/issues/9066
    self.assertEqual(tree_util.tree_structure((3,)),
                     tree_util.treedef_tuple((tree_util.tree_structure(3),)))

  def testFlattenOrder(self):
    flat1, _ = tree_util.tree_flatten([0, ((1, 2), 3, (4, (5, 6, 7))), 8, 9])
    flat2, _ = tree_util.tree_flatten([0, ((1, 2), 3, (4, (5, 6, 7))), 8, 9])
    flat3, _ = tree_util.tree_flatten([0, ((1, (2, 3)), (4, (5, 6, 7))), 8, 9])
    self.assertEqual(flat1, list(range(10)))
    self.assertEqual(flat2, list(range(10)))
    self.assertEqual(flat3, list(range(10)))

  @parameterized.parameters(
      (
          [(1, 2), None, ATuple(foo=3, bar=7)],
          [({"foo": 7}, (3, 4)), None, ATuple(foo=(11, 9), bar=None)],
          [{"foo": 7}, (3, 4), (11, 9), None],
      ),
      ([1], [{"a": 7}], [{"a": 7}]),
      ([1], [[7]], [[7]]),
      ([1], [(7,)], [(7,)]),
      ((1, 2), ({"a": 7}, {"a": 8}), [{"a": 7}, {"a": 8}]),
      ((1,), ([7],), [[7]]),
      ((1,), ((7,),), [(7,)]),
      ({"a": 1, "b": (2, 3)}, {"a": [7], "b": ([8], (9,))}, [[7], [8], (9,)]),
      ({"a": 1}, {"a": (7,)}, [(7,)]),
      ({"a": 1}, {"a": {"a": 7}}, [{"a": 7}]),
      (None, None, [])
  )
  def testFlattenUpTo(self, tree, xs, expected):
    _, tree_def = tree_util.tree_flatten(tree)
    out = tree_def.flatten_up_to(xs)
    self.assertEqual(out, expected)

  @parameterized.parameters(
      ([1, 2], [7], re.escape("List arity mismatch: 1 != 2; list: [7].")),
      ((1,), (7, 8), re.escape("Tuple arity mismatch: 2 != 1; tuple: (7, 8).")),
      (
          {"a": 1},
          {"a": 7, "b": 8},
          re.escape(
              "Dict key mismatch; expected keys: ['a']; dict: {'a': 7, 'b': 8}."
          ),
      ),
      (
          {"a": 1},
          {"b": 7},
          re.escape("Dict key mismatch; expected keys: ['a']; dict: {'b': 7}."),
      ),
      ([1], {"a": 7}, re.escape("Expected list, got {'a': 7}.")),
      ([1], (7,), re.escape("Expected list, got (7,).")),
      ((1,), [7], re.escape("Expected tuple, got [7].")),
      ((1,), {"b": 7}, re.escape("Expected tuple, got {'b': 7}.")),
      ({"a": 1}, (7,), re.escape("Expected dict, got (7,).")),
      ({"a": 1}, [7], re.escape("Expected dict, got [7].")),
      ([[1]], [7], re.escape("Expected list, got 7.")),
      ([[1]], [(7,)], re.escape("Expected list, got (7,).")),
      ([[1]], [{"a": 7}], re.escape("Expected list, got {'a': 7}.")),
      ([(1,)], [7], re.escape("Expected tuple, got 7.")),
      ([(1,)], [[7]], re.escape("Expected tuple, got [7].")),
      ([(1,)], [{"a": 7}], re.escape("Expected tuple, got {'a': 7}.")),
      ([{"a": 1}], [7], re.escape("Expected dict, got 7.")),
      ([{"a": 1}], [[7]], re.escape("Expected dict, got [7].")),
      ([{"a": 1}], [(7,)], re.escape("Expected dict, got (7,).")),
      (
          [{"a": 1}],
          [{"b": 7}],
          re.escape("Dict key mismatch; expected keys: ['a']; dict: {'b': 7}."),
      ),
      (([1],), (7,), re.escape("Expected list, got 7.")),
      (([1],), ((7,),), re.escape("Expected list, got (7,).")),
      (([1],), ({"a": 7},), re.escape("Expected list, got {'a': 7}.")),
      (((1,),), (7,), re.escape("Expected tuple, got 7.")),
      (((1,),), ([7],), re.escape("Expected tuple, got [7].")),
      (((1,),), ({"a": 7},), re.escape("Expected tuple, got {'a': 7}.")),
      (({"a": 1},), (7,), re.escape("Expected dict, got 7.")),
      (({"a": 1},), ([7],), re.escape("Expected dict, got [7].")),
      (({"a": 1},), ((7,),), re.escape("Expected dict, got (7,).")),
      (
          ({"a": 1},),
          ({"b": 7},),
          re.escape("Dict key mismatch; expected keys: ['a']; dict: {'b': 7}."),
      ),
      ({"a": [1]}, {"a": 7}, re.escape("Expected list, got 7.")),
      ({"a": [1]}, {"a": (7,)}, re.escape("Expected list, got (7,).")),
      ({"a": [1]}, {"a": {"a": 7}}, re.escape("Expected list, got {'a': 7}.")),
      ({"a": (1,)}, {"a": 7}, re.escape("Expected tuple, got 7.")),
      ({"a": (1,)}, {"a": [7]}, re.escape("Expected tuple, got [7].")),
      (
          {"a": (1,)},
          {"a": {"a": 7}},
          re.escape("Expected tuple, got {'a': 7}."),
      ),
      ({"a": {"a": 1}}, {"a": 7}, re.escape("Expected dict, got 7.")),
      ({"a": {"a": 1}}, {"a": [7]}, re.escape("Expected dict, got [7].")),
      ({"a": {"a": 1}}, {"a": (7,)}, re.escape("Expected dict, got (7,).")),
      (
          {"a": {"a": 1}},
          {"a": {"b": 7}},
          re.escape("Dict key mismatch; expected keys: ['a']; dict: {'b': 7}."),
      ),
      (
          [ATuple(foo=1, bar=2)],
          [(1, 2)],
          re.escape("Expected named tuple, got (1, 2)."),
      ),
      (
          [ATuple(foo=1, bar=2)],
          [ATuple2(foo=1, bar=2)],
          re.escape("Named tuple type mismatch"),
      ),
      (
          [AnObject(x=[1], y=(2,), z={"a": [1]})],
          [([1], (2,), {"a": [1]})],
          re.escape("Custom node type mismatch"),
      ),
      (
          (None, [2], re.escape("Expected None, got [2]."))
      ),
  )
  def testFlattenUpToErrors(self, tree, xs, error):
    _, tree_def = tree_util.tree_flatten(tree)
    with self.assertRaisesRegex(ValueError, error):
      tree_def.flatten_up_to(xs)

  def testTreeMap(self):
    x = ((1, 2), (3, 4, 5))
    y = (([3], None), ({"foo": "bar"}, 7, [5, 6]))
    out = tree_util.tree_map(lambda *xs: tuple(xs), x, y)
    self.assertEqual(out, (((1, [3]), (2, None)),
                           ((3, {"foo": "bar"}), (4, 7), (5, [5, 6]))))

  def testTreeMapWithIsLeafArgument(self):
    x = ((1, 2), [3, 4, 5])
    y = (([3], None), ({"foo": "bar"}, 7, [5, 6]))
    out = tree_util.tree_map(lambda *xs: tuple(xs), x, y,
                             is_leaf=lambda n: isinstance(n, list))
    self.assertEqual(out, (((1, [3]), (2, None)),
                           (([3, 4, 5], ({"foo": "bar"}, 7, [5, 6])))))

  def testTreeReduceWithIsLeafArgument(self):
    out = tree_util.tree_reduce(lambda x, y: x + y, [(1, 2), [(3, 4), (5, 6)]],
                                is_leaf=lambda l: isinstance(l, tuple))
    self.assertEqual(out, (1, 2, 3, 4, 5, 6))

  @parameterized.parameters(
      tree_util.tree_leaves,
      lambda tree, is_leaf: tree_util.tree_flatten(tree, is_leaf)[0])
  def testFlattenIsLeaf(self, leaf_fn):
    x = [(1, 2), (3, 4), (5, 6)]
    leaves = leaf_fn(x, is_leaf=lambda t: False)
    self.assertEqual(leaves, [1, 2, 3, 4, 5, 6])
    leaves = leaf_fn(x, is_leaf=lambda t: isinstance(t, tuple))
    self.assertEqual(leaves, x)
    leaves = leaf_fn(x, is_leaf=lambda t: isinstance(t, list))
    self.assertEqual(leaves, [x])
    leaves = leaf_fn(x, is_leaf=lambda t: True)
    self.assertEqual(leaves, [x])

    y = [[[(1,)], [[(2,)], {"a": (3,)}]]]
    leaves = leaf_fn(y, is_leaf=lambda t: isinstance(t, tuple))
    self.assertEqual(leaves, [(1,), (2,), (3,)])

  @parameterized.parameters(
      tree_util.tree_structure,
      lambda tree, is_leaf: tree_util.tree_flatten(tree, is_leaf)[1])
  def testStructureIsLeaf(self, structure_fn):
    x = [(1, 2), (3, 4), (5, 6)]
    treedef = structure_fn(x, is_leaf=lambda t: False)
    self.assertEqual(treedef.num_leaves, 6)
    treedef = structure_fn(x, is_leaf=lambda t: isinstance(t, tuple))
    self.assertEqual(treedef.num_leaves, 3)
    treedef = structure_fn(x, is_leaf=lambda t: isinstance(t, list))
    self.assertEqual(treedef.num_leaves, 1)
    treedef = structure_fn(x, is_leaf=lambda t: True)
    self.assertEqual(treedef.num_leaves, 1)

    y = [[[(1,)], [[(2,)], {"a": (3,)}]]]
    treedef = structure_fn(y, is_leaf=lambda t: isinstance(t, tuple))
    self.assertEqual(treedef.num_leaves, 3)

  @parameterized.parameters(*TREES)
  def testRoundtripIsLeaf(self, tree):
    xs, treedef = tree_util.tree_flatten(
        tree, is_leaf=lambda t: isinstance(t, tuple))
    recon_tree = tree_util.tree_unflatten(treedef, xs)
    self.assertEqual(recon_tree, tree)

  @parameterized.parameters(*TREES)
  def testAllLeavesWithTrees(self, tree):
    leaves = tree_util.tree_leaves(tree)
    self.assertTrue(tree_util.all_leaves(leaves))
    self.assertFalse(tree_util.all_leaves([tree]))

  @parameterized.parameters(*LEAVES)
  def testAllLeavesWithLeaves(self, leaf):
    self.assertTrue(tree_util.all_leaves([leaf]))

  @parameterized.parameters(*TREES)
  def testCompose(self, tree):
    treedef = tree_util.tree_structure(tree)
    inner_treedef = tree_util.tree_structure(["*", "*", "*"])
    composed_treedef = treedef.compose(inner_treedef)
    expected_leaves = treedef.num_leaves * inner_treedef.num_leaves
    self.assertEqual(composed_treedef.num_leaves, expected_leaves)
    expected_nodes = ((treedef.num_nodes - treedef.num_leaves) +
                      (inner_treedef.num_nodes * treedef.num_leaves))
    self.assertEqual(composed_treedef.num_nodes, expected_nodes)
    leaves = [1] * expected_leaves
    composed = tree_util.tree_unflatten(composed_treedef, leaves)
    self.assertEqual(leaves, tree_util.tree_leaves(composed))

  @parameterized.parameters(*TREES)
  def testTranspose(self, tree):
    outer_treedef = tree_util.tree_structure(tree)
    if not outer_treedef.num_leaves:
      self.skipTest("Skipping empty tree")
    def make_inner(x):
      return [x, x, x]
    inner_treedef = tree_util.tree_structure(make_inner(1))
    nested = tree_util.tree_map(make_inner, tree)
    actual = tree_util.tree_transpose(outer_treedef, inner_treedef, nested)
    self.assertEqual(actual, make_inner(tree))

  @parameterized.parameters(*TREES)
  def testTransposeInferInnerTreedef(self, tree):
    if isinstance(tree, FlatCache):
      # The tree_map construction below fails for FlatCache, because
      # the cached metadata becomes out of sync.
      self.skipTest("Test does not work properly for FlatCache.")
    outer_treedef = tree_util.tree_structure(tree)
    if not outer_treedef.num_leaves:
      self.skipTest("Skipping empty tree")
    def make_inner(x):
      return [x, {'a': x}, (x,)]
    nested = tree_util.tree_map(make_inner, tree)
    actual = tree_util.tree_transpose(outer_treedef, None, nested)
    self.assertEqual(actual, make_inner(tree))

  def testTransposeMismatchOuter(self):
    tree = {"a": [1, 2], "b": [3, 4]}
    outer_treedef = tree_util.tree_structure({"a": 1, "b": 2, "c": 3})
    inner_treedef = tree_util.tree_structure([1, 2])
    with self.assertRaisesRegex(TypeError, "Mismatch"):
      tree_util.tree_transpose(outer_treedef, inner_treedef, tree)

  def testTransposeMismatchInner(self):
    tree = {"a": [1, 2], "b": [3, 4]}
    outer_treedef = tree_util.tree_structure({"a": 1, "b": 2})
    inner_treedef = tree_util.tree_structure([1, 2, 3])
    with self.assertRaisesRegex(TypeError, "Mismatch"):
      tree_util.tree_transpose(outer_treedef, inner_treedef, tree)

  def testTransposeWithCustomObject(self):
    outer_treedef = tree_util.tree_structure(FlatCache({"a": 1, "b": 2}))
    inner_treedef = tree_util.tree_structure([1, 2])
    expected = [FlatCache({"a": 3, "b": 5}), FlatCache({"a": 4, "b": 6})]
    actual = tree_util.tree_transpose(outer_treedef, inner_treedef,
                                      FlatCache({"a": [3, 4], "b": [5, 6]}))
    self.assertEqual(expected, actual)

  @parameterized.parameters([(*t, s) for t, s in zip(TREES, TREE_STRINGS)])
  def testStringRepresentation(self, tree, correct_string):
    """Checks that the string representation of a tree works."""
    treedef = tree_util.tree_structure(tree)
    print(TREES)
    self.assertRegex(str(treedef), correct_string)

  def testTreeDefWithEmptyDictStringRepresentation(self):
    self.assertEqual(str(tree_util.tree_structure({})), "PyTreeDef({})")

  @parameterized.parameters(*TREES)
  def testPickleRoundTrip(self, tree):
    leaves, treedef = tree_util.tree_flatten(tree)
    treedef_restored = pickle.loads(pickle.dumps(treedef))
    self.assertEqual(treedef, treedef_restored)
    reconstituted = treedef_restored.unflatten(leaves)
    self.assertEqual(tree, reconstituted)

  def testDictKeysSortable(self):
    d = {"a": 1, 2: "b"}
    with self.assertRaisesRegex(
        (TypeError, ValueError),
        "('<' not supported|Comparator raised exception).*"):
      _, _ = tree_util.tree_flatten(d)

  def testFlattenDictKeyOrder(self):
    d = {"b": 2, "a": 1, "c": {"b": 2, "a": 1}}
    leaves, treedef = tree_util.tree_flatten(d)
    self.assertEqual(leaves, [1, 2, 1, 2])
    self.assertEqual(
        str(treedef), "PyTreeDef({'a': *, 'b': *, 'c': {'a': *, 'b': *}})"
    )
    restored_d = tree_util.tree_unflatten(treedef, leaves)
    self.assertEqual(list(restored_d.keys()), ["a", "b", "c"])

  def testFlattenDefaultDictKeyOrder(self):
    d = collections.defaultdict(int,
                                {"b": 2, "a": 1, "c": {"b": 2, "a": 1}})
    leaves, treedef = tree_util.tree_flatten(d)
    self.assertEqual(leaves, [1, 2, 1, 2])
    restored_d = tree_util.tree_unflatten(treedef, leaves)
    self.assertEqual(list(restored_d.keys()), ["a", "b", "c"])

  def testWalk(self):
    d = {"b": 2, "a": 1, "c": {"b": 2, "a": 1}}
    leaves, treedef = tree_util.tree_flatten(d)

    nodes_visited = []
    node_data_visited = []
    leaves_visited = []

    def f_node(node, node_data):
      nodes_visited.append(node)
      node_data_visited.append(node_data)

    def f_leaf(leaf):
      leaves_visited.append(leaf)

    treedef.walk(f_node, f_leaf, leaves)
    self.assertEqual(leaves_visited, [1, 2, 1, 2])
    self.assertEqual(nodes_visited, [(None, None), (None, None, None)])
    self.assertEqual(node_data_visited, [["a", "b"], ["a", "b", "c"]])

  @parameterized.parameters(*(TREES_WITH_KEYPATH + LEAVES))
  def testRoundtripWithPath(self, inputs):
    key_leaves, treedef = tree_util.tree_flatten_with_path(inputs)
    actual = tree_util.tree_unflatten(treedef, [leaf for _, leaf in key_leaves])
    self.assertEqual(actual, inputs)

  def testTreeMapWithPath(self):
    tree = [{i: i for i in range(10)}]
    all_zeros = tree_util.tree_map_with_path(
        lambda kp, val: val - kp[1].key + kp[0].idx, tree
    )
    self.assertEqual(all_zeros, [{i: 0 for i in range(10)}])

  def testTreeMapWithPathMultipleTrees(self):
    tree1 = [AnObject2(x=12,
                       y={'cin': [1, 4, 10], 'bar': None},
                       z='constantdef'),
              5]
    tree2 = [AnObject2(x=2,
                       y={'cin': [2, 2, 2], 'bar': None},
                       z='constantdef'),
              2]
    from_two_trees = tree_util.tree_map_with_path(
        lambda kp, a, b: a + b, tree1, tree2
    )
    from_one_tree = tree_util.tree_map(lambda a: a + 2, tree1)
    self.assertEqual(from_two_trees, from_one_tree)

  def testTreeLeavesWithPath(self):
    tree = [{i: i for i in range(10)}]
    actual = tree_util.tree_leaves_with_path(tree)
    expected = [((tree_util.SequenceKey(0), tree_util.DictKey(i)), i)
                for i in range(10)]
    self.assertEqual(actual, expected)

  def testKeyStr(self):
    tree1 = [ATuple(12, {'cin': [1, 4, 10], 'bar': None}), jnp.arange(5)]
    flattened, _ = tree_util.tree_flatten_with_path(tree1)
    strs = [f"{tree_util.keystr(kp)}: {x}" for kp, x in flattened]
    self.assertEqual(
        strs,
        [
            "[0].foo: 12",
            "[0].bar['cin'][0]: 1",
            "[0].bar['cin'][1]: 4",
            "[0].bar['cin'][2]: 10",
            "[1]: [0 1 2 3 4]",
        ],
    )

  def testTreeMapWithPathWithIsLeafArgument(self):
    x = ((1, 2), [3, 4, 5])
    y = (([3], jnp.array(0)), ([0], 7, [5, 6]))
    out = tree_util.tree_map_with_path(
        lambda kp, *xs: (kp[0].idx, *xs), x, y,
        is_leaf=lambda n: isinstance(n, list))
    self.assertEqual(out, (((0, 1, [3]),
                            (0, 2, jnp.array(0))),
                           (1, [3, 4, 5], ([0], 7, [5, 6]))))

  def testFlattenWithPathWithIsLeafArgument(self):
    def is_empty(x):
      try:
        children, _ = flatten_one_level(x)
      except ValueError:
        return True  # Cannot flatten x; means it must be a leaf
      return len(children) == 0

    EmptyTuple = collections.namedtuple("EmptyTuple", ())
    tree1 = {'a': 1,
             'sub': [jnp.array((1, 2)), ATuple(foo=(), bar=[None])],
             'obj': AnObject2(x=EmptyTuple(), y=0, z='constantdef')}
    flattened, _ = tree_util.tree_flatten_with_path(tree1, is_empty)
    strs = [f"{tree_util.keystr(kp)}: {x}" for kp, x in flattened]
    self.assertEqual(
        strs,
        [
            "['a']: 1",
            "['obj']x: EmptyTuple()",
            "['obj']y: 0",
            "['sub'][0]: [1 2]",
            "['sub'][1].foo: ()",
            "['sub'][1].bar[0]: None",
        ],
    )

  def testFlattenOneLevel(self):
    EmptyTuple = collections.namedtuple("EmptyTuple", ())
    tree1 = {'a': 1,
             'sub': [jnp.array((1, 2)), ATuple(foo=(), bar=[None])],
             'obj': AnObject2(x=EmptyTuple(), y=0, z='constantdef')}
    self.assertEqual(flatten_one_level(tree1["sub"])[0],
                     tree1["sub"])
    self.assertEqual(flatten_one_level(tree1["sub"][1])[0],
                     [(), [None]])
    self.assertEqual(flatten_one_level(tree1["obj"])[0],
                     [EmptyTuple(), 0])
    with self.assertRaisesRegex(ValueError, "can't tree-flatten type"):
      flatten_one_level(1)
    with self.assertRaisesRegex(ValueError, "can't tree-flatten type"):
      flatten_one_level(jnp.array((1, 2)))

  def testOptionalFlatten(self):
    @tree_util.register_pytree_with_keys_class
    class FooClass:
      def __init__(self, x, y):
        self.x = x
        self.y = y
      def tree_flatten(self):
        return ((self.x, self.y), 'treedef')
      def tree_flatten_with_keys(self):
        return (((tree_util.GetAttrKey('x'), self.x),
                 (tree_util.GetAttrKey('x'), self.y)), 'treedef')
      @classmethod
      def tree_unflatten(cls, _, children):
        return cls(*children)

    tree = FooClass(x=1, y=2)
    self.assertEqual(
        str(tree_util.tree_flatten(tree)[1]),
        "PyTreeDef(CustomNode(FooClass[treedef], [*, *]))",
    )
    self.assertEqual(
        str(tree_util.tree_flatten_with_path(tree)[1]),
        "PyTreeDef(CustomNode(FooClass[treedef], [*, *]))",
    )
    self.assertEqual(tree_util.tree_flatten(tree)[0],
                     [l for _, l in tree_util.tree_flatten_with_path(tree)[0]])

  def testPyTreeWithoutKeysIsntTreatedAsLeaf(self):
    leaves, _ = tree_util.tree_flatten_with_path(Special([1, 2], [3, 4]))
    self.assertLen(leaves, 4)

  def testNamedTupleRegisteredWithoutKeysIsntTreatedAsLeaf(self):
    leaves, _ = tree_util.tree_flatten_with_path(ATuple2(1, 'hi'))
    self.assertLen(leaves, 1)

  def testBadFlattenNonTuple(self):
    t = BadFlattenNonTuple(3, 4)
    with self.assertRaisesRegex(
        ValueError,
        "The to_iterable function for a custom PyTree node should return a"
        r" \(children, aux_data\) tuple, got 'hello'",
    ):
      tree_util.tree_flatten(t)

  def testBadFlattenBadArityTuple(self):
    t = BadFlattenBadArityTuple(3, 4)
    with self.assertRaisesRegex(
        ValueError,
        "The to_iterable function for a custom PyTree node should return a"
        r" \(children, aux_data\) tuple, got \(2, 3, 4\)",
    ):
      tree_util.tree_flatten(t)

  def testBadFlattenNonIterableLeaves(self):
    t = BadFlattenNonIterableLeaves(3, 4)
    with self.assertRaisesRegex(
        ValueError,
        "The to_iterable function for a custom PyTree node should return a"
        r" \(children, aux_data\) tuple where 'children' is iterable, got "
        r"\(7, 7\)",
    ):
      tree_util.tree_flatten(t)


class StaticTest(parameterized.TestCase):

  @parameterized.parameters(
      (StaticInt(2),),
      (StaticTuple((2, None)),),
      (StaticDict(foo=2),),
  )
  def test_trace_just_once_with_same_static(self, y):
    num_called = 0

    @jax.jit
    def fn(x: int, static_y: StaticInt):
      nonlocal num_called
      num_called += 1
      unstatic_y = type(static_y).__base__(static_y)
      [y] = tree_util.tree_leaves(unstatic_y)
      return x + y

    fn(1, y)
    fn(3, y)
    self.assertEqual(num_called, 1)

  def test_name(self):
    self.assertEqual(StaticInt.__name__, "StaticInt")
    self.assertEqual(BlackBox.__name__, "BlackBox")

  @parameterized.parameters(
      (StaticInt(2), StaticInt(4)),
      (StaticTuple((2, None)), StaticTuple((4, None))),
      (StaticDict(foo=2), StaticDict(foo=4)),
  )
  def test_trace_twice_with_different_static(self, y1, y2):
    num_called = 0

    @jax.jit
    def fn(x: int, static_y: StaticInt):
      nonlocal num_called
      num_called += 1
      unstatic_y = type(static_y).__base__(static_y)
      [y] = tree_util.tree_leaves(unstatic_y)
      return x + y

    fn(1, y1)
    fn(3, y2)
    self.assertEqual(num_called, 2)

  def test_trace_just_once_if_static_looks_constant(self):
    num_called = 0

    @jax.jit
    def fn(x: int, static_y: BlackBox):
      nonlocal num_called
      num_called += 1
      return x + static_y.value

    self.assertEqual(fn(1, BlackBox(2)), 3)
    self.assertEqual(fn(3, BlackBox(1)), 5)
    self.assertEqual(num_called, 1)

  def test_serialize_treedef(self):
    tree_structure = jax.tree_util.tree_structure([1, [2], (3,), {'a': 4, 'b': 5}])
    serialized = tree_structure.serialize_using_proto()
    new_structure = jax.tree_util.PyTreeDef.deserialize_using_proto(
      jax.tree_util.default_registry,
      serialized
    )
    self.assertEqual(tree_structure, new_structure)


class RavelUtilTest(jtu.JaxTestCase):

  def testFloats(self):
    tree = [
        jnp.array([3.0], jnp.float32),
        jnp.array([[1.0, 2.0], [3.0, 4.0]], jnp.float32),
    ]
    raveled, unravel = flatten_util.ravel_pytree(tree)
    self.assertEqual(raveled.dtype, jnp.float32)
    tree_ = unravel(raveled)
    self.assertAllClose(tree, tree_, atol=0., rtol=0.)

  def testInts(self):
    tree = [jnp.array([3], jnp.int32),
            jnp.array([[1, 2], [3, 4]], jnp.int32)]
    raveled, unravel = flatten_util.ravel_pytree(tree)
    self.assertEqual(raveled.dtype, jnp.int32)
    tree_ = unravel(raveled)
    self.assertAllClose(tree, tree_, atol=0., rtol=0.)

  @jax.numpy_dtype_promotion('standard')  # Explicitly exercises implicit dtype promotion.
  def testMixedFloatInt(self):
    tree = [jnp.array([3], jnp.int32),
            jnp.array([[1., 2.], [3., 4.]], jnp.float32)]
    raveled, unravel = flatten_util.ravel_pytree(tree)
    self.assertEqual(raveled.dtype, jnp.promote_types(jnp.float32, jnp.int32))
    tree_ = unravel(raveled)
    self.assertAllClose(tree, tree_, atol=0., rtol=0.)

  @jax.numpy_dtype_promotion('standard')  # Explicitly exercises implicit dtype promotion.
  def testMixedIntBool(self):
    tree = [jnp.array([0], jnp.bool_),
            jnp.array([[1, 2], [3, 4]], jnp.int32)]
    raveled, unravel = flatten_util.ravel_pytree(tree)
    self.assertEqual(raveled.dtype, jnp.promote_types(jnp.bool_, jnp.int32))
    tree_ = unravel(raveled)
    self.assertAllClose(tree, tree_, atol=0., rtol=0.)

  @jax.numpy_dtype_promotion('standard')  # Explicitly exercises implicit dtype promotion.
  def testMixedFloatComplex(self):
    tree = [jnp.array([1.], jnp.float32),
            jnp.array([[1, 2 + 3j], [3, 4]], jnp.complex64)]
    raveled, unravel = flatten_util.ravel_pytree(tree)
    self.assertEqual(raveled.dtype, jnp.promote_types(jnp.float32, jnp.complex64))
    tree_ = unravel(raveled)
    self.assertAllClose(tree, tree_, atol=0., rtol=0.)

  def testEmpty(self):
    tree = []
    raveled, unravel = flatten_util.ravel_pytree(tree)
    self.assertEqual(raveled.dtype, jnp.float32)  # convention
    tree_ = unravel(raveled)
    self.assertAllClose(tree, tree_, atol=0., rtol=0.)

  def testDtypePolymorphicUnravel(self):
    # https://github.com/jax-ml/jax/issues/7809
    x = jnp.arange(10, dtype=jnp.float32)
    x_flat, unravel = flatten_util.ravel_pytree(x)
    y = x_flat < 5.3
    x_ = unravel(y)
    self.assertEqual(x_.dtype, y.dtype)

  @jax.numpy_dtype_promotion('standard')  # Explicitly exercises implicit dtype promotion.
  def testDtypeMonomorphicUnravel(self):
    # https://github.com/jax-ml/jax/issues/7809
    x1 = jnp.arange(10, dtype=jnp.float32)
    x2 = jnp.arange(10, dtype=jnp.int32)
    x_flat, unravel = flatten_util.ravel_pytree((x1, x2))
    y = x_flat < 5.3
    with self.assertRaisesRegex(TypeError, 'but expected dtype'):
      _ = unravel(y)

  def test_no_recompile(self):
    x1 = jnp.array([1, 2])
    x2 = jnp.array([3, 4])
    x_flat1, unravel1 = flatten_util.ravel_pytree((x1, x2))
    x_flat2, unravel2 = flatten_util.ravel_pytree((x1, x2))
    num_traces = 0

    def run(flat, unravel):
      nonlocal num_traces
      num_traces += 1
      flat = flat + 1
      return unravel(flat)

    run = jax.jit(run, static_argnums=1)

    run(x_flat1, unravel1)
    run(x_flat2, unravel2)
    self.assertEqual(num_traces, 1)


class TreePrefixErrorsTest(jtu.JaxTestCase):

  def test_different_types(self):
    e, = prefix_errors((1, 2), [1, 2])
    expected = ("pytree structure error: different types at key path\n"
                "    in_axes")
    with self.assertRaisesRegex(ValueError, expected):
      raise e('in_axes')

  def test_different_types_nested(self):
    e, = prefix_errors(((1,), (2,)), ([3], (4,)))
    expected = ("pytree structure error: different types at key path\n"
                r"    in_axes\[0\]")
    with self.assertRaisesRegex(ValueError, expected):
      raise e('in_axes')

  def test_different_types_multiple(self):
    e1, e2 = prefix_errors(((1,), (2,)), ([3], [4]))
    expected = ("pytree structure error: different types at key path\n"
                r"    in_axes\[0\]")
    with self.assertRaisesRegex(ValueError, expected):
      raise e1('in_axes')
    expected = ("pytree structure error: different types at key path\n"
                r"    in_axes\[1\]")
    with self.assertRaisesRegex(ValueError, expected):
      raise e2('in_axes')

  def test_different_num_children_tuple(self):
    e, = prefix_errors((1,), (2, 3))
    expected = ("pytree structure error: different lengths of tuple "
                "at key path\n"
                "    in_axes")
    with self.assertRaisesRegex(ValueError, expected):
      raise e('in_axes')

  def test_different_num_children_list(self):
    e, = prefix_errors([1], [2, 3])
    expected = ("pytree structure error: different lengths of list "
                "at key path\n"
                "    in_axes")
    with self.assertRaisesRegex(ValueError, expected):
      raise e('in_axes')


  def test_different_num_children_generic(self):
    e, = prefix_errors({'hi': 1}, {'hi': 2, 'bye': 3})
    expected = ("pytree structure error: different numbers of pytree children "
                "at key path\n"
                "    in_axes")
    with self.assertRaisesRegex(ValueError, expected):
      raise e('in_axes')

  def test_different_num_children_nested(self):
    e, = prefix_errors([[1]], [[2, 3]])
    expected = ("pytree structure error: different lengths of list "
                "at key path\n"
                r"    in_axes\[0\]")
    with self.assertRaisesRegex(ValueError, expected):
      raise e('in_axes')

  def test_different_num_children_multiple(self):
    e1, e2 = prefix_errors([[1], [2]], [[3, 4], [5, 6]])
    expected = ("pytree structure error: different lengths of list "
                "at key path\n"
                r"    in_axes\[0\]")
    with self.assertRaisesRegex(ValueError, expected):
      raise e1('in_axes')
    expected = ("pytree structure error: different lengths of list "
                "at key path\n"
                r"    in_axes\[1\]")
    with self.assertRaisesRegex(ValueError, expected):
      raise e2('in_axes')

  def test_different_num_children_print_key_diff(self):
    e, = prefix_errors({'a': 1}, {'a': 2, 'b': 3})
    expected = ("so the symmetric difference on key sets is\n"
                "    b")
    with self.assertRaisesRegex(ValueError, expected):
      raise e('in_axes')

  def test_different_metadata(self):
    e, = prefix_errors({1: 2}, {3: 4})
    expected = ("pytree structure error: different pytree metadata "
                "at key path\n"
                "    in_axes")
    with self.assertRaisesRegex(ValueError, expected):
      raise e('in_axes')

  def test_different_metadata_nested(self):
    e, = prefix_errors([{1: 2}], [{3: 4}])
    expected = ("pytree structure error: different pytree metadata "
                "at key path\n"
                r"    in_axes\[0\]")
    with self.assertRaisesRegex(ValueError, expected):
      raise e('in_axes')

  def test_different_metadata_multiple(self):
    e1, e2 = prefix_errors([{1: 2}, {3: 4}], [{3: 4}, {5: 6}])
    expected = ("pytree structure error: different pytree metadata "
                "at key path\n"
                r"    in_axes\[0\]")
    with self.assertRaisesRegex(ValueError, expected):
      raise e1('in_axes')
    expected = ("pytree structure error: different pytree metadata "
                "at key path\n"
                r"    in_axes\[1\]")
    with self.assertRaisesRegex(ValueError, expected):
      raise e2('in_axes')

  def test_fallback_keypath(self):
    e, = prefix_errors(Special(1, [2]), Special(3, 4))
    expected = ("pytree structure error: different types at key path\n"
                r"    in_axes\[<flat index 1>\]")
    with self.assertRaisesRegex(ValueError, expected):
      raise e('in_axes')

  def test_no_errors(self):
    () = prefix_errors((1, 2), ((11, 12, 13), 2))

  def test_different_structure_no_children(self):
    e, = prefix_errors({}, {'a': []})
    expected = ("pytree structure error: different numbers of pytree children "
                "at key path\n"
                "    in_axes")
    with self.assertRaisesRegex(ValueError, expected):
      raise e('in_axes')

  def test_curly_braces_in_keys_no_children(self):
    e, = prefix_errors({"{oops}": {}}, {})
    expected = ("pytree structure error: different numbers of pytree children "
                "at key path\n"
                "    in_axes")
    with self.assertRaisesRegex(ValueError, expected):
      raise e('in_axes')

  def test_curly_braces_in_keys_list_length(self):
    e, = prefix_errors({"{oops}": []}, {"{oops}": [{}]})
    expected = ("pytree structure error: different lengths of list "
                "at key path\n"
                r"    in_axes\['{oops}'\]")
    with self.assertRaisesRegex(ValueError, expected):
      raise e('in_axes')

  def test_curly_braces_in_keys_different_lengths(self):
    e, = prefix_errors({"{oops}": {}}, {"{oops}": 1})
    expected = ("pytree structure error: different types at key path\n"
                r"    in_axes\['{oops}'\]")
    with self.assertRaisesRegex(ValueError, expected):
      raise e('in_axes')

  def test_curly_braces_in_keys_different_metadata(self):
    e, = prefix_errors({"{oops}": {"{a}": 1}}, {"{oops}": {"{b}": 1}})
    expected = ("pytree structure error: different pytree metadata "
                "at key path\n"
                r"    in_axes\['{oops}'\]")
    with self.assertRaisesRegex(ValueError, expected):
      raise e('in_axes')


class TreeAliasTest(jtu.JaxTestCase):
  """Simple smoke-tests for tree_util aliases under jax.tree"""

  def test_tree_all(self):
    obj = [True, True, (True, False)]
    self.assertEqual(
      jax.tree.all(obj),
      tree_util.tree_all(obj),
    )

  def test_tree_all_is_leaf(self):
    obj = [True, True, (True, False)]
    is_leaf = lambda x: isinstance(x, tuple)
    self.assertEqual(
      jax.tree.all(obj, is_leaf=is_leaf),
      tree_util.tree_all(obj, is_leaf=is_leaf),
    )

  def test_tree_flatten(self):
    obj = [1, 2, (3, 4)]
    self.assertEqual(
      jax.tree.flatten(obj),
      tree_util.tree_flatten(obj),
    )

  def test_tree_flatten_is_leaf(self):
    obj = [1, 2, (3, 4)]
    is_leaf = lambda x: isinstance(x, tuple)
    self.assertEqual(
      jax.tree.flatten(obj, is_leaf=is_leaf),
      tree_util.tree_flatten(obj, is_leaf=is_leaf),
    )

  def test_tree_leaves(self):
    obj = [1, 2, (3, 4)]
    self.assertEqual(
      jax.tree.leaves(obj),
      tree_util.tree_leaves(obj),
    )

  def test_tree_leaves_is_leaf(self):
    obj = [1, 2, (3, 4)]
    is_leaf = lambda x: isinstance(x, tuple)
    self.assertEqual(
      jax.tree.leaves(obj, is_leaf=is_leaf),
      tree_util.tree_leaves(obj, is_leaf=is_leaf),
    )

  def test_tree_map(self):
    func = lambda x: x * 2
    obj = [1, 2, (3, 4)]
    self.assertEqual(
      jax.tree.map(func, obj),
      tree_util.tree_map(func, obj),
    )

  def test_tree_map_is_leaf(self):
    func = lambda x: x * 2
    obj = [1, 2, (3, 4)]
    is_leaf = lambda x: isinstance(x, tuple)
    self.assertEqual(
      jax.tree.map(func, obj, is_leaf=is_leaf),
      tree_util.tree_map(func, obj, is_leaf=is_leaf),
    )

  def test_tree_reduce(self):
    func = lambda a, b: a + b
    obj = [1, 2, (3, 4)]
    self.assertEqual(
      jax.tree.reduce(func, obj),
      tree_util.tree_reduce(func, obj),
    )

  def test_tree_reduce_is_leaf(self):
    func = lambda a, b: a + b
    obj = [(1, 2), (3, 4)]
    is_leaf = lambda x: isinstance(x, tuple)
    self.assertEqual(
      jax.tree.reduce(func, obj, is_leaf=is_leaf),
      tree_util.tree_reduce(func, obj, is_leaf=is_leaf),
    )

  def test_tree_structure(self):
    obj = [1, 2, (3, 4)]
    self.assertEqual(
      jax.tree.structure(obj),
      tree_util.tree_structure(obj),
    )

  def test_tree_structure_is_leaf(self):
    obj = [1, 2, (3, 4)]
    is_leaf = lambda x: isinstance(x, tuple)
    self.assertEqual(
      jax.tree.structure(obj, is_leaf=is_leaf),
      tree_util.tree_structure(obj, is_leaf=is_leaf),
    )

  def test_tree_transpose(self):
    obj = [(1, 2), (3, 4), (5, 6)]
    outer_treedef = tree_util.tree_structure(['*', '*', '*'])
    inner_treedef = tree_util.tree_structure(('*', '*'))
    self.assertEqual(
      jax.tree.transpose(outer_treedef, inner_treedef, obj),
      tree_util.tree_transpose(outer_treedef, inner_treedef, obj)
    )

  def test_tree_unflatten(self):
    leaves, treedef = jax.tree.flatten([1, 2, (3, 4)])
    self.assertEqual(
      jax.tree.unflatten(treedef, leaves),
      tree_util.tree_unflatten(treedef, leaves)
    )


class RegistrationTest(jtu.JaxTestCase):

  def test_register_dataclass_missing_fields(self):
    @dataclasses.dataclass
    class Foo:
      x: int
      y: int
      z: float = dataclasses.field(init=False)

    with self.assertRaisesRegex(
        ValueError,
        "data_fields and meta_fields must include all dataclass fields.*"
        "Missing fields: {'y'}",
    ):
      tree_util.register_dataclass(Foo, data_fields=["x"], meta_fields=[])

    # ``z`` is not required, because it's not included in ``__init__``.
    tree_util.register_dataclass(Foo, data_fields=["x"], meta_fields=["y"])

  def test_register_dataclass_unexpected_fields(self):
    @dataclasses.dataclass
    class Foo:
      x: int
      y: float

    with self.assertRaisesRegex(
        ValueError,
        "data_fields and meta_fields must include all dataclass fields.*"
        "Unexpected fields: {'z'}",
    ):
      tree_util.register_dataclass(
          Foo, data_fields=["x"], meta_fields=["y", "z"]
      )

  def test_register_dataclass_drop_fields(self):
    @dataclasses.dataclass
    class Foo:
      x: int
      y: int = dataclasses.field(default=42)

    # ``y`` is explicitly excluded.
    tree_util.register_dataclass(
        Foo, data_fields=["x"], meta_fields=[], drop_fields=["y"]
    )

  def test_register_dataclass_invalid_plain_class(self):
    class Foo:
      x: int
      y: int

      def __init__(self, x, y):
        self.x = x
        self.y = y

    # ``y`` is missing, but no validation is done for plain classes.
    tree_util.register_dataclass(Foo, data_fields=["x"], meta_fields=[])


if __name__ == "__main__":
  absltest.main(testLoader=jtu.JaxTestLoader())

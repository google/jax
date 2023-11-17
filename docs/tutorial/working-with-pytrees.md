---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
kernelspec:
  display_name: Python 3
  language: python
  name: python3
language_info:
  name: python
  file_extension: .py
---

(pytrees)=

# Working with Pytrees

*Author: Vladimir Mikulik, Matteo Hessel*

Often, in JAX, we will want to operate over nested collections of arrays called *pytrees*; for instance, dicts of arrays, or lists of lists of dicts. This section will explain how to use them, give some useful snippets and point out common gotchas.

## What is a pytree?

In machine learning, we often work with tree-like containers, for instance to hold:

* Model parameters
* Dataset entries
* RL agent observations

They also often arise naturally when working in bulk with datasets (e.g., lists of lists of dicts).

In JAX, we refer to all tree-like structure built out of container-like Python objects as *pytrees*. Classes are considered container-like if they are in the pytree registry, which by default includes lists, tuples, and dicts. Any object whose type is *not* in the pytree container registry will be treated as a leaf node in the tree.

The pytree registry can be extended to include user defined container classes, by simply registring a pair of functions that specify 1) how to convert an instance of the container type to a (children, metadata) pair and 2) how to convert such a pair back to an instance of the container type. JAX will use these functions to canonicalize any tree of registered container objects into a flat tuple, and then reassemble the tree-like container before returning the processed data to the user.

Some example pytrees:

```{code-cell}
import jax
import jax.numpy as jnp

example_trees = [
    [1, 'a', object()],
    (1, (2, 3), ()),
    [1, {'k1': 2, 'k2': (3, 4)}, 5],
    {'a': 2, 'b': (2, 3)},
    jnp.array([1, 2, 3]),
]

# Let's see how many leaves they have:
for pytree in example_trees:
  leaves = jax.tree_util.tree_leaves(pytree)
  print(f"{repr(pytree):<45} has {len(leaves)} leaves: {leaves}")
```

We've also introduced our first `jax.tree_*` function, which allowed us to extract the flattened leaves from the trees.

## Working with pytrees

JAX provides a number of utilities to operate over pytrees. These can be found in the `jax.tree_util` subpackage.

The most commonly used pytree function is `tree_map`. It works analogously to Python's native `map`, but transparently operates over entire pytrees:

```{code-cell}
list_of_lists = [
    [1, 2, 3],
    [1, 2],
    [1, 2, 3, 4]
]

jax.tree_map(lambda x: x*2, list_of_lists)
```

`jax.tree_map` also allows to map a N-ary function over multiple arguments:

```{code-cell}
another_list_of_lists = list_of_lists
jax.tree_map(lambda x, y: x+y, list_of_lists, another_list_of_lists)
```

When using multiple arguments with `jax.tree_map`, the structure of the inputs must exactly match. That is, lists must have the same number of elements, dicts must have the same keys, etc.

## Example: model parameters

A simple example of training an MLP displays some ways in which pytree operations come in useful:

```
import numpy as np

def init_mlp_params(layer_widths):
  params = []
  for n_in, n_out in zip(layer_widths[:-1], layer_widths[1:]):
    params.append(
        dict(weights=np.random.normal(size=(n_in, n_out)) * np.sqrt(2/n_in),
             biases=np.ones(shape=(n_out,))
            )
    )
  return params

params = init_mlp_params([1, 128, 128, 1])
```

We can use `jax.tree_map` to check that the shapes of our parameters are what we expect:

```{code-cell}
jax.tree_map(lambda x: x.shape, params)
```

Now, let's train our MLP:

```{code-cell}
def forward(params, x):
  *hidden, last = params
  for layer in hidden:
    x = jax.nn.relu(x @ layer['weights'] + layer['biases'])
  return x @ last['weights'] + last['biases']

def loss_fn(params, x, y):
  return jnp.mean((forward(params, x) - y) ** 2)

LEARNING_RATE = 0.0001

@jax.jit
def update(params, x, y):

  grads = jax.grad(loss_fn)(params, x, y)
  # Note that `grads` is a pytree with the same structure as `params`.
  # `jax.grad` is one of the many JAX functions that has
  # built-in support for pytrees.

  # This is handy, because we can apply the SGD update using tree utils:
  return jax.tree_map(
      lambda p, g: p - LEARNING_RATE * g, params, grads
  )
```

## Custom pytree nodes

In all previous examples, we've only been considering pytrees of lists, tuples, and dicts; everything else was considered a leaf. This is because, if you define your own container class, it will be considered a leaf unless you register it with JAX. This is the case even if your container class has trees inside it.

```{code-cell}
class Special(object):
  def __init__(self, x, y):
    self.x = x
    self.y = y

jax.tree_util.tree_leaves([
    Special(0, 1),
    Special(2, 4),
])
```

Accordingly, if we try to use a `tree_map` expecting our leaves to be the elements inside the container, we will get an error:

```{code-cell}
try:
    jax.tree_map(lambda x: x + 1,
    [
      Special(0, 1),
      Special(2, 4),
    ])
except TypeError as e:
    print(f'TypeError: {e}')
```

The set of Python types that are considered internal pytree nodes is extensible, through a global registry of types, and values of registered types are traversed recursively. To register a new type, you can use `register_pytree_node()`:

```{code-cell}
from jax.tree_util import register_pytree_node

class RegisteredSpecial(Special):
  def __repr__(self):
    return "RegisteredSpecial(x={}, y={})".format(self.x, self.y)

def special_flatten(v):
  """Specifies a flattening recipe.

  Params:
    v: the value of registered type to flatten.
  Returns:
    a pair of an iterable with the children to be flattened recursively,
    and some opaque auxiliary data to pass back to the unflattening recipe.
    The auxiliary data is stored in the treedef for use during unflattening.
    The auxiliary data could be used, e.g., for dictionary keys.
  """
  children = (v.x, v.y)
  aux_data = None
  return (children, aux_data)

def special_unflatten(aux_data, children):
  """Specifies an unflattening recipe.

  Params:
    aux_data: the opaque data that was specified during flattening of the
      current treedef.
    children: the unflattened children

  Returns:
    a re-constructed object of the registered type, using the specified
    children and auxiliary data.
  """
  return RegisteredSpecial(*children)

# Global registration
register_pytree_node(
    RegisteredSpecial,
    special_flatten,    # tell JAX what are the children nodes
    special_unflatten   # tell JAX how to pack back into a RegisteredSpecial
)
```

You can now traverse the special Container structure:

```{code-cell}
jax.tree_map(lambda x: x + 1,
[
  RegisteredSpecial(0, 1),
  RegisteredSpecial(2, 4),
])
```

Modern Python comes equipped with helpful tools to make defining containers easier. Some of these will work with JAX out-of-the-box, but others require more care. For instance, a `NamedTuple` subclass doesn't need to be registered to be considered a pytree node type:

```{code-cell}
from typing import NamedTuple, Any

class MyOtherContainer(NamedTuple):
  name: str
  a: Any
  b: Any
  c: Any

# NamedTuple subclasses are handled as pytree nodes, so
# this will work out-of-the-box:
jax.tree_util.tree_leaves([
    MyOtherContainer('Alice', 1, 2, 3),
    MyOtherContainer('Bob', 4, 5, 6)
])
```

Notice that the `name` field now appears as a leaf, as all tuple elements are children.

That's the price we pay for not having to register the class the hard way.

## Pytree and JAX's transformations

Many JAX functions, like `jax.lax.scan()`, operate over pytrees of arrays.

Furthemore, all JAX function transformations can be applied to functions that accept as input and produce as output pytrees of arrays.

Some JAX function transformations take optional parameters that specify how certain input or output values should be treated (e.g. the `in_axes` and `out_axes` arguments to `vmap()`). These parameters can also be pytrees, and their structure must correspond to the pytree structure of the corresponding arguments. In particular, to be able to “match up” leaves in these parameter pytrees with values in the argument pytrees, the parameter pytrees are often constrained to be tree prefixes of the argument pytrees.

For example, if we pass the following input to `vmap()` (note that the input arguments to a function are considered a tuple):

```
(a1, {"k1": a2, "k2": a3})
```

We can use the following `in_axes` pytree to specify that only the`k2` argument is mapped (`axis=0`) and the rest aren’t mapped over (`axis=None`):

```
(None, {"k1": None, "k2": 0})
```

The optional parameter pytree structure must match that of the main input pytree. However, the optional parameters can optionally be specified as a “prefix” pytree, meaning that a single leaf value can be applied to an entire sub-pytree. For example, if we have the same `vmap()` input as above, but wish to only map over the dictionary argument, we can use:

```
(None, 0)  # equivalent to (None, {"k1": 0, "k2": 0})
```

Or, if we want every argument to be mapped, we can simply write a single leaf value that is applied over the entire argument tuple pytree:

```
0
```

This happens to be the default `in_axes` value for `vmap()``!

The same logic applies to other optional parameters that refer to specific input or output values of a transformed function, e.g. vmap’s `out_axes`.

## Explicit key paths

In a pytree each leaf has a _key path_. A key path for a leaf is a `list` of _keys_, where the length of the list is equal to the depth of the leaf in the pytree . Each _key_ is a [hashable object](https://docs.python.org/3/glossary.html#term-hashable) that represents an index into the corresponding pytree node type. The type of the key depends on the pytree node type; for example, the type of keys for `dict`s is different from the type of keys for `tuple`s.

For built-in pytree node types, the set of keys for any pytree node instance is unique. For a pytree comprising nodes with this property, the key path for each leaf is unique.

The APIs for working with key paths are:

* [`jax.tree_util.tree_flatten_with_path`](https://jax.readthedocs.io/en/latest/_autosummary/jax.tree_util.tree_flatten_with_path.html): Works similarly with `jax.tree_util.tree_flatten`, but returns key paths.

* [`jax.tree_util.tree_map_with_path`](https://jax.readthedocs.io/en/latest/_autosummary/jax.tree_util.tree_map_with_path.html): Works similarly with `jax.tree_util.tree_map`, but the function also takes key paths as arguments.

* [`jax.tree_util.keystr`](https://jax.readthedocs.io/en/latest/_autosummary/jax.tree_util.keystr.html): Given a general key path, returns a reader-friendly string expression.

One use case is to print debugging information related to a certain leaf value:

```{code-cell}
import collections
ATuple = collections.namedtuple("ATuple", ('name'))

tree = [1, {'k1': 2, 'k2': (3, 4)}, ATuple('foo')]
flattened, _ = jax.tree_util.tree_flatten_with_path(tree)
for key_path, value in flattened:
    print(f'Value of tree{jax.tree_util.keystr(key_path)}: {value}')
```

To express key paths, JAX provides a few default key types for the built-in pytree node types, namely:

*  `SequenceKey(idx: int)`: for lists and tuples.
*  `DictKey(key: Hashable)`: for dictionaries.
*  `GetAttrKey(name: str)`: for `namedtuple`s and preferably custom pytree nodes (more in the next section)

You are free to define your own key types for your own custom nodes. They will work with `jax.tree_util.keystr` as long as their `__str__()` method is also overridden with a reader-friendly expression.

```{code-cell}
for key_path, _ in flattened:
    print(f'Key path of tree{jax.tree_util.keystr(key_path)}: {repr(key_path)}')
```

## Common pytree gotchas and patterns

### Gotchas

#### Mistaking nodes for leaves
A common problem to look out for is accidentally introducing tree nodes instead of leaves:

```{code-cell}
a_tree = [jnp.zeros((2, 3)), jnp.zeros((3, 4))]

# Try to make another tree with ones instead of zeros
shapes = jax.tree_map(lambda x: x.shape, a_tree)
jax.tree_map(jnp.ones, shapes)
```

What happened is that the `shape` of an array is a tuple, which is a pytree node, with its elements as leaves. Thus, in the map, instead of calling `jnp.ones` on e.g. `(2, 3)`, it's called on `2` and `3`.

The solution will depend on the specifics, but there are two broadly applicable options:
* rewrite the code to avoid the intermediate `tree_map`.
* convert the tuple into an `np.array` or `jnp.array`, which makes the entire
sequence a leaf.

#### Handling of None
`jax.tree_utils` treats `None` as the absense of a node, not as a leaf:

```{code-cell}
jax.tree_util.tree_leaves([None, None, None])
```

Note that this is different from how the (now deprecated) dm_tree library used
to treat `None``.

#### Custom PyTrees and Initialization

One common gotcha with user-defined PyTree objects is that JAX transformations occasionally initialize them with unexpected values, so that any input validation done at initialization may fail. For example:

```{code-cell}
class MyTree:
  def __init__(self, a):
    self.a = jnp.asarray(a)

register_pytree_node(MyTree, lambda tree: ((tree.a,), None),
    lambda _, args: MyTree(*args))

tree = MyTree(jnp.arange(5.0))

jax.vmap(lambda x: x)(tree)      # Error because object() is passed to MyTree.
jax.jacobian(lambda x: x)(tree)  # Error because MyTree(...) is passed to MyTree
```

In the first case, JAX’s internals use arrays of `object()` values to infer the structure of the tree; in the second case, the jacobian of a function mapping a tree to a tree is defined as a tree of trees.

For this reason, the `__init__` and `__new__` methods of custom PyTree classes should generally avoid doing any array conversion or other input validation, or else anticipate and handle these special cases. For example:

```{code-cell}
class MyTree:
  def __init__(self, a):
    if not (type(a) is object or a is None or isinstance(a, MyTree)):
      a = jnp.asarray(a)
    self.a = a
```

Another possibility is to structure your tree_unflatten function so that it avoids calling __init__; for example:

```{code-cell}
def tree_unflatten(aux_data, children):
  del aux_data  # unused in this class
  obj = object.__new__(MyTree)
  obj.a = a
  return obj
```

If you go this route, make sure that your tree_unflatten function stays in-sync with __init__ if and when the code is updated.

### Patterns

#### Transposing trees

If you would like to transpose a pytree, i.e. turn a list of trees into a tree of lists, you can do so using `jax.tree_map`:

```{code-cell}
def tree_transpose(list_of_trees):
  """Convert a list of trees of identical structure into a single tree of lists."""
  return jax.tree_map(lambda *xs: list(xs), *list_of_trees)


# Convert a dataset from row-major to column-major:
episode_steps = [dict(t=1, obs=3), dict(t=2, obs=4)]
tree_transpose(episode_steps)
```

For more complicated transposes, JAX provides `jax.tree_transpose`, which is more verbose, but allows you specify the structure of the inner and outer Pytree for more flexibility:

```{code-cell}
jax.tree_transpose(
  outer_treedef = jax.tree_structure([0 for e in episode_steps]),
  inner_treedef = jax.tree_structure(episode_steps[0]),
  pytree_to_transpose = episode_steps
)
```

## Internal pytree handling

WARNING: This is primarily JAX internal documentation, end-users are not supposed to need to understand this.

JAX flattens pytrees into lists of leaves at the `api.py` boundary (and also in control flow primitives). This keeps downstream JAX internals simpler: transformations like `grad()`, `jit()`, and `vmap()` can handle user functions that accept and return the myriad different Python containers, while all the other parts of the system can operate on functions that only take (multiple) array arguments and always return a flat list of arrays.

When JAX flattens a pytree it will produce a list of leaves and a `treedef` object that encodes the structure of the original value. The `treedef` can then be used to construct a matching structured value after transforming the leaves. Pytrees are tree-like, rather than DAG-like or graph-like, in that we handle them assuming referential transparency and that they can’t contain reference cycles.

Here is a simple example:

```{code-cell}
from jax.tree_util import tree_flatten, tree_unflatten
import jax.numpy as jnp

# The structured value to be transformed
value_structured = [1., (2., 3.)]

# The leaves in value_flat correspond to the `*` markers in value_tree
value_flat, value_tree = tree_flatten(value_structured)
print(f"{value_flat=}\n{value_tree=}")

# Transform the flat value list using an element-wise numeric transformer
transformed_flat = list(map(lambda v: v * 2., value_flat))
print(f"{transformed_flat=}")

# Reconstruct the structured output, using the original
transformed_structured = tree_unflatten(value_tree, transformed_flat)
print(f"{transformed_structured=}")
```

## More Information

For more information on pytrees in JAX and the operations that are available, see the [Pytrees](https://jax.readthedocs.io/en/latest/pytrees.html) section in the JAX documentation.

For transparently applying arithmetic operationa to pytrees consider using the jax-ecosystem library [tree_math](https://github.com/google/tree-math)
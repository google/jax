(pallas_grids_and_blockspecs)=

# Grids and BlockSpecs

<!--* freshness: { reviewed: '2024-06-01' } *-->

(pallas_grid)=
## `grid`, a.k.a. kernels in a loop

When using {func}`jax.experimental.pallas.pallas_call` the kernel function
is executed multiple times on different inputs, as specified via the `grid` argument
to `pallas_call`. Conceptually:
```python
pl.pallas_call(some_kernel, grid=(n,))(...)
```
maps to
```python
for i in range(n):
  some_kernel(...)
```
Grids can be generalized to be multi-dimensional, corresponding to nested
loops. For example,

```python
pl.pallas_call(some_kernel, grid=(n, m))(...)
```
is equivalent to
```python
for i in range(n):
  for j in range(m):
    some_kernel(...)
```
This generalizes to any tuple of integers (a length `d` grid will correspond
to `d` nested loops).
The kernel is executed as many times
as `prod(grid)`.
The default grid value `()` results in one
kernel invocation.
Each of these invocations is referred to as a "program".
To access which program (i.e. which element of the grid) the kernel is currently
executing, we use {func}`jax.experimental.pallas.program_id`.
For example, for invocation `(1, 2)`, `program_id(axis=0)` returns `1` and
`program_id(axis=1)` returns `2`.
You can also use {func}`jax.experimental.pallas.num_programs` to get the
grid size for a given axis.

Here's an example kernel that uses a `grid` and `program_id`.

```python
>>> import jax
>>> from jax.experimental import pallas as pl

>>> def iota_kernel(o_ref):
...   i = pl.program_id(0)
...   o_ref[i] = i

```

We now execute it using `pallas_call` with an additional `grid` argument.

```python
>>> def iota(size: int):
...   return pl.pallas_call(iota_kernel,
...                         out_shape=jax.ShapeDtypeStruct((size,), jnp.int32),
...                         grid=(size,), interpret=True)()
>>> iota(8)
Array([0, 1, 2, 3, 4, 5, 6, 7], dtype=int32)

```

On GPUs, each program is executed in parallel on separate thread blocks.
Thus, we need to think about race conditions on writes to HBM.
A reasonable approach is to write our kernels in such a way that different
programs write to disjoint places in HBM to avoid these parallel writes.

On TPUs, programs are executed in a combination of parallel and sequential
(depending on the architecture) so there are slightly different considerations.
See [the Pallas TPU documentation](https://jax.readthedocs.io/en/latest/pallas/tpu/details.html#noteworthy-properties-and-restrictions).

(pallas_blockspec)=

## `BlockSpec`, a.k.a. how to chunk up inputs

In conjunction with the `grid` argument, we need to provide Pallas
the information on how to slice up the input for each invocation.
Specifically, we need to provide a mapping between *the iteration of the loop*
to *which block of our inputs and outputs to be operated on*.
This is provided via {class}`jax.experimental.pallas.BlockSpec` objects.

Before we get into the details of `BlockSpec`s, you may want
to revisit the
[Pallas Quickstart BlockSpecs example](https://jax.readthedocs.io/en/latest/pallas/quickstart.html#block-specs-by-example).

`BlockSpec`s are provided to `pallas_call` via the
`in_specs` and `out_specs`, one for each input and output respectively.

First, we discuss the semantics of `BlockSpec` when `indexing_mode == pl.Blocked()`.

Informally, the `index_map` of the `BlockSpec` takes as arguments
the invocation indices (as many as the length of the `grid` tuple),
and returns **block indices** (one block index for each axis of
the overall array). Each block index is then multiplied by the
corresponding axis size from `block_shape`
to get the actual element index on the corresponding array axis.

```{note}
Not all block shapes are supported.
  * On TPU, only blocks with rank at least 1 are supported.
    Furthermore, the last two dimensions of your block shape must be equal to
    the respective dimension of the overall array, or be divisible
    by 8 and 128 respectively. For blocks of rank 1, the block dimension
    must be equal to the array dimension, or be divisible by
    `128 * (32 / bitwidth(dtype))`.

  * On GPU, only blocks with a size that is power of 2 are supported.
```

If the block shape does not divide evenly the overall shape then the
last iteration on each axis will still receive references to blocks
of `block_shape` but the elements that are out-of-bounds are padded
on input and discarded on output. The values of the padding are unspecified, and
you should assume they is garbage. In the `interpret=True` mode, we
pad with NaN for floating-point values, to give users a chance to
spot accessing out-of-bounds elements, but this behavior should not
be depended upon. Note that at least one of the
elements in each block must be within bounds.

More precisely, the slices for each axis of the input `x` of
shape `x_shape` are computed as in the function `slice_for_invocation`
below:

```python
>>> def slices_for_invocation(x_shape: tuple[int, ...],
...                           x_spec: pl.BlockSpec,
...                           grid: tuple[int, ...],
...                           invocation_indices: tuple[int, ...]) -> tuple[slice, ...]:
...   assert len(invocation_indices) == len(grid)
...   assert all(0 <= i < grid_size for i, grid_size in zip(invocation_indices, grid))
...   block_indices = x_spec.index_map(*invocation_indices)
...   assert len(x_shape) == len(x_spec.block_shape) == len(block_indices)
...   elem_indices = []
...   for x_size, block_size, block_idx in zip(x_shape, x_spec.block_shape, block_indices):
...     start_idx = block_idx * block_size
...     # At least one element of the block must be within bounds
...     assert start_idx < x_size
...     elem_indices.append(slice(start_idx, start_idx + block_size))
...   return elem_indices

```

For example:
```python
>>> slices_for_invocation(x_shape=(100, 100),
...                       x_spec = pl.BlockSpec((10, 20), lambda i, j: (i, j)),
...                       grid = (10, 5),
...                       invocation_indices = (2, 4))
[slice(20, 30, None), slice(80, 100, None)]

>>> # Same shape of the array and blocks, but we iterate over each block 4 times
>>> slices_for_invocation(x_shape=(100, 100),
...                       x_spec = pl.BlockSpec((10, 20), lambda i, j, k: (i, j)),
...                       grid = (10, 5, 4),
...                       invocation_indices = (2, 4, 0))
[slice(20, 30, None), slice(80, 100, None)]

>>> # An example when the block is partially out-of-bounds in the 2nd axis.
>>> slices_for_invocation(x_shape=(100, 90),
...                       x_spec = pl.BlockSpec((10, 20), lambda i, j: (i, j)),
...                       grid = (10, 5),
...                       invocation_indices = (2, 4))
[slice(20, 30, None), slice(80, 100, None)]

```

The function `show_program_ids` defined below uses Pallas to show the
invocation indices. The `iota_2D_kernel` will fill each output block
with a decimal number where the first digit represents the invocation
index over the first axis, and the second the invocation index
over the second axis:

```python
>>> def show_program_ids(x_shape, block_shape, grid,
...                      index_map=lambda i, j: (i, j),
...                      indexing_mode=pl.Blocked()):
...   def program_ids_kernel(o_ref):  # Fill the output block with 10*program_id(1) + program_id(0)
...     axes = 0
...     for axis in range(len(grid)):
...       axes += pl.program_id(axis) * 10**(len(grid) - 1 - axis)
...     o_ref[...] = jnp.full(o_ref.shape, axes)
...   res = pl.pallas_call(program_ids_kernel,
...                        out_shape=jax.ShapeDtypeStruct(x_shape, dtype=np.int32),
...                        grid=grid,
...                        in_specs=[],
...                        out_specs=pl.BlockSpec(block_shape, index_map, indexing_mode=indexing_mode),
...                        interpret=True)()
...   print(res)

```

For example:
```python
>>> show_program_ids(x_shape=(8, 6), block_shape=(2, 3), grid=(4, 2),
...                  index_map=lambda i, j: (i, j))
[[ 0  0  0  1  1  1]
 [ 0  0  0  1  1  1]
 [10 10 10 11 11 11]
 [10 10 10 11 11 11]
 [20 20 20 21 21 21]
 [20 20 20 21 21 21]
 [30 30 30 31 31 31]
 [30 30 30 31 31 31]]

>>> # An example with out-of-bounds accesses
>>> show_program_ids(x_shape=(7, 5), block_shape=(2, 3), grid=(4, 2),
...                  index_map=lambda i, j: (i, j))
[[ 0  0  0  1  1]
 [ 0  0  0  1  1]
 [10 10 10 11 11]
 [10 10 10 11 11]
 [20 20 20 21 21]
 [20 20 20 21 21]
 [30 30 30 31 31]]

>>> # It is allowed for the shape to be smaller than block_shape
>>> show_program_ids(x_shape=(1, 2), block_shape=(2, 3), grid=(1, 1),
...                  index_map=lambda i, j: (i, j))
[[0 0]]

```

When multiple invocations write to the same elements of the output
array the result is platform dependent.

In the example below, we have a 3D grid with the last grid dimension
not used in the block selection (`index_map=lambda i, j, k: (i, j)`).
Hence, we iterate over the same output block 10 times.
The output shown below was generated on CPU using `interpret=True`
mode, which at the moment executes the invocation sequentially.
On TPUs, programs are executed in a combination of parallel and sequential,
and this function generates the output shown.
See [the Pallas TPU documentation](https://jax.readthedocs.io/en/latest/pallas/tpu/details.html#noteworthy-properties-and-restrictions).

```python
>>> show_program_ids(x_shape=(8, 6), block_shape=(2, 3), grid=(4, 2, 10),
...                  index_map=lambda i, j, k: (i, j))
[[  9   9   9  19  19  19]
 [  9   9   9  19  19  19]
 [109 109 109 119 119 119]
 [109 109 109 119 119 119]
 [209 209 209 219 219 219]
 [209 209 209 219 219 219]
 [309 309 309 319 319 319]
 [309 309 309 319 319 319]]

```

A `None` value appearing as a dimension value in the `block_shape` behaves
as the value `1`, except that the corresponding
block axis is squeezed. In the example below, observe that the
shape of the `o_ref` is (2,) when the block shape was specified as
`(None, 2)` (the leading dimension was squeezed).

```python
>>> def kernel(o_ref):
...   assert o_ref.shape == (2,)
...   o_ref[...] = jnp.full((2,), 10 * pl.program_id(1) + pl.program_id(0))
>>> pl.pallas_call(kernel,
...                jax.ShapeDtypeStruct((3, 4), dtype=np.int32),
...                out_specs=pl.BlockSpec((None, 2), lambda i, j: (i, j)),
...                grid=(3, 2), interpret=True)()
Array([[ 0,  0, 10, 10],
       [ 1,  1, 11, 11],
       [ 2,  2, 12, 12]], dtype=int32)

```

When we construct a `BlockSpec` we can use the value `None` for the
`block_shape` parameter, in which case the shape of the overall array
is used as `block_shape`.
And if we use the value `None` for the `index_map` parameter
then a default index map function that returns a tuple of zeros is
used: `index_map=lambda *invocation_indices: (0,) * len(block_shape)`.

```python
>>> show_program_ids(x_shape=(4, 4), block_shape=None, grid=(2, 3),
...                  index_map=None)
[[12 12 12 12]
 [12 12 12 12]
 [12 12 12 12]
 [12 12 12 12]]

>>> show_program_ids(x_shape=(4, 4), block_shape=(4, 4), grid=(2, 3),
...                  index_map=None)
[[12 12 12 12]
 [12 12 12 12]
 [12 12 12 12]
 [12 12 12 12]]

```

### The "unblocked" indexing mode

The behavior documented above applies to the `indexing_mode=pl.Blocked()`.
When using the `pl.Unblocked` indexing mode the values returned by the
index map function are used directly as the array indices, without first
scaling them by the block size.
When using the unblocked mode you can specify virtual padding
of the array as a tuple of low-high paddings for each dimension: the
behavior is as if the overall array is padded on input. No guarantees
are made for the padding values in the unblocked mode, similarly to the padding
values for the blocked indexing mode when the block shape does not divide the
overall array shape.

The unblocked mode is currently supported only on TPUs.


```python
>>> # unblocked without padding
>>> show_program_ids(x_shape=(8, 6), block_shape=(2, 3), grid=(4, 2),
...                  index_map=lambda i, j: (2*i, 3*j),
...                  indexing_mode=pl.Unblocked())
    [[ 0  0  0  1  1  1]
     [ 0  0  0  1  1  1]
     [10 10 10 11 11 11]
     [10 10 10 11 11 11]
     [20 20 20 21 21 21]
     [20 20 20 21 21 21]
     [30 30 30 31 31 31]
     [30 30 30 31 31 31]]

>>> # unblocked, first pad the array with 1 row and 2 columns.
>>> show_program_ids(x_shape=(7, 7), block_shape=(2, 3), grid=(4, 3),
...                  index_map=lambda i, j: (2*i, 3*j),
...                  indexing_mode=pl.Unblocked(((1, 0), (2, 0))))
    [[ 0  1  1  1  2  2  2]
     [10 11 11 11 12 12 12]
     [10 11 11 11 12 12 12]
     [20 21 21 21 22 22 22]
     [20 21 21 21 22 22 22]
     [30 31 31 31 32 32 32]
     [30 31 31 31 32 32 32]]

```

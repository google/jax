(pallas_grids_and_blockspecs)=

# Grids and BlockSpecs

(pallas_grid)=
### `grid`, a.k.a. kernels in a loop

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
as `prod(grid)`. Each of these invocations is referred to as a "program".
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

### `BlockSpec`, a.k.a. how to chunk up inputs

```{note}
The documentation here applies to the ``indexing_mode == Blocked``, which
is the default.
The documentation for the ``indexing_mode == Unblocked`` is coming.
```

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

Informally, the `index_map` of the `BlockSpec` takes as arguments
the invocation indices (as many as the length of the `grid` tuple),
and returns **block indices** (one block index for each axis of
the overall array). Each block index is then multiplied by the
corresponding axis size from `block_shape`
to get the actual element index on the corresponding array axis.

```{note}
This documentation applies to the case when the block shape divides
the array shape.
The documentation for the other cases is pending.
```

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
...     assert block_size <= x_size  # Blocks must be smaller than the array
...     start_idx = block_idx * block_size
...     # For now, we document only the case when the entire iteration is in bounds
...     assert start_idx + block_size <= x_size
...     elem_indices.append(slice(start_idx, start_idx + block_size))
...   return elem_indices

```

For example:
```python
>>> slices_for_invocation(x_shape=(100, 100),
...                       x_spec = pl.BlockSpec(lambda i, j: (i, j), (10, 20)),
...                       grid = (10, 5),    
...                       invocation_indices = (2, 3))
[slice(20, 30, None), slice(60, 80, None)]

>>> # Same shape of the array and blocks, but we iterate over each block 4 times
>>> slices_for_invocation(x_shape=(100, 100),
...                       x_spec = pl.BlockSpec(lambda i, j, k: (i, j), (10, 20)),
...                       grid = (10, 5, 4),    
...                       invocation_indices = (2, 3, 0))
[slice(20, 30, None), slice(60, 80, None)]

```

The function `show_invocations` defined below uses Pallas to show the
invocation indices. The `iota_2D_kernel` will fill each output block
with a decimal number where the first digit represents the invocation
index over the first axis, and the second the invocation index
over the second axis:

```python
>>> def show_invocations(x_shape, block_shape, grid, out_index_map=lambda i, j: (i, j)):
...   def iota_2D_kernel(o_ref):
...    axes = 0
...    for axis in range(len(grid)):
...      axes += pl.program_id(axis) * 10**(len(grid) - 1 - axis)
...    o_ref[...] = jnp.full(o_ref.shape, axes)
...   res = pl.pallas_call(iota_2D_kernel,
...                        out_shape=jax.ShapeDtypeStruct(x_shape, dtype=np.int32),
...                        grid=grid,
...                        in_specs=[],
...                        out_specs=pl.BlockSpec(out_index_map, block_shape),
...                        interpret=True)()
...   print(res)

```

For example:
```python
>>> show_invocations(x_shape=(8, 6), block_shape=(2, 3), grid=(4, 2))
[[ 0  0  0  1  1  1]
 [ 0  0  0  1  1  1]
 [10 10 10 11 11 11]
 [10 10 10 11 11 11]
 [20 20 20 21 21 21]
 [20 20 20 21 21 21]
 [30 30 30 31 31 31]
 [30 30 30 31 31 31]]

```

When multiple invocations write to the same elements of the output
array the result is platform dependent.

In the example below, we have a 3D grid with the last grid dimension
not used in the block selection (`out_index_map=lambda i, j, k: (i, j)`).
Hence, we iterate over the same output block 10 times.
The output shown below was generated on CPU using `interpret=True`
mode, which at the moment executes the invocation sequentially.
On TPUs, programs are executed in a combination of parallel and sequential,
and this function generates the output shown. 
See [the Pallas TPU documentation](https://jax.readthedocs.io/en/latest/pallas/tpu/details.html#noteworthy-properties-and-restrictions).

```python
>>> show_invocations(x_shape=(8, 6), block_shape=(2, 3), grid=(4, 2, 10),
...                  out_index_map=lambda i, j, k: (i, j))
[[  9   9   9  19  19  19]
 [  9   9   9  19  19  19]
 [109 109 109 119 119 119]
 [109 109 109 119 119 119]
 [209 209 209 219 219 219]
 [209 209 209 219 219 219]
 [309 309 309 319 319 319]
 [309 309 309 319 319 319]]

```

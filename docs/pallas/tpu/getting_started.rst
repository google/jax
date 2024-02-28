Writing TPU kernels with Pallas
===============================

This page focuses on the details that are important when attempting to run
Pallas kernels on Google TPUs. For one, the TPU backend is still in an
experimental phase, and only a subset of JAX NumPy will be accepted.
Furthermore, writing performant code for TPUs might require thinking carefully
about the native capabilities of the hardware. While many patterns that are
unnatural to the hardware will be accepted, they might end up requiring
software emulation, and can slow down the computation.

.. warning::
  This feature should still be considered experimental as work is still in
  progress (in particular on improving the error messages).

.. note::
  While all the features described here are experimental, we remain very serious
  about maintaining their correctness. As such, it might not be uncommon to
  see a "not implemented" error while attempting to write TPU kernels. But, if
  a kernel is accepted by the compiler, it *must* return the expected results.

  If you see unexpected outputs, please compare them against a kernel run with
  ``interpret=True`` passed in to ``pallas_call``. If the results diverge,
  please file a `bug report <https://github.com/google/jax/issues/new/choose>`_.

What is a TPU?
--------------

.. image:: https://lh3.googleusercontent.com/PBWR5LFWaz8Nx4F7vRstDjt_nvUYdfxe9H3O9i3KMam_RmmwIOQMr1GAq3RUfowET2cK5kAcb_zGpw=e14-rw-lo-sc0xffffff-w2540
   :width: 400
   :align: center
   :alt: A TPUv4 board

TPU is a hardware accelerator developed at Google. You can think of TPUs as
GPUs, but specialized for machine learning workloads specifically. As such,
their architecture differs quite significantly. However, we believe that Pallas
can make it easy to start writing TPU kernels, even without having a full
understanding of the underlying hardware. Having said that, understanding the
hardware well will certainly make it easier to write performant kernels.

In a nutshell, the main difference between TPUs and GPUs is that TPUs are
sequential machines with a very wide vector register (kind of like a CPU!).
At the same time, they allow the software to schedule certain operations in the
background, making them execute asynchronously with respect to the main
instruction stream. This includes things like HBM memory accesses
(which cannot be issued directly, but instead have to be prefetched to
lower levels of the memory hierarchy by the DMA subunits), matrix multiplies
(supported by the MXU unit) or matrix transpositions and permutes (supported by
the XLU unit).

If you're interested in learning more about the TPU architecture
in detail, we recommend reading a collection of papers published over the
years. While many of them talk about specific TPU generations, many of the
ideas described transfer to later generations as well.

* `A Domain-Specific Supercomputer for Training Deep Neural Networks <https://dl.acm.org/doi/10.1145/3360307>`_
* `The Design Process for Google's Training Chips: TPUv2 and TPUv3 <https://ieeexplore.ieee.org/document/9351692>`_
* `Ten Lessons From Three Generations Shaped Google's TPUv4i : Industrial Product <https://ieeexplore.ieee.org/document/9499913>`_
* `TPU v4: An Optically Reconfigurable Supercomputer for Machine Learning with Hardware Support for Embeddings <https://dl.acm.org/doi/abs/10.1145/3579371.3589350>`_


Noteworthy properties and restrictions
--------------------------------------

``BlockSpec``\s and grid iteration
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

``BlockSpec``\s generally behave as expected in Pallas --- every invocation of
the kernel body gets access to slices of the inputs and is meant to initialize a slice
of the output.

.. warning::
  Not all window shapes are supported. If the last two dimensions of your input
  are larger than 8 and 128 respectively, the window shape in those dimensions
  must be a multiple of the respective factor. If the input dimension is smaller,
  the window should span the full dimension.

One interesting aspect of Pallas TPU kernels is the way they handle memory spaces:
While the inputs to ``pallas_call`` will often reside in HBM (the main TPU
memory), the references passed in to the kernel body will point to buffers in
lower levels of memory hierarchy (VMEM or SMEM). This enables the kernel body
to write and read them at very high speeds, while all the communication with
HBM (which has very high latency) is handled by the compiler and overlapped
with compute.

What's more, compared to GPUs, TPUs are actually highly sequential machines.
Ergo, the grid is generally not processed in parallel, but sequentially,
in lexicographic order (though see the `Multicore TPU configurations`_ section
for exceptions). This unlocks some interesting capabilities:

* When two (lexicographically) consecutive grid indices use the same slice of
  an input, the HBM transfer for the second iteration is skipped, as the data is
  already available.

* Multiple invocations of the kernel body can write to the same slice of the
  output, without any risk of race conditions. However, we do require that all
  invocations that write to a particular slice are consecutive.

The "consecutive" restriction on the output usually means that the some prefix
of the grid dimensions always vary the slice of the output an invocation needs
to access, while the output window remains constant for the remaining suffix.

For example, when implementing a Pallas TPU kernel for matrix multiplication,
one would generally use a 3 dimensional grid: the first two dimensions would
correspond to slicing along the first axis of the left operand and the second
axis of the second operand. The third and *last* grid axis would tile the
reduction dimension. The grid axis corresponding to the reduction dimension has
to be the last one, since the output window does not vary along this axis.
The output reference can be then used as an accumulator for partial results.

.. note::
  VMEM is fairly large for such a low-level memory hierarchy (16MB+), making it
  possible to use large window sizes. And, oftentimes, the larger the window
  size, the better the eventual hardware utilization will be. However, it is possible to
  specify a window size that (together with space necessary to hold
  spilled vector registers) exceeds the size of VMEM. In this case, you will likely see a
  low-level compiler error message complaining about an out-of-memory error.

Dimension ordering is meaningful
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

In JAX programs, the ordering of intermediate arrays inside ``jax.jit`` usually
has no impact on performance, as the compiler is free to rearrange them.
However, as Pallas is meant to expose lower-level capabilities, the dimension
order can have great impact on the quality of generated code.

Recall that the TPUs perform bulk of the computation on 2D vector registers.
Pallas TPU will only ever consider mapping the last two dimensions of
intermediate arrays to those vector register dimensions (sublanes and lanes
respectively). An array of shape ``(n, 1, 1)`` is guaranteed to require at least
``n`` vector registers to represent. If ``n`` becomes too large, this can lead
to spills, and potential VMEM OOM errors due to an overly large memory footprint.
But it also might not --- the low-level compiler is free to rearrange the
instructions to lower the register pressure, and is in fact very good at it.
Still, it is a good rule of thumb to keep the last two dimensions large
(especially the last dimension), while keeping the leading dimensions small.

Multicore TPU configurations
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

In newer TPU generations, the two cores on a chip are often abstracted as a
single device. To take advantage of multiple cores, Pallas has to break the
sequential grid execution guarantees, and will need to parallelize one of the
grid axes over cores. This is an opt-in procedure. To allow that,
``pallas_call`` requires an extra parameter named ``dimension_semantics``:

..
  pallas_call(
      ...,
      mosaic_params=dict(
        dimension_semantics=["parallel", "parallel", "arbitrary"]
      ),
    )

That parameter is a list, with as many entries as many axes there are in the
grid. Only ``parallel`` dimensions can be partitioned over cores. As a rule of
thumb, the dimensions are parallel, unless the output window does not vary.
As such, ``dimension_semantics`` is always a number of ``parallel`` axes
followed by a number of ``arbitrary`` axes.

While partitioning a kernel over a 2-core TPU device often leads to a 2x
speedup, it can be in fact significantly smaller. This is especially true if
different instances of the body have highly varying cost. If all of the expensive
steps get mapped to one core, but all cheap steps are assigned to the other, the
second core will be sitting idle until the first one completes its tasks.

Pallas TPU generally favors partitioning axes of a size that is a multiple of the
number of TPU cores, and prefers to partition leading grid axes.

Placing operands in SMEM
^^^^^^^^^^^^^^^^^^^^^^^^

Most of the compute on the TPU will happen on the vector unit. Still, there are
many cases where it is useful to perform a number of scalar operations, e.g., to
carry out control-flow. For that reason, TPUs come with a separate
scalar unit, and a separate scalar memory (SMEM) attached to it.
As a rule of thumb, any data used to perform control-flow decisions should
be placed in SMEM.

SMEM is a low-latency memory that supports random access, but lets you only
read and write 32-bit values with a single instruction (very small compared to
the 4KBi granularity of VMEM transactions, but much more flexible due to lack
of alignment requirements!).

The scalar memory is also very useful when implementing kernels that do not
access the tiles of inputs in a regular pattern, such as when writing
block-sparse kernels. In Pallas, this can be achieved by replacing the
``grid`` argument to ``pallas_call`` with a ``grid_spec`` of
``PrefetchScalarGridSpec`` with a non-zero ``num_scalar_prefetch`` argument.
If ``num_scalar_prefetch`` is ``n``, then the first ``n`` arguments to
``pallas_call`` will be placed in SMEM. No ``BlockSpec``\s should be specified
for those arguments. But, the ``BlockSpec``\s for all subsequent arguments will
receive not only the grid indices, but also the SMEM references to the leading
operands.

.. note::
  We are working on implementing examples for this feature. Stay tuned!

Supported data types
^^^^^^^^^^^^^^^^^^^^

At the moment Pallas TPU only supports the following data types:

* ``jnp.float32``
* ``jnp.bfloat16``
* ``jnp.int*``  (all precisions, except for ``jnp.int4``)
* ``jnp.uint*``  (all precisions)

Computation placement
^^^^^^^^^^^^^^^^^^^^^

All scalar (i.e. 0D) arrays will be stored in scalar registers, and operations
on then will be executed on the scalar core.  All other operations (even on
single-element, but 1D+ arrays) will be executed on the vector core.

Supported operations
--------------------

Matrix multiplication
^^^^^^^^^^^^^^^^^^^^^

Matrix multiplication always produces results in the float32 format.
If your inputs are not float32, we recommend using ``lax.dot`` with
``preferred_element_type`` set to ``jnp.float32``.

When using ``lax.dot_general``, it is possible to fuse transpositions of
the last two dimensions of matrix multiplication operands into the operation,
which can improve overall kernel performance.

Precision control
"""""""""""""""""

Pallas TPU lowering is aware of ``jax.default_matmul_precision``. For best
performance (and lowest precision), use ``bfloat16``. If you care about
numerical accuracy, you might want to set the precision to ``float32``.

.. warning::
  Even if you pass in 32-bit operands to a matrix multiplication, they will be
  rounded to ``bfloat16`` unless ``float32`` precision is requested.

Transposition
^^^^^^^^^^^^^

If the value has at least 4 dimensions, arbitrary transpositions of all but
the last two axes are free.
Otherwise, only the transposition of the last two axes is implemented.
Note that some transpositions of the last two dimensions can be fused into
matrix multiplication.

Accessing memory
^^^^^^^^^^^^^^^^

Arbitrary slices of references can be read or updated, subject to implementation
constraints. Currently, no restrictions are placed on inputs that are 32-bit wide,
but only some slicing patterns are supported for narrower types. Reads and
writes that are aligned to multiples of, and have a length that is a multiple
of 8 and 128 respectively in the last two dimensions are always supported.

Reads and writes to vector memory generally happen on tiles of shape ``(8, 128)``.
As such, when reading or writing to references that have at least two dimensions,
the best performance is achieved when the base offset of the memory access
has indices divisible by the tiling, and the size of the read region is a
multiple of the tile size.

Elementwise operations
^^^^^^^^^^^^^^^^^^^^^^

Many elementwise operations are supported. It is worth noting that the hardware
generally only supports elementwise compute using 32-bit types. When loading
operands that use lower-precision types, they should generally be upcast to a
32-bit type before applying elementwise ops.

It is worth noting that they can vary *significantly* in their cost. As such, we
outline three categories of supported operations: cheap (🟢), medium (🌕) and
expensive (🔴).

============================ ========
 Operation                    Cost
============================ ========
 ``jnp.add``, ``+``            🟢
 ``jnp.sub``, ``-``            🟢
 ``jnp.mul``, ``*``            🟢
 ``/``, ``//``, ``%``          🌕
 ``jnp.max``, ``jnp.min``      🟢
 ``jnp.where`` (select)        🟢
 ``jnp.abs``                   🟢
 ``|``, ``^``, ``&``, ``~``    🟢
 ``<<``, ``>>``                🟢
 Comparisons (``==``, ...)     🟢
 Type casts (``.astype``)      🟢
 ``jnp.exp``                   🌕
 ``jnp.tanh``                  🌕
 ``jnp.pow``                   🌕
 ``jnp.sin``                   🔴
 ``jnp.cos``                   🔴
============================ ========

Many JAX functions are implemented in terms of other JAX primitives, so this
list might not be comprehensive. For example, ``jax.nn.relu`` is implemented
in terms of comparisons and ``jnp.where`` and will work in Pallas kernels too.

Array constructors
^^^^^^^^^^^^^^^^^^

All constant array constructors are supported (``jnp.ones``, ``jnp.zeros``,
``jnp.full``). Notably, the ``jax.random`` module is **not** compatible with
Pallas as of today.

Reductions
^^^^^^^^^^

Sum, maximum and minimum reductions are supported, but only on a single array
axis at a time.

Reductions over the last array dimension are generally the slowest.
Reductions over the second last dimension are faster, but still slower than
over the leading dimensions.

Broadcasting
^^^^^^^^^^^^

The performance characteristics of broadcasting are very similar to those
of reductions. Broadcasting along all but the two trailing dimensions is
always supported and free. Broadcasting along the second to last dimension is
slower, while broadcasting along the last dimension is the slowest.

Reshapes
^^^^^^^^

As usual, reshapes in all dimensions but the last two dimensions are supported
and free.

The only two supported cases when a reshape can modify the last two dimensions
of an array is when (1) some leading dimensions are flattened onto the second
to last dimension, or (2) it adds a dimension that was just removed by a
reduction.

Control flow
^^^^^^^^^^^^

The TPU backend features limited support for control flow at the moment. The
currently supported functions are ``cond``, ``fori_loop`` and ``for_loop``.
However, loop primitives get fully unrolled during the compilation at the
moment, so try to keep the loop trip count reasonably small.

Overusing control flow can lead to significant regressions in low-level code
generation, and it is recommended to try to squeeze as many computationaly
expensive operations into a single basic block as possible.

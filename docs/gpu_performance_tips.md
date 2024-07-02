# GPU performance tips

<!--* freshness: { reviewed: '2024-06-10' } *-->

This document focuses on performance tips for neural network workloads

## Matmul precision

On recent GPU generations, such as the Nvidia A100 generation or later, it can
be a good idea to perform most computations in `bfloat16` precision. For
example, if using [Flax](https://github.com/google/flax), instantiate `Dense`
layers using `flax.linen.Dense(..., dtype=jax.numpy.bfloat16)`. Here are some
code examples:
* In the [Flax LM1B
  example](https://github.com/google/flax/tree/main/examples/lm1b), `Dense`
  modules are [instantiated with a configurable
  dtype](https://github.com/google/flax/blob/fd8fd76a4af5307a61f85bac98feab9b26d60db8/examples/lm1b/models.py#L188)
  which [defaults](https://github.com/google/flax/blob/fd8fd76a4af5307a61f85bac98feab9b26d60db8/examples/lm1b/configs/default.py#L112) to
  [bfloat16](https://github.com/google/flax/blob/c0087535d7f2e5bfcbf2a7be6825b9f5055a54c6/examples/lm1b/train.py#L431).
* In [MaxText](https://github.com/google/maxtext), `DenseGeneral` modules are
  also [instantiated with a configurable
  dtype](https://github.com/google/maxtext/blob/07dc6ce27ced1246407d0de311d4a0d6a9fd46d8/MaxText/layers.py#L592)
  that [defaults to
  bfloat16](https://github.com/google/maxtext/blob/07dc6ce27ced1246407d0de311d4a0d6a9fd46d8/MaxText/configs/base.yml#L41).

## XLA performance flags

```{note}
  JAX-Toolbox also has a page on [NVIDIA XLA performance FLAGS](https://github.com/NVIDIA/JAX-Toolbox/blob/main/rosetta/docs/GPU_performance.md).
```

The existence and exact behavior of XLA flags may be `jaxlib`-version dependent.

As of `jaxlib==0.4.18` (released [Oct 6
2023](https://pypi.org/project/jaxlib/#history)), setting these XLA flags can
improve performance. Some are related to communication between GPUs, and so are
only relevant when running computations on multiple devices, while others are
related to code generation on each device.

Some of these may be set by default in future releases.

These flags can be set via the `XLA_FLAGS` shell environment variable. For
example, we can add this to the top of a Python file:
```python
import os
os.environ['XLA_FLAGS'] = (
    '--xla_gpu_enable_triton_softmax_fusion=true '
    '--xla_gpu_triton_gemm_any=True '
    '--xla_gpu_enable_async_collectives=true '
    '--xla_gpu_enable_latency_hiding_scheduler=true '
    '--xla_gpu_enable_highest_priority_async_stream=true '
)
```

For more examples, see also [XLA Flags recommended for Pax
training on Nvidia GPUs](https://github.com/NVIDIA/JAX-Toolbox/blob/main/rosetta/rosetta/projects/pax/README.md#xla-flags).


### Code generation flags

* **--xla_gpu_enable_triton_softmax_fusion** This flag enables an automatic
  softmax fusion, based on pattern-matching backed by Triton code generation.
  The default value is False.
* **--xla_gpu_triton_gemm_any** Use the Triton-based GEMM (matmul) emitter for
  any GEMM that it supports. The default value is False.

### Communication flags

* **--xla_gpu_enable_latency_hiding_scheduler** This flag enables latency hiding
  schedulers to overlap asynchronous communication with computation efficiently.
  The default value is False.
* **--xla_gpu_enable_pipelined_collectives** When using pipeline parallelism,
  this flag enables overlapping the (i+1)-th layer weight `AllGather` with the
  i-th layer computation. It also enables overlapping (i+1)-th layer
  weight `Reduce`/`ReduceScatter` with i-th layer's computation. The default
  value is False. **There are some bugs when this flag is turned on.**
* **--xla_gpu_collective_permute_decomposer_threshold** This flag is useful when
  performing [GSPMD pipelining](https://arxiv.org/abs/2105.04663). Setting a
  nonzero threshold decomposes `CollectivePermute`s into
  `CollectivePermuteReceiveDone` and `CollectivePermuteSendDone` pairs, so that
  computation can be performed between each corresponding
  `ReceiveDone`/`SendDone` pair and hence achieve more overlap. By default the
  threshold is 0 and there is no decomposition. Setting it to threshold > 0 such
  as `--xla_gpu_collective_permute_decomposer_threshold=1024` can enable this
  feature.
* **--xla_gpu_all_gather_combine_threshold_bytes**
  **--xla_gpu_reduce_scatter_combine_threshold_bytes**
  **--xla_gpu_all_reduce_combine_threshold_bytes**
  These flags tune when to combine multiple small
  `AllGather`/`ReduceScatter`/`AllReduce` into one big
  `AllGather`/`ReduceScatter`/`AllReduce` to reduce time spent on cross-device
  communication. For example, for the `AllGather`/`ReduceScatter` thresholds
  on a Transformer-based workload, consider tuning them high enough so as to
  combine at least a Transformer Layer's weight `AllGather`/`ReduceScatter`. By
  default, the `combine_threshold_bytes` is set to 256.

## NCCL flags

These Nvidia NCCL flag values may be useful for single-host multi-device
computations on Nvidia GPUs:

```python
os.environ.update({
  "NCCL_LL128_BUFFSIZE": "-2",
  "NCCL_LL_BUFFSIZE": "-2",
   "NCCL_PROTO": "SIMPLE,LL,LL128",
 })
```

These NCCL flags could improve single-host communication speed. These flags
don't seem useful for multi-host communication yet.

## Multi-Process

We recommand using one process per GPU and not one per node.  In some
cases, this can speed up jitted computation. The
{func}`jax.distributed.initialize` API will automatically understand
that configuration when run under SLURM. However, this only a rule of
thumb and it may be useful to test both one process per GPU and one
process per node on your use case.

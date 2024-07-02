# Using JAX in multi-host and multi-process environments

<!--* freshness: { reviewed: '2024-06-10' } *-->

## Introduction

This guide explains how to use JAX in environments such as
GPU clusters and [Cloud TPU](https://cloud.google.com/tpu) pods where
accelerators are spread across multiple CPU hosts or JAX processes. We’ll refer
to these as “multi-process” environments.

This guide specifically focuses on how to use collective communication
operations (e.g. {func}`jax.lax.psum` ) in multi-process settings, although
other communication methods may be useful too depending on your use case (e.g.
RPC, [mpi4jax](https://github.com/mpi4jax/mpi4jax)). If you’re not already
familiar with JAX’s collective operations, we recommend starting with the
{doc}`/sharded-computation` section. An important requirement of
multi-process environments in JAX is direct communication links between
accelerators, e.g. the high-speed interconnects for Cloud TPUs or
[NCCL](https://developer.nvidia.com/nccl) for GPUs. These links allow
collective operations to run across multiple processes’ worth of accelerators
with high performance.

## Multi-process programming model

Key concepts:

  * You must run at least one JAX process per host.
  * You should initialize the cluster with {func}`jax.distributed.initialize`.
  * Each process has a
    distinct set of *local* devices it can address. The *global* devices are the set
    of all devices across all processes.
  * Use standard JAX parallelism APIs like {func}`~jax.jit` (see
    {doc}`/sharded-computation` tutorial) and
    {func}`~jax.experimental.shard_map.shard_map`. jax.jit only accepts
    globally shaped arrays. shard_map allows you to drop to per-device
    shape.
  * Make sure all processes run the same parallel computations in the same
    order.
  * Make sure all processes has the same number of local devices.
  * Make sure all devices are the same (e.g., all V100, or all H100).

### Launching JAX processes

Unlike other distributed systems where a single controller node manages many
worker nodes, JAX uses a “multi-controller” programming model where each JAX
Python process runs independently, sometimes referred to as a {term}`Single
Program, Multiple Data (SPMD)<SPMD>` model. Generally, the same JAX Python
program is run in each process, with only slight differences between each
process’s execution (e.g. different processes will load different input data).
Furthermore, **you must manually run your JAX program on each host!** JAX
doesn’t automatically start multiple processes from a single program invocation.

(The requirement for multiple processes is why this guide isn’t offered as a
notebook -- we don’t currently have a good way to manage multiple Python
processes from a single notebook.)

### Initializing the cluster

To initialize the cluster, you should call {func}`jax.distributed.initialize` at
the start of each process. {func}`jax.distributed.initialize` must be called
early in the program, before any JAX computations are executed.

The API {func}`jax.distributed.initialize` takes several arguments, namely:

  * `coordinator_address`: the IP address of process 0 in your cluster, together
    with a port available on that process. Process 0 will start a JAX service
    exposed via that IP address and port, to which the other processes in the
    cluster will connect.
  * `coordinator_bind_address`: the IP address and port to which the JAX service
    on process 0 in your cluster will bind. By default, it will bind to all
    available interfaces using the same port as `coordinator_address`.
  * `num_processes`: the number of processes in the cluster
  * `process_id`: the ID number of this process, in the range `[0 ..
    num_processes)`.
  * `local_device_ids`: Restricts the visible devices of the current process to
    ``local_device_ids``.

For example on GPU, a typical usage is:

```python
import jax

jax.distributed.initialize(coordinator_address="192.168.0.1:1234",
                           num_processes=2,
                           process_id=0)
```

On Cloud TPU, Slurm and Open MPI environments, you can simply call {func}`jax.distributed.initialize()` with no
arguments. Default values for the arguments will be chosen automatically.
When running on GPUs with Slurm and Open MPI, it is assumed that one process is started per GPU, i.e. each process will
be assigned only one visible local device. Otherwise it is assumed that one process is started per host,
i.e. each process will be assigned all local devices.
The Open MPI auto-initialization is only used when the JAX processes are launched via `mpirun`/`mpiexec`.

```python
import jax

jax.distributed.initialize()
```

On TPU at present calling {func}`jax.distributed.initialize` is optional, but
recommended since it enables additional checkpointing and health checking features.

### Local vs. global devices

Before we get to running multi-process computations from your program, it’s
important to understand the distinction between *local* and *global* devices.

**A process’s *local* devices are those that it can directly address and launch
computations on.** For example, on a GPU cluster, each host can only launch
computations on the directly attached GPUs. On a Cloud TPU pod, each host can
only launch computations on the 8 TPU cores attached directly to that host (see
the
[Cloud TPU System Architecture](https://cloud.google.com/tpu/docs/system-architecture)
documentation for more details). You can see a process’s local devices via
{func}`jax.local_devices()`.

**The *global* devices are the devices across all processes.** A computation can
span devices across processes and perform collective operations via the direct
communication links between devices, as long as each process launches the
computation on its local devices. You can see all available global devices via
{func}`jax.devices()`. A process’s local devices are always a subset of the
global devices.

### Running multi-process computations

So how do you actually run a computation involving cross-process communication?
**Use the same parallel evaluation APIs that you would in a single process!**

For example, {func}`~jax.experimental.shard_map.shard_map` can be used
to run a parallel computation across multiple processes. (If you’re
not already familiar with how to use `shard_map` to run across
multiple devices within a single process, check out the
{doc}`/sharded-computation` tutorial.)  Conceptually, this can be
thought of as running a pmap over a single array sharded across hosts,
where each host “sees” only its local shard of the input and output.

Here’s an example of multi-process pmap in action:

```python
# The following is run in parallel on each host on a GPU cluster or TPU pod slice.
>>> import jax
>>> jax.distributed.initialize()  # On GPU, see above for the necessary arguments.
>>> jax.device_count()  # total number of accelerator devices in the cluster
32
>>> jax.local_device_count()  # number of accelerator devices attached to this host
8
# The psum is performed over all mapped devices across the pod slice
>>> xs = jax.numpy.ones(jax.local_device_count())
>>> jax.pmap(lambda x: jax.lax.psum(x, 'i'), axis_name='i')(xs)
ShardedDeviceArray([32., 32., 32., 32., 32., 32., 32., 32.], dtype=float32)
```

**It’s very important that all processes run the same cross-process computations
in the same order.** Running the same JAX Python program in each process is
usually sufficient. Some common pitfalls to look out for that may cause
differently-ordered computations despite running the same program:

*   Processes passing differently-shaped inputs to the same parallel function
    can cause hangs or incorrect return values. Differently-shaped inputs are
    safe so long as they result in identically-shaped per-device data shards
    across processes; e.g. passing in different leading batch sizes in order to
    run on different numbers of local devices per process is ok, but having each
    process pad its batch to a different max example length is not.

*   “Last batch” issues where a parallel function is called in a (training)
    loop, and one or more processes exit the loop earlier than the rest. This
    will cause the rest to hang waiting for the already-finished processes to
    start the computation.

*   Conditions based on non-deterministic ordering of collections can cause code
    processes to hang. For example, iterating over
    `set` on current Python versions or `dict` [before Python 3.7](https://mail.python.org/pipermail/python-dev/2017-December/151283.html)
    may result in a different ordering on different processes, even with the
    same insertion order.

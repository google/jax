# Custom operations for GPUs with C++ and CUDA

JAX ships with a large number of built-in operations, but users occasionally run into a situation where they need a new operation that is not supported by JAX.

To accommodate such scenarios, JAX allows users to define custom operations and this tutorial is to explain how we can define one for GPUs and use it in single-GPU and multi-GPU environments.

This tutorial contains information from [Extending JAX with custom C++ and CUDA code](https://github.com/dfm/extending-jax) and
supposes that you are familiar with [JAX primitive](https://jax.readthedocs.io/en/latest/notebooks/How_JAX_primitives_work.html).

## RMS normalization

For this tutorial, we are going to add the RMS normalization as a custom operation in JAX.
Note that the RMS normalization can be expressed with [`jax.numpy`](https://jax.readthedocs.io/en/latest/jax.numpy.html) directly. However, we are using it as an example to show the process of creating a custom operation for GPUs.
The CUDA code in `gpu_ops/rms_norm_kernels.cu` for this operation has been borrowed from [Apex](https://github.com/NVIDIA/apex/blob/master/csrc/layer_norm_cuda_kernel.cu) and adapted to eliminate any dependency on PyTorch.


## High-level steps

This tutorial shows how to write both a custom operation and its gradient.

In C:
You need to follow these steps in C for each new JAX primitive:
* Have CUDA kernel(s).
* Create a C function that dispatches the CUDA kernel that will be called by XLA.
* Create a descriptor to convey information needed for the computation.
  * The types, the shapes and other attributes.
* Bind C functions to Python
  * To create the descriptor and to call the primitive during execution.

In Python:
You need to follow these steps in Python:
* Define a new JAX primitive (instruction/operation)
* Write Python functions to build the graph nodes with the primitive.
* Define its abstract evaluation.
* Define its lowering to MLIR.
* [Optional] Define the gradient.
* [Optional] Use [xmap](https://jax.readthedocs.io/en/latest/notebooks/xmap_tutorial.html), [custom_partitioning](https://jax.readthedocs.io/en/latest/jax.experimental.custom_partitioning.html) or [shard_map](https://jax.readthedocs.io/en/latest/jep/14273-shard-map.html) functions for fast multi-GPU.


## C code

See [`gpu_ops` code listing](#gpu_ops-code-listing) for a complete code listing of C++ and CUDA files.
`gpu_ops/rms_norm_kernels.cu` defines the following functions, which are declared with the XLA custom function signature.
These functions are responsible for launching RMS normalization kernels with the given `buffers` on the specified `stream`.

```cpp
namespace gpu_ops {
    
void rms_forward_affine_mixed_dtypes(cudaStream_t stream, void **buffers,
                                     const char *opaque,
                                     std::size_t opaque_len);

void rms_backward_affine(cudaStream_t stream, void **buffers,
                         const char *opaque, std::size_t opaque_len);

} // namespace gpu_ops
```

* `stream` is the CUDA stream to be used to execute any kernel on the GPU.
* `buffers` has all pointers to input buffers followed by all pointers to output buffers.
* `opaque` is a buffer for any extra information that is being passed to the custom functions and `opaque_len` is the length of `opaque`.

For this tutorial, an `RMSNormDescriptor` object will be passed to these functions as `opaque`.

```cpp
namespace gpu_ops {

enum ElementType { BF16, F16, F32, F64 };

struct RMSNormDescriptor {
  int n1;
  int n2;
  double eps;
  ElementType x_type;
  ElementType w_type;
  int part_grad_size;
};

} // namespace gpu_ops
```

Now, we need to expose these functions as well as `ElementType` and `RMSNormDescriptor` as a Python module, `gpu_ops`, through `pybind11`.

```cpp
pybind11::dict RMSNormRegistrations() {
  pybind11::dict dict;
  dict["rms_forward_affine_mixed_dtype"] =
      gpu_ops::EncapsulateFunction(gpu_ops::rms_forward_affine_mixed_dtypes);
  dict["rms_backward_affine"] =
      gpu_ops::EncapsulateFunction(gpu_ops::rms_backward_affine);
  return dict;
}

PYBIND11_MODULE(gpu_ops, m) {
  m.def("get_rms_norm_registrations", &RMSNormRegistrations);
  m.def("create_rms_norm_descriptor",
        [](int n1, int n2, double eps, gpu_ops::ElementType x_type,
           gpu_ops::ElementType w_type, int part_grad_size) {
          return gpu_ops::PackDescriptor(gpu_ops::RMSNormDescriptor{
              n1, n2, eps, x_type, w_type, part_grad_size});
        });

  pybind11::enum_<gpu_ops::ElementType>(m, "ElementType")
      .value("BF16", gpu_ops::ElementType::BF16)
      .value("F16", gpu_ops::ElementType::F16)
      .value("F32", gpu_ops::ElementType::F32)
      .value("F64", gpu_ops::ElementType::F64);

}
```

## Build `gpu_ops` extension module

We build the `gpu_ops` Python extension module with the aforementioned code.
(See [`gpu_ops` code listing](#gpu_ops-code-listing) for a complete code listing of C++ and CUDA files.)

```shell
python -m pip install pybind11==2.10.1
mkdir -p build
pybind_include_path=$(python -c "import pybind11; print(pybind11.get_include())")
python_executable=$(python -c 'import sys; print(sys.executable)')


nvcc --threads 4 -Xcompiler -Wall -ldl --expt-relaxed-constexpr -O3 -DNDEBUG -Xcompiler -O3 --generate-code=arch=compute_70,code=[compute_70,sm_70] --generate-code=arch=compute_75,code=[compute_75,sm_75] --generate-code=arch=compute_80,code=[compute_80,sm_80] --generate-code=arch=compute_86,code=[compute_86,sm_86] -Xcompiler=-fPIC -Xcompiler=-fvisibility=hidden -x cu -c gpu_ops/rms_norm_kernels.cu -o build/rms_norm_kernels.cu.o
c++ -I/usr/local/cuda/include -I$pybind_include_path $(${python_executable}-config --cflags) -O3 -DNDEBUG -O3 -fPIC -fvisibility=hidden -flto -fno-fat-lto-objects -o build/gpu_ops.cpp.o -c gpu_ops/gpu_ops.cpp
c++ -fPIC -O3 -DNDEBUG -O3 -flto -shared  -o build/gpu_ops$(${python_executable}-config --extension-suffix) build/gpu_ops.cpp.o build/rms_norm_kernels.cu.o -L/usr/local/cuda/lib64  -lcudadevrt -lcudart_static -lrt -lpthread -ldl
strip build/gpu_ops$(${python_executable}-config --extension-suffix)
```

## Add RMS normalization to JAX as custom call

`gpu_ops` is just a Python extension module and we need more work to plug it into JAX.

### Create primitives

We first create primitives, `_rms_norm_fwd_p` and `_rms_norm_bwd_p`, which the custom functions can be mapped to.
We set the `multiple_results` attribute to `True` for these operations, which means that the operation produces multiple outputs as a tuple.
When it is set to `False`, the operation produces a single output without a tuple.
For more details, see [How JAX primitives work](https://jax.readthedocs.io/en/latest/notebooks/How_JAX_primitives_work.html).

```python
from functools import partial

import jax
import jax.numpy as jnp
import jax._src.test_util as jtu
from build import gpu_ops
from jax import core, dtypes
from jax.interpreters import xla
from jax.lib import xla_client


# Create _rms_norm_fwd_p for forward operation.
_rms_norm_fwd_p = core.Primitive("rms_norm_fwd")
_rms_norm_fwd_p.multiple_results = True
_rms_norm_fwd_p.def_impl(partial(xla.apply_primitive, _rms_norm_fwd_p))


def rms_norm_fwd(x, weight, eps=1e-05):
    output, invvar = _rms_norm_fwd_p.bind(x, weight, eps=eps)
    return output


# Create _rms_norm_bwd_p for backward operation.
_rms_norm_bwd_p = core.Primitive("rms_norm_bwd")
_rms_norm_bwd_p.multiple_results = True
_rms_norm_bwd_p.def_impl(partial(xla.apply_primitive, _rms_norm_bwd_p))


def rms_norm_bwd(g, invvar, x, weight, eps):
    grad_input, grad_weight, part_grad = _rms_norm_bwd_p.bind(
        g, invvar, x, weight, eps=eps
    )
    return grad_input, grad_weight
```

### Lowering to MLIR custom call

To map the custom functions to the new primitives, `_rms_norm_fwd_p` and `_rms_norm_bwd_p`, we need to:

* Register custom functions as custom call targets with `xla_client.register_custom_call_target`, and
* Register lowering functions that lower the primitives to MLIR custom calls with the registered custom call targets.

The functions `_rms_norm_fwd_cuda_lowering` and `_rms_norm_bwd_cuda_lowering` below lower the primitives to MLIR custom call operations with the custom targets from `gpu_ops`.  These functions are registered with `jax.interpreters.mlir.register_lowering`.

Note that an `RMSNormDescriptor` object is created in the lowering function, and passed to the custom call as `opaque`.

```python
from functools import reduce

from jax.interpreters import mlir
from jax.interpreters.mlir import ir
from jaxlib.hlo_helpers import custom_call


# Register functions defined in gpu_ops as custom call target for GPUs
for _name, _value in gpu_ops.get_rms_norm_registrations().items():
    xla_client.register_custom_call_target(_name, _value, platform="gpu")


def element_type_to_descriptor_type_mapping(element_type):
    _element_type_to_descriptor_type_mapping = {
        ir.BF16Type.get(): gpu_ops.ElementType.BF16,
        ir.F16Type.get(): gpu_ops.ElementType.F16,
        ir.F32Type.get(): gpu_ops.ElementType.F32,
        ir.F64Type.get(): gpu_ops.ElementType.F64,
    }
    return _element_type_to_descriptor_type_mapping.get(element_type)


def default_layouts(*shapes):
    return [range(len(shape) - 1, -1, -1) for shape in shapes]


def _rms_norm_fwd_cuda_lowering(ctx, x, weight, eps):
    x_type = ir.RankedTensorType(x.type)
    x_shape = x_type.shape
    w_type = ir.RankedTensorType(weight.type)
    w_shape = w_type.shape
    iv_element_type = (
        ir.F32Type.get()
        if x_type.element_type in [ir.F16Type.get(), ir.BF16Type.get()]
        else x_type.element_type
    )

    n2 = reduce(lambda x, y: x * y, w_shape)
    n1 = reduce(lambda x, y: x * y, x_shape) // n2

    opaque = gpu_ops.create_rms_norm_descriptor(
        n1,
        n2,
        eps,
        element_type_to_descriptor_type_mapping(x_type.element_type),
        element_type_to_descriptor_type_mapping(w_type.element_type),
        0,  # unused
    )
    out = custom_call(
        b"rms_forward_affine_mixed_dtype",
        result_types=[
            ir.RankedTensorType.get(x_shape, w_type.element_type),
            ir.RankedTensorType.get((n1,), iv_element_type),
        ],
        operands=[x, weight],
        backend_config=opaque,
        operand_layouts=default_layouts(x_shape, w_shape),
        result_layouts=default_layouts(x_shape, (n1,)),
    ).results
    return out


mlir.register_lowering(
    _rms_norm_fwd_p,
    _rms_norm_fwd_cuda_lowering,
    platform="gpu",
)


def _rms_norm_bwd_cuda_lowering(ctx, grad_output, invvar, x, weight, eps):
    x_type = ir.RankedTensorType(x.type)
    x_shape = x_type.shape
    w_type = ir.RankedTensorType(weight.type)
    w_shape = w_type.shape
    iv_type = ir.RankedTensorType(invvar.type)

    n2 = reduce(lambda x, y: x * y, w_shape)
    n1 = reduce(lambda x, y: x * y, x_shape) // n2

    part_grad_shape = ctx.avals_out[-1].shape

    opaque = gpu_ops.create_rms_norm_descriptor(
        n1,
        n2,
        eps,
        element_type_to_descriptor_type_mapping(x_type.element_type),
        element_type_to_descriptor_type_mapping(w_type.element_type),
        part_grad_shape[0],
    )
    out = custom_call(
        b"rms_backward_affine",
        result_types=[
            ir.RankedTensorType.get(x_shape, x_type.element_type),
            ir.RankedTensorType.get(w_shape, w_type.element_type),
            ir.RankedTensorType.get(part_grad_shape, iv_type.element_type),
        ],
        operands=[grad_output, invvar, x, weight],
        backend_config=opaque,
        operand_layouts=default_layouts(x_shape, (n1,), x_shape, w_shape),
        result_layouts=default_layouts(x_shape, w_shape, part_grad_shape),
    ).results
    return out


mlir.register_lowering(
    _rms_norm_bwd_p,
    _rms_norm_bwd_cuda_lowering,
    platform="gpu",
)
```

## Let's test it

```python
per_core_batch_size=4
seq_len=512
emb_dim=512
x = jax.random.normal(
    jax.random.PRNGKey(0),
    shape=(jax.local_device_count() * per_core_batch_size, seq_len, emb_dim),
    dtype=jnp.bfloat16,
)
norm_shape = x.shape[-2:]
weight = jnp.ones(norm_shape, dtype=jnp.bfloat16)
```

### Test forward function

```python
out = rms_norm_fwd(x, weight)
```
```python
---------------------------------------------------------------------------
NotImplementedError                       Traceback (most recent call last)
Cell In [5], line 1
----> 1 out = rms_norm_fwd(x, weight)

...

NotImplementedError: Abstract evaluation for 'rms_norm_fwd' not implemented
```

## Abstract evaluation

The test above failed with `NotImplementedError: Abstract evaluation for 'rms_norm_fwd' not implemented`.  Why did the test fail?  What does it mean?

As part of the execution, JAX performs abstract evaluation.  As JAX has no knowledge about the new primitives, it doesn't know how to compute the output shapes and output data types, thus can't evaluate these operations abstractly.

We need to provide a function for abstract evaluation of each primitive.
These abstract evaluation functions compute the shape and the data type of the outputs, but don't compute actual values for the operations.

These functions are passed to `.def_abstract_eval` method to be registered with the corresponding primitives.

See [How JAX primitives work](https://jax.readthedocs.io/en/latest/notebooks/How_JAX_primitives_work.html#abstract-evaluation-rules) for more information on abstract evaluation.

```python
from functools import reduce
from operator import mul

from jax.core import ShapedArray


def _rms_norm_fwd_abstract(x, weight, eps):
    w_dtype = dtypes.canonicalize_dtype(weight.dtype)
    iv_dtype = dtypes.canonicalize_dtype(x.dtype)
    if iv_dtype in [jnp.float16, jnp.bfloat16]:
        iv_dtype = jnp.float32
    n2 = reduce(mul, weight.shape)
    n1 = reduce(mul, x.shape) // n2
    return (
        ShapedArray(x.shape, w_dtype, named_shape=x.named_shape),  # output
        ShapedArray((n1,), iv_dtype, named_shape=x.named_shape),  # invvar
    )


_rms_norm_fwd_p.def_abstract_eval(_rms_norm_fwd_abstract)


def _rms_norm_bwd_abstract(grad_output, invvar, x, weight, eps):
    iv_dtype = dtypes.canonicalize_dtype(invvar.dtype)
    w_dtype = dtypes.canonicalize_dtype(weight.dtype)
    x_dtype = dtypes.canonicalize_dtype(x.dtype)
    n2 = reduce(lambda x, y: x * y, weight.shape)
    n1 = reduce(lambda x, y: x * y, x.shape) // n2
    part_grad_shape = (16, n2)
    assert dtypes.canonicalize_dtype(grad_output.dtype) == w_dtype
    assert grad_output.shape == x.shape
    assert invvar.shape == (n1,)
    assert (
        iv_dtype == jnp.float32 if x_dtype in [jnp.float16, jnp.bfloat16] else x_dtype
    )
    assert grad_output.named_shape == x.named_shape
    weight_named_shape = (
        weight_named_shape if weight.named_shape else x.named_shape
    )
    return (
        ShapedArray(
            x.shape, x_dtype, named_shape=x.named_shape
        ),  # grad input
        ShapedArray(
            weight.shape, w_dtype, named_shape=weight_named_shape
        ),  # grad weight
        ShapedArray(
            part_grad_shape, iv_dtype, named_shape=weight_named_shape
        ),  # part grad
    )


_rms_norm_bwd_p.def_abstract_eval(_rms_norm_bwd_abstract)
```

## Let's test it again

### Test the forward function

```python
out = rms_norm_fwd(x, weight)
```

### Test the backward function

Now let's test the backward operation using `jax.grad` and `jtu.check_grads`.

```python
def loss(x, weight):
    predictions = rms_norm_fwd(x, weight)
    return -jnp.mean(predictions**2)


loss_grad = jax.grad(loss)
out = loss_grad(x, weight)
jtu.check_grads(loss, (x, weight), modes=["rev"], order=1)
```
```python
---------------------------------------------------------------------------
NotImplementedError                       Traceback (most recent call last)
Cell In [8], line 7
      3     return -jnp.mean(predictions**2)
      6 loss_grad = jax.grad(loss)
----> 7 out = loss_grad(x, weight)

...

NotImplementedError: Differentiation rule for 'rms_norm_fwd' not implemented
```

## Differentiation rule

The backward operation failed with the error `NotImplementedError: Differentiation rule for 'rms_norm_fwd' not implemented`.  It means that, although we have defined `rms_norm_fwd` and `rms_norm_bwd`, JAX doesn't know the relationship between them.

We can teach JAX that `rms_norm_bwd` is the backward operation for `rms_norm_fwd`, using `jax.custom_vjp` and its convention.  As the first step, we need to refine the definition of `rms_norm_fwd` and `rms_norm_bwd`.

```python
# rms_norm_fwd was previously defined as
#
# def rms_norm_fwd(x, weight, eps=1e-05):
#     output, invvar = _rms_norm_fwd_p.bind(x, weight, eps=eps)
#     return output
#
def rms_norm_fwd(x, weight, eps=1e-05):
    output, invvar = _rms_norm_fwd_p.bind(x, weight, eps=eps)
    return output, (invvar, x, weight)


# rms_norm_bwd was previously defined as
#
# def rms_norm_bwd(g, invvar, x, weight, eps):
#     grad_input, grad_weight, part_grad = _rms_norm_bwd_p.bind(
#         g, invvar, x, weight, eps=eps
#     )
#     return grad_input, grad_weight
#
def rms_norm_bwd(eps, res, g):
    invvar, x, weight = res
    grad_input, grad_weight, part_grad = _rms_norm_bwd_p.bind(
        g, invvar, x, weight, eps=eps
    )
    return grad_input, grad_weight
```

`rms_norm_fwd` now returns an extra output `(invvar, x, weight)` for the residual data and `rms_norm_bwd` takes `eps`, `res`, and `g` as the parameters.

Once the relationship between `rms_norm_fwd` and `rms_norm_bwd` is established through `jax.custom_vjp`, JAX will ensure that the residual data from `rms_norm_fwd` is passed to `rms_norm_bwd` as `res` for backward operation.
For non-differentiable parameters such as `eps`, JAX ensures that they are passed to the backward operation before the residual data.  That's why `eps` precedes `res` in the parameter list of `rms_norm_bwd`.

Now that `rms_norm_fwd` returns the residual data, which is not needed for simple RMS normalization operation, we define a wrapper around it, `rms_norm`.  It simply calls `rms_norm_fwd` and returns only `output`.  Note that `rms_norm` is annotated with `@partial(jax.custom_vjp, nondiff_argnums=(2,))` and we are passing `rms_norm_fwd` and `rms_norm_bwd` to `rms_norm.defvjp`.  It teaches JAX that, when `rms_norm` is differentiated, `rms_norm_fwd` is to be used for forward operation, and `rms_norm_bwd` is to be used for backward operation.

See [Custom derivative rules for JAX-transformable Python functions](https://jax.readthedocs.io/en/latest/notebooks/Custom_derivative_rules_for_Python_code.html#use-jax-custom-vjp-to-define-custom-reverse-mode-only-rules) for more information on `jax.custom_vjp`.

```python
@partial(jax.custom_vjp, nondiff_argnums=(2,))
def rms_norm(x, weight, eps=1e-05):
    output, _ = rms_norm_fwd(x, weight, eps=eps)
    return output


rms_norm.defvjp(rms_norm_fwd, rms_norm_bwd)
```

With the refinement we have made, the backward operation test works with a modification: `loss` now calls `rms_norm` instead of `rms_norm_fwd`.

```python
def loss(x, weight):
    predictions = rms_norm(x, weight)
    return -jnp.mean(predictions**2)


loss_grad = jax.grad(loss)
out = loss_grad(x, weight)
jtu.check_grads(loss, (x, weight), modes=["rev"], order=1)
```

## Let's test it on multiple devices

We are using `jax.experimental.pjit.pjit` for parallel execution on multiple devices, and we produce reference values with sequential execution on a single device.

### Test the forward function

Let's first test the forward operation on multiple devices.  We are creating a simple 1D mesh and sharding `x` on all devices.

```python
from jax.sharding import Mesh, PartitionSpec
from jax.experimental.pjit import pjit


mesh = Mesh(jax.local_devices(), ("x",))
ref = rms_norm(x, weight)
pjitted = pjit(
    rms_norm,
    # Shard x by batch dimension and replicate weight on all devices.
    in_shardings=(PartitionSpec("x", None, None), PartitionSpec(None, None)),
    # Shard the output by batch dimension.
    out_shardings=PartitionSpec("x", None, None),
)

with mesh:
    print(pjitted.lower(x, weight).compile().runtime_executable().hlo_modules()[0].to_string())
    out = pjitted(x, weight)

jnp.allclose(ref, out, atol=1e-5, rtol=1e-5)
```
```python
HloModule pjit_rms_norm, entry_computation_layout={(bf16[4,512,512]{2,1,0},bf16[512,512]{1,0})->bf16[4,512,512]{2,1,0}}

%fused_computation (param_1: bf16[32,512,512], param_1.3: u32[]) -> bf16[4,512,512] {
  %param_1 = bf16[32,512,512]{2,1,0} parameter(0)
  %param_1.3 = u32[] parameter(1)
  %convert.2 = s32[] convert(u32[] %param_1.3), metadata={op_name="pjit(rms_norm)/jit(main)/rms_norm_fwd[eps=1e-05]" source_file="/tmp/ipykernel_25235/3343076723.py" source_line=8}
  %constant_9 = s32[] constant(4), metadata={op_name="pjit(rms_norm)/jit(main)/rms_norm_fwd[eps=1e-05]" source_file="/tmp/ipykernel_25235/3343076723.py" source_line=8}
  %multiply.3 = s32[] multiply(s32[] %convert.2, s32[] %constant_9), metadata={op_name="pjit(rms_norm)/jit(main)/rms_norm_fwd[eps=1e-05]" source_file="/tmp/ipykernel_25235/3343076723.py" source_line=8}
  %constant_8 = s32[] constant(0), metadata={op_name="pjit(rms_norm)/jit(main)/rms_norm_fwd[eps=1e-05]" source_file="/tmp/ipykernel_25235/3343076723.py" source_line=8}
  ROOT %dynamic-slice.2 = bf16[4,512,512]{2,1,0} dynamic-slice(bf16[32,512,512]{2,1,0} %param_1, s32[] %multiply.3, s32[] %constant_8, s32[] %constant_8), dynamic_slice_sizes={4,512,512}, metadata={op_name="pjit(rms_norm)/jit(main)/rms_norm_fwd[eps=1e-05]" source_file="/tmp/ipykernel_25235/3343076723.py" source_line=8}
}

ENTRY %main.7_spmd (param: bf16[4,512,512], param.1: bf16[512,512]) -> bf16[4,512,512] {
  %param = bf16[4,512,512]{2,1,0} parameter(0), sharding={devices=[8,1,1]0,1,2,3,4,5,6,7}
  %all-gather = bf16[32,512,512]{2,1,0} all-gather(bf16[4,512,512]{2,1,0} %param), channel_id=1, replica_groups={{0,1,2,3,4,5,6,7}}, dimensions={0}, use_global_device_ids=true, metadata={op_name="pjit(rms_norm)/jit(main)/rms_norm_fwd[eps=1e-05]" source_file="/tmp/ipykernel_25235/3343076723.py" source_line=8}
  %param.1 = bf16[512,512]{1,0} parameter(1), sharding={replicated}
  %custom-call.0 = (bf16[32,512,512]{2,1,0}, f32[32]{0}) custom-call(bf16[32,512,512]{2,1,0} %all-gather, bf16[512,512]{1,0} %param.1), custom_call_target="rms_forward_affine_mixed_dtype", operand_layout_constraints={bf16[32,512,512]{2,1,0}, bf16[512,512]{1,0}}, api_version=API_VERSION_STATUS_RETURNING, metadata={op_name="pjit(rms_norm)/jit(main)/rms_norm_fwd[eps=1e-05]" source_file="/tmp/ipykernel_25235/3343076723.py" source_line=8}, backend_config=" \000\000\000\000\000\004\000\361h\343\210\265\370\344>\000\000\000\000\000\000\000\000\000\000\000\000\255\177\000\000"
  %get-tuple-element = bf16[32,512,512]{2,1,0} get-tuple-element((bf16[32,512,512]{2,1,0}, f32[32]{0}) %custom-call.0), index=0, metadata={op_name="pjit(rms_norm)/jit(main)/rms_norm_fwd[eps=1e-05]" source_file="/tmp/ipykernel_25235/3343076723.py" source_line=8}
  %partition-id = u32[] partition-id(), metadata={op_name="pjit(rms_norm)/jit(main)/rms_norm_fwd[eps=1e-05]" source_file="/tmp/ipykernel_25235/3343076723.py" source_line=8}
  ROOT %fusion = bf16[4,512,512]{2,1,0} fusion(bf16[32,512,512]{2,1,0} %get-tuple-element, u32[] %partition-id), kind=kLoop, calls=%fused_computation, metadata={op_name="pjit(rms_norm)/jit(main)/rms_norm_fwd[eps=1e-05]" source_file="/tmp/ipykernel_25235/3343076723.py" source_line=8}
}
```
```python
True
```

The values have been computed correctly for forward operation, however, the generated HLO modules shows an `all-gather` operation to replicate `x` on all devices, incurring large communication overhead.

As XLA does not have enough knowledge about the custom functions to shard input tensors, it decides to replicate them to produce correct values before making the custom call.

To avoid this duplication, we can:
- [custom_partitioning](https://jax.readthedocs.io/en/latest/jax.experimental.custom_partitioning.html): to make it behave like all native JAX operations (but more complicated)
- Use manual sharding
  - [xmap](https://jax.readthedocs.io/en/latest/notebooks/xmap_tutorial.html): deprecated and bugged in some cases when combined with grad
  - [shard_map](https://jax.readthedocs.io/en/latest/jep/14273-shard-map.html): experimental but should work

We show examples for xmap and custom_partitioning below.

### Shard the forward function with xmap

```python
jax.config.update("experimental_xmap_spmd_lowering", True)
jax.config.update("experimental_xmap_spmd_lowering_manual", True)
```

We need to modify the test code to use the xmap manual sharding with the custom operation.

We first define a function that wraps `rms_norm` with `xmap`.  As the size of the data axis that is being sharded must match the size of the corresponding mesh axis in the xmap manual sharding mode, we reshape `x` with the new shape `(device_count, x.shape[0] // device_count, *x.shape[1:])`, and `device_count` represents the size of the corresponding mesh axis.

After running `rms_norm` through `xmap`, we reshape the output to match the shape of `x` to match the expectation from clients.

```python
from jax.experimental.maps import xmap


def xmap_rms_norm(x, weight, *, device_count):
    reshaped = x.reshape(device_count, x.shape[0] // device_count, *x.shape[1:])
    xmapped = xmap(
        rms_norm,
        in_axes=(("x", None, None, None), (None, None)),
        out_axes=("x", None, None, None),
        axis_resources={"x": "x"},
    )
    reshaped_out = xmapped(reshaped, weight)
    return reshaped_out.reshape(x.shape)
```

Now we need to run `xmap_rms_norm`, not `rms_norm` through `pjit`.

```python
with mesh:

    pjitted = pjit(
        partial(xmap_rms_norm, device_count=jax.local_device_count()),
        # Shard x by batch dimension and replicate weight on all devices.
        in_shardings=(
            PartitionSpec("x", None, None),
            PartitionSpec(None, None),
        ),
        # Shard the output by batch dimension.
        out_shardings=PartitionSpec("x", None, None),
    )
    print(pjitted.lower(x, weight).compile().runtime_executable().hlo_modules()[0].to_string())
    out = pjitted(x, weight)

jnp.allclose(ref, out, atol=1e-5, rtol=1e-5)
```
```python
HloModule pjit__unnamed_wrapped_function_, entry_computation_layout={(bf16[4,512,512]{2,1,0},bf16[512,512]{1,0})->bf16[4,512,512]{2,1,0}}

ENTRY %main.17_spmd (param: bf16[4,512,512], param.1: bf16[512,512]) -> bf16[4,512,512] {
  %param = bf16[4,512,512]{2,1,0} parameter(0), sharding={devices=[8,1,1]0,1,2,3,4,5,6,7}, metadata={op_name="pjit(<unnamed wrapped function>)/jit(main)/xmap(rms_norm)/squeeze[dimensions=(0,)]" source_file="/tmp/ipykernel_25235/3123505662.py" source_line=13}
  %param.1 = bf16[512,512]{1,0} parameter(1), sharding={replicated}, metadata={op_name="pjit(<unnamed wrapped function>)/jit(main)/xmap(rms_norm)/full_to_shard[axes=OrderedDict() mesh=Mesh(device_ids=array([0, 1, 2, 3, 4, 5, 6, 7]), axis_names=(\'x\',)) manual_axes=(\'x\',)]" source_file="/tmp/ipykernel_25235/3123505662.py" source_line=13}
  %custom-call.0 = (bf16[4,512,512]{2,1,0}, f32[4]{0}) custom-call(bf16[4,512,512]{2,1,0} %param, bf16[512,512]{1,0} %param.1), custom_call_target="rms_forward_affine_mixed_dtype", operand_layout_constraints={bf16[4,512,512]{2,1,0}, bf16[512,512]{1,0}}, api_version=API_VERSION_STATUS_RETURNING, metadata={op_name="pjit(<unnamed wrapped function>)/jit(main)/xmap(rms_norm)/rms_norm_fwd[eps=1e-05]" source_file="/tmp/ipykernel_25235/3343076723.py" source_line=8}, backend_config="\004\000\000\000\000\000\004\000\361h\343\210\265\370\344>\000\000\000\000\000\000\000\000\000\000\000\000\027\177\000\000"
  ROOT %get-tuple-element = bf16[4,512,512]{2,1,0} get-tuple-element((bf16[4,512,512]{2,1,0}, f32[4]{0}) %custom-call.0), index=0, metadata={op_name="pjit(<unnamed wrapped function>)/jit(main)/xmap(rms_norm)/rms_norm_fwd[eps=1e-05]" source_file="/tmp/ipykernel_25235/3343076723.py" source_line=8}
}
```
```python
True
```

With this modification, the `all-gather` operation is eliminated and the custom call is made on each shard of `x`.

### Shard the forward function with custom_partitioning

When using custom_partitioning, it isn't needed to define the JAX config
experimental_xmap_spmd_lowering and
experimental_xmap_spmd_lowering_manual.

We create an helper function to help with all the JAX registration needed.

```python
def register_primitive(cls):
...
```

We define 2 JAX primitives, one inner primitive that map to the
real kernel we want to warp in JAX. And an outer primitive that will
be used with the custom_partitioning registration and for the
gradient. (And if you implement the interface to support vmat, it will
also be on the outer primitive).

JAX custom_partitioning implementation are callbacks from XLA to Python during XLA sharding logic.
XLA sharding goes in two phases: a sharding propagation phase and a partition phase.
The propagation phase is when XLA plan the sharding to be created. It is the partition phase that create the sharded graph.
For XLA to be able to shard our custom operations, it needs us to define 2 extra functions:
infer_sharding_from_operands() and partition(). They are used in the first and second phase respectively.

The infer_sharding_from_operands() function must do what its name say: infer the output sharding from the input sharding.

The partition() function will do a few things:
- tell which input sharding will be expected. XLA will reshad if needed.
- tell the final version of the output sharding.
- give a function that will create the new instruction from the sharded inputs.

See the code comments for more explanation:

```python
    def infer_sharding_from_operands(eps : float, mesh : jax.sharding.Mesh,
                                     arg_infos : tuple[jax._src.api.ShapeDtypeStruct],
                                     result_infos : tuple[jax._src.core.ShapedArray]):
        del eps, result_infos  # Not needed for this example.
        x_info, weight_info = arg_infos
        assert len(x_info.shape) == 3
        assert len(weight_info.shape) == 2
        # partition() will force all dims of all inputs to be replicated except the
        # first dim of x that will be kept as is.
        # This is because the implementaion can only be sharded on the batch dimensions.

        x_spec = arg_infos[0].sharding.spec
        # None mean that we replicate on that dimension.
        output_sharding = NamedSharding(mesh, PartitionSpec(x_spec[0], None, None))
        invvar_sharding = NamedSharding(mesh, PartitionSpec(x_spec[0]))
        return (output_sharding, invvar_sharding)

    def partition(eps : float, mesh : jax.sharding.Mesh,
                  arg_infos : tuple[jax._src.api.ShapeDtypeStruct],
                  result_infos : tuple[jax._src.api.ShapeDtypeStruct]):
        del result_infos  # Not needed for this example.
        x_info, weight_info = arg_infos
        assert len(x_info.shape) == 3
        assert len(weight_info.shape) == 2
        x_spec = arg_infos[0].sharding.spec
        # We only support sharding on the batch dimensions.
        # Force sharding on all others dimensions with None.
        arg_shardings = (NamedSharding(mesh, PartitionSpec(x_spec[0], None, None)),
                         NamedSharding(mesh, PartitionSpec(None, None)))
        invvar_sharding = NamedSharding(mesh, PartitionSpec(x_spec[0]))
        output_shardings = (arg_shardings[0], invvar_sharding)
        # Sharded_impl only accepts positional arugments
        # And they should be Jax traceable variables
        impl = partial(RmsNormFwdClass.impl, eps=eps)

        return mesh, impl, output_shardings, arg_shardings
register_primitive(RmsNormFwdClass)

```


### Shard the backward function with xmap

We are moving onto the backward operation using `jax.grad` on multiple devices.

Similarly to the forward operation test, we are creating a simple 1D mesh and sharding `x` on all devices.

We also define the `loss` function with `xmap_rms_norm` instead of `rms_norm`

```python
def ref_loss(x, weight):
    predictions = rms_norm(x, weight)
    return -jnp.mean(predictions**2)


ref = jax.grad(ref_loss, argnums=(0, 1))(x, weight)


# Re-define loss to use xmap_rms_norm instead of rms_norm
def xmap_loss(x, weight, *, device_count):
    predictions = xmap_rms_norm(x, weight, device_count=device_count)
    return -jnp.mean(predictions**2)


with mesh:

    pjitted = pjit(
        jax.grad(partial(xmap_loss, device_count=jax.local_device_count()), argnums=(0, 1)),
        # Shard x by batch dimension and replicate weight on all devices.
        in_shardings=(
            PartitionSpec("x", None, None),
            PartitionSpec(None, None),
        ),
        # Shard the output by batch dimension and replicate weight grad on all devices.
        out_shardings=(
            PartitionSpec("x", None, None),
            PartitionSpec(None, None),
        ),
    )
    out = pjitted(x, weight)

for r, o in zip(ref, out):
    print(jnp.allclose(r, o, atol=1e-5, rtol=1e-5))
```
```python
True
True
```

We can inspect the generated jaxpr, which is the JAX internal representation, to make sure `jax.grad` inserts a `psum` for the gradient accumulation across the devices when needed.

```python
with mesh:
    
    print(jax.make_jaxpr(pjitted)(x, weight))
```
```python
{ lambda ; a:bf16[32,512,512] b:bf16[512,512]. let
    c:bf16[32,512,512] d:bf16[512,512] = pjit[
      donated_invars=(False, False)
      in_positional_semantics=(<_PositionalSemantics.GLOBAL: 1>, <_PositionalSemantics.GLOBAL: 1>)
      in_shardings=(GSPMDSharding({devices=[8,1,1]0,1,2,3,4,5,6,7}), GSPMDSharding({replicated}))
      jaxpr={ lambda ; e:bf16[32,512,512] f:bf16[512,512]. let
          g:bf16[8,4,512,512] = reshape[
            dimensions=None
            new_sizes=(8, 4, 512, 512)
          ] e
          h:bf16[8,4,512,512] i:f32[8,4] j:bf16[8,4,512,512] k:bf16[512,512] = xmap[
            axis_resources=FrozenDict({'x': ('x',)})
            backend=None
            call_jaxpr={ lambda ; l:bf16[4,512,512;x:8] m:bf16[512,512]. let
                n:bf16[4,512,512;x:8] o:f32[4;x:8] = rms_norm_fwd[eps=1e-05] l m
              in (n, o, l, m) }
            donated_invars=(False, False)
            global_axis_sizes=FrozenDict({'x': 8})
            in_axes=(FrozenDict({'x': 0}), FrozenDict({}))
            in_positional_semantics=(<_PositionalSemantics.GLOBAL: 1>, <_PositionalSemantics.GLOBAL: 1>)
            name=rms_norm
            out_axes=(FrozenDict({'x': 0}), FrozenDict({'x': 0}), FrozenDict({'x': 0}), FrozenDict({}))
            out_positional_semantics=_PositionalSemantics.GLOBAL
            resource_env=ResourceEnv(Mesh(device_ids=array([0, 1, 2, 3, 4, 5, 6, 7]), axis_names=('x',)), ())
            spmd_in_axes=None
            spmd_out_axes=None
          ] g f
          p:bf16[32,512,512] = reshape[dimensions=None new_sizes=(32, 512, 512)] h
          q:bf16[32,512,512] = integer_pow[y=2] p
          r:bf16[32,512,512] = integer_pow[y=1] p
          s:bf16[32,512,512] = mul 2 r
          t:f32[32,512,512] = convert_element_type[
            new_dtype=float32
            weak_type=False
          ] q
          u:f32[] = reduce_sum[axes=(0, 1, 2)] t
          v:bf16[] = convert_element_type[new_dtype=bfloat16 weak_type=False] u
          w:bf16[] = div v 8.38861e+06
          _:bf16[] = neg w
          x:bf16[] = neg 1
          y:bf16[] = div x 8.38861e+06
          z:f32[] = convert_element_type[new_dtype=float32 weak_type=False] y
          ba:f32[32,512,512] = broadcast_in_dim[
            broadcast_dimensions=()
            shape=(32, 512, 512)
          ] z
          bb:bf16[32,512,512] = convert_element_type[
            new_dtype=bfloat16
            weak_type=False
          ] ba
          bc:bf16[32,512,512] = mul bb s
          bd:bf16[8,4,512,512] = reshape[
            dimensions=None
            new_sizes=(8, 4, 512, 512)
          ] bc
          be:bf16[8,4,512,512] bf:bf16[512,512] = xmap[
            axis_resources=FrozenDict({'x': ('x',)})
            backend=None
            call_jaxpr={ lambda ; bg:f32[4;x:8] bh:bf16[4,512,512;x:8] bi:bf16[512,512]
                bj:bf16[4,512,512;x:8]. let
                bk:bf16[4,512,512;x:8] bl:bf16[512,512;x:8] _:f32[16,262144;x:8] = rms_norm_bwd[
                  eps=1e-05
                ] bj bg bh bi
                bm:bf16[512,512] = psum[axes=('x',) axis_index_groups=None] bl
              in (bk, bm) }
            donated_invars=(False, False, False, False)
            global_axis_sizes=FrozenDict({'x': 8})
            in_axes=(FrozenDict({'x': 0}), FrozenDict({'x': 0}), FrozenDict({}), FrozenDict({'x': 0}))
            in_positional_semantics=(<_PositionalSemantics.GLOBAL: 1>, <_PositionalSemantics.GLOBAL: 1>)
            name=transpose(rms_norm)
            out_axes=(FrozenDict({'x': 0}), FrozenDict({}))
            out_positional_semantics=_PositionalSemantics.GLOBAL
            resource_env=ResourceEnv(Mesh(device_ids=array([0, 1, 2, 3, 4, 5, 6, 7]), axis_names=('x',)), ())
            spmd_in_axes=None
            spmd_out_axes=None
          ] i j k bd
          bn:bf16[32,512,512] = reshape[
            dimensions=None
            new_sizes=(32, 512, 512)
          ] be
        in (bn, bf) }
      name=<unnamed function>
      out_positional_semantics=_PositionalSemantics.GLOBAL
      out_shardings=(GSPMDSharding({devices=[8,1,1]0,1,2,3,4,5,6,7}), GSPMDSharding({replicated}))
      resource_env=ResourceEnv(Mesh(device_ids=array([0, 1, 2, 3, 4, 5, 6, 7]), axis_names=('x',)), ())
    ] a b
  in (c, d) }
```

We see that `bm:bf16[512,512] = psum[axes=('x',) axis_index_groups=None] bl` has been added after the call to `rms_norm_bwd` to reduce `grad_weight` across the devices on the axis `"x"`, but there is no `psum` for `grad_input`.

This is controlled by `named_shape` passed to the `ShapedArray` construction in abstract evaluation and the axes given to `xmap`.

The following code snippet from `_rms_norm_bwd_abstract` shows that `grad_input` has the exact same shape, type, and named shape as `x` does, which means `grad_input` is sharded the same way as `x`, hence no need for a `psum` for `grad_input`.
In contrast, `grad_weight` has the same shape and type as `weight` does, but, when `weight.named_shape` is empty, `x.named_shape` is used for `grad_weight`.  In `in_axes` of our `xmap` call, `weight` has no named axis and `weight.named_shape` is empty, but `grad_weight` now has a named axis `"x"` in `grad_weight.named_shape`.
This makes `jax.grad` insert `psum` on the axis `"x"` for `grad_weight`.

```
weight_named_shape = (
    weight_named_shape if weight.named_shape else x.named_shape
)
...
return (
    ShapedArray(
        x.shape, x_dtype, named_shape=x.named_shape
    ),  # grad input
    ShapedArray(
        weight.shape, w_dtype, named_shape=weight_named_shape
    ),  # grad weight
    ....
)
```

### Shard the backward function with custom_partitioning

```python
    def infer_sharding_from_operands(eps : float, mesh : jax.sharding.Mesh,
                                     arg_infos : tuple[jax._src.api.ShapeDtypeStruct],
                                     result_infos : tuple[jax._src.core.ShapedArray]):
        del eps, result_infos  # Not needed for this example.
        g_info, invvar_info, x_info, weight_info = arg_infos
        assert len(g_info.shape) == 3
        assert len(invvar_info.shape) == 1
        assert len(x_info.shape) == 3
        assert len(weight_info.shape) == 2
        # partition() will force all dims to be replicated except the batch dimension.
        x_spec = x_info.sharding.spec
        output_sharding = NamedSharding(mesh, PartitionSpec(x_spec[0], None, None))
        invvar_sharding = NamedSharding(mesh, PartitionSpec(None, None))
        return (output_sharding, invvar_sharding, output_sharding, )

    @staticmethod
    def partition(eps : float, mesh : jax.sharding.Mesh,
                  arg_infos : tuple[jax._src.api.ShapeDtypeStruct],
                  result_infos : tuple[jax._src.api.ShapeDtypeStruct]):
        del result_infos  # Not needed for this example.
        g_info, invvar_info, x_info, weight_info = arg_infos
        assert len(g_info.shape) == 3
        assert len(invvar_info.shape) == 1
        assert len(x_info.shape) == 3
        assert len(weight_info.shape) == 2

        # We only support sharding on the batch dimensions.
        # Force sharding on all others dimensions with None.
        # Also force gx, x and invvar to have the same batch sharding/replication.
        x_spec = x_info.sharding.spec
        arg_shardings = (NamedSharding(mesh, PartitionSpec(x_spec[0], None, None)),
                         NamedSharding(mesh, PartitionSpec(x_spec[0],)),
                         NamedSharding(mesh, PartitionSpec(x_spec[0], None, None)),
                         NamedSharding(mesh, PartitionSpec(None, None)))

        output_sharding = NamedSharding(mesh, PartitionSpec(x_spec[0], None, None))
        invvar_sharding = NamedSharding(mesh, PartitionSpec(None, None))
        output_shardings = (output_sharding, invvar_sharding, invvar_sharding)


        # Sharded_impl only accepts positional arugments
        # And they should be Jax traceable variables
        def impl(g, invvar, x, weight):
            grad_input, grad_weight, part_grad = _rms_norm_bwd_p.bind(
                g, invvar, x, weight, eps=eps
            )
            # We need to sum the weight gradient from all partition.
            global_weight = grad_weight
            if x_spec[0]:
                global_weight = jax.lax.psum(grad_weight, x_spec[0])
            return grad_input, global_weight, part_grad
        return mesh, impl, output_shardings, arg_shardings
register_primitive(RmsNormBwdClass)
```

## Let's put it together

Here is the complete code.
TODO: Update the code!
```
```

## Appendix

### `gpu_ops` code listing

#### `gpu_ops/kernel_helpers.h`

```cpp
// This header is not specific to our application and you'll probably want
// something like this for any extension you're building. This includes the
// infrastructure needed to serialize descriptors that are used with the
// "opaque" parameter of the GPU custom call. In our example we'll use this
// parameter to pass the size of our problem.

#ifndef _GPU_OPS_KERNEL_HELPERS_H_
#define _GPU_OPS_KERNEL_HELPERS_H_

#include <cstdint>
#include <stdexcept>
#include <string>
#include <type_traits>

#define JAX_APEX_WARP_SIZE 32

namespace gpu_ops {

// https://en.cppreference.com/w/cpp/numeric/bit_cast
template <class To, class From>
typename std::enable_if<sizeof(To) == sizeof(From) &&
                            std::is_trivially_copyable<From>::value &&
                            std::is_trivially_copyable<To>::value,
                        To>::type
bit_cast(const From &src) noexcept {
  static_assert(std::is_trivially_constructible<To>::value,
                "This implementation additionally requires destination type to "
                "be trivially constructible");

  To dst;
  memcpy(&dst, &src, sizeof(To));
  return dst;
}

template <typename T> std::string PackDescriptorAsString(const T &descriptor) {
  return std::string(bit_cast<const char *>(&descriptor), sizeof(T));
}

template <typename T>
const T *UnpackDescriptor(const char *opaque, std::size_t opaque_len) {
  if (opaque_len != sizeof(T)) {
    throw std::runtime_error("Invalid opaque object size");
  }
  return bit_cast<const T *>(opaque);
}

} // namespace gpu_ops

#endif
```

#### `gpu_ops/kernels.h`

```cpp
#ifndef _GPU_OPS_KERNELS_H_
#define _GPU_OPS_KERNELS_H_

#include <cuda_runtime_api.h>

#include <cstddef>
#include <cstdint>

namespace gpu_ops {

enum ElementType { BF16, F16, F32, F64 };

struct RMSNormDescriptor {
  int n1;
  int n2;
  double eps;
  ElementType x_type;
  ElementType w_type;
  int part_grad_size;
};

void rms_forward_affine_mixed_dtypes(cudaStream_t stream, void **buffers,
                                     const char *opaque,
                                     std::size_t opaque_len);
void rms_backward_affine(cudaStream_t stream, void **buffers,
                         const char *opaque, std::size_t opaque_len);
} // namespace gpu_ops

#endif
```

#### `gpu_ops/pybind11_kernel_helpers.h`

```cpp
// This header extends kernel_helpers.h with the pybind11 specific interface to
// serializing descriptors. It also adds a pybind11 function for wrapping our
// custom calls in a Python capsule. This is separate from kernel_helpers so
// that the CUDA code itself doesn't include pybind11. I don't think that this
// is strictly necessary, but they do it in jaxlib, so let's do it here too.

#ifndef _GPU_OPS_PYBIND11_KERNEL_HELPERS_H_
#define _GPU_OPS_PYBIND11_KERNEL_HELPERS_H_

#include <pybind11/pybind11.h>

#include "kernel_helpers.h"

namespace gpu_ops {

template <typename T> pybind11::bytes PackDescriptor(const T &descriptor) {
  return pybind11::bytes(PackDescriptorAsString(descriptor));
}

template <typename T> pybind11::capsule EncapsulateFunction(T *fn) {
  return pybind11::capsule(bit_cast<void *>(fn), "xla._CUSTOM_CALL_TARGET");
}

} // namespace gpu_ops

#endif
```

#### `gpu_ops/gpu_ops.cpp`

```cpp
#include "kernels.h"
#include "pybind11_kernel_helpers.h"

namespace {
pybind11::dict RMSNormRegistrations() {
  pybind11::dict dict;
  dict["rms_forward_affine_mixed_dtype"] =
      gpu_ops::EncapsulateFunction(gpu_ops::rms_forward_affine_mixed_dtypes);
  dict["rms_backward_affine"] =
      gpu_ops::EncapsulateFunction(gpu_ops::rms_backward_affine);
  return dict;
}

PYBIND11_MODULE(gpu_ops, m) {
  m.def("get_rms_norm_registrations", &RMSNormRegistrations);
  m.def("create_rms_norm_descriptor",
        [](int n1, int n2, double eps, gpu_ops::ElementType x_type,
           gpu_ops::ElementType w_type, int part_grad_size) {
          return gpu_ops::PackDescriptor(gpu_ops::RMSNormDescriptor{
              n1, n2, eps, x_type, w_type, part_grad_size});
        });

  pybind11::enum_<gpu_ops::ElementType>(m, "ElementType")
      .value("BF16", gpu_ops::ElementType::BF16)
      .value("F16", gpu_ops::ElementType::F16)
      .value("F32", gpu_ops::ElementType::F32)
      .value("F64", gpu_ops::ElementType::F64);

}
} // namespace
```

#### `gpu_ops/rms_norm_kernels.cu`

```cpp
#include "kernel_helpers.h"
#include "kernels.h"
#include "stdio.h"
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <iostream>

namespace {

#define DISPATCH_DOUBLE_FLOAT_HALF_AND_BFLOAT_INOUT_TYPES(TYPEIN, TYPEOUT,     \
                                                          NAME, ...)           \
  switch (TYPEIN) {                                                            \
  case gpu_ops::ElementType::F64: {                                            \
    using scalar_t_in = double;                                                \
    using accscalar_t = double;                                                \
    switch (TYPEOUT) {                                                         \
    case gpu_ops::ElementType::F64: {                                          \
      using scalar_t_out = double;                                             \
      __VA_ARGS__;                                                             \
      break;                                                                   \
    }                                                                          \
    case gpu_ops::ElementType::F32: {                                          \
      using scalar_t_out = float;                                              \
      __VA_ARGS__;                                                             \
      break;                                                                   \
    }                                                                          \
    case gpu_ops::ElementType::F16: {                                          \
      using scalar_t_out = __half;                                             \
      __VA_ARGS__;                                                             \
      break;                                                                   \
    }                                                                          \
    case gpu_ops::ElementType::BF16: {                                         \
      using scalar_t_out = __nv_bfloat16;                                      \
      __VA_ARGS__;                                                             \
      break;                                                                   \
    }                                                                          \
    default:                                                                   \
      break;                                                                   \
    }                                                                          \
    break;                                                                     \
  }                                                                            \
  case gpu_ops::ElementType::F32: {                                            \
    using scalar_t_in = float;                                                 \
    using accscalar_t = float;                                                 \
    switch (TYPEOUT) {                                                         \
    case gpu_ops::ElementType::F64: {                                          \
      using scalar_t_out = double;                                             \
      __VA_ARGS__;                                                             \
      break;                                                                   \
    }                                                                          \
    case gpu_ops::ElementType::F32: {                                          \
      using scalar_t_out = float;                                              \
      __VA_ARGS__;                                                             \
      break;                                                                   \
    }                                                                          \
    case gpu_ops::ElementType::F16: {                                          \
      using scalar_t_out = __half;                                             \
      __VA_ARGS__;                                                             \
      break;                                                                   \
    }                                                                          \
    case gpu_ops::ElementType::BF16: {                                         \
      using scalar_t_out = __nv_bfloat16;                                      \
      __VA_ARGS__;                                                             \
      break;                                                                   \
    }                                                                          \
    default:                                                                   \
      break;                                                                   \
    }                                                                          \
    break;                                                                     \
  }                                                                            \
  case gpu_ops::ElementType::F16: {                                            \
    using scalar_t_in = __half;                                                \
    using accscalar_t = float;                                                 \
    switch (TYPEOUT) {                                                         \
    case gpu_ops::ElementType::F64: {                                          \
      using scalar_t_out = double;                                             \
      __VA_ARGS__;                                                             \
      break;                                                                   \
    }                                                                          \
    case gpu_ops::ElementType::F32: {                                          \
      using scalar_t_out = float;                                              \
      __VA_ARGS__;                                                             \
      break;                                                                   \
    }                                                                          \
    case gpu_ops::ElementType::F16: {                                          \
      using scalar_t_out = __half;                                             \
      __VA_ARGS__;                                                             \
      break;                                                                   \
    }                                                                          \
    case gpu_ops::ElementType::BF16: {                                         \
      using scalar_t_out = __nv_bfloat16;                                      \
      __VA_ARGS__;                                                             \
      break;                                                                   \
    }                                                                          \
    default:                                                                   \
      break;                                                                   \
    }                                                                          \
    break;                                                                     \
  }                                                                            \
  case gpu_ops::ElementType::BF16: {                                           \
    using scalar_t_in = __nv_bfloat16;                                         \
    using accscalar_t = float;                                                 \
    switch (TYPEOUT) {                                                         \
    case gpu_ops::ElementType::F64: {                                          \
      using scalar_t_out = double;                                             \
      __VA_ARGS__;                                                             \
      break;                                                                   \
    }                                                                          \
    case gpu_ops::ElementType::F32: {                                          \
      using scalar_t_out = float;                                              \
      __VA_ARGS__;                                                             \
      break;                                                                   \
    }                                                                          \
    case gpu_ops::ElementType::F16: {                                          \
      using scalar_t_out = __half;                                             \
      __VA_ARGS__;                                                             \
      break;                                                                   \
    }                                                                          \
    case gpu_ops::ElementType::BF16: {                                         \
      using scalar_t_out = __nv_bfloat16;                                      \
      __VA_ARGS__;                                                             \
      break;                                                                   \
    }                                                                          \
    default:                                                                   \
      break;                                                                   \
    }                                                                          \
    break;                                                                     \
  }                                                                            \
  default:                                                                     \
    break;                                                                     \
  }

template <typename U>
__device__ void cuWelfordOnlineSum(const U curr, U &mu, U &sigma2, U &count) {
  count = count + U(1);
  U delta = curr - mu;
  U lmean = mu + delta / count;
  mu = lmean;
  U delta2 = curr - lmean;
  sigma2 = sigma2 + delta * delta2;
}

template <typename U>
__device__ void cuChanOnlineSum(const U muB, const U sigma2B, const U countB,
                                U &mu, U &sigma2, U &count) {
  U delta = muB - mu;
  U nA = count;
  U nB = countB;
  count = count + countB;
  U nX = count;
  if (nX > U(0)) {
    nA = nA / nX;
    nB = nB / nX;
    mu = nA * mu + nB * muB;
    sigma2 = sigma2 + sigma2B + delta * delta * nA * nB * nX;
  } else {
    mu = U(0);
    sigma2 = U(0);
  }
}

template <typename U> __device__ void cuRMSOnlineSum(const U curr, U &sigma2) {
  sigma2 = sigma2 + curr * curr;
}

template <typename U>
__device__ void cuChanRMSOnlineSum(const U sigma2B, U &sigma2) {
  sigma2 = sigma2 + sigma2B;
}

template <typename T, typename U>
__device__ void cuWelfordMuSigma2(const T *__restrict__ vals, const int n1,
                                  const int n2, const int i1, U &mu, U &sigma2,
                                  U *buf, bool rms_only) {
  // Assumptions:
  // 1) blockDim.x == warpSize
  // 2) Tensor is contiguous
  // 3) 2*blockDim.y*sizeof(U)+blockDim.y*sizeof(int) shared memory available.
  //
  // compute variance and mean over n2
  U count = U(0);
  mu = U(0);
  sigma2 = U(0);
  if (i1 < n1) {
    // one warp normalizes one n1 index,
    // synchronization is implicit
    // initialize with standard Welford algorithm
    const int numx = blockDim.x * blockDim.y;
    const int thrx = threadIdx.x + threadIdx.y * blockDim.x;
    const T *lvals = vals + i1 * n2;
    int l = 4 * thrx;
    for (; l + 3 < n2; l += 4 * numx) {
      for (int k = 0; k < 4; ++k) {
        U curr = static_cast<U>(lvals[l + k]);
        if (!rms_only) {
          cuWelfordOnlineSum<U>(curr, mu, sigma2, count);
        } else {
          cuRMSOnlineSum<U>(curr, sigma2);
        }
      }
    }
    for (; l < n2; ++l) {
      U curr = static_cast<U>(lvals[l]);
      if (!rms_only) {
        cuWelfordOnlineSum<U>(curr, mu, sigma2, count);
      } else {
        cuRMSOnlineSum<U>(curr, sigma2);
      }
    }
    // intra-warp reductions
    for (int l = 0; l <= 4; ++l) {
      int srcLaneB = (threadIdx.x + (1 << l)) & 31;
      U sigma2B = __shfl_sync(0xffffffff, sigma2, srcLaneB, warpSize);
      if (!rms_only) {
        U muB = __shfl_sync(0xffffffff, mu, srcLaneB, warpSize);
        U countB = __shfl_sync(0xffffffff, count, srcLaneB, warpSize);
        cuChanOnlineSum<U>(muB, sigma2B, countB, mu, sigma2, count);
      } else {
        cuChanRMSOnlineSum<U>(sigma2B, sigma2);
      }
    }
    // threadIdx.x == 0 has correct values for each warp
    // inter-warp reductions
    if (blockDim.y > 1) {
      U *ubuf = (U *)buf;
      U *ibuf = (U *)(ubuf + blockDim.y);
      for (int offset = blockDim.y / 2; offset > 0; offset /= 2) {
        // upper half of warps write to shared
        if (threadIdx.x == 0 && threadIdx.y >= offset &&
            threadIdx.y < 2 * offset) {
          const int wrt_y = threadIdx.y - offset;
          if (!rms_only) {
            ubuf[2 * wrt_y] = mu;
            ibuf[wrt_y] = count;
          }
          ubuf[2 * wrt_y + 1] = sigma2;
        }
        __syncthreads();
        // lower half merges
        if (threadIdx.x == 0 && threadIdx.y < offset) {
          U sigma2B = ubuf[2 * threadIdx.y + 1];
          if (!rms_only) {
            U muB = ubuf[2 * threadIdx.y];
            U countB = ibuf[threadIdx.y];
            cuChanOnlineSum<U>(muB, sigma2B, countB, mu, sigma2, count);
          } else {
            cuChanRMSOnlineSum<U>(sigma2B, sigma2);
          }
        }
        __syncthreads();
      }
      // threadIdx.x = 0 && threadIdx.y == 0 only thread that has correct values
      if (threadIdx.x == 0 && threadIdx.y == 0) {
        if (!rms_only) {
          ubuf[0] = mu;
        }
        ubuf[1] = sigma2;
      }
      __syncthreads();
      if (!rms_only) {
        mu = ubuf[0];
      }
      sigma2 = ubuf[1] / U(n2);
      // don't care about final value of count, we know count == n2
    } else {
      if (!rms_only) {
        mu = __shfl_sync(0xffffffff, mu, 0, warpSize);
      }
      sigma2 = __shfl_sync(0xffffffff, sigma2 / U(n2), 0, warpSize);
    }
  }
}

template <>
__device__ void cuWelfordMuSigma2(const __half *__restrict__ vals, const int n1,
                                  const int n2, const int i1, float &mu,
                                  float &sigma2, float *buf, bool rms_only) {
  // Assumptions:
  // 1) blockDim.x == warpSize
  // 2) Tensor is contiguous
  // 3) 2*blockDim.y*sizeof(U)+blockDim.y*sizeof(int) shared memory available.
  //
  // compute variance and mean over n2
  float count = 0.0f;
  mu = float(0);
  sigma2 = float(0);
  if (i1 < n1) {
    // one warp normalizes one n1 index,
    // synchronization is implicit
    // initialize with standard Welford algorithm
    const int numx = blockDim.x * blockDim.y;
    const int thrx = threadIdx.x + threadIdx.y * blockDim.x;
    const __half *lvals = vals + i1 * n2;
    int l = 8 * thrx;
    if ((((size_t)lvals) & 3) != 0) {
      // 16 bit alignment
      // first thread consumes first point
      if (thrx == 0) {
        float curr = static_cast<float>(lvals[0]);
        if (!rms_only) {
          cuWelfordOnlineSum(curr, mu, sigma2, count);
        } else {
          cuRMSOnlineSum(curr, sigma2);
        }
      }
      ++l;
    }
    // at this point, lvals[l] are 32 bit aligned for all threads.
    for (; l + 7 < n2; l += 8 * numx) {
      for (int k = 0; k < 8; k += 2) {
        float2 curr = __half22float2(*((__half2 *)(lvals + l + k)));
        if (!rms_only) {
          cuWelfordOnlineSum(curr.x, mu, sigma2, count);
          cuWelfordOnlineSum(curr.y, mu, sigma2, count);
        } else {
          cuRMSOnlineSum(curr.x, sigma2);
          cuRMSOnlineSum(curr.y, sigma2);
        }
      }
    }
    for (; l < n2; ++l) {
      float curr = static_cast<float>(lvals[l]);
      if (!rms_only) {
        cuWelfordOnlineSum(curr, mu, sigma2, count);
      } else {
        cuRMSOnlineSum(curr, sigma2);
      }
    }
    // intra-warp reductions
    for (int l = 0; l <= 4; ++l) {
      int srcLaneB = (threadIdx.x + (1 << l)) & 31;
      float sigma2B = __shfl_sync(0xffffffff, sigma2, srcLaneB, warpSize);
      if (!rms_only) {
        float muB = __shfl_sync(0xffffffff, mu, srcLaneB, warpSize);
        float countB = __shfl_sync(0xffffffff, count, srcLaneB, warpSize);
        cuChanOnlineSum(muB, sigma2B, countB, mu, sigma2, count);
      } else {
        cuChanRMSOnlineSum(sigma2B, sigma2);
      }
    }
    // threadIdx.x == 0 has correct values for each warp
    // inter-warp reductions
    if (blockDim.y > 1) {
      float *ubuf = (float *)buf;
      float *ibuf = (float *)(ubuf + blockDim.y);
      for (int offset = blockDim.y / 2; offset > 0; offset /= 2) {
        // upper half of warps write to shared
        if (threadIdx.x == 0 && threadIdx.y >= offset &&
            threadIdx.y < 2 * offset) {
          const int wrt_y = threadIdx.y - offset;
          ubuf[2 * wrt_y + 1] = sigma2;
          if (!rms_only) {
            ubuf[2 * wrt_y] = mu;
            ibuf[wrt_y] = count;
          }
        }
        __syncthreads();
        // lower half merges
        if (threadIdx.x == 0 && threadIdx.y < offset) {
          float sigma2B = ubuf[2 * threadIdx.y + 1];
          if (!rms_only) {
            float muB = ubuf[2 * threadIdx.y];
            float countB = ibuf[threadIdx.y];
            cuChanOnlineSum(muB, sigma2B, countB, mu, sigma2, count);
          } else {
            cuChanRMSOnlineSum(sigma2B, sigma2);
          }
        }
        __syncthreads();
      }
      // threadIdx.x = 0 && threadIdx.y == 0 only thread that has correct values
      if (threadIdx.x == 0 && threadIdx.y == 0) {
        if (!rms_only) {
          ubuf[0] = mu;
        }
        ubuf[1] = sigma2;
      }
      __syncthreads();
      if (!rms_only) {
        mu = ubuf[0];
      }
      sigma2 = ubuf[1] / float(n2);
      // don't care about final value of count, we know count == n2
    } else {
      if (!rms_only) {
        mu = __shfl_sync(0xffffffff, mu, 0, warpSize);
      }
      sigma2 = __shfl_sync(0xffffffff, sigma2 / float(n2), 0, warpSize);
    }
  }
}

// This is the un-specialized struct.  Note that we prevent instantiation of
// this struct by putting an undefined symbol in the function body so it won't
// compile.
//  template <typename T>
//  struct SharedMemory
//  {
//      // Ensure that we won't compile any un-specialized types
//      __device__ T *getPointer()
//      {
//          extern __device__ void error(void);
//          error();
//          return NULL;
//      }
//  };
// https://github.com/NVIDIA/apex/issues/246
template <typename T> struct SharedMemory;

template <> struct SharedMemory<float> {
  __device__ float *getPointer() {
    extern __shared__ float s_float[];
    return s_float;
  }
};

template <> struct SharedMemory<double> {
  __device__ double *getPointer() {
    extern __shared__ double s_double[];
    return s_double;
  }
};

template <typename T, typename U, typename V>
__device__ void cuApplyLayerNorm_(V *__restrict__ output_vals,
                                  U *__restrict__ mean, U *__restrict__ invvar,
                                  const T *__restrict__ vals, const int n1,
                                  const int n2, const U epsilon,
                                  const V *__restrict__ gamma,
                                  const V *__restrict__ beta, bool rms_only) {
  // Assumptions:
  // 1) blockDim.x == warpSize
  // 2) Tensors are contiguous
  //
  for (auto i1 = blockIdx.y; i1 < n1; i1 += gridDim.y) {
    SharedMemory<U> shared;
    U *buf = shared.getPointer();
    U mu, sigma2;
    cuWelfordMuSigma2(vals, n1, n2, i1, mu, sigma2, buf, rms_only);

    const T *lvals = vals + i1 * n2;
    V *ovals = output_vals + i1 * n2;
    U c_invvar = rsqrt(sigma2 + epsilon);
    const int numx = blockDim.x * blockDim.y;
    const int thrx = threadIdx.x + threadIdx.y * blockDim.x;
    if (gamma != NULL && (beta != NULL || rms_only)) {
      for (int i = thrx; i < n2; i += numx) {
        U curr = static_cast<U>(lvals[i]);
        if (!rms_only) {
          ovals[i] =
              gamma[i] * static_cast<V>(c_invvar * (curr - mu)) + beta[i];
        } else {
          ovals[i] = gamma[i] * static_cast<V>(c_invvar * curr);
        }
      }
    } else {
      for (int i = thrx; i < n2; i += numx) {
        U curr = static_cast<U>(lvals[i]);
        if (!rms_only) {
          ovals[i] = static_cast<V>(c_invvar * (curr - mu));
        } else {
          ovals[i] = static_cast<V>(c_invvar * curr);
        }
      }
    }
    if (threadIdx.x == 0 && threadIdx.y == 0) {
      if (!rms_only) {
        mean[i1] = mu;
      }
      invvar[i1] = c_invvar;
    }
    __syncthreads();
  }
}

template <typename T, typename U, typename V = T>
__global__ void
cuApplyRMSNorm(V *__restrict__ output_vals, U *__restrict__ invvar,
               const T *__restrict__ vals, const int n1, const int n2,
               const U epsilon, const V *__restrict__ gamma) {
  cuApplyLayerNorm_<T, U, V>(output_vals, NULL, invvar, vals, n1, n2, epsilon,
                             gamma, NULL, true);
}

template <typename T, typename U, typename V = T>
void HostApplyRMSNorm(cudaStream_t stream, V *output, U *invvar, const T *input,
                      int n1, int n2, double epsilon, const V *gamma) {
  auto getMaxGridY = []() {
    int device;
    int val;
    cudaGetDevice(&device);
    cudaDeviceGetAttribute(&val, cudaDevAttrMaxGridDimY, device);
    return val;
  };
  const dim3 threads(32, 4, 1);
  const uint64_t maxGridY = getMaxGridY();
  const dim3 blocks(1, std::min((uint64_t)n1, maxGridY), 1);
  int nshared =
      threads.y > 1 ? threads.y * sizeof(U) + (threads.y / 2) * sizeof(U) : 0;
  cuApplyRMSNorm<<<blocks, threads, nshared, stream>>>(
      output, invvar, input, n1, n2, U(epsilon), gamma);
}

template <typename T, typename U, typename V>
__device__ void cuLoadWriteStridedInputs(
    const int i1_block, const int thr_load_row_off, const int thr_load_col_off,
    const int i2_off, const int row_stride, U *warp_buf1, U *warp_buf2,
    const T *input, const V *dout, const int i1_end, const int n2,
    const U *__restrict__ mean, const U *__restrict__ invvar, bool rms_only) {
  int i1 = i1_block + thr_load_row_off;
  if (i1 < i1_end) {
    U curr_mean;
    if (!rms_only) {
      curr_mean = mean[i1];
    }
    U curr_invvar = invvar[i1];
    for (int k = 0; k < blockDim.y; ++k) {
      int i2 = i2_off + k;
      int load_idx = i1 * n2 + i2;
      int write_idx = thr_load_row_off * row_stride + thr_load_col_off + k;
      if (i2 < n2) {
        U curr_input = static_cast<U>(input[load_idx]);
        U curr_dout = static_cast<U>(dout[load_idx]);
        if (!rms_only) {
          warp_buf1[write_idx] = curr_dout;
          warp_buf2[write_idx] =
              curr_dout * (curr_input - curr_mean) * curr_invvar;
        } else {
          warp_buf2[write_idx] = curr_dout * (curr_input)*curr_invvar;
        }
      } else {
        if (!rms_only) {
          warp_buf1[write_idx] = U(0);
        }
        warp_buf2[write_idx] = U(0);
      }
    }
  } else {
    for (int k = 0; k < blockDim.y; ++k) {
      int write_idx = thr_load_row_off * row_stride + thr_load_col_off + k;
      if (!rms_only) {
        warp_buf1[write_idx] = U(0);
      }
      warp_buf2[write_idx] = U(0);
    }
  }
}

template <typename T, typename U, typename V>
__device__ void cuLoadAddStridedInputs(
    const int i1_block, const int thr_load_row_off, const int thr_load_col_off,
    const int i2_off, const int row_stride, U *warp_buf1, U *warp_buf2,
    const T *input, const V *dout, const int i1_end, const int n2,
    const U *__restrict__ mean, const U *__restrict__ invvar, bool rms_only) {
  int i1 = i1_block + thr_load_row_off;
  if (i1 < i1_end) {
    U curr_mean;
    if (!rms_only) {
      curr_mean = mean[i1];
    }
    U curr_invvar = invvar[i1];
    for (int k = 0; k < blockDim.y; ++k) {
      int i2 = i2_off + k;
      int load_idx = i1 * n2 + i2;
      int write_idx = thr_load_row_off * row_stride + thr_load_col_off + k;
      if (i2 < n2) {
        U curr_input = static_cast<U>(input[load_idx]);
        U curr_dout = static_cast<U>(dout[load_idx]);
        if (!rms_only) {
          warp_buf1[write_idx] += curr_dout;
          warp_buf2[write_idx] +=
              curr_dout * (curr_input - curr_mean) * curr_invvar;
        } else {
          warp_buf2[write_idx] += curr_dout * (curr_input)*curr_invvar;
        }
      }
    }
  }
}

template <typename T, typename U, typename V>
__global__ void cuComputePartGradGammaBeta(
    const V *__restrict__ dout, const T *__restrict__ input, const int n1,
    const int n2, const U *__restrict__ mean, const U *__restrict__ invvar,
    U epsilon, U *part_grad_gamma, U *part_grad_beta, bool rms_only) {
  const int numsegs_n1 =
      (n1 + blockDim.y * blockDim.y - 1) / (blockDim.y * blockDim.y);
  const int segs_per_block = (numsegs_n1 + gridDim.y - 1) / gridDim.y;
  const int i1_beg = blockIdx.y * segs_per_block * blockDim.y * blockDim.y;
  const int i1_beg_plus_one =
      (blockIdx.y + 1) * segs_per_block * blockDim.y * blockDim.y;
  const int i1_end = i1_beg_plus_one < n1 ? i1_beg_plus_one : n1;
  const int row_stride = blockDim.x + 1;
  const int thr_load_col_off = (threadIdx.x * blockDim.y) & (blockDim.x - 1);
  const int thr_load_row_off =
      (threadIdx.x * blockDim.y) / blockDim.x + threadIdx.y * blockDim.y;
  const int i2_off = blockIdx.x * blockDim.x + thr_load_col_off;
  SharedMemory<U> shared;
  U *buf = shared.getPointer(); // buf has at least blockDim.x * blockDim.y *
                                // blockDim.y + (blockDim.y -
                                // 1)*(blockDim.x/blockDim.y) elements
  U *warp_buf1 = (U *)buf;
  U *warp_buf2 = warp_buf1 + blockDim.y * blockDim.y * row_stride;
  // compute partial sums from strided inputs
  // do this to increase number of loads in flight
  cuLoadWriteStridedInputs(i1_beg, thr_load_row_off, thr_load_col_off, i2_off,
                           row_stride, warp_buf1, warp_buf2, input, dout,
                           i1_end, n2, mean, invvar, rms_only);
  for (int i1_block = i1_beg + blockDim.y * blockDim.y; i1_block < i1_end;
       i1_block += blockDim.y * blockDim.y) {
    cuLoadAddStridedInputs(i1_block, thr_load_row_off, thr_load_col_off, i2_off,
                           row_stride, warp_buf1, warp_buf2, input, dout,
                           i1_end, n2, mean, invvar, rms_only);
  }
  __syncthreads();
  // inter-warp reductions
  // sum within each warp
  U acc1 = U(0);
  U acc2 = U(0);
  for (int k = 0; k < blockDim.y; ++k) {
    int row1 = threadIdx.y + k * blockDim.y;
    int idx1 = row1 * row_stride + threadIdx.x;
    if (!rms_only) {
      acc1 += warp_buf1[idx1];
    }
    acc2 += warp_buf2[idx1];
  }
  if (!rms_only) {
    warp_buf1[threadIdx.y * row_stride + threadIdx.x] = acc1;
  }
  warp_buf2[threadIdx.y * row_stride + threadIdx.x] = acc2;
  __syncthreads();
  // sum all warps
  for (int offset = blockDim.y / 2; offset > 1; offset /= 2) {
    if (threadIdx.y < offset) {
      int row1 = threadIdx.y;
      int row2 = threadIdx.y + offset;
      int idx1 = row1 * row_stride + threadIdx.x;
      int idx2 = row2 * row_stride + threadIdx.x;
      if (!rms_only) {
        warp_buf1[idx1] += warp_buf1[idx2];
      }
      warp_buf2[idx1] += warp_buf2[idx2];
    }
    __syncthreads();
  }
  int i2 = blockIdx.x * blockDim.x + threadIdx.x;
  if (threadIdx.y == 0 && i2 < n2) {
    int row1 = threadIdx.y;
    int row2 = threadIdx.y + 1;
    int idx1 = row1 * row_stride + threadIdx.x;
    int idx2 = row2 * row_stride + threadIdx.x;
    if (!rms_only) {
      part_grad_beta[blockIdx.y * n2 + i2] = warp_buf1[idx1] + warp_buf1[idx2];
    }
    part_grad_gamma[blockIdx.y * n2 + i2] = warp_buf2[idx1] + warp_buf2[idx2];
  }
}

template <typename U, typename V>
__global__ void
cuComputeGradGammaBeta(const U *part_grad_gamma, const U *part_grad_beta,
                       const int part_size, const int n1, const int n2,
                       V *grad_gamma, V *grad_beta, bool rms_only) {
  // sum partial gradients for gamma and beta
  SharedMemory<U> shared;
  U *buf = shared.getPointer();
  int i2 = blockIdx.x * blockDim.x + threadIdx.x;
  if (i2 < n2) {
    // each warp does sequential reductions until reduced part_size is num_warps
    int num_warp_reductions = part_size / blockDim.y;
    U sum_gamma = U(0);
    U sum_beta = U(0);
    const U *part_grad_gamma_ptr =
        part_grad_gamma + threadIdx.y * num_warp_reductions * n2 + i2;
    const U *part_grad_beta_ptr =
        part_grad_beta + threadIdx.y * num_warp_reductions * n2 + i2;
    for (int warp_offset = 0; warp_offset < num_warp_reductions;
         ++warp_offset) {
      sum_gamma += part_grad_gamma_ptr[warp_offset * n2];
      if (!rms_only) {
        sum_beta += part_grad_beta_ptr[warp_offset * n2];
      }
    }
    // inter-warp reductions
    const int nbsize3 = blockDim.x * blockDim.y / 2;
    for (int offset = blockDim.y / 2; offset >= 1; offset /= 2) {
      // top half write to shared memory
      if (threadIdx.y >= offset && threadIdx.y < 2 * offset) {
        const int write_idx = (threadIdx.y - offset) * blockDim.x + threadIdx.x;
        buf[write_idx] = sum_gamma;
        if (!rms_only) {
          buf[write_idx + nbsize3] = sum_beta;
        }
      }
      __syncthreads();
      // bottom half sums
      if (threadIdx.y < offset) {
        const int read_idx = threadIdx.y * blockDim.x + threadIdx.x;
        sum_gamma += buf[read_idx];
        if (!rms_only) {
          sum_beta += buf[read_idx + nbsize3];
        }
      }
      __syncthreads();
    }
    // write out fully summed gradients
    if (threadIdx.y == 0) {
      grad_gamma[i2] = sum_gamma;
      if (!rms_only) {
        grad_beta[i2] = sum_beta;
      }
    }
  }
}

template <typename T, typename U, typename V>
__global__ void
cuComputeGradInput(const V *__restrict__ dout, const T *__restrict__ input,
                   const int n1, const int n2, const U *__restrict__ mean,
                   const U *__restrict__ invvar, U epsilon, const V *gamma,
                   T *grad_input, bool rms_only) {
  for (auto i1 = blockIdx.y; i1 < n1; i1 += gridDim.y) {
    U sum_loss1 = U(0);
    U sum_loss2 = U(0);
    U c_mean;
    if (!rms_only) {
      c_mean = mean[i1];
    }
    const U c_invvar = invvar[i1];
    const T *k_input = input + i1 * n2;
    const V *k_dout = dout + i1 * n2;
    const int numx = blockDim.x * blockDim.y;
    const int thrx = threadIdx.x + threadIdx.y * blockDim.x;
    if (gamma != NULL) {
      int l = 4 * thrx;
      for (; l + 3 < n2; l += 4 * numx) {
        for (int k = 0; k < 4; ++k) {
          const U c_h = static_cast<U>(k_input[l + k]);
          const U c_loss = static_cast<U>(k_dout[l + k]);
          if (!rms_only) {
            sum_loss1 += c_loss * static_cast<U>(gamma[l + k]);
            sum_loss2 += c_loss * static_cast<U>(gamma[l + k]) *
                         (c_h - c_mean) * c_invvar;
          } else {
            sum_loss2 += c_loss * static_cast<U>(gamma[l + k]) * (c_h)*c_invvar;
          }
        }
      }
      for (; l < n2; ++l) {
        const U c_h = static_cast<U>(k_input[l]);
        const U c_loss = static_cast<U>(k_dout[l]);
        if (!rms_only) {
          sum_loss1 += c_loss * static_cast<U>(gamma[l]);
          sum_loss2 +=
              c_loss * static_cast<U>(gamma[l]) * (c_h - c_mean) * c_invvar;
        } else {
          sum_loss2 += c_loss * static_cast<U>(gamma[l]) * (c_h)*c_invvar;
        }
      }
    } else {
      int l = 4 * thrx;
      for (; l + 3 < n2; l += 4 * numx) {
        for (int k = 0; k < 4; ++k) {
          const U c_h = static_cast<U>(k_input[l + k]);
          const U c_loss = static_cast<U>(k_dout[l + k]);
          if (!rms_only) {
            sum_loss1 += c_loss;
            sum_loss2 += c_loss * (c_h - c_mean) * c_invvar;
          } else {
            sum_loss2 += c_loss * (c_h)*c_invvar;
          }
        }
      }
      for (; l < n2; ++l) {
        const U c_h = static_cast<U>(k_input[l]);
        const U c_loss = static_cast<U>(k_dout[l]);
        if (!rms_only) {
          sum_loss1 += c_loss;
          sum_loss2 += c_loss * (c_h - c_mean) * c_invvar;
        } else {
          sum_loss2 += c_loss * (c_h)*c_invvar;
        }
      }
    }
    // intra-warp reductions
    for (int mask = blockDim.x / 2; mask > 0; mask /= 2) {
      if (!rms_only) {
        sum_loss1 += __shfl_xor_sync(0xffffffff, sum_loss1, mask, warpSize);
      }
      sum_loss2 += __shfl_xor_sync(0xffffffff, sum_loss2, mask, warpSize);
    }
    // inter-warp reductions
    if (blockDim.y > 1) {
      SharedMemory<U> shared;
      U *buf = shared.getPointer();
      for (int offset = blockDim.y / 2; offset > 0; offset /= 2) {
        // upper half of warps write to shared
        if (threadIdx.y >= offset && threadIdx.y < 2 * offset) {
          const int wrt_i = (threadIdx.y - offset) * blockDim.x + threadIdx.x;
          if (!rms_only) {
            buf[2 * wrt_i] = sum_loss1;
          }
          buf[2 * wrt_i + 1] = sum_loss2;
        }
        __syncthreads();
        // lower half merges
        if (threadIdx.y < offset) {
          const int read_i = threadIdx.y * blockDim.x + threadIdx.x;
          if (!rms_only) {
            sum_loss1 += buf[2 * read_i];
          }
          sum_loss2 += buf[2 * read_i + 1];
        }
        __syncthreads();
      }
      if (threadIdx.y == 0) {
        if (!rms_only) {
          buf[2 * threadIdx.x] = sum_loss1;
        }
        buf[2 * threadIdx.x + 1] = sum_loss2;
      }
      __syncthreads();
      if (threadIdx.y != 0) {
        if (!rms_only) {
          sum_loss1 = buf[2 * threadIdx.x];
        }
        sum_loss2 = buf[2 * threadIdx.x + 1];
      }
    }
    // all threads now have the two sums over l
    U fH = (U)n2;
    U term1 = (U(1) / fH) * c_invvar;
    T *k_grad_input = grad_input + i1 * n2;
    if (gamma != NULL) {
      for (int l = thrx; l < n2; l += numx) {
        const U c_h = static_cast<U>(k_input[l]);
        const U c_loss = static_cast<U>(k_dout[l]);
        U f_grad_input = fH * c_loss * static_cast<U>(gamma[l]);
        if (!rms_only) {
          f_grad_input -= sum_loss1;
          f_grad_input -= (c_h - c_mean) * c_invvar * sum_loss2;
        } else {
          f_grad_input -= (c_h)*c_invvar * sum_loss2;
        }
        f_grad_input *= term1;
        k_grad_input[l] = static_cast<T>(f_grad_input);
      }
    } else {
      for (int l = thrx; l < n2; l += numx) {
        const U c_h = static_cast<U>(k_input[l]);
        const U c_loss = static_cast<U>(k_dout[l]);
        U f_grad_input = fH * c_loss;
        if (!rms_only) {
          f_grad_input -= sum_loss1;
          f_grad_input -= (c_h - c_mean) * c_invvar * sum_loss2;
        } else {
          f_grad_input -= (c_h)*c_invvar * sum_loss2;
        }
        f_grad_input *= term1;
        k_grad_input[l] = static_cast<T>(f_grad_input);
      }
    }
    // prevent race where buf is written again before reads are done
    __syncthreads();
  }
}

template <typename T, typename U = float, typename V = T>
void HostRMSNormGradient(cudaStream_t stream, const V *dout, const U *invvar,
                         const T *input, int n1, int n2, const V *gamma,
                         double epsilon, T *grad_input, V *grad_gamma,
                         int part_size, U *part_grad_gamma) {
  auto getMaxGridY = []() {
    int device;
    int val;
    cudaGetDevice(&device);
    cudaDeviceGetAttribute(&val, cudaDevAttrMaxGridDimY, device);
    return val;
  };
  const uint64_t maxGridY = getMaxGridY();
  if (gamma != NULL) {
    const dim3 threads2(32, 4, 1);
    const dim3 blocks2((n2 + threads2.x - 1) / threads2.x, part_size, 1);
    const int nshared2_a =
        2 * sizeof(U) * threads2.y * threads2.y * (threads2.x + 1);
    const int nshared2_b = threads2.x * threads2.y * sizeof(U);
    const int nshared2 = nshared2_a > nshared2_b ? nshared2_a : nshared2_b;
    // note (mkozuki): I can hard code part_grad_gamma's dtype as float given
    // that the `cuda_layer_norm_gradient` doesn't support double.
    cuComputePartGradGammaBeta<<<blocks2, threads2, nshared2, stream>>>(
        dout, input, n1, n2,
        invvar,                                               // unused
        invvar, U(epsilon), part_grad_gamma, part_grad_gamma, /* unused */
        true);

    const dim3 threads3(32, 8, 1);
    const dim3 blocks3((n2 + threads2.x - 1) / threads2.x, 1, 1);
    const int nshared3 = threads3.x * threads3.y * sizeof(U);
    cuComputeGradGammaBeta<<<blocks3, threads3, nshared3, stream>>>(
        part_grad_gamma, part_grad_gamma,          /* unused */
        part_size, n1, n2, grad_gamma, grad_gamma, /* unused */
        true);
  }

  // compute grad_input
  const dim3 blocks1(1, std::min((uint64_t)n1, maxGridY), 1);
  const dim3 threads1(32, 4, 1);
  int nshared = threads1.y > 1 ? threads1.y * threads1.x * sizeof(U) : 0;
  cuComputeGradInput<<<blocks1, threads1, nshared, stream>>>(
      dout, input, n1, n2, invvar, /* unused */
      invvar, U(epsilon), gamma, grad_input, true);
}

} // namespace

namespace gpu_ops {

void rms_forward_affine_mixed_dtypes(cudaStream_t stream, void **buffers,
                                     const char *opaque,
                                     std::size_t opaque_len) {
  const RMSNormDescriptor &d =
      *UnpackDescriptor<RMSNormDescriptor>(opaque, opaque_len);

  DISPATCH_DOUBLE_FLOAT_HALF_AND_BFLOAT_INOUT_TYPES(
      d.x_type, d.w_type, "rms_norm_cuda_kernel",
      HostApplyRMSNorm<scalar_t_in, accscalar_t, scalar_t_out>(
          stream, static_cast<scalar_t_out *>(buffers[2]),
          static_cast<accscalar_t *>(buffers[3]),
          static_cast<scalar_t_in *>(buffers[0]), d.n1, d.n2, d.eps,
          /*gamma=*/static_cast<scalar_t_out *>(buffers[1]));)
}

void rms_backward_affine(cudaStream_t stream, void **buffers,
                         const char *opaque, std::size_t opaque_len) {
  const RMSNormDescriptor &d =
      *UnpackDescriptor<RMSNormDescriptor>(opaque, opaque_len);

  DISPATCH_DOUBLE_FLOAT_HALF_AND_BFLOAT_INOUT_TYPES(
      d.x_type, d.w_type, "cuComputeGradInputRMS",
      HostRMSNormGradient(
          stream,
          /*dout=*/static_cast<scalar_t_out *>(buffers[0]),
          /*invvar=*/static_cast<accscalar_t *>(buffers[1]),
          /*input=*/static_cast<scalar_t_in *>(buffers[2]), d.n1, d.n2,
          // TMJ pass NULL argument for gamma, beta, grad_gamma and grad_beta
          // if gamma Tensor is NULL on input.
          /*gamma=*/static_cast<scalar_t_out *>(buffers[3]), d.eps,
          /*grad_input=*/static_cast<scalar_t_in *>(buffers[4]),
          /*grad_gamma=*/static_cast<scalar_t_out *>(buffers[5]),
          d.part_grad_size,
          /*part_grad_gamma=*/static_cast<accscalar_t *>(buffers[6]));)
}

} // namespace gpu_ops
```

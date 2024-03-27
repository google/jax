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
* [Optional] Use [custom_partitioning](https://jax.readthedocs.io/en/latest/jax.experimental.custom_partitioning.html) or [shard_map](https://jax.readthedocs.io/en/latest/jep/14273-shard-map.html) functions for fast multi-GPU.


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

The values have been computed correctly for forward operation, however, the generated HLO modules show an `all-gather` operation to replicate `x` on all devices, incurring large communication overhead.

As XLA does not have enough knowledge about the custom functions to shard input tensors, it decides to replicate them to produce correct values before making the custom call.

To avoid this duplication, we can:
- [custom_partitioning](https://jax.readthedocs.io/en/latest/jax.experimental.custom_partitioning.html): to make it behave like all native JAX operations (but more complicated)
- Use manual sharding
  - [shard_map](https://jax.readthedocs.io/en/latest/jep/14273-shard-map.html): the new replacement for xmap
  - [xmap](https://jax.readthedocs.io/en/latest/notebooks/xmap_tutorial.html) (now deprecated)

This example demonstrates the use of custom_partitioning.

### Shard the forward function with custom_partitioning

We first create a helper function to help with all the JAX/XLA callback registration required.

```python
def register_primitive(cls):
    """
    register jax primitive

    The order of calls. Each operation is composed of two primitives: Inner and Outer.

    Inner, only the basic to wrap the custom_call itself.
    - impl to XLA custom_call in C.
    - abstract to know the static shapes
    - lower to StableHLO XLA custom_call.
    Outer, mostly all the rest:
    - impl: Bind to the inner primitive. Not used for real computation, but only for tracing. So we only need to bind.
    - abstract: same
    - lower to StableHLO custom_p. (XLA will call the python callback from it)
    - custom_p
    - vmap: could be added here.
    VJP is based on Outer, but not handled in this function.
    """

    def name_of_wrapper_p():
        return cls.name + "_wrapper"

    inner_p = core.Primitive(cls.name)
    dispatch.prim_requires_devices_during_lowering.add(inner_p)
    inner_p.multiple_results = cls.multiple_results
    inner_p.def_impl(partial(xla.apply_primitive, inner_p))
    inner_p.def_abstract_eval(cls.abstract)
    mlir.register_lowering(inner_p, cls.lowering, platform='cuda')
    cls.inner_primitive = inner_p

    outer_p = core.Primitive(name_of_wrapper_p())
    dispatch.prim_requires_devices_during_lowering.add(outer_p)
    outer_p.multiple_results = cls.multiple_results
    outer_p.def_impl(cls.impl)
    outer_p.def_abstract_eval(cls.abstract)
    batching.primitive_batchers[outer_p] = cls.batcher
    outer_p_lower = custom_partitioning(cls.impl, static_argnums=cls.impl_static_args)
    outer_p_lower.def_partition(infer_sharding_from_operands=cls.infer_sharding_from_operands,
                                partition=cls.partition)
    mlir.register_lowering(outer_p,
                           mlir.lower_fun(outer_p_lower, multiple_results=cls.multiple_results))
    cls.outer_primitive = outer_p
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
class RmsNormFwdClass:
    name = "rms_forward_affine_mixed_dtype"
    multiple_results = True
    impl_static_args = (2,)    # eps
    inner_primitive = None
    outer_primitive = None

    @staticmethod
    def infer_sharding_from_operands(eps : float, mesh : jax.sharding.Mesh,
                                     arg_infos : Tuple[jax._src.api.ShapeDtypeStruct],
                                     result_infos : Tuple[jax._src.core.ShapedArray]):
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

    @staticmethod
    def partition(eps : float, mesh : jax.sharding.Mesh,
                  arg_infos : Tuple[jax._src.api.ShapeDtypeStruct],
                  result_infos : Tuple[jax._src.api.ShapeDtypeStruct]):
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
Next we define the primitive for the backward pass of RMSNorm

### Shard the backward function with custom_partitioning

```python
class RmsNormBwdClass:
    name = "rms_norm_bwd"
    multiple_results = True
    impl_static_args = (4,)    # eps
    inner_primitive = None
    outer_primitive = None

    @staticmethod
    def infer_sharding_from_operands(eps : float, mesh : jax.sharding.Mesh,
                                     arg_infos : Tuple[jax._src.api.ShapeDtypeStruct],
                                     result_infos : Tuple[jax._src.core.ShapedArray]):
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
                  arg_infos : Tuple[jax._src.api.ShapeDtypeStruct],
                  result_infos : Tuple[jax._src.api.ShapeDtypeStruct]):
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
Plumbing to establish the forward and backward primtives with a custom_vjp rule as before:

```python
@partial(jax.custom_vjp, nondiff_argnums=(2,))
def custom_p_rms_norm(x, weight, eps=1e-05):
    output, _ = custom_p_rms_norm_fwd(x, weight, eps=eps)
    return output
  
def custom_p_rms_norm_fwd(x, weight, eps=1e-05):
    output, invvar = RmsNormFwdClass.outer_primitive.bind(x, weight, eps=eps)
    return output, (invvar, x, weight)

def custom_p_rms_norm_bwd(eps, res, g):
    invvar, x, weight = res
    grad_input, grad_weight, part_grad = RmsNormBwdClass.outer_primitive.bind(
        g, invvar, x, weight, eps=eps)
    return grad_input, grad_weight

custom_p_rms_norm.defvjp(custom_p_rms_norm_fwd, custom_p_rms_norm_bwd)
```

With that we have completely defined our custom RMS norm primitive with custom_partitioning. To check for correctness we define the following loss functions: ref_loss is the reference value to compare against, while custom_p_loss uses our new primitive that implements custom_partitioning.

```python
def ref_loss(x, weight):
    predictions = rms_norm(x, weight)
    return -jnp.mean(predictions**2)


ref = jax.grad(ref_loss, argnums=(0, 1))(x, weight)

def custom_p_loss(x, weight):
    predictions = custom_p_rms_norm(x, weight)
    return -jnp.mean(predictions**2)
```

# Check for correctness

```python
with Mesh(jax.local_devices(), ("x",)):
    def run_and_verify(loss):
        pjitted = pjit(
            jax.grad(loss, argnums=(0, 1)),
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
        hlo = pjitted.lower(x, weight).compile().as_text()
        out = pjitted(x, weight)
        print(hlo)
        assert "all-reduce-done" in hlo, "The gradient will produce wrong value!"
        if "all-gather-start" in hlo:
            print("NOT OPTIMIZED, ALL_GATHER in the graph!")
        return out

    custom_p_out = run_and_verify(custom_p_loss)


for r, o in zip(ref_out, custom_p_out):
    print(jnp.allclose(r, o, atol=1e-6, rtol=1e-6))
```
```python
HloModule pjit_custom_p_loss, is_scheduled=true, entry_computation_layout={(f16[4,512,512]{2,1,0}, f16[512,512]{1,0})->(f16[4,512,512]{2,1,0}, f16[512,512]{1,0})}, allow_spmd_sharding_propagation_to_parameters={false,false}, allow_spmd_sharding_propagation_to_output={false,false}, num_partitions=4, frontend_attributes={fingerprint_before_lhs="d7b9bc40de002332dd665ff2ab537b76"}

%fused_multiply (param_0: f16[4,512,512]) -> f16[4,512,512] {
  %param_0 = f16[4,512,512]{2,1,0} parameter(0)
  %constant_4_1 = f16[] constant(-4.7684e-07)
  %broadcast.8.1 = f16[4,512,512]{2,1,0} broadcast(f16[] %constant_4_1), dimensions={}, metadata={op_name="pjit(custom_p_loss)/jit(main)/mul" source_file="/opt/jax/docs/Custom_Operation_for_GPUs.py" source_line=484}
  ROOT %multiply.5.1 = f16[4,512,512]{2,1,0} multiply(f16[4,512,512]{2,1,0} %param_0, f16[4,512,512]{2,1,0} %broadcast.8.1), metadata={op_name="pjit(custom_p_loss)/jit(main)/mul" source_file="/opt/jax/docs/Custom_Operation_for_GPUs.py" source_line=484}
}

%region_0.9._custom_call_lowering_rule (Arg_0.10.0: f16[], Arg_1.11.0: f16[]) -> f16[] {
  %Arg_1.11.0 = f16[] parameter(1)
  %Arg_0.10.0 = f16[] parameter(0)
  ROOT %add.2.0 = f16[] add(f16[] %Arg_0.10.0, f16[] %Arg_1.11.0), metadata={op_name="jit(main)/add" source_file="/opt/jax/docs/Custom_Operation_for_GPUs.py" source_line=433}
}

ENTRY %main.23_spmd (param.2: f16[4,512,512], param.1.0: f16[512,512]) -> (f16[4,512,512], f16[512,512]) {
  %param.1.0 = f16[512,512]{1,0} parameter(1), sharding={replicated}
  %param.2 = f16[4,512,512]{2,1,0} parameter(0), sharding={devices=[4,1,1]<=[4]}
  %custom-call.3.0 = (f16[4,512,512]{2,1,0}, f32[4]{0}) custom-call(f16[4,512,512]{2,1,0} %param.2, f16[512,512]{1,0} %param.1.0), custom_call_target="rms_forward_affine_mixed_dtype", operand_layout_constraints={f16[4,512,512]{2,1,0}, f16[512,512]{1,0}}, api_version=API_VERSION_STATUS_RETURNING, metadata={op_name="pjit(custom_p_loss)/jit(main)/custom_partitioning[partition=<function RmsNormFwdClass.partition at 0x7ff99e3980d0> propagate_user_sharding=None infer_sharding_from_operands=<function RmsNormFwdClass.infer_sharding_from_operands at 0x7ff99e398040> decode_shardings=True in_tree=PyTreeDef((*, *)) out_tree=PyTreeDef((*, *)) static_args=[1e-05]]" source_file="/opt/jax/docs/Custom_Operation_for_GPUs.py" source_line=440}, backend_config="\004\000\000\000\000\000\004\000\361h\343\210\265\370\344>\001\000\000\000\001\000\000\000\000\000\000\000$V\000\000"
  %get-tuple-element.14 = f16[4,512,512]{2,1,0} get-tuple-element((f16[4,512,512]{2,1,0}, f32[4]{0}) %custom-call.3.0), index=0, metadata={op_name="pjit(custom_p_loss)/jit(main)/custom_partitioning[partition=<function RmsNormFwdClass.partition at 0x7ff99e3980d0> propagate_user_sharding=None infer_sharding_from_operands=<function RmsNormFwdClass.infer_sharding_from_operands at 0x7ff99e398040> decode_shardings=True in_tree=PyTreeDef((*, *)) out_tree=PyTreeDef((*, *)) static_args=[1e-05]]" source_file="/opt/jax/docs/Custom_Operation_for_GPUs.py" source_line=440}
  %loop_multiply_fusion = f16[4,512,512]{2,1,0} fusion(f16[4,512,512]{2,1,0} %get-tuple-element.14), kind=kLoop, calls=%fused_multiply, metadata={op_name="pjit(custom_p_loss)/jit(main)/mul" source_file="/opt/jax/docs/Custom_Operation_for_GPUs.py" source_line=484}
  %get-tuple-element.1.0 = f32[4]{0} get-tuple-element((f16[4,512,512]{2,1,0}, f32[4]{0}) %custom-call.3.0), index=1, metadata={op_name="pjit(custom_p_loss)/jit(main)/custom_partitioning[partition=<function RmsNormFwdClass.partition at 0x7ff99e3980d0> propagate_user_sharding=None infer_sharding_from_operands=<function RmsNormFwdClass.infer_sharding_from_operands at 0x7ff99e398040> decode_shardings=True in_tree=PyTreeDef((*, *)) out_tree=PyTreeDef((*, *)) static_args=[1e-05]]" source_file="/opt/jax/docs/Custom_Operation_for_GPUs.py" source_line=440}
  %custom-call.5.0 = (f16[4,512,512]{2,1,0}, f16[512,512]{1,0}, f32[16,262144]{1,0}) custom-call(f16[4,512,512]{2,1,0} %loop_multiply_fusion, f32[4]{0} %get-tuple-element.1.0, f16[4,512,512]{2,1,0} %param.2, f16[512,512]{1,0} %param.1.0), custom_call_target="rms_backward_affine", operand_layout_constraints={f16[4,512,512]{2,1,0}, f32[4]{0}, f16[4,512,512]{2,1,0}, f16[512,512]{1,0}}, api_version=API_VERSION_STATUS_RETURNING, metadata={op_name="pjit(custom_p_loss)/jit(main)/custom_partitioning[partition=<function RmsNormBwdClass.partition at 0x7ff99e3985e0> propagate_user_sharding=None infer_sharding_from_operands=<function RmsNormBwdClass.infer_sharding_from_operands at 0x7ff99e398550> decode_shardings=True in_tree=PyTreeDef((*, *, *, *)) out_tree=PyTreeDef((*, *, *)) static_args=[1e-05]]" source_file="/opt/jax/docs/Custom_Operation_for_GPUs.py" source_line=483}, backend_config="\004\000\000\000\000\000\004\000\361h\343\210\265\370\344>\001\000\000\000\001\000\000\000\020\000\000\000$V\000\000"
  %get-tuple-element.7.0 = f16[512,512]{1,0} get-tuple-element((f16[4,512,512]{2,1,0}, f16[512,512]{1,0}, f32[16,262144]{1,0}) %custom-call.5.0), index=1, metadata={op_name="pjit(custom_p_loss)/jit(main)/custom_partitioning[partition=<function RmsNormBwdClass.partition at 0x7ff99e3985e0> propagate_user_sharding=None infer_sharding_from_operands=<function RmsNormBwdClass.infer_sharding_from_operands at 0x7ff99e398550> decode_shardings=True in_tree=PyTreeDef((*, *, *, *)) out_tree=PyTreeDef((*, *, *)) static_args=[1e-05]]" source_file="/opt/jax/docs/Custom_Operation_for_GPUs.py" source_line=483}
  %all-reduce-start = f16[512,512]{1,0} all-reduce-start(f16[512,512]{1,0} %get-tuple-element.7.0), channel_id=1, replica_groups={{0,1,2,3}}, use_global_device_ids=true, to_apply=%region_0.9._custom_call_lowering_rule, metadata={op_name="pjit(custom_p_loss)/jit(main)/custom_partitioning[partition=<function RmsNormBwdClass.partition at 0x7ff99e3985e0> propagate_user_sharding=None infer_sharding_from_operands=<function RmsNormBwdClass.infer_sharding_from_operands at 0x7ff99e398550> decode_shardings=True in_tree=PyTreeDef((*, *, *, *)) out_tree=PyTreeDef((*, *, *)) static_args=[1e-05]]" source_file="/opt/jax/docs/Custom_Operation_for_GPUs.py" source_line=483}, backend_config={"operation_queue_id":"0","wait_on_operation_queues":[],"collective_backend_config":{"is_sync":true,"no_parallel_custom_call":false}}
  %all-reduce-done = f16[512,512]{1,0} all-reduce-done(f16[512,512]{1,0} %all-reduce-start), metadata={op_name="pjit(custom_p_loss)/jit(main)/custom_partitioning[partition=<function RmsNormBwdClass.partition at 0x7ff99e3985e0> propagate_user_sharding=None infer_sharding_from_operands=<function RmsNormBwdClass.infer_sharding_from_operands at 0x7ff99e398550> decode_shardings=True in_tree=PyTreeDef((*, *, *, *)) out_tree=PyTreeDef((*, *, *)) static_args=[1e-05]]" source_file="/opt/jax/docs/Custom_Operation_for_GPUs.py" source_line=483}
  %get-tuple-element.12.0 = f16[4,512,512]{2,1,0} get-tuple-element((f16[4,512,512]{2,1,0}, f16[512,512]{1,0}, f32[16,262144]{1,0}) %custom-call.5.0), index=0, metadata={op_name="pjit(custom_p_loss)/jit(main)/custom_partitioning[partition=<function RmsNormBwdClass.partition at 0x7ff99e3985e0> propagate_user_sharding=None infer_sharding_from_operands=<function RmsNormBwdClass.infer_sharding_from_operands at 0x7ff99e398550> decode_shardings=True in_tree=PyTreeDef((*, *, *, *)) out_tree=PyTreeDef((*, *, *)) static_args=[1e-05]]" source_file="/opt/jax/docs/Custom_Operation_for_GPUs.py" source_line=483}
  ROOT %tuple.1.0 = (f16[4,512,512]{2,1,0}, f16[512,512]{1,0}) tuple(f16[4,512,512]{2,1,0} %get-tuple-element.12.0, f16[512,512]{1,0} %all-reduce-done)
}
```
```python
True
True
```

Now there are no all-gathers in the HLO, sharding is respected and only gradients are accumulated via an all-reduce.


## Let's put it together

The complete definition of the primitives using custom_partitioning can be found in [Custom_Operation_for_GPUs.py](Custom_Operation_for_GPUs.py) and the corresponding C++ code the defines python bindings in addition to the kernel implementations can be found below:

### `gpu_ops` code listing

[gpu_ops/kernel_helpers.h](gpu_ops/kernel_helpers.h) \
[gpu_ops/kernels.h](gpu_ops/kernels.h) \
[gpu_ops/pybind11_kernel_helpers.h](gpu_ops/pybind11_kernel_helpers.h) \
[gpu_ops/gpu_ops.cpp](gpu_ops/gpu_ops.cpp) \
[gpu_ops/rms_norm_kernels.cu](gpu_ops/rms_norm_kernels.cu) 

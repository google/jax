import jax
import jax.numpy as jnp
import numpy as np
from jax.lib import xla_client
from jax._src.lib.mlir import ir
from jaxlib.hlo_helpers import custom_call
from jax.interpreters import mlir

import mul

# XLA needs uppercase, "cuda" isn't recognized
XLA_PLATFORM = "CUDA"

# JAX needs lowercase, "CUDA" isn't recognized
JAX_PLATFORM = "cuda"

# 0 = original ("opaque"), 1 = FFI
XLA_CUSTOM_CALL_API_VERSION = 1

# version 4 accepts MLIR dictionaries in backend_config
STABLEHLO_CUSTOM_CALL_API_VERSION = 4

# register the XLA FFI binding pointer with XLA under the name "mul"
xla_client.register_custom_call_target(
    name="mul",
    fn=mul.registrations["mul"],
    platform=XLA_PLATFORM,
    api_version=XLA_CUSTOM_CALL_API_VERSION
)

mul_p = jax.core.Primitive("mul")
def mul(a, b):
    return mul_p.bind(a, b)

def _mul_abstract_eval(a, b):
    assert a.shape == b.shape
    assert a.dtype == b.dtype
    return jax.core.ShapedArray(a.shape, a.dtype)

mul_p.def_abstract_eval(_mul_abstract_eval)

def _mul_lowering(ctx, a, b):
    typ = ir.RankedTensorType(a.type)
    shape = typ.shape
    u64 = ir.IntegerType.get_unsigned(64)
    n = np.prod(shape)
    # default XLA layout that just means row major
    layout = range(len(shape)-1, -1, -1)
    return custom_call(
        "mul",
        api_version=STABLEHLO_CUSTOM_CALL_API_VERSION,
        result_types=[typ],  # same type as inputs
        operands=[a,b],
        backend_config=dict(n=ir.IntegerAttr.get(u64, n)),
        operand_layouts=[layout, layout],
        result_layouts=[layout],
    ).results

mlir.register_lowering(mul_p, _mul_lowering, platform=JAX_PLATFORM)

a = 2. * jnp.ones((2,3))
b = 3. * jnp.ones((2,3))
assert (jax.jit(mul)(a, b) == (6)).all()

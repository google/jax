# Copyright 2021 Google LLC
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
# pylint: disable=unused-import
from absl import logging
import os

from tensorflow.python.framework import versions
from tensorflow.python.dlpack.dlpack import from_dlpack as experimental_dlpack_from_dlpack
from tensorflow.python.dlpack.dlpack import to_dlpack as experimental_dlpack_to_dlpack
from tensorflow.python.eager.backprop import GradientTape
from tensorflow.python.eager.context import device
from tensorflow.python.eager.def_function import function
from tensorflow.python.framework.constant_op import constant
from tensorflow.python.framework.device_spec import DeviceSpecV2  # MODIFIED MANUALLY
from tensorflow.python.framework.dtypes import as_dtype
from tensorflow.python.framework.dtypes import bfloat16
from tensorflow.python.framework.dtypes import bool
from tensorflow.python.framework.dtypes import complex128
from tensorflow.python.framework.dtypes import complex64
from tensorflow.python.framework.dtypes import float16
from tensorflow.python.framework.dtypes import float32
from tensorflow.python.framework.dtypes import float64
from tensorflow.python.framework.dtypes import int16
from tensorflow.python.framework.dtypes import int32
from tensorflow.python.framework.dtypes import int64
from tensorflow.python.framework.dtypes import int8
from tensorflow.python.framework.dtypes import resource
from tensorflow.python.framework.dtypes import uint16
from tensorflow.python.framework.dtypes import uint32
from tensorflow.python.framework.dtypes import uint64
from tensorflow.python.framework.dtypes import uint8
from tensorflow.python.framework.ops import Tensor
from tensorflow.python.framework.ops import convert_to_tensor
from tensorflow.python.framework.ops import enable_eager_execution
from tensorflow.python.framework.ops import name_scope
from tensorflow.python.framework.tensor_shape import Dimension as compat_v1_Dimension
from tensorflow.python.framework.tensor_shape import TensorShape
from tensorflow.python.framework.tensor_spec import TensorSpec
from tensorflow.python.module.module import Module
from tensorflow.python.ops.linalg.linalg_impl import einsum
from tensorflow.python.ops.linalg_ops import eig
from tensorflow.python.ops.linalg_ops import eigvals
from tensorflow.python.ops.array_ops import bitcast
from tensorflow.python.ops.array_ops import broadcast_to
from tensorflow.python.ops.array_ops import concat
from tensorflow.python.ops.array_ops import expand_dims
from tensorflow.python.ops.array_ops import gather
from tensorflow.python.ops.array_ops import gather_nd
from tensorflow.python.ops.array_ops import identity
from tensorflow.python.ops.array_ops import invert_permutation
from tensorflow.python.ops.array_ops import ones
from tensorflow.python.ops.array_ops import ones_like
from tensorflow.python.ops.array_ops import pad
from tensorflow.python.ops.array_ops import prevent_gradient as raw_ops_PreventGradient  # MODIFIED MANUALLY
from tensorflow.python.ops.array_ops import rank
from tensorflow.python.ops.array_ops import reshape
from tensorflow.python.ops.array_ops import reverse
from tensorflow.python.ops.array_ops import scatter_nd
from tensorflow.python.ops.array_ops import shape
from tensorflow.python.ops.array_ops import size
from tensorflow.python.ops.array_ops import slice
from tensorflow.python.ops.array_ops import squeeze
from tensorflow.python.ops.array_ops import stack
from tensorflow.python.ops.array_ops import stop_gradient
from tensorflow.python.ops.array_ops import strided_slice
from tensorflow.python.ops.array_ops import tensor_scatter_nd_update
from tensorflow.python.ops.array_ops import transpose
from tensorflow.python.ops.array_ops import where_v2 as where
from tensorflow.python.ops.array_ops import zeros
from tensorflow.python.ops.array_ops import zeros_like
from tensorflow.python.ops.bitwise_ops import bitwise_and
from tensorflow.python.ops.bitwise_ops import bitwise_or
from tensorflow.python.ops.bitwise_ops import bitwise_xor
from tensorflow.python.ops.bitwise_ops import invert as bitwise_invert
from tensorflow.python.ops.bitwise_ops import left_shift as bitwise_left_shift
from tensorflow.python.ops.bitwise_ops import right_shift as bitwise_right_shift
from tensorflow.python.ops.bitwise_ops import population_count as raw_ops_PopulationCount  # ADDED MANUALLY
from tensorflow.python.ops.clip_ops import clip_by_value
from tensorflow.python.ops.control_flow_ops import switch_case
from tensorflow.python.ops.control_flow_ops import while_loop_v2 as while_loop
from tensorflow.python.ops.custom_gradient import custom_gradient
from tensorflow.python.ops.linalg.linalg_impl import adjoint as linalg_adjoint
from tensorflow.python.ops.linalg.linalg_impl import cholesky as linalg_cholesky
from tensorflow.python.ops.linalg.linalg_impl import eigh as linalg_eigh
from tensorflow.python.ops.linalg.linalg_impl import qr as linalg_qr
from tensorflow.python.ops.linalg.linalg_impl import set_diag as linalg_set_diag
from tensorflow.python.ops.linalg.linalg_impl import svd as linalg_svd
from tensorflow.python.ops.linalg.linalg_impl import triangular_solve as linalg_triangular_solve
from tensorflow.python.ops.map_fn import map_fn
from tensorflow.python.ops.math_ops import abs as math_abs
from tensorflow.python.ops.math_ops import acosh as math_acosh
from tensorflow.python.ops.math_ops import add
from tensorflow.python.ops.math_ops import add_v2 as raw_ops_AddV2  # ADDED MANUALLY
from tensorflow.python.ops.math_ops import argmax as math_argmax
from tensorflow.python.ops.math_ops import argmin as math_argmin
from tensorflow.python.ops.math_ops import asinh as math_asinh
from tensorflow.python.ops.math_ops import atan2 as math_atan2
from tensorflow.python.ops.math_ops import atanh as math_atanh
from tensorflow.python.ops.math_ops import betainc as math_betainc
from tensorflow.python.ops.math_ops import cast
from tensorflow.python.ops.math_ops import ceil as math_ceil
from tensorflow.python.ops.math_ops import complex
from tensorflow.python.ops.math_ops import conj as math_conj
from tensorflow.python.ops.math_ops import cos as math_cos
from tensorflow.python.ops.math_ops import cosh as math_cosh
from tensorflow.python.ops.math_ops import digamma as math_digamma
from tensorflow.python.ops.math_ops import equal
from tensorflow.python.ops.math_ops import erf as math_erf
from tensorflow.python.ops.math_ops import erfc as math_erfc
from tensorflow.python.ops.math_ops import erfinv as math_erfinv
from tensorflow.python.ops.math_ops import exp as math_exp
from tensorflow.python.ops.math_ops import expm1 as math_expm1
from tensorflow.python.ops.math_ops import floor as math_floor
from tensorflow.python.ops.math_ops import floordiv as math_floordiv
from tensorflow.python.ops.math_ops import floormod as math_floormod
from tensorflow.python.ops.math_ops import greater
from tensorflow.python.ops.math_ops import greater_equal
from tensorflow.python.ops.math_ops import igamma as math_igamma
from tensorflow.python.ops.math_ops import igammac as math_igammac
from tensorflow.python.ops.math_ops import imag as math_imag
from tensorflow.python.ops.math_ops import is_finite as math_is_finite
from tensorflow.python.ops.math_ops import less
from tensorflow.python.ops.math_ops import less_equal
from tensorflow.python.ops.math_ops import lgamma as math_lgamma
from tensorflow.python.ops.math_ops import log as math_log
from tensorflow.python.ops.math_ops import log1p as math_log1p
from tensorflow.python.ops.math_ops import logical_and
from tensorflow.python.ops.math_ops import logical_not
from tensorflow.python.ops.math_ops import logical_or
from tensorflow.python.ops.math_ops import logical_xor as math_logical_xor
from tensorflow.python.ops.math_ops import matmul
from tensorflow.python.ops.math_ops import maximum
from tensorflow.python.ops.math_ops import minimum
from tensorflow.python.ops.math_ops import multiply
from tensorflow.python.ops.math_ops import negative as math_negative
from tensorflow.python.ops.math_ops import nextafter as math_nextafter
from tensorflow.python.ops.math_ops import not_equal
from tensorflow.python.ops.math_ops import pow as math_pow
from tensorflow.python.ops.math_ops import range
from tensorflow.python.ops.math_ops import real as math_real
from tensorflow.python.ops.math_ops import reciprocal as math_reciprocal
from tensorflow.python.ops.math_ops import reduce_all
from tensorflow.python.ops.math_ops import reduce_any
from tensorflow.python.ops.math_ops import reduce_max
from tensorflow.python.ops.math_ops import reduce_min
from tensorflow.python.ops.math_ops import reduce_prod
from tensorflow.python.ops.math_ops import reduce_sum
from tensorflow.python.ops.math_ops import round as math_round
from tensorflow.python.ops.math_ops import rsqrt as math_rsqrt
from tensorflow.python.ops.math_ops import sign as math_sign
from tensorflow.python.ops.math_ops import sin as math_sin
from tensorflow.python.ops.math_ops import sinh as math_sinh
from tensorflow.python.ops.math_ops import sqrt as math_sqrt
from tensorflow.python.ops.math_ops import subtract
from tensorflow.python.ops.math_ops import tan as math_tan
from tensorflow.python.ops.math_ops import tanh as math_tanh
from tensorflow.python.ops.math_ops import truediv as math_truediv
from tensorflow.python.ops.math_ops import unsorted_segment_max as math_unsorted_segment_max
from tensorflow.python.ops.math_ops import unsorted_segment_min as math_unsorted_segment_min
from tensorflow.python.ops.math_ops import unsorted_segment_prod as math_unsorted_segment_prod
from tensorflow.python.ops.math_ops import unsorted_segment_sum as math_unsorted_segment_sum
from tensorflow.python.ops.nn_impl import depthwise_conv2d as nn_depthwise_conv2d
from tensorflow.python.ops.nn_ops import approx_max_k as math_approx_max_k
from tensorflow.python.ops.nn_ops import approx_min_k as math_approx_min_k
from tensorflow.python.ops.nn_ops import conv2d as nn_conv2d
from tensorflow.python.ops.nn_ops import conv2d_transpose as nn_conv2d_transpose
from tensorflow.python.ops.nn_ops import pool as nn_pool
from tensorflow.python.ops.nn_ops import top_k as math_top_k
from tensorflow.python.ops.random_ops import random_uniform
from tensorflow.python.ops.signal.fft_ops import fft as signal_fft
from tensorflow.python.ops.signal.fft_ops import fft2d as signal_fft2d
from tensorflow.python.ops.signal.fft_ops import fft3d as signal_fft3d
from tensorflow.python.ops.signal.fft_ops import ifft as signal_ifft
from tensorflow.python.ops.signal.fft_ops import ifft2d as signal_ifft2d
from tensorflow.python.ops.signal.fft_ops import ifft3d as signal_ifft3d
from tensorflow.python.ops.signal.fft_ops import irfft as signal_irfft
from tensorflow.python.ops.signal.fft_ops import irfft2d as signal_irfft2d
from tensorflow.python.ops.signal.fft_ops import irfft3d as signal_irfft3d
from tensorflow.python.ops.signal.fft_ops import rfft as signal_rfft
from tensorflow.python.ops.signal.fft_ops import rfft2d as signal_rfft2d
from tensorflow.python.ops.signal.fft_ops import rfft3d as signal_rfft3d
from tensorflow.python.ops.special_math_ops import bessel_i0e as math_bessel_i0e
from tensorflow.python.ops.special_math_ops import bessel_i1e as math_bessel_i1e
from tensorflow.python.ops.stateless_random_ops import Algorithm
from tensorflow.python.ops.variables import Variable
from tensorflow.python.ops.unconnected_gradients import UnconnectedGradients
from tensorflow.python.util.nest import assert_same_structure as nest_assert_same_structure
from tensorflow.python.util.nest import flatten as nest_flatten
from tensorflow.python.util.nest import map_structure as nest_map_structure

from pathlib import Path

# Aliases used to keep imports in the source files simple, so we can for
# instance do "tf.DeviceSpec_from_string".
DeviceSpec_from_string = DeviceSpecV2.from_string

random_Algorithm_AUTO_SELECT = Algorithm.AUTO_SELECT
random_Algorithm_PHILOX = Algorithm.PHILOX
random_Algorithm_THREEFRY = Algorithm.THREEFRY
UnconnectedGradients_ZERO = UnconnectedGradients.ZERO

__version__ = versions.__version__

# When using "import tensorflow as tf":
# dirname(tf.__file__) = //third_party/py/tensorflow
#
# We get the same output using the "versions" import as follows:
# dir(versions.__file__) = /third_party/tensorflow/python/framework
# Path(versions.__file__).parents[1] = /third_party/tensorflow/python/
#
# //third_party/py/tensorflow points to /third_party/tensorflow/python/
__file__ = str(Path(versions.__file__).parents[1])


# Simulate tf.__init__

# pylint: disable=g-bad-import-order,g-import-not-at-top,g-direct-tensorflow-import
try:
  ;   # Marker for internal import
except ImportError:
  os.environ["TF2_BEHAVIOR"] = "1"

from tensorflow.python import tf2
try:
  ;   # Marker for internal import
  tf2.disable()
  logging.warning("TF2 behavior is disabled by default because there is "
                   "a BUILD dependency on //learning/brain/public:disable_tf2. "
                   "Remove it if the intention is to run with V2 behavior.")
except ImportError:
  tf2.enable()

from tensorflow.python.compat import v2_compat as _compat
# Enable TF2 behaviors
if tf2.enabled():
  _compat.enable_v2_behavior()
else:
  _compat.disable_v2_behavior()

# Copyright 2024 The JAX Authors.
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

import datetime
from numpy import array, int32

# Pasted from the test output (see export_back_compat_test_util.py module docstring)
data_2024_08_08 = dict(
    testdata_version=1,
    platform='cuda',
    custom_call_targets=['cu_lu_pivots_to_permutation'],
    serialized_date=datetime.date(2024, 8, 8),
    inputs=(),
    expected_outputs=(array([[[0, 1, 2, 3, 4, 5, 6, 7],
        [4, 5, 6, 7, 0, 1, 2, 3],
        [0, 1, 2, 3, 4, 5, 6, 7]],

       [[0, 1, 2, 3, 4, 5, 6, 7],
        [0, 1, 2, 3, 4, 5, 6, 7],
        [0, 1, 2, 3, 4, 5, 6, 7]]], dtype=int32),),
    mlir_module_text=r"""
module @jit__lambda_ attributes {jax.uses_shape_polymorphism = false, mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} {
  func.func public @main() -> (tensor<2x3x8xi32> {jax.result_info = "", mhlo.layout_mode = "default"}) {
    %0 = stablehlo.iota dim = 0 : tensor<24xi32> loc(#loc4)
    %1 = stablehlo.reshape %0 : (tensor<24xi32>) -> tensor<2x3x4xi32> loc(#loc5)
    %c = stablehlo.constant dense<2> : tensor<i64> loc(#loc6)
    %c_0 = stablehlo.constant dense<3> : tensor<i64> loc(#loc6)
    %c_1 = stablehlo.constant dense<4> : tensor<i64> loc(#loc6)
    %2 = stablehlo.custom_call @cu_lu_pivots_to_permutation(%1) {mhlo.backend_config = {permutation_size = 8 : i32}, operand_layouts = [dense<[2, 1, 0]> : tensor<3xindex>], result_layouts = [dense<[2, 1, 0]> : tensor<3xindex>]} : (tensor<2x3x4xi32>) -> tensor<2x3x8xi32> loc(#loc6)
    return %2 : tensor<2x3x8xi32> loc(#loc)
  } loc(#loc)
} loc(#loc)
#loc = loc(unknown)
#loc1 = loc("third_party/py/jax/tests/export_back_compat_test.py":347:26)
#loc2 = loc("third_party/py/jax/tests/export_back_compat_test.py":347:14)
#loc3 = loc("third_party/py/jax/tests/export_back_compat_test.py":348:11)
#loc4 = loc("jit(<lambda>)/jit(main)/iota[dtype=int32 shape=(24,) dimension=0]"(#loc1))
#loc5 = loc("jit(<lambda>)/jit(main)/reshape[new_sizes=(2, 3, 4) dimensions=None]"(#loc2))
#loc6 = loc("jit(<lambda>)/jit(main)/lu_pivots_to_permutation[permutation_size=8]"(#loc3))
""",
    mlir_module_serialized=b"ML\xefR\x01StableHLO_v0.9.0\x00\x01\x1d\x05\x01\x03\x01\x03\x05\x03\r\x07\t\x0b\r\x0f\x11\x03\xa7}\x17\x01Q\x0f\x07\x0b\x0b\x0f\x0b+\x0b\x0f\x0b\x0b\x0b3\x0b\x0b\x0b\x0b\x13\x0b\x0f\x0b\x17\x0f\x0b\x17\x13\x0b\x17\x13\x13S\x0b\x0b\x0b\x0b\x0b\x0b\x0b\x0b\x0b\x03-\x0b\x0b\x0f\x0b\x0f\x1b\x0b\x0b\x0b\x0b\x0b\x0f///\x0b\x0b\x0b\x13\x0b\x0fo\x01\x05\x0b\x0f\x03\x13\x0f\x07\x1b\x07\x13\x13\x1b\x13\x07\x02Z\x04\x1d57\x1f\x05\x13\x05\x15\x11\x03\x05\x05\x17\x03\t\x0f\x11\x13\t\x15\t\x0b\x17\x05\x19\x11\x01\x00\x05\x1b\x05\x1d\x05\x1f\x03\x0b\x1bQ\x1dW\x1fY\x0bc!e\x05!\x05#\x05%\x05'\x03\x03%g\x05)\x1d)+\x05+\x17\x05n\x055\x1d/1\x05-\x17\x05n\x05\x1d\x03\x03\x07i\x05/\x17\x05r\x05\x17\x03\x03\x07k\x03\x03\x07m\x03\x13?oASCqEQGsIuKUMQOU\x051\x053\x055\x057\x059\x05;\x05=\x05?\x05A\x03\x01\x1dC\x03\x03{#\r\x03\x03[\r\x05]S_a\x1dE\x1dG\x1dI\x1dK\x1dM\x13\x0b\x01\x1f\x05\x11\x02\x00\x00\x00\x00\x00\x00\x00\x1f\x05\x11\x03\x00\x00\x00\x00\x00\x00\x00\x1f\x05\x11\x04\x00\x00\x00\x00\x00\x00\x00\x0b\x03\x1dO\x05\x01\r\x03wy\x1dQ\x13\x07!\x1f\x131\x02\x00\x00\x00\x00\x00\x00\x00\x01\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x01\t\x01\x02\x02)\x01\x0b\x1b)\x07\t\r!\x07\x1d\x11\x01\x03\t)\x03a\x07)\x07\t\r\x11\x07)\x03\r\x15\x13\x04{\x05\x01\x11\x03\r\x07\x03\x01\x05\x05\x11\x03\x19\x07\x03\r\x1d\x07\x03'#\x03\x0f\t\x06-\x03\x11\x03\x01\x03\x03\x013\x03\x05\x03\x03\x019\x03\x05\x03\x03\x01;\x03\x05\x0b\x07\x01=\x03\t\x03\x03\r\x04\x03\x03\x0b\x06\x03\x01\x05\x01\x00f\x0cS#9\x0f\x0b\x11#!\x03\x1f/!)!)#\x1f\x19\x8b\x8b\x85\x1f\x1f\x15\x1d\x15\x1b%)9\x13\ri\x15\x1f\x17\x11\x11\x19\x0f\x0b\x11builtin\x00vhlo\x00module\x00constant_v1\x00func_v1\x00iota_v1\x00reshape_v1\x00custom_call_v1\x00return_v1\x00third_party/py/jax/tests/export_back_compat_test.py\x00value\x00sym_name\x00jax.uses_shape_polymorphism\x00mhlo.num_partitions\x00mhlo.num_replicas\x00jit__lambda_\x00arg_attrs\x00function_type\x00res_attrs\x00sym_visibility\x00iota_dimension\x00jit(<lambda>)/jit(main)/iota[dtype=int32 shape=(24,) dimension=0]\x00jit(<lambda>)/jit(main)/reshape[new_sizes=(2, 3, 4) dimensions=None]\x00jit(<lambda>)/jit(main)/lu_pivots_to_permutation[permutation_size=8]\x00api_version\x00backend_config\x00call_target_name\x00called_computations\x00has_side_effect\x00mhlo.backend_config\x00operand_layouts\x00output_operand_aliases\x00result_layouts\x00\x00jax.result_info\x00mhlo.layout_mode\x00default\x00main\x00public\x00cu_lu_pivots_to_permutation\x00permutation_size\x00",
    xla_call_module_version=9,
    nr_devices=1,
)  # End paste

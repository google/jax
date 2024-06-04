# Copyright 2023 The JAX Authors.
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

# ruff: noqa

import datetime
from numpy import array, float32, complex64

data_2023_06_19 = {}


# Pasted from the test output (see back_compat_test.py module docstring)
data_2023_06_19["f32"] = dict(
    testdata_version=1,
    platform='cpu',
    custom_call_targets=['lapack_spotrf'],
    serialized_date=datetime.date(2023, 6, 19),
    inputs=(array([[ 24.343887,  13.603932,  20.50489 ,  12.063956],
       [ 13.603932,  58.879757, -31.84056 ,  16.328012],
       [ 20.50489 , -31.84056 ,  66.890755,  -9.92216 ],
       [ 12.063956,  16.328012,  -9.92216 ,  23.640734]], dtype=float32),),
    expected_outputs=(array([[ 4.9339523,  0.       ,  0.       ,  0.       ],
       [ 2.7572079,  7.1608353,  0.       ,  0.       ],
       [ 4.155875 , -6.0466647,  3.6134892,  0.       ],
       [ 2.4450896,  1.3387254, -3.3177967,  2.2050648]], dtype=float32),),
    mlir_module_text=r"""
#loc = loc(unknown)
module @jit_cholesky attributes {mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} {
  func.func public @main(%arg0: tensor<4x4xf32> {jax.arg_info = "x", mhlo.sharding = "{replicated}"} loc(unknown)) -> (tensor<4x4xf32> {jax.result_info = ""}) {
    %0 = stablehlo.transpose %arg0, dims = [1, 0] : (tensor<4x4xf32>) -> tensor<4x4xf32> loc(#loc2)
    %1 = stablehlo.add %arg0, %0 : tensor<4x4xf32> loc(#loc3)
    %2 = stablehlo.constant dense<2.000000e+00> : tensor<f32> loc(#loc)
    %3 = stablehlo.broadcast_in_dim %2, dims = [] : (tensor<f32>) -> tensor<4x4xf32> loc(#loc4)
    %4 = stablehlo.divide %1, %3 : tensor<4x4xf32> loc(#loc4)
    %5 = stablehlo.constant dense<1> : tensor<i32> loc(#loc5)
    %6 = stablehlo.constant dense<1> : tensor<i32> loc(#loc5)
    %7 = stablehlo.constant dense<4> : tensor<i32> loc(#loc5)
    %8:2 = stablehlo.custom_call @lapack_spotrf(%5, %6, %7, %4) {api_version = 2 : i32, operand_layouts = [dense<> : tensor<0xindex>, dense<> : tensor<0xindex>, dense<> : tensor<0xindex>, dense<[0, 1]> : tensor<2xindex>], output_operand_aliases = [#stablehlo.output_operand_alias<output_tuple_indices = [0], operand_index = 3, operand_tuple_indices = []>], result_layouts = [dense<[0, 1]> : tensor<2xindex>, dense<> : tensor<0xindex>]} : (tensor<i32>, tensor<i32>, tensor<i32>, tensor<4x4xf32>) -> (tensor<4x4xf32>, tensor<i32>) loc(#loc5)
    %9 = stablehlo.constant dense<0> : tensor<i32> loc(#loc5)
    %10 = stablehlo.broadcast_in_dim %9, dims = [] : (tensor<i32>) -> tensor<i32> loc(#loc5)
    %11 = stablehlo.compare  EQ, %8#1, %10,  SIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1> loc(#loc5)
    %12 = stablehlo.broadcast_in_dim %11, dims = [] : (tensor<i1>) -> tensor<1x1xi1> loc(#loc5)
    %13 = stablehlo.constant dense<0x7FC00000> : tensor<f32> loc(#loc5)
    %14 = stablehlo.broadcast_in_dim %13, dims = [] : (tensor<f32>) -> tensor<4x4xf32> loc(#loc5)
    %15 = stablehlo.broadcast_in_dim %12, dims = [0, 1] : (tensor<1x1xi1>) -> tensor<4x4xi1> loc(#loc5)
    %16 = stablehlo.select %15, %8#0, %14 : tensor<4x4xi1>, tensor<4x4xf32> loc(#loc5)
    %17 = call @tril(%16) : (tensor<4x4xf32>) -> tensor<4x4xf32> loc(#loc6)
    return %17 : tensor<4x4xf32> loc(#loc)
  } loc(#loc)
  func.func private @tril(%arg0: tensor<4x4xf32> loc(unknown)) -> tensor<4x4xf32> {
    %0 = stablehlo.iota dim = 0 : tensor<4x4xi32> loc(#loc7)
    %1 = stablehlo.constant dense<0> : tensor<i32> loc(#loc6)
    %2 = stablehlo.broadcast_in_dim %1, dims = [] : (tensor<i32>) -> tensor<4x4xi32> loc(#loc8)
    %3 = stablehlo.add %0, %2 : tensor<4x4xi32> loc(#loc8)
    %4 = stablehlo.iota dim = 1 : tensor<4x4xi32> loc(#loc9)
    %5 = stablehlo.compare  GE, %3, %4,  SIGNED : (tensor<4x4xi32>, tensor<4x4xi32>) -> tensor<4x4xi1> loc(#loc10)
    %6 = stablehlo.constant dense<0.000000e+00> : tensor<f32> loc(#loc6)
    %7 = stablehlo.broadcast_in_dim %6, dims = [] : (tensor<f32>) -> tensor<4x4xf32> loc(#loc11)
    %8 = stablehlo.select %5, %arg0, %7 : tensor<4x4xi1>, tensor<4x4xf32> loc(#loc12)
    return %8 : tensor<4x4xf32> loc(#loc6)
  } loc(#loc6)
} loc(#loc)
#loc1 = loc("/Users/necula/Source/jax/jax/experimental/jax2tf/tests/back_compat_test.py":292:0)
#loc2 = loc("jit(cholesky)/jit(main)/transpose[permutation=(1, 0)]"(#loc1))
#loc3 = loc("jit(cholesky)/jit(main)/add"(#loc1))
#loc4 = loc("jit(cholesky)/jit(main)/div"(#loc1))
#loc5 = loc("jit(cholesky)/jit(main)/cholesky"(#loc1))
#loc6 = loc("jit(cholesky)/jit(main)/pjit[in_shardings=(UnspecifiedValue,) out_shardings=(UnspecifiedValue,) resource_env=None donated_invars=(False,) name=tril keep_unused=False inline=False]"(#loc1))
#loc7 = loc("jit(cholesky)/jit(main)/jit(tril)/iota[dtype=int32 shape=(4, 4) dimension=0]"(#loc1))
#loc8 = loc("jit(cholesky)/jit(main)/jit(tril)/add"(#loc1))
#loc9 = loc("jit(cholesky)/jit(main)/jit(tril)/iota[dtype=int32 shape=(4, 4) dimension=1]"(#loc1))
#loc10 = loc("jit(cholesky)/jit(main)/jit(tril)/ge"(#loc1))
#loc11 = loc("jit(cholesky)/jit(main)/jit(tril)/broadcast_in_dim[shape=(4, 4) broadcast_dimensions=()]"(#loc1))
#loc12 = loc("jit(cholesky)/jit(main)/jit(tril)/select_n"(#loc1))
""",
    mlir_module_serialized=b'ML\xefR\x01StableHLO_v0.9.0\x00\x01)\x05\x01\x03\x01\x03\x05\x03\x19\x07\t\x0b\r\x0f\x11\x13\x15\x17\x19\x1b\x1d\x03"\x02\xd9%\x01\x87\x0f\x17\x07\x0b\x13\x0f\x0b\x0f\x0b\x0b\x0b\x0b\x0b\x13\x0b\x0f\x0b\x0b\x0f\x13#\x0b\x0b\x0b33\x0b\x0b\x13\x0f\x0b\x0b\x13\x0f\x0b\x1b\x0f\x0b\x13\x0f\x0b\x0f\x0b\x13\x0b\x0f\x0b\x0f\x0b\x13\x0b\x0b\x13K\x0b\x0b\x0b\x0b\x0b\x0b\x0b\x0b\x1b\x13\x13\x13\x0b\x03S\x0f\x0b\x0b\x0f\x0b\x0bO\x0f\x1b\x0b\x0b\x0b\x0b\x0f\x13\x0b\x0b\x0b\x0b\x0b\x0f\x1f\x0f\x0f\x0b\x1fO\x1f\x1f\x1f\x0b\x0b\x0b\x0b\x1b\x0f\x17\x13\x0b\x1fO\x01\x03\x0f\x03#\x17\x0f\x0f\x17\x07\x07\x07\x07\x17\x13\x07\x17\x13\x13\x13\x0f\x17\x02J\x07\x1dg\x03\x177\x92\x04\x01\x1f\x05\x1f\x03\x03\x1d\xb3\x1d5\x03\x05!\x11\x01\x05\x05#\x05%\x05\'\x05)\x05+\x03\x03\x07\xb1\x05-\x1d?\x03\x05/\x051\x1de\x03\x03\x03\x07\xbf\x03\x07+\x0f-\x0f\r/\x053\x055\x057\x03\x0b\x11\x95\x13\x89\x15\xa1\r\xa7\x17\xa9\x03\x0b\x11\x8d\x13\x89\x15\x8d\r\x8f\x17\xad\x059\x05;\x03\x03\x19\xaf\x1d=\x03\x05=\x05?\x03\x03\x19\xb5\x1dE\x03\x05A\x03\x05!\x91#\xb7\x1dK\x03\x05C\x03\x03\x07\xb9\x1dQ\x03\x05E\x1dU\x03\x05G\x03\x03Y\xbb\x05I\x1d]\x03\x05K\x1da\x03\x05M\x03\x03\x07\xbd\x05O\x05Q\x03\x03\x07\xc1\x03\x11m\xc3o\x8bq\xc5s\xc7u\xc9w\xcby\xcd{\xd1\x05S\x05U\x05W\x05Y\x05[\x05]\x05_\x05a\x03\x05!\x91#\xd3\x03\x03\x07\xd5\x03\x03\x1d\xd7\x03\x03\x85\x8f\x05c\x1f\x1d\x01#\x19\x1de\x03\x03\xab\x1dg\t\x07\x1f\x1f!\x00\x00\x00\x00\x00\x00\x00\x00\x01\x00\x00\x00\x00\x00\x00\x00\x03\x03\x97\r\x05\x99\x9b\x9d\x9f\x1di\x1dk\x1dm\x1do\x03\x03\xa3\r\x03\xa5\x8b\x1dq\x1ds\x1du\r\x01\x1dw\x13\x0b\x01\x1f\x05\t\x00\x00\x00\x00\x1f\x1b\x01\x13\x0b\x05\x07\x05\x1f\x07\t\x00\x00\x00\x00\x1f\x15!\x01\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x1f\x07\t\x00\x00\x00@\x1f\x05\t\x01\x00\x00\x00\x1f\x05\t\x04\x00\x00\x00\x0b\x05\x1dy\x03\x01\x05\x01\x03\t\x87\x87\x87\x93\x03\x03\xcf\x15\x03\x01\r\x01\x03\x05\x93\x87\x07\x01\x1f\x07\t\x00\x00\xc0\x7f\x1f\x15!\x00\x00\x00\x00\x00\x00\x00\x00\x01\x00\x00\x00\x00\x00\x00\x00\x01\x02\x02)\x05\x11\x11\x0f)\x01\x11)\x01\x0f)\x05\x11\x11\x11\x1d\x01\t\x1b)\x05\x11\x11\r)\x03\t\x0b\x13\x11\x03\x03\x03\x03)\x03\x01\x0b)\x03\x01\x17)\x03\t\x17)\x01\r)\x05\x05\x05\r\x04\xd6\x03\x05\x01\x11\x05)\x07\x03\x01\t\x07\x11\x051\x05\x03)O\x03\x03\x05\x13\x07[W\x03\x03\x03\x01\x0b\x06_\x03\x03\x05\x01\x03\x03\x03\x05c\x03\x07\x05\x07%\t\x03\x03\x03\x07\x15\x06%\x03\x03\x05\x05\t\x03\x03\x01\'\x03\x05\x03\x03\x01\'\x03\x05\x03\x03\x01i\x03\x05\x17\x07\x01k\x05\x03\x05\t\r\x0f\x11\x0b\x03\x03\x01\x1b\x03\x05\x05\x07\x01\t\x03\x05\x03\x17\r\x07\x01}\x03!\x05\x15\x19\x05\x07\x01\t\x03#\x03\x1b\x03\x03\x01\x7f\x03\x07\x05\x07\x01\t\x03\x03\x03\x1f\x05\x07\x01\x81\x03\x13\x03\x1d\x0f\x06\x01\x03\x03\x07#\x13!\x19\x07\x0b\x83\x03\x03\x03%\x11\x04\x05\x03\'\x07\x11\x0b3\x05\x03\x15+\x03\x03\x05\t\x03;9\x03\t\x03\x03\x0b\x1b\x03\x05\x05\x07\x1f\t\x03\t\x03\x05\x0b\x06\x1f\x03\t\x05\x03\x07\t\x03CA\x03\t\r\x07IG\x03\x13\x05\t\x0b\x03\x03\x0bM\x03\x07\x05\x07O\t\x03\x03\x03\x0f\x0f\x06S\x03\x03\x07\r\x01\x11\x11\x04\x0b\x03\x13\x06\x03\x01\x05\x01\x00\n\x16{\x1d\x11\x0f\x0b!\x1b\x1d\x05\x1b\x0b\x03\x0f\x1f/!!)#\x1f\x19C99m\x19W\xb3K\x9bM\x9b\x97\xd2\x02\x1b%)+\x1b+\x1f\x1f\x15\x1d\x15\x13\r\x11\x1f\x15\x1b\x15\x15\x17\x0f\x11\x11)\x19\x0f\x0b\x11builtin\x00vhlo\x00module\x00constant_v1\x00broadcast_in_dim_v1\x00func_v1\x00iota_v1\x00add_v1\x00compare_v1\x00select_v1\x00return_v1\x00transpose_v1\x00divide_v1\x00custom_call_v1\x00call_v1\x00value\x00sym_name\x00arg_attrs\x00function_type\x00res_attrs\x00sym_visibility\x00iota_dimension\x00broadcast_dimensions\x00compare_type\x00comparison_direction\x00mhlo.num_partitions\x00mhlo.num_replicas\x00jit_cholesky\x00jit(cholesky)/jit(main)/pjit[in_shardings=(UnspecifiedValue,) out_shardings=(UnspecifiedValue,) resource_env=None donated_invars=(False,) name=tril keep_unused=False inline=False]\x00/Users/necula/Source/jax/jax/experimental/jax2tf/tests/back_compat_test.py\x00jit(cholesky)/jit(main)/jit(tril)/iota[dtype=int32 shape=(4, 4) dimension=0]\x00jit(cholesky)/jit(main)/jit(tril)/add\x00jit(cholesky)/jit(main)/jit(tril)/iota[dtype=int32 shape=(4, 4) dimension=1]\x00jit(cholesky)/jit(main)/jit(tril)/ge\x00jit(cholesky)/jit(main)/jit(tril)/broadcast_in_dim[shape=(4, 4) broadcast_dimensions=()]\x00jit(cholesky)/jit(main)/jit(tril)/select_n\x00permutation\x00jit(cholesky)/jit(main)/transpose[permutation=(1, 0)]\x00jit(cholesky)/jit(main)/add\x00jit(cholesky)/jit(main)/div\x00jit(cholesky)/jit(main)/cholesky\x00api_version\x00backend_config\x00call_target_name\x00called_computations\x00has_side_effect\x00operand_layouts\x00output_operand_aliases\x00result_layouts\x00callee\x00\x00tril\x00jax.arg_info\x00x\x00mhlo.sharding\x00{replicated}\x00jax.result_info\x00main\x00public\x00private\x00lapack_spotrf\x00',
    xla_call_module_version=6,
)  # End paste


# Pasted from the test output (see back_compat_test.py module docstring)
data_2023_06_19["f64"] = dict(
    testdata_version=1,
    platform='cpu',
    custom_call_targets=['lapack_dpotrf'],
    serialized_date=datetime.date(2023, 6, 19),
    inputs=(array([[ 23.022171138130666 , -16.79765603341739  ,   0.9133449305189146,
        -25.36636199966769  ],
       [-16.79765603341739  ,  31.655770252600092 ,  -1.5189878284433445,
         20.0344758332268   ],
       [  0.9133449305189146,  -1.5189878284433445,  10.940134497877208 ,
          8.169020034607513 ],
       [-25.36636199966769  ,  20.0344758332268   ,   8.169020034607513 ,
         37.054603917509596 ]]),),
    expected_outputs=(array([[ 4.7981424674691215 ,  0.                 ,  0.                 ,
         0.                 ],
       [-3.500866459740129  ,  4.404509539513645  ,  0.                 ,
         0.                 ],
       [ 0.19035385812557523, -0.1935707899825621 ,  3.2964268922333835 ,
         0.                 ],
       [-5.286704630312426  ,  0.3465604732420997 ,  2.8037778311164425 ,
         1.060228174247855  ]]),),
    mlir_module_text=r"""
#loc = loc(unknown)
module @jit_cholesky attributes {mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} {
  func.func public @main(%arg0: tensor<4x4xf64> {jax.arg_info = "x", mhlo.sharding = "{replicated}"} loc(unknown)) -> (tensor<4x4xf64> {jax.result_info = ""}) {
    %0 = stablehlo.transpose %arg0, dims = [1, 0] : (tensor<4x4xf64>) -> tensor<4x4xf64> loc(#loc2)
    %1 = stablehlo.add %arg0, %0 : tensor<4x4xf64> loc(#loc3)
    %2 = stablehlo.constant dense<2.000000e+00> : tensor<f64> loc(#loc)
    %3 = stablehlo.broadcast_in_dim %2, dims = [] : (tensor<f64>) -> tensor<4x4xf64> loc(#loc4)
    %4 = stablehlo.divide %1, %3 : tensor<4x4xf64> loc(#loc4)
    %5 = stablehlo.constant dense<1> : tensor<i32> loc(#loc5)
    %6 = stablehlo.constant dense<1> : tensor<i32> loc(#loc5)
    %7 = stablehlo.constant dense<4> : tensor<i32> loc(#loc5)
    %8:2 = stablehlo.custom_call @lapack_dpotrf(%5, %6, %7, %4) {api_version = 2 : i32, operand_layouts = [dense<> : tensor<0xindex>, dense<> : tensor<0xindex>, dense<> : tensor<0xindex>, dense<[0, 1]> : tensor<2xindex>], output_operand_aliases = [#stablehlo.output_operand_alias<output_tuple_indices = [0], operand_index = 3, operand_tuple_indices = []>], result_layouts = [dense<[0, 1]> : tensor<2xindex>, dense<> : tensor<0xindex>]} : (tensor<i32>, tensor<i32>, tensor<i32>, tensor<4x4xf64>) -> (tensor<4x4xf64>, tensor<i32>) loc(#loc5)
    %9 = stablehlo.constant dense<0> : tensor<i32> loc(#loc5)
    %10 = stablehlo.broadcast_in_dim %9, dims = [] : (tensor<i32>) -> tensor<i32> loc(#loc5)
    %11 = stablehlo.compare  EQ, %8#1, %10,  SIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1> loc(#loc5)
    %12 = stablehlo.broadcast_in_dim %11, dims = [] : (tensor<i1>) -> tensor<1x1xi1> loc(#loc5)
    %13 = stablehlo.constant dense<0x7FF8000000000000> : tensor<f64> loc(#loc5)
    %14 = stablehlo.broadcast_in_dim %13, dims = [] : (tensor<f64>) -> tensor<4x4xf64> loc(#loc5)
    %15 = stablehlo.broadcast_in_dim %12, dims = [0, 1] : (tensor<1x1xi1>) -> tensor<4x4xi1> loc(#loc5)
    %16 = stablehlo.select %15, %8#0, %14 : tensor<4x4xi1>, tensor<4x4xf64> loc(#loc5)
    %17 = call @tril(%16) : (tensor<4x4xf64>) -> tensor<4x4xf64> loc(#loc6)
    return %17 : tensor<4x4xf64> loc(#loc)
  } loc(#loc)
  func.func private @tril(%arg0: tensor<4x4xf64> loc(unknown)) -> tensor<4x4xf64> {
    %0 = stablehlo.iota dim = 0 : tensor<4x4xi32> loc(#loc7)
    %1 = stablehlo.constant dense<0> : tensor<i32> loc(#loc6)
    %2 = stablehlo.broadcast_in_dim %1, dims = [] : (tensor<i32>) -> tensor<4x4xi32> loc(#loc8)
    %3 = stablehlo.add %0, %2 : tensor<4x4xi32> loc(#loc8)
    %4 = stablehlo.iota dim = 1 : tensor<4x4xi32> loc(#loc9)
    %5 = stablehlo.compare  GE, %3, %4,  SIGNED : (tensor<4x4xi32>, tensor<4x4xi32>) -> tensor<4x4xi1> loc(#loc10)
    %6 = stablehlo.constant dense<0.000000e+00> : tensor<f64> loc(#loc6)
    %7 = stablehlo.broadcast_in_dim %6, dims = [] : (tensor<f64>) -> tensor<4x4xf64> loc(#loc11)
    %8 = stablehlo.select %5, %arg0, %7 : tensor<4x4xi1>, tensor<4x4xf64> loc(#loc12)
    return %8 : tensor<4x4xf64> loc(#loc6)
  } loc(#loc6)
} loc(#loc)
#loc1 = loc("/Users/necula/Source/jax/jax/experimental/jax2tf/tests/back_compat_test.py":292:0)
#loc2 = loc("jit(cholesky)/jit(main)/transpose[permutation=(1, 0)]"(#loc1))
#loc3 = loc("jit(cholesky)/jit(main)/add"(#loc1))
#loc4 = loc("jit(cholesky)/jit(main)/div"(#loc1))
#loc5 = loc("jit(cholesky)/jit(main)/cholesky"(#loc1))
#loc6 = loc("jit(cholesky)/jit(main)/pjit[in_shardings=(UnspecifiedValue,) out_shardings=(UnspecifiedValue,) resource_env=None donated_invars=(False,) name=tril keep_unused=False inline=False]"(#loc1))
#loc7 = loc("jit(cholesky)/jit(main)/jit(tril)/iota[dtype=int32 shape=(4, 4) dimension=0]"(#loc1))
#loc8 = loc("jit(cholesky)/jit(main)/jit(tril)/add"(#loc1))
#loc9 = loc("jit(cholesky)/jit(main)/jit(tril)/iota[dtype=int32 shape=(4, 4) dimension=1]"(#loc1))
#loc10 = loc("jit(cholesky)/jit(main)/jit(tril)/ge"(#loc1))
#loc11 = loc("jit(cholesky)/jit(main)/jit(tril)/broadcast_in_dim[shape=(4, 4) broadcast_dimensions=()]"(#loc1))
#loc12 = loc("jit(cholesky)/jit(main)/jit(tril)/select_n"(#loc1))
""",
    mlir_module_serialized=b'ML\xefR\x01StableHLO_v0.9.0\x00\x01)\x05\x01\x03\x01\x03\x05\x03\x19\x07\t\x0b\r\x0f\x11\x13\x15\x17\x19\x1b\x1d\x03"\x02\xd9%\x01\x87\x0f\x17\x07\x0b\x13\x0f\x0b\x0f\x0b\x0b\x0b\x0b\x0b\x13\x0b\x0f\x0b\x0b\x0f\x13#\x0b\x0b\x0b33\x0b\x0b\x13\x0f\x0b\x0b\x13\x0f\x0b\x1b\x0f\x0b\x13\x0f\x0b\x0f\x0b\x13\x0b\x0f\x0b\x0f\x0b\x13\x0b\x0b\x13K\x0b\x0b\x0b\x0b\x0b\x0b\x0b\x0b\x1b\x13\x13\x13\x0b\x03S\x0f\x0b\x0b\x0f\x0b\x0bO\x0f\x1b\x0b\x0b\x0b\x0b\x0f\x13\x0b\x0b\x0b\x0b\x0b\x0f\x1f\x0f\x0f\x0b/O/\x1f\x1f\x0b\x0b\x0b\x0b\x1b\x0f\x17\x13\x0b/O\x01\x03\x0f\x03#\x17\x0f\x0f\x17\x07\x07\x07\x07\x17\x13\x07\x17\x13\x13\x13\x0f\x17\x02z\x07\x1dg\x03\x177\x92\x04\x01\x1f\x05\x1f\x03\x03\x1d\xb3\x1d5\x03\x05!\x11\x01\x05\x05#\x05%\x05\'\x05)\x05+\x03\x03\x07\xb1\x05-\x1d?\x03\x05/\x051\x1de\x03\x03\x03\x07\xbf\x03\x07+\x0f-\x0f\r/\x053\x055\x057\x03\x0b\x11\x95\x13\x89\x15\xa1\r\xa7\x17\xa9\x03\x0b\x11\x8d\x13\x89\x15\x8d\r\x8f\x17\xad\x059\x05;\x03\x03\x19\xaf\x1d=\x03\x05=\x05?\x03\x03\x19\xb5\x1dE\x03\x05A\x03\x05!\x91#\xb7\x1dK\x03\x05C\x03\x03\x07\xb9\x1dQ\x03\x05E\x1dU\x03\x05G\x03\x03Y\xbb\x05I\x1d]\x03\x05K\x1da\x03\x05M\x03\x03\x07\xbd\x05O\x05Q\x03\x03\x07\xc1\x03\x11m\xc3o\x8bq\xc5s\xc7u\xc9w\xcby\xcd{\xd1\x05S\x05U\x05W\x05Y\x05[\x05]\x05_\x05a\x03\x05!\x91#\xd3\x03\x03\x07\xd5\x03\x03\x1d\xd7\x03\x03\x85\x8f\x05c\x1f\x1d\x01#\x19\x1de\x03\x03\xab\x1dg\t\x07\x1f\x1f!\x00\x00\x00\x00\x00\x00\x00\x00\x01\x00\x00\x00\x00\x00\x00\x00\x03\x03\x97\r\x05\x99\x9b\x9d\x9f\x1di\x1dk\x1dm\x1do\x03\x03\xa3\r\x03\xa5\x8b\x1dq\x1ds\x1du\r\x01\x1dw\x13\x0b\x01\x1f\x05\t\x00\x00\x00\x00\x1f\x1b\x01\x13\x0b\x05\x07\x05\x1f\x07\x11\x00\x00\x00\x00\x00\x00\x00\x00\x1f\x15!\x01\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x1f\x07\x11\x00\x00\x00\x00\x00\x00\x00@\x1f\x05\t\x01\x00\x00\x00\x1f\x05\t\x04\x00\x00\x00\x0b\x05\x1dy\x03\x01\x05\x01\x03\t\x87\x87\x87\x93\x03\x03\xcf\x15\x03\x01\r\x01\x03\x05\x93\x87\x07\x01\x1f\x07\x11\x00\x00\x00\x00\x00\x00\xf8\x7f\x1f\x15!\x00\x00\x00\x00\x00\x00\x00\x00\x01\x00\x00\x00\x00\x00\x00\x00\x01\x02\x02)\x05\x11\x11\x0f)\x01\x11)\x01\x0f)\x05\x11\x11\x11\x1d\x01\x0b\x1b)\x05\x11\x11\r)\x03\t\x0b\x13\x11\x03\x03\x03\x03)\x03\x01\x0b)\x03\x01\x17)\x03\t\x17)\x01\r)\x05\x05\x05\r\x04\xd6\x03\x05\x01\x11\x05)\x07\x03\x01\t\x07\x11\x051\x05\x03)O\x03\x03\x05\x13\x07[W\x03\x03\x03\x01\x0b\x06_\x03\x03\x05\x01\x03\x03\x03\x05c\x03\x07\x05\x07%\t\x03\x03\x03\x07\x15\x06%\x03\x03\x05\x05\t\x03\x03\x01\'\x03\x05\x03\x03\x01\'\x03\x05\x03\x03\x01i\x03\x05\x17\x07\x01k\x05\x03\x05\t\r\x0f\x11\x0b\x03\x03\x01\x1b\x03\x05\x05\x07\x01\t\x03\x05\x03\x17\r\x07\x01}\x03!\x05\x15\x19\x05\x07\x01\t\x03#\x03\x1b\x03\x03\x01\x7f\x03\x07\x05\x07\x01\t\x03\x03\x03\x1f\x05\x07\x01\x81\x03\x13\x03\x1d\x0f\x06\x01\x03\x03\x07#\x13!\x19\x07\x0b\x83\x03\x03\x03%\x11\x04\x05\x03\'\x07\x11\x0b3\x05\x03\x15+\x03\x03\x05\t\x03;9\x03\t\x03\x03\x0b\x1b\x03\x05\x05\x07\x1f\t\x03\t\x03\x05\x0b\x06\x1f\x03\t\x05\x03\x07\t\x03CA\x03\t\r\x07IG\x03\x13\x05\t\x0b\x03\x03\x0bM\x03\x07\x05\x07O\t\x03\x03\x03\x0f\x0f\x06S\x03\x03\x07\r\x01\x11\x11\x04\x0b\x03\x13\x06\x03\x01\x05\x01\x00\n\x16{\x1d\x11\x0f\x0b!\x1b\x1d\x05\x1b\x0b\x03\x0f\x1f/!!)#\x1f\x19C99m\x19W\xb3K\x9bM\x9b\x97\xd2\x02\x1b%)+\x1b+\x1f\x1f\x15\x1d\x15\x13\r\x11\x1f\x15\x1b\x15\x15\x17\x0f\x11\x11)\x19\x0f\x0b\x11builtin\x00vhlo\x00module\x00constant_v1\x00broadcast_in_dim_v1\x00func_v1\x00iota_v1\x00add_v1\x00compare_v1\x00select_v1\x00return_v1\x00transpose_v1\x00divide_v1\x00custom_call_v1\x00call_v1\x00value\x00sym_name\x00arg_attrs\x00function_type\x00res_attrs\x00sym_visibility\x00iota_dimension\x00broadcast_dimensions\x00compare_type\x00comparison_direction\x00mhlo.num_partitions\x00mhlo.num_replicas\x00jit_cholesky\x00jit(cholesky)/jit(main)/pjit[in_shardings=(UnspecifiedValue,) out_shardings=(UnspecifiedValue,) resource_env=None donated_invars=(False,) name=tril keep_unused=False inline=False]\x00/Users/necula/Source/jax/jax/experimental/jax2tf/tests/back_compat_test.py\x00jit(cholesky)/jit(main)/jit(tril)/iota[dtype=int32 shape=(4, 4) dimension=0]\x00jit(cholesky)/jit(main)/jit(tril)/add\x00jit(cholesky)/jit(main)/jit(tril)/iota[dtype=int32 shape=(4, 4) dimension=1]\x00jit(cholesky)/jit(main)/jit(tril)/ge\x00jit(cholesky)/jit(main)/jit(tril)/broadcast_in_dim[shape=(4, 4) broadcast_dimensions=()]\x00jit(cholesky)/jit(main)/jit(tril)/select_n\x00permutation\x00jit(cholesky)/jit(main)/transpose[permutation=(1, 0)]\x00jit(cholesky)/jit(main)/add\x00jit(cholesky)/jit(main)/div\x00jit(cholesky)/jit(main)/cholesky\x00api_version\x00backend_config\x00call_target_name\x00called_computations\x00has_side_effect\x00operand_layouts\x00output_operand_aliases\x00result_layouts\x00callee\x00\x00tril\x00jax.arg_info\x00x\x00mhlo.sharding\x00{replicated}\x00jax.result_info\x00main\x00public\x00private\x00lapack_dpotrf\x00',
    xla_call_module_version=6,
)  # End paste



# Pasted from the test output (see back_compat_test.py module docstring)
data_2023_06_19["c64"] = dict(
    testdata_version=1,
    platform='cpu',
    custom_call_targets=['lapack_cpotrf'],
    serialized_date=datetime.date(2023, 6, 19),
    inputs=(array([[ 38.089394 +6.36582342e-09j,   3.3509154+3.13455486e+01j,
         -0.5972489-3.80308151e+01j, -19.04205  +1.22770605e+01j],
       [  3.3509154-3.13455486e+01j,  73.875755 +4.06565448e-09j,
        -12.427276 -1.23379612e+01j,  41.542507 -9.63993359e+00j],
       [ -0.5972489+3.80308151e+01j, -12.427276 +1.23379612e+01j,
         73.04141  -4.18667753e-07j,   8.193126 -2.60565052e+01j],
       [-19.04205  -1.22770605e+01j,  41.542507 +9.63993359e+00j,
          8.193126 +2.60565052e+01j,  52.977036 -1.09952367e-07j]],
      dtype=complex64),),
    expected_outputs=(array([[ 6.1716604 +0.j       ,  0.        +0.j       ,
         0.        +0.j       ,  0.        +0.j       ],
       [ 0.542952  -5.078949j ,  6.912687  +0.j       ,
         0.        +0.j       ,  0.        +0.j       ],
       [-0.09677281+6.162169j ,  2.7373738 +1.3719271j,
         5.0679703 +0.j       ,  0.        +0.j       ],
       [-3.0854013 -1.9892638j,  4.7903748 +3.8177056j,
         0.3555784 +0.5865844j,  1.2276335 +0.j       ]], dtype=complex64),),
    mlir_module_text=r"""
#loc = loc(unknown)
module @jit_cholesky attributes {mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} {
  func.func public @main(%arg0: tensor<4x4xcomplex<f32>> {jax.arg_info = "x", mhlo.sharding = "{replicated}"} loc(unknown)) -> (tensor<4x4xcomplex<f32>> {jax.result_info = ""}) {
    %0 = stablehlo.transpose %arg0, dims = [1, 0] : (tensor<4x4xcomplex<f32>>) -> tensor<4x4xcomplex<f32>> loc(#loc2)
    %1 = stablehlo.real %0 : (tensor<4x4xcomplex<f32>>) -> tensor<4x4xf32> loc(#loc3)
    %2 = stablehlo.imag %0 : (tensor<4x4xcomplex<f32>>) -> tensor<4x4xf32> loc(#loc4)
    %3 = stablehlo.negate %2 : tensor<4x4xf32> loc(#loc5)
    %4 = stablehlo.complex %1, %3 : tensor<4x4xcomplex<f32>> loc(#loc6)
    %5 = stablehlo.add %arg0, %4 : tensor<4x4xcomplex<f32>> loc(#loc7)
    %6 = stablehlo.constant dense<(2.000000e+00,0.000000e+00)> : tensor<complex<f32>> loc(#loc)
    %7 = stablehlo.broadcast_in_dim %6, dims = [] : (tensor<complex<f32>>) -> tensor<4x4xcomplex<f32>> loc(#loc8)
    %8 = stablehlo.divide %5, %7 : tensor<4x4xcomplex<f32>> loc(#loc8)
    %9 = stablehlo.constant dense<1> : tensor<i32> loc(#loc9)
    %10 = stablehlo.constant dense<1> : tensor<i32> loc(#loc9)
    %11 = stablehlo.constant dense<4> : tensor<i32> loc(#loc9)
    %12:2 = stablehlo.custom_call @lapack_cpotrf(%9, %10, %11, %8) {api_version = 2 : i32, operand_layouts = [dense<> : tensor<0xindex>, dense<> : tensor<0xindex>, dense<> : tensor<0xindex>, dense<[0, 1]> : tensor<2xindex>], output_operand_aliases = [#stablehlo.output_operand_alias<output_tuple_indices = [0], operand_index = 3, operand_tuple_indices = []>], result_layouts = [dense<[0, 1]> : tensor<2xindex>, dense<> : tensor<0xindex>]} : (tensor<i32>, tensor<i32>, tensor<i32>, tensor<4x4xcomplex<f32>>) -> (tensor<4x4xcomplex<f32>>, tensor<i32>) loc(#loc9)
    %13 = stablehlo.constant dense<0> : tensor<i32> loc(#loc9)
    %14 = stablehlo.broadcast_in_dim %13, dims = [] : (tensor<i32>) -> tensor<i32> loc(#loc9)
    %15 = stablehlo.compare  EQ, %12#1, %14,  SIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1> loc(#loc9)
    %16 = stablehlo.broadcast_in_dim %15, dims = [] : (tensor<i1>) -> tensor<1x1xi1> loc(#loc9)
    %17 = stablehlo.constant dense<(0x7FC00000,0x7FC00000)> : tensor<complex<f32>> loc(#loc9)
    %18 = stablehlo.broadcast_in_dim %17, dims = [] : (tensor<complex<f32>>) -> tensor<4x4xcomplex<f32>> loc(#loc9)
    %19 = stablehlo.broadcast_in_dim %16, dims = [0, 1] : (tensor<1x1xi1>) -> tensor<4x4xi1> loc(#loc9)
    %20 = stablehlo.select %19, %12#0, %18 : tensor<4x4xi1>, tensor<4x4xcomplex<f32>> loc(#loc9)
    %21 = call @tril(%20) : (tensor<4x4xcomplex<f32>>) -> tensor<4x4xcomplex<f32>> loc(#loc10)
    return %21 : tensor<4x4xcomplex<f32>> loc(#loc)
  } loc(#loc)
  func.func private @tril(%arg0: tensor<4x4xcomplex<f32>> loc(unknown)) -> tensor<4x4xcomplex<f32>> {
    %0 = stablehlo.iota dim = 0 : tensor<4x4xi32> loc(#loc11)
    %1 = stablehlo.constant dense<0> : tensor<i32> loc(#loc10)
    %2 = stablehlo.broadcast_in_dim %1, dims = [] : (tensor<i32>) -> tensor<4x4xi32> loc(#loc12)
    %3 = stablehlo.add %0, %2 : tensor<4x4xi32> loc(#loc12)
    %4 = stablehlo.iota dim = 1 : tensor<4x4xi32> loc(#loc13)
    %5 = stablehlo.compare  GE, %3, %4,  SIGNED : (tensor<4x4xi32>, tensor<4x4xi32>) -> tensor<4x4xi1> loc(#loc14)
    %6 = stablehlo.constant dense<(0.000000e+00,0.000000e+00)> : tensor<complex<f32>> loc(#loc10)
    %7 = stablehlo.broadcast_in_dim %6, dims = [] : (tensor<complex<f32>>) -> tensor<4x4xcomplex<f32>> loc(#loc15)
    %8 = stablehlo.select %5, %arg0, %7 : tensor<4x4xi1>, tensor<4x4xcomplex<f32>> loc(#loc16)
    return %8 : tensor<4x4xcomplex<f32>> loc(#loc10)
  } loc(#loc10)
} loc(#loc)
#loc1 = loc("/Users/necula/Source/jax/jax/experimental/jax2tf/tests/back_compat_test.py":292:0)
#loc2 = loc("jit(cholesky)/jit(main)/transpose[permutation=(1, 0)]"(#loc1))
#loc3 = loc("jit(cholesky)/jit(main)/real"(#loc1))
#loc4 = loc("jit(cholesky)/jit(main)/imag"(#loc1))
#loc5 = loc("jit(cholesky)/jit(main)/neg"(#loc1))
#loc6 = loc("jit(cholesky)/jit(main)/complex"(#loc1))
#loc7 = loc("jit(cholesky)/jit(main)/add"(#loc1))
#loc8 = loc("jit(cholesky)/jit(main)/div"(#loc1))
#loc9 = loc("jit(cholesky)/jit(main)/cholesky"(#loc1))
#loc10 = loc("jit(cholesky)/jit(main)/pjit[in_shardings=(UnspecifiedValue,) out_shardings=(UnspecifiedValue,) resource_env=None donated_invars=(False,) name=tril keep_unused=False inline=False]"(#loc1))
#loc11 = loc("jit(cholesky)/jit(main)/jit(tril)/iota[dtype=int32 shape=(4, 4) dimension=0]"(#loc1))
#loc12 = loc("jit(cholesky)/jit(main)/jit(tril)/add"(#loc1))
#loc13 = loc("jit(cholesky)/jit(main)/jit(tril)/iota[dtype=int32 shape=(4, 4) dimension=1]"(#loc1))
#loc14 = loc("jit(cholesky)/jit(main)/jit(tril)/ge"(#loc1))
#loc15 = loc("jit(cholesky)/jit(main)/jit(tril)/broadcast_in_dim[shape=(4, 4) broadcast_dimensions=()]"(#loc1))
#loc16 = loc("jit(cholesky)/jit(main)/jit(tril)/select_n"(#loc1))
""",
    mlir_module_serialized=b"ML\xefR\x01StableHLO_v0.9.0\x00\x011\x05\x01\x03\x01\x03\x05\x03!\x07\t\x0b\r\x0f\x11\x13\x15\x17\x19\x1b\x1d\x1f!#%\x03J\x02\xe9)\x01\x97\x17\x0f\x07\x0b\x13\x0f\x0b\x0f\x0b\x0b\x0b\x0b\x0b\x13\x0b\x0f\x0b\x0b\x0f\x13#\x0b\x0b\x0b33\x0b\x0b\x13\x0f\x0b\x0b\x13\x0f\x0b\x1b\x0f\x0b\x13\x0f\x0b\x0f\x0b\x13\x0b\x0f\x0b\x0f\x0b\x0f\x0b\x0f\x0b\x0f\x0b\x0f\x0b\x13\x0b\x0b\x13K\x0b\x0b\x0b\x0b\x0b\x0b\x0b\x0b\x1b\x13\x13\x13\x0b\x03S\x0f\x0b\x0b\x0f\x0b\x0bO\x0f\x1b\x0b\x0b\x0b\x0b\x0f\x13\x0b\x0b\x0b\x0b\x0b\x0f\x1f\x0f\x0f\x0b/O/\x1f\x1f\x0b\x0b\x0b\x0b\x1b\x0f\x17\x13\x0b/O\x01\x03\x0f\x03'\x17\x0f\x0f\x17\x07\x07\x17\x0b\x07\x07\x17\x13\x07\x17\x13\x13\x13\x0f\x17\x02\xe6\x07\x177\x92\x04\x01\x1dw\x01\x1f\x05'\x03\x03\x1d\xc3\x1d5\x01\x05)\x11\x01\x05\x05+\x05-\x05/\x051\x053\x03\x03\x07\xc1\x055\x1d?\x01\x057\x059\x1du\x01\x03\x03\x07\xcf\x03\x07+\x0f-\x0f\r/\x05;\x05=\x05?\x03\x0b\x11\xa5\x13\x99\x15\xb1\r\xb7\x17\xb9\x03\x0b\x11\x9d\x13\x99\x15\x9d\r\x9f\x17\xbd\x05A\x05C\x03\x03\x19\xbf\x1d=\x01\x05E\x05G\x03\x03\x19\xc5\x1dE\x01\x05I\x03\x05!\xa1#\xc7\x1dK\x01\x05K\x03\x03\x07\xc9\x1dQ\x01\x05M\x1dU\x01\x05O\x03\x03Y\xcb\x05Q\x1d]\x01\x05S\x1da\x01\x05U\x1de\x01\x05W\x1di\x01\x05Y\x1dm\x01\x05[\x1dq\x01\x05]\x03\x03\x07\xcd\x05_\x05a\x03\x03\x07\xd1\x03\x11}\xd3\x7f\x9b\x81\xd5\x83\xd7\x85\xd9\x87\xdb\x89\xdd\x8b\xe1\x05c\x05e\x05g\x05i\x05k\x05m\x05o\x05q\x03\x05!\xa1#\xe3\x03\x03\x07\xe5\x03\x03\x1d\xe7\x03\x03\x95\x9f\x05s\x1f!\x01#\x1d\x1du\x03\x03\xbb\x1dw\t\x07\x1f#!\x00\x00\x00\x00\x00\x00\x00\x00\x01\x00\x00\x00\x00\x00\x00\x00\x03\x03\xa7\r\x05\xa9\xab\xad\xaf\x1dy\x1d{\x1d}\x1d\x7f\x03\x03\xb3\r\x03\xb5\x9b\x1d\x81\x1d\x83\x1d\x85\r\x01\x1d\x87\x13\x0b\x01\x1f\x05\t\x00\x00\x00\x00\x1f\x1f\x01\x13\x0b\x05\x07\x05\x1f\x07\x11\x00\x00\x00\x00\x00\x00\x00\x00\x1f\x19!\x01\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x1f\x07\x11\x00\x00\x00@\x00\x00\x00\x00\x1f\x05\t\x01\x00\x00\x00\x1f\x05\t\x04\x00\x00\x00\x0b\x05\x1d\x89\x03\x01\x05\x01\x03\t\x97\x97\x97\xa3\x03\x03\xdf\x15\x03\x01\r\x01\x03\x05\xa3\x97\x07\x01\x1f\x07\x11\x00\x00\xc0\x7f\x00\x00\xc0\x7f\x1f\x19!\x00\x00\x00\x00\x00\x00\x00\x00\x01\x00\x00\x00\x00\x00\x00\x00\x01\x02\x02)\x05\x11\x11\x11)\x01\x15)\x01\x11)\x05\x11\x11\x15\x1d\x01)\x05\x11\x11\x13\x03\x13\t\x1b)\x05\x11\x11\r)\x03\t\x0b\x13\x11\x03\x03\x03\x03)\x03\x01\x0b)\x03\x01\x1b)\x03\t\x1b)\x01\r)\x05\x05\x05\r\x04J\x04\x05\x01\x11\x05)\x07\x03\x01\t\x07\x11\x051\x05\x031_\x03\x03\x05\x13\x07[W\x03\x03\x03\x01\x15\x06_\x03\x0f\x03\x03\x17\x06c\x03\x0f\x03\x03\x19\x06g\x03\x0f\x03\x07\x1b\x06k\x03\x03\x05\x05\t\x0b\x06o\x03\x03\x05\x01\x0b\x03\x03\x05s\x03\x07\x05\x07%\t\x03\x03\x03\x0f\x1d\x06%\x03\x03\x05\r\x11\x03\x03\x03'\x03\x05\x03\x03\x03'\x03\x05\x03\x03\x03y\x03\x05\x1f\x07\x03{\x05\x03\x05\t\x15\x17\x19\x13\x03\x03\x03\x1b\x03\x05\x05\x07\x03\t\x03\x05\x03\x1f\r\x07\x03\x8d\x03%\x05\x1d!\x05\x07\x03\t\x03'\x03#\x03\x03\x03\x8f\x03\x07\x05\x07\x03\t\x03\x03\x03'\x05\x07\x03\x91\x03\x17\x03%\x0f\x06\x03\x03\x03\x07+\x1b)!\x07\x0b\x93\x03\x03\x03-\x11\x04\x05\x03/\x07\x11\x0b3\x05\x03\x15+\x03\x03\x05\t\x03;9\x03\t\x03\x03\x0b\x1b\x03\x05\x05\x07\x1f\t\x03\t\x03\x05\x0b\x06\x1f\x03\t\x05\x03\x07\t\x03CA\x03\t\r\x07IG\x03\x17\x05\t\x0b\x03\x03\x0bM\x03\x07\x05\x07O\t\x03\x03\x03\x0f\x0f\x06S\x03\x03\x07\r\x01\x11\x11\x04\x0b\x03\x13\x06\x03\x01\x05\x01\x00\x96\x18\x8b\x1d\x11\x0f\x0b!\x1b\x1d\x05\x1b\x0b\x03\x0f\x1f/!!)#\x1f\x19C99A9;;m\x19W\xb3K\x9bM\x9b\x97\xd2\x02\x1b%)+\x1b+\x1f\x1f\x15\x1d\x15\x13\r\x11\x1f\x15\x17\x15\x11\x11\x1b\x15\x15\x17\x0f\x11\x11)\x19\x0f\x0b\x11builtin\x00vhlo\x00module\x00constant_v1\x00broadcast_in_dim_v1\x00func_v1\x00iota_v1\x00add_v1\x00compare_v1\x00select_v1\x00return_v1\x00transpose_v1\x00real_v1\x00imag_v1\x00negate_v1\x00complex_v1\x00divide_v1\x00custom_call_v1\x00call_v1\x00value\x00sym_name\x00arg_attrs\x00function_type\x00res_attrs\x00sym_visibility\x00iota_dimension\x00broadcast_dimensions\x00compare_type\x00comparison_direction\x00mhlo.num_partitions\x00mhlo.num_replicas\x00jit_cholesky\x00jit(cholesky)/jit(main)/pjit[in_shardings=(UnspecifiedValue,) out_shardings=(UnspecifiedValue,) resource_env=None donated_invars=(False,) name=tril keep_unused=False inline=False]\x00/Users/necula/Source/jax/jax/experimental/jax2tf/tests/back_compat_test.py\x00jit(cholesky)/jit(main)/jit(tril)/iota[dtype=int32 shape=(4, 4) dimension=0]\x00jit(cholesky)/jit(main)/jit(tril)/add\x00jit(cholesky)/jit(main)/jit(tril)/iota[dtype=int32 shape=(4, 4) dimension=1]\x00jit(cholesky)/jit(main)/jit(tril)/ge\x00jit(cholesky)/jit(main)/jit(tril)/broadcast_in_dim[shape=(4, 4) broadcast_dimensions=()]\x00jit(cholesky)/jit(main)/jit(tril)/select_n\x00permutation\x00jit(cholesky)/jit(main)/transpose[permutation=(1, 0)]\x00jit(cholesky)/jit(main)/real\x00jit(cholesky)/jit(main)/imag\x00jit(cholesky)/jit(main)/neg\x00jit(cholesky)/jit(main)/complex\x00jit(cholesky)/jit(main)/add\x00jit(cholesky)/jit(main)/div\x00jit(cholesky)/jit(main)/cholesky\x00api_version\x00backend_config\x00call_target_name\x00called_computations\x00has_side_effect\x00operand_layouts\x00output_operand_aliases\x00result_layouts\x00callee\x00\x00tril\x00jax.arg_info\x00x\x00mhlo.sharding\x00{replicated}\x00jax.result_info\x00main\x00public\x00private\x00lapack_cpotrf\x00",
    xla_call_module_version=6,
)  # End paste


# Pasted from the test output (see back_compat_test.py module docstring)
data_2023_06_19["c128"] = dict(
    testdata_version=1,
    platform='cpu',
    custom_call_targets=['lapack_zpotrf'],
    serialized_date=datetime.date(2023, 6, 19),
    inputs=(array([[ 77.35445791180521 -6.4555004827448569e-16j,
         16.89356598261691 -5.4959586590823566e+00j,
        -21.124380423202325+6.4431220601700787e+01j,
         55.385054340628855+2.5198457006849742e+00j],
       [ 16.89356598261691 +5.4959586590823566e+00j,
         67.125263428637   -3.2921739472953976e-16j,
         25.14078382035968 +1.2783276691803774e+01j,
         51.116221409460884-2.2635508887939348e+00j],
       [-21.124380423202325-6.4431220601700787e+01j,
         25.14078382035968 -1.2783276691803774e+01j,
        107.43449297637208 -2.8959717546347756e-15j,
         12.493792156221616-5.7556567757218694e+01j],
       [ 55.385054340628855-2.5198457006849715e+00j,
         51.116221409460884+2.2635508887939326e+00j,
         12.493792156221616+5.7556567757218708e+01j,
         78.9856503203742  +2.0971925518284437e-16j]]),),
    expected_outputs=(array([[ 8.795138311124232 +0.j                  ,
         0.                +0.j                  ,
         0.                +0.j                  ,
         0.                +0.j                  ],
       [ 1.9207845726825759+0.624885984127274j   ,
         7.940111306576433 +0.j                  ,
         0.                +0.j                  ,
         0.                +0.j                  ],
       [-2.401824698593298 -7.325776846534311j   ,
         4.3238621722485755-0.026813746599595675j,
         5.413152651345813 +0.j                  ,
         0.                +0.j                  ],
       [ 6.297235174866659 -0.28650438589440164j ,
         4.936910868956218 +0.849977768846063j   ,
         0.7751580530200595+1.279980716041562j   ,
         3.451611642915363 +0.j                  ]]),),
    mlir_module_text=r"""
#loc = loc(unknown)
module @jit_cholesky attributes {mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} {
  func.func public @main(%arg0: tensor<4x4xcomplex<f64>> {jax.arg_info = "x", mhlo.sharding = "{replicated}"} loc(unknown)) -> (tensor<4x4xcomplex<f64>> {jax.result_info = ""}) {
    %0 = stablehlo.transpose %arg0, dims = [1, 0] : (tensor<4x4xcomplex<f64>>) -> tensor<4x4xcomplex<f64>> loc(#loc2)
    %1 = stablehlo.real %0 : (tensor<4x4xcomplex<f64>>) -> tensor<4x4xf64> loc(#loc3)
    %2 = stablehlo.imag %0 : (tensor<4x4xcomplex<f64>>) -> tensor<4x4xf64> loc(#loc4)
    %3 = stablehlo.negate %2 : tensor<4x4xf64> loc(#loc5)
    %4 = stablehlo.complex %1, %3 : tensor<4x4xcomplex<f64>> loc(#loc6)
    %5 = stablehlo.add %arg0, %4 : tensor<4x4xcomplex<f64>> loc(#loc7)
    %6 = stablehlo.constant dense<(2.000000e+00,0.000000e+00)> : tensor<complex<f64>> loc(#loc)
    %7 = stablehlo.broadcast_in_dim %6, dims = [] : (tensor<complex<f64>>) -> tensor<4x4xcomplex<f64>> loc(#loc8)
    %8 = stablehlo.divide %5, %7 : tensor<4x4xcomplex<f64>> loc(#loc8)
    %9 = stablehlo.constant dense<1> : tensor<i32> loc(#loc9)
    %10 = stablehlo.constant dense<1> : tensor<i32> loc(#loc9)
    %11 = stablehlo.constant dense<4> : tensor<i32> loc(#loc9)
    %12:2 = stablehlo.custom_call @lapack_zpotrf(%9, %10, %11, %8) {api_version = 2 : i32, operand_layouts = [dense<> : tensor<0xindex>, dense<> : tensor<0xindex>, dense<> : tensor<0xindex>, dense<[0, 1]> : tensor<2xindex>], output_operand_aliases = [#stablehlo.output_operand_alias<output_tuple_indices = [0], operand_index = 3, operand_tuple_indices = []>], result_layouts = [dense<[0, 1]> : tensor<2xindex>, dense<> : tensor<0xindex>]} : (tensor<i32>, tensor<i32>, tensor<i32>, tensor<4x4xcomplex<f64>>) -> (tensor<4x4xcomplex<f64>>, tensor<i32>) loc(#loc9)
    %13 = stablehlo.constant dense<0> : tensor<i32> loc(#loc9)
    %14 = stablehlo.broadcast_in_dim %13, dims = [] : (tensor<i32>) -> tensor<i32> loc(#loc9)
    %15 = stablehlo.compare  EQ, %12#1, %14,  SIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1> loc(#loc9)
    %16 = stablehlo.broadcast_in_dim %15, dims = [] : (tensor<i1>) -> tensor<1x1xi1> loc(#loc9)
    %17 = stablehlo.constant dense<(0x7FF8000000000000,0x7FF8000000000000)> : tensor<complex<f64>> loc(#loc9)
    %18 = stablehlo.broadcast_in_dim %17, dims = [] : (tensor<complex<f64>>) -> tensor<4x4xcomplex<f64>> loc(#loc9)
    %19 = stablehlo.broadcast_in_dim %16, dims = [0, 1] : (tensor<1x1xi1>) -> tensor<4x4xi1> loc(#loc9)
    %20 = stablehlo.select %19, %12#0, %18 : tensor<4x4xi1>, tensor<4x4xcomplex<f64>> loc(#loc9)
    %21 = call @tril(%20) : (tensor<4x4xcomplex<f64>>) -> tensor<4x4xcomplex<f64>> loc(#loc10)
    return %21 : tensor<4x4xcomplex<f64>> loc(#loc)
  } loc(#loc)
  func.func private @tril(%arg0: tensor<4x4xcomplex<f64>> loc(unknown)) -> tensor<4x4xcomplex<f64>> {
    %0 = stablehlo.iota dim = 0 : tensor<4x4xi32> loc(#loc11)
    %1 = stablehlo.constant dense<0> : tensor<i32> loc(#loc10)
    %2 = stablehlo.broadcast_in_dim %1, dims = [] : (tensor<i32>) -> tensor<4x4xi32> loc(#loc12)
    %3 = stablehlo.add %0, %2 : tensor<4x4xi32> loc(#loc12)
    %4 = stablehlo.iota dim = 1 : tensor<4x4xi32> loc(#loc13)
    %5 = stablehlo.compare  GE, %3, %4,  SIGNED : (tensor<4x4xi32>, tensor<4x4xi32>) -> tensor<4x4xi1> loc(#loc14)
    %6 = stablehlo.constant dense<(0.000000e+00,0.000000e+00)> : tensor<complex<f64>> loc(#loc10)
    %7 = stablehlo.broadcast_in_dim %6, dims = [] : (tensor<complex<f64>>) -> tensor<4x4xcomplex<f64>> loc(#loc15)
    %8 = stablehlo.select %5, %arg0, %7 : tensor<4x4xi1>, tensor<4x4xcomplex<f64>> loc(#loc16)
    return %8 : tensor<4x4xcomplex<f64>> loc(#loc10)
  } loc(#loc10)
} loc(#loc)
#loc1 = loc("/Users/necula/Source/jax/jax/experimental/jax2tf/tests/back_compat_test.py":292:0)
#loc2 = loc("jit(cholesky)/jit(main)/transpose[permutation=(1, 0)]"(#loc1))
#loc3 = loc("jit(cholesky)/jit(main)/real"(#loc1))
#loc4 = loc("jit(cholesky)/jit(main)/imag"(#loc1))
#loc5 = loc("jit(cholesky)/jit(main)/neg"(#loc1))
#loc6 = loc("jit(cholesky)/jit(main)/complex"(#loc1))
#loc7 = loc("jit(cholesky)/jit(main)/add"(#loc1))
#loc8 = loc("jit(cholesky)/jit(main)/div"(#loc1))
#loc9 = loc("jit(cholesky)/jit(main)/cholesky"(#loc1))
#loc10 = loc("jit(cholesky)/jit(main)/pjit[in_shardings=(UnspecifiedValue,) out_shardings=(UnspecifiedValue,) resource_env=None donated_invars=(False,) name=tril keep_unused=False inline=False]"(#loc1))
#loc11 = loc("jit(cholesky)/jit(main)/jit(tril)/iota[dtype=int32 shape=(4, 4) dimension=0]"(#loc1))
#loc12 = loc("jit(cholesky)/jit(main)/jit(tril)/add"(#loc1))
#loc13 = loc("jit(cholesky)/jit(main)/jit(tril)/iota[dtype=int32 shape=(4, 4) dimension=1]"(#loc1))
#loc14 = loc("jit(cholesky)/jit(main)/jit(tril)/ge"(#loc1))
#loc15 = loc("jit(cholesky)/jit(main)/jit(tril)/broadcast_in_dim[shape=(4, 4) broadcast_dimensions=()]"(#loc1))
#loc16 = loc("jit(cholesky)/jit(main)/jit(tril)/select_n"(#loc1))
""",
    mlir_module_serialized=b"ML\xefR\x01StableHLO_v0.9.0\x00\x011\x05\x01\x03\x01\x03\x05\x03!\x07\t\x0b\r\x0f\x11\x13\x15\x17\x19\x1b\x1d\x1f!#%\x03J\x02\xe9)\x01\x97\x17\x0f\x07\x0b\x13\x0f\x0b\x0f\x0b\x0b\x0b\x0b\x0b\x13\x0b\x0f\x0b\x0b\x0f\x13#\x0b\x0b\x0b33\x0b\x0b\x13\x0f\x0b\x0b\x13\x0f\x0b\x1b\x0f\x0b\x13\x0f\x0b\x0f\x0b\x13\x0b\x0f\x0b\x0f\x0b\x0f\x0b\x0f\x0b\x0f\x0b\x0f\x0b\x13\x0b\x0b\x13K\x0b\x0b\x0b\x0b\x0b\x0b\x0b\x0b\x1b\x13\x13\x13\x0b\x03S\x0f\x0b\x0b\x0f\x0b\x0bO\x0f\x1b\x0b\x0b\x0b\x0b\x0f\x13\x0b\x0b\x0b\x0b\x0b\x0f\x1f\x0f\x0f\x0bOOO\x1f\x1f\x0b\x0b\x0b\x0b\x1b\x0f\x17\x13\x0bOO\x01\x03\x0f\x03'\x17\x0f\x0f\x17\x07\x07\x17\x0b\x07\x07\x17\x13\x07\x17\x13\x13\x13\x0f\x17\x02F\x08\x177\x92\x04\x01\x1dw\x01\x1f\x05'\x03\x03\x1d\xc3\x1d5\x01\x05)\x11\x01\x05\x05+\x05-\x05/\x051\x053\x03\x03\x07\xc1\x055\x1d?\x01\x057\x059\x1du\x01\x03\x03\x07\xcf\x03\x07+\x0f-\x0f\r/\x05;\x05=\x05?\x03\x0b\x11\xa5\x13\x99\x15\xb1\r\xb7\x17\xb9\x03\x0b\x11\x9d\x13\x99\x15\x9d\r\x9f\x17\xbd\x05A\x05C\x03\x03\x19\xbf\x1d=\x01\x05E\x05G\x03\x03\x19\xc5\x1dE\x01\x05I\x03\x05!\xa1#\xc7\x1dK\x01\x05K\x03\x03\x07\xc9\x1dQ\x01\x05M\x1dU\x01\x05O\x03\x03Y\xcb\x05Q\x1d]\x01\x05S\x1da\x01\x05U\x1de\x01\x05W\x1di\x01\x05Y\x1dm\x01\x05[\x1dq\x01\x05]\x03\x03\x07\xcd\x05_\x05a\x03\x03\x07\xd1\x03\x11}\xd3\x7f\x9b\x81\xd5\x83\xd7\x85\xd9\x87\xdb\x89\xdd\x8b\xe1\x05c\x05e\x05g\x05i\x05k\x05m\x05o\x05q\x03\x05!\xa1#\xe3\x03\x03\x07\xe5\x03\x03\x1d\xe7\x03\x03\x95\x9f\x05s\x1f!\x01#\x1d\x1du\x03\x03\xbb\x1dw\t\x07\x1f#!\x00\x00\x00\x00\x00\x00\x00\x00\x01\x00\x00\x00\x00\x00\x00\x00\x03\x03\xa7\r\x05\xa9\xab\xad\xaf\x1dy\x1d{\x1d}\x1d\x7f\x03\x03\xb3\r\x03\xb5\x9b\x1d\x81\x1d\x83\x1d\x85\r\x01\x1d\x87\x13\x0b\x01\x1f\x05\t\x00\x00\x00\x00\x1f\x1f\x01\x13\x0b\x05\x07\x05\x1f\x07!\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x1f\x19!\x01\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x1f\x07!\x00\x00\x00\x00\x00\x00\x00@\x00\x00\x00\x00\x00\x00\x00\x00\x1f\x05\t\x01\x00\x00\x00\x1f\x05\t\x04\x00\x00\x00\x0b\x05\x1d\x89\x03\x01\x05\x01\x03\t\x97\x97\x97\xa3\x03\x03\xdf\x15\x03\x01\r\x01\x03\x05\xa3\x97\x07\x01\x1f\x07!\x00\x00\x00\x00\x00\x00\xf8\x7f\x00\x00\x00\x00\x00\x00\xf8\x7f\x1f\x19!\x00\x00\x00\x00\x00\x00\x00\x00\x01\x00\x00\x00\x00\x00\x00\x00\x01\x02\x02)\x05\x11\x11\x11)\x01\x15)\x01\x11)\x05\x11\x11\x15\x1d\x01)\x05\x11\x11\x13\x03\x13\x0b\x1b)\x05\x11\x11\r)\x03\t\x0b\x13\x11\x03\x03\x03\x03)\x03\x01\x0b)\x03\x01\x1b)\x03\t\x1b)\x01\r)\x05\x05\x05\r\x04J\x04\x05\x01\x11\x05)\x07\x03\x01\t\x07\x11\x051\x05\x031_\x03\x03\x05\x13\x07[W\x03\x03\x03\x01\x15\x06_\x03\x0f\x03\x03\x17\x06c\x03\x0f\x03\x03\x19\x06g\x03\x0f\x03\x07\x1b\x06k\x03\x03\x05\x05\t\x0b\x06o\x03\x03\x05\x01\x0b\x03\x03\x05s\x03\x07\x05\x07%\t\x03\x03\x03\x0f\x1d\x06%\x03\x03\x05\r\x11\x03\x03\x03'\x03\x05\x03\x03\x03'\x03\x05\x03\x03\x03y\x03\x05\x1f\x07\x03{\x05\x03\x05\t\x15\x17\x19\x13\x03\x03\x03\x1b\x03\x05\x05\x07\x03\t\x03\x05\x03\x1f\r\x07\x03\x8d\x03%\x05\x1d!\x05\x07\x03\t\x03'\x03#\x03\x03\x03\x8f\x03\x07\x05\x07\x03\t\x03\x03\x03'\x05\x07\x03\x91\x03\x17\x03%\x0f\x06\x03\x03\x03\x07+\x1b)!\x07\x0b\x93\x03\x03\x03-\x11\x04\x05\x03/\x07\x11\x0b3\x05\x03\x15+\x03\x03\x05\t\x03;9\x03\t\x03\x03\x0b\x1b\x03\x05\x05\x07\x1f\t\x03\t\x03\x05\x0b\x06\x1f\x03\t\x05\x03\x07\t\x03CA\x03\t\r\x07IG\x03\x17\x05\t\x0b\x03\x03\x0bM\x03\x07\x05\x07O\t\x03\x03\x03\x0f\x0f\x06S\x03\x03\x07\r\x01\x11\x11\x04\x0b\x03\x13\x06\x03\x01\x05\x01\x00\x96\x18\x8b\x1d\x11\x0f\x0b!\x1b\x1d\x05\x1b\x0b\x03\x0f\x1f/!!)#\x1f\x19C99A9;;m\x19W\xb3K\x9bM\x9b\x97\xd2\x02\x1b%)+\x1b+\x1f\x1f\x15\x1d\x15\x13\r\x11\x1f\x15\x17\x15\x11\x11\x1b\x15\x15\x17\x0f\x11\x11)\x19\x0f\x0b\x11builtin\x00vhlo\x00module\x00constant_v1\x00broadcast_in_dim_v1\x00func_v1\x00iota_v1\x00add_v1\x00compare_v1\x00select_v1\x00return_v1\x00transpose_v1\x00real_v1\x00imag_v1\x00negate_v1\x00complex_v1\x00divide_v1\x00custom_call_v1\x00call_v1\x00value\x00sym_name\x00arg_attrs\x00function_type\x00res_attrs\x00sym_visibility\x00iota_dimension\x00broadcast_dimensions\x00compare_type\x00comparison_direction\x00mhlo.num_partitions\x00mhlo.num_replicas\x00jit_cholesky\x00jit(cholesky)/jit(main)/pjit[in_shardings=(UnspecifiedValue,) out_shardings=(UnspecifiedValue,) resource_env=None donated_invars=(False,) name=tril keep_unused=False inline=False]\x00/Users/necula/Source/jax/jax/experimental/jax2tf/tests/back_compat_test.py\x00jit(cholesky)/jit(main)/jit(tril)/iota[dtype=int32 shape=(4, 4) dimension=0]\x00jit(cholesky)/jit(main)/jit(tril)/add\x00jit(cholesky)/jit(main)/jit(tril)/iota[dtype=int32 shape=(4, 4) dimension=1]\x00jit(cholesky)/jit(main)/jit(tril)/ge\x00jit(cholesky)/jit(main)/jit(tril)/broadcast_in_dim[shape=(4, 4) broadcast_dimensions=()]\x00jit(cholesky)/jit(main)/jit(tril)/select_n\x00permutation\x00jit(cholesky)/jit(main)/transpose[permutation=(1, 0)]\x00jit(cholesky)/jit(main)/real\x00jit(cholesky)/jit(main)/imag\x00jit(cholesky)/jit(main)/neg\x00jit(cholesky)/jit(main)/complex\x00jit(cholesky)/jit(main)/add\x00jit(cholesky)/jit(main)/div\x00jit(cholesky)/jit(main)/cholesky\x00api_version\x00backend_config\x00call_target_name\x00called_computations\x00has_side_effect\x00operand_layouts\x00output_operand_aliases\x00result_layouts\x00callee\x00\x00tril\x00jax.arg_info\x00x\x00mhlo.sharding\x00{replicated}\x00jax.result_info\x00main\x00public\x00private\x00lapack_zpotrf\x00",
    xla_call_module_version=6,
)  # End paste

data_2024_05_31 = {}


# Pasted from the test output (see export_back_compat_test_util.py module docstring)
data_2024_05_31["c128"] = dict(
    testdata_version=1,
    platform='cpu',
    custom_call_targets=['lapack_zpotrf_ffi'],
    serialized_date=datetime.date(2024, 5, 31),
    inputs=(array([[ 77.35445791180521 -6.4555004827448569e-16j,
         16.89356598261691 -5.4959586590823566e+00j,
        -21.124380423202325+6.4431220601700787e+01j,
         55.385054340628855+2.5198457006849742e+00j],
       [ 16.89356598261691 +5.4959586590823566e+00j,
         67.125263428637   -3.2921739472953976e-16j,
         25.14078382035968 +1.2783276691803774e+01j,
         51.116221409460884-2.2635508887939348e+00j],
       [-21.124380423202325-6.4431220601700787e+01j,
         25.14078382035968 -1.2783276691803774e+01j,
        107.43449297637208 -2.8959717546347756e-15j,
         12.493792156221616-5.7556567757218694e+01j],
       [ 55.385054340628855-2.5198457006849715e+00j,
         51.116221409460884+2.2635508887939326e+00j,
         12.493792156221616+5.7556567757218708e+01j,
         78.9856503203742  +2.0971925518284437e-16j]]),),
    expected_outputs=(array([[ 8.795138311124232 +0.j                  ,
         0.                +0.j                  ,
         0.                +0.j                  ,
         0.                +0.j                  ],
       [ 1.920784572682576 +0.6248859841272741j  ,
         7.940111306576432 +0.j                  ,
         0.                +0.j                  ,
         0.                +0.j                  ],
       [-2.4018246985932983-7.325776846534312j   ,
         4.323862172248577 -0.026813746599595487j,
         5.413152651345812 +0.j                  ,
         0.                +0.j                  ],
       [ 6.2972351748666595-0.2865043858944017j  ,
         4.936910868956218 +0.8499777688460634j  ,
         0.775158053020059 +1.2799807160415595j  ,
         3.451611642915358 +0.j                  ]]),),
    mlir_module_text=r"""
#loc1 = loc("x")
#loc2 = loc("third_party/py/jax/tests/export_back_compat_test.py":182:4)
#loc11 = loc("jit(cholesky)/jit(main)/pjit[in_shardings=(UnspecifiedValue,) out_shardings=(UnspecifiedValue,) in_layouts=(None,) out_layouts=(None,) resource_env=None donated_invars=(False,) name=tril keep_unused=False inline=False]"(#loc2))
module @jit_cholesky attributes {jax.uses_shape_polymorphism = false, mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} {
  func.func public @main(%arg0: tensor<4x4xcomplex<f64>> {mhlo.layout_mode = "default"} loc("x")) -> (tensor<4x4xcomplex<f64>> {jax.result_info = "", mhlo.layout_mode = "default"}) {
    %0 = stablehlo.transpose %arg0, dims = [1, 0] : (tensor<4x4xcomplex<f64>>) -> tensor<4x4xcomplex<f64>> loc(#loc3)
    %1 = stablehlo.real %0 : (tensor<4x4xcomplex<f64>>) -> tensor<4x4xf64> loc(#loc4)
    %2 = stablehlo.imag %0 : (tensor<4x4xcomplex<f64>>) -> tensor<4x4xf64> loc(#loc5)
    %3 = stablehlo.negate %2 : tensor<4x4xf64> loc(#loc6)
    %4 = stablehlo.complex %1, %3 : tensor<4x4xcomplex<f64>> loc(#loc7)
    %5 = stablehlo.add %arg0, %4 : tensor<4x4xcomplex<f64>> loc(#loc8)
    %cst = stablehlo.constant dense<(2.000000e+00,0.000000e+00)> : tensor<complex<f64>> loc(#loc)
    %6 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<complex<f64>>) -> tensor<4x4xcomplex<f64>> loc(#loc9)
    %7 = stablehlo.divide %5, %6 : tensor<4x4xcomplex<f64>> loc(#loc9)
    %c = stablehlo.constant dense<4> : tensor<i64> loc(#loc10)
    %c_0 = stablehlo.constant dense<4> : tensor<i64> loc(#loc10)
    %8:2 = stablehlo.custom_call @lapack_zpotrf_ffi(%7) {mhlo.backend_config = {uplo = 76 : ui8}, operand_layouts = [dense<[0, 1]> : tensor<2xindex>], output_operand_aliases = [#stablehlo.output_operand_alias<output_tuple_indices = [0], operand_index = 0, operand_tuple_indices = []>], result_layouts = [dense<[0, 1]> : tensor<2xindex>, dense<> : tensor<0xindex>]} : (tensor<4x4xcomplex<f64>>) -> (tensor<4x4xcomplex<f64>>, tensor<i32>) loc(#loc10)
    %c_1 = stablehlo.constant dense<0> : tensor<i32> loc(#loc10)
    %9 = stablehlo.broadcast_in_dim %c_1, dims = [] : (tensor<i32>) -> tensor<i32> loc(#loc10)
    %10 = stablehlo.compare  EQ, %8#1, %9,  SIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1> loc(#loc10)
    %11 = stablehlo.broadcast_in_dim %10, dims = [] : (tensor<i1>) -> tensor<1x1xi1> loc(#loc10)
    %cst_2 = stablehlo.constant dense<(0x7FF8000000000000,0x7FF8000000000000)> : tensor<complex<f64>> loc(#loc10)
    %12 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<complex<f64>>) -> tensor<4x4xcomplex<f64>> loc(#loc10)
    %13 = stablehlo.broadcast_in_dim %11, dims = [0, 1] : (tensor<1x1xi1>) -> tensor<4x4xi1> loc(#loc10)
    %14 = stablehlo.select %13, %8#0, %12 : tensor<4x4xi1>, tensor<4x4xcomplex<f64>> loc(#loc10)
    %15 = call @tril(%14) : (tensor<4x4xcomplex<f64>>) -> tensor<4x4xcomplex<f64>> loc(#loc11)
    return %15 : tensor<4x4xcomplex<f64>> loc(#loc)
  } loc(#loc)
  func.func private @tril(%arg0: tensor<4x4xcomplex<f64>> {mhlo.layout_mode = "default"} loc("jit(cholesky)/jit(main)/pjit[in_shardings=(UnspecifiedValue,) out_shardings=(UnspecifiedValue,) in_layouts=(None,) out_layouts=(None,) resource_env=None donated_invars=(False,) name=tril keep_unused=False inline=False]"(#loc2))) -> (tensor<4x4xcomplex<f64>> {mhlo.layout_mode = "default"}) {
    %0 = stablehlo.iota dim = 0 : tensor<4x4xi32> loc(#loc12)
    %c = stablehlo.constant dense<0> : tensor<i32> loc(#loc11)
    %1 = stablehlo.broadcast_in_dim %c, dims = [] : (tensor<i32>) -> tensor<4x4xi32> loc(#loc13)
    %2 = stablehlo.add %0, %1 : tensor<4x4xi32> loc(#loc13)
    %3 = stablehlo.iota dim = 1 : tensor<4x4xi32> loc(#loc14)
    %4 = stablehlo.compare  GE, %2, %3,  SIGNED : (tensor<4x4xi32>, tensor<4x4xi32>) -> tensor<4x4xi1> loc(#loc15)
    %cst = stablehlo.constant dense<(0.000000e+00,0.000000e+00)> : tensor<complex<f64>> loc(#loc11)
    %5 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<complex<f64>>) -> tensor<4x4xcomplex<f64>> loc(#loc16)
    %6 = stablehlo.select %4, %arg0, %5 : tensor<4x4xi1>, tensor<4x4xcomplex<f64>> loc(#loc17)
    return %6 : tensor<4x4xcomplex<f64>> loc(#loc11)
  } loc(#loc11)
} loc(#loc)
#loc = loc(unknown)
#loc3 = loc("jit(cholesky)/jit(main)/transpose[permutation=(1, 0)]"(#loc2))
#loc4 = loc("jit(cholesky)/jit(main)/real"(#loc2))
#loc5 = loc("jit(cholesky)/jit(main)/imag"(#loc2))
#loc6 = loc("jit(cholesky)/jit(main)/neg"(#loc2))
#loc7 = loc("jit(cholesky)/jit(main)/complex"(#loc2))
#loc8 = loc("jit(cholesky)/jit(main)/add"(#loc2))
#loc9 = loc("jit(cholesky)/jit(main)/div"(#loc2))
#loc10 = loc("jit(cholesky)/jit(main)/cholesky"(#loc2))
#loc12 = loc("jit(cholesky)/jit(main)/jit(tril)/iota[dtype=int32 shape=(4, 4) dimension=0]"(#loc2))
#loc13 = loc("jit(cholesky)/jit(main)/jit(tril)/add"(#loc2))
#loc14 = loc("jit(cholesky)/jit(main)/jit(tril)/iota[dtype=int32 shape=(4, 4) dimension=1]"(#loc2))
#loc15 = loc("jit(cholesky)/jit(main)/jit(tril)/ge"(#loc2))
#loc16 = loc("jit(cholesky)/jit(main)/jit(tril)/broadcast_in_dim[shape=(4, 4) broadcast_dimensions=()]"(#loc2))
#loc17 = loc("jit(cholesky)/jit(main)/jit(tril)/select_n"(#loc2))
""",
    mlir_module_serialized=b"ML\xefR\x01StableHLO_v0.9.0\x00\x011\x05\x01\x03\x01\x03\x05\x03!\x07\t\x0b\r\x0f\x11\x13\x15\x17\x19\x1b\x1d\x1f!#%\x03^\x02\xed/\x01\x9f\x17\x0f\x0f\x13\x07\x0b\x0b\x0f\x0b\x0b\x0b\x0b\x0b\x13\x0b\x0f\x0b\x0b\x0f\x13+\x0b\x0f\x0b\x0b\x0b33\x0b\x0b\x13\x0f\x0b\x0b\x13\x0f\x0b\x1b\x0f\x0b\x13\x0f\x0b\x0f\x0b\x0f\x0b\x13\x0b\x0f\x0b\x0f\x0b\x0f\x0b\x0f\x0b\x0f\x0b\x0f\x0b\x13\x0b\x0bS\x0b\x0b\x0b\x0b\x0b\x0b\x0b\x0b\x0b\x1b\x13\x13\x13\x0b\x03O\x0f\x0b\x0b\x0b\x0b\x0b\x0bO\x13\x0f\x1b\x0b\x0b\x0b\x0b\x0f\x1f\x0f\x0f\x0bOOO/\x0b\x0b\x0b\x0b\x13\x0b\x0f\x0f\x0f\x17\x13\x0f\x0bOO\x01\x05\x0b\x0f\x03+\x17\x0f\x07\x0f\x17\x07\x17\x0f\x0b\x07\x07\x17\x13\x07\x17\x13\x07\x13\x13\x0f\x17\x02v\x08\x17;\xda\x02\t\x1d\x7f\x01\x1d9\x01\x03\x03\x1d\xc1\x1f\x05'\x05)\x11\x03\x05\x05+\x05-\x05/\x051\x053\x03\x03\x0b\xbf\x055\x1dC\x01\x057\x059\x1d}\x01\x03\x03\x0b\xcd\x03\t+-/\x0f1\x0f\r3\x05;\x11\x01\x00\x05=\x05?\x05A\x03\x0b\x11\x9f\x13\xa5\x15\xb1\r\xb7\x17\xb9\x03\x0b\x11\x9f\x13\xa5\x15\x9f\r\xa9\x17\xbb\x05C\x05E\x03\x03\x19\xbd\x1dA\x01\x05G\x05I\x03\x03\x19\xc3\x1dI\x01\x05K\x03\x05!\xab#\xc5\x1dO\x01\x05M\x03\x03\x0b\xc7\x1dU\x01\x05O\x1dY\x01\x05Q\x1d]\t\x05S\x03\x03a\xc9\x05U\x1de\x01\x05W\x1di\x01\x05Y\x1dm\x01\x05[\x1dq\x01\x05]\x1du\x01\x05_\x1dy\x01\x05a\x03\x03\x0b\xcb\x05c\x05e\x03\x13\x83\xcf\x85\xa7\x87\xd1\x89\xd3\x8b\xd5\x8d\xd7\x8f\xdd\x91\xdf\x93\xe3\x05g\x05i\x05k\x05m\x05o\x05q\x05s\x05u\x05w\x03\x05!\xab#\xe7\x03\x03\x0b\xe9\x03\x03\x1d\xeb\x03\x03\x9d\xa9\x05y\x03\x03\xaf\x1d{\x1d}#!\x1d\x7f\x1d\x81\t\x07\x1f'!\x00\x00\x00\x00\x00\x00\x00\x00\x01\x00\x00\x00\x00\x00\x00\x00\r\x03\xa1\xa3\x03\x03\xb3\r\x05\xb5\xa7\xa1\xa3\x1d\x83\x1d\x85\x1d\x87\x1d\x89\x13\t\x01\x1f\x0b\t\x00\x00\x00\x00\x1f#\x01\x13\t\x05\x07\x05\x1f\x07!\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x1f\x1d!\x01\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x1f\x07!\x00\x00\x00\x00\x00\x00\x00@\x00\x00\x00\x00\x00\x00\x00\x00\x1f\x13\x11\x04\x00\x00\x00\x00\x00\x00\x00\x0b\x03\x1d\x8b\x03\x01\x05\x01\r\x03\xd9\xdb\x1d\x8d\x13%L\x03\x03\xad\x03\x03\xe1\x15\x03\x01\x01\x01\x03\x05\xad\xe5\x1f)\x01\x07\x01\x1f\x07!\x00\x00\x00\x00\x00\x00\xf8\x7f\x00\x00\x00\x00\x00\x00\xf8\x7f\x1f\x1d!\x00\x00\x00\x00\x00\x00\x00\x00\x01\x00\x00\x00\x00\x00\x00\x00\x01\t\x01\x02\x02)\x05\x11\x11\x15)\x01\x15\x1d)\x01\x19)\x05\x11\x11\x19\x01)\x05\x11\x11\x17)\x01\t\x03\x17\x0b\x1b)\x05\x11\x11\x0f)\x03\t\t\x13\x11\x03\x05\x03\x05)\x03\x01\t!)\x03\t\x1f)\x03\x01\x1f)\x01\x0f)\x05\x05\x05\x0f\x04&\x04\x05\x01\x11\t)\x07\x03\x01\t\x07\x11\t5\x07\x03/[\x03\x05[\x13\x07c_\x03\x05\x03\x01\x15\x06g\x03\x11\x03\x03\x17\x06k\x03\x11\x03\x03\x19\x06o\x03\x11\x03\x07\x1b\x06s\x03\x05\x05\x05\t\x0b\x06w\x03\x05\x05\x01\x0b\x03\x03\t{\x03\x07\x05\x07%\x07\x03\x05\x03\x0f\x1d\x06%\x03\x05\x05\r\x11\x03\x03\x03'\x03\x13\x03\x03\x03'\x03\x13\x1f\x07\x03\x81\x05\x05\x0b\x03\x13\x03\x03\x03\x1b\x03\x0b\x05\x07\x03\x07\x03\x0b\x03\x1d\r\x07\x03\x95\x03+\x05\x1b\x1f\x05\x07\x03\x07\x03-\x03!\x03\x03\x03\x97\x03\x07\x05\x07\x03\x07\x03\x05\x03%\x05\x07\x03\x99\x03\x1b\x03#\x0f\x06\x03\x03\x05\x07)\x19'!\x07\x05\x9b\x03\x05\x03+\x11\x04\t\x03-\x07\x11\x057\x07\x03\x15+\x03\x05\x05\t\x03?=\x03\r\x03\x03\x05\x1b\x03\x0b\x05\x07\x1f\x07\x03\r\x03\x05\x0b\x06\x1f\x03\r\x05\x03\x07\t\x03GE\x03\r\r\x07MK\x03\x1b\x05\t\x0b\x03\x03\x05Q\x03\x07\x05\x07S\x07\x03\x05\x03\x0f\x0f\x06W\x03\x05\x07\r\x01\x11\x11\x04\x05\x03\x13\x06\x03\x01\x05\x01\x00\x86\x19\x8f\x0b%\x11\x0f\x0b!\x0b\x03\x11#\x0f\x1f/!)!)#\x1f\x19C99A9;;m\x19\x05W\xb3K\x9bM\x9bin\x03\x1b%)9+\x1b+\x1f\x1f\x15\x1d\x15\x13\r\x11\x1f\x15\x17\x15\x11\x11\x1b\x15\x15\x17\x0f\x11\x11)\x19\x0f\x0b\x11builtin\x00vhlo\x00module\x00constant_v1\x00broadcast_in_dim_v1\x00func_v1\x00iota_v1\x00add_v1\x00compare_v1\x00select_v1\x00return_v1\x00transpose_v1\x00real_v1\x00imag_v1\x00negate_v1\x00complex_v1\x00divide_v1\x00custom_call_v1\x00call_v1\x00value\x00sym_name\x00arg_attrs\x00function_type\x00res_attrs\x00sym_visibility\x00iota_dimension\x00broadcast_dimensions\x00compare_type\x00comparison_direction\x00jax.uses_shape_polymorphism\x00mhlo.num_partitions\x00mhlo.num_replicas\x00jit_cholesky\x00jit(cholesky)/jit(main)/pjit[in_shardings=(UnspecifiedValue,) out_shardings=(UnspecifiedValue,) in_layouts=(None,) out_layouts=(None,) resource_env=None donated_invars=(False,) name=tril keep_unused=False inline=False]\x00third_party/py/jax/tests/export_back_compat_test.py\x00jit(cholesky)/jit(main)/jit(tril)/iota[dtype=int32 shape=(4, 4) dimension=0]\x00jit(cholesky)/jit(main)/jit(tril)/add\x00jit(cholesky)/jit(main)/jit(tril)/iota[dtype=int32 shape=(4, 4) dimension=1]\x00jit(cholesky)/jit(main)/jit(tril)/ge\x00jit(cholesky)/jit(main)/jit(tril)/broadcast_in_dim[shape=(4, 4) broadcast_dimensions=()]\x00jit(cholesky)/jit(main)/jit(tril)/select_n\x00x\x00permutation\x00jit(cholesky)/jit(main)/transpose[permutation=(1, 0)]\x00jit(cholesky)/jit(main)/real\x00jit(cholesky)/jit(main)/imag\x00jit(cholesky)/jit(main)/neg\x00jit(cholesky)/jit(main)/complex\x00jit(cholesky)/jit(main)/add\x00jit(cholesky)/jit(main)/div\x00jit(cholesky)/jit(main)/cholesky\x00api_version\x00backend_config\x00call_target_name\x00called_computations\x00has_side_effect\x00mhlo.backend_config\x00operand_layouts\x00output_operand_aliases\x00result_layouts\x00callee\x00mhlo.layout_mode\x00default\x00\x00tril\x00jax.result_info\x00main\x00public\x00private\x00lapack_zpotrf_ffi\x00uplo\x00",
    xla_call_module_version=9,
    nr_devices=1,
)  # End paste


# Pasted from the test output (see export_back_compat_test_util.py module docstring)
data_2024_05_31["c64"] = dict(
    testdata_version=1,
    platform='cpu',
    custom_call_targets=['lapack_cpotrf_ffi'],
    serialized_date=datetime.date(2024, 5, 31),
    inputs=(array([[ 38.089394  +0.j      ,   3.3509152+31.34555j ,
         -0.5972495-38.030815j, -19.042051 +12.27706j ],
       [  3.3509152-31.34555j ,  73.87575   +0.j      ,
        -12.427277 -12.337961j,  41.542507  -9.639935j],
       [ -0.5972495+38.030815j, -12.427277 +12.337961j,
         73.04141   +0.j      ,   8.193127 -26.056503j],
       [-19.042051 -12.27706j ,  41.542507  +9.639935j,
          8.193127 +26.056503j,  52.977036  +0.j      ]], dtype=complex64),),
    expected_outputs=(array([[ 6.1716604 +0.j       ,  0.        +0.j       ,
         0.        +0.j       ,  0.        +0.j       ],
       [ 0.542952  -5.0789495j,  6.912686  +0.j       ,
         0.        +0.j       ,  0.        +0.j       ],
       [-0.0967729 +6.162169j ,  2.7373748 +1.3719275j,
         5.0679693 +0.j       ,  0.        +0.j       ],
       [-3.0854018 -1.9892637j,  4.790376  +3.8177073j,
         0.35557628+0.5865825j,  1.227622  +0.j       ]], dtype=complex64),),
    mlir_module_text=r"""
#loc1 = loc("x")
#loc2 = loc("third_party/py/jax/tests/export_back_compat_test.py":182:4)
#loc11 = loc("jit(cholesky)/jit(main)/pjit[in_shardings=(UnspecifiedValue,) out_shardings=(UnspecifiedValue,) in_layouts=(None,) out_layouts=(None,) resource_env=None donated_invars=(False,) name=tril keep_unused=False inline=False]"(#loc2))
module @jit_cholesky attributes {jax.uses_shape_polymorphism = false, mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} {
  func.func public @main(%arg0: tensor<4x4xcomplex<f32>> {mhlo.layout_mode = "default"} loc("x")) -> (tensor<4x4xcomplex<f32>> {jax.result_info = "", mhlo.layout_mode = "default"}) {
    %0 = stablehlo.transpose %arg0, dims = [1, 0] : (tensor<4x4xcomplex<f32>>) -> tensor<4x4xcomplex<f32>> loc(#loc3)
    %1 = stablehlo.real %0 : (tensor<4x4xcomplex<f32>>) -> tensor<4x4xf32> loc(#loc4)
    %2 = stablehlo.imag %0 : (tensor<4x4xcomplex<f32>>) -> tensor<4x4xf32> loc(#loc5)
    %3 = stablehlo.negate %2 : tensor<4x4xf32> loc(#loc6)
    %4 = stablehlo.complex %1, %3 : tensor<4x4xcomplex<f32>> loc(#loc7)
    %5 = stablehlo.add %arg0, %4 : tensor<4x4xcomplex<f32>> loc(#loc8)
    %cst = stablehlo.constant dense<(2.000000e+00,0.000000e+00)> : tensor<complex<f32>> loc(#loc)
    %6 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<complex<f32>>) -> tensor<4x4xcomplex<f32>> loc(#loc9)
    %7 = stablehlo.divide %5, %6 : tensor<4x4xcomplex<f32>> loc(#loc9)
    %c = stablehlo.constant dense<4> : tensor<i64> loc(#loc10)
    %c_0 = stablehlo.constant dense<4> : tensor<i64> loc(#loc10)
    %8:2 = stablehlo.custom_call @lapack_cpotrf_ffi(%7) {mhlo.backend_config = {uplo = 76 : ui8}, operand_layouts = [dense<[0, 1]> : tensor<2xindex>], output_operand_aliases = [#stablehlo.output_operand_alias<output_tuple_indices = [0], operand_index = 0, operand_tuple_indices = []>], result_layouts = [dense<[0, 1]> : tensor<2xindex>, dense<> : tensor<0xindex>]} : (tensor<4x4xcomplex<f32>>) -> (tensor<4x4xcomplex<f32>>, tensor<i32>) loc(#loc10)
    %c_1 = stablehlo.constant dense<0> : tensor<i32> loc(#loc10)
    %9 = stablehlo.broadcast_in_dim %c_1, dims = [] : (tensor<i32>) -> tensor<i32> loc(#loc10)
    %10 = stablehlo.compare  EQ, %8#1, %9,  SIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1> loc(#loc10)
    %11 = stablehlo.broadcast_in_dim %10, dims = [] : (tensor<i1>) -> tensor<1x1xi1> loc(#loc10)
    %cst_2 = stablehlo.constant dense<(0x7FC00000,0x7FC00000)> : tensor<complex<f32>> loc(#loc10)
    %12 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<complex<f32>>) -> tensor<4x4xcomplex<f32>> loc(#loc10)
    %13 = stablehlo.broadcast_in_dim %11, dims = [0, 1] : (tensor<1x1xi1>) -> tensor<4x4xi1> loc(#loc10)
    %14 = stablehlo.select %13, %8#0, %12 : tensor<4x4xi1>, tensor<4x4xcomplex<f32>> loc(#loc10)
    %15 = call @tril(%14) : (tensor<4x4xcomplex<f32>>) -> tensor<4x4xcomplex<f32>> loc(#loc11)
    return %15 : tensor<4x4xcomplex<f32>> loc(#loc)
  } loc(#loc)
  func.func private @tril(%arg0: tensor<4x4xcomplex<f32>> {mhlo.layout_mode = "default"} loc("jit(cholesky)/jit(main)/pjit[in_shardings=(UnspecifiedValue,) out_shardings=(UnspecifiedValue,) in_layouts=(None,) out_layouts=(None,) resource_env=None donated_invars=(False,) name=tril keep_unused=False inline=False]"(#loc2))) -> (tensor<4x4xcomplex<f32>> {mhlo.layout_mode = "default"}) {
    %0 = stablehlo.iota dim = 0 : tensor<4x4xi32> loc(#loc12)
    %c = stablehlo.constant dense<0> : tensor<i32> loc(#loc11)
    %1 = stablehlo.broadcast_in_dim %c, dims = [] : (tensor<i32>) -> tensor<4x4xi32> loc(#loc13)
    %2 = stablehlo.add %0, %1 : tensor<4x4xi32> loc(#loc13)
    %3 = stablehlo.iota dim = 1 : tensor<4x4xi32> loc(#loc14)
    %4 = stablehlo.compare  GE, %2, %3,  SIGNED : (tensor<4x4xi32>, tensor<4x4xi32>) -> tensor<4x4xi1> loc(#loc15)
    %cst = stablehlo.constant dense<(0.000000e+00,0.000000e+00)> : tensor<complex<f32>> loc(#loc11)
    %5 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<complex<f32>>) -> tensor<4x4xcomplex<f32>> loc(#loc16)
    %6 = stablehlo.select %4, %arg0, %5 : tensor<4x4xi1>, tensor<4x4xcomplex<f32>> loc(#loc17)
    return %6 : tensor<4x4xcomplex<f32>> loc(#loc11)
  } loc(#loc11)
} loc(#loc)
#loc = loc(unknown)
#loc3 = loc("jit(cholesky)/jit(main)/transpose[permutation=(1, 0)]"(#loc2))
#loc4 = loc("jit(cholesky)/jit(main)/real"(#loc2))
#loc5 = loc("jit(cholesky)/jit(main)/imag"(#loc2))
#loc6 = loc("jit(cholesky)/jit(main)/neg"(#loc2))
#loc7 = loc("jit(cholesky)/jit(main)/complex"(#loc2))
#loc8 = loc("jit(cholesky)/jit(main)/add"(#loc2))
#loc9 = loc("jit(cholesky)/jit(main)/div"(#loc2))
#loc10 = loc("jit(cholesky)/jit(main)/cholesky"(#loc2))
#loc12 = loc("jit(cholesky)/jit(main)/jit(tril)/iota[dtype=int32 shape=(4, 4) dimension=0]"(#loc2))
#loc13 = loc("jit(cholesky)/jit(main)/jit(tril)/add"(#loc2))
#loc14 = loc("jit(cholesky)/jit(main)/jit(tril)/iota[dtype=int32 shape=(4, 4) dimension=1]"(#loc2))
#loc15 = loc("jit(cholesky)/jit(main)/jit(tril)/ge"(#loc2))
#loc16 = loc("jit(cholesky)/jit(main)/jit(tril)/broadcast_in_dim[shape=(4, 4) broadcast_dimensions=()]"(#loc2))
#loc17 = loc("jit(cholesky)/jit(main)/jit(tril)/select_n"(#loc2))
""",
    mlir_module_serialized=b"ML\xefR\x01StableHLO_v0.9.0\x00\x011\x05\x01\x03\x01\x03\x05\x03!\x07\t\x0b\r\x0f\x11\x13\x15\x17\x19\x1b\x1d\x1f!#%\x03^\x02\xed/\x01\x9f\x17\x0f\x0f\x13\x07\x0b\x0b\x0f\x0b\x0b\x0b\x0b\x0b\x13\x0b\x0f\x0b\x0b\x0f\x13+\x0b\x0f\x0b\x0b\x0b33\x0b\x0b\x13\x0f\x0b\x0b\x13\x0f\x0b\x1b\x0f\x0b\x13\x0f\x0b\x0f\x0b\x0f\x0b\x13\x0b\x0f\x0b\x0f\x0b\x0f\x0b\x0f\x0b\x0f\x0b\x0f\x0b\x13\x0b\x0bS\x0b\x0b\x0b\x0b\x0b\x0b\x0b\x0b\x0b\x1b\x13\x13\x13\x0b\x03O\x0f\x0b\x0b\x0b\x0b\x0b\x0bO\x13\x0f\x1b\x0b\x0b\x0b\x0b\x0f\x1f\x0f\x0f\x0b/O//\x0b\x0b\x0b\x0b\x13\x0b\x0f\x0f\x0f\x17\x13\x0f\x0b/O\x01\x05\x0b\x0f\x03+\x17\x0f\x07\x0f\x17\x07\x17\x0f\x0b\x07\x07\x17\x13\x07\x17\x13\x07\x13\x13\x0f\x17\x02\x16\x08\x17;\xda\x02\t\x1d\x7f\x01\x1d9\x01\x03\x03\x1d\xc1\x1f\x05'\x05)\x11\x03\x05\x05+\x05-\x05/\x051\x053\x03\x03\x0b\xbf\x055\x1dC\x01\x057\x059\x1d}\x01\x03\x03\x0b\xcd\x03\t+-/\x0f1\x0f\r3\x05;\x11\x01\x00\x05=\x05?\x05A\x03\x0b\x11\x9f\x13\xa5\x15\xb1\r\xb7\x17\xb9\x03\x0b\x11\x9f\x13\xa5\x15\x9f\r\xa9\x17\xbb\x05C\x05E\x03\x03\x19\xbd\x1dA\x01\x05G\x05I\x03\x03\x19\xc3\x1dI\x01\x05K\x03\x05!\xab#\xc5\x1dO\x01\x05M\x03\x03\x0b\xc7\x1dU\x01\x05O\x1dY\x01\x05Q\x1d]\t\x05S\x03\x03a\xc9\x05U\x1de\x01\x05W\x1di\x01\x05Y\x1dm\x01\x05[\x1dq\x01\x05]\x1du\x01\x05_\x1dy\x01\x05a\x03\x03\x0b\xcb\x05c\x05e\x03\x13\x83\xcf\x85\xa7\x87\xd1\x89\xd3\x8b\xd5\x8d\xd7\x8f\xdd\x91\xdf\x93\xe3\x05g\x05i\x05k\x05m\x05o\x05q\x05s\x05u\x05w\x03\x05!\xab#\xe7\x03\x03\x0b\xe9\x03\x03\x1d\xeb\x03\x03\x9d\xa9\x05y\x03\x03\xaf\x1d{\x1d}#!\x1d\x7f\x1d\x81\t\x07\x1f'!\x00\x00\x00\x00\x00\x00\x00\x00\x01\x00\x00\x00\x00\x00\x00\x00\r\x03\xa1\xa3\x03\x03\xb3\r\x05\xb5\xa7\xa1\xa3\x1d\x83\x1d\x85\x1d\x87\x1d\x89\x13\t\x01\x1f\x0b\t\x00\x00\x00\x00\x1f#\x01\x13\t\x05\x07\x05\x1f\x07\x11\x00\x00\x00\x00\x00\x00\x00\x00\x1f\x1d!\x01\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x1f\x07\x11\x00\x00\x00@\x00\x00\x00\x00\x1f\x13\x11\x04\x00\x00\x00\x00\x00\x00\x00\x0b\x03\x1d\x8b\x03\x01\x05\x01\r\x03\xd9\xdb\x1d\x8d\x13%L\x03\x03\xad\x03\x03\xe1\x15\x03\x01\x01\x01\x03\x05\xad\xe5\x1f)\x01\x07\x01\x1f\x07\x11\x00\x00\xc0\x7f\x00\x00\xc0\x7f\x1f\x1d!\x00\x00\x00\x00\x00\x00\x00\x00\x01\x00\x00\x00\x00\x00\x00\x00\x01\t\x01\x02\x02)\x05\x11\x11\x15)\x01\x15\x1d)\x01\x19)\x05\x11\x11\x19\x01)\x05\x11\x11\x17)\x01\t\x03\x17\t\x1b)\x05\x11\x11\x0f)\x03\t\t\x13\x11\x03\x05\x03\x05)\x03\x01\t!)\x03\t\x1f)\x03\x01\x1f)\x01\x0f)\x05\x05\x05\x0f\x04&\x04\x05\x01\x11\t)\x07\x03\x01\t\x07\x11\t5\x07\x03/[\x03\x05[\x13\x07c_\x03\x05\x03\x01\x15\x06g\x03\x11\x03\x03\x17\x06k\x03\x11\x03\x03\x19\x06o\x03\x11\x03\x07\x1b\x06s\x03\x05\x05\x05\t\x0b\x06w\x03\x05\x05\x01\x0b\x03\x03\t{\x03\x07\x05\x07%\x07\x03\x05\x03\x0f\x1d\x06%\x03\x05\x05\r\x11\x03\x03\x03'\x03\x13\x03\x03\x03'\x03\x13\x1f\x07\x03\x81\x05\x05\x0b\x03\x13\x03\x03\x03\x1b\x03\x0b\x05\x07\x03\x07\x03\x0b\x03\x1d\r\x07\x03\x95\x03+\x05\x1b\x1f\x05\x07\x03\x07\x03-\x03!\x03\x03\x03\x97\x03\x07\x05\x07\x03\x07\x03\x05\x03%\x05\x07\x03\x99\x03\x1b\x03#\x0f\x06\x03\x03\x05\x07)\x19'!\x07\x05\x9b\x03\x05\x03+\x11\x04\t\x03-\x07\x11\x057\x07\x03\x15+\x03\x05\x05\t\x03?=\x03\r\x03\x03\x05\x1b\x03\x0b\x05\x07\x1f\x07\x03\r\x03\x05\x0b\x06\x1f\x03\r\x05\x03\x07\t\x03GE\x03\r\r\x07MK\x03\x1b\x05\t\x0b\x03\x03\x05Q\x03\x07\x05\x07S\x07\x03\x05\x03\x0f\x0f\x06W\x03\x05\x07\r\x01\x11\x11\x04\x05\x03\x13\x06\x03\x01\x05\x01\x00\x86\x19\x8f\x0b%\x11\x0f\x0b!\x0b\x03\x11#\x0f\x1f/!)!)#\x1f\x19C99A9;;m\x19\x05W\xb3K\x9bM\x9bin\x03\x1b%)9+\x1b+\x1f\x1f\x15\x1d\x15\x13\r\x11\x1f\x15\x17\x15\x11\x11\x1b\x15\x15\x17\x0f\x11\x11)\x19\x0f\x0b\x11builtin\x00vhlo\x00module\x00constant_v1\x00broadcast_in_dim_v1\x00func_v1\x00iota_v1\x00add_v1\x00compare_v1\x00select_v1\x00return_v1\x00transpose_v1\x00real_v1\x00imag_v1\x00negate_v1\x00complex_v1\x00divide_v1\x00custom_call_v1\x00call_v1\x00value\x00sym_name\x00arg_attrs\x00function_type\x00res_attrs\x00sym_visibility\x00iota_dimension\x00broadcast_dimensions\x00compare_type\x00comparison_direction\x00jax.uses_shape_polymorphism\x00mhlo.num_partitions\x00mhlo.num_replicas\x00jit_cholesky\x00jit(cholesky)/jit(main)/pjit[in_shardings=(UnspecifiedValue,) out_shardings=(UnspecifiedValue,) in_layouts=(None,) out_layouts=(None,) resource_env=None donated_invars=(False,) name=tril keep_unused=False inline=False]\x00third_party/py/jax/tests/export_back_compat_test.py\x00jit(cholesky)/jit(main)/jit(tril)/iota[dtype=int32 shape=(4, 4) dimension=0]\x00jit(cholesky)/jit(main)/jit(tril)/add\x00jit(cholesky)/jit(main)/jit(tril)/iota[dtype=int32 shape=(4, 4) dimension=1]\x00jit(cholesky)/jit(main)/jit(tril)/ge\x00jit(cholesky)/jit(main)/jit(tril)/broadcast_in_dim[shape=(4, 4) broadcast_dimensions=()]\x00jit(cholesky)/jit(main)/jit(tril)/select_n\x00x\x00permutation\x00jit(cholesky)/jit(main)/transpose[permutation=(1, 0)]\x00jit(cholesky)/jit(main)/real\x00jit(cholesky)/jit(main)/imag\x00jit(cholesky)/jit(main)/neg\x00jit(cholesky)/jit(main)/complex\x00jit(cholesky)/jit(main)/add\x00jit(cholesky)/jit(main)/div\x00jit(cholesky)/jit(main)/cholesky\x00api_version\x00backend_config\x00call_target_name\x00called_computations\x00has_side_effect\x00mhlo.backend_config\x00operand_layouts\x00output_operand_aliases\x00result_layouts\x00callee\x00mhlo.layout_mode\x00default\x00\x00tril\x00jax.result_info\x00main\x00public\x00private\x00lapack_cpotrf_ffi\x00uplo\x00",
    xla_call_module_version=9,
    nr_devices=1,
)  # End paste


# Pasted from the test output (see export_back_compat_test_util.py module docstring)
data_2024_05_31["f32"] = dict(
    testdata_version=1,
    platform='cpu',
    custom_call_targets=['lapack_spotrf_ffi'],
    serialized_date=datetime.date(2024, 5, 31),
    inputs=(array([[ 24.343887,  13.603931,  20.50489 ,  12.063957],
       [ 13.603931,  58.879753, -31.84056 ,  16.328014],
       [ 20.50489 , -31.84056 ,  66.89075 ,  -9.92216 ],
       [ 12.063957,  16.328014,  -9.92216 ,  23.640732]], dtype=float32),),
    expected_outputs=(array([[ 4.9339523,  0.       ,  0.       ,  0.       ],
       [ 2.7572076,  7.160835 ,  0.       ,  0.       ],
       [ 4.155875 , -6.0466657,  3.613486 ,  0.       ],
       [ 2.4450898,  1.3387257, -3.3177993,  2.2050598]], dtype=float32),),
    mlir_module_text=r"""
#loc1 = loc("x")
#loc2 = loc("third_party/py/jax/tests/export_back_compat_test.py":182:4)
#loc7 = loc("jit(cholesky)/jit(main)/pjit[in_shardings=(UnspecifiedValue,) out_shardings=(UnspecifiedValue,) in_layouts=(None,) out_layouts=(None,) resource_env=None donated_invars=(False,) name=tril keep_unused=False inline=False]"(#loc2))
module @jit_cholesky attributes {jax.uses_shape_polymorphism = false, mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} {
  func.func public @main(%arg0: tensor<4x4xf32> {mhlo.layout_mode = "default"} loc("x")) -> (tensor<4x4xf32> {jax.result_info = "", mhlo.layout_mode = "default"}) {
    %0 = stablehlo.transpose %arg0, dims = [1, 0] : (tensor<4x4xf32>) -> tensor<4x4xf32> loc(#loc3)
    %1 = stablehlo.add %arg0, %0 : tensor<4x4xf32> loc(#loc4)
    %cst = stablehlo.constant dense<2.000000e+00> : tensor<f32> loc(#loc)
    %2 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<4x4xf32> loc(#loc5)
    %3 = stablehlo.divide %1, %2 : tensor<4x4xf32> loc(#loc5)
    %c = stablehlo.constant dense<4> : tensor<i64> loc(#loc6)
    %c_0 = stablehlo.constant dense<4> : tensor<i64> loc(#loc6)
    %4:2 = stablehlo.custom_call @lapack_spotrf_ffi(%3) {mhlo.backend_config = {uplo = 76 : ui8}, operand_layouts = [dense<[0, 1]> : tensor<2xindex>], output_operand_aliases = [#stablehlo.output_operand_alias<output_tuple_indices = [0], operand_index = 0, operand_tuple_indices = []>], result_layouts = [dense<[0, 1]> : tensor<2xindex>, dense<> : tensor<0xindex>]} : (tensor<4x4xf32>) -> (tensor<4x4xf32>, tensor<i32>) loc(#loc6)
    %c_1 = stablehlo.constant dense<0> : tensor<i32> loc(#loc6)
    %5 = stablehlo.broadcast_in_dim %c_1, dims = [] : (tensor<i32>) -> tensor<i32> loc(#loc6)
    %6 = stablehlo.compare  EQ, %4#1, %5,  SIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1> loc(#loc6)
    %7 = stablehlo.broadcast_in_dim %6, dims = [] : (tensor<i1>) -> tensor<1x1xi1> loc(#loc6)
    %cst_2 = stablehlo.constant dense<0x7FC00000> : tensor<f32> loc(#loc6)
    %8 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<4x4xf32> loc(#loc6)
    %9 = stablehlo.broadcast_in_dim %7, dims = [0, 1] : (tensor<1x1xi1>) -> tensor<4x4xi1> loc(#loc6)
    %10 = stablehlo.select %9, %4#0, %8 : tensor<4x4xi1>, tensor<4x4xf32> loc(#loc6)
    %11 = call @tril(%10) : (tensor<4x4xf32>) -> tensor<4x4xf32> loc(#loc7)
    return %11 : tensor<4x4xf32> loc(#loc)
  } loc(#loc)
  func.func private @tril(%arg0: tensor<4x4xf32> {mhlo.layout_mode = "default"} loc("jit(cholesky)/jit(main)/pjit[in_shardings=(UnspecifiedValue,) out_shardings=(UnspecifiedValue,) in_layouts=(None,) out_layouts=(None,) resource_env=None donated_invars=(False,) name=tril keep_unused=False inline=False]"(#loc2))) -> (tensor<4x4xf32> {mhlo.layout_mode = "default"}) {
    %0 = stablehlo.iota dim = 0 : tensor<4x4xi32> loc(#loc8)
    %c = stablehlo.constant dense<0> : tensor<i32> loc(#loc7)
    %1 = stablehlo.broadcast_in_dim %c, dims = [] : (tensor<i32>) -> tensor<4x4xi32> loc(#loc9)
    %2 = stablehlo.add %0, %1 : tensor<4x4xi32> loc(#loc9)
    %3 = stablehlo.iota dim = 1 : tensor<4x4xi32> loc(#loc10)
    %4 = stablehlo.compare  GE, %2, %3,  SIGNED : (tensor<4x4xi32>, tensor<4x4xi32>) -> tensor<4x4xi1> loc(#loc11)
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32> loc(#loc7)
    %5 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<4x4xf32> loc(#loc12)
    %6 = stablehlo.select %4, %arg0, %5 : tensor<4x4xi1>, tensor<4x4xf32> loc(#loc13)
    return %6 : tensor<4x4xf32> loc(#loc7)
  } loc(#loc7)
} loc(#loc)
#loc = loc(unknown)
#loc3 = loc("jit(cholesky)/jit(main)/transpose[permutation=(1, 0)]"(#loc2))
#loc4 = loc("jit(cholesky)/jit(main)/add"(#loc2))
#loc5 = loc("jit(cholesky)/jit(main)/div"(#loc2))
#loc6 = loc("jit(cholesky)/jit(main)/cholesky"(#loc2))
#loc8 = loc("jit(cholesky)/jit(main)/jit(tril)/iota[dtype=int32 shape=(4, 4) dimension=0]"(#loc2))
#loc9 = loc("jit(cholesky)/jit(main)/jit(tril)/add"(#loc2))
#loc10 = loc("jit(cholesky)/jit(main)/jit(tril)/iota[dtype=int32 shape=(4, 4) dimension=1]"(#loc2))
#loc11 = loc("jit(cholesky)/jit(main)/jit(tril)/ge"(#loc2))
#loc12 = loc("jit(cholesky)/jit(main)/jit(tril)/broadcast_in_dim[shape=(4, 4) broadcast_dimensions=()]"(#loc2))
#loc13 = loc("jit(cholesky)/jit(main)/jit(tril)/select_n"(#loc2))
""",
    mlir_module_serialized=b"ML\xefR\x01StableHLO_v0.9.0\x00\x01)\x05\x01\x03\x01\x03\x05\x03\x19\x07\t\x0b\r\x0f\x11\x13\x15\x17\x19\x1b\x1d\x036\x02\xdd+\x01\x8f\x17\x0f\x0f\x13\x07\x0b\x0b\x0f\x0b\x0b\x0b\x0b\x0b\x13\x0b\x0f\x0b\x0b\x0f\x13+\x0b\x0f\x0b\x0b\x0b33\x0b\x0b\x13\x0f\x0b\x0b\x13\x0f\x0b\x1b\x0f\x0b\x13\x0f\x0b\x0f\x0b\x0f\x0b\x13\x0b\x0f\x0b\x0f\x0b\x13\x0b\x0bS\x0b\x0b\x0b\x0b\x0b\x0b\x0b\x0b\x0b\x1b\x13\x13\x13\x0b\x03O\x0f\x0b\x0b\x0b\x0b\x0b\x0bO\x13\x0f\x1b\x0b\x0b\x0b\x0b\x0f\x1f\x0f\x0f\x0b\x1fO\x1f/\x0b\x0b\x0b\x0b\x13\x0b\x0f\x0f\x0f\x17\x13\x0f\x0b\x1fO\x01\x05\x0b\x0f\x03'\x17\x0f\x07\x0f\x17\x07\x0f\x07\x07\x17\x13\x07\x17\x13\x07\x13\x13\x0f\x17\x02z\x07\x17;\xda\x02\t\x1do\x01\x1d9\x01\x03\x03\x1d\xb1\x1f\x05\x1f\x05!\x11\x03\x05\x05#\x05%\x05'\x05)\x05+\x03\x03\x0b\xaf\x05-\x1dC\x01\x05/\x051\x1dm\x01\x03\x03\x0b\xbd\x03\t+-/\x0f1\x0f\r3\x053\x11\x01\x00\x055\x057\x059\x03\x0b\x11\x8f\x13\x95\x15\xa1\r\xa7\x17\xa9\x03\x0b\x11\x8f\x13\x95\x15\x8f\r\x99\x17\xab\x05;\x05=\x03\x03\x19\xad\x1dA\x01\x05?\x05A\x03\x03\x19\xb3\x1dI\x01\x05C\x03\x05!\x9b#\xb5\x1dO\x01\x05E\x03\x03\x0b\xb7\x1dU\x01\x05G\x1dY\x01\x05I\x1d]\t\x05K\x03\x03a\xb9\x05M\x1de\x01\x05O\x1di\x01\x05Q\x03\x03\x0b\xbb\x05S\x05U\x03\x13s\xbfu\x97w\xc1y\xc3{\xc5}\xc7\x7f\xcd\x81\xcf\x83\xd3\x05W\x05Y\x05[\x05]\x05_\x05a\x05c\x05e\x05g\x03\x05!\x9b#\xd7\x03\x03\x0b\xd9\x03\x03\x1d\xdb\x03\x03\x8d\x99\x05i\x03\x03\x9f\x1dk\x1dm#\x1d\x1do\x1dq\t\x07\x1f#!\x00\x00\x00\x00\x00\x00\x00\x00\x01\x00\x00\x00\x00\x00\x00\x00\r\x03\x91\x93\x03\x03\xa3\r\x05\xa5\x97\x91\x93\x1ds\x1du\x1dw\x1dy\x13\t\x01\x1f\x0b\t\x00\x00\x00\x00\x1f\x1f\x01\x13\t\x05\x07\x05\x1f\x07\t\x00\x00\x00\x00\x1f\x19!\x01\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x1f\x07\t\x00\x00\x00@\x1f\x11\x11\x04\x00\x00\x00\x00\x00\x00\x00\x0b\x03\x1d{\x03\x01\x05\x01\r\x03\xc9\xcb\x1d}\x13!L\x03\x03\x9d\x03\x03\xd1\x15\x03\x01\x01\x01\x03\x05\x9d\xd5\x1f%\x01\x07\x01\x1f\x07\t\x00\x00\xc0\x7f\x1f\x19!\x00\x00\x00\x00\x00\x00\x00\x00\x01\x00\x00\x00\x00\x00\x00\x00\x01\t\x01\x02\x02)\x05\x11\x11\x13)\x01\x13\x1d)\x01\x15)\x05\x11\x11\x15\x01)\x01\t\t\x1b)\x05\x11\x11\x0f)\x03\t\t\x13\x11\x03\x05\x03\x05)\x03\x01\t!)\x03\t\x1b)\x03\x01\x1b)\x01\x0f)\x05\x05\x05\x0f\x04\xb2\x03\x05\x01\x11\t)\x07\x03\x01\t\x07\x11\t5\x07\x03'K\x03\x05[\x13\x07c_\x03\x05\x03\x01\x0b\x06g\x03\x05\x05\x01\x03\x03\x03\tk\x03\x07\x05\x07%\x07\x03\x05\x03\x07\x15\x06%\x03\x05\x05\x05\t\x03\x03\x03'\x03\x11\x03\x03\x03'\x03\x11\x17\x07\x03q\x05\x05\x0b\x03\x0b\x03\x03\x03\x1b\x03\x0b\x05\x07\x03\x07\x03\x0b\x03\x15\r\x07\x03\x85\x03'\x05\x13\x17\x05\x07\x03\x07\x03)\x03\x19\x03\x03\x03\x87\x03\x07\x05\x07\x03\x07\x03\x05\x03\x1d\x05\x07\x03\x89\x03\x17\x03\x1b\x0f\x06\x03\x03\x05\x07!\x11\x1f\x19\x07\x05\x8b\x03\x05\x03#\x11\x04\t\x03%\x07\x11\x057\x07\x03\x15+\x03\x05\x05\t\x03?=\x03\r\x03\x03\x05\x1b\x03\x0b\x05\x07\x1f\x07\x03\r\x03\x05\x0b\x06\x1f\x03\r\x05\x03\x07\t\x03GE\x03\r\r\x07MK\x03\x17\x05\t\x0b\x03\x03\x05Q\x03\x07\x05\x07S\x07\x03\x05\x03\x0f\x0f\x06W\x03\x05\x07\r\x01\x11\x11\x04\x05\x03\x13\x06\x03\x01\x05\x01\x00\xfa\x16\x7f\x0b%\x11\x0f\x0b!\x0b\x03\x11#\x0f\x1f/!)!)#\x1f\x19C99m\x19\x05W\xb3K\x9bM\x9bin\x03\x1b%)9+\x1b+\x1f\x1f\x15\x1d\x15\x13\r\x11\x1f\x15\x1b\x15\x15\x17\x0f\x11\x11)\x19\x0f\x0b\x11builtin\x00vhlo\x00module\x00constant_v1\x00broadcast_in_dim_v1\x00func_v1\x00iota_v1\x00add_v1\x00compare_v1\x00select_v1\x00return_v1\x00transpose_v1\x00divide_v1\x00custom_call_v1\x00call_v1\x00value\x00sym_name\x00arg_attrs\x00function_type\x00res_attrs\x00sym_visibility\x00iota_dimension\x00broadcast_dimensions\x00compare_type\x00comparison_direction\x00jax.uses_shape_polymorphism\x00mhlo.num_partitions\x00mhlo.num_replicas\x00jit_cholesky\x00jit(cholesky)/jit(main)/pjit[in_shardings=(UnspecifiedValue,) out_shardings=(UnspecifiedValue,) in_layouts=(None,) out_layouts=(None,) resource_env=None donated_invars=(False,) name=tril keep_unused=False inline=False]\x00third_party/py/jax/tests/export_back_compat_test.py\x00jit(cholesky)/jit(main)/jit(tril)/iota[dtype=int32 shape=(4, 4) dimension=0]\x00jit(cholesky)/jit(main)/jit(tril)/add\x00jit(cholesky)/jit(main)/jit(tril)/iota[dtype=int32 shape=(4, 4) dimension=1]\x00jit(cholesky)/jit(main)/jit(tril)/ge\x00jit(cholesky)/jit(main)/jit(tril)/broadcast_in_dim[shape=(4, 4) broadcast_dimensions=()]\x00jit(cholesky)/jit(main)/jit(tril)/select_n\x00x\x00permutation\x00jit(cholesky)/jit(main)/transpose[permutation=(1, 0)]\x00jit(cholesky)/jit(main)/add\x00jit(cholesky)/jit(main)/div\x00jit(cholesky)/jit(main)/cholesky\x00api_version\x00backend_config\x00call_target_name\x00called_computations\x00has_side_effect\x00mhlo.backend_config\x00operand_layouts\x00output_operand_aliases\x00result_layouts\x00callee\x00mhlo.layout_mode\x00default\x00\x00tril\x00jax.result_info\x00main\x00public\x00private\x00lapack_spotrf_ffi\x00uplo\x00",
    xla_call_module_version=9,
    nr_devices=1,
)  # End paste


# Pasted from the test output (see export_back_compat_test_util.py module docstring)
data_2024_05_31["f64"] = dict(
    testdata_version=1,
    platform='cpu',
    custom_call_targets=['lapack_dpotrf_ffi'],
    serialized_date=datetime.date(2024, 5, 31),
    inputs=(array([[ 23.022171138130666 , -16.79765603341739  ,   0.9133449305189147,
        -25.36636199966769  ],
       [-16.79765603341739  ,  31.655770252600092 ,  -1.5189878284433447,
         20.0344758332268   ],
       [  0.9133449305189147,  -1.5189878284433447,  10.940134497877208 ,
          8.169020034607513 ],
       [-25.36636199966769  ,  20.0344758332268   ,   8.169020034607513 ,
         37.054603917509596 ]]),),
    expected_outputs=(array([[ 4.7981424674691215 ,  0.                 ,  0.                 ,
         0.                 ],
       [-3.500866459740129  ,  4.404509539513645  ,  0.                 ,
         0.                 ],
       [ 0.19035385812557526, -0.19357078998256214,  3.2964268922333835 ,
         0.                 ],
       [-5.286704630312426  ,  0.34656047324209965,  2.803777831116442  ,
         1.060228174247857  ]]),),
    mlir_module_text=r"""
#loc1 = loc("x")
#loc2 = loc("third_party/py/jax/tests/export_back_compat_test.py":182:4)
#loc7 = loc("jit(cholesky)/jit(main)/pjit[in_shardings=(UnspecifiedValue,) out_shardings=(UnspecifiedValue,) in_layouts=(None,) out_layouts=(None,) resource_env=None donated_invars=(False,) name=tril keep_unused=False inline=False]"(#loc2))
module @jit_cholesky attributes {jax.uses_shape_polymorphism = false, mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} {
  func.func public @main(%arg0: tensor<4x4xf64> {mhlo.layout_mode = "default"} loc("x")) -> (tensor<4x4xf64> {jax.result_info = "", mhlo.layout_mode = "default"}) {
    %0 = stablehlo.transpose %arg0, dims = [1, 0] : (tensor<4x4xf64>) -> tensor<4x4xf64> loc(#loc3)
    %1 = stablehlo.add %arg0, %0 : tensor<4x4xf64> loc(#loc4)
    %cst = stablehlo.constant dense<2.000000e+00> : tensor<f64> loc(#loc)
    %2 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f64>) -> tensor<4x4xf64> loc(#loc5)
    %3 = stablehlo.divide %1, %2 : tensor<4x4xf64> loc(#loc5)
    %c = stablehlo.constant dense<4> : tensor<i64> loc(#loc6)
    %c_0 = stablehlo.constant dense<4> : tensor<i64> loc(#loc6)
    %4:2 = stablehlo.custom_call @lapack_dpotrf_ffi(%3) {mhlo.backend_config = {uplo = 76 : ui8}, operand_layouts = [dense<[0, 1]> : tensor<2xindex>], output_operand_aliases = [#stablehlo.output_operand_alias<output_tuple_indices = [0], operand_index = 0, operand_tuple_indices = []>], result_layouts = [dense<[0, 1]> : tensor<2xindex>, dense<> : tensor<0xindex>]} : (tensor<4x4xf64>) -> (tensor<4x4xf64>, tensor<i32>) loc(#loc6)
    %c_1 = stablehlo.constant dense<0> : tensor<i32> loc(#loc6)
    %5 = stablehlo.broadcast_in_dim %c_1, dims = [] : (tensor<i32>) -> tensor<i32> loc(#loc6)
    %6 = stablehlo.compare  EQ, %4#1, %5,  SIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1> loc(#loc6)
    %7 = stablehlo.broadcast_in_dim %6, dims = [] : (tensor<i1>) -> tensor<1x1xi1> loc(#loc6)
    %cst_2 = stablehlo.constant dense<0x7FF8000000000000> : tensor<f64> loc(#loc6)
    %8 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f64>) -> tensor<4x4xf64> loc(#loc6)
    %9 = stablehlo.broadcast_in_dim %7, dims = [0, 1] : (tensor<1x1xi1>) -> tensor<4x4xi1> loc(#loc6)
    %10 = stablehlo.select %9, %4#0, %8 : tensor<4x4xi1>, tensor<4x4xf64> loc(#loc6)
    %11 = call @tril(%10) : (tensor<4x4xf64>) -> tensor<4x4xf64> loc(#loc7)
    return %11 : tensor<4x4xf64> loc(#loc)
  } loc(#loc)
  func.func private @tril(%arg0: tensor<4x4xf64> {mhlo.layout_mode = "default"} loc("jit(cholesky)/jit(main)/pjit[in_shardings=(UnspecifiedValue,) out_shardings=(UnspecifiedValue,) in_layouts=(None,) out_layouts=(None,) resource_env=None donated_invars=(False,) name=tril keep_unused=False inline=False]"(#loc2))) -> (tensor<4x4xf64> {mhlo.layout_mode = "default"}) {
    %0 = stablehlo.iota dim = 0 : tensor<4x4xi32> loc(#loc8)
    %c = stablehlo.constant dense<0> : tensor<i32> loc(#loc7)
    %1 = stablehlo.broadcast_in_dim %c, dims = [] : (tensor<i32>) -> tensor<4x4xi32> loc(#loc9)
    %2 = stablehlo.add %0, %1 : tensor<4x4xi32> loc(#loc9)
    %3 = stablehlo.iota dim = 1 : tensor<4x4xi32> loc(#loc10)
    %4 = stablehlo.compare  GE, %2, %3,  SIGNED : (tensor<4x4xi32>, tensor<4x4xi32>) -> tensor<4x4xi1> loc(#loc11)
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f64> loc(#loc7)
    %5 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f64>) -> tensor<4x4xf64> loc(#loc12)
    %6 = stablehlo.select %4, %arg0, %5 : tensor<4x4xi1>, tensor<4x4xf64> loc(#loc13)
    return %6 : tensor<4x4xf64> loc(#loc7)
  } loc(#loc7)
} loc(#loc)
#loc = loc(unknown)
#loc3 = loc("jit(cholesky)/jit(main)/transpose[permutation=(1, 0)]"(#loc2))
#loc4 = loc("jit(cholesky)/jit(main)/add"(#loc2))
#loc5 = loc("jit(cholesky)/jit(main)/div"(#loc2))
#loc6 = loc("jit(cholesky)/jit(main)/cholesky"(#loc2))
#loc8 = loc("jit(cholesky)/jit(main)/jit(tril)/iota[dtype=int32 shape=(4, 4) dimension=0]"(#loc2))
#loc9 = loc("jit(cholesky)/jit(main)/jit(tril)/add"(#loc2))
#loc10 = loc("jit(cholesky)/jit(main)/jit(tril)/iota[dtype=int32 shape=(4, 4) dimension=1]"(#loc2))
#loc11 = loc("jit(cholesky)/jit(main)/jit(tril)/ge"(#loc2))
#loc12 = loc("jit(cholesky)/jit(main)/jit(tril)/broadcast_in_dim[shape=(4, 4) broadcast_dimensions=()]"(#loc2))
#loc13 = loc("jit(cholesky)/jit(main)/jit(tril)/select_n"(#loc2))
""",
    mlir_module_serialized=b"ML\xefR\x01StableHLO_v0.9.0\x00\x01)\x05\x01\x03\x01\x03\x05\x03\x19\x07\t\x0b\r\x0f\x11\x13\x15\x17\x19\x1b\x1d\x036\x02\xdd+\x01\x8f\x17\x0f\x0f\x13\x07\x0b\x0b\x0f\x0b\x0b\x0b\x0b\x0b\x13\x0b\x0f\x0b\x0b\x0f\x13+\x0b\x0f\x0b\x0b\x0b33\x0b\x0b\x13\x0f\x0b\x0b\x13\x0f\x0b\x1b\x0f\x0b\x13\x0f\x0b\x0f\x0b\x0f\x0b\x13\x0b\x0f\x0b\x0f\x0b\x13\x0b\x0bS\x0b\x0b\x0b\x0b\x0b\x0b\x0b\x0b\x0b\x1b\x13\x13\x13\x0b\x03O\x0f\x0b\x0b\x0b\x0b\x0b\x0bO\x13\x0f\x1b\x0b\x0b\x0b\x0b\x0f\x1f\x0f\x0f\x0b/O//\x0b\x0b\x0b\x0b\x13\x0b\x0f\x0f\x0f\x17\x13\x0f\x0b/O\x01\x05\x0b\x0f\x03'\x17\x0f\x07\x0f\x17\x07\x0f\x07\x07\x17\x13\x07\x17\x13\x07\x13\x13\x0f\x17\x02\xaa\x07\x17;\xda\x02\t\x1do\x01\x1d9\x01\x03\x03\x1d\xb1\x1f\x05\x1f\x05!\x11\x03\x05\x05#\x05%\x05'\x05)\x05+\x03\x03\x0b\xaf\x05-\x1dC\x01\x05/\x051\x1dm\x01\x03\x03\x0b\xbd\x03\t+-/\x0f1\x0f\r3\x053\x11\x01\x00\x055\x057\x059\x03\x0b\x11\x8f\x13\x95\x15\xa1\r\xa7\x17\xa9\x03\x0b\x11\x8f\x13\x95\x15\x8f\r\x99\x17\xab\x05;\x05=\x03\x03\x19\xad\x1dA\x01\x05?\x05A\x03\x03\x19\xb3\x1dI\x01\x05C\x03\x05!\x9b#\xb5\x1dO\x01\x05E\x03\x03\x0b\xb7\x1dU\x01\x05G\x1dY\x01\x05I\x1d]\t\x05K\x03\x03a\xb9\x05M\x1de\x01\x05O\x1di\x01\x05Q\x03\x03\x0b\xbb\x05S\x05U\x03\x13s\xbfu\x97w\xc1y\xc3{\xc5}\xc7\x7f\xcd\x81\xcf\x83\xd3\x05W\x05Y\x05[\x05]\x05_\x05a\x05c\x05e\x05g\x03\x05!\x9b#\xd7\x03\x03\x0b\xd9\x03\x03\x1d\xdb\x03\x03\x8d\x99\x05i\x03\x03\x9f\x1dk\x1dm#\x1d\x1do\x1dq\t\x07\x1f#!\x00\x00\x00\x00\x00\x00\x00\x00\x01\x00\x00\x00\x00\x00\x00\x00\r\x03\x91\x93\x03\x03\xa3\r\x05\xa5\x97\x91\x93\x1ds\x1du\x1dw\x1dy\x13\t\x01\x1f\x0b\t\x00\x00\x00\x00\x1f\x1f\x01\x13\t\x05\x07\x05\x1f\x07\x11\x00\x00\x00\x00\x00\x00\x00\x00\x1f\x19!\x01\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x1f\x07\x11\x00\x00\x00\x00\x00\x00\x00@\x1f\x11\x11\x04\x00\x00\x00\x00\x00\x00\x00\x0b\x03\x1d{\x03\x01\x05\x01\r\x03\xc9\xcb\x1d}\x13!L\x03\x03\x9d\x03\x03\xd1\x15\x03\x01\x01\x01\x03\x05\x9d\xd5\x1f%\x01\x07\x01\x1f\x07\x11\x00\x00\x00\x00\x00\x00\xf8\x7f\x1f\x19!\x00\x00\x00\x00\x00\x00\x00\x00\x01\x00\x00\x00\x00\x00\x00\x00\x01\t\x01\x02\x02)\x05\x11\x11\x13)\x01\x13\x1d)\x01\x15)\x05\x11\x11\x15\x01)\x01\t\x0b\x1b)\x05\x11\x11\x0f)\x03\t\t\x13\x11\x03\x05\x03\x05)\x03\x01\t!)\x03\t\x1b)\x03\x01\x1b)\x01\x0f)\x05\x05\x05\x0f\x04\xb2\x03\x05\x01\x11\t)\x07\x03\x01\t\x07\x11\t5\x07\x03'K\x03\x05[\x13\x07c_\x03\x05\x03\x01\x0b\x06g\x03\x05\x05\x01\x03\x03\x03\tk\x03\x07\x05\x07%\x07\x03\x05\x03\x07\x15\x06%\x03\x05\x05\x05\t\x03\x03\x03'\x03\x11\x03\x03\x03'\x03\x11\x17\x07\x03q\x05\x05\x0b\x03\x0b\x03\x03\x03\x1b\x03\x0b\x05\x07\x03\x07\x03\x0b\x03\x15\r\x07\x03\x85\x03'\x05\x13\x17\x05\x07\x03\x07\x03)\x03\x19\x03\x03\x03\x87\x03\x07\x05\x07\x03\x07\x03\x05\x03\x1d\x05\x07\x03\x89\x03\x17\x03\x1b\x0f\x06\x03\x03\x05\x07!\x11\x1f\x19\x07\x05\x8b\x03\x05\x03#\x11\x04\t\x03%\x07\x11\x057\x07\x03\x15+\x03\x05\x05\t\x03?=\x03\r\x03\x03\x05\x1b\x03\x0b\x05\x07\x1f\x07\x03\r\x03\x05\x0b\x06\x1f\x03\r\x05\x03\x07\t\x03GE\x03\r\r\x07MK\x03\x17\x05\t\x0b\x03\x03\x05Q\x03\x07\x05\x07S\x07\x03\x05\x03\x0f\x0f\x06W\x03\x05\x07\r\x01\x11\x11\x04\x05\x03\x13\x06\x03\x01\x05\x01\x00\xfa\x16\x7f\x0b%\x11\x0f\x0b!\x0b\x03\x11#\x0f\x1f/!)!)#\x1f\x19C99m\x19\x05W\xb3K\x9bM\x9bin\x03\x1b%)9+\x1b+\x1f\x1f\x15\x1d\x15\x13\r\x11\x1f\x15\x1b\x15\x15\x17\x0f\x11\x11)\x19\x0f\x0b\x11builtin\x00vhlo\x00module\x00constant_v1\x00broadcast_in_dim_v1\x00func_v1\x00iota_v1\x00add_v1\x00compare_v1\x00select_v1\x00return_v1\x00transpose_v1\x00divide_v1\x00custom_call_v1\x00call_v1\x00value\x00sym_name\x00arg_attrs\x00function_type\x00res_attrs\x00sym_visibility\x00iota_dimension\x00broadcast_dimensions\x00compare_type\x00comparison_direction\x00jax.uses_shape_polymorphism\x00mhlo.num_partitions\x00mhlo.num_replicas\x00jit_cholesky\x00jit(cholesky)/jit(main)/pjit[in_shardings=(UnspecifiedValue,) out_shardings=(UnspecifiedValue,) in_layouts=(None,) out_layouts=(None,) resource_env=None donated_invars=(False,) name=tril keep_unused=False inline=False]\x00third_party/py/jax/tests/export_back_compat_test.py\x00jit(cholesky)/jit(main)/jit(tril)/iota[dtype=int32 shape=(4, 4) dimension=0]\x00jit(cholesky)/jit(main)/jit(tril)/add\x00jit(cholesky)/jit(main)/jit(tril)/iota[dtype=int32 shape=(4, 4) dimension=1]\x00jit(cholesky)/jit(main)/jit(tril)/ge\x00jit(cholesky)/jit(main)/jit(tril)/broadcast_in_dim[shape=(4, 4) broadcast_dimensions=()]\x00jit(cholesky)/jit(main)/jit(tril)/select_n\x00x\x00permutation\x00jit(cholesky)/jit(main)/transpose[permutation=(1, 0)]\x00jit(cholesky)/jit(main)/add\x00jit(cholesky)/jit(main)/div\x00jit(cholesky)/jit(main)/cholesky\x00api_version\x00backend_config\x00call_target_name\x00called_computations\x00has_side_effect\x00mhlo.backend_config\x00operand_layouts\x00output_operand_aliases\x00result_layouts\x00callee\x00mhlo.layout_mode\x00default\x00\x00tril\x00jax.result_info\x00main\x00public\x00private\x00lapack_dpotrf_ffi\x00uplo\x00",
    xla_call_module_version=9,
    nr_devices=1,
)  # End paste

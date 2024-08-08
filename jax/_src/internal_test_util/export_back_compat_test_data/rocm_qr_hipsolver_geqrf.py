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

import datetime
from numpy import array, float32

data_2024_08_05 = {}

# Pasted from the test output (see export_back_compat_test_util.py module docstring)
data_2024_08_05["unbatched"] = dict(
    testdata_version=1,
    platform='rocm',
    custom_call_targets=['hipsolver_geqrf', 'hipsolver_orgqr'],
    serialized_date=datetime.date(2024, 8, 5),
    inputs=(),
    expected_outputs=(array([[ 0.        ,  0.9128709 ,  0.40824834],
       [-0.4472136 ,  0.3651484 , -0.81649655],
       [-0.8944272 , -0.18257423,  0.40824828]], dtype=float32), array([[-6.7082038e+00, -8.0498447e+00, -9.3914852e+00],
       [ 0.0000000e+00,  1.0954450e+00,  2.1908898e+00],
       [ 0.0000000e+00,  0.0000000e+00,  1.6371473e-09]], dtype=float32)),
    mlir_module_text=r"""
#loc2 = loc("/release/jax/tests/export_back_compat_test.py":346:0)
#loc9 = loc("jit(<lambda>)/jit(main)/pjit[in_shardings=(UnspecifiedValue,) out_shardings=(UnspecifiedValue,) in_layouts=(None,) out_layouts=(None,) resource_env=None donated_invars=(False,) name=triu keep_unused=False inline=False]"(#loc2))
module @jit__lambda_ attributes {jax.uses_shape_polymorphism = false, mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} {
  func.func public @main() -> (tensor<3x3xf32> {jax.result_info = "[0]", mhlo.layout_mode = "default"}, tensor<3x3xf32> {jax.result_info = "[1]", mhlo.layout_mode = "default"}) {
    %0 = stablehlo.iota dim = 0 : tensor<9xf32> loc(#loc3)
    %1 = stablehlo.reshape %0 : (tensor<9xf32>) -> tensor<3x3xf32> loc(#loc4)
    %2:4 = stablehlo.custom_call @hipsolver_geqrf(%1) {api_version = 2 : i32, backend_config = "\00\00\00\00\01\00\00\00\03\00\00\00\03\00\00\00\00\01\00\00", operand_layouts = [dense<[0, 1]> : tensor<2xindex>], output_operand_aliases = [#stablehlo.output_operand_alias<output_tuple_indices = [0], operand_index = 0, operand_tuple_indices = []>], result_layouts = [dense<[0, 1]> : tensor<2xindex>, dense<0> : tensor<1xindex>, dense<> : tensor<0xindex>, dense<0> : tensor<1xindex>]} : (tensor<3x3xf32>) -> (tensor<3x3xf32>, tensor<3xf32>, tensor<i32>, tensor<256xf32>) loc(#loc5)
    %c = stablehlo.constant dense<0> : tensor<i32> loc(#loc5)
    %3 = stablehlo.broadcast_in_dim %c, dims = [] : (tensor<i32>) -> tensor<i32> loc(#loc5)
    %4 = stablehlo.compare  EQ, %2#2, %3,  SIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1> loc(#loc5)
    %5 = stablehlo.broadcast_in_dim %4, dims = [] : (tensor<i1>) -> tensor<1x1xi1> loc(#loc5)
    %cst = stablehlo.constant dense<0x7FC00000> : tensor<f32> loc(#loc5)
    %6 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<3x3xf32> loc(#loc5)
    %7 = stablehlo.broadcast_in_dim %5, dims = [0, 1] : (tensor<1x1xi1>) -> tensor<3x3xi1> loc(#loc5)
    %8 = stablehlo.select %7, %2#0, %6 : tensor<3x3xi1>, tensor<3x3xf32> loc(#loc5)
    %9 = stablehlo.broadcast_in_dim %4, dims = [] : (tensor<i1>) -> tensor<1xi1> loc(#loc5)
    %cst_0 = stablehlo.constant dense<0x7FC00000> : tensor<f32> loc(#loc5)
    %10 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<3xf32> loc(#loc5)
    %11 = stablehlo.broadcast_in_dim %9, dims = [0] : (tensor<1xi1>) -> tensor<3xi1> loc(#loc5)
    %12 = stablehlo.select %11, %2#1, %10 : tensor<3xi1>, tensor<3xf32> loc(#loc5)
    %cst_1 = stablehlo.constant dense<0.000000e+00> : tensor<f32> loc(#loc6)
    %13 = stablehlo.pad %8, %cst_1, low = [0, 0], high = [0, 0], interior = [0, 0] : (tensor<3x3xf32>, tensor<f32>) -> tensor<3x3xf32> loc(#loc7)
    %14:3 = stablehlo.custom_call @hipsolver_orgqr(%13, %12) {api_version = 2 : i32, backend_config = "\00\00\00\00\01\00\00\00\03\00\00\00\03\00\00\00\03\00\00\00\80\00\00\00", operand_layouts = [dense<[0, 1]> : tensor<2xindex>, dense<0> : tensor<1xindex>], output_operand_aliases = [#stablehlo.output_operand_alias<output_tuple_indices = [0], operand_index = 0, operand_tuple_indices = []>], result_layouts = [dense<[0, 1]> : tensor<2xindex>, dense<> : tensor<0xindex>, dense<0> : tensor<1xindex>]} : (tensor<3x3xf32>, tensor<3xf32>) -> (tensor<3x3xf32>, tensor<i32>, tensor<128xf32>) loc(#loc8)
    %c_2 = stablehlo.constant dense<0> : tensor<i32> loc(#loc8)
    %15 = stablehlo.broadcast_in_dim %c_2, dims = [] : (tensor<i32>) -> tensor<i32> loc(#loc8)
    %16 = stablehlo.compare  EQ, %14#1, %15,  SIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1> loc(#loc8)
    %17 = stablehlo.broadcast_in_dim %16, dims = [] : (tensor<i1>) -> tensor<1x1xi1> loc(#loc8)
    %cst_3 = stablehlo.constant dense<0x7FC00000> : tensor<f32> loc(#loc8)
    %18 = stablehlo.broadcast_in_dim %cst_3, dims = [] : (tensor<f32>) -> tensor<3x3xf32> loc(#loc8)
    %19 = stablehlo.broadcast_in_dim %17, dims = [0, 1] : (tensor<1x1xi1>) -> tensor<3x3xi1> loc(#loc8)
    %20 = stablehlo.select %19, %14#0, %18 : tensor<3x3xi1>, tensor<3x3xf32> loc(#loc8)
    %21 = call @triu(%8) : (tensor<3x3xf32>) -> tensor<3x3xf32> loc(#loc9)
    return %20, %21 : tensor<3x3xf32>, tensor<3x3xf32> loc(#loc)
  } loc(#loc)
  func.func private @triu(%arg0: tensor<3x3xf32> {mhlo.layout_mode = "default"} loc("jit(<lambda>)/jit(main)/pjit[in_shardings=(UnspecifiedValue,) out_shardings=(UnspecifiedValue,) in_layouts=(None,) out_layouts=(None,) resource_env=None donated_invars=(False,) name=triu keep_unused=False inline=False]"(#loc2))) -> (tensor<3x3xf32> {mhlo.layout_mode = "default"}) {
    %0 = stablehlo.iota dim = 0 : tensor<3x3xi32> loc(#loc10)
    %c = stablehlo.constant dense<-1> : tensor<i32> loc(#loc9)
    %1 = stablehlo.broadcast_in_dim %c, dims = [] : (tensor<i32>) -> tensor<3x3xi32> loc(#loc11)
    %2 = stablehlo.add %0, %1 : tensor<3x3xi32> loc(#loc11)
    %3 = stablehlo.iota dim = 1 : tensor<3x3xi32> loc(#loc12)
    %4 = stablehlo.compare  GE, %2, %3,  SIGNED : (tensor<3x3xi32>, tensor<3x3xi32>) -> tensor<3x3xi1> loc(#loc13)
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32> loc(#loc9)
    %5 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<3x3xf32> loc(#loc14)
    %6 = stablehlo.select %4, %5, %arg0 : tensor<3x3xi1>, tensor<3x3xf32> loc(#loc15)
    return %6 : tensor<3x3xf32> loc(#loc9)
  } loc(#loc9)
} loc(#loc)
#loc = loc(unknown)
#loc1 = loc("/release/jax/tests/export_back_compat_test.py":345:0)
#loc3 = loc("jit(<lambda>)/jit(main)/iota[dtype=float32 shape=(9,) dimension=0]"(#loc1))
#loc4 = loc("jit(<lambda>)/jit(main)/reshape[new_sizes=(3, 3) dimensions=None]"(#loc1))
#loc5 = loc("jit(<lambda>)/jit(main)/geqrf"(#loc2))
#loc6 = loc("jit(<lambda>)/jit(main)/qr[full_matrices=True]"(#loc2))
#loc7 = loc("jit(<lambda>)/jit(main)/pad[padding_config=((0, 0, 0), (0, 0, 0))]"(#loc2))
#loc8 = loc("jit(<lambda>)/jit(main)/householder_product"(#loc2))
#loc10 = loc("jit(<lambda>)/jit(main)/jit(triu)/iota[dtype=int32 shape=(3, 3) dimension=0]"(#loc2))
#loc11 = loc("jit(<lambda>)/jit(main)/jit(triu)/add"(#loc2))
#loc12 = loc("jit(<lambda>)/jit(main)/jit(triu)/iota[dtype=int32 shape=(3, 3) dimension=1]"(#loc2))
#loc13 = loc("jit(<lambda>)/jit(main)/jit(triu)/ge"(#loc2))
#loc14 = loc("jit(<lambda>)/jit(main)/jit(triu)/broadcast_in_dim[shape=(3, 3) broadcast_dimensions=()]"(#loc2))
#loc15 = loc("jit(<lambda>)/jit(main)/jit(triu)/select_n"(#loc2))
""",
    mlir_module_serialized=b"ML\xefR\x01StableHLO_v0.9.0\x00\x01)\x05\x01\x03\x01\x03\x05\x03\x19\x07\t\x0b\r\x0f\x11\x13\x15\x17\x19\x1b\x1d\x03~\x02\xf39\x01\x99\x0f\x17\x13\x0f\x0f\x0b\x0b\x07\x0b\x13\x0f\x0b\x0b\x0b\x0b\x0b\x13\x0b\x0f\x0b\x0b\x13\x17\x0b\x0b\x0b\x0b\x0b\x0b\x0b\x0b\x13\x1b\x13+\x0b\x0f\x0b\x0b\x0b33\x0b\x0f\x0b\x13\x0b\x13\x0f\x0b\x1b\x0f\x0b\x0f\x0b\x0f\x0b\x0f\x0b\x0f\x0bK\x0b\x13\x0f\x0b#\x0b\x0b\x0b\x0f\x0bK\x0b\x13\x0b\x03[O/\x0b\x0b\x0b/\x0b\x0f\x0b\x0b\x0b\x0b\x0f\x0f\x0b\x13\x1b\x0b\x1b\x0b\x0b\x0b\x13\x0b\x0b\x0f\x1f\x0f\x0f\x0b\x1f\x0b\x0b\x0f\x17\x1b\x1f\x0b\x1fO/\x0b\x0b\x13\x17\x01\x05\x0b\x0f\x035\x17\x0f\x0f\x07\x07\x07\x17\x17\x13\x07\x07\x0f\x17\x13\x17\x17\x13\x13\x17\x13\x13\x13\x13\x13\x13\x17\x02\xde\x08\x1d}\x03\x17\x1fj\x05\x01\x03\x03\x11\xcf\x1d\x93\x03\x1dU\x03\x05\x1f\x05!\x1f\x05#\x03\x03\x0b\xe5\x11\x03\x05\x05%\x05'\x05)\x05+\x05-\x03\x03#\xcb\x05/\x1d]\x03\x051\x053\x03\x03\x0b\xd5\x17\x1ff\x05\x01\x055\x057\x059\x05;\x05=\x05?\x05A\x05C\x03\x03\x0b\xe1\x03\x05'\xab)\xe3\x03\x03\x11\xe7\x03\tGIK\x15M\x15\rO\x05E\x11\x01\x00\x05G\x05I\x05K\x03\x0b\x17\x9d\x19\xb5\x1b\xb7\r\xc1\x1d\xc3\x03\x0b\x17\xa7\x19\xc7\x1b\xa7\r\xa9\x1d\xc9\x05M\x1dY\x03\x05O\x03\x03\x0b\xcd\x05Q\x03\x03#\xd1\x1dc\x03\x05S\x03\x05'\xab)\xd3\x1di\x03\x05U\x1dm\x03\x05W\x1dq\x03\x05Y\x1du-\x05[\x1dy-\x05]\x03\x11/\xad1\xd73\xd95\x9d7\xaf9\xdb;\xb1=\xdf\x05_\x03\x03\x11\xe9\x1d\x83\x03\x05a\x03\x07\x87\xa3\x89\xa3\x8b\xa3\x05c\x05e\x05g\x1d\x8f\x03\x05i\x03\x11/\xad1\xeb3\xed5\x9d7\xaf9\xef;\xb1=\xf1\x05k\x03\x03\x97\xa9\x05m\x1f+!\x00\x00\x00\x00\x00\x00\x00\x00\x01\x00\x00\x00\x00\x00\x00\x00\x1f-\x11\x00\x00\x00\x00\x00\x00\x00\x00\x03\x01\x1do\x1dq\x1f\x1f\x11\x00\x00\x00\x00\x00\x00\x00\x00\x1ds\x03\x03\xc5\x1du\t\x07\x0b\x05\x05\x01\x03\x03\xdd\x1f/\x01#!\x03\x05\xb9\xbd\r\x05\xa5\xbb\x9f\xa1\x1dw\r\x05\xa5\xbf\x9f\xa1\x1dy\x1d{\x1d}\r\x03\x9f\xa1##\x1d\x7f\x13\r\x01\x1f\x07\t\xff\xff\xff\xff\x1f%\x01\x13\r\x05\x07\x05\x1f\t\t\x00\x00\x00\x00\x1d\x81\x1d\x83\x03\x03\x99\x15\x03\x01\x01\x01\x03\t\x99\x9b\xb3\x9b\x1f\x07\t\x00\x00\x00\x00\x07\x01\x1f\t\t\x00\x00\xc0\x7f\x1f\x1f!\x00\x00\x00\x00\x00\x00\x00\x00\x01\x00\x00\x00\x00\x00\x00\x00\x1f5\x11\x00\x00\x00\x00\x00\x00\x00\x00\x1d\x85\x1d\x87\x03\x05\x99\x9b\x03\x07\x99\xb3\x9b\x01\t\x01\x02\x02)\x05\r\r\x0b)\x01\x19)\x01\x0b\t\x1d\x01)\x05\r\r\x19)\x05\r\r\x0f)\x03\r\x0b\x13\x1b)\x01\x0f)\x05\x05\x05\x0f)\x03\t\r\x11\x01\x05\x05\x05\x11\x03\x05\x03\x05)\x03\x01\r)\x03%\x0b)\x03\x02\x08\x0b)\x03\t\x17)\x03\x05\x17)\x03\x01\x17)\x03\x05\x0f)\x03\r\x0f)\x03\x05\r)\x03\x02\x04\x0b\x04\x1a\x05\x05\x01\x11\x0fE\x07\x03\x01\t\r\x11\x0fQ\x07\x03Cu\t\x03s!\x03'\x15\x06w\x03\x05\x03\x01\x11\x07\x01{\t\x05\x15\x07)\x03\x03\x05\x03\x01?\x03\x07\x03\x07\x01\x05\x03\x07\x03\r\x0b\x07\x01A\x03\x1b\x05\t\x0f\x03\x07\x01\x05\x03\x1d\x03\x11\x05\x03\x01\x13\x03\t\x03\x07\x01\x05\x03\x05\x03\x15\x03\x07\x01C\x03\x13\x03\x13\x07\x06\x01\x03\x05\x07\x19\x05\x17\x03\x07\x01\x05\x031\x03\x11\x05\x03\x01\x13\x03\t\x03\x07\x01\x05\x03\x15\x03\x1f\x03\x07\x01\x7f\x033\x03\x1d\x07\x06\x01\x03\x15\x07#\x07!\x05\x03\x81+\x03\t\x17\x07\x8d\x85\x03\x05\x05\x1b'\x11\x07\x07\x91\x07\x05\x077\x05)%\x05\x03\x07?\x03\x07\x03\x07\x07\x05\x03\x07\x031\x0b\x07\x07A\x03\x1b\x05-3\x03\x07\x07\x05\x03\x1d\x035\x05\x03\x07\x13\x03\t\x03\x07\x07\x05\x03\x05\x039\x03\x07\x07C\x03\x13\x037\x07\x06\x07\x03\x05\x07=+;\x19\x07\t\x95\x03\x05\x03\x1b\x0f\x04\x0f\x05?A\r\x11\tS\x07\x03\x15+\x03\x05\t\t\x03W!\x03\x11\x05\x03\t[\x03\x07\x03\x07%\x05\x03\x11\x03\x05\x13\x06%\x03\x11\x05\x03\x07\t\x03a_\x03\x11\x0b\x07ge\x03\x13\x05\t\x0b\x05\x03\t+\x03\t\x03\x07k\x05\x03\x05\x03\x0f\x07\x06o\x03\x05\x07\r\x11\x01\x0f\x04\t\x03\x13\x06\x03\x01\x05\x01\x00\xea\x1a\x89!3!+\x11\x0f\x0b\t\t\x0b!\x11#\x0fY\x87##%_=\x85\x87W\xb3K\x9bM\x9bn\x03\x1b%)9\x1f/!!)#\x1f\x19+\x1b\x1f]\x1f\x15\x1d\x15+\x13\r\x11\x0f\x17\x0f\x1f\x15\x11\x17\x11\x15\x19)\x0f\x0b\x11builtin\x00vhlo\x00module\x00broadcast_in_dim_v1\x00constant_v1\x00select_v1\x00iota_v1\x00compare_v1\x00func_v1\x00return_v1\x00custom_call_v1\x00add_v1\x00reshape_v1\x00pad_v1\x00call_v1\x00value\x00sym_name\x00broadcast_dimensions\x00arg_attrs\x00function_type\x00res_attrs\x00sym_visibility\x00/release/jax/tests/export_back_compat_test.py\x00iota_dimension\x00compare_type\x00comparison_direction\x00api_version\x00backend_config\x00call_target_name\x00called_computations\x00has_side_effect\x00operand_layouts\x00output_operand_aliases\x00result_layouts\x00jax.uses_shape_polymorphism\x00mhlo.num_partitions\x00mhlo.num_replicas\x00jit__lambda_\x00jit(<lambda>)/jit(main)/pjit[in_shardings=(UnspecifiedValue,) out_shardings=(UnspecifiedValue,) in_layouts=(None,) out_layouts=(None,) resource_env=None donated_invars=(False,) name=triu keep_unused=False inline=False]\x00jit(<lambda>)/jit(main)/jit(triu)/iota[dtype=int32 shape=(3, 3) dimension=0]\x00jit(<lambda>)/jit(main)/jit(triu)/add\x00jit(<lambda>)/jit(main)/jit(triu)/iota[dtype=int32 shape=(3, 3) dimension=1]\x00jit(<lambda>)/jit(main)/jit(triu)/ge\x00jit(<lambda>)/jit(main)/jit(triu)/broadcast_in_dim[shape=(3, 3) broadcast_dimensions=()]\x00jit(<lambda>)/jit(main)/jit(triu)/select_n\x00jit(<lambda>)/jit(main)/iota[dtype=float32 shape=(9,) dimension=0]\x00jit(<lambda>)/jit(main)/reshape[new_sizes=(3, 3) dimensions=None]\x00jit(<lambda>)/jit(main)/geqrf\x00jit(<lambda>)/jit(main)/qr[full_matrices=True]\x00edge_padding_high\x00edge_padding_low\x00interior_padding\x00jit(<lambda>)/jit(main)/pad[padding_config=((0, 0, 0), (0, 0, 0))]\x00jit(<lambda>)/jit(main)/householder_product\x00callee\x00mhlo.layout_mode\x00default\x00jax.result_info\x00triu\x00[0]\x00[1]\x00main\x00public\x00private\x00\x00\x00\x00\x00\x01\x00\x00\x00\x03\x00\x00\x00\x03\x00\x00\x00\x00\x01\x00\x00\x00hipsolver_geqrf\x00\x00\x00\x00\x00\x01\x00\x00\x00\x03\x00\x00\x00\x03\x00\x00\x00\x03\x00\x00\x00\x80\x00\x00\x00\x00hipsolver_orgqr\x00",
    xla_call_module_version=9,
    nr_devices=1,
)  # End paste


# Pasted from the test output (see export_back_compat_test_util.py module docstring)
data_2024_08_05["batched"] = dict(
    testdata_version=1,
    platform='rocm',
    custom_call_targets=['hipblas_geqrf_batched', 'hipsolver_orgqr'],
    serialized_date=datetime.date(2024, 8, 5),
    inputs=(),
    expected_outputs=(array([[[ 0.        ,  0.9128709 ,  0.40824834],
        [-0.4472136 ,  0.3651484 , -0.81649655],
        [-0.8944272 , -0.18257423,  0.40824828]],

       [[-0.42426407,  0.8082888 ,  0.4082513 ],
        [-0.5656854 ,  0.11547317, -0.81649613],
        [-0.7071068 , -0.5773518 ,  0.40824607]]], dtype=float32), array([[[-6.7082038e+00, -8.0498447e+00, -9.3914852e+00],
        [ 0.0000000e+00,  1.0954450e+00,  2.1908898e+00],
        [ 0.0000000e+00,  0.0000000e+00,  1.6371473e-09]],

       [[-2.1213203e+01, -2.2910259e+01, -2.4607313e+01],
        [ 0.0000000e+00,  3.4641036e-01,  6.9281983e-01],
        [ 0.0000000e+00,  0.0000000e+00,  8.3555670e-07]]], dtype=float32)),
    mlir_module_text=r"""
#loc2 = loc("/release/jax/tests/export_back_compat_test.py":346:0)
#loc9 = loc("jit(<lambda>)/jit(main)/pjit[in_shardings=(UnspecifiedValue,) out_shardings=(UnspecifiedValue,) in_layouts=(None,) out_layouts=(None,) resource_env=None donated_invars=(False,) name=triu keep_unused=False inline=False]"(#loc2))
module @jit__lambda_ attributes {jax.uses_shape_polymorphism = false, mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} {
  func.func public @main() -> (tensor<2x3x3xf32> {jax.result_info = "[0]", mhlo.layout_mode = "default"}, tensor<2x3x3xf32> {jax.result_info = "[1]", mhlo.layout_mode = "default"}) {
    %0 = stablehlo.iota dim = 0 : tensor<18xf32> loc(#loc3)
    %1 = stablehlo.reshape %0 : (tensor<18xf32>) -> tensor<2x3x3xf32> loc(#loc4)
    %2:4 = stablehlo.custom_call @hipblas_geqrf_batched(%1) {api_version = 2 : i32, backend_config = "\00\00\00\00\02\00\00\00\03\00\00\00\03\00\00\00", operand_layouts = [dense<[1, 2, 0]> : tensor<3xindex>], output_operand_aliases = [#stablehlo.output_operand_alias<output_tuple_indices = [0], operand_index = 0, operand_tuple_indices = []>], result_layouts = [dense<[1, 2, 0]> : tensor<3xindex>, dense<[1, 0]> : tensor<2xindex>, dense<0> : tensor<1xindex>, dense<0> : tensor<1xindex>]} : (tensor<2x3x3xf32>) -> (tensor<2x3x3xf32>, tensor<2x3xf32>, tensor<16xi8>, tensor<16xi8>) loc(#loc5)
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32> loc(#loc6)
    %3 = stablehlo.pad %2#0, %cst, low = [0, 0, 0], high = [0, 0, 0], interior = [0, 0, 0] : (tensor<2x3x3xf32>, tensor<f32>) -> tensor<2x3x3xf32> loc(#loc7)
    %4:3 = stablehlo.custom_call @hipsolver_orgqr(%3, %2#1) {api_version = 2 : i32, backend_config = "\00\00\00\00\02\00\00\00\03\00\00\00\03\00\00\00\03\00\00\00\80\00\00\00", operand_layouts = [dense<[1, 2, 0]> : tensor<3xindex>, dense<[1, 0]> : tensor<2xindex>], output_operand_aliases = [#stablehlo.output_operand_alias<output_tuple_indices = [0], operand_index = 0, operand_tuple_indices = []>], result_layouts = [dense<[1, 2, 0]> : tensor<3xindex>, dense<0> : tensor<1xindex>, dense<0> : tensor<1xindex>]} : (tensor<2x3x3xf32>, tensor<2x3xf32>) -> (tensor<2x3x3xf32>, tensor<2xi32>, tensor<128xf32>) loc(#loc8)
    %c = stablehlo.constant dense<0> : tensor<i32> loc(#loc8)
    %5 = stablehlo.broadcast_in_dim %c, dims = [] : (tensor<i32>) -> tensor<2xi32> loc(#loc8)
    %6 = stablehlo.compare  EQ, %4#1, %5,  SIGNED : (tensor<2xi32>, tensor<2xi32>) -> tensor<2xi1> loc(#loc8)
    %7 = stablehlo.broadcast_in_dim %6, dims = [0] : (tensor<2xi1>) -> tensor<2x1x1xi1> loc(#loc8)
    %cst_0 = stablehlo.constant dense<0x7FC00000> : tensor<f32> loc(#loc8)
    %8 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<2x3x3xf32> loc(#loc8)
    %9 = stablehlo.broadcast_in_dim %7, dims = [0, 1, 2] : (tensor<2x1x1xi1>) -> tensor<2x3x3xi1> loc(#loc8)
    %10 = stablehlo.select %9, %4#0, %8 : tensor<2x3x3xi1>, tensor<2x3x3xf32> loc(#loc8)
    %11 = call @triu(%2#0) : (tensor<2x3x3xf32>) -> tensor<2x3x3xf32> loc(#loc9)
    return %10, %11 : tensor<2x3x3xf32>, tensor<2x3x3xf32> loc(#loc)
  } loc(#loc)
  func.func private @triu(%arg0: tensor<2x3x3xf32> {mhlo.layout_mode = "default"} loc("jit(<lambda>)/jit(main)/pjit[in_shardings=(UnspecifiedValue,) out_shardings=(UnspecifiedValue,) in_layouts=(None,) out_layouts=(None,) resource_env=None donated_invars=(False,) name=triu keep_unused=False inline=False]"(#loc2))) -> (tensor<2x3x3xf32> {mhlo.layout_mode = "default"}) {
    %0 = stablehlo.iota dim = 0 : tensor<3x3xi32> loc(#loc10)
    %c = stablehlo.constant dense<-1> : tensor<i32> loc(#loc9)
    %1 = stablehlo.broadcast_in_dim %c, dims = [] : (tensor<i32>) -> tensor<3x3xi32> loc(#loc11)
    %2 = stablehlo.add %0, %1 : tensor<3x3xi32> loc(#loc11)
    %3 = stablehlo.iota dim = 1 : tensor<3x3xi32> loc(#loc12)
    %4 = stablehlo.compare  GE, %2, %3,  SIGNED : (tensor<3x3xi32>, tensor<3x3xi32>) -> tensor<3x3xi1> loc(#loc13)
    %5 = stablehlo.broadcast_in_dim %4, dims = [1, 2] : (tensor<3x3xi1>) -> tensor<2x3x3xi1> loc(#loc14)
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32> loc(#loc9)
    %6 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<2x3x3xf32> loc(#loc15)
    %7 = stablehlo.select %5, %6, %arg0 : tensor<2x3x3xi1>, tensor<2x3x3xf32> loc(#loc16)
    return %7 : tensor<2x3x3xf32> loc(#loc9)
  } loc(#loc9)
} loc(#loc)
#loc = loc(unknown)
#loc1 = loc("/release/jax/tests/export_back_compat_test.py":345:0)
#loc3 = loc("jit(<lambda>)/jit(main)/iota[dtype=float32 shape=(18,) dimension=0]"(#loc1))
#loc4 = loc("jit(<lambda>)/jit(main)/reshape[new_sizes=(2, 3, 3) dimensions=None]"(#loc1))
#loc5 = loc("jit(<lambda>)/jit(main)/geqrf"(#loc2))
#loc6 = loc("jit(<lambda>)/jit(main)/qr[full_matrices=True]"(#loc2))
#loc7 = loc("jit(<lambda>)/jit(main)/pad[padding_config=((0, 0, 0), (0, 0, 0), (0, 0, 0))]"(#loc2))
#loc8 = loc("jit(<lambda>)/jit(main)/householder_product"(#loc2))
#loc10 = loc("jit(<lambda>)/jit(main)/jit(triu)/iota[dtype=int32 shape=(3, 3) dimension=0]"(#loc2))
#loc11 = loc("jit(<lambda>)/jit(main)/jit(triu)/add"(#loc2))
#loc12 = loc("jit(<lambda>)/jit(main)/jit(triu)/iota[dtype=int32 shape=(3, 3) dimension=1]"(#loc2))
#loc13 = loc("jit(<lambda>)/jit(main)/jit(triu)/ge"(#loc2))
#loc14 = loc("jit(<lambda>)/jit(main)/jit(triu)/broadcast_in_dim[shape=(2, 3, 3) broadcast_dimensions=(1, 2)]"(#loc2))
#loc15 = loc("jit(<lambda>)/jit(main)/jit(triu)/broadcast_in_dim[shape=(2, 3, 3) broadcast_dimensions=()]"(#loc2))
#loc16 = loc("jit(<lambda>)/jit(main)/jit(triu)/select_n"(#loc2))
""",
    mlir_module_serialized=b"ML\xefR\x01StableHLO_v0.9.0\x00\x01)\x05\x01\x03\x01\x03\x05\x03\x19\x07\t\x0b\r\x0f\x11\x13\x15\x17\x19\x1b\x1d\x03\x96\x02\xfb=\x01\x9f\x17\x0f\x0f\x0b\x13\x0b\x0b\x07\x0f\x0b\x0b\x0b\x0b\x0b\x13\x0b\x0f\x0b\x0b\x13\x17\x0b\x0b\x0b\x0b\x0b\x0b\x0b\x0b+\x0b\x0f\x0b\x0b\x0b33\x0b\x0f\x0b\x13\x0b\x13\x0f\x0b\x1b\x0f\x0b\x13\x0f\x0b\x0f\x0b\x0f\x0b\x0f\x0b\x0f\x0bK\x0f\x0b\x0f\x0b#\x0b\x0b\x0b\x0f\x0bK\x0b\x13\x1b\x13\x13\x13\x13\x0b\x03]o/\x0b\x0b\x0b/\x0b\x0f\x0b\x0b\x0b\x0b\x0fO\x0b\x13\x1b\x0b\x1b\x0b\x0b\x0b\x13\x0b\x0b\x0f\x1f\x0f\x0f\x0bO\x1f\x0b\x0b\x0f\x17\x1b\x0b\x0b\x13\x17\x1f\x0b/\x1fo\x01\x05\x0b\x0f\x039\x1b\x07\x07\x0f\x17\x0f\x07\x07\x07\x1b\x13\x13\x13\x17\x17\x13\x17\x13\x13\x17\x07\x13\x13\x13\x17\x13\x1b\x13\x02\xf6\t\x17\x1bj\x05\x01\x1d\x8f\x01\x1dK\x01\x05\x1f\x03\x03\x0b\xd5\x05!\x05#\x1f\x11\x03\x05\x05%\x05'\x05)\x05+\x05-\x03\x03\x1f\xd1\x05/\x1dS\x01\x051\x053\x03\x03\x07\xdd\x17\x1bf\x05\x01\x055\x057\x059\x05;\x05=\x05?\x05A\x05C\x03\t=?A\x11C\x11\rE\x05E\x11\x01\x00\x05G\x05I\x05K\x03\x0b\x13\xa3\x15\xbb\x17\xbd\r\xc7\x19\xc9\x03\x0b\x13\xad\x15\xcd\x17\xad\r\xaf\x19\xcf\x05M\x1dO\x01\x05O\x03\x03\x07\xd3\x05Q\x03\x03\x1f\xd7\x1dY\x01\x05S\x03\x05#\xb1%\xd9\x1d_\x01\x05U\x03\x03\x0b\xdb\x1de\x01\x05W\x1di\x01\x05Y\x1dm\x01\x05[\x1dq)\x05]\x1du)\x05_\x03\x11+\xb3-\xdf/\xe11\xa33\xb55\xe37\xb79\xe7\x1d{\x01\x05a\x1d\x7f\x01\x05c\x03\x07\x83\xa9\x85\xa9\x87\xa9\x05e\x05g\x05i\x1d\x8b\x01\x05k\x03\x11+\xb3-\xe9/\xeb1\xa33\xb55\xed7\xb79\xef\x05m\x03\x03\x07\xf1\x03\x05#\xb1%\xf3\x03\x03\x0b\xf5\x03\x03\x07\xf7\x03\x03\x0b\xf9\x03\x03\x9d\xaf\x05o\x1f/1\x01\x00\x00\x00\x00\x00\x00\x00\x02\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x1f3\x11\x00\x00\x00\x00\x00\x00\x00\x00\x03\x01\x1dq\x1ds\x1f\x1b\x11\x00\x00\x00\x00\x00\x00\x00\x00\x1du\x03\x03\xcb\x1dw\t\x07\x0b\x05\x05\x01\x03\x03\xe5\x1f1!\x01\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00#\x1f\x03\x05\xbf\xc3\r\x05\xab\xc1\xa5\xa7\x1dy\r\x05\xab\xc5\xa5\xa7\x1d{\x1d}\x1d\x7f\r\x03\xa5\xa7#!\x1d\x81\x13\x07\x01\x1f\x0f\t\xff\xff\xff\xff\x1f#\x01\x13\x07\x05\x07\x05\x1f'!\x01\x00\x00\x00\x00\x00\x00\x00\x02\x00\x00\x00\x00\x00\x00\x00\x1f\x0b\t\x00\x00\x00\x00\x1d\x83\x1d\x85\x03\x03\x9f\x15\x03\x01\x01\x01\x03\t\x9f\xb9\xa1\xa1\x1d\x87\x1d\x89\x03\x05\x9f\xb9\x03\x07\x9f\xa1\xa1\x1f\x0f\t\x00\x00\x00\x00\x07\x01\x1f;\x11\x00\x00\x00\x00\x00\x00\x00\x00\x1f\x0b\t\x00\x00\xc0\x7f\x1f\x1b1\x00\x00\x00\x00\x00\x00\x00\x00\x01\x00\x00\x00\x00\x00\x00\x00\x02\x00\x00\x00\x00\x00\x00\x00\x01\t\x01\x02\x02)\x07\t\r\r\t\x1d\t)\x01\t)\x05\r\r\x13)\x01\x13\x01\x1b\x13)\x07\t\r\r\x11)\x03A-)\x03\r\x07)\x03\t\x13\x11\x01\x05\x05\x05\x11\x03\x05\x03\x05)\x03\x01\x07)\x05\r\r\x11)\x03\t\x07)\x03I\t)\x05\t\r\t\x17)\x03\r\x15)\x03\t\x15)\x03\x05\x15)\x03\x02\x04\t)\x03\t\x11)\x07\t\x05\x05\x11)\x03\x05\x07\x04\xa6\x03\x05\x01\x11\x0f;\x07\x03\x01\t\t\x11\x0fG\x07\x03)A\x07\x03o\x1d\x03)\x15\x06s\x03\x05\x03\x01\x11\x07yw\t\x05+\x19\x19\x03\x03\x05\x03}'\x03\x0b\x17\x07\x89\x81\x03\x05\x05\x05\r\x11\x07\x03\x8d\x07\x05\x1d5\x05\x0f\x07\x05\x03\x03\x91\x03\x0f\x03\x07\x03\t\x03\x1d\x03\x17\x0b\x07\x03\x93\x037\x05\x13\x19\x03\x07\x03\x95\x039\x03\x1b\x05\x03\x03\x97\x03\x0b\x03\x07\x03\t\x03\x05\x03\x1f\x03\x07\x03\x99\x03\x17\x03\x1d\r\x06\x03\x03\x05\x07#\x11!\x19\x07\x05\x9b\x03\x05\x03\x05\x0f\x04\x0f\x05%'\t\x11\x05I\x07\x03\x17/\x03\x05\x05\x07\x03M\x1d\x03\r\x05\x03\x05Q\x03\x0f\x03\x07!\t\x03\r\x03\x05\x13\x06!\x03\r\x05\x03\x07\x07\x03WU\x03\r\x0b\x07][\x03%\x05\t\x0b\x03\x07ca\x03\x17\x03\r\x05\x03\x05'\x03\x0b\x03\x07g\t\x03\x05\x03\x11\r\x06k\x03\x05\x07\x0f\x13\x01\x0f\x04\x05\x03\x15\x06\x03\x01\x05\x01\x00\xbe\x1c\x8b!3-#\x11\x0f\x0b\t\t\x0b!\x11#\x0fY\x9d##%_=\x8b\x89W\xb9\xc1K\x9bM\x9bn\x03\x1b%)9\x1f/!!)#\x1f\x19+\x1b\x1f]\x1f\x15\x1d\x15\x13+\r\x11\x0f\x17\x0f\x1f\x15\x15\x17\x11\x11\x19)\x0f\x0b\x11builtin\x00vhlo\x00module\x00broadcast_in_dim_v1\x00constant_v1\x00iota_v1\x00func_v1\x00compare_v1\x00select_v1\x00return_v1\x00custom_call_v1\x00add_v1\x00reshape_v1\x00pad_v1\x00call_v1\x00value\x00broadcast_dimensions\x00sym_name\x00arg_attrs\x00function_type\x00res_attrs\x00sym_visibility\x00/release/jax/tests/export_back_compat_test.py\x00iota_dimension\x00compare_type\x00comparison_direction\x00api_version\x00backend_config\x00call_target_name\x00called_computations\x00has_side_effect\x00operand_layouts\x00output_operand_aliases\x00result_layouts\x00jax.uses_shape_polymorphism\x00mhlo.num_partitions\x00mhlo.num_replicas\x00jit__lambda_\x00jit(<lambda>)/jit(main)/pjit[in_shardings=(UnspecifiedValue,) out_shardings=(UnspecifiedValue,) in_layouts=(None,) out_layouts=(None,) resource_env=None donated_invars=(False,) name=triu keep_unused=False inline=False]\x00jit(<lambda>)/jit(main)/jit(triu)/iota[dtype=int32 shape=(3, 3) dimension=0]\x00jit(<lambda>)/jit(main)/jit(triu)/add\x00jit(<lambda>)/jit(main)/jit(triu)/iota[dtype=int32 shape=(3, 3) dimension=1]\x00jit(<lambda>)/jit(main)/jit(triu)/ge\x00jit(<lambda>)/jit(main)/jit(triu)/broadcast_in_dim[shape=(2, 3, 3) broadcast_dimensions=(1, 2)]\x00jit(<lambda>)/jit(main)/jit(triu)/broadcast_in_dim[shape=(2, 3, 3) broadcast_dimensions=()]\x00jit(<lambda>)/jit(main)/jit(triu)/select_n\x00jit(<lambda>)/jit(main)/iota[dtype=float32 shape=(18,) dimension=0]\x00jit(<lambda>)/jit(main)/reshape[new_sizes=(2, 3, 3) dimensions=None]\x00jit(<lambda>)/jit(main)/geqrf\x00jit(<lambda>)/jit(main)/qr[full_matrices=True]\x00edge_padding_high\x00edge_padding_low\x00interior_padding\x00jit(<lambda>)/jit(main)/pad[padding_config=((0, 0, 0), (0, 0, 0), (0, 0, 0))]\x00jit(<lambda>)/jit(main)/householder_product\x00callee\x00mhlo.layout_mode\x00default\x00jax.result_info\x00triu\x00[0]\x00[1]\x00main\x00public\x00private\x00\x00\x00\x00\x00\x02\x00\x00\x00\x03\x00\x00\x00\x03\x00\x00\x00\x00hipblas_geqrf_batched\x00\x00\x00\x00\x00\x02\x00\x00\x00\x03\x00\x00\x00\x03\x00\x00\x00\x03\x00\x00\x00\x80\x00\x00\x00\x00hipsolver_orgqr\x00",
    xla_call_module_version=9,
    nr_devices=1,
)  # End paste

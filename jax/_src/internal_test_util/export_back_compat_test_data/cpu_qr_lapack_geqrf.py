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

data_2023_03_17 = {}

# Pasted from the test output (see back_compat_test.py module docstring)
data_2023_03_17["f32"] = dict(
    testdata_version=1,
    platform='cpu',
    custom_call_targets=['lapack_sgeqrf', 'lapack_sorgqr'],
    serialized_date=datetime.date(2023, 3, 17),
    inputs=(),
    expected_outputs=(array([[ 0.        ,  0.91287076,  0.4082487 ],
       [-0.44721356,  0.36514866, -0.8164965 ],
       [-0.8944271 , -0.18257445,  0.40824816]], dtype=float32), array([[-6.7082043e+00, -8.0498438e+00, -9.3914852e+00],
       [ 0.0000000e+00,  1.0954441e+00,  2.1908894e+00],
       [ 0.0000000e+00,  0.0000000e+00,  7.1525574e-07]], dtype=float32)),
    mlir_module_text=r"""
module @jit__lambda_ {
  func.func public @main() -> (tensor<3x3xf32> {jax.result_info = "[0]"}, tensor<3x3xf32> {jax.result_info = "[1]"}) {
    %0 = stablehlo.iota dim = 0 : tensor<9xf32>
    %1 = stablehlo.reshape %0 : (tensor<9xf32>) -> tensor<3x3xf32>
    %2 = stablehlo.constant dense<1> : tensor<i32>
    %3 = stablehlo.constant dense<3> : tensor<i32>
    %4 = stablehlo.constant dense<3> : tensor<i32>
    %5 = stablehlo.constant dense<96> : tensor<i32>
    %6 = stablehlo.custom_call @lapack_sgeqrf(%2, %3, %4, %5, %1) {api_version = 2 : i32, operand_layouts = [dense<> : tensor<0xindex>, dense<> : tensor<0xindex>, dense<> : tensor<0xindex>, dense<> : tensor<0xindex>, dense<[0, 1]> : tensor<2xindex>], output_operand_aliases = [#stablehlo.output_operand_alias<output_tuple_indices = [0], operand_index = 4, operand_tuple_indices = []>], result_layouts = [dense<[0, 1]> : tensor<2xindex>, dense<0> : tensor<1xindex>, dense<> : tensor<0xindex>, dense<0> : tensor<1xindex>]} : (tensor<i32>, tensor<i32>, tensor<i32>, tensor<i32>, tensor<3x3xf32>) -> tuple<tensor<3x3xf32>, tensor<3xf32>, tensor<i32>, tensor<96xf32>>
    %7 = stablehlo.get_tuple_element %6[0] : (tuple<tensor<3x3xf32>, tensor<3xf32>, tensor<i32>, tensor<96xf32>>) -> tensor<3x3xf32>
    %8 = stablehlo.get_tuple_element %6[1] : (tuple<tensor<3x3xf32>, tensor<3xf32>, tensor<i32>, tensor<96xf32>>) -> tensor<3xf32>
    %9 = stablehlo.get_tuple_element %6[2] : (tuple<tensor<3x3xf32>, tensor<3xf32>, tensor<i32>, tensor<96xf32>>) -> tensor<i32>
    %10 = stablehlo.get_tuple_element %6[3] : (tuple<tensor<3x3xf32>, tensor<3xf32>, tensor<i32>, tensor<96xf32>>) -> tensor<96xf32>
    %11 = stablehlo.constant dense<0> : tensor<i32>
    %12 = stablehlo.broadcast_in_dim %11, dims = [] : (tensor<i32>) -> tensor<i32>
    %13 = stablehlo.compare  EQ, %9, %12,  SIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1>
    %14 = stablehlo.broadcast_in_dim %13, dims = [] : (tensor<i1>) -> tensor<1x1xi1>
    %15 = stablehlo.constant dense<0x7FC00000> : tensor<f32>
    %16 = stablehlo.broadcast_in_dim %15, dims = [] : (tensor<f32>) -> tensor<3x3xf32>
    %17 = stablehlo.broadcast_in_dim %14, dims = [0, 1] : (tensor<1x1xi1>) -> tensor<3x3xi1>
    %18 = stablehlo.select %17, %7, %16 : tensor<3x3xi1>, tensor<3x3xf32>
    %19 = stablehlo.broadcast_in_dim %13, dims = [] : (tensor<i1>) -> tensor<1xi1>
    %20 = stablehlo.constant dense<0x7FC00000> : tensor<f32>
    %21 = stablehlo.broadcast_in_dim %20, dims = [] : (tensor<f32>) -> tensor<3xf32>
    %22 = stablehlo.broadcast_in_dim %19, dims = [0] : (tensor<1xi1>) -> tensor<3xi1>
    %23 = stablehlo.select %22, %8, %21 : tensor<3xi1>, tensor<3xf32>
    %24 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %25 = stablehlo.pad %18, %24, low = [0, 0], high = [0, 0], interior = [0, 0] : (tensor<3x3xf32>, tensor<f32>) -> tensor<3x3xf32>
    %26 = stablehlo.constant dense<1> : tensor<i32>
    %27 = stablehlo.constant dense<3> : tensor<i32>
    %28 = stablehlo.constant dense<3> : tensor<i32>
    %29 = stablehlo.constant dense<3> : tensor<i32>
    %30 = stablehlo.constant dense<96> : tensor<i32>
    %31 = stablehlo.custom_call @lapack_sorgqr(%26, %27, %28, %29, %30, %25, %23) {api_version = 2 : i32, operand_layouts = [dense<> : tensor<0xindex>, dense<> : tensor<0xindex>, dense<> : tensor<0xindex>, dense<> : tensor<0xindex>, dense<> : tensor<0xindex>, dense<[0, 1]> : tensor<2xindex>, dense<0> : tensor<1xindex>], output_operand_aliases = [#stablehlo.output_operand_alias<output_tuple_indices = [0], operand_index = 5, operand_tuple_indices = []>], result_layouts = [dense<[0, 1]> : tensor<2xindex>, dense<> : tensor<0xindex>, dense<0> : tensor<1xindex>]} : (tensor<i32>, tensor<i32>, tensor<i32>, tensor<i32>, tensor<i32>, tensor<3x3xf32>, tensor<3xf32>) -> tuple<tensor<3x3xf32>, tensor<i32>, tensor<96xf32>>
    %32 = stablehlo.get_tuple_element %31[0] : (tuple<tensor<3x3xf32>, tensor<i32>, tensor<96xf32>>) -> tensor<3x3xf32>
    %33 = stablehlo.get_tuple_element %31[1] : (tuple<tensor<3x3xf32>, tensor<i32>, tensor<96xf32>>) -> tensor<i32>
    %34 = stablehlo.get_tuple_element %31[2] : (tuple<tensor<3x3xf32>, tensor<i32>, tensor<96xf32>>) -> tensor<96xf32>
    %35 = stablehlo.constant dense<0> : tensor<i32>
    %36 = stablehlo.broadcast_in_dim %35, dims = [] : (tensor<i32>) -> tensor<i32>
    %37 = stablehlo.compare  EQ, %33, %36,  SIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1>
    %38 = stablehlo.broadcast_in_dim %37, dims = [] : (tensor<i1>) -> tensor<1x1xi1>
    %39 = stablehlo.constant dense<0x7FC00000> : tensor<f32>
    %40 = stablehlo.broadcast_in_dim %39, dims = [] : (tensor<f32>) -> tensor<3x3xf32>
    %41 = stablehlo.broadcast_in_dim %38, dims = [0, 1] : (tensor<1x1xi1>) -> tensor<3x3xi1>
    %42 = stablehlo.select %41, %32, %40 : tensor<3x3xi1>, tensor<3x3xf32>
    %43 = call @triu(%18) : (tensor<3x3xf32>) -> tensor<3x3xf32>
    return %42, %43 : tensor<3x3xf32>, tensor<3x3xf32>
  }
  func.func private @triu(%arg0: tensor<3x3xf32>) -> tensor<3x3xf32> {
    %0 = stablehlo.iota dim = 0 : tensor<3x3xi32>
    %1 = stablehlo.constant dense<-1> : tensor<i32>
    %2 = stablehlo.broadcast_in_dim %1, dims = [] : (tensor<i32>) -> tensor<3x3xi32>
    %3 = stablehlo.add %0, %2 : tensor<3x3xi32>
    %4 = stablehlo.iota dim = 1 : tensor<3x3xi32>
    %5 = stablehlo.compare  GE, %3, %4,  SIGNED : (tensor<3x3xi32>, tensor<3x3xi32>) -> tensor<3x3xi1>
    %6 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %7 = stablehlo.broadcast_in_dim %6, dims = [] : (tensor<f32>) -> tensor<3x3xf32>
    %8 = stablehlo.select %5, %7, %arg0 : tensor<3x3xi1>, tensor<3x3xf32>
    return %8 : tensor<3x3xf32>
  }
}
""",
    mlir_module_serialized=b"ML\xefR\x03MLIRxxx-trunk\x00\x01+\x05\x01\x05\x01\x03\x05\x03\x1b\x07\t\x0b\r\x0f\x11\x13\x15\x17\x19\x1b\x1d\x1f\x03\xa2\x02\n\x027\x01\x9b\x0f\x0f\x17\x13\x0b\x0f\x13\x07\x0b\x0b\x0b\x13\x0b\x0b\x0b\x0b\x0b\x13\x0b\x0f\x0b\x0b\x13\x17\x13\x13\x0b\x0b\x0b\x0b\x0b\x0b\x0b\x0b\x13\x13\x13\x13\x1b\x13\x13\x0b33\x0b\x0f\x0b\x13\x0b\x13\x0f\x0b\x1b\x0f\x0b\x0f\x0b\x0f\x0b\x0f\x0b\x0f\x0b\x0bK\x13\x13\x0f\x0b#\x0b\x0b\x0b\x0f\x0b\x0bK\x03g\x0fO/\x0b/\x0b\x0f\x0b\x0b\x0b\x0b\x0b\x0b\x13\x13\x0b\x13\x0b\x0b\x0b\x0b\x0b\x0b\x0f\x1f\x0f\x0f\x0b\x1f\x1f\x1f\x1f\x0b\x1f\x0f\x17\x1b\x0f\x0f\x0f\x0f\x1f\x0b\x1fO/\x0b'\x0f\x17\x17\x01\x05\x17\x0b\x037\x0f\x17\x0f\x07\x07\x07\x07\x17\x13\x17\x17\x07\x0f\x17\x13\x17\x17\x13\x13\x1b\x13\x13\x13\x13\x13\x13\x17\x02\xae\t\x1d\x7f\x05\x1d\x97\x05\x17!\xee\x05\x01\x03\x03\x15\xcd\x05!\x1dY\x05\x03\x03\t\xd7\x1f\x05#\x05%\x05'\x03\x03\t\xf1\x05)\x05+\x05-\x05/\x051\x03\x03%\xc9\x053\x1da\x05\x055\x057\x03\x03\t\xd3\x17!\xea\x05\x01\x03\x03\t\xd5\x03\x03\t\xd9\x059\x05;\x05=\x05?\x05A\x05C\x05E\x05G\x03\x03\x11\xe5\x03\x03\x11\xe7\x03\x03\x11\xe9\x03\x03\t\xed\x03\x05)\xab+\xef\x03\x03\x15\xf3\x03\x03\x13S\x05I\x03\x0b\x19\xa1\x1b\xb3\x1d\xb5\x13\xbf\x1f\xc1\x03\x0b\x19\xa7\x1b\xc5\x1d\xa7\x13\xa9\x1f\xc7\x05K\x1d]\x05\x05M\x03\x03\t\xcb\x05O\x03\x03%\xcf\x1dg\x05\x05Q\x03\x05)\xab+\xd1\x1dm\x05\x05S\x1dq\x05\x05U\x1du\x05\x05W\x1dy/\x05Y\x1d}/\x05[\x05]\x03\x115\xad7\xaf9\xdb;\xa1=\xb1?\xddA\xdfC\xe3\x03\x03\x11\xeb\x03\x03\x15\xf5\x1d\x89\x05\x05_\x03\x07\x8d\xa3\x8f\xa3\x91\xa3\x05a\x05c\x05e\x1d\x95\x05\x05g\x05i\x03\x115\xad7\xaf9\xf7;\xa1=\xb1?\xf9A\xfbC\xff\x1f)\x01\x1f+!\x00\x00\x00\x00\x00\x00\x00\x00\x01\x00\x00\x00\x00\x00\x00\x00\x1f-\x11\x00\x00\x00\x00\x00\x00\x00\x00\x03\x01\x1f\x1d\x11\x00\x00\x00\x00\x00\x00\x00\x00\x1dk\x03\x03\xc3\x1dm\t\x07\x0b\x05\x1do\x05\x01#\x1f\x03\x05\xb7\xbb\r\x03\xa5\xb9\x1dq\r\x03\xa5\xbd\x1ds\x1du\x1dw\r\x01#!\x1dy\x13\x0b\x01\x1f\x01\t\xff\xff\xff\xff\x1f#\x01\x13\x0b\x05\x07\x05\x1f\x05\t\x00\x00\x00\x00\x1f\x01\t\x01\x00\x00\x00\x1f\x01\t\x03\x00\x00\x00\x1f\x01\t`\x00\x00\x00\x1d{\x03\x0b\x9b\x9b\x9b\x9b\x9d\x03\x03\xe1\x15\x03\x01\x11\x01\x03\t\x9d\x9f\x9b\x9f\x13\x07\x01\x13\x07\x05\x13\x07\t\x13\x07\r\x1f\x01\t\x00\x00\x00\x00\x07\x01\x1f\x05\t\x00\x00\xc0\x7f\x1f\x1d!\x00\x00\x00\x00\x00\x00\x00\x00\x01\x00\x00\x00\x00\x00\x00\x00\x1f3\x11\x00\x00\x00\x00\x00\x00\x00\x00\x1d}\x03\x0f\x9b\x9b\x9b\x9b\x9b\x9d\x9f\x03\x03\xfd\x15\x03\x01\x15\x01\x03\x07\x9d\x9b\x9f\x03\x03\x06\x02\xa9\x05\x7f)\x01\x07)\x05\r\r\t)\x01\t\x1b\t\x1d\x01)\x05\r\r\x07)\x03\r\t)\x03\x02\x03\t)\x05\r\r\r\x13)\x01\r)\x05\x05\x05\r)\x03\t\x0b\x11\x01\x05\x03\x03\x11\x03\x03\x03\x03)\x03\x01\x0b)\x03%\t/\t\x03\x11\x01\x13)\x03\x01\x17)\x03\t\x17)\x03\x05\x17)\x03\x05\r)\x03\r\r)\x03\x05\x0b/\x07\x03\x01\x13\x04\xe6\x06\x05\x01\x11\x0fQ\x07\x03\x01\t\x0f\x11\x0fU\x05\x03Y\xb5\x0b\x03w#\x03%\x17\x06{\x03\x03\x03\x01\x03\x03\x011\x03\x01\x03\x03\x01\r\x03\x01\x03\x03\x01\r\x03\x01\x03\x03\x013\x03\x01\x13\x07\x01\x81\x03'\x0b\x05\x07\t\x0b\x03\x07\x07\x01E\x03\x03\x03\r\x07\x07\x01G\x03\x11\x03\r\x07\x07\x01I\x03\x01\x03\r\x07\x07\x01\x83\x03\x13\x03\r\x03\x03\x01K\x03\x01\x05\x07\x01\x07\x03\x01\x03\x17\r\x07\x01M\x03\x19\x05\x13\x19\x05\x07\x01\x07\x03\x1b\x03\x1b\x03\x03\x01\x17\x03\x05\x05\x07\x01\x07\x03\x03\x03\x1f\x05\x07\x01O\x03\x15\x03\x1d\t\x06\x01\x03\x03\x07#\x0f!\x05\x07\x01\x07\x03/\x03\x1b\x03\x03\x01\x17\x03\x05\x05\x07\x01\x07\x03\x11\x03)\x05\x07\x01\x85\x031\x03'\t\x06\x01\x03\x11\x07-\x11+\x03\x03\x87-\x03\x05\x19\x07\x93\x8b\x03\x03\x05%1\x03\x03\x031\x03\x01\x03\x03\x03\r\x03\x01\x03\x03\x03\r\x03\x01\x03\x03\x03\r\x03\x01\x03\x03\x033\x03\x01\x13\x07\x03\x99\x035\x0f579;=3/\x07\x07\x03E\x03\x03\x03?\x07\x07\x03G\x03\x01\x03?\x07\x07\x03I\x03\x13\x03?\x03\x03\x03K\x03\x01\x05\x07\x03\x07\x03\x01\x03G\r\x07\x03M\x03\x19\x05CI\x05\x07\x03\x07\x03\x1b\x03K\x03\x03\x03\x17\x03\x05\x05\x07\x03\x07\x03\x03\x03O\x05\x07\x03O\x03\x15\x03M\t\x06\x03\x03\x03\x07SAQ\x1b\x07\x0b\x02\x02\x03\x03\x03%\x11\x04\x0f\x05UW\x0f\x11\x0bW\x05\x03\x15+\x03\x03\x0f\x0b\x03[#\x03\x0f\x03\x03\x0b_\x03\x01\x05\x07'\x07\x03\x0f\x03\x05\x15\x06'\x03\x0f\x05\x03\x07\x0b\x03ec\x03\x0f\r\x07ki\x03\x15\x05\t\x0b\x03\x03\x0b-\x03\x05\x05\x07o\x07\x03\x03\x03\x0f\t\x06s\x03\x03\x07\r\x11\x01\x11\x04\x0b\x03\x13\x06\x03\x01\x05\x01\x00\xc6\x18\x81\x0f\x1d\x1d\x11\x0f\x0b\t\t\x03\x0b!Y\x87##%_=\x85\x87W\xb3K\x9bM\x9b\xd2\x02\x1b\x1f/!!)#\x1f\x19+\x1b\x1f\x83\x1f\x15\x1d\x15+\x13\r\r\x11\x0f\x17\x0f\x1f\x15\x11\x17\x11\x15+)\x19\x0f\x0b\x11builtin\x00vhlo\x00module\x00constant_v1\x00broadcast_in_dim_v1\x00get_tuple_element_v1\x00select_v1\x00iota_v1\x00compare_v1\x00func_v1\x00return_v1\x00custom_call_v1\x00add_v1\x00reshape_v1\x00pad_v1\x00call_v1\x00value\x00index\x00sym_name\x00broadcast_dimensions\x00arg_attrs\x00function_type\x00res_attrs\x00sym_visibility\x00third_party/py/jax/experimental/jax2tf/tests/back_compat_test.py\x00iota_dimension\x00compare_type\x00comparison_direction\x00api_version\x00backend_config\x00call_target_name\x00called_computations\x00has_side_effect\x00operand_layouts\x00output_operand_aliases\x00result_layouts\x00jit__lambda_\x00jit(<lambda>)/jit(main)/pjit[in_shardings=(UnspecifiedValue,) out_shardings=(UnspecifiedValue,) resource_env=None donated_invars=(False,) name=triu keep_unused=False inline=False]\x00jit(<lambda>)/jit(main)/jit(triu)/iota[dtype=int32 shape=(3, 3) dimension=0]\x00jit(<lambda>)/jit(main)/jit(triu)/add\x00jit(<lambda>)/jit(main)/jit(triu)/iota[dtype=int32 shape=(3, 3) dimension=1]\x00jit(<lambda>)/jit(main)/jit(triu)/ge\x00jit(<lambda>)/jit(main)/jit(triu)/broadcast_in_dim[shape=(3, 3) broadcast_dimensions=()]\x00jit(<lambda>)/jit(main)/jit(triu)/select_n\x00jit(<lambda>)/jit(main)/iota[dtype=float32 shape=(9,) dimension=0]\x00jit(<lambda>)/jit(main)/reshape[new_sizes=(3, 3) dimensions=None]\x00jit(<lambda>)/jit(main)/geqrf\x00jit(<lambda>)/jit(main)/qr[full_matrices=True]\x00edge_padding_high\x00edge_padding_low\x00interior_padding\x00jit(<lambda>)/jit(main)/pad[padding_config=((0, 0, 0), (0, 0, 0))]\x00jit(<lambda>)/jit(main)/householder_product\x00jax.result_info\x00triu\x00\x00[0]\x00[1]\x00main\x00public\x00private\x00lapack_sgeqrf\x00lapack_sorgqr\x00callee\x00",
    xla_call_module_version=4,
)  # End paste


# Pasted from the test output (see back_compat_test.py module docstring)
data_2023_03_17["f64"] = dict(
    testdata_version=1,
    platform='cpu',
    custom_call_targets=['lapack_dgeqrf', 'lapack_dorgqr'],
    serialized_date=datetime.date(2023, 3, 17),
    inputs=(),
    expected_outputs=(array([[ 0.                 ,  0.9128709291752773 ,  0.40824829046386235],
       [-0.447213595499958  ,  0.3651483716701102 , -0.8164965809277263 ],
       [-0.894427190999916  , -0.1825741858350548 ,  0.40824829046386324]]), array([[-6.7082039324993694e+00, -8.0498447189992444e+00,
        -9.3914855054991175e+00],
       [ 0.0000000000000000e+00,  1.0954451150103341e+00,
         2.1908902300206665e+00],
       [ 0.0000000000000000e+00,  0.0000000000000000e+00,
        -8.8817841970012523e-16]])),
    mlir_module_text=r"""
module @jit__lambda_ {
  func.func public @main() -> (tensor<3x3xf64> {jax.result_info = "[0]"}, tensor<3x3xf64> {jax.result_info = "[1]"}) {
    %0 = stablehlo.iota dim = 0 : tensor<9xf64>
    %1 = stablehlo.reshape %0 : (tensor<9xf64>) -> tensor<3x3xf64>
    %2 = stablehlo.constant dense<1> : tensor<i32>
    %3 = stablehlo.constant dense<3> : tensor<i32>
    %4 = stablehlo.constant dense<3> : tensor<i32>
    %5 = stablehlo.constant dense<96> : tensor<i32>
    %6 = stablehlo.custom_call @lapack_dgeqrf(%2, %3, %4, %5, %1) {api_version = 2 : i32, operand_layouts = [dense<> : tensor<0xindex>, dense<> : tensor<0xindex>, dense<> : tensor<0xindex>, dense<> : tensor<0xindex>, dense<[0, 1]> : tensor<2xindex>], output_operand_aliases = [#stablehlo.output_operand_alias<output_tuple_indices = [0], operand_index = 4, operand_tuple_indices = []>], result_layouts = [dense<[0, 1]> : tensor<2xindex>, dense<0> : tensor<1xindex>, dense<> : tensor<0xindex>, dense<0> : tensor<1xindex>]} : (tensor<i32>, tensor<i32>, tensor<i32>, tensor<i32>, tensor<3x3xf64>) -> tuple<tensor<3x3xf64>, tensor<3xf64>, tensor<i32>, tensor<96xf64>>
    %7 = stablehlo.get_tuple_element %6[0] : (tuple<tensor<3x3xf64>, tensor<3xf64>, tensor<i32>, tensor<96xf64>>) -> tensor<3x3xf64>
    %8 = stablehlo.get_tuple_element %6[1] : (tuple<tensor<3x3xf64>, tensor<3xf64>, tensor<i32>, tensor<96xf64>>) -> tensor<3xf64>
    %9 = stablehlo.get_tuple_element %6[2] : (tuple<tensor<3x3xf64>, tensor<3xf64>, tensor<i32>, tensor<96xf64>>) -> tensor<i32>
    %10 = stablehlo.get_tuple_element %6[3] : (tuple<tensor<3x3xf64>, tensor<3xf64>, tensor<i32>, tensor<96xf64>>) -> tensor<96xf64>
    %11 = stablehlo.constant dense<0> : tensor<i32>
    %12 = stablehlo.broadcast_in_dim %11, dims = [] : (tensor<i32>) -> tensor<i32>
    %13 = stablehlo.compare  EQ, %9, %12,  SIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1>
    %14 = stablehlo.broadcast_in_dim %13, dims = [] : (tensor<i1>) -> tensor<1x1xi1>
    %15 = stablehlo.constant dense<0x7FF8000000000000> : tensor<f64>
    %16 = stablehlo.broadcast_in_dim %15, dims = [] : (tensor<f64>) -> tensor<3x3xf64>
    %17 = stablehlo.broadcast_in_dim %14, dims = [0, 1] : (tensor<1x1xi1>) -> tensor<3x3xi1>
    %18 = stablehlo.select %17, %7, %16 : tensor<3x3xi1>, tensor<3x3xf64>
    %19 = stablehlo.broadcast_in_dim %13, dims = [] : (tensor<i1>) -> tensor<1xi1>
    %20 = stablehlo.constant dense<0x7FF8000000000000> : tensor<f64>
    %21 = stablehlo.broadcast_in_dim %20, dims = [] : (tensor<f64>) -> tensor<3xf64>
    %22 = stablehlo.broadcast_in_dim %19, dims = [0] : (tensor<1xi1>) -> tensor<3xi1>
    %23 = stablehlo.select %22, %8, %21 : tensor<3xi1>, tensor<3xf64>
    %24 = stablehlo.constant dense<0.000000e+00> : tensor<f64>
    %25 = stablehlo.pad %18, %24, low = [0, 0], high = [0, 0], interior = [0, 0] : (tensor<3x3xf64>, tensor<f64>) -> tensor<3x3xf64>
    %26 = stablehlo.constant dense<1> : tensor<i32>
    %27 = stablehlo.constant dense<3> : tensor<i32>
    %28 = stablehlo.constant dense<3> : tensor<i32>
    %29 = stablehlo.constant dense<3> : tensor<i32>
    %30 = stablehlo.constant dense<96> : tensor<i32>
    %31 = stablehlo.custom_call @lapack_dorgqr(%26, %27, %28, %29, %30, %25, %23) {api_version = 2 : i32, operand_layouts = [dense<> : tensor<0xindex>, dense<> : tensor<0xindex>, dense<> : tensor<0xindex>, dense<> : tensor<0xindex>, dense<> : tensor<0xindex>, dense<[0, 1]> : tensor<2xindex>, dense<0> : tensor<1xindex>], output_operand_aliases = [#stablehlo.output_operand_alias<output_tuple_indices = [0], operand_index = 5, operand_tuple_indices = []>], result_layouts = [dense<[0, 1]> : tensor<2xindex>, dense<> : tensor<0xindex>, dense<0> : tensor<1xindex>]} : (tensor<i32>, tensor<i32>, tensor<i32>, tensor<i32>, tensor<i32>, tensor<3x3xf64>, tensor<3xf64>) -> tuple<tensor<3x3xf64>, tensor<i32>, tensor<96xf64>>
    %32 = stablehlo.get_tuple_element %31[0] : (tuple<tensor<3x3xf64>, tensor<i32>, tensor<96xf64>>) -> tensor<3x3xf64>
    %33 = stablehlo.get_tuple_element %31[1] : (tuple<tensor<3x3xf64>, tensor<i32>, tensor<96xf64>>) -> tensor<i32>
    %34 = stablehlo.get_tuple_element %31[2] : (tuple<tensor<3x3xf64>, tensor<i32>, tensor<96xf64>>) -> tensor<96xf64>
    %35 = stablehlo.constant dense<0> : tensor<i32>
    %36 = stablehlo.broadcast_in_dim %35, dims = [] : (tensor<i32>) -> tensor<i32>
    %37 = stablehlo.compare  EQ, %33, %36,  SIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1>
    %38 = stablehlo.broadcast_in_dim %37, dims = [] : (tensor<i1>) -> tensor<1x1xi1>
    %39 = stablehlo.constant dense<0x7FF8000000000000> : tensor<f64>
    %40 = stablehlo.broadcast_in_dim %39, dims = [] : (tensor<f64>) -> tensor<3x3xf64>
    %41 = stablehlo.broadcast_in_dim %38, dims = [0, 1] : (tensor<1x1xi1>) -> tensor<3x3xi1>
    %42 = stablehlo.select %41, %32, %40 : tensor<3x3xi1>, tensor<3x3xf64>
    %43 = call @triu(%18) : (tensor<3x3xf64>) -> tensor<3x3xf64>
    return %42, %43 : tensor<3x3xf64>, tensor<3x3xf64>
  }
  func.func private @triu(%arg0: tensor<3x3xf64>) -> tensor<3x3xf64> {
    %0 = stablehlo.iota dim = 0 : tensor<3x3xi32>
    %1 = stablehlo.constant dense<-1> : tensor<i32>
    %2 = stablehlo.broadcast_in_dim %1, dims = [] : (tensor<i32>) -> tensor<3x3xi32>
    %3 = stablehlo.add %0, %2 : tensor<3x3xi32>
    %4 = stablehlo.iota dim = 1 : tensor<3x3xi32>
    %5 = stablehlo.compare  GE, %3, %4,  SIGNED : (tensor<3x3xi32>, tensor<3x3xi32>) -> tensor<3x3xi1>
    %6 = stablehlo.constant dense<0.000000e+00> : tensor<f64>
    %7 = stablehlo.broadcast_in_dim %6, dims = [] : (tensor<f64>) -> tensor<3x3xf64>
    %8 = stablehlo.select %5, %7, %arg0 : tensor<3x3xi1>, tensor<3x3xf64>
    return %8 : tensor<3x3xf64>
  }
}
""",
    mlir_module_serialized=b"ML\xefR\x03MLIRxxx-trunk\x00\x01+\x05\x01\x05\x01\x03\x05\x03\x1b\x07\t\x0b\r\x0f\x11\x13\x15\x17\x19\x1b\x1d\x1f\x03\xa2\x02\n\x027\x01\x9b\x0f\x0f\x17\x13\x0b\x0f\x13\x07\x0b\x0b\x0b\x13\x0b\x0b\x0b\x0b\x0b\x13\x0b\x0f\x0b\x0b\x13\x17\x13\x13\x0b\x0b\x0b\x0b\x0b\x0b\x0b\x0b\x13\x13\x13\x13\x1b\x13\x13\x0b33\x0b\x0f\x0b\x13\x0b\x13\x0f\x0b\x1b\x0f\x0b\x0f\x0b\x0f\x0b\x0f\x0b\x0f\x0b\x0bK\x13\x13\x0f\x0b#\x0b\x0b\x0b\x0f\x0b\x0bK\x03g\x0fO/\x0b/\x0b\x0f\x0b\x0b\x0b\x0b\x0b\x0b\x13\x13\x0b\x13\x0b\x0b\x0b\x0b\x0b\x0b\x0f\x1f\x0f\x0f\x0b/\x1f\x1f\x1f\x0b\x1f\x0f\x17\x1b\x0f\x0f\x0f\x0f\x1f\x0b/O/\x0b'\x0f\x17\x17\x01\x05\x17\x0b\x037\x0f\x17\x0f\x07\x07\x07\x07\x17\x13\x17\x17\x07\x0f\x17\x13\x17\x17\x13\x13\x1b\x13\x13\x13\x13\x13\x13\x17\x02\xce\t\x1d\x7f\x05\x1d\x97\x05\x17!\xee\x05\x01\x03\x03\x15\xcd\x05!\x1dY\x05\x03\x03\t\xd7\x1f\x05#\x05%\x05'\x03\x03\t\xf1\x05)\x05+\x05-\x05/\x051\x03\x03%\xc9\x053\x1da\x05\x055\x057\x03\x03\t\xd3\x17!\xea\x05\x01\x03\x03\t\xd5\x03\x03\t\xd9\x059\x05;\x05=\x05?\x05A\x05C\x05E\x05G\x03\x03\x11\xe5\x03\x03\x11\xe7\x03\x03\x11\xe9\x03\x03\t\xed\x03\x05)\xab+\xef\x03\x03\x15\xf3\x03\x03\x13S\x05I\x03\x0b\x19\xa1\x1b\xb3\x1d\xb5\x13\xbf\x1f\xc1\x03\x0b\x19\xa7\x1b\xc5\x1d\xa7\x13\xa9\x1f\xc7\x05K\x1d]\x05\x05M\x03\x03\t\xcb\x05O\x03\x03%\xcf\x1dg\x05\x05Q\x03\x05)\xab+\xd1\x1dm\x05\x05S\x1dq\x05\x05U\x1du\x05\x05W\x1dy/\x05Y\x1d}/\x05[\x05]\x03\x115\xad7\xaf9\xdb;\xa1=\xb1?\xddA\xdfC\xe3\x03\x03\x11\xeb\x03\x03\x15\xf5\x1d\x89\x05\x05_\x03\x07\x8d\xa3\x8f\xa3\x91\xa3\x05a\x05c\x05e\x1d\x95\x05\x05g\x05i\x03\x115\xad7\xaf9\xf7;\xa1=\xb1?\xf9A\xfbC\xff\x1f)\x01\x1f+!\x00\x00\x00\x00\x00\x00\x00\x00\x01\x00\x00\x00\x00\x00\x00\x00\x1f-\x11\x00\x00\x00\x00\x00\x00\x00\x00\x03\x01\x1f\x1d\x11\x00\x00\x00\x00\x00\x00\x00\x00\x1dk\x03\x03\xc3\x1dm\t\x07\x0b\x05\x1do\x05\x01#\x1f\x03\x05\xb7\xbb\r\x03\xa5\xb9\x1dq\r\x03\xa5\xbd\x1ds\x1du\x1dw\r\x01#!\x1dy\x13\x0b\x01\x1f\x01\t\xff\xff\xff\xff\x1f#\x01\x13\x0b\x05\x07\x05\x1f\x05\x11\x00\x00\x00\x00\x00\x00\x00\x00\x1f\x01\t\x01\x00\x00\x00\x1f\x01\t\x03\x00\x00\x00\x1f\x01\t`\x00\x00\x00\x1d{\x03\x0b\x9b\x9b\x9b\x9b\x9d\x03\x03\xe1\x15\x03\x01\x11\x01\x03\t\x9d\x9f\x9b\x9f\x13\x07\x01\x13\x07\x05\x13\x07\t\x13\x07\r\x1f\x01\t\x00\x00\x00\x00\x07\x01\x1f\x05\x11\x00\x00\x00\x00\x00\x00\xf8\x7f\x1f\x1d!\x00\x00\x00\x00\x00\x00\x00\x00\x01\x00\x00\x00\x00\x00\x00\x00\x1f3\x11\x00\x00\x00\x00\x00\x00\x00\x00\x1d}\x03\x0f\x9b\x9b\x9b\x9b\x9b\x9d\x9f\x03\x03\xfd\x15\x03\x01\x15\x01\x03\x07\x9d\x9b\x9f\x03\x03\x06\x02\xa9\x05\x7f)\x01\x07)\x05\r\r\t)\x01\t\x1b\x0b\x1d\x01)\x05\r\r\x07)\x03\r\t)\x03\x02\x03\t)\x05\r\r\r\x13)\x01\r)\x05\x05\x05\r)\x03\t\x0b\x11\x01\x05\x03\x03\x11\x03\x03\x03\x03)\x03\x01\x0b)\x03%\t/\t\x03\x11\x01\x13)\x03\x01\x17)\x03\t\x17)\x03\x05\x17)\x03\x05\r)\x03\r\r)\x03\x05\x0b/\x07\x03\x01\x13\x04\xe6\x06\x05\x01\x11\x0fQ\x07\x03\x01\t\x0f\x11\x0fU\x05\x03Y\xb5\x0b\x03w#\x03%\x17\x06{\x03\x03\x03\x01\x03\x03\x011\x03\x01\x03\x03\x01\r\x03\x01\x03\x03\x01\r\x03\x01\x03\x03\x013\x03\x01\x13\x07\x01\x81\x03'\x0b\x05\x07\t\x0b\x03\x07\x07\x01E\x03\x03\x03\r\x07\x07\x01G\x03\x11\x03\r\x07\x07\x01I\x03\x01\x03\r\x07\x07\x01\x83\x03\x13\x03\r\x03\x03\x01K\x03\x01\x05\x07\x01\x07\x03\x01\x03\x17\r\x07\x01M\x03\x19\x05\x13\x19\x05\x07\x01\x07\x03\x1b\x03\x1b\x03\x03\x01\x17\x03\x05\x05\x07\x01\x07\x03\x03\x03\x1f\x05\x07\x01O\x03\x15\x03\x1d\t\x06\x01\x03\x03\x07#\x0f!\x05\x07\x01\x07\x03/\x03\x1b\x03\x03\x01\x17\x03\x05\x05\x07\x01\x07\x03\x11\x03)\x05\x07\x01\x85\x031\x03'\t\x06\x01\x03\x11\x07-\x11+\x03\x03\x87-\x03\x05\x19\x07\x93\x8b\x03\x03\x05%1\x03\x03\x031\x03\x01\x03\x03\x03\r\x03\x01\x03\x03\x03\r\x03\x01\x03\x03\x03\r\x03\x01\x03\x03\x033\x03\x01\x13\x07\x03\x99\x035\x0f579;=3/\x07\x07\x03E\x03\x03\x03?\x07\x07\x03G\x03\x01\x03?\x07\x07\x03I\x03\x13\x03?\x03\x03\x03K\x03\x01\x05\x07\x03\x07\x03\x01\x03G\r\x07\x03M\x03\x19\x05CI\x05\x07\x03\x07\x03\x1b\x03K\x03\x03\x03\x17\x03\x05\x05\x07\x03\x07\x03\x03\x03O\x05\x07\x03O\x03\x15\x03M\t\x06\x03\x03\x03\x07SAQ\x1b\x07\x0b\x02\x02\x03\x03\x03%\x11\x04\x0f\x05UW\x0f\x11\x0bW\x05\x03\x15+\x03\x03\x0f\x0b\x03[#\x03\x0f\x03\x03\x0b_\x03\x01\x05\x07'\x07\x03\x0f\x03\x05\x15\x06'\x03\x0f\x05\x03\x07\x0b\x03ec\x03\x0f\r\x07ki\x03\x15\x05\t\x0b\x03\x03\x0b-\x03\x05\x05\x07o\x07\x03\x03\x03\x0f\t\x06s\x03\x03\x07\r\x11\x01\x11\x04\x0b\x03\x13\x06\x03\x01\x05\x01\x00\xc6\x18\x81\x0f\x1d\x1d\x11\x0f\x0b\t\t\x03\x0b!Y\x87##%_=\x85\x87W\xb3K\x9bM\x9b\xd2\x02\x1b\x1f/!!)#\x1f\x19+\x1b\x1f\x83\x1f\x15\x1d\x15+\x13\r\r\x11\x0f\x17\x0f\x1f\x15\x11\x17\x11\x15+)\x19\x0f\x0b\x11builtin\x00vhlo\x00module\x00constant_v1\x00broadcast_in_dim_v1\x00get_tuple_element_v1\x00select_v1\x00iota_v1\x00compare_v1\x00func_v1\x00return_v1\x00custom_call_v1\x00add_v1\x00reshape_v1\x00pad_v1\x00call_v1\x00value\x00index\x00sym_name\x00broadcast_dimensions\x00arg_attrs\x00function_type\x00res_attrs\x00sym_visibility\x00third_party/py/jax/experimental/jax2tf/tests/back_compat_test.py\x00iota_dimension\x00compare_type\x00comparison_direction\x00api_version\x00backend_config\x00call_target_name\x00called_computations\x00has_side_effect\x00operand_layouts\x00output_operand_aliases\x00result_layouts\x00jit__lambda_\x00jit(<lambda>)/jit(main)/pjit[in_shardings=(UnspecifiedValue,) out_shardings=(UnspecifiedValue,) resource_env=None donated_invars=(False,) name=triu keep_unused=False inline=False]\x00jit(<lambda>)/jit(main)/jit(triu)/iota[dtype=int32 shape=(3, 3) dimension=0]\x00jit(<lambda>)/jit(main)/jit(triu)/add\x00jit(<lambda>)/jit(main)/jit(triu)/iota[dtype=int32 shape=(3, 3) dimension=1]\x00jit(<lambda>)/jit(main)/jit(triu)/ge\x00jit(<lambda>)/jit(main)/jit(triu)/broadcast_in_dim[shape=(3, 3) broadcast_dimensions=()]\x00jit(<lambda>)/jit(main)/jit(triu)/select_n\x00jit(<lambda>)/jit(main)/iota[dtype=float64 shape=(9,) dimension=0]\x00jit(<lambda>)/jit(main)/reshape[new_sizes=(3, 3) dimensions=None]\x00jit(<lambda>)/jit(main)/geqrf\x00jit(<lambda>)/jit(main)/qr[full_matrices=True]\x00edge_padding_high\x00edge_padding_low\x00interior_padding\x00jit(<lambda>)/jit(main)/pad[padding_config=((0, 0, 0), (0, 0, 0))]\x00jit(<lambda>)/jit(main)/householder_product\x00jax.result_info\x00triu\x00\x00[0]\x00[1]\x00main\x00public\x00private\x00lapack_dgeqrf\x00lapack_dorgqr\x00callee\x00",
    xla_call_module_version=4,
)  # End paste


# Pasted from the test output (see back_compat_test.py module docstring)
data_2023_03_17["c64"] = dict(
    testdata_version=1,
    platform='cpu',
    custom_call_targets=['lapack_cgeqrf', 'lapack_cungqr'],
    serialized_date=datetime.date(2023, 3, 17),
    inputs=(),
    expected_outputs=(array([[ 0.        +0.j,  0.91287076+0.j,  0.4082487 +0.j],
       [-0.44721356-0.j,  0.36514866+0.j, -0.8164965 +0.j],
       [-0.8944271 -0.j, -0.18257445+0.j,  0.40824816+0.j]],
      dtype=complex64), array([[-6.7082043e+00+0.j, -8.0498438e+00+0.j, -9.3914852e+00+0.j],
       [ 0.0000000e+00+0.j,  1.0954441e+00+0.j,  2.1908894e+00+0.j],
       [ 0.0000000e+00+0.j,  0.0000000e+00+0.j,  7.1525574e-07+0.j]],
      dtype=complex64)),
    mlir_module_text=r"""
module @jit__lambda_ {
  func.func public @main() -> (tensor<3x3xcomplex<f32>> {jax.result_info = "[0]"}, tensor<3x3xcomplex<f32>> {jax.result_info = "[1]"}) {
    %0 = stablehlo.iota dim = 0 : tensor<9xcomplex<f32>>
    %1 = stablehlo.reshape %0 : (tensor<9xcomplex<f32>>) -> tensor<3x3xcomplex<f32>>
    %2 = stablehlo.constant dense<1> : tensor<i32>
    %3 = stablehlo.constant dense<3> : tensor<i32>
    %4 = stablehlo.constant dense<3> : tensor<i32>
    %5 = stablehlo.constant dense<96> : tensor<i32>
    %6 = stablehlo.custom_call @lapack_cgeqrf(%2, %3, %4, %5, %1) {api_version = 2 : i32, operand_layouts = [dense<> : tensor<0xindex>, dense<> : tensor<0xindex>, dense<> : tensor<0xindex>, dense<> : tensor<0xindex>, dense<[0, 1]> : tensor<2xindex>], output_operand_aliases = [#stablehlo.output_operand_alias<output_tuple_indices = [0], operand_index = 4, operand_tuple_indices = []>], result_layouts = [dense<[0, 1]> : tensor<2xindex>, dense<0> : tensor<1xindex>, dense<> : tensor<0xindex>, dense<0> : tensor<1xindex>]} : (tensor<i32>, tensor<i32>, tensor<i32>, tensor<i32>, tensor<3x3xcomplex<f32>>) -> tuple<tensor<3x3xcomplex<f32>>, tensor<3xcomplex<f32>>, tensor<i32>, tensor<96xcomplex<f32>>>
    %7 = stablehlo.get_tuple_element %6[0] : (tuple<tensor<3x3xcomplex<f32>>, tensor<3xcomplex<f32>>, tensor<i32>, tensor<96xcomplex<f32>>>) -> tensor<3x3xcomplex<f32>>
    %8 = stablehlo.get_tuple_element %6[1] : (tuple<tensor<3x3xcomplex<f32>>, tensor<3xcomplex<f32>>, tensor<i32>, tensor<96xcomplex<f32>>>) -> tensor<3xcomplex<f32>>
    %9 = stablehlo.get_tuple_element %6[2] : (tuple<tensor<3x3xcomplex<f32>>, tensor<3xcomplex<f32>>, tensor<i32>, tensor<96xcomplex<f32>>>) -> tensor<i32>
    %10 = stablehlo.get_tuple_element %6[3] : (tuple<tensor<3x3xcomplex<f32>>, tensor<3xcomplex<f32>>, tensor<i32>, tensor<96xcomplex<f32>>>) -> tensor<96xcomplex<f32>>
    %11 = stablehlo.constant dense<0> : tensor<i32>
    %12 = stablehlo.broadcast_in_dim %11, dims = [] : (tensor<i32>) -> tensor<i32>
    %13 = stablehlo.compare  EQ, %9, %12,  SIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1>
    %14 = stablehlo.broadcast_in_dim %13, dims = [] : (tensor<i1>) -> tensor<1x1xi1>
    %15 = stablehlo.constant dense<(0x7FC00000,0x7FC00000)> : tensor<complex<f32>>
    %16 = stablehlo.broadcast_in_dim %15, dims = [] : (tensor<complex<f32>>) -> tensor<3x3xcomplex<f32>>
    %17 = stablehlo.broadcast_in_dim %14, dims = [0, 1] : (tensor<1x1xi1>) -> tensor<3x3xi1>
    %18 = stablehlo.select %17, %7, %16 : tensor<3x3xi1>, tensor<3x3xcomplex<f32>>
    %19 = stablehlo.broadcast_in_dim %13, dims = [] : (tensor<i1>) -> tensor<1xi1>
    %20 = stablehlo.constant dense<(0x7FC00000,0x7FC00000)> : tensor<complex<f32>>
    %21 = stablehlo.broadcast_in_dim %20, dims = [] : (tensor<complex<f32>>) -> tensor<3xcomplex<f32>>
    %22 = stablehlo.broadcast_in_dim %19, dims = [0] : (tensor<1xi1>) -> tensor<3xi1>
    %23 = stablehlo.select %22, %8, %21 : tensor<3xi1>, tensor<3xcomplex<f32>>
    %24 = stablehlo.constant dense<(0.000000e+00,0.000000e+00)> : tensor<complex<f32>>
    %25 = stablehlo.pad %18, %24, low = [0, 0], high = [0, 0], interior = [0, 0] : (tensor<3x3xcomplex<f32>>, tensor<complex<f32>>) -> tensor<3x3xcomplex<f32>>
    %26 = stablehlo.constant dense<1> : tensor<i32>
    %27 = stablehlo.constant dense<3> : tensor<i32>
    %28 = stablehlo.constant dense<3> : tensor<i32>
    %29 = stablehlo.constant dense<3> : tensor<i32>
    %30 = stablehlo.constant dense<96> : tensor<i32>
    %31 = stablehlo.custom_call @lapack_cungqr(%26, %27, %28, %29, %30, %25, %23) {api_version = 2 : i32, operand_layouts = [dense<> : tensor<0xindex>, dense<> : tensor<0xindex>, dense<> : tensor<0xindex>, dense<> : tensor<0xindex>, dense<> : tensor<0xindex>, dense<[0, 1]> : tensor<2xindex>, dense<0> : tensor<1xindex>], output_operand_aliases = [#stablehlo.output_operand_alias<output_tuple_indices = [0], operand_index = 5, operand_tuple_indices = []>], result_layouts = [dense<[0, 1]> : tensor<2xindex>, dense<> : tensor<0xindex>, dense<0> : tensor<1xindex>]} : (tensor<i32>, tensor<i32>, tensor<i32>, tensor<i32>, tensor<i32>, tensor<3x3xcomplex<f32>>, tensor<3xcomplex<f32>>) -> tuple<tensor<3x3xcomplex<f32>>, tensor<i32>, tensor<96xcomplex<f32>>>
    %32 = stablehlo.get_tuple_element %31[0] : (tuple<tensor<3x3xcomplex<f32>>, tensor<i32>, tensor<96xcomplex<f32>>>) -> tensor<3x3xcomplex<f32>>
    %33 = stablehlo.get_tuple_element %31[1] : (tuple<tensor<3x3xcomplex<f32>>, tensor<i32>, tensor<96xcomplex<f32>>>) -> tensor<i32>
    %34 = stablehlo.get_tuple_element %31[2] : (tuple<tensor<3x3xcomplex<f32>>, tensor<i32>, tensor<96xcomplex<f32>>>) -> tensor<96xcomplex<f32>>
    %35 = stablehlo.constant dense<0> : tensor<i32>
    %36 = stablehlo.broadcast_in_dim %35, dims = [] : (tensor<i32>) -> tensor<i32>
    %37 = stablehlo.compare  EQ, %33, %36,  SIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1>
    %38 = stablehlo.broadcast_in_dim %37, dims = [] : (tensor<i1>) -> tensor<1x1xi1>
    %39 = stablehlo.constant dense<(0x7FC00000,0x7FC00000)> : tensor<complex<f32>>
    %40 = stablehlo.broadcast_in_dim %39, dims = [] : (tensor<complex<f32>>) -> tensor<3x3xcomplex<f32>>
    %41 = stablehlo.broadcast_in_dim %38, dims = [0, 1] : (tensor<1x1xi1>) -> tensor<3x3xi1>
    %42 = stablehlo.select %41, %32, %40 : tensor<3x3xi1>, tensor<3x3xcomplex<f32>>
    %43 = call @triu(%18) : (tensor<3x3xcomplex<f32>>) -> tensor<3x3xcomplex<f32>>
    return %42, %43 : tensor<3x3xcomplex<f32>>, tensor<3x3xcomplex<f32>>
  }
  func.func private @triu(%arg0: tensor<3x3xcomplex<f32>>) -> tensor<3x3xcomplex<f32>> {
    %0 = stablehlo.iota dim = 0 : tensor<3x3xi32>
    %1 = stablehlo.constant dense<-1> : tensor<i32>
    %2 = stablehlo.broadcast_in_dim %1, dims = [] : (tensor<i32>) -> tensor<3x3xi32>
    %3 = stablehlo.add %0, %2 : tensor<3x3xi32>
    %4 = stablehlo.iota dim = 1 : tensor<3x3xi32>
    %5 = stablehlo.compare  GE, %3, %4,  SIGNED : (tensor<3x3xi32>, tensor<3x3xi32>) -> tensor<3x3xi1>
    %6 = stablehlo.constant dense<(0.000000e+00,0.000000e+00)> : tensor<complex<f32>>
    %7 = stablehlo.broadcast_in_dim %6, dims = [] : (tensor<complex<f32>>) -> tensor<3x3xcomplex<f32>>
    %8 = stablehlo.select %5, %7, %arg0 : tensor<3x3xi1>, tensor<3x3xcomplex<f32>>
    return %8 : tensor<3x3xcomplex<f32>>
  }
}
""",
    mlir_module_serialized=b"ML\xefR\x03MLIRxxx-trunk\x00\x01+\x05\x01\x05\x01\x03\x05\x03\x1b\x07\t\x0b\r\x0f\x11\x13\x15\x17\x19\x1b\x1d\x1f\x03\xa6\x02\n\x029\x01\x9b\x0f\x0f\x17\x13\x0b\x0f\x13\x07\x0b\x0b\x0b\x13\x0b\x0b\x0b\x0b\x0b\x13\x0b\x0f\x0b\x0b\x13\x17\x13\x13\x0b\x0b\x0b\x0b\x0b\x0b\x0b\x0b\x13\x13\x13\x13\x1b\x13\x13\x0b33\x0b\x0f\x0b\x13\x0b\x13\x0f\x0b\x1b\x0f\x0b\x0f\x0b\x0f\x0b\x0f\x0b\x0f\x0b\x0bK\x13\x13\x0f\x0b#\x0b\x0b\x0b\x0f\x0b\x0bK\x03g\x0fO/\x0b/\x0b\x0f\x0b\x0b\x0b\x0b\x0b\x0b\x13\x13\x0b\x13\x0b\x0b\x0b\x0b\x0b\x0b\x0f\x1f\x0f\x0f\x0b/\x1f\x1f\x1f\x0b\x1f\x0f\x17\x1b\x0f\x0f\x0f\x0f\x1f\x0b/O/\x0b'\x0f\x17\x17\x01\x05\x17\x0b\x039\x0f\x17\x0f\x07\x0b\x07\x07\x17\x13\x17\x17\x07\x0f\x17\x13\x17\x07\x17\x13\x13\x1b\x13\x13\x13\x13\x13\x13\x17\x02\xd6\t\x1d\x7f\x05\x1d\x97\x05\x17!\xee\x05\x01\x03\x03\x15\xcd\x05!\x1dY\x05\x03\x03\t\xd7\x1f\x05#\x05%\x05'\x03\x03\t\xf1\x05)\x05+\x05-\x05/\x051\x03\x03%\xc9\x053\x1da\x05\x055\x057\x03\x03\t\xd3\x17!\xea\x05\x01\x03\x03\t\xd5\x03\x03\t\xd9\x059\x05;\x05=\x05?\x05A\x05C\x05E\x05G\x03\x03\x11\xe5\x03\x03\x11\xe7\x03\x03\x11\xe9\x03\x03\t\xed\x03\x05)\xab+\xef\x03\x03\x15\xf3\x03\x03\x13S\x05I\x03\x0b\x19\xa1\x1b\xb3\x1d\xb5\x13\xbf\x1f\xc1\x03\x0b\x19\xa7\x1b\xc5\x1d\xa7\x13\xa9\x1f\xc7\x05K\x1d]\x05\x05M\x03\x03\t\xcb\x05O\x03\x03%\xcf\x1dg\x05\x05Q\x03\x05)\xab+\xd1\x1dm\x05\x05S\x1dq\x05\x05U\x1du\x05\x05W\x1dy/\x05Y\x1d}/\x05[\x05]\x03\x115\xad7\xaf9\xdb;\xa1=\xb1?\xddA\xdfC\xe3\x03\x03\x11\xeb\x03\x03\x15\xf5\x1d\x89\x05\x05_\x03\x07\x8d\xa3\x8f\xa3\x91\xa3\x05a\x05c\x05e\x1d\x95\x05\x05g\x05i\x03\x115\xad7\xaf9\xf7;\xa1=\xb1?\xf9A\xfbC\xff\x1f+\x01\x1f-!\x00\x00\x00\x00\x00\x00\x00\x00\x01\x00\x00\x00\x00\x00\x00\x00\x1f/\x11\x00\x00\x00\x00\x00\x00\x00\x00\x03\x01\x1f\x1d\x11\x00\x00\x00\x00\x00\x00\x00\x00\x1dk\x03\x03\xc3\x1dm\t\x07\x0b\x05\x1do\x05\x01#\x1f\x03\x05\xb7\xbb\r\x03\xa5\xb9\x1dq\r\x03\xa5\xbd\x1ds\x1du\x1dw\r\x01##\x1dy\x13\x0b\x01\x1f\x01\t\xff\xff\xff\xff\x1f%\x01\x13\x0b\x05\x07\x05\x1f\x05\x11\x00\x00\x00\x00\x00\x00\x00\x00\x1f\x01\t\x01\x00\x00\x00\x1f\x01\t\x03\x00\x00\x00\x1f\x01\t`\x00\x00\x00\x1d{\x03\x0b\x9b\x9b\x9b\x9b\x9d\x03\x03\xe1\x15\x03\x01\x11\x01\x03\t\x9d\x9f\x9b\x9f\x13\x07\x01\x13\x07\x05\x13\x07\t\x13\x07\r\x1f\x01\t\x00\x00\x00\x00\x07\x01\x1f\x05\x11\x00\x00\xc0\x7f\x00\x00\xc0\x7f\x1f\x1d!\x00\x00\x00\x00\x00\x00\x00\x00\x01\x00\x00\x00\x00\x00\x00\x00\x1f5\x11\x00\x00\x00\x00\x00\x00\x00\x00\x1d}\x03\x0f\x9b\x9b\x9b\x9b\x9b\x9d\x9f\x03\x03\xfd\x15\x03\x01\x15\x01\x03\x07\x9d\x9b\x9f\x03\x03\x06\x02\xa9\x05\x7f)\x01\x07)\x05\r\r\t)\x01\t\x1b\x03!\x1d\x01)\x05\r\r\x07)\x03\r\t)\x03\x02\x03\t)\x05\r\r\r\x13)\x01\r)\x05\x05\x05\r)\x03\t\x0b\x11\x01\x05\x03\x03\t\x11\x03\x03\x03\x03)\x03\x01\x0b)\x03%\t/\t\x03\x11\x01\x13)\x03\x01\x17)\x03\t\x17)\x03\x05\x17)\x03\x05\r)\x03\r\r)\x03\x05\x0b/\x07\x03\x01\x13\x04\xe6\x06\x05\x01\x11\x0fQ\x07\x03\x01\t\x0f\x11\x0fU\x05\x03Y\xb5\x0b\x03w#\x03'\x17\x06{\x03\x03\x03\x01\x03\x03\x011\x03\x01\x03\x03\x01\r\x03\x01\x03\x03\x01\r\x03\x01\x03\x03\x013\x03\x01\x13\x07\x01\x81\x03)\x0b\x05\x07\t\x0b\x03\x07\x07\x01E\x03\x03\x03\r\x07\x07\x01G\x03\x11\x03\r\x07\x07\x01I\x03\x01\x03\r\x07\x07\x01\x83\x03\x13\x03\r\x03\x03\x01K\x03\x01\x05\x07\x01\x07\x03\x01\x03\x17\r\x07\x01M\x03\x19\x05\x13\x19\x05\x07\x01\x07\x03\x1b\x03\x1b\x03\x03\x01\x17\x03\x05\x05\x07\x01\x07\x03\x03\x03\x1f\x05\x07\x01O\x03\x15\x03\x1d\t\x06\x01\x03\x03\x07#\x0f!\x05\x07\x01\x07\x031\x03\x1b\x03\x03\x01\x17\x03\x05\x05\x07\x01\x07\x03\x11\x03)\x05\x07\x01\x85\x033\x03'\t\x06\x01\x03\x11\x07-\x11+\x03\x03\x87-\x03\x05\x19\x07\x93\x8b\x03\x03\x05%1\x03\x03\x031\x03\x01\x03\x03\x03\r\x03\x01\x03\x03\x03\r\x03\x01\x03\x03\x03\r\x03\x01\x03\x03\x033\x03\x01\x13\x07\x03\x99\x037\x0f579;=3/\x07\x07\x03E\x03\x03\x03?\x07\x07\x03G\x03\x01\x03?\x07\x07\x03I\x03\x13\x03?\x03\x03\x03K\x03\x01\x05\x07\x03\x07\x03\x01\x03G\r\x07\x03M\x03\x19\x05CI\x05\x07\x03\x07\x03\x1b\x03K\x03\x03\x03\x17\x03\x05\x05\x07\x03\x07\x03\x03\x03O\x05\x07\x03O\x03\x15\x03M\t\x06\x03\x03\x03\x07SAQ\x1b\x07\x0b\x02\x02\x03\x03\x03%\x11\x04\x0f\x05UW\x0f\x11\x0bW\x05\x03\x15+\x03\x03\x0f\x0b\x03[#\x03\x0f\x03\x03\x0b_\x03\x01\x05\x07'\x07\x03\x0f\x03\x05\x15\x06'\x03\x0f\x05\x03\x07\x0b\x03ec\x03\x0f\r\x07ki\x03\x15\x05\t\x0b\x03\x03\x0b-\x03\x05\x05\x07o\x07\x03\x03\x03\x0f\t\x06s\x03\x03\x07\r\x11\x01\x11\x04\x0b\x03\x13\x06\x03\x01\x05\x01\x00\xce\x18\x81\x0f\x1d\x1d\x11\x0f\x0b\t\t\x03\x0b!Y\x87##%_=\x85\x8bW\xb3K\x9bM\x9b\xd2\x02\x1b\x1f/!!)#\x1f\x19+\x1b\x1f\x83\x1f\x15\x1d\x15+\x13\r\r\x11\x0f\x17\x0f\x1f\x15\x11\x17\x11\x15+)\x19\x0f\x0b\x11builtin\x00vhlo\x00module\x00constant_v1\x00broadcast_in_dim_v1\x00get_tuple_element_v1\x00select_v1\x00iota_v1\x00compare_v1\x00func_v1\x00return_v1\x00custom_call_v1\x00add_v1\x00reshape_v1\x00pad_v1\x00call_v1\x00value\x00index\x00sym_name\x00broadcast_dimensions\x00arg_attrs\x00function_type\x00res_attrs\x00sym_visibility\x00third_party/py/jax/experimental/jax2tf/tests/back_compat_test.py\x00iota_dimension\x00compare_type\x00comparison_direction\x00api_version\x00backend_config\x00call_target_name\x00called_computations\x00has_side_effect\x00operand_layouts\x00output_operand_aliases\x00result_layouts\x00jit__lambda_\x00jit(<lambda>)/jit(main)/pjit[in_shardings=(UnspecifiedValue,) out_shardings=(UnspecifiedValue,) resource_env=None donated_invars=(False,) name=triu keep_unused=False inline=False]\x00jit(<lambda>)/jit(main)/jit(triu)/iota[dtype=int32 shape=(3, 3) dimension=0]\x00jit(<lambda>)/jit(main)/jit(triu)/add\x00jit(<lambda>)/jit(main)/jit(triu)/iota[dtype=int32 shape=(3, 3) dimension=1]\x00jit(<lambda>)/jit(main)/jit(triu)/ge\x00jit(<lambda>)/jit(main)/jit(triu)/broadcast_in_dim[shape=(3, 3) broadcast_dimensions=()]\x00jit(<lambda>)/jit(main)/jit(triu)/select_n\x00jit(<lambda>)/jit(main)/iota[dtype=complex64 shape=(9,) dimension=0]\x00jit(<lambda>)/jit(main)/reshape[new_sizes=(3, 3) dimensions=None]\x00jit(<lambda>)/jit(main)/geqrf\x00jit(<lambda>)/jit(main)/qr[full_matrices=True]\x00edge_padding_high\x00edge_padding_low\x00interior_padding\x00jit(<lambda>)/jit(main)/pad[padding_config=((0, 0, 0), (0, 0, 0))]\x00jit(<lambda>)/jit(main)/householder_product\x00jax.result_info\x00triu\x00\x00[0]\x00[1]\x00main\x00public\x00private\x00lapack_cgeqrf\x00lapack_cungqr\x00callee\x00",
    xla_call_module_version=4,
)  # End paste


# Pasted from the test output (see back_compat_test.py module docstring)
data_2023_03_17["c128"] = dict(
    testdata_version=1,
    platform='cpu',
    custom_call_targets=['lapack_zgeqrf', 'lapack_zungqr'],
    serialized_date=datetime.date(2023, 3, 17),
    inputs=(),
    expected_outputs=(array([[ 0.                 +0.j,  0.9128709291752773 +0.j,
         0.40824829046386235+0.j],
       [-0.447213595499958  -0.j,  0.3651483716701102 +0.j,
        -0.8164965809277263 +0.j],
       [-0.894427190999916  -0.j, -0.1825741858350548 +0.j,
         0.40824829046386324+0.j]]), array([[-6.7082039324993694e+00+0.j, -8.0498447189992444e+00+0.j,
        -9.3914855054991175e+00+0.j],
       [ 0.0000000000000000e+00+0.j,  1.0954451150103341e+00+0.j,
         2.1908902300206665e+00+0.j],
       [ 0.0000000000000000e+00+0.j,  0.0000000000000000e+00+0.j,
        -8.8817841970012523e-16+0.j]])),
    mlir_module_text=r"""
module @jit__lambda_ {
  func.func public @main() -> (tensor<3x3xcomplex<f64>> {jax.result_info = "[0]"}, tensor<3x3xcomplex<f64>> {jax.result_info = "[1]"}) {
    %0 = stablehlo.iota dim = 0 : tensor<9xcomplex<f64>>
    %1 = stablehlo.reshape %0 : (tensor<9xcomplex<f64>>) -> tensor<3x3xcomplex<f64>>
    %2 = stablehlo.constant dense<1> : tensor<i32>
    %3 = stablehlo.constant dense<3> : tensor<i32>
    %4 = stablehlo.constant dense<3> : tensor<i32>
    %5 = stablehlo.constant dense<96> : tensor<i32>
    %6 = stablehlo.custom_call @lapack_zgeqrf(%2, %3, %4, %5, %1) {api_version = 2 : i32, operand_layouts = [dense<> : tensor<0xindex>, dense<> : tensor<0xindex>, dense<> : tensor<0xindex>, dense<> : tensor<0xindex>, dense<[0, 1]> : tensor<2xindex>], output_operand_aliases = [#stablehlo.output_operand_alias<output_tuple_indices = [0], operand_index = 4, operand_tuple_indices = []>], result_layouts = [dense<[0, 1]> : tensor<2xindex>, dense<0> : tensor<1xindex>, dense<> : tensor<0xindex>, dense<0> : tensor<1xindex>]} : (tensor<i32>, tensor<i32>, tensor<i32>, tensor<i32>, tensor<3x3xcomplex<f64>>) -> tuple<tensor<3x3xcomplex<f64>>, tensor<3xcomplex<f64>>, tensor<i32>, tensor<96xcomplex<f64>>>
    %7 = stablehlo.get_tuple_element %6[0] : (tuple<tensor<3x3xcomplex<f64>>, tensor<3xcomplex<f64>>, tensor<i32>, tensor<96xcomplex<f64>>>) -> tensor<3x3xcomplex<f64>>
    %8 = stablehlo.get_tuple_element %6[1] : (tuple<tensor<3x3xcomplex<f64>>, tensor<3xcomplex<f64>>, tensor<i32>, tensor<96xcomplex<f64>>>) -> tensor<3xcomplex<f64>>
    %9 = stablehlo.get_tuple_element %6[2] : (tuple<tensor<3x3xcomplex<f64>>, tensor<3xcomplex<f64>>, tensor<i32>, tensor<96xcomplex<f64>>>) -> tensor<i32>
    %10 = stablehlo.get_tuple_element %6[3] : (tuple<tensor<3x3xcomplex<f64>>, tensor<3xcomplex<f64>>, tensor<i32>, tensor<96xcomplex<f64>>>) -> tensor<96xcomplex<f64>>
    %11 = stablehlo.constant dense<0> : tensor<i32>
    %12 = stablehlo.broadcast_in_dim %11, dims = [] : (tensor<i32>) -> tensor<i32>
    %13 = stablehlo.compare  EQ, %9, %12,  SIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1>
    %14 = stablehlo.broadcast_in_dim %13, dims = [] : (tensor<i1>) -> tensor<1x1xi1>
    %15 = stablehlo.constant dense<(0x7FF8000000000000,0x7FF8000000000000)> : tensor<complex<f64>>
    %16 = stablehlo.broadcast_in_dim %15, dims = [] : (tensor<complex<f64>>) -> tensor<3x3xcomplex<f64>>
    %17 = stablehlo.broadcast_in_dim %14, dims = [0, 1] : (tensor<1x1xi1>) -> tensor<3x3xi1>
    %18 = stablehlo.select %17, %7, %16 : tensor<3x3xi1>, tensor<3x3xcomplex<f64>>
    %19 = stablehlo.broadcast_in_dim %13, dims = [] : (tensor<i1>) -> tensor<1xi1>
    %20 = stablehlo.constant dense<(0x7FF8000000000000,0x7FF8000000000000)> : tensor<complex<f64>>
    %21 = stablehlo.broadcast_in_dim %20, dims = [] : (tensor<complex<f64>>) -> tensor<3xcomplex<f64>>
    %22 = stablehlo.broadcast_in_dim %19, dims = [0] : (tensor<1xi1>) -> tensor<3xi1>
    %23 = stablehlo.select %22, %8, %21 : tensor<3xi1>, tensor<3xcomplex<f64>>
    %24 = stablehlo.constant dense<(0.000000e+00,0.000000e+00)> : tensor<complex<f64>>
    %25 = stablehlo.pad %18, %24, low = [0, 0], high = [0, 0], interior = [0, 0] : (tensor<3x3xcomplex<f64>>, tensor<complex<f64>>) -> tensor<3x3xcomplex<f64>>
    %26 = stablehlo.constant dense<1> : tensor<i32>
    %27 = stablehlo.constant dense<3> : tensor<i32>
    %28 = stablehlo.constant dense<3> : tensor<i32>
    %29 = stablehlo.constant dense<3> : tensor<i32>
    %30 = stablehlo.constant dense<96> : tensor<i32>
    %31 = stablehlo.custom_call @lapack_zungqr(%26, %27, %28, %29, %30, %25, %23) {api_version = 2 : i32, operand_layouts = [dense<> : tensor<0xindex>, dense<> : tensor<0xindex>, dense<> : tensor<0xindex>, dense<> : tensor<0xindex>, dense<> : tensor<0xindex>, dense<[0, 1]> : tensor<2xindex>, dense<0> : tensor<1xindex>], output_operand_aliases = [#stablehlo.output_operand_alias<output_tuple_indices = [0], operand_index = 5, operand_tuple_indices = []>], result_layouts = [dense<[0, 1]> : tensor<2xindex>, dense<> : tensor<0xindex>, dense<0> : tensor<1xindex>]} : (tensor<i32>, tensor<i32>, tensor<i32>, tensor<i32>, tensor<i32>, tensor<3x3xcomplex<f64>>, tensor<3xcomplex<f64>>) -> tuple<tensor<3x3xcomplex<f64>>, tensor<i32>, tensor<96xcomplex<f64>>>
    %32 = stablehlo.get_tuple_element %31[0] : (tuple<tensor<3x3xcomplex<f64>>, tensor<i32>, tensor<96xcomplex<f64>>>) -> tensor<3x3xcomplex<f64>>
    %33 = stablehlo.get_tuple_element %31[1] : (tuple<tensor<3x3xcomplex<f64>>, tensor<i32>, tensor<96xcomplex<f64>>>) -> tensor<i32>
    %34 = stablehlo.get_tuple_element %31[2] : (tuple<tensor<3x3xcomplex<f64>>, tensor<i32>, tensor<96xcomplex<f64>>>) -> tensor<96xcomplex<f64>>
    %35 = stablehlo.constant dense<0> : tensor<i32>
    %36 = stablehlo.broadcast_in_dim %35, dims = [] : (tensor<i32>) -> tensor<i32>
    %37 = stablehlo.compare  EQ, %33, %36,  SIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1>
    %38 = stablehlo.broadcast_in_dim %37, dims = [] : (tensor<i1>) -> tensor<1x1xi1>
    %39 = stablehlo.constant dense<(0x7FF8000000000000,0x7FF8000000000000)> : tensor<complex<f64>>
    %40 = stablehlo.broadcast_in_dim %39, dims = [] : (tensor<complex<f64>>) -> tensor<3x3xcomplex<f64>>
    %41 = stablehlo.broadcast_in_dim %38, dims = [0, 1] : (tensor<1x1xi1>) -> tensor<3x3xi1>
    %42 = stablehlo.select %41, %32, %40 : tensor<3x3xi1>, tensor<3x3xcomplex<f64>>
    %43 = call @triu(%18) : (tensor<3x3xcomplex<f64>>) -> tensor<3x3xcomplex<f64>>
    return %42, %43 : tensor<3x3xcomplex<f64>>, tensor<3x3xcomplex<f64>>
  }
  func.func private @triu(%arg0: tensor<3x3xcomplex<f64>>) -> tensor<3x3xcomplex<f64>> {
    %0 = stablehlo.iota dim = 0 : tensor<3x3xi32>
    %1 = stablehlo.constant dense<-1> : tensor<i32>
    %2 = stablehlo.broadcast_in_dim %1, dims = [] : (tensor<i32>) -> tensor<3x3xi32>
    %3 = stablehlo.add %0, %2 : tensor<3x3xi32>
    %4 = stablehlo.iota dim = 1 : tensor<3x3xi32>
    %5 = stablehlo.compare  GE, %3, %4,  SIGNED : (tensor<3x3xi32>, tensor<3x3xi32>) -> tensor<3x3xi1>
    %6 = stablehlo.constant dense<(0.000000e+00,0.000000e+00)> : tensor<complex<f64>>
    %7 = stablehlo.broadcast_in_dim %6, dims = [] : (tensor<complex<f64>>) -> tensor<3x3xcomplex<f64>>
    %8 = stablehlo.select %5, %7, %arg0 : tensor<3x3xi1>, tensor<3x3xcomplex<f64>>
    return %8 : tensor<3x3xcomplex<f64>>
  }
}
""",
    mlir_module_serialized=b"ML\xefR\x03MLIRxxx-trunk\x00\x01+\x05\x01\x05\x01\x03\x05\x03\x1b\x07\t\x0b\r\x0f\x11\x13\x15\x17\x19\x1b\x1d\x1f\x03\xa6\x02\n\x029\x01\x9b\x0f\x0f\x17\x13\x0b\x0f\x13\x07\x0b\x0b\x0b\x13\x0b\x0b\x0b\x0b\x0b\x13\x0b\x0f\x0b\x0b\x13\x17\x13\x13\x0b\x0b\x0b\x0b\x0b\x0b\x0b\x0b\x13\x13\x13\x13\x1b\x13\x13\x0b33\x0b\x0f\x0b\x13\x0b\x13\x0f\x0b\x1b\x0f\x0b\x0f\x0b\x0f\x0b\x0f\x0b\x0f\x0b\x0bK\x13\x13\x0f\x0b#\x0b\x0b\x0b\x0f\x0b\x0bK\x03g\x0fO/\x0b/\x0b\x0f\x0b\x0b\x0b\x0b\x0b\x0b\x13\x13\x0b\x13\x0b\x0b\x0b\x0b\x0b\x0b\x0f\x1f\x0f\x0f\x0bO\x1f\x1f\x1f\x0b\x1f\x0f\x17\x1b\x0f\x0f\x0f\x0f\x1f\x0bOO/\x0b'\x0f\x17\x17\x01\x05\x17\x0b\x039\x0f\x17\x0f\x07\x0b\x07\x07\x17\x13\x17\x17\x07\x0f\x17\x13\x17\x07\x17\x13\x13\x1b\x13\x13\x13\x13\x13\x13\x17\x02\x16\n\x1d\x7f\x05\x1d\x97\x05\x17!\xee\x05\x01\x03\x03\x15\xcd\x05!\x1dY\x05\x03\x03\t\xd7\x1f\x05#\x05%\x05'\x03\x03\t\xf1\x05)\x05+\x05-\x05/\x051\x03\x03%\xc9\x053\x1da\x05\x055\x057\x03\x03\t\xd3\x17!\xea\x05\x01\x03\x03\t\xd5\x03\x03\t\xd9\x059\x05;\x05=\x05?\x05A\x05C\x05E\x05G\x03\x03\x11\xe5\x03\x03\x11\xe7\x03\x03\x11\xe9\x03\x03\t\xed\x03\x05)\xab+\xef\x03\x03\x15\xf3\x03\x03\x13S\x05I\x03\x0b\x19\xa1\x1b\xb3\x1d\xb5\x13\xbf\x1f\xc1\x03\x0b\x19\xa7\x1b\xc5\x1d\xa7\x13\xa9\x1f\xc7\x05K\x1d]\x05\x05M\x03\x03\t\xcb\x05O\x03\x03%\xcf\x1dg\x05\x05Q\x03\x05)\xab+\xd1\x1dm\x05\x05S\x1dq\x05\x05U\x1du\x05\x05W\x1dy/\x05Y\x1d}/\x05[\x05]\x03\x115\xad7\xaf9\xdb;\xa1=\xb1?\xddA\xdfC\xe3\x03\x03\x11\xeb\x03\x03\x15\xf5\x1d\x89\x05\x05_\x03\x07\x8d\xa3\x8f\xa3\x91\xa3\x05a\x05c\x05e\x1d\x95\x05\x05g\x05i\x03\x115\xad7\xaf9\xf7;\xa1=\xb1?\xf9A\xfbC\xff\x1f+\x01\x1f-!\x00\x00\x00\x00\x00\x00\x00\x00\x01\x00\x00\x00\x00\x00\x00\x00\x1f/\x11\x00\x00\x00\x00\x00\x00\x00\x00\x03\x01\x1f\x1d\x11\x00\x00\x00\x00\x00\x00\x00\x00\x1dk\x03\x03\xc3\x1dm\t\x07\x0b\x05\x1do\x05\x01#\x1f\x03\x05\xb7\xbb\r\x03\xa5\xb9\x1dq\r\x03\xa5\xbd\x1ds\x1du\x1dw\r\x01##\x1dy\x13\x0b\x01\x1f\x01\t\xff\xff\xff\xff\x1f%\x01\x13\x0b\x05\x07\x05\x1f\x05!\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x1f\x01\t\x01\x00\x00\x00\x1f\x01\t\x03\x00\x00\x00\x1f\x01\t`\x00\x00\x00\x1d{\x03\x0b\x9b\x9b\x9b\x9b\x9d\x03\x03\xe1\x15\x03\x01\x11\x01\x03\t\x9d\x9f\x9b\x9f\x13\x07\x01\x13\x07\x05\x13\x07\t\x13\x07\r\x1f\x01\t\x00\x00\x00\x00\x07\x01\x1f\x05!\x00\x00\x00\x00\x00\x00\xf8\x7f\x00\x00\x00\x00\x00\x00\xf8\x7f\x1f\x1d!\x00\x00\x00\x00\x00\x00\x00\x00\x01\x00\x00\x00\x00\x00\x00\x00\x1f5\x11\x00\x00\x00\x00\x00\x00\x00\x00\x1d}\x03\x0f\x9b\x9b\x9b\x9b\x9b\x9d\x9f\x03\x03\xfd\x15\x03\x01\x15\x01\x03\x07\x9d\x9b\x9f\x03\x03\x06\x02\xa9\x05\x7f)\x01\x07)\x05\r\r\t)\x01\t\x1b\x03!\x1d\x01)\x05\r\r\x07)\x03\r\t)\x03\x02\x03\t)\x05\r\r\r\x13)\x01\r)\x05\x05\x05\r)\x03\t\x0b\x11\x01\x05\x03\x03\x0b\x11\x03\x03\x03\x03)\x03\x01\x0b)\x03%\t/\t\x03\x11\x01\x13)\x03\x01\x17)\x03\t\x17)\x03\x05\x17)\x03\x05\r)\x03\r\r)\x03\x05\x0b/\x07\x03\x01\x13\x04\xe6\x06\x05\x01\x11\x0fQ\x07\x03\x01\t\x0f\x11\x0fU\x05\x03Y\xb5\x0b\x03w#\x03'\x17\x06{\x03\x03\x03\x01\x03\x03\x011\x03\x01\x03\x03\x01\r\x03\x01\x03\x03\x01\r\x03\x01\x03\x03\x013\x03\x01\x13\x07\x01\x81\x03)\x0b\x05\x07\t\x0b\x03\x07\x07\x01E\x03\x03\x03\r\x07\x07\x01G\x03\x11\x03\r\x07\x07\x01I\x03\x01\x03\r\x07\x07\x01\x83\x03\x13\x03\r\x03\x03\x01K\x03\x01\x05\x07\x01\x07\x03\x01\x03\x17\r\x07\x01M\x03\x19\x05\x13\x19\x05\x07\x01\x07\x03\x1b\x03\x1b\x03\x03\x01\x17\x03\x05\x05\x07\x01\x07\x03\x03\x03\x1f\x05\x07\x01O\x03\x15\x03\x1d\t\x06\x01\x03\x03\x07#\x0f!\x05\x07\x01\x07\x031\x03\x1b\x03\x03\x01\x17\x03\x05\x05\x07\x01\x07\x03\x11\x03)\x05\x07\x01\x85\x033\x03'\t\x06\x01\x03\x11\x07-\x11+\x03\x03\x87-\x03\x05\x19\x07\x93\x8b\x03\x03\x05%1\x03\x03\x031\x03\x01\x03\x03\x03\r\x03\x01\x03\x03\x03\r\x03\x01\x03\x03\x03\r\x03\x01\x03\x03\x033\x03\x01\x13\x07\x03\x99\x037\x0f579;=3/\x07\x07\x03E\x03\x03\x03?\x07\x07\x03G\x03\x01\x03?\x07\x07\x03I\x03\x13\x03?\x03\x03\x03K\x03\x01\x05\x07\x03\x07\x03\x01\x03G\r\x07\x03M\x03\x19\x05CI\x05\x07\x03\x07\x03\x1b\x03K\x03\x03\x03\x17\x03\x05\x05\x07\x03\x07\x03\x03\x03O\x05\x07\x03O\x03\x15\x03M\t\x06\x03\x03\x03\x07SAQ\x1b\x07\x0b\x02\x02\x03\x03\x03%\x11\x04\x0f\x05UW\x0f\x11\x0bW\x05\x03\x15+\x03\x03\x0f\x0b\x03[#\x03\x0f\x03\x03\x0b_\x03\x01\x05\x07'\x07\x03\x0f\x03\x05\x15\x06'\x03\x0f\x05\x03\x07\x0b\x03ec\x03\x0f\r\x07ki\x03\x15\x05\t\x0b\x03\x03\x0b-\x03\x05\x05\x07o\x07\x03\x03\x03\x0f\t\x06s\x03\x03\x07\r\x11\x01\x11\x04\x0b\x03\x13\x06\x03\x01\x05\x01\x00\xd2\x18\x81\x0f\x1d\x1d\x11\x0f\x0b\t\t\x03\x0b!Y\x87##%_=\x85\x8dW\xb3K\x9bM\x9b\xd2\x02\x1b\x1f/!!)#\x1f\x19+\x1b\x1f\x83\x1f\x15\x1d\x15+\x13\r\r\x11\x0f\x17\x0f\x1f\x15\x11\x17\x11\x15+)\x19\x0f\x0b\x11builtin\x00vhlo\x00module\x00constant_v1\x00broadcast_in_dim_v1\x00get_tuple_element_v1\x00select_v1\x00iota_v1\x00compare_v1\x00func_v1\x00return_v1\x00custom_call_v1\x00add_v1\x00reshape_v1\x00pad_v1\x00call_v1\x00value\x00index\x00sym_name\x00broadcast_dimensions\x00arg_attrs\x00function_type\x00res_attrs\x00sym_visibility\x00third_party/py/jax/experimental/jax2tf/tests/back_compat_test.py\x00iota_dimension\x00compare_type\x00comparison_direction\x00api_version\x00backend_config\x00call_target_name\x00called_computations\x00has_side_effect\x00operand_layouts\x00output_operand_aliases\x00result_layouts\x00jit__lambda_\x00jit(<lambda>)/jit(main)/pjit[in_shardings=(UnspecifiedValue,) out_shardings=(UnspecifiedValue,) resource_env=None donated_invars=(False,) name=triu keep_unused=False inline=False]\x00jit(<lambda>)/jit(main)/jit(triu)/iota[dtype=int32 shape=(3, 3) dimension=0]\x00jit(<lambda>)/jit(main)/jit(triu)/add\x00jit(<lambda>)/jit(main)/jit(triu)/iota[dtype=int32 shape=(3, 3) dimension=1]\x00jit(<lambda>)/jit(main)/jit(triu)/ge\x00jit(<lambda>)/jit(main)/jit(triu)/broadcast_in_dim[shape=(3, 3) broadcast_dimensions=()]\x00jit(<lambda>)/jit(main)/jit(triu)/select_n\x00jit(<lambda>)/jit(main)/iota[dtype=complex128 shape=(9,) dimension=0]\x00jit(<lambda>)/jit(main)/reshape[new_sizes=(3, 3) dimensions=None]\x00jit(<lambda>)/jit(main)/geqrf\x00jit(<lambda>)/jit(main)/qr[full_matrices=True]\x00edge_padding_high\x00edge_padding_low\x00interior_padding\x00jit(<lambda>)/jit(main)/pad[padding_config=((0, 0, 0), (0, 0, 0))]\x00jit(<lambda>)/jit(main)/householder_product\x00jax.result_info\x00triu\x00\x00[0]\x00[1]\x00main\x00public\x00private\x00lapack_zgeqrf\x00lapack_zungqr\x00callee\x00",
    xla_call_module_version=4,
)  # End paste

data_2024_05_31 = {}


# Pasted from the test output (see export_back_compat_test_util.py module docstring)
data_2024_05_31["c128"] = dict(
    testdata_version=1,
    platform='cpu',
    custom_call_targets=['lapack_zgeqrf_ffi', 'lapack_zungqr_ffi'],
    serialized_date=datetime.date(2024, 5, 31),
    inputs=(),
    expected_outputs=(array([[ 0.                 +0.j,  0.9128709291752773 +0.j,
         0.40824829046386235+0.j],
       [-0.447213595499958  -0.j,  0.3651483716701102 +0.j,
        -0.8164965809277263 +0.j],
       [-0.894427190999916  -0.j, -0.1825741858350548 +0.j,
         0.40824829046386324+0.j]]), array([[-6.7082039324993694e+00+0.j, -8.0498447189992444e+00+0.j,
        -9.3914855054991175e+00+0.j],
       [ 0.0000000000000000e+00+0.j,  1.0954451150103341e+00+0.j,
         2.1908902300206665e+00+0.j],
       [ 0.0000000000000000e+00+0.j,  0.0000000000000000e+00+0.j,
        -8.8817841970012523e-16+0.j]])),
    mlir_module_text=r"""
#loc3 = loc("third_party/py/jax/tests/export_back_compat_test.py":342:11)
#loc10 = loc("jit(<lambda>)/jit(main)/pjit[in_shardings=(UnspecifiedValue,) out_shardings=(UnspecifiedValue,) in_layouts=(None,) out_layouts=(None,) resource_env=None donated_invars=(False,) name=triu keep_unused=False inline=False]"(#loc3))
module @jit__lambda_ attributes {jax.uses_shape_polymorphism = false, mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} {
  func.func public @main() -> (tensor<3x3xcomplex<f64>> {jax.result_info = "[0]", mhlo.layout_mode = "default"}, tensor<3x3xcomplex<f64>> {jax.result_info = "[1]", mhlo.layout_mode = "default"}) {
    %0 = stablehlo.iota dim = 0 : tensor<9xcomplex<f64>> loc(#loc4)
    %1 = stablehlo.reshape %0 : (tensor<9xcomplex<f64>>) -> tensor<3x3xcomplex<f64>> loc(#loc5)
    %c = stablehlo.constant dense<3> : tensor<i64> loc(#loc6)
    %c_0 = stablehlo.constant dense<3> : tensor<i64> loc(#loc6)
    %2:4 = stablehlo.custom_call @lapack_zgeqrf_ffi(%1) {mhlo.backend_config = {}, operand_layouts = [dense<[0, 1]> : tensor<2xindex>], output_operand_aliases = [#stablehlo.output_operand_alias<output_tuple_indices = [0], operand_index = 0, operand_tuple_indices = []>], result_layouts = [dense<[0, 1]> : tensor<2xindex>, dense<0> : tensor<1xindex>, dense<> : tensor<0xindex>, dense<0> : tensor<1xindex>]} : (tensor<3x3xcomplex<f64>>) -> (tensor<3x3xcomplex<f64>>, tensor<3xcomplex<f64>>, tensor<i32>, tensor<96xcomplex<f64>>) loc(#loc6)
    %c_1 = stablehlo.constant dense<0> : tensor<i32> loc(#loc6)
    %3 = stablehlo.broadcast_in_dim %c_1, dims = [] : (tensor<i32>) -> tensor<i32> loc(#loc6)
    %4 = stablehlo.compare  EQ, %2#2, %3,  SIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1> loc(#loc6)
    %5 = stablehlo.broadcast_in_dim %4, dims = [] : (tensor<i1>) -> tensor<1x1xi1> loc(#loc6)
    %cst = stablehlo.constant dense<(0x7FF8000000000000,0x7FF8000000000000)> : tensor<complex<f64>> loc(#loc6)
    %6 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<complex<f64>>) -> tensor<3x3xcomplex<f64>> loc(#loc6)
    %7 = stablehlo.broadcast_in_dim %5, dims = [0, 1] : (tensor<1x1xi1>) -> tensor<3x3xi1> loc(#loc6)
    %8 = stablehlo.select %7, %2#0, %6 : tensor<3x3xi1>, tensor<3x3xcomplex<f64>> loc(#loc6)
    %9 = stablehlo.broadcast_in_dim %4, dims = [] : (tensor<i1>) -> tensor<1xi1> loc(#loc6)
    %cst_2 = stablehlo.constant dense<(0x7FF8000000000000,0x7FF8000000000000)> : tensor<complex<f64>> loc(#loc6)
    %10 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<complex<f64>>) -> tensor<3xcomplex<f64>> loc(#loc6)
    %11 = stablehlo.broadcast_in_dim %9, dims = [0] : (tensor<1xi1>) -> tensor<3xi1> loc(#loc6)
    %12 = stablehlo.select %11, %2#1, %10 : tensor<3xi1>, tensor<3xcomplex<f64>> loc(#loc6)
    %cst_3 = stablehlo.constant dense<(0.000000e+00,0.000000e+00)> : tensor<complex<f64>> loc(#loc7)
    %13 = stablehlo.pad %8, %cst_3, low = [0, 0], high = [0, 0], interior = [0, 0] : (tensor<3x3xcomplex<f64>>, tensor<complex<f64>>) -> tensor<3x3xcomplex<f64>> loc(#loc8)
    %c_4 = stablehlo.constant dense<3> : tensor<i64> loc(#loc9)
    %c_5 = stablehlo.constant dense<3> : tensor<i64> loc(#loc9)
    %c_6 = stablehlo.constant dense<3> : tensor<i64> loc(#loc9)
    %14:3 = stablehlo.custom_call @lapack_zungqr_ffi(%13, %12) {mhlo.backend_config = {}, operand_layouts = [dense<[0, 1]> : tensor<2xindex>, dense<0> : tensor<1xindex>], output_operand_aliases = [#stablehlo.output_operand_alias<output_tuple_indices = [0], operand_index = 0, operand_tuple_indices = []>], result_layouts = [dense<[0, 1]> : tensor<2xindex>, dense<> : tensor<0xindex>, dense<0> : tensor<1xindex>]} : (tensor<3x3xcomplex<f64>>, tensor<3xcomplex<f64>>) -> (tensor<3x3xcomplex<f64>>, tensor<i32>, tensor<96xcomplex<f64>>) loc(#loc9)
    %c_7 = stablehlo.constant dense<0> : tensor<i32> loc(#loc9)
    %15 = stablehlo.broadcast_in_dim %c_7, dims = [] : (tensor<i32>) -> tensor<i32> loc(#loc9)
    %16 = stablehlo.compare  EQ, %14#1, %15,  SIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1> loc(#loc9)
    %17 = stablehlo.broadcast_in_dim %16, dims = [] : (tensor<i1>) -> tensor<1x1xi1> loc(#loc9)
    %cst_8 = stablehlo.constant dense<(0x7FF8000000000000,0x7FF8000000000000)> : tensor<complex<f64>> loc(#loc9)
    %18 = stablehlo.broadcast_in_dim %cst_8, dims = [] : (tensor<complex<f64>>) -> tensor<3x3xcomplex<f64>> loc(#loc9)
    %19 = stablehlo.broadcast_in_dim %17, dims = [0, 1] : (tensor<1x1xi1>) -> tensor<3x3xi1> loc(#loc9)
    %20 = stablehlo.select %19, %14#0, %18 : tensor<3x3xi1>, tensor<3x3xcomplex<f64>> loc(#loc9)
    %21 = call @triu(%8) : (tensor<3x3xcomplex<f64>>) -> tensor<3x3xcomplex<f64>> loc(#loc10)
    return %20, %21 : tensor<3x3xcomplex<f64>>, tensor<3x3xcomplex<f64>> loc(#loc)
  } loc(#loc)
  func.func private @triu(%arg0: tensor<3x3xcomplex<f64>> {mhlo.layout_mode = "default"} loc("jit(<lambda>)/jit(main)/pjit[in_shardings=(UnspecifiedValue,) out_shardings=(UnspecifiedValue,) in_layouts=(None,) out_layouts=(None,) resource_env=None donated_invars=(False,) name=triu keep_unused=False inline=False]"(#loc3))) -> (tensor<3x3xcomplex<f64>> {mhlo.layout_mode = "default"}) {
    %0 = stablehlo.iota dim = 0 : tensor<3x3xi32> loc(#loc11)
    %c = stablehlo.constant dense<-1> : tensor<i32> loc(#loc10)
    %1 = stablehlo.broadcast_in_dim %c, dims = [] : (tensor<i32>) -> tensor<3x3xi32> loc(#loc12)
    %2 = stablehlo.add %0, %1 : tensor<3x3xi32> loc(#loc12)
    %3 = stablehlo.iota dim = 1 : tensor<3x3xi32> loc(#loc13)
    %4 = stablehlo.compare  GE, %2, %3,  SIGNED : (tensor<3x3xi32>, tensor<3x3xi32>) -> tensor<3x3xi1> loc(#loc14)
    %cst = stablehlo.constant dense<(0.000000e+00,0.000000e+00)> : tensor<complex<f64>> loc(#loc10)
    %5 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<complex<f64>>) -> tensor<3x3xcomplex<f64>> loc(#loc15)
    %6 = stablehlo.select %4, %5, %arg0 : tensor<3x3xi1>, tensor<3x3xcomplex<f64>> loc(#loc16)
    return %6 : tensor<3x3xcomplex<f64>> loc(#loc10)
  } loc(#loc10)
} loc(#loc)
#loc = loc(unknown)
#loc1 = loc("third_party/py/jax/tests/export_back_compat_test.py":341:26)
#loc2 = loc("third_party/py/jax/tests/export_back_compat_test.py":341:14)
#loc4 = loc("jit(<lambda>)/jit(main)/iota[dtype=complex128 shape=(9,) dimension=0]"(#loc1))
#loc5 = loc("jit(<lambda>)/jit(main)/reshape[new_sizes=(3, 3) dimensions=None]"(#loc2))
#loc6 = loc("jit(<lambda>)/jit(main)/geqrf"(#loc3))
#loc7 = loc("jit(<lambda>)/jit(main)/qr[full_matrices=True]"(#loc3))
#loc8 = loc("jit(<lambda>)/jit(main)/pad[padding_config=((0, 0, 0), (0, 0, 0))]"(#loc3))
#loc9 = loc("jit(<lambda>)/jit(main)/householder_product"(#loc3))
#loc11 = loc("jit(<lambda>)/jit(main)/jit(triu)/iota[dtype=int32 shape=(3, 3) dimension=0]"(#loc3))
#loc12 = loc("jit(<lambda>)/jit(main)/jit(triu)/add"(#loc3))
#loc13 = loc("jit(<lambda>)/jit(main)/jit(triu)/iota[dtype=int32 shape=(3, 3) dimension=1]"(#loc3))
#loc14 = loc("jit(<lambda>)/jit(main)/jit(triu)/ge"(#loc3))
#loc15 = loc("jit(<lambda>)/jit(main)/jit(triu)/broadcast_in_dim[shape=(3, 3) broadcast_dimensions=()]"(#loc3))
#loc16 = loc("jit(<lambda>)/jit(main)/jit(triu)/select_n"(#loc3))
""",
    mlir_module_serialized=b"ML\xefR\x01StableHLO_v0.9.0\x00\x01)\x05\x01\x03\x01\x03\x05\x03\x19\x07\t\x0b\r\x0f\x11\x13\x15\x17\x19\x1b\x1d\x03\x92\x02\xfb;\x01\x9f\x0f\x0f\x17\x13\x0f\x0b\x13\x0b\x07\x0b\x0b\x13\x0f\x0b\x0b\x0b\x0b\x13\x0b\x0f\x0b\x0b\x13\x0b\x0b\x0b\x0b\x0b\x0b\x0b\x0b\x0b\x13\x1b\x13+\x0b\x0f\x0b\x0b\x0b33\x0b\x0f\x0b\x13\x0b\x13\x0f\x0b\x1b\x0f\x0b\x0f\x0b\x0f\x0b\x0f\x0b\x17\x0f\x0b\x17\x0bS\x13\x0f\x0b#\x0b\x0b\x0b\x0f\x0b\x0bS\x13\x0b\x03]O/\x0b\x0b\x0b/\x0b\x0f\x0b\x0b\x0b\x0b\x0b\x0b\x0f\x0f\x0b\x13\x1b\x0b\x1b\x0b\x0b\x0b\x13\x0b\x0b\x0f\x1f\x0f\x0f\x0bO/\x0b\x0f\x17\x1b\x1f\x0bOO/\x0b\x13\x17\x01\x05\x0b\x0f\x037\x17\x0f\x0f\x07\x0f\x0b\x07\x17\x17\x13\x07\x07\x17\x0f\x17\x13\x17\x07\x17\x13\x13\x13\x13\x13\x13\x13\x13\x02\xa6\t\x1d\x81\x05\x1d\x97\x05\x17\x13Z\x05\x17\x03\x03\x15\xd9\x1dW\x05\x05\x1f\x03\x03\x0b\xe1\x05!\x1f\x05#\x05%\x03\x03\x0b\xef\x11\x03\x05\x05'\x05)\x05+\x05-\x03\x03%\xd5\x05/\x1d_\x05\x051\x053\x03\x03\x0b\xdf\x055\x057\x059\x05;\x05=\x05?\x05A\x05C\x05E\x03\x03\x0b\xeb\x03\x05)\xb1+\xed\x03\x03\x15\xf1\x03\tIKM\x19O\x19\x0fQ\x05G\x11\x01\x00\x05I\x05K\x05M\x03\x0b\x1b\xa3\x1d\xbf\x1f\xc1\x0f\xcb!\xcd\x03\x0b\x1b\xad\x1d\xd1\x1f\xad\x0f\xaf!\xd3\x05O\x1d[\x05\x05Q\x03\x03\x0b\xd7\x05S\x03\x03%\xdb\x1de\x05\x05U\x03\x05)\xb1+\xdd\x1dk\x05\x05W\x1do\x05\x05Y\x1ds\x05\x05[\x1dwy\x05]\x17\x13V\x055\x1d}\x7f\x05_\x17\x13V\x05\x1d\x05a\x03\x13/\xb31\xb53\xe35\xa37\xb79\xb9;\xe5=\xbb?\xe9\x03\x03\x15\xf3\x1d\x89\x05\x05c\x03\x07\x8d\xa9\x8f\xa9\x91\xa9\x05e\x05g\x05i\x1d\x95\x05\x05k\x05m\x03\x13/\xb31\xb53\xf55\xa37\xb79\xb9;\xf7=\xbb?\xf9\x03\x03\x9d\xaf\x05o\x1f/!\x00\x00\x00\x00\x00\x00\x00\x00\x01\x00\x00\x00\x00\x00\x00\x00\x1f1\x11\x00\x00\x00\x00\x00\x00\x00\x00\x03\x01\x1dq\x1ds\x1f#\x11\x00\x00\x00\x00\x00\x00\x00\x00\x1du\x03\x03\xcf\x1dw\t\x07\x0b\x03\x1dy\x05\x01\r\x01\x03\x03\xe7\x1f3\x01#%\x03\x05\xc3\xc7\r\x05\xab\xc5\xa5\xa7\x1d{\r\x05\xab\xc9\xa5\xa7\x1d}\x1d\x7f\x1d\x81\r\x03\xa5\xa7#)\x1d\x83\x13\x0b\x01\x1f\x07\t\xff\xff\xff\xff\x1f+\x01\x13\x0b\x05\x07\x05\x1f\t!\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x1f\r\x11\x03\x00\x00\x00\x00\x00\x00\x00\x1d\x85\x03\x03\x9f\x15\x03\x01\x01\x01\x03\t\x9f\xa1\xbd\xa1\x1f\x07\t\x00\x00\x00\x00\x07\x01\x1f\t!\x00\x00\x00\x00\x00\x00\xf8\x7f\x00\x00\x00\x00\x00\x00\xf8\x7f\x1f#!\x00\x00\x00\x00\x00\x00\x00\x00\x01\x00\x00\x00\x00\x00\x00\x00\x1f9\x11\x00\x00\x00\x00\x00\x00\x00\x00\x1d\x87\x03\x05\x9f\xa1\x03\x07\x9f\xbd\xa1\x01\t\x01\x02\x02)\x05\r\r\x0f)\x01\x1b)\x01\x0f\x1d)\x01\x0b\x03'\x01)\x05\r\r\x1b)\x05\r\r\x11)\x03\r\x0f\x13\x1b)\x03\x02\x03\x0f)\x01\x11)\x05\x05\x05\x11)\x03\t\x0b\x11\x01\x05\x05\x05\x0b\x11\x03\x05\x03\x05)\x03\x01\x0b)\x03%\x0f)\x03\t\x19)\x03\x05\x19)\x03\x01\x19)\x03\x05\x11)\x03\r\x11)\x03\x05\x0b\x04\x92\x05\x05\x01\x11\x11G\x07\x03\x01\t\r\x11\x11S\x07\x03M\x89\t\x03u#\x03-\x15\x06{\x03\x05\x03\x01\x03\x03\x01\r\x03\r\x03\x03\x01\r\x03\r\x11\x07\x01\x83\t\x05\x17\x07\x1d\x03\x03\x03\x03\x01A\x03\x07\x05\x07\x01\x07\x03\x07\x03\x11\x0b\x07\x01C\x03\x1f\x05\r\x13\x05\x07\x01\x07\x03!\x03\x15\x03\x03\x01\x17\x03\t\x05\x07\x01\x07\x03\x05\x03\x19\x05\x07\x01E\x03\x15\x03\x17\x07\x06\x01\x03\x05\x07\x1d\t\x1b\x05\x07\x01\x07\x035\x03\x15\x03\x03\x01\x17\x03\t\x05\x07\x01\x07\x03\x17\x03#\x05\x07\x01\x85\x037\x03!\x07\x06\x01\x03\x17\x07'\x0b%\x03\x03\x87-\x03\t\x17\x07\x93\x8b\x03\x05\x05\x1f+\x03\x03\x03\r\x03\r\x03\x03\x03\r\x03\r\x03\x03\x03\r\x03\r\x11\x07\x03\x99\x07\x05\x07\x1d\x05-)\x03\x03\x03A\x03\x07\x05\x07\x03\x07\x03\x07\x03;\x0b\x07\x03C\x03\x1f\x057=\x05\x07\x03\x07\x03!\x03?\x03\x03\x03\x17\x03\t\x05\x07\x03\x07\x03\x05\x03C\x05\x07\x03E\x03\x15\x03A\x07\x06\x03\x03\x05\x07G5E\x19\x07\t\x9b\x03\x05\x03\x1f\x0f\x04\x11\x05IK\r\x11\tU\x07\x03\x15+\x03\x05\t\t\x03Y#\x03\x13\x03\x03\t]\x03\x07\x05\x07'\x07\x03\x13\x03\x05\x13\x06'\x03\x13\x05\x03\x07\t\x03ca\x03\x13\x0b\x07ig\x03\x15\x05\t\x0b\x03\x03\t-\x03\t\x05\x07m\x07\x03\x05\x03\x0f\x07\x06q\x03\x05\x07\r\x11\x01\x0f\x04\t\x03\x13\x06\x03\x01\x05\x01\x00\xba\x1a\x89%%\x11\x0f\x0b\t\t\x03\x0b!\x11#\x0fY\x87##%_=\x85\x8dW\xb3K\x9bM\x9bn\x03\x1b%)9\x1f/!)!)#\x1f\x19+\x1b\x1f\x1f\x15\x1d\x15+i\x13\r\x11\x0f\x17\x0f\x1f\x15\x11\x17\x11\x15)\x19\x0f\x0b\x11builtin\x00vhlo\x00module\x00constant_v1\x00broadcast_in_dim_v1\x00select_v1\x00iota_v1\x00compare_v1\x00func_v1\x00return_v1\x00custom_call_v1\x00add_v1\x00reshape_v1\x00pad_v1\x00call_v1\x00value\x00sym_name\x00third_party/py/jax/tests/export_back_compat_test.py\x00broadcast_dimensions\x00arg_attrs\x00function_type\x00res_attrs\x00sym_visibility\x00iota_dimension\x00compare_type\x00comparison_direction\x00api_version\x00backend_config\x00call_target_name\x00called_computations\x00has_side_effect\x00mhlo.backend_config\x00operand_layouts\x00output_operand_aliases\x00result_layouts\x00jax.uses_shape_polymorphism\x00mhlo.num_partitions\x00mhlo.num_replicas\x00jit__lambda_\x00jit(<lambda>)/jit(main)/pjit[in_shardings=(UnspecifiedValue,) out_shardings=(UnspecifiedValue,) in_layouts=(None,) out_layouts=(None,) resource_env=None donated_invars=(False,) name=triu keep_unused=False inline=False]\x00jit(<lambda>)/jit(main)/jit(triu)/iota[dtype=int32 shape=(3, 3) dimension=0]\x00jit(<lambda>)/jit(main)/jit(triu)/add\x00jit(<lambda>)/jit(main)/jit(triu)/iota[dtype=int32 shape=(3, 3) dimension=1]\x00jit(<lambda>)/jit(main)/jit(triu)/ge\x00jit(<lambda>)/jit(main)/jit(triu)/broadcast_in_dim[shape=(3, 3) broadcast_dimensions=()]\x00jit(<lambda>)/jit(main)/jit(triu)/select_n\x00jit(<lambda>)/jit(main)/iota[dtype=complex128 shape=(9,) dimension=0]\x00jit(<lambda>)/jit(main)/reshape[new_sizes=(3, 3) dimensions=None]\x00jit(<lambda>)/jit(main)/geqrf\x00jit(<lambda>)/jit(main)/qr[full_matrices=True]\x00edge_padding_high\x00edge_padding_low\x00interior_padding\x00jit(<lambda>)/jit(main)/pad[padding_config=((0, 0, 0), (0, 0, 0))]\x00jit(<lambda>)/jit(main)/householder_product\x00callee\x00mhlo.layout_mode\x00default\x00jax.result_info\x00triu\x00\x00[0]\x00[1]\x00main\x00public\x00private\x00lapack_zgeqrf_ffi\x00lapack_zungqr_ffi\x00",
    xla_call_module_version=9,
    nr_devices=1,
)  # End paste


# Pasted from the test output (see export_back_compat_test_util.py module docstring)
data_2024_05_31["c64"] = dict(
    testdata_version=1,
    platform='cpu',
    custom_call_targets=['lapack_cgeqrf_ffi', 'lapack_cungqr_ffi'],
    serialized_date=datetime.date(2024, 5, 31),
    inputs=(),
    expected_outputs=(array([[ 0.        +0.j,  0.91287076+0.j,  0.4082487 +0.j],
       [-0.44721356-0.j,  0.36514866+0.j, -0.8164965 +0.j],
       [-0.8944271 -0.j, -0.18257445+0.j,  0.40824816+0.j]],
      dtype=complex64), array([[-6.7082043e+00+0.j, -8.0498438e+00+0.j, -9.3914852e+00+0.j],
       [ 0.0000000e+00+0.j,  1.0954441e+00+0.j,  2.1908894e+00+0.j],
       [ 0.0000000e+00+0.j,  0.0000000e+00+0.j,  7.1525574e-07+0.j]],
      dtype=complex64)),
    mlir_module_text=r"""
#loc3 = loc("third_party/py/jax/tests/export_back_compat_test.py":342:11)
#loc10 = loc("jit(<lambda>)/jit(main)/pjit[in_shardings=(UnspecifiedValue,) out_shardings=(UnspecifiedValue,) in_layouts=(None,) out_layouts=(None,) resource_env=None donated_invars=(False,) name=triu keep_unused=False inline=False]"(#loc3))
module @jit__lambda_ attributes {jax.uses_shape_polymorphism = false, mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} {
  func.func public @main() -> (tensor<3x3xcomplex<f32>> {jax.result_info = "[0]", mhlo.layout_mode = "default"}, tensor<3x3xcomplex<f32>> {jax.result_info = "[1]", mhlo.layout_mode = "default"}) {
    %0 = stablehlo.iota dim = 0 : tensor<9xcomplex<f32>> loc(#loc4)
    %1 = stablehlo.reshape %0 : (tensor<9xcomplex<f32>>) -> tensor<3x3xcomplex<f32>> loc(#loc5)
    %c = stablehlo.constant dense<3> : tensor<i64> loc(#loc6)
    %c_0 = stablehlo.constant dense<3> : tensor<i64> loc(#loc6)
    %2:4 = stablehlo.custom_call @lapack_cgeqrf_ffi(%1) {mhlo.backend_config = {}, operand_layouts = [dense<[0, 1]> : tensor<2xindex>], output_operand_aliases = [#stablehlo.output_operand_alias<output_tuple_indices = [0], operand_index = 0, operand_tuple_indices = []>], result_layouts = [dense<[0, 1]> : tensor<2xindex>, dense<0> : tensor<1xindex>, dense<> : tensor<0xindex>, dense<0> : tensor<1xindex>]} : (tensor<3x3xcomplex<f32>>) -> (tensor<3x3xcomplex<f32>>, tensor<3xcomplex<f32>>, tensor<i32>, tensor<96xcomplex<f32>>) loc(#loc6)
    %c_1 = stablehlo.constant dense<0> : tensor<i32> loc(#loc6)
    %3 = stablehlo.broadcast_in_dim %c_1, dims = [] : (tensor<i32>) -> tensor<i32> loc(#loc6)
    %4 = stablehlo.compare  EQ, %2#2, %3,  SIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1> loc(#loc6)
    %5 = stablehlo.broadcast_in_dim %4, dims = [] : (tensor<i1>) -> tensor<1x1xi1> loc(#loc6)
    %cst = stablehlo.constant dense<(0x7FC00000,0x7FC00000)> : tensor<complex<f32>> loc(#loc6)
    %6 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<complex<f32>>) -> tensor<3x3xcomplex<f32>> loc(#loc6)
    %7 = stablehlo.broadcast_in_dim %5, dims = [0, 1] : (tensor<1x1xi1>) -> tensor<3x3xi1> loc(#loc6)
    %8 = stablehlo.select %7, %2#0, %6 : tensor<3x3xi1>, tensor<3x3xcomplex<f32>> loc(#loc6)
    %9 = stablehlo.broadcast_in_dim %4, dims = [] : (tensor<i1>) -> tensor<1xi1> loc(#loc6)
    %cst_2 = stablehlo.constant dense<(0x7FC00000,0x7FC00000)> : tensor<complex<f32>> loc(#loc6)
    %10 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<complex<f32>>) -> tensor<3xcomplex<f32>> loc(#loc6)
    %11 = stablehlo.broadcast_in_dim %9, dims = [0] : (tensor<1xi1>) -> tensor<3xi1> loc(#loc6)
    %12 = stablehlo.select %11, %2#1, %10 : tensor<3xi1>, tensor<3xcomplex<f32>> loc(#loc6)
    %cst_3 = stablehlo.constant dense<(0.000000e+00,0.000000e+00)> : tensor<complex<f32>> loc(#loc7)
    %13 = stablehlo.pad %8, %cst_3, low = [0, 0], high = [0, 0], interior = [0, 0] : (tensor<3x3xcomplex<f32>>, tensor<complex<f32>>) -> tensor<3x3xcomplex<f32>> loc(#loc8)
    %c_4 = stablehlo.constant dense<3> : tensor<i64> loc(#loc9)
    %c_5 = stablehlo.constant dense<3> : tensor<i64> loc(#loc9)
    %c_6 = stablehlo.constant dense<3> : tensor<i64> loc(#loc9)
    %14:3 = stablehlo.custom_call @lapack_cungqr_ffi(%13, %12) {mhlo.backend_config = {}, operand_layouts = [dense<[0, 1]> : tensor<2xindex>, dense<0> : tensor<1xindex>], output_operand_aliases = [#stablehlo.output_operand_alias<output_tuple_indices = [0], operand_index = 0, operand_tuple_indices = []>], result_layouts = [dense<[0, 1]> : tensor<2xindex>, dense<> : tensor<0xindex>, dense<0> : tensor<1xindex>]} : (tensor<3x3xcomplex<f32>>, tensor<3xcomplex<f32>>) -> (tensor<3x3xcomplex<f32>>, tensor<i32>, tensor<96xcomplex<f32>>) loc(#loc9)
    %c_7 = stablehlo.constant dense<0> : tensor<i32> loc(#loc9)
    %15 = stablehlo.broadcast_in_dim %c_7, dims = [] : (tensor<i32>) -> tensor<i32> loc(#loc9)
    %16 = stablehlo.compare  EQ, %14#1, %15,  SIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1> loc(#loc9)
    %17 = stablehlo.broadcast_in_dim %16, dims = [] : (tensor<i1>) -> tensor<1x1xi1> loc(#loc9)
    %cst_8 = stablehlo.constant dense<(0x7FC00000,0x7FC00000)> : tensor<complex<f32>> loc(#loc9)
    %18 = stablehlo.broadcast_in_dim %cst_8, dims = [] : (tensor<complex<f32>>) -> tensor<3x3xcomplex<f32>> loc(#loc9)
    %19 = stablehlo.broadcast_in_dim %17, dims = [0, 1] : (tensor<1x1xi1>) -> tensor<3x3xi1> loc(#loc9)
    %20 = stablehlo.select %19, %14#0, %18 : tensor<3x3xi1>, tensor<3x3xcomplex<f32>> loc(#loc9)
    %21 = call @triu(%8) : (tensor<3x3xcomplex<f32>>) -> tensor<3x3xcomplex<f32>> loc(#loc10)
    return %20, %21 : tensor<3x3xcomplex<f32>>, tensor<3x3xcomplex<f32>> loc(#loc)
  } loc(#loc)
  func.func private @triu(%arg0: tensor<3x3xcomplex<f32>> {mhlo.layout_mode = "default"} loc("jit(<lambda>)/jit(main)/pjit[in_shardings=(UnspecifiedValue,) out_shardings=(UnspecifiedValue,) in_layouts=(None,) out_layouts=(None,) resource_env=None donated_invars=(False,) name=triu keep_unused=False inline=False]"(#loc3))) -> (tensor<3x3xcomplex<f32>> {mhlo.layout_mode = "default"}) {
    %0 = stablehlo.iota dim = 0 : tensor<3x3xi32> loc(#loc11)
    %c = stablehlo.constant dense<-1> : tensor<i32> loc(#loc10)
    %1 = stablehlo.broadcast_in_dim %c, dims = [] : (tensor<i32>) -> tensor<3x3xi32> loc(#loc12)
    %2 = stablehlo.add %0, %1 : tensor<3x3xi32> loc(#loc12)
    %3 = stablehlo.iota dim = 1 : tensor<3x3xi32> loc(#loc13)
    %4 = stablehlo.compare  GE, %2, %3,  SIGNED : (tensor<3x3xi32>, tensor<3x3xi32>) -> tensor<3x3xi1> loc(#loc14)
    %cst = stablehlo.constant dense<(0.000000e+00,0.000000e+00)> : tensor<complex<f32>> loc(#loc10)
    %5 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<complex<f32>>) -> tensor<3x3xcomplex<f32>> loc(#loc15)
    %6 = stablehlo.select %4, %5, %arg0 : tensor<3x3xi1>, tensor<3x3xcomplex<f32>> loc(#loc16)
    return %6 : tensor<3x3xcomplex<f32>> loc(#loc10)
  } loc(#loc10)
} loc(#loc)
#loc = loc(unknown)
#loc1 = loc("third_party/py/jax/tests/export_back_compat_test.py":341:26)
#loc2 = loc("third_party/py/jax/tests/export_back_compat_test.py":341:14)
#loc4 = loc("jit(<lambda>)/jit(main)/iota[dtype=complex64 shape=(9,) dimension=0]"(#loc1))
#loc5 = loc("jit(<lambda>)/jit(main)/reshape[new_sizes=(3, 3) dimensions=None]"(#loc2))
#loc6 = loc("jit(<lambda>)/jit(main)/geqrf"(#loc3))
#loc7 = loc("jit(<lambda>)/jit(main)/qr[full_matrices=True]"(#loc3))
#loc8 = loc("jit(<lambda>)/jit(main)/pad[padding_config=((0, 0, 0), (0, 0, 0))]"(#loc3))
#loc9 = loc("jit(<lambda>)/jit(main)/householder_product"(#loc3))
#loc11 = loc("jit(<lambda>)/jit(main)/jit(triu)/iota[dtype=int32 shape=(3, 3) dimension=0]"(#loc3))
#loc12 = loc("jit(<lambda>)/jit(main)/jit(triu)/add"(#loc3))
#loc13 = loc("jit(<lambda>)/jit(main)/jit(triu)/iota[dtype=int32 shape=(3, 3) dimension=1]"(#loc3))
#loc14 = loc("jit(<lambda>)/jit(main)/jit(triu)/ge"(#loc3))
#loc15 = loc("jit(<lambda>)/jit(main)/jit(triu)/broadcast_in_dim[shape=(3, 3) broadcast_dimensions=()]"(#loc3))
#loc16 = loc("jit(<lambda>)/jit(main)/jit(triu)/select_n"(#loc3))
""",
    mlir_module_serialized=b"ML\xefR\x01StableHLO_v0.9.0\x00\x01)\x05\x01\x03\x01\x03\x05\x03\x19\x07\t\x0b\r\x0f\x11\x13\x15\x17\x19\x1b\x1d\x03\x92\x02\xfb;\x01\x9f\x0f\x0f\x17\x13\x0f\x0b\x13\x0b\x07\x0b\x0b\x13\x0f\x0b\x0b\x0b\x0b\x13\x0b\x0f\x0b\x0b\x13\x0b\x0b\x0b\x0b\x0b\x0b\x0b\x0b\x0b\x13\x1b\x13+\x0b\x0f\x0b\x0b\x0b33\x0b\x0f\x0b\x13\x0b\x13\x0f\x0b\x1b\x0f\x0b\x0f\x0b\x0f\x0b\x0f\x0b\x17\x0f\x0b\x17\x0bS\x13\x0f\x0b#\x0b\x0b\x0b\x0f\x0b\x0bS\x13\x0b\x03]O/\x0b\x0b\x0b/\x0b\x0f\x0b\x0b\x0b\x0b\x0b\x0b\x0f\x0f\x0b\x13\x1b\x0b\x1b\x0b\x0b\x0b\x13\x0b\x0b\x0f\x1f\x0f\x0f\x0b//\x0b\x0f\x17\x1b\x1f\x0b/O/\x0b\x13\x17\x01\x05\x0b\x0f\x037\x17\x0f\x0f\x07\x0f\x0b\x07\x17\x17\x13\x07\x07\x17\x0f\x17\x13\x17\x07\x17\x13\x13\x13\x13\x13\x13\x13\x13\x02f\t\x1d\x81\x05\x1d\x97\x05\x17\x13Z\x05\x17\x03\x03\x15\xd9\x1dW\x05\x05\x1f\x03\x03\x0b\xe1\x05!\x1f\x05#\x05%\x03\x03\x0b\xef\x11\x03\x05\x05'\x05)\x05+\x05-\x03\x03%\xd5\x05/\x1d_\x05\x051\x053\x03\x03\x0b\xdf\x055\x057\x059\x05;\x05=\x05?\x05A\x05C\x05E\x03\x03\x0b\xeb\x03\x05)\xb1+\xed\x03\x03\x15\xf1\x03\tIKM\x19O\x19\x0fQ\x05G\x11\x01\x00\x05I\x05K\x05M\x03\x0b\x1b\xa3\x1d\xbf\x1f\xc1\x0f\xcb!\xcd\x03\x0b\x1b\xad\x1d\xd1\x1f\xad\x0f\xaf!\xd3\x05O\x1d[\x05\x05Q\x03\x03\x0b\xd7\x05S\x03\x03%\xdb\x1de\x05\x05U\x03\x05)\xb1+\xdd\x1dk\x05\x05W\x1do\x05\x05Y\x1ds\x05\x05[\x1dwy\x05]\x17\x13V\x055\x1d}\x7f\x05_\x17\x13V\x05\x1d\x05a\x03\x13/\xb31\xb53\xe35\xa37\xb79\xb9;\xe5=\xbb?\xe9\x03\x03\x15\xf3\x1d\x89\x05\x05c\x03\x07\x8d\xa9\x8f\xa9\x91\xa9\x05e\x05g\x05i\x1d\x95\x05\x05k\x05m\x03\x13/\xb31\xb53\xf55\xa37\xb79\xb9;\xf7=\xbb?\xf9\x03\x03\x9d\xaf\x05o\x1f/!\x00\x00\x00\x00\x00\x00\x00\x00\x01\x00\x00\x00\x00\x00\x00\x00\x1f1\x11\x00\x00\x00\x00\x00\x00\x00\x00\x03\x01\x1dq\x1ds\x1f#\x11\x00\x00\x00\x00\x00\x00\x00\x00\x1du\x03\x03\xcf\x1dw\t\x07\x0b\x03\x1dy\x05\x01\r\x01\x03\x03\xe7\x1f3\x01#%\x03\x05\xc3\xc7\r\x05\xab\xc5\xa5\xa7\x1d{\r\x05\xab\xc9\xa5\xa7\x1d}\x1d\x7f\x1d\x81\r\x03\xa5\xa7#)\x1d\x83\x13\x0b\x01\x1f\x07\t\xff\xff\xff\xff\x1f+\x01\x13\x0b\x05\x07\x05\x1f\t\x11\x00\x00\x00\x00\x00\x00\x00\x00\x1f\r\x11\x03\x00\x00\x00\x00\x00\x00\x00\x1d\x85\x03\x03\x9f\x15\x03\x01\x01\x01\x03\t\x9f\xa1\xbd\xa1\x1f\x07\t\x00\x00\x00\x00\x07\x01\x1f\t\x11\x00\x00\xc0\x7f\x00\x00\xc0\x7f\x1f#!\x00\x00\x00\x00\x00\x00\x00\x00\x01\x00\x00\x00\x00\x00\x00\x00\x1f9\x11\x00\x00\x00\x00\x00\x00\x00\x00\x1d\x87\x03\x05\x9f\xa1\x03\x07\x9f\xbd\xa1\x01\t\x01\x02\x02)\x05\r\r\x0f)\x01\x1b)\x01\x0f\x1d)\x01\x0b\x03'\x01)\x05\r\r\x1b)\x05\r\r\x11)\x03\r\x0f\x13\x1b)\x03\x02\x03\x0f)\x01\x11)\x05\x05\x05\x11)\x03\t\x0b\x11\x01\x05\x05\x05\t\x11\x03\x05\x03\x05)\x03\x01\x0b)\x03%\x0f)\x03\t\x19)\x03\x05\x19)\x03\x01\x19)\x03\x05\x11)\x03\r\x11)\x03\x05\x0b\x04\x92\x05\x05\x01\x11\x11G\x07\x03\x01\t\r\x11\x11S\x07\x03M\x89\t\x03u#\x03-\x15\x06{\x03\x05\x03\x01\x03\x03\x01\r\x03\r\x03\x03\x01\r\x03\r\x11\x07\x01\x83\t\x05\x17\x07\x1d\x03\x03\x03\x03\x01A\x03\x07\x05\x07\x01\x07\x03\x07\x03\x11\x0b\x07\x01C\x03\x1f\x05\r\x13\x05\x07\x01\x07\x03!\x03\x15\x03\x03\x01\x17\x03\t\x05\x07\x01\x07\x03\x05\x03\x19\x05\x07\x01E\x03\x15\x03\x17\x07\x06\x01\x03\x05\x07\x1d\t\x1b\x05\x07\x01\x07\x035\x03\x15\x03\x03\x01\x17\x03\t\x05\x07\x01\x07\x03\x17\x03#\x05\x07\x01\x85\x037\x03!\x07\x06\x01\x03\x17\x07'\x0b%\x03\x03\x87-\x03\t\x17\x07\x93\x8b\x03\x05\x05\x1f+\x03\x03\x03\r\x03\r\x03\x03\x03\r\x03\r\x03\x03\x03\r\x03\r\x11\x07\x03\x99\x07\x05\x07\x1d\x05-)\x03\x03\x03A\x03\x07\x05\x07\x03\x07\x03\x07\x03;\x0b\x07\x03C\x03\x1f\x057=\x05\x07\x03\x07\x03!\x03?\x03\x03\x03\x17\x03\t\x05\x07\x03\x07\x03\x05\x03C\x05\x07\x03E\x03\x15\x03A\x07\x06\x03\x03\x05\x07G5E\x19\x07\t\x9b\x03\x05\x03\x1f\x0f\x04\x11\x05IK\r\x11\tU\x07\x03\x15+\x03\x05\t\t\x03Y#\x03\x13\x03\x03\t]\x03\x07\x05\x07'\x07\x03\x13\x03\x05\x13\x06'\x03\x13\x05\x03\x07\t\x03ca\x03\x13\x0b\x07ig\x03\x15\x05\t\x0b\x03\x03\t-\x03\t\x05\x07m\x07\x03\x05\x03\x0f\x07\x06q\x03\x05\x07\r\x11\x01\x0f\x04\t\x03\x13\x06\x03\x01\x05\x01\x00\xb6\x1a\x89%%\x11\x0f\x0b\t\t\x03\x0b!\x11#\x0fY\x87##%_=\x85\x8bW\xb3K\x9bM\x9bn\x03\x1b%)9\x1f/!)!)#\x1f\x19+\x1b\x1f\x1f\x15\x1d\x15+i\x13\r\x11\x0f\x17\x0f\x1f\x15\x11\x17\x11\x15)\x19\x0f\x0b\x11builtin\x00vhlo\x00module\x00constant_v1\x00broadcast_in_dim_v1\x00select_v1\x00iota_v1\x00compare_v1\x00func_v1\x00return_v1\x00custom_call_v1\x00add_v1\x00reshape_v1\x00pad_v1\x00call_v1\x00value\x00sym_name\x00third_party/py/jax/tests/export_back_compat_test.py\x00broadcast_dimensions\x00arg_attrs\x00function_type\x00res_attrs\x00sym_visibility\x00iota_dimension\x00compare_type\x00comparison_direction\x00api_version\x00backend_config\x00call_target_name\x00called_computations\x00has_side_effect\x00mhlo.backend_config\x00operand_layouts\x00output_operand_aliases\x00result_layouts\x00jax.uses_shape_polymorphism\x00mhlo.num_partitions\x00mhlo.num_replicas\x00jit__lambda_\x00jit(<lambda>)/jit(main)/pjit[in_shardings=(UnspecifiedValue,) out_shardings=(UnspecifiedValue,) in_layouts=(None,) out_layouts=(None,) resource_env=None donated_invars=(False,) name=triu keep_unused=False inline=False]\x00jit(<lambda>)/jit(main)/jit(triu)/iota[dtype=int32 shape=(3, 3) dimension=0]\x00jit(<lambda>)/jit(main)/jit(triu)/add\x00jit(<lambda>)/jit(main)/jit(triu)/iota[dtype=int32 shape=(3, 3) dimension=1]\x00jit(<lambda>)/jit(main)/jit(triu)/ge\x00jit(<lambda>)/jit(main)/jit(triu)/broadcast_in_dim[shape=(3, 3) broadcast_dimensions=()]\x00jit(<lambda>)/jit(main)/jit(triu)/select_n\x00jit(<lambda>)/jit(main)/iota[dtype=complex64 shape=(9,) dimension=0]\x00jit(<lambda>)/jit(main)/reshape[new_sizes=(3, 3) dimensions=None]\x00jit(<lambda>)/jit(main)/geqrf\x00jit(<lambda>)/jit(main)/qr[full_matrices=True]\x00edge_padding_high\x00edge_padding_low\x00interior_padding\x00jit(<lambda>)/jit(main)/pad[padding_config=((0, 0, 0), (0, 0, 0))]\x00jit(<lambda>)/jit(main)/householder_product\x00callee\x00mhlo.layout_mode\x00default\x00jax.result_info\x00triu\x00\x00[0]\x00[1]\x00main\x00public\x00private\x00lapack_cgeqrf_ffi\x00lapack_cungqr_ffi\x00",
    xla_call_module_version=9,
    nr_devices=1,
)  # End paste


# Pasted from the test output (see export_back_compat_test_util.py module docstring)
data_2024_05_31["f32"] = dict(
    testdata_version=1,
    platform='cpu',
    custom_call_targets=['lapack_sgeqrf_ffi', 'lapack_sorgqr_ffi'],
    serialized_date=datetime.date(2024, 5, 31),
    inputs=(),
    expected_outputs=(array([[ 0.        ,  0.91287076,  0.4082487 ],
       [-0.44721356,  0.36514866, -0.8164965 ],
       [-0.8944271 , -0.18257445,  0.40824816]], dtype=float32), array([[-6.7082043e+00, -8.0498438e+00, -9.3914852e+00],
       [ 0.0000000e+00,  1.0954441e+00,  2.1908894e+00],
       [ 0.0000000e+00,  0.0000000e+00,  7.1525574e-07]], dtype=float32)),
    mlir_module_text=r"""
#loc3 = loc("third_party/py/jax/tests/export_back_compat_test.py":342:11)
#loc10 = loc("jit(<lambda>)/jit(main)/pjit[in_shardings=(UnspecifiedValue,) out_shardings=(UnspecifiedValue,) in_layouts=(None,) out_layouts=(None,) resource_env=None donated_invars=(False,) name=triu keep_unused=False inline=False]"(#loc3))
module @jit__lambda_ attributes {jax.uses_shape_polymorphism = false, mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} {
  func.func public @main() -> (tensor<3x3xf32> {jax.result_info = "[0]", mhlo.layout_mode = "default"}, tensor<3x3xf32> {jax.result_info = "[1]", mhlo.layout_mode = "default"}) {
    %0 = stablehlo.iota dim = 0 : tensor<9xf32> loc(#loc4)
    %1 = stablehlo.reshape %0 : (tensor<9xf32>) -> tensor<3x3xf32> loc(#loc5)
    %c = stablehlo.constant dense<3> : tensor<i64> loc(#loc6)
    %c_0 = stablehlo.constant dense<3> : tensor<i64> loc(#loc6)
    %2:4 = stablehlo.custom_call @lapack_sgeqrf_ffi(%1) {mhlo.backend_config = {}, operand_layouts = [dense<[0, 1]> : tensor<2xindex>], output_operand_aliases = [#stablehlo.output_operand_alias<output_tuple_indices = [0], operand_index = 0, operand_tuple_indices = []>], result_layouts = [dense<[0, 1]> : tensor<2xindex>, dense<0> : tensor<1xindex>, dense<> : tensor<0xindex>, dense<0> : tensor<1xindex>]} : (tensor<3x3xf32>) -> (tensor<3x3xf32>, tensor<3xf32>, tensor<i32>, tensor<96xf32>) loc(#loc6)
    %c_1 = stablehlo.constant dense<0> : tensor<i32> loc(#loc6)
    %3 = stablehlo.broadcast_in_dim %c_1, dims = [] : (tensor<i32>) -> tensor<i32> loc(#loc6)
    %4 = stablehlo.compare  EQ, %2#2, %3,  SIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1> loc(#loc6)
    %5 = stablehlo.broadcast_in_dim %4, dims = [] : (tensor<i1>) -> tensor<1x1xi1> loc(#loc6)
    %cst = stablehlo.constant dense<0x7FC00000> : tensor<f32> loc(#loc6)
    %6 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<3x3xf32> loc(#loc6)
    %7 = stablehlo.broadcast_in_dim %5, dims = [0, 1] : (tensor<1x1xi1>) -> tensor<3x3xi1> loc(#loc6)
    %8 = stablehlo.select %7, %2#0, %6 : tensor<3x3xi1>, tensor<3x3xf32> loc(#loc6)
    %9 = stablehlo.broadcast_in_dim %4, dims = [] : (tensor<i1>) -> tensor<1xi1> loc(#loc6)
    %cst_2 = stablehlo.constant dense<0x7FC00000> : tensor<f32> loc(#loc6)
    %10 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<3xf32> loc(#loc6)
    %11 = stablehlo.broadcast_in_dim %9, dims = [0] : (tensor<1xi1>) -> tensor<3xi1> loc(#loc6)
    %12 = stablehlo.select %11, %2#1, %10 : tensor<3xi1>, tensor<3xf32> loc(#loc6)
    %cst_3 = stablehlo.constant dense<0.000000e+00> : tensor<f32> loc(#loc7)
    %13 = stablehlo.pad %8, %cst_3, low = [0, 0], high = [0, 0], interior = [0, 0] : (tensor<3x3xf32>, tensor<f32>) -> tensor<3x3xf32> loc(#loc8)
    %c_4 = stablehlo.constant dense<3> : tensor<i64> loc(#loc9)
    %c_5 = stablehlo.constant dense<3> : tensor<i64> loc(#loc9)
    %c_6 = stablehlo.constant dense<3> : tensor<i64> loc(#loc9)
    %14:3 = stablehlo.custom_call @lapack_sorgqr_ffi(%13, %12) {mhlo.backend_config = {}, operand_layouts = [dense<[0, 1]> : tensor<2xindex>, dense<0> : tensor<1xindex>], output_operand_aliases = [#stablehlo.output_operand_alias<output_tuple_indices = [0], operand_index = 0, operand_tuple_indices = []>], result_layouts = [dense<[0, 1]> : tensor<2xindex>, dense<> : tensor<0xindex>, dense<0> : tensor<1xindex>]} : (tensor<3x3xf32>, tensor<3xf32>) -> (tensor<3x3xf32>, tensor<i32>, tensor<96xf32>) loc(#loc9)
    %c_7 = stablehlo.constant dense<0> : tensor<i32> loc(#loc9)
    %15 = stablehlo.broadcast_in_dim %c_7, dims = [] : (tensor<i32>) -> tensor<i32> loc(#loc9)
    %16 = stablehlo.compare  EQ, %14#1, %15,  SIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1> loc(#loc9)
    %17 = stablehlo.broadcast_in_dim %16, dims = [] : (tensor<i1>) -> tensor<1x1xi1> loc(#loc9)
    %cst_8 = stablehlo.constant dense<0x7FC00000> : tensor<f32> loc(#loc9)
    %18 = stablehlo.broadcast_in_dim %cst_8, dims = [] : (tensor<f32>) -> tensor<3x3xf32> loc(#loc9)
    %19 = stablehlo.broadcast_in_dim %17, dims = [0, 1] : (tensor<1x1xi1>) -> tensor<3x3xi1> loc(#loc9)
    %20 = stablehlo.select %19, %14#0, %18 : tensor<3x3xi1>, tensor<3x3xf32> loc(#loc9)
    %21 = call @triu(%8) : (tensor<3x3xf32>) -> tensor<3x3xf32> loc(#loc10)
    return %20, %21 : tensor<3x3xf32>, tensor<3x3xf32> loc(#loc)
  } loc(#loc)
  func.func private @triu(%arg0: tensor<3x3xf32> {mhlo.layout_mode = "default"} loc("jit(<lambda>)/jit(main)/pjit[in_shardings=(UnspecifiedValue,) out_shardings=(UnspecifiedValue,) in_layouts=(None,) out_layouts=(None,) resource_env=None donated_invars=(False,) name=triu keep_unused=False inline=False]"(#loc3))) -> (tensor<3x3xf32> {mhlo.layout_mode = "default"}) {
    %0 = stablehlo.iota dim = 0 : tensor<3x3xi32> loc(#loc11)
    %c = stablehlo.constant dense<-1> : tensor<i32> loc(#loc10)
    %1 = stablehlo.broadcast_in_dim %c, dims = [] : (tensor<i32>) -> tensor<3x3xi32> loc(#loc12)
    %2 = stablehlo.add %0, %1 : tensor<3x3xi32> loc(#loc12)
    %3 = stablehlo.iota dim = 1 : tensor<3x3xi32> loc(#loc13)
    %4 = stablehlo.compare  GE, %2, %3,  SIGNED : (tensor<3x3xi32>, tensor<3x3xi32>) -> tensor<3x3xi1> loc(#loc14)
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32> loc(#loc10)
    %5 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<3x3xf32> loc(#loc15)
    %6 = stablehlo.select %4, %5, %arg0 : tensor<3x3xi1>, tensor<3x3xf32> loc(#loc16)
    return %6 : tensor<3x3xf32> loc(#loc10)
  } loc(#loc10)
} loc(#loc)
#loc = loc(unknown)
#loc1 = loc("third_party/py/jax/tests/export_back_compat_test.py":341:26)
#loc2 = loc("third_party/py/jax/tests/export_back_compat_test.py":341:14)
#loc4 = loc("jit(<lambda>)/jit(main)/iota[dtype=float32 shape=(9,) dimension=0]"(#loc1))
#loc5 = loc("jit(<lambda>)/jit(main)/reshape[new_sizes=(3, 3) dimensions=None]"(#loc2))
#loc6 = loc("jit(<lambda>)/jit(main)/geqrf"(#loc3))
#loc7 = loc("jit(<lambda>)/jit(main)/qr[full_matrices=True]"(#loc3))
#loc8 = loc("jit(<lambda>)/jit(main)/pad[padding_config=((0, 0, 0), (0, 0, 0))]"(#loc3))
#loc9 = loc("jit(<lambda>)/jit(main)/householder_product"(#loc3))
#loc11 = loc("jit(<lambda>)/jit(main)/jit(triu)/iota[dtype=int32 shape=(3, 3) dimension=0]"(#loc3))
#loc12 = loc("jit(<lambda>)/jit(main)/jit(triu)/add"(#loc3))
#loc13 = loc("jit(<lambda>)/jit(main)/jit(triu)/iota[dtype=int32 shape=(3, 3) dimension=1]"(#loc3))
#loc14 = loc("jit(<lambda>)/jit(main)/jit(triu)/ge"(#loc3))
#loc15 = loc("jit(<lambda>)/jit(main)/jit(triu)/broadcast_in_dim[shape=(3, 3) broadcast_dimensions=()]"(#loc3))
#loc16 = loc("jit(<lambda>)/jit(main)/jit(triu)/select_n"(#loc3))
""",
    mlir_module_serialized=b"ML\xefR\x01StableHLO_v0.9.0\x00\x01)\x05\x01\x03\x01\x03\x05\x03\x19\x07\t\x0b\r\x0f\x11\x13\x15\x17\x19\x1b\x1d\x03\x8e\x02\xfb9\x01\x9f\x0f\x0f\x17\x13\x0f\x0b\x13\x0b\x07\x0b\x0b\x13\x0f\x0b\x0b\x0b\x0b\x13\x0b\x0f\x0b\x0b\x13\x0b\x0b\x0b\x0b\x0b\x0b\x0b\x0b\x0b\x13\x1b\x13+\x0b\x0f\x0b\x0b\x0b33\x0b\x0f\x0b\x13\x0b\x13\x0f\x0b\x1b\x0f\x0b\x0f\x0b\x0f\x0b\x0f\x0b\x17\x0f\x0b\x17\x0bS\x13\x0f\x0b#\x0b\x0b\x0b\x0f\x0b\x0bS\x13\x0b\x03]O/\x0b\x0b\x0b/\x0b\x0f\x0b\x0b\x0b\x0b\x0b\x0b\x0f\x0f\x0b\x13\x1b\x0b\x1b\x0b\x0b\x0b\x13\x0b\x0b\x0f\x1f\x0f\x0f\x0b\x1f/\x0b\x0f\x17\x1b\x1f\x0b\x1fO/\x0b\x13\x17\x01\x05\x0b\x0f\x035\x17\x0f\x0f\x07\x0f\x07\x07\x17\x17\x13\x07\x07\x17\x0f\x17\x13\x17\x17\x13\x13\x13\x13\x13\x13\x13\x13\x02>\t\x1d\x81\x05\x1d\x97\x05\x17\x13Z\x05\x17\x03\x03\x15\xd9\x1dW\x05\x05\x1f\x03\x03\x0b\xe1\x05!\x1f\x05#\x05%\x03\x03\x0b\xef\x11\x03\x05\x05'\x05)\x05+\x05-\x03\x03%\xd5\x05/\x1d_\x05\x051\x053\x03\x03\x0b\xdf\x055\x057\x059\x05;\x05=\x05?\x05A\x05C\x05E\x03\x03\x0b\xeb\x03\x05)\xb1+\xed\x03\x03\x15\xf1\x03\tIKM\x19O\x19\x0fQ\x05G\x11\x01\x00\x05I\x05K\x05M\x03\x0b\x1b\xa3\x1d\xbf\x1f\xc1\x0f\xcb!\xcd\x03\x0b\x1b\xad\x1d\xd1\x1f\xad\x0f\xaf!\xd3\x05O\x1d[\x05\x05Q\x03\x03\x0b\xd7\x05S\x03\x03%\xdb\x1de\x05\x05U\x03\x05)\xb1+\xdd\x1dk\x05\x05W\x1do\x05\x05Y\x1ds\x05\x05[\x1dwy\x05]\x17\x13V\x055\x1d}\x7f\x05_\x17\x13V\x05\x1d\x05a\x03\x13/\xb31\xb53\xe35\xa37\xb79\xb9;\xe5=\xbb?\xe9\x03\x03\x15\xf3\x1d\x89\x05\x05c\x03\x07\x8d\xa9\x8f\xa9\x91\xa9\x05e\x05g\x05i\x1d\x95\x05\x05k\x05m\x03\x13/\xb31\xb53\xf55\xa37\xb79\xb9;\xf7=\xbb?\xf9\x03\x03\x9d\xaf\x05o\x1f-!\x00\x00\x00\x00\x00\x00\x00\x00\x01\x00\x00\x00\x00\x00\x00\x00\x1f/\x11\x00\x00\x00\x00\x00\x00\x00\x00\x03\x01\x1dq\x1ds\x1f#\x11\x00\x00\x00\x00\x00\x00\x00\x00\x1du\x03\x03\xcf\x1dw\t\x07\x0b\x03\x1dy\x05\x01\r\x01\x03\x03\xe7\x1f1\x01#%\x03\x05\xc3\xc7\r\x05\xab\xc5\xa5\xa7\x1d{\r\x05\xab\xc9\xa5\xa7\x1d}\x1d\x7f\x1d\x81\r\x03\xa5\xa7#'\x1d\x83\x13\x0b\x01\x1f\x07\t\xff\xff\xff\xff\x1f)\x01\x13\x0b\x05\x07\x05\x1f\t\t\x00\x00\x00\x00\x1f\r\x11\x03\x00\x00\x00\x00\x00\x00\x00\x1d\x85\x03\x03\x9f\x15\x03\x01\x01\x01\x03\t\x9f\xa1\xbd\xa1\x1f\x07\t\x00\x00\x00\x00\x07\x01\x1f\t\t\x00\x00\xc0\x7f\x1f#!\x00\x00\x00\x00\x00\x00\x00\x00\x01\x00\x00\x00\x00\x00\x00\x00\x1f7\x11\x00\x00\x00\x00\x00\x00\x00\x00\x1d\x87\x03\x05\x9f\xa1\x03\x07\x9f\xbd\xa1\x01\t\x01\x02\x02)\x05\r\r\x0f)\x01\x1b)\x01\x0f\x1d)\x01\x0b\t\x01)\x05\r\r\x1b)\x05\r\r\x11)\x03\r\x0f\x13\x1b)\x03\x02\x03\x0f)\x01\x11)\x05\x05\x05\x11)\x03\t\x0b\x11\x01\x05\x05\x05\x11\x03\x05\x03\x05)\x03\x01\x0b)\x03%\x0f)\x03\t\x19)\x03\x05\x19)\x03\x01\x19)\x03\x05\x11)\x03\r\x11)\x03\x05\x0b\x04\x92\x05\x05\x01\x11\x11G\x07\x03\x01\t\r\x11\x11S\x07\x03M\x89\t\x03u#\x03+\x15\x06{\x03\x05\x03\x01\x03\x03\x01\r\x03\r\x03\x03\x01\r\x03\r\x11\x07\x01\x83\t\x05\x17\x07\x1d\x03\x03\x03\x03\x01A\x03\x07\x05\x07\x01\x07\x03\x07\x03\x11\x0b\x07\x01C\x03\x1f\x05\r\x13\x05\x07\x01\x07\x03!\x03\x15\x03\x03\x01\x17\x03\t\x05\x07\x01\x07\x03\x05\x03\x19\x05\x07\x01E\x03\x15\x03\x17\x07\x06\x01\x03\x05\x07\x1d\t\x1b\x05\x07\x01\x07\x033\x03\x15\x03\x03\x01\x17\x03\t\x05\x07\x01\x07\x03\x17\x03#\x05\x07\x01\x85\x035\x03!\x07\x06\x01\x03\x17\x07'\x0b%\x03\x03\x87-\x03\t\x17\x07\x93\x8b\x03\x05\x05\x1f+\x03\x03\x03\r\x03\r\x03\x03\x03\r\x03\r\x03\x03\x03\r\x03\r\x11\x07\x03\x99\x07\x05\x07\x1d\x05-)\x03\x03\x03A\x03\x07\x05\x07\x03\x07\x03\x07\x03;\x0b\x07\x03C\x03\x1f\x057=\x05\x07\x03\x07\x03!\x03?\x03\x03\x03\x17\x03\t\x05\x07\x03\x07\x03\x05\x03C\x05\x07\x03E\x03\x15\x03A\x07\x06\x03\x03\x05\x07G5E\x19\x07\t\x9b\x03\x05\x03\x1f\x0f\x04\x11\x05IK\r\x11\tU\x07\x03\x15+\x03\x05\t\t\x03Y#\x03\x13\x03\x03\t]\x03\x07\x05\x07'\x07\x03\x13\x03\x05\x13\x06'\x03\x13\x05\x03\x07\t\x03ca\x03\x13\x0b\x07ig\x03\x15\x05\t\x0b\x03\x03\t-\x03\t\x05\x07m\x07\x03\x05\x03\x0f\x07\x06q\x03\x05\x07\r\x11\x01\x0f\x04\t\x03\x13\x06\x03\x01\x05\x01\x00\xae\x1a\x89%%\x11\x0f\x0b\t\t\x03\x0b!\x11#\x0fY\x87##%_=\x85\x87W\xb3K\x9bM\x9bn\x03\x1b%)9\x1f/!)!)#\x1f\x19+\x1b\x1f\x1f\x15\x1d\x15+i\x13\r\x11\x0f\x17\x0f\x1f\x15\x11\x17\x11\x15)\x19\x0f\x0b\x11builtin\x00vhlo\x00module\x00constant_v1\x00broadcast_in_dim_v1\x00select_v1\x00iota_v1\x00compare_v1\x00func_v1\x00return_v1\x00custom_call_v1\x00add_v1\x00reshape_v1\x00pad_v1\x00call_v1\x00value\x00sym_name\x00third_party/py/jax/tests/export_back_compat_test.py\x00broadcast_dimensions\x00arg_attrs\x00function_type\x00res_attrs\x00sym_visibility\x00iota_dimension\x00compare_type\x00comparison_direction\x00api_version\x00backend_config\x00call_target_name\x00called_computations\x00has_side_effect\x00mhlo.backend_config\x00operand_layouts\x00output_operand_aliases\x00result_layouts\x00jax.uses_shape_polymorphism\x00mhlo.num_partitions\x00mhlo.num_replicas\x00jit__lambda_\x00jit(<lambda>)/jit(main)/pjit[in_shardings=(UnspecifiedValue,) out_shardings=(UnspecifiedValue,) in_layouts=(None,) out_layouts=(None,) resource_env=None donated_invars=(False,) name=triu keep_unused=False inline=False]\x00jit(<lambda>)/jit(main)/jit(triu)/iota[dtype=int32 shape=(3, 3) dimension=0]\x00jit(<lambda>)/jit(main)/jit(triu)/add\x00jit(<lambda>)/jit(main)/jit(triu)/iota[dtype=int32 shape=(3, 3) dimension=1]\x00jit(<lambda>)/jit(main)/jit(triu)/ge\x00jit(<lambda>)/jit(main)/jit(triu)/broadcast_in_dim[shape=(3, 3) broadcast_dimensions=()]\x00jit(<lambda>)/jit(main)/jit(triu)/select_n\x00jit(<lambda>)/jit(main)/iota[dtype=float32 shape=(9,) dimension=0]\x00jit(<lambda>)/jit(main)/reshape[new_sizes=(3, 3) dimensions=None]\x00jit(<lambda>)/jit(main)/geqrf\x00jit(<lambda>)/jit(main)/qr[full_matrices=True]\x00edge_padding_high\x00edge_padding_low\x00interior_padding\x00jit(<lambda>)/jit(main)/pad[padding_config=((0, 0, 0), (0, 0, 0))]\x00jit(<lambda>)/jit(main)/householder_product\x00callee\x00mhlo.layout_mode\x00default\x00jax.result_info\x00triu\x00\x00[0]\x00[1]\x00main\x00public\x00private\x00lapack_sgeqrf_ffi\x00lapack_sorgqr_ffi\x00",
    xla_call_module_version=9,
    nr_devices=1,
)  # End paste


# Pasted from the test output (see export_back_compat_test_util.py module docstring)
data_2024_05_31["f64"] = dict(
    testdata_version=1,
    platform='cpu',
    custom_call_targets=['lapack_dgeqrf_ffi', 'lapack_dorgqr_ffi'],
    serialized_date=datetime.date(2024, 5, 31),
    inputs=(),
    expected_outputs=(array([[ 0.                 ,  0.9128709291752773 ,  0.40824829046386235],
       [-0.447213595499958  ,  0.3651483716701102 , -0.8164965809277263 ],
       [-0.894427190999916  , -0.1825741858350548 ,  0.40824829046386324]]), array([[-6.7082039324993694e+00, -8.0498447189992444e+00,
        -9.3914855054991175e+00],
       [ 0.0000000000000000e+00,  1.0954451150103341e+00,
         2.1908902300206665e+00],
       [ 0.0000000000000000e+00,  0.0000000000000000e+00,
        -8.8817841970012523e-16]])),
    mlir_module_text=r"""
#loc3 = loc("third_party/py/jax/tests/export_back_compat_test.py":342:11)
#loc10 = loc("jit(<lambda>)/jit(main)/pjit[in_shardings=(UnspecifiedValue,) out_shardings=(UnspecifiedValue,) in_layouts=(None,) out_layouts=(None,) resource_env=None donated_invars=(False,) name=triu keep_unused=False inline=False]"(#loc3))
module @jit__lambda_ attributes {jax.uses_shape_polymorphism = false, mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} {
  func.func public @main() -> (tensor<3x3xf64> {jax.result_info = "[0]", mhlo.layout_mode = "default"}, tensor<3x3xf64> {jax.result_info = "[1]", mhlo.layout_mode = "default"}) {
    %0 = stablehlo.iota dim = 0 : tensor<9xf64> loc(#loc4)
    %1 = stablehlo.reshape %0 : (tensor<9xf64>) -> tensor<3x3xf64> loc(#loc5)
    %c = stablehlo.constant dense<3> : tensor<i64> loc(#loc6)
    %c_0 = stablehlo.constant dense<3> : tensor<i64> loc(#loc6)
    %2:4 = stablehlo.custom_call @lapack_dgeqrf_ffi(%1) {mhlo.backend_config = {}, operand_layouts = [dense<[0, 1]> : tensor<2xindex>], output_operand_aliases = [#stablehlo.output_operand_alias<output_tuple_indices = [0], operand_index = 0, operand_tuple_indices = []>], result_layouts = [dense<[0, 1]> : tensor<2xindex>, dense<0> : tensor<1xindex>, dense<> : tensor<0xindex>, dense<0> : tensor<1xindex>]} : (tensor<3x3xf64>) -> (tensor<3x3xf64>, tensor<3xf64>, tensor<i32>, tensor<96xf64>) loc(#loc6)
    %c_1 = stablehlo.constant dense<0> : tensor<i32> loc(#loc6)
    %3 = stablehlo.broadcast_in_dim %c_1, dims = [] : (tensor<i32>) -> tensor<i32> loc(#loc6)
    %4 = stablehlo.compare  EQ, %2#2, %3,  SIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1> loc(#loc6)
    %5 = stablehlo.broadcast_in_dim %4, dims = [] : (tensor<i1>) -> tensor<1x1xi1> loc(#loc6)
    %cst = stablehlo.constant dense<0x7FF8000000000000> : tensor<f64> loc(#loc6)
    %6 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f64>) -> tensor<3x3xf64> loc(#loc6)
    %7 = stablehlo.broadcast_in_dim %5, dims = [0, 1] : (tensor<1x1xi1>) -> tensor<3x3xi1> loc(#loc6)
    %8 = stablehlo.select %7, %2#0, %6 : tensor<3x3xi1>, tensor<3x3xf64> loc(#loc6)
    %9 = stablehlo.broadcast_in_dim %4, dims = [] : (tensor<i1>) -> tensor<1xi1> loc(#loc6)
    %cst_2 = stablehlo.constant dense<0x7FF8000000000000> : tensor<f64> loc(#loc6)
    %10 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f64>) -> tensor<3xf64> loc(#loc6)
    %11 = stablehlo.broadcast_in_dim %9, dims = [0] : (tensor<1xi1>) -> tensor<3xi1> loc(#loc6)
    %12 = stablehlo.select %11, %2#1, %10 : tensor<3xi1>, tensor<3xf64> loc(#loc6)
    %cst_3 = stablehlo.constant dense<0.000000e+00> : tensor<f64> loc(#loc7)
    %13 = stablehlo.pad %8, %cst_3, low = [0, 0], high = [0, 0], interior = [0, 0] : (tensor<3x3xf64>, tensor<f64>) -> tensor<3x3xf64> loc(#loc8)
    %c_4 = stablehlo.constant dense<3> : tensor<i64> loc(#loc9)
    %c_5 = stablehlo.constant dense<3> : tensor<i64> loc(#loc9)
    %c_6 = stablehlo.constant dense<3> : tensor<i64> loc(#loc9)
    %14:3 = stablehlo.custom_call @lapack_dorgqr_ffi(%13, %12) {mhlo.backend_config = {}, operand_layouts = [dense<[0, 1]> : tensor<2xindex>, dense<0> : tensor<1xindex>], output_operand_aliases = [#stablehlo.output_operand_alias<output_tuple_indices = [0], operand_index = 0, operand_tuple_indices = []>], result_layouts = [dense<[0, 1]> : tensor<2xindex>, dense<> : tensor<0xindex>, dense<0> : tensor<1xindex>]} : (tensor<3x3xf64>, tensor<3xf64>) -> (tensor<3x3xf64>, tensor<i32>, tensor<96xf64>) loc(#loc9)
    %c_7 = stablehlo.constant dense<0> : tensor<i32> loc(#loc9)
    %15 = stablehlo.broadcast_in_dim %c_7, dims = [] : (tensor<i32>) -> tensor<i32> loc(#loc9)
    %16 = stablehlo.compare  EQ, %14#1, %15,  SIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1> loc(#loc9)
    %17 = stablehlo.broadcast_in_dim %16, dims = [] : (tensor<i1>) -> tensor<1x1xi1> loc(#loc9)
    %cst_8 = stablehlo.constant dense<0x7FF8000000000000> : tensor<f64> loc(#loc9)
    %18 = stablehlo.broadcast_in_dim %cst_8, dims = [] : (tensor<f64>) -> tensor<3x3xf64> loc(#loc9)
    %19 = stablehlo.broadcast_in_dim %17, dims = [0, 1] : (tensor<1x1xi1>) -> tensor<3x3xi1> loc(#loc9)
    %20 = stablehlo.select %19, %14#0, %18 : tensor<3x3xi1>, tensor<3x3xf64> loc(#loc9)
    %21 = call @triu(%8) : (tensor<3x3xf64>) -> tensor<3x3xf64> loc(#loc10)
    return %20, %21 : tensor<3x3xf64>, tensor<3x3xf64> loc(#loc)
  } loc(#loc)
  func.func private @triu(%arg0: tensor<3x3xf64> {mhlo.layout_mode = "default"} loc("jit(<lambda>)/jit(main)/pjit[in_shardings=(UnspecifiedValue,) out_shardings=(UnspecifiedValue,) in_layouts=(None,) out_layouts=(None,) resource_env=None donated_invars=(False,) name=triu keep_unused=False inline=False]"(#loc3))) -> (tensor<3x3xf64> {mhlo.layout_mode = "default"}) {
    %0 = stablehlo.iota dim = 0 : tensor<3x3xi32> loc(#loc11)
    %c = stablehlo.constant dense<-1> : tensor<i32> loc(#loc10)
    %1 = stablehlo.broadcast_in_dim %c, dims = [] : (tensor<i32>) -> tensor<3x3xi32> loc(#loc12)
    %2 = stablehlo.add %0, %1 : tensor<3x3xi32> loc(#loc12)
    %3 = stablehlo.iota dim = 1 : tensor<3x3xi32> loc(#loc13)
    %4 = stablehlo.compare  GE, %2, %3,  SIGNED : (tensor<3x3xi32>, tensor<3x3xi32>) -> tensor<3x3xi1> loc(#loc14)
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f64> loc(#loc10)
    %5 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f64>) -> tensor<3x3xf64> loc(#loc15)
    %6 = stablehlo.select %4, %5, %arg0 : tensor<3x3xi1>, tensor<3x3xf64> loc(#loc16)
    return %6 : tensor<3x3xf64> loc(#loc10)
  } loc(#loc10)
} loc(#loc)
#loc = loc(unknown)
#loc1 = loc("third_party/py/jax/tests/export_back_compat_test.py":341:26)
#loc2 = loc("third_party/py/jax/tests/export_back_compat_test.py":341:14)
#loc4 = loc("jit(<lambda>)/jit(main)/iota[dtype=float64 shape=(9,) dimension=0]"(#loc1))
#loc5 = loc("jit(<lambda>)/jit(main)/reshape[new_sizes=(3, 3) dimensions=None]"(#loc2))
#loc6 = loc("jit(<lambda>)/jit(main)/geqrf"(#loc3))
#loc7 = loc("jit(<lambda>)/jit(main)/qr[full_matrices=True]"(#loc3))
#loc8 = loc("jit(<lambda>)/jit(main)/pad[padding_config=((0, 0, 0), (0, 0, 0))]"(#loc3))
#loc9 = loc("jit(<lambda>)/jit(main)/householder_product"(#loc3))
#loc11 = loc("jit(<lambda>)/jit(main)/jit(triu)/iota[dtype=int32 shape=(3, 3) dimension=0]"(#loc3))
#loc12 = loc("jit(<lambda>)/jit(main)/jit(triu)/add"(#loc3))
#loc13 = loc("jit(<lambda>)/jit(main)/jit(triu)/iota[dtype=int32 shape=(3, 3) dimension=1]"(#loc3))
#loc14 = loc("jit(<lambda>)/jit(main)/jit(triu)/ge"(#loc3))
#loc15 = loc("jit(<lambda>)/jit(main)/jit(triu)/broadcast_in_dim[shape=(3, 3) broadcast_dimensions=()]"(#loc3))
#loc16 = loc("jit(<lambda>)/jit(main)/jit(triu)/select_n"(#loc3))
""",
    mlir_module_serialized=b"ML\xefR\x01StableHLO_v0.9.0\x00\x01)\x05\x01\x03\x01\x03\x05\x03\x19\x07\t\x0b\r\x0f\x11\x13\x15\x17\x19\x1b\x1d\x03\x8e\x02\xfb9\x01\x9f\x0f\x0f\x17\x13\x0f\x0b\x13\x0b\x07\x0b\x0b\x13\x0f\x0b\x0b\x0b\x0b\x13\x0b\x0f\x0b\x0b\x13\x0b\x0b\x0b\x0b\x0b\x0b\x0b\x0b\x0b\x13\x1b\x13+\x0b\x0f\x0b\x0b\x0b33\x0b\x0f\x0b\x13\x0b\x13\x0f\x0b\x1b\x0f\x0b\x0f\x0b\x0f\x0b\x0f\x0b\x17\x0f\x0b\x17\x0bS\x13\x0f\x0b#\x0b\x0b\x0b\x0f\x0b\x0bS\x13\x0b\x03]O/\x0b\x0b\x0b/\x0b\x0f\x0b\x0b\x0b\x0b\x0b\x0b\x0f\x0f\x0b\x13\x1b\x0b\x1b\x0b\x0b\x0b\x13\x0b\x0b\x0f\x1f\x0f\x0f\x0b//\x0b\x0f\x17\x1b\x1f\x0b/O/\x0b\x13\x17\x01\x05\x0b\x0f\x035\x17\x0f\x0f\x07\x0f\x07\x07\x17\x17\x13\x07\x07\x17\x0f\x17\x13\x17\x17\x13\x13\x13\x13\x13\x13\x13\x13\x02^\t\x1d\x81\x05\x1d\x97\x05\x17\x13Z\x05\x17\x03\x03\x15\xd9\x1dW\x05\x05\x1f\x03\x03\x0b\xe1\x05!\x1f\x05#\x05%\x03\x03\x0b\xef\x11\x03\x05\x05'\x05)\x05+\x05-\x03\x03%\xd5\x05/\x1d_\x05\x051\x053\x03\x03\x0b\xdf\x055\x057\x059\x05;\x05=\x05?\x05A\x05C\x05E\x03\x03\x0b\xeb\x03\x05)\xb1+\xed\x03\x03\x15\xf1\x03\tIKM\x19O\x19\x0fQ\x05G\x11\x01\x00\x05I\x05K\x05M\x03\x0b\x1b\xa3\x1d\xbf\x1f\xc1\x0f\xcb!\xcd\x03\x0b\x1b\xad\x1d\xd1\x1f\xad\x0f\xaf!\xd3\x05O\x1d[\x05\x05Q\x03\x03\x0b\xd7\x05S\x03\x03%\xdb\x1de\x05\x05U\x03\x05)\xb1+\xdd\x1dk\x05\x05W\x1do\x05\x05Y\x1ds\x05\x05[\x1dwy\x05]\x17\x13V\x055\x1d}\x7f\x05_\x17\x13V\x05\x1d\x05a\x03\x13/\xb31\xb53\xe35\xa37\xb79\xb9;\xe5=\xbb?\xe9\x03\x03\x15\xf3\x1d\x89\x05\x05c\x03\x07\x8d\xa9\x8f\xa9\x91\xa9\x05e\x05g\x05i\x1d\x95\x05\x05k\x05m\x03\x13/\xb31\xb53\xf55\xa37\xb79\xb9;\xf7=\xbb?\xf9\x03\x03\x9d\xaf\x05o\x1f-!\x00\x00\x00\x00\x00\x00\x00\x00\x01\x00\x00\x00\x00\x00\x00\x00\x1f/\x11\x00\x00\x00\x00\x00\x00\x00\x00\x03\x01\x1dq\x1ds\x1f#\x11\x00\x00\x00\x00\x00\x00\x00\x00\x1du\x03\x03\xcf\x1dw\t\x07\x0b\x03\x1dy\x05\x01\r\x01\x03\x03\xe7\x1f1\x01#%\x03\x05\xc3\xc7\r\x05\xab\xc5\xa5\xa7\x1d{\r\x05\xab\xc9\xa5\xa7\x1d}\x1d\x7f\x1d\x81\r\x03\xa5\xa7#'\x1d\x83\x13\x0b\x01\x1f\x07\t\xff\xff\xff\xff\x1f)\x01\x13\x0b\x05\x07\x05\x1f\t\x11\x00\x00\x00\x00\x00\x00\x00\x00\x1f\r\x11\x03\x00\x00\x00\x00\x00\x00\x00\x1d\x85\x03\x03\x9f\x15\x03\x01\x01\x01\x03\t\x9f\xa1\xbd\xa1\x1f\x07\t\x00\x00\x00\x00\x07\x01\x1f\t\x11\x00\x00\x00\x00\x00\x00\xf8\x7f\x1f#!\x00\x00\x00\x00\x00\x00\x00\x00\x01\x00\x00\x00\x00\x00\x00\x00\x1f7\x11\x00\x00\x00\x00\x00\x00\x00\x00\x1d\x87\x03\x05\x9f\xa1\x03\x07\x9f\xbd\xa1\x01\t\x01\x02\x02)\x05\r\r\x0f)\x01\x1b)\x01\x0f\x1d)\x01\x0b\x0b\x01)\x05\r\r\x1b)\x05\r\r\x11)\x03\r\x0f\x13\x1b)\x03\x02\x03\x0f)\x01\x11)\x05\x05\x05\x11)\x03\t\x0b\x11\x01\x05\x05\x05\x11\x03\x05\x03\x05)\x03\x01\x0b)\x03%\x0f)\x03\t\x19)\x03\x05\x19)\x03\x01\x19)\x03\x05\x11)\x03\r\x11)\x03\x05\x0b\x04\x92\x05\x05\x01\x11\x11G\x07\x03\x01\t\r\x11\x11S\x07\x03M\x89\t\x03u#\x03+\x15\x06{\x03\x05\x03\x01\x03\x03\x01\r\x03\r\x03\x03\x01\r\x03\r\x11\x07\x01\x83\t\x05\x17\x07\x1d\x03\x03\x03\x03\x01A\x03\x07\x05\x07\x01\x07\x03\x07\x03\x11\x0b\x07\x01C\x03\x1f\x05\r\x13\x05\x07\x01\x07\x03!\x03\x15\x03\x03\x01\x17\x03\t\x05\x07\x01\x07\x03\x05\x03\x19\x05\x07\x01E\x03\x15\x03\x17\x07\x06\x01\x03\x05\x07\x1d\t\x1b\x05\x07\x01\x07\x033\x03\x15\x03\x03\x01\x17\x03\t\x05\x07\x01\x07\x03\x17\x03#\x05\x07\x01\x85\x035\x03!\x07\x06\x01\x03\x17\x07'\x0b%\x03\x03\x87-\x03\t\x17\x07\x93\x8b\x03\x05\x05\x1f+\x03\x03\x03\r\x03\r\x03\x03\x03\r\x03\r\x03\x03\x03\r\x03\r\x11\x07\x03\x99\x07\x05\x07\x1d\x05-)\x03\x03\x03A\x03\x07\x05\x07\x03\x07\x03\x07\x03;\x0b\x07\x03C\x03\x1f\x057=\x05\x07\x03\x07\x03!\x03?\x03\x03\x03\x17\x03\t\x05\x07\x03\x07\x03\x05\x03C\x05\x07\x03E\x03\x15\x03A\x07\x06\x03\x03\x05\x07G5E\x19\x07\t\x9b\x03\x05\x03\x1f\x0f\x04\x11\x05IK\r\x11\tU\x07\x03\x15+\x03\x05\t\t\x03Y#\x03\x13\x03\x03\t]\x03\x07\x05\x07'\x07\x03\x13\x03\x05\x13\x06'\x03\x13\x05\x03\x07\t\x03ca\x03\x13\x0b\x07ig\x03\x15\x05\t\x0b\x03\x03\t-\x03\t\x05\x07m\x07\x03\x05\x03\x0f\x07\x06q\x03\x05\x07\r\x11\x01\x0f\x04\t\x03\x13\x06\x03\x01\x05\x01\x00\xae\x1a\x89%%\x11\x0f\x0b\t\t\x03\x0b!\x11#\x0fY\x87##%_=\x85\x87W\xb3K\x9bM\x9bn\x03\x1b%)9\x1f/!)!)#\x1f\x19+\x1b\x1f\x1f\x15\x1d\x15+i\x13\r\x11\x0f\x17\x0f\x1f\x15\x11\x17\x11\x15)\x19\x0f\x0b\x11builtin\x00vhlo\x00module\x00constant_v1\x00broadcast_in_dim_v1\x00select_v1\x00iota_v1\x00compare_v1\x00func_v1\x00return_v1\x00custom_call_v1\x00add_v1\x00reshape_v1\x00pad_v1\x00call_v1\x00value\x00sym_name\x00third_party/py/jax/tests/export_back_compat_test.py\x00broadcast_dimensions\x00arg_attrs\x00function_type\x00res_attrs\x00sym_visibility\x00iota_dimension\x00compare_type\x00comparison_direction\x00api_version\x00backend_config\x00call_target_name\x00called_computations\x00has_side_effect\x00mhlo.backend_config\x00operand_layouts\x00output_operand_aliases\x00result_layouts\x00jax.uses_shape_polymorphism\x00mhlo.num_partitions\x00mhlo.num_replicas\x00jit__lambda_\x00jit(<lambda>)/jit(main)/pjit[in_shardings=(UnspecifiedValue,) out_shardings=(UnspecifiedValue,) in_layouts=(None,) out_layouts=(None,) resource_env=None donated_invars=(False,) name=triu keep_unused=False inline=False]\x00jit(<lambda>)/jit(main)/jit(triu)/iota[dtype=int32 shape=(3, 3) dimension=0]\x00jit(<lambda>)/jit(main)/jit(triu)/add\x00jit(<lambda>)/jit(main)/jit(triu)/iota[dtype=int32 shape=(3, 3) dimension=1]\x00jit(<lambda>)/jit(main)/jit(triu)/ge\x00jit(<lambda>)/jit(main)/jit(triu)/broadcast_in_dim[shape=(3, 3) broadcast_dimensions=()]\x00jit(<lambda>)/jit(main)/jit(triu)/select_n\x00jit(<lambda>)/jit(main)/iota[dtype=float64 shape=(9,) dimension=0]\x00jit(<lambda>)/jit(main)/reshape[new_sizes=(3, 3) dimensions=None]\x00jit(<lambda>)/jit(main)/geqrf\x00jit(<lambda>)/jit(main)/qr[full_matrices=True]\x00edge_padding_high\x00edge_padding_low\x00interior_padding\x00jit(<lambda>)/jit(main)/pad[padding_config=((0, 0, 0), (0, 0, 0))]\x00jit(<lambda>)/jit(main)/householder_product\x00callee\x00mhlo.layout_mode\x00default\x00jax.result_info\x00triu\x00\x00[0]\x00[1]\x00main\x00public\x00private\x00lapack_dgeqrf_ffi\x00lapack_dorgqr_ffi\x00",
    xla_call_module_version=9,
    nr_devices=1,
)  # End paste

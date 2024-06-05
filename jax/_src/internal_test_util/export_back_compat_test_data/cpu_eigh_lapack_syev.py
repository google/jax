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

data_2023_03_17 = dict(
    # Pasted from the test output (see back_compat_test.py module docstring)
    f32=dict(
        testdata_version=1,
        platform='cpu',
        custom_call_targets=['lapack_ssyevd'],
        serialized_date=datetime.date(2023, 3, 17),
        inputs=(),
        expected_outputs=(array([[-0.6185769    , -0.20142993   , -0.09725195   ,  0.62983674   ,
            -0.07926044   ,  0.3605001    , -0.019093221  , -0.18446997   ],
          [-0.47070873   ,  0.29325768   , -0.19454119   , -0.6394365    ,
            0.0622955    ,  0.33249345   ,  0.28112718   , -0.22856665   ],
          [-0.32284075   , -0.12361939   ,  0.20547704   , -0.18307868   ,
            0.47294614   , -0.3170349    , -0.6373532    , -0.27266347   ],
          [-0.17497246   , -0.079641335  ,  0.15042791   , -0.15416273   ,
            -0.815209     , -0.38054234   , -0.083263926  , -0.31676024   ],
          [-0.027104253  , -0.26490977   ,  0.32271704   ,  0.08653544   ,
            0.30305928   , -0.33998996   ,  0.6926741    , -0.360857     ],
          [ 0.12076397   ,  0.43288827   , -0.64385164   ,  0.2652551    ,
            0.09482376   , -0.37435007   ,  0.00091664493, -0.40495378   ],
          [ 0.26863196   ,  0.51607686   ,  0.53846526   ,  0.16969058   ,
            -0.021670295  ,  0.35755336   , -0.113144726  , -0.4490505    ],
          [ 0.4165004    , -0.57262254   , -0.2814425    , -0.17463988   ,
            -0.01698498   ,  0.3613705    , -0.12186296   , -0.49314725   ]],
          dtype=float32), array([-2.4598808e+01, -3.3105560e-05, -3.1002426e-05, -1.0103593e-05,
          -1.0022322e-05,  4.0141886e-06,  9.5510331e-06,  2.7659882e+02],
          dtype=float32)),
        mlir_module_text=r"""
module @jit__lambda_ {
  func.func public @main() -> (tensor<8x8xf32> {jax.result_info = "[0]"}, tensor<8xf32> {jax.result_info = "[1]"}) {
    %0 = stablehlo.iota dim = 0 : tensor<64xf32>
    %1 = stablehlo.reshape %0 : (tensor<64xf32>) -> tensor<8x8xf32>
    %2 = stablehlo.transpose %1, dims = [1, 0] : (tensor<8x8xf32>) -> tensor<8x8xf32>
    %3 = stablehlo.add %1, %2 : tensor<8x8xf32>
    %4 = stablehlo.constant dense<2.000000e+00> : tensor<f32>
    %5 = stablehlo.broadcast_in_dim %4, dims = [] : (tensor<f32>) -> tensor<8x8xf32>
    %6 = stablehlo.divide %3, %5 : tensor<8x8xf32>
    %7 = call @tril(%6) : (tensor<8x8xf32>) -> tensor<8x8xf32>
    %8 = stablehlo.constant dense<1> : tensor<i32>
    %9 = stablehlo.constant dense<1> : tensor<i32>
    %10 = stablehlo.constant dense<8> : tensor<i32>
    %11 = stablehlo.custom_call @lapack_ssyevd(%8, %9, %10, %7) {api_version = 2 : i32, operand_layouts = [dense<> : tensor<0xindex>, dense<> : tensor<0xindex>, dense<> : tensor<0xindex>, dense<[0, 1]> : tensor<2xindex>], output_operand_aliases = [#stablehlo.output_operand_alias<output_tuple_indices = [0], operand_index = 3, operand_tuple_indices = []>], result_layouts = [dense<[0, 1]> : tensor<2xindex>, dense<0> : tensor<1xindex>, dense<> : tensor<0xindex>, dense<0> : tensor<1xindex>, dense<0> : tensor<1xindex>]} : (tensor<i32>, tensor<i32>, tensor<i32>, tensor<8x8xf32>) -> tuple<tensor<8x8xf32>, tensor<8xf32>, tensor<i32>, tensor<177xf32>, tensor<43xi32>>
    %12 = stablehlo.get_tuple_element %11[0] : (tuple<tensor<8x8xf32>, tensor<8xf32>, tensor<i32>, tensor<177xf32>, tensor<43xi32>>) -> tensor<8x8xf32>
    %13 = stablehlo.get_tuple_element %11[1] : (tuple<tensor<8x8xf32>, tensor<8xf32>, tensor<i32>, tensor<177xf32>, tensor<43xi32>>) -> tensor<8xf32>
    %14 = stablehlo.get_tuple_element %11[2] : (tuple<tensor<8x8xf32>, tensor<8xf32>, tensor<i32>, tensor<177xf32>, tensor<43xi32>>) -> tensor<i32>
    %15 = stablehlo.get_tuple_element %11[3] : (tuple<tensor<8x8xf32>, tensor<8xf32>, tensor<i32>, tensor<177xf32>, tensor<43xi32>>) -> tensor<177xf32>
    %16 = stablehlo.get_tuple_element %11[4] : (tuple<tensor<8x8xf32>, tensor<8xf32>, tensor<i32>, tensor<177xf32>, tensor<43xi32>>) -> tensor<43xi32>
    %17 = stablehlo.constant dense<0> : tensor<i32>
    %18 = stablehlo.broadcast_in_dim %17, dims = [] : (tensor<i32>) -> tensor<i32>
    %19 = stablehlo.compare  EQ, %14, %18,  SIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1>
    %20 = stablehlo.broadcast_in_dim %19, dims = [] : (tensor<i1>) -> tensor<1x1xi1>
    %21 = stablehlo.constant dense<0x7FC00000> : tensor<f32>
    %22 = stablehlo.broadcast_in_dim %21, dims = [] : (tensor<f32>) -> tensor<8x8xf32>
    %23 = stablehlo.broadcast_in_dim %20, dims = [0, 1] : (tensor<1x1xi1>) -> tensor<8x8xi1>
    %24 = stablehlo.select %23, %12, %22 : tensor<8x8xi1>, tensor<8x8xf32>
    %25 = stablehlo.broadcast_in_dim %19, dims = [] : (tensor<i1>) -> tensor<1xi1>
    %26 = stablehlo.constant dense<0x7FC00000> : tensor<f32>
    %27 = stablehlo.broadcast_in_dim %26, dims = [] : (tensor<f32>) -> tensor<8xf32>
    %28 = stablehlo.broadcast_in_dim %25, dims = [0] : (tensor<1xi1>) -> tensor<8xi1>
    %29 = stablehlo.select %28, %13, %27 : tensor<8xi1>, tensor<8xf32>
    return %24, %29 : tensor<8x8xf32>, tensor<8xf32>
  }
  func.func private @tril(%arg0: tensor<8x8xf32>) -> tensor<8x8xf32> {
    %0 = stablehlo.iota dim = 0 : tensor<8x8xi32>
    %1 = stablehlo.constant dense<0> : tensor<i32>
    %2 = stablehlo.broadcast_in_dim %1, dims = [] : (tensor<i32>) -> tensor<8x8xi32>
    %3 = stablehlo.add %0, %2 : tensor<8x8xi32>
    %4 = stablehlo.iota dim = 1 : tensor<8x8xi32>
    %5 = stablehlo.compare  GE, %3, %4,  SIGNED : (tensor<8x8xi32>, tensor<8x8xi32>) -> tensor<8x8xi1>
    %6 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %7 = stablehlo.broadcast_in_dim %6, dims = [] : (tensor<f32>) -> tensor<8x8xf32>
    %8 = stablehlo.select %5, %arg0, %7 : tensor<8x8xi1>, tensor<8x8xf32>
    return %8 : tensor<8x8xf32>
  }
}
""",
        mlir_module_serialized=b"ML\xefR\x03MLIRxxx-trunk\x00\x01-\x05\x01\x05\x01\x03\x05\x03\x1d\x07\t\x0b\r\x0f\x11\x13\x15\x17\x19\x1b\x1d\x1f!\x03z\x02\xf77\x01\x9b\x0f\x17\x13\x0b\x07\x0f\x0b\x0b\x0b\x0b\x17\x0b\x0b\x0b\x0b\x13\x0b\x13\x0f\x0b\x0b\x17\x0f\x13\x13\x13\x0b33\x0b\x0f\x0b\x0b\x13\x0f\x0b\x1b\x0f\x0b\x13\x0f\x0b\x0f\x0b\x0f\x0b\x0f\x0b\x13\x0b\x0f\x0b\x0f\x0b\x13\x0b\x13\x0b\x0b\x13K\x0b\x0b\x0b\x0b\x0b\x0b\x0b\x0b\x13\x13\x13\x13\x13\x1b\x13\x13\x03]\x0f/\x0b\x0b\x0f\x0b\x0bO\x0b\x13\x13\x0b\x13\x0b\x0b\x0b\x0b\x0b\x0b\x0f\x1f\x0f\x0f\x0b\x1fO\x1f\x1f\x1f\x0b\x0b\x0b\x0b\x1b\x0f\x17\x1f\x0f\x0f\x0f\x0f\x0f\x0b\x1fO/\x037\x17\x0f\x07\x0f\x07\x13\x07\x07\x17\x07\x17\x13\x17\x13\x17\x17\x13\x17\x1f\x13\x13\x13\x0f\x17\x13\x13\x13\x02\n\t\x1du\x03\x17\x11\xf6\x04\x01\x03\x03\x13\xc5\x05#\x1f\x1d;\x03\x05%\x05'\x05)\x05+\x17\x11\xf2\x04\x01\x05-\x05/\x051\x053\x03\x03!\xc1\x055\x03\x03\x07\xc3\x1dA\x03\x057\x059\x17\x11\xea\x04\x01\x1do\x15\x03\x03\x07\xd1\x03\x03\x07\xf1\x03\x03\x0f5\x05;\x03\x0b\x17\x9f\x19\xab\x1b\xad\x0f\xb7\x1d\xb9\x03\x0b\x17\xa3\x19\xbd\x1b\xa3\x0f\xa5\x1d\xbf\x05=\x1d?\x03\x05?\x05A\x03\x03!\xc7\x1dG\x03\x05C\x03\x05'\xa7)\xc9\x1dM\x03\x05E\x03\x03\x07\xcb\x1dS\x03\x05G\x1dW\x03\x05I\x1d[+\x05K\x1d_+\x05M\x03\x03c\xcd\x05O\x1dg\x15\x05Q\x1dk\x15\x05S\x03\x03\x07\xcf\x05U\x03\x03s\xa5\x05W\x05Y\x03\x03\x07\xd3\x03\x11{\xd5}\xd7\x7f\xd9\x81\x9f\x83\xdb\x85\xdd\x87\xdf\x89\xe3\x05[\x05]\x05_\x05a\x05c\x05e\x05g\x05i\x03\x03\r\xe5\x03\x03\r\xe7\x03\x03\r\xe9\x03\x03\r\xeb\x03\x03\r\xed\x03\x05'\xa7)\xef\x03\x03\x13\xf3\x03\x03\x13\xf5\x1f'\x01\x1f+\x11\x00\x00\x00\x00\x00\x00\x00\x00\x03\x01\x1dk\x03\x03\xbb\x1dm\t\x07\x1f)!\x00\x00\x00\x00\x00\x00\x00\x00\x01\x00\x00\x00\x00\x00\x00\x00#\x1d\x03\x05\xaf\xb3\r\x03\xa1\xb1\x1do\r\x03\xa1\xb5\x1dq\x1ds\x1du\r\x01#\x1f\x1dw\x13\r\x01\x1f\x03\t\x00\x00\x00\x00\x1f!\x01\x13\r\x05\x07\x05\x1f\x07\t\x00\x00\x00\x00\x1f\x17!\x01\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x1f\x07\t\x00\x00\x00@\x1f\x03\t\x01\x00\x00\x00\x1f\x03\t\x08\x00\x00\x00\x0b\x05\x1dy\x1d{\x05\x01\x03\t\x9b\x9b\x9b\xa9\x03\x03\xe1\x15\x03\x01\r\x01\x03\x0b\xa9\x9d\x9b\x9d\x9d\x13\x05\x01\x13\x05\x05\x13\x05\t\x13\x05\r\x13\x05\x11\x07\x01\x1f\x07\t\x00\x00\xc0\x7f\x1f\x17!\x00\x00\x00\x00\x00\x00\x00\x00\x01\x00\x00\x00\x00\x00\x00\x00\x1f5\x11\x00\x00\x00\x00\x00\x00\x00\x00)\x05!!\t)\x01\x05\x1b)\x01\t\t)\x03!\t\x1d\x01)\x05!!\x05\x13)\x05!!\x0f)\x03\t\r)\x03\x8a\x05\t)\x03\xad\x05\x11\x01\x05\x01\x0b\x11\x03\x01\x03\x01)\x03\x01\r)\x03\x02\x02\t/\x0b\x01\x0b\x03\x19\x1b)\x03\x01\x13)\x03\t\x13)\x03\x05\x13)\x01\x0f)\x05\x05\x05\x0f)\x03\x05\x0f)\x03!\x0f)\x03\x05\r\x04:\x05\x05\x01\x11\t3\x07\x03\x01\t\r\x11\t7\x05\x03=}\t\x03Y\x1f\x03#\x15\x06]\x03\x01\x03\x01\x17\x07ea\x03\x01\x03\x03\x0f\x06i\x03\x01\x05\x03\x05\x05\x03\tm\x03\x07\x03\x07-\x05\x03\x01\x03\t\x19\x06-\x03\x01\x05\x07\x0b\x1b\x07\x0bq\x03\x01\x03\r\x05\x03\x01/\x03\x03\x05\x03\x01/\x03\x03\x05\x03\x01w\x03\x03\x1d\x07\x01y\x03%\t\x11\x13\x15\x0f\x07\x07\x01\x8b\x03\x01\x03\x17\x07\x07\x01\x8d\x03\x0b\x03\x17\x07\x07\x01\x8f\x03\x03\x03\x17\x07\x07\x01\x91\x03\x19\x03\x17\x07\x07\x01\x93\x03\x1b\x03\x17\x05\x03\x01#\x03\x03\x03\x07\x01\x05\x03\x03\x03#\x11\x07\x01\x95\x03-\x05\x1d%\x03\x07\x01\x05\x03/\x03'\x05\x03\x011\x03\x07\x03\x07\x01\x05\x03\x01\x03+\x03\x07\x01\x97\x03\x15\x03)\x0b\x06\x01\x03\x01\x07/\x19-\x03\x07\x01\x05\x031\x03'\x05\x03\x011\x03\x07\x03\x07\x01\x05\x03\x0b\x035\x03\x07\x01\x99\x033\x033\x0b\x06\x01\x03\x0b\x079\x1b7\x13\x04\t\x051;\r\x11\x0b9\x05\x03\x15+\x03\x01\t\t\x03=\x1f\x03\x11\x05\x03\x0b#\x03\x03\x03\x07%\x05\x03\x11\x03\x05\x0f\x06%\x03\x11\x05\x03\x07\t\x03EC\x03\x11\x11\x07KI\x03\x15\x05\t\x0b\x05\x03\x0bO\x03\x07\x03\x07Q\x05\x03\x01\x03\x0f\x0b\x06U\x03\x01\x07\r\x01\x11\x13\x04\x0b\x03\x13\x06\x03\x01\x05\x01\x00\xb2\x19}\x1d\x03\x11\x0f\x0b\t\t\x0b!\x1f/!!)#\x1f\x19\x7f\x0f99m\x19\x85\x89W\xb3K\x9bM\x9b\x96\x04\x1b+\x1b\x1f\x1f\x15\x1d\x15+\x83\x13\r\r\x1f\x11\x15\x1b\x17\x15\x17\x0f\x11\x15\x11+\x19)\x0f\x0b\x11builtin\x00vhlo\x00module\x00broadcast_in_dim_v1\x00constant_v1\x00get_tuple_element_v1\x00iota_v1\x00select_v1\x00func_v1\x00add_v1\x00compare_v1\x00return_v1\x00reshape_v1\x00transpose_v1\x00divide_v1\x00call_v1\x00custom_call_v1\x00value\x00index\x00sym_name\x00third_party/py/jax/experimental/jax2tf/tests/back_compat_test.py\x00broadcast_dimensions\x00arg_attrs\x00function_type\x00res_attrs\x00sym_visibility\x00iota_dimension\x00compare_type\x00comparison_direction\x00jit__lambda_\x00jit(<lambda>)/jit(main)/pjit[in_shardings=(UnspecifiedValue,) out_shardings=(UnspecifiedValue,) resource_env=None donated_invars=(False,) name=tril in_positional_semantics=(<_PositionalSemantics.GLOBAL: 1>,) out_positional_semantics=_PositionalSemantics.GLOBAL keep_unused=False inline=False]\x00jit(<lambda>)/jit(main)/jit(tril)/iota[dtype=int32 shape=(8, 8) dimension=0]\x00jit(<lambda>)/jit(main)/jit(tril)/add\x00jit(<lambda>)/jit(main)/jit(tril)/iota[dtype=int32 shape=(8, 8) dimension=1]\x00jit(<lambda>)/jit(main)/jit(tril)/ge\x00jit(<lambda>)/jit(main)/jit(tril)/broadcast_in_dim[shape=(8, 8) broadcast_dimensions=()]\x00jit(<lambda>)/jit(main)/jit(tril)/select_n\x00jit(<lambda>)/jit(main)/iota[dtype=float32 shape=(64,) dimension=0]\x00jit(<lambda>)/jit(main)/reshape[new_sizes=(8, 8) dimensions=None]\x00permutation\x00jit(<lambda>)/jit(main)/transpose[permutation=(1, 0)]\x00jit(<lambda>)/jit(main)/add\x00jit(<lambda>)/jit(main)/div\x00callee\x00jit(<lambda>)/jit(main)/eigh[lower=True sort_eigenvalues=True]\x00api_version\x00backend_config\x00call_target_name\x00called_computations\x00has_side_effect\x00operand_layouts\x00output_operand_aliases\x00result_layouts\x00jax.result_info\x00tril\x00[0]\x00[1]\x00main\x00public\x00private\x00\x00lapack_ssyevd\x00",
        xla_call_module_version=4,
    ),  # End paste

    # Pasted from the test output (see back_compat_test.py module docstring)
    f64=dict(
        testdata_version=1,
        platform='cpu',
        custom_call_targets=['lapack_dsyevd'],
        serialized_date=datetime.date(2023, 3, 17),
        inputs=(),
        expected_outputs=(array([[-6.1857700048412056e-01,  2.4081403770912022e-01,
            3.5662489253627483e-01, -6.3034019033669797e-01,
            1.0043483479985752e-16, -2.8842036081919542e-02,
            7.7164692943283169e-25, -1.8446994643771725e-01],
          [-4.7070881487314614e-01,  4.7473787464450845e-01,
            -4.8036836210243367e-01,  4.3802686872516400e-01,
            1.7961797619639258e-01,  8.3080980076741355e-03,
            2.1415294457221756e-01, -2.2856669794666584e-01],
          [-3.2284062926217072e-01, -5.4336490915553370e-01,
            2.2181041859724990e-01,  2.9947877954402297e-01,
            -3.6491813600134632e-01,  3.2867679819727436e-01,
            3.8223299448843473e-01, -2.7266344945561438e-01],
          [-1.7497244365119530e-01, -8.9251550609769414e-02,
            -6.3518515114898394e-02,  1.9162997359209971e-01,
            -2.2087281326110139e-01,  5.9957027043505064e-02,
            -8.7632498908241274e-01, -3.1676020096456303e-01],
          [-2.7104258040220038e-02, -3.3772873786627672e-01,
            2.5901386593721748e-01,  1.7032650752287815e-01,
            6.7521217612940332e-01, -4.5036136532965476e-01,
            -1.2279030059078447e-02, -3.6085695247351163e-01],
          [ 1.2076392757075530e-01, -3.3834734096469254e-01,
            -6.5506827461665540e-01, -5.0472498521116749e-01,
            6.9987430903492118e-02,  1.0595648906599275e-01,
            8.3443844143082022e-02, -4.0495370398246017e-01],
          [ 2.6863211318173097e-01,  2.2958613191407318e-01,
            6.3952843755683941e-02,  1.8776775771084137e-02,
            -5.3523731432241317e-01, -5.9199531677602002e-01,
            1.7916671834524248e-01, -4.4905045549140887e-01],
          [ 4.1650029879270661e-01,  3.6355449432857079e-01,
            2.9755313100756142e-01,  1.6826270392615944e-02,
            1.9621068035557282e-01,  5.6830030587314817e-01,
            2.9607517592514246e-02, -4.9314720700035747e-01]]), array([-2.4598804776133626e+01, -4.6567755957874661e-14,
          -1.9932120610662194e-14, -5.7323356091157378e-15,
          -4.5459724251334835e-16,  4.0479851042511616e-14,
            9.2325194924982089e-14,  2.7659880477613365e+02])),
        mlir_module_text=r"""
module @jit__lambda_ {
  func.func public @main() -> (tensor<8x8xf64> {jax.result_info = "[0]"}, tensor<8xf64> {jax.result_info = "[1]"}) {
    %0 = stablehlo.iota dim = 0 : tensor<64xf64>
    %1 = stablehlo.reshape %0 : (tensor<64xf64>) -> tensor<8x8xf64>
    %2 = stablehlo.transpose %1, dims = [1, 0] : (tensor<8x8xf64>) -> tensor<8x8xf64>
    %3 = stablehlo.add %1, %2 : tensor<8x8xf64>
    %4 = stablehlo.constant dense<2.000000e+00> : tensor<f64>
    %5 = stablehlo.broadcast_in_dim %4, dims = [] : (tensor<f64>) -> tensor<8x8xf64>
    %6 = stablehlo.divide %3, %5 : tensor<8x8xf64>
    %7 = call @tril(%6) : (tensor<8x8xf64>) -> tensor<8x8xf64>
    %8 = stablehlo.constant dense<1> : tensor<i32>
    %9 = stablehlo.constant dense<1> : tensor<i32>
    %10 = stablehlo.constant dense<8> : tensor<i32>
    %11 = stablehlo.custom_call @lapack_dsyevd(%8, %9, %10, %7) {api_version = 2 : i32, operand_layouts = [dense<> : tensor<0xindex>, dense<> : tensor<0xindex>, dense<> : tensor<0xindex>, dense<[0, 1]> : tensor<2xindex>], output_operand_aliases = [#stablehlo.output_operand_alias<output_tuple_indices = [0], operand_index = 3, operand_tuple_indices = []>], result_layouts = [dense<[0, 1]> : tensor<2xindex>, dense<0> : tensor<1xindex>, dense<> : tensor<0xindex>, dense<0> : tensor<1xindex>, dense<0> : tensor<1xindex>]} : (tensor<i32>, tensor<i32>, tensor<i32>, tensor<8x8xf64>) -> tuple<tensor<8x8xf64>, tensor<8xf64>, tensor<i32>, tensor<177xf64>, tensor<43xi32>>
    %12 = stablehlo.get_tuple_element %11[0] : (tuple<tensor<8x8xf64>, tensor<8xf64>, tensor<i32>, tensor<177xf64>, tensor<43xi32>>) -> tensor<8x8xf64>
    %13 = stablehlo.get_tuple_element %11[1] : (tuple<tensor<8x8xf64>, tensor<8xf64>, tensor<i32>, tensor<177xf64>, tensor<43xi32>>) -> tensor<8xf64>
    %14 = stablehlo.get_tuple_element %11[2] : (tuple<tensor<8x8xf64>, tensor<8xf64>, tensor<i32>, tensor<177xf64>, tensor<43xi32>>) -> tensor<i32>
    %15 = stablehlo.get_tuple_element %11[3] : (tuple<tensor<8x8xf64>, tensor<8xf64>, tensor<i32>, tensor<177xf64>, tensor<43xi32>>) -> tensor<177xf64>
    %16 = stablehlo.get_tuple_element %11[4] : (tuple<tensor<8x8xf64>, tensor<8xf64>, tensor<i32>, tensor<177xf64>, tensor<43xi32>>) -> tensor<43xi32>
    %17 = stablehlo.constant dense<0> : tensor<i32>
    %18 = stablehlo.broadcast_in_dim %17, dims = [] : (tensor<i32>) -> tensor<i32>
    %19 = stablehlo.compare  EQ, %14, %18,  SIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1>
    %20 = stablehlo.broadcast_in_dim %19, dims = [] : (tensor<i1>) -> tensor<1x1xi1>
    %21 = stablehlo.constant dense<0x7FF8000000000000> : tensor<f64>
    %22 = stablehlo.broadcast_in_dim %21, dims = [] : (tensor<f64>) -> tensor<8x8xf64>
    %23 = stablehlo.broadcast_in_dim %20, dims = [0, 1] : (tensor<1x1xi1>) -> tensor<8x8xi1>
    %24 = stablehlo.select %23, %12, %22 : tensor<8x8xi1>, tensor<8x8xf64>
    %25 = stablehlo.broadcast_in_dim %19, dims = [] : (tensor<i1>) -> tensor<1xi1>
    %26 = stablehlo.constant dense<0x7FF8000000000000> : tensor<f64>
    %27 = stablehlo.broadcast_in_dim %26, dims = [] : (tensor<f64>) -> tensor<8xf64>
    %28 = stablehlo.broadcast_in_dim %25, dims = [0] : (tensor<1xi1>) -> tensor<8xi1>
    %29 = stablehlo.select %28, %13, %27 : tensor<8xi1>, tensor<8xf64>
    return %24, %29 : tensor<8x8xf64>, tensor<8xf64>
  }
  func.func private @tril(%arg0: tensor<8x8xf64>) -> tensor<8x8xf64> {
    %0 = stablehlo.iota dim = 0 : tensor<8x8xi32>
    %1 = stablehlo.constant dense<0> : tensor<i32>
    %2 = stablehlo.broadcast_in_dim %1, dims = [] : (tensor<i32>) -> tensor<8x8xi32>
    %3 = stablehlo.add %0, %2 : tensor<8x8xi32>
    %4 = stablehlo.iota dim = 1 : tensor<8x8xi32>
    %5 = stablehlo.compare  GE, %3, %4,  SIGNED : (tensor<8x8xi32>, tensor<8x8xi32>) -> tensor<8x8xi1>
    %6 = stablehlo.constant dense<0.000000e+00> : tensor<f64>
    %7 = stablehlo.broadcast_in_dim %6, dims = [] : (tensor<f64>) -> tensor<8x8xf64>
    %8 = stablehlo.select %5, %arg0, %7 : tensor<8x8xi1>, tensor<8x8xf64>
    return %8 : tensor<8x8xf64>
  }
}
""",
        mlir_module_serialized=b"ML\xefR\x03MLIRxxx-trunk\x00\x01-\x05\x01\x05\x01\x03\x05\x03\x1d\x07\t\x0b\r\x0f\x11\x13\x15\x17\x19\x1b\x1d\x1f!\x03z\x02\xf77\x01\x9b\x0f\x17\x13\x0b\x07\x0f\x0b\x0b\x0b\x0b\x17\x0b\x0b\x0b\x0b\x13\x0b\x13\x0f\x0b\x0b\x17\x0f\x13\x13\x13\x0b33\x0b\x0f\x0b\x0b\x13\x0f\x0b\x1b\x0f\x0b\x13\x0f\x0b\x0f\x0b\x0f\x0b\x0f\x0b\x13\x0b\x0f\x0b\x0f\x0b\x13\x0b\x13\x0b\x0b\x13K\x0b\x0b\x0b\x0b\x0b\x0b\x0b\x0b\x13\x13\x13\x13\x13\x1b\x13\x13\x03]\x0f/\x0b\x0b\x0f\x0b\x0bO\x0b\x13\x13\x0b\x13\x0b\x0b\x0b\x0b\x0b\x0b\x0f\x1f\x0f\x0f\x0b/O/\x1f\x1f\x0b\x0b\x0b\x0b\x1b\x0f\x17\x1f\x0f\x0f\x0f\x0f\x0f\x0b/O/\x037\x17\x0f\x07\x0f\x07\x13\x07\x07\x17\x07\x17\x13\x17\x13\x17\x17\x13\x17\x1f\x13\x13\x13\x0f\x17\x13\x13\x13\x02:\t\x1du\x03\x17\x11\xf6\x04\x01\x03\x03\x13\xc5\x05#\x1f\x1d;\x03\x05%\x05'\x05)\x05+\x17\x11\xf2\x04\x01\x05-\x05/\x051\x053\x03\x03!\xc1\x055\x03\x03\x07\xc3\x1dA\x03\x057\x059\x17\x11\xea\x04\x01\x1do\x15\x03\x03\x07\xd1\x03\x03\x07\xf1\x03\x03\x0f5\x05;\x03\x0b\x17\x9f\x19\xab\x1b\xad\x0f\xb7\x1d\xb9\x03\x0b\x17\xa3\x19\xbd\x1b\xa3\x0f\xa5\x1d\xbf\x05=\x1d?\x03\x05?\x05A\x03\x03!\xc7\x1dG\x03\x05C\x03\x05'\xa7)\xc9\x1dM\x03\x05E\x03\x03\x07\xcb\x1dS\x03\x05G\x1dW\x03\x05I\x1d[+\x05K\x1d_+\x05M\x03\x03c\xcd\x05O\x1dg\x15\x05Q\x1dk\x15\x05S\x03\x03\x07\xcf\x05U\x03\x03s\xa5\x05W\x05Y\x03\x03\x07\xd3\x03\x11{\xd5}\xd7\x7f\xd9\x81\x9f\x83\xdb\x85\xdd\x87\xdf\x89\xe3\x05[\x05]\x05_\x05a\x05c\x05e\x05g\x05i\x03\x03\r\xe5\x03\x03\r\xe7\x03\x03\r\xe9\x03\x03\r\xeb\x03\x03\r\xed\x03\x05'\xa7)\xef\x03\x03\x13\xf3\x03\x03\x13\xf5\x1f'\x01\x1f+\x11\x00\x00\x00\x00\x00\x00\x00\x00\x03\x01\x1dk\x03\x03\xbb\x1dm\t\x07\x1f)!\x00\x00\x00\x00\x00\x00\x00\x00\x01\x00\x00\x00\x00\x00\x00\x00#\x1d\x03\x05\xaf\xb3\r\x03\xa1\xb1\x1do\r\x03\xa1\xb5\x1dq\x1ds\x1du\r\x01#\x1f\x1dw\x13\r\x01\x1f\x03\t\x00\x00\x00\x00\x1f!\x01\x13\r\x05\x07\x05\x1f\x07\x11\x00\x00\x00\x00\x00\x00\x00\x00\x1f\x17!\x01\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x1f\x07\x11\x00\x00\x00\x00\x00\x00\x00@\x1f\x03\t\x01\x00\x00\x00\x1f\x03\t\x08\x00\x00\x00\x0b\x05\x1dy\x1d{\x05\x01\x03\t\x9b\x9b\x9b\xa9\x03\x03\xe1\x15\x03\x01\r\x01\x03\x0b\xa9\x9d\x9b\x9d\x9d\x13\x05\x01\x13\x05\x05\x13\x05\t\x13\x05\r\x13\x05\x11\x07\x01\x1f\x07\x11\x00\x00\x00\x00\x00\x00\xf8\x7f\x1f\x17!\x00\x00\x00\x00\x00\x00\x00\x00\x01\x00\x00\x00\x00\x00\x00\x00\x1f5\x11\x00\x00\x00\x00\x00\x00\x00\x00)\x05!!\t)\x01\x05\x1b)\x01\t\x0b)\x03!\t\x1d\x01)\x05!!\x05\x13)\x05!!\x0f)\x03\t\r)\x03\x8a\x05\t)\x03\xad\x05\x11\x01\x05\x01\x0b\x11\x03\x01\x03\x01)\x03\x01\r)\x03\x02\x02\t/\x0b\x01\x0b\x03\x19\x1b)\x03\x01\x13)\x03\t\x13)\x03\x05\x13)\x01\x0f)\x05\x05\x05\x0f)\x03\x05\x0f)\x03!\x0f)\x03\x05\r\x04:\x05\x05\x01\x11\t3\x07\x03\x01\t\r\x11\t7\x05\x03=}\t\x03Y\x1f\x03#\x15\x06]\x03\x01\x03\x01\x17\x07ea\x03\x01\x03\x03\x0f\x06i\x03\x01\x05\x03\x05\x05\x03\tm\x03\x07\x03\x07-\x05\x03\x01\x03\t\x19\x06-\x03\x01\x05\x07\x0b\x1b\x07\x0bq\x03\x01\x03\r\x05\x03\x01/\x03\x03\x05\x03\x01/\x03\x03\x05\x03\x01w\x03\x03\x1d\x07\x01y\x03%\t\x11\x13\x15\x0f\x07\x07\x01\x8b\x03\x01\x03\x17\x07\x07\x01\x8d\x03\x0b\x03\x17\x07\x07\x01\x8f\x03\x03\x03\x17\x07\x07\x01\x91\x03\x19\x03\x17\x07\x07\x01\x93\x03\x1b\x03\x17\x05\x03\x01#\x03\x03\x03\x07\x01\x05\x03\x03\x03#\x11\x07\x01\x95\x03-\x05\x1d%\x03\x07\x01\x05\x03/\x03'\x05\x03\x011\x03\x07\x03\x07\x01\x05\x03\x01\x03+\x03\x07\x01\x97\x03\x15\x03)\x0b\x06\x01\x03\x01\x07/\x19-\x03\x07\x01\x05\x031\x03'\x05\x03\x011\x03\x07\x03\x07\x01\x05\x03\x0b\x035\x03\x07\x01\x99\x033\x033\x0b\x06\x01\x03\x0b\x079\x1b7\x13\x04\t\x051;\r\x11\x0b9\x05\x03\x15+\x03\x01\t\t\x03=\x1f\x03\x11\x05\x03\x0b#\x03\x03\x03\x07%\x05\x03\x11\x03\x05\x0f\x06%\x03\x11\x05\x03\x07\t\x03EC\x03\x11\x11\x07KI\x03\x15\x05\t\x0b\x05\x03\x0bO\x03\x07\x03\x07Q\x05\x03\x01\x03\x0f\x0b\x06U\x03\x01\x07\r\x01\x11\x13\x04\x0b\x03\x13\x06\x03\x01\x05\x01\x00\xb2\x19}\x1d\x03\x11\x0f\x0b\t\t\x0b!\x1f/!!)#\x1f\x19\x7f\x0f99m\x19\x85\x89W\xb3K\x9bM\x9b\x96\x04\x1b+\x1b\x1f\x1f\x15\x1d\x15+\x83\x13\r\r\x1f\x11\x15\x1b\x17\x15\x17\x0f\x11\x15\x11+\x19)\x0f\x0b\x11builtin\x00vhlo\x00module\x00broadcast_in_dim_v1\x00constant_v1\x00get_tuple_element_v1\x00iota_v1\x00select_v1\x00func_v1\x00add_v1\x00compare_v1\x00return_v1\x00reshape_v1\x00transpose_v1\x00divide_v1\x00call_v1\x00custom_call_v1\x00value\x00index\x00sym_name\x00third_party/py/jax/experimental/jax2tf/tests/back_compat_test.py\x00broadcast_dimensions\x00arg_attrs\x00function_type\x00res_attrs\x00sym_visibility\x00iota_dimension\x00compare_type\x00comparison_direction\x00jit__lambda_\x00jit(<lambda>)/jit(main)/pjit[in_shardings=(UnspecifiedValue,) out_shardings=(UnspecifiedValue,) resource_env=None donated_invars=(False,) name=tril in_positional_semantics=(<_PositionalSemantics.GLOBAL: 1>,) out_positional_semantics=_PositionalSemantics.GLOBAL keep_unused=False inline=False]\x00jit(<lambda>)/jit(main)/jit(tril)/iota[dtype=int32 shape=(8, 8) dimension=0]\x00jit(<lambda>)/jit(main)/jit(tril)/add\x00jit(<lambda>)/jit(main)/jit(tril)/iota[dtype=int32 shape=(8, 8) dimension=1]\x00jit(<lambda>)/jit(main)/jit(tril)/ge\x00jit(<lambda>)/jit(main)/jit(tril)/broadcast_in_dim[shape=(8, 8) broadcast_dimensions=()]\x00jit(<lambda>)/jit(main)/jit(tril)/select_n\x00jit(<lambda>)/jit(main)/iota[dtype=float64 shape=(64,) dimension=0]\x00jit(<lambda>)/jit(main)/reshape[new_sizes=(8, 8) dimensions=None]\x00permutation\x00jit(<lambda>)/jit(main)/transpose[permutation=(1, 0)]\x00jit(<lambda>)/jit(main)/add\x00jit(<lambda>)/jit(main)/div\x00callee\x00jit(<lambda>)/jit(main)/eigh[lower=True sort_eigenvalues=True]\x00api_version\x00backend_config\x00call_target_name\x00called_computations\x00has_side_effect\x00operand_layouts\x00output_operand_aliases\x00result_layouts\x00jax.result_info\x00tril\x00[0]\x00[1]\x00main\x00public\x00private\x00\x00lapack_dsyevd\x00",
        xla_call_module_version=4,
    ),  # End paste

    # Pasted from the test output (see back_compat_test.py module docstring)
    c64=dict(
        testdata_version=1,
        platform='cpu',
        custom_call_targets=['lapack_cheevd'],
        serialized_date=datetime.date(2023, 3, 17),
        inputs=(),
        expected_outputs=(array([[-0.6185769    +0.j, -0.20142993   +0.j, -0.09725195   +0.j,
            0.62983674   +0.j, -0.07926044   +0.j,  0.3605001    -0.j,
            -0.019093221  +0.j, -0.18446997   +0.j],
          [-0.47070873   +0.j,  0.29325768   +0.j, -0.19454116   +0.j,
            -0.6394365    +0.j,  0.06229549   +0.j,  0.33249345   +0.j,
            0.28112718   +0.j, -0.22856665   +0.j],
          [-0.32284075   +0.j, -0.12361939   +0.j,  0.20547704   +0.j,
            -0.18307868   +0.j,  0.47294614   +0.j, -0.3170349    +0.j,
            -0.6373532    +0.j, -0.27266347   +0.j],
          [-0.17497246   +0.j, -0.079641335  +0.j,  0.15042792   +0.j,
            -0.15416273   +0.j, -0.815209     +0.j, -0.38054234   +0.j,
            -0.083263926  +0.j, -0.31676024   +0.j],
          [-0.027104257  +0.j, -0.26490977   +0.j,  0.32271704   +0.j,
            0.08653544   +0.j,  0.30305928   +0.j, -0.33998996   +0.j,
            0.6926741    +0.j, -0.360857     +0.j],
          [ 0.120763965  +0.j,  0.43288827   +0.j, -0.64385164   +0.j,
            0.2652551    +0.j,  0.094823755  +0.j, -0.37435007   +0.j,
            0.00091664493+0.j, -0.40495378   +0.j],
          [ 0.26863196   +0.j,  0.51607686   +0.j,  0.53846526   +0.j,
            0.16969058   +0.j, -0.0216703    +0.j,  0.35755336   +0.j,
            -0.113144726  +0.j, -0.4490505    +0.j],
          [ 0.4165004    +0.j, -0.57262254   +0.j, -0.28144246   +0.j,
            -0.17463988   +0.j, -0.016984984  +0.j,  0.3613705    +0.j,
            -0.12186296   +0.j, -0.49314725   +0.j]], dtype=complex64), array([-2.4598808e+01, -3.3105560e-05, -3.1002426e-05, -1.0103593e-05,
          -1.0022322e-05,  4.0141886e-06,  9.5510331e-06,  2.7659882e+02],
          dtype=float32)),
        mlir_module_text=r"""
module @jit__lambda_ {
  func.func public @main() -> (tensor<8x8xcomplex<f32>> {jax.result_info = "[0]"}, tensor<8xf32> {jax.result_info = "[1]"}) {
    %0 = stablehlo.iota dim = 0 : tensor<64xcomplex<f32>>
    %1 = stablehlo.reshape %0 : (tensor<64xcomplex<f32>>) -> tensor<8x8xcomplex<f32>>
    %2 = stablehlo.transpose %1, dims = [1, 0] : (tensor<8x8xcomplex<f32>>) -> tensor<8x8xcomplex<f32>>
    %3 = stablehlo.real %2 : (tensor<8x8xcomplex<f32>>) -> tensor<8x8xf32>
    %4 = stablehlo.imag %2 : (tensor<8x8xcomplex<f32>>) -> tensor<8x8xf32>
    %5 = stablehlo.negate %4 : tensor<8x8xf32>
    %6 = stablehlo.complex %3, %5 : tensor<8x8xcomplex<f32>>
    %7 = stablehlo.add %1, %6 : tensor<8x8xcomplex<f32>>
    %8 = stablehlo.constant dense<(2.000000e+00,0.000000e+00)> : tensor<complex<f32>>
    %9 = stablehlo.broadcast_in_dim %8, dims = [] : (tensor<complex<f32>>) -> tensor<8x8xcomplex<f32>>
    %10 = stablehlo.divide %7, %9 : tensor<8x8xcomplex<f32>>
    %11 = call @tril(%10) : (tensor<8x8xcomplex<f32>>) -> tensor<8x8xcomplex<f32>>
    %12 = stablehlo.constant dense<1> : tensor<i32>
    %13 = stablehlo.constant dense<1> : tensor<i32>
    %14 = stablehlo.constant dense<8> : tensor<i32>
    %15 = stablehlo.custom_call @lapack_cheevd(%12, %13, %14, %11) {api_version = 2 : i32, operand_layouts = [dense<> : tensor<0xindex>, dense<> : tensor<0xindex>, dense<> : tensor<0xindex>, dense<[0, 1]> : tensor<2xindex>], output_operand_aliases = [#stablehlo.output_operand_alias<output_tuple_indices = [0], operand_index = 3, operand_tuple_indices = []>], result_layouts = [dense<[0, 1]> : tensor<2xindex>, dense<0> : tensor<1xindex>, dense<> : tensor<0xindex>, dense<0> : tensor<1xindex>, dense<0> : tensor<1xindex>, dense<0> : tensor<1xindex>]} : (tensor<i32>, tensor<i32>, tensor<i32>, tensor<8x8xcomplex<f32>>) -> tuple<tensor<8x8xcomplex<f32>>, tensor<8xf32>, tensor<i32>, tensor<81xcomplex<f32>>, tensor<169xf32>, tensor<43xi32>>
    %16 = stablehlo.get_tuple_element %15[0] : (tuple<tensor<8x8xcomplex<f32>>, tensor<8xf32>, tensor<i32>, tensor<81xcomplex<f32>>, tensor<169xf32>, tensor<43xi32>>) -> tensor<8x8xcomplex<f32>>
    %17 = stablehlo.get_tuple_element %15[1] : (tuple<tensor<8x8xcomplex<f32>>, tensor<8xf32>, tensor<i32>, tensor<81xcomplex<f32>>, tensor<169xf32>, tensor<43xi32>>) -> tensor<8xf32>
    %18 = stablehlo.get_tuple_element %15[2] : (tuple<tensor<8x8xcomplex<f32>>, tensor<8xf32>, tensor<i32>, tensor<81xcomplex<f32>>, tensor<169xf32>, tensor<43xi32>>) -> tensor<i32>
    %19 = stablehlo.get_tuple_element %15[3] : (tuple<tensor<8x8xcomplex<f32>>, tensor<8xf32>, tensor<i32>, tensor<81xcomplex<f32>>, tensor<169xf32>, tensor<43xi32>>) -> tensor<81xcomplex<f32>>
    %20 = stablehlo.get_tuple_element %15[4] : (tuple<tensor<8x8xcomplex<f32>>, tensor<8xf32>, tensor<i32>, tensor<81xcomplex<f32>>, tensor<169xf32>, tensor<43xi32>>) -> tensor<169xf32>
    %21 = stablehlo.get_tuple_element %15[5] : (tuple<tensor<8x8xcomplex<f32>>, tensor<8xf32>, tensor<i32>, tensor<81xcomplex<f32>>, tensor<169xf32>, tensor<43xi32>>) -> tensor<43xi32>
    %22 = stablehlo.constant dense<0> : tensor<i32>
    %23 = stablehlo.broadcast_in_dim %22, dims = [] : (tensor<i32>) -> tensor<i32>
    %24 = stablehlo.compare  EQ, %18, %23,  SIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1>
    %25 = stablehlo.broadcast_in_dim %24, dims = [] : (tensor<i1>) -> tensor<1x1xi1>
    %26 = stablehlo.constant dense<(0x7FC00000,0x7FC00000)> : tensor<complex<f32>>
    %27 = stablehlo.broadcast_in_dim %26, dims = [] : (tensor<complex<f32>>) -> tensor<8x8xcomplex<f32>>
    %28 = stablehlo.broadcast_in_dim %25, dims = [0, 1] : (tensor<1x1xi1>) -> tensor<8x8xi1>
    %29 = stablehlo.select %28, %16, %27 : tensor<8x8xi1>, tensor<8x8xcomplex<f32>>
    %30 = stablehlo.broadcast_in_dim %24, dims = [] : (tensor<i1>) -> tensor<1xi1>
    %31 = stablehlo.constant dense<0x7FC00000> : tensor<f32>
    %32 = stablehlo.broadcast_in_dim %31, dims = [] : (tensor<f32>) -> tensor<8xf32>
    %33 = stablehlo.broadcast_in_dim %30, dims = [0] : (tensor<1xi1>) -> tensor<8xi1>
    %34 = stablehlo.select %33, %17, %32 : tensor<8xi1>, tensor<8xf32>
    return %29, %34 : tensor<8x8xcomplex<f32>>, tensor<8xf32>
  }
  func.func private @tril(%arg0: tensor<8x8xcomplex<f32>>) -> tensor<8x8xcomplex<f32>> {
    %0 = stablehlo.iota dim = 0 : tensor<8x8xi32>
    %1 = stablehlo.constant dense<0> : tensor<i32>
    %2 = stablehlo.broadcast_in_dim %1, dims = [] : (tensor<i32>) -> tensor<8x8xi32>
    %3 = stablehlo.add %0, %2 : tensor<8x8xi32>
    %4 = stablehlo.iota dim = 1 : tensor<8x8xi32>
    %5 = stablehlo.compare  GE, %3, %4,  SIGNED : (tensor<8x8xi32>, tensor<8x8xi32>) -> tensor<8x8xi1>
    %6 = stablehlo.constant dense<(0.000000e+00,0.000000e+00)> : tensor<complex<f32>>
    %7 = stablehlo.broadcast_in_dim %6, dims = [] : (tensor<complex<f32>>) -> tensor<8x8xcomplex<f32>>
    %8 = stablehlo.select %5, %arg0, %7 : tensor<8x8xi1>, tensor<8x8xcomplex<f32>>
    return %8 : tensor<8x8xcomplex<f32>>
  }
}
""",
        mlir_module_serialized=b"ML\xefR\x03MLIRxxx-trunk\x00\x015\x05\x01\x05\x01\x03\x05\x03%\x07\t\x0b\r\x0f\x11\x13\x15\x17\x19\x1b\x1d\x1f!#%')\x03\xc6\x02\x1e\x02?\x01\xa9\x0f\x17\x13\x0b\x17\x0b\x07\x0f\x0b\x0b\x0b\x0b\x0b\x0b\x0b\x13\x0b\x13\x0f\x0b\x0b\x17\x0f\x13\x13\x0b33\x0b\x0f\x0b\x0b\x13\x0f\x0b\x1b\x0f\x0b\x13\x0f\x0b\x0f\x0b\x0f\x0b\x0f\x0b\x13\x0b\x0f\x0b\x0f\x0b\x0f\x0b\x0f\x0b\x0f\x0b\x0f\x0b\x13\x0b\x13\x0b\x0b\x13K\x0b\x0b\x0b\x0b\x0b\x0b\x0b\x0b\x13\x13\x13\x13\x13\x13\x1b\x17\x03a\x0f/\x0b\x0b\x0f\x0b\x0bO\x0b\x13\x13\x0b\x13\x0b\x0b\x0b\x0b\x0b\x0b\x0f\x1f\x0f\x0f\x0b/O/\x1f\x1f\x0b\x0b\x0b\x0b\x1b\x0f\x17#\x0f\x0f\x0f\x0f\x0f\x0f\x0b/O\x1f/\x01\x07\x17\x17\x17\x03?\x17\x0f\x07\x0f\x07\x13\x07\x07\x0b\x17\x17\x07\x17\x13\x17\x17\x13\x0f\x17\x17\x13\x17#\x13\x13\x13\x0f\x17\x13\x13\x13\x02&\n\x1d\x83\x03\x17\x13\xf6\x04\x01\x03\x03\x15\xd3\x05+\x17\x13\xf2\x04\x01\x05-\x1f\x1d9\x03\x05/\x051\x053\x055\x057\x059\x05;\x03\x03!\xcf\x05=\x03\x03\x07\xd1\x1d?\x03\x05?\x05A\x17\x13\xea\x04\x01\x1d}\t\x03\x03\x07\xdf\x03\x03\x113\x05C\x03\x0b\x17\xad\x19\xb9\x1b\xbb\x11\xc5\x1d\xc7\x03\x0b\x17\xb1\x19\xcb\x1b\xb1\x11\xb3\x1d\xcd\x05E\x1d=\x03\x05G\x05I\x03\x03!\xd5\x1dE\x03\x05K\x03\x05'\xb5)\xd7\x1dK\x03\x05M\x03\x03\x07\xd9\x1dQ\x03\x05O\x1dU\x03\x05Q\x1dY+\x05S\x1d]+\x05U\x03\x03a\xdb\x05W\x1de\t\x05Y\x1di\t\x05[\x1dm\t\x05]\x1dq\t\x05_\x1du\t\x05a\x1dy\t\x05c\x03\x03\x07\xdd\x05e\x03\x03\x81\xb3\x05g\x05i\x03\x03\x07\xe1\x03\x11\x89\xe3\x8b\xe5\x8d\xe7\x8f\xad\x91\xe9\x93\xeb\x95\xed\x97\xf1\x05k\x05m\x05o\x05q\x05s\x05u\x05w\x05y\x03\x03\x0b\xf3\x03\x03\x0b\xf5\x03\x03\x0b\xf7\x03\x03\x0b\xf9\x03\x03\x0b\xfb\x03\x03\x0b\xfd\x03\x05'\xb5)\xff\x03\x03\x07\x02\x02\x1f/\x01\x1f3\x11\x00\x00\x00\x00\x00\x00\x00\x00\x03\x01\x1d{\x03\x03\xc9\x1d}\t\x07\x1f1!\x00\x00\x00\x00\x00\x00\x00\x00\x01\x00\x00\x00\x00\x00\x00\x00#%\x03\x05\xbd\xc1\r\x03\xaf\xbf\x1d\x7f\r\x03\xaf\xc3\x1d\x81\x1d\x83\x1d\x85\r\x01#'\x1d\x87\x13\r\x01\x1f\x03\t\x00\x00\x00\x00\x1f)\x01\x13\r\x05\x07\x05\x1f\x07\x11\x00\x00\x00\x00\x00\x00\x00\x00\x1f\x1b!\x01\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x1f\x07\x11\x00\x00\x00@\x00\x00\x00\x00\x1f\x03\t\x01\x00\x00\x00\x1f\x03\t\x08\x00\x00\x00\x0b\x05\x1d\x89\x1d\x8b\x05\x01\x03\t\xa9\xa9\xa9\xb7\x03\x03\xef\x15\x03\x01\r\x01\x03\r\xb7\xab\xa9\xab\xab\xab\x13\x05\x01\x13\x05\x05\x13\x05\t\x13\x05\r\x13\x05\x11\x13\x05\x15\x07\x01\x1f\x07\x11\x00\x00\xc0\x7f\x00\x00\xc0\x7f\x1f\x1b!\x00\x00\x00\x00\x00\x00\x00\x00\x01\x00\x00\x00\x00\x00\x00\x00\x1f#\t\x00\x00\xc0\x7f\x1f=\x11\x00\x00\x00\x00\x00\x00\x00\x00\x03\x03\x15\x06\x02\x03\x03\x07\n\x02\x03\x03\x15\x0e\x02)\x05!!\x11)\x01\x05\x1b)\x01\x11\t)\x03!\t\x1d\x01\x03\t)\x05!!\x05)\x05!!\t\x13)\x05!!\x0f)\x03\t\r)\x03\x8a\x02\x11)\x03J\x05\t)\x03\xad\x05)\x01\t\x11\x01\x05\x01\x0b\x11\x03\x01\x03\x01)\x03\x01\r)\x03\x02\x02\x11/\r\x01\x0b\x03\x1d\x1f!)\x03\x01\x17)\x03\t\x17)\x03\x05\x17)\x01\x0f)\x05\x05\x05\x0f)\x03\x05\x0f)\x03!\x0f)\x03\x05\r\x04\xda\x05\x05\x01\x11\r1\x07\x03\x01\t\r\x11\r5\x05\x03G\x91\t\x03W\x1f\x03+\x15\x06[\x03\x01\x03\x01\x17\x07c_\x03\x01\x03\x03\x19\x06g\x03\x15\x03\x05\x1b\x06k\x03\x15\x03\x05\x1d\x06o\x03\x15\x03\t\x1f\x06s\x03\x01\x05\x07\x0b\x0f\x06w\x03\x01\x05\x03\r\x05\x03\r{\x03\x07\x03\x07-\x05\x03\x01\x03\x11!\x06-\x03\x01\x05\x0f\x13#\x07\x0f\x7f\x03\x01\x03\x15\x05\x03\x01/\x03\x03\x05\x03\x01/\x03\x03\x05\x03\x01\x85\x03\x03%\x07\x01\x87\x03-\t\x19\x1b\x1d\x17\x07\x07\x01\x99\x03\x01\x03\x1f\x07\x07\x01\x9b\x03\x0b\x03\x1f\x07\x07\x01\x9d\x03\x03\x03\x1f\x07\x07\x01\x9f\x03\x1d\x03\x1f\x07\x07\x01\xa1\x03\x1f\x03\x1f\x07\x07\x01\xa3\x03!\x03\x1f\x05\x03\x01#\x03\x03\x03\x07\x01\x05\x03\x03\x03-\x11\x07\x01\xa5\x035\x05%/\x03\x07\x01\x05\x037\x031\x05\x03\x01\xa7\x03\x07\x03\x07\x01\x05\x03\x01\x035\x03\x07\x01\x12\x02\x03\x19\x033\x0b\x06\x01\x03\x01\x079!7\x03\x07\x01\x05\x039\x031\x05\x03\x01\x16\x02\x03#\x03\x07\x01\x05\x03\x0b\x03?\x03\x07\x01\x1a\x02\x03;\x03=\x0b\x06\x01\x03\x0b\x07C#A\x13\x04\r\x05;E\r\x11\x0f7\x05\x03\x15+\x03\x01\r\t\x03;\x1f\x03\x13\x05\x03\x0f#\x03\x03\x03\x07%\x05\x03\x13\x03\x05\x0f\x06%\x03\x13\x05\x03\x07\t\x03CA\x03\x13\x11\x07IG\x03\x19\x05\t\x0b\x05\x03\x0fM\x03\x07\x03\x07O\x05\x03\x01\x03\x0f\x0b\x06S\x03\x01\x07\r\x01\x11\x13\x04\x0f\x03\x13\x06\x03\x01\x05\x01\x00F\x1c\x8d\x1d\x03\x11\x0f\x0b\t\t\x0b!\x1f/!!)#\x1f\x19\x7f\x0f99A9;;m\x19\x85\x8dW\xb3K\x9bM\x9b\x96\x04\x1b+\x1b\x1f\x1f\x15\x1d\x15+\x83\x13\r\r\x1f\x11\x15\x17\x15\x11\x11\x1b\x17\x15\x17\x0f\x11\x15\x11+\x19)\x0f\x0b\x11builtin\x00vhlo\x00module\x00broadcast_in_dim_v1\x00constant_v1\x00get_tuple_element_v1\x00iota_v1\x00select_v1\x00func_v1\x00add_v1\x00compare_v1\x00return_v1\x00reshape_v1\x00transpose_v1\x00real_v1\x00imag_v1\x00negate_v1\x00complex_v1\x00divide_v1\x00call_v1\x00custom_call_v1\x00value\x00index\x00sym_name\x00third_party/py/jax/experimental/jax2tf/tests/back_compat_test.py\x00broadcast_dimensions\x00arg_attrs\x00function_type\x00res_attrs\x00sym_visibility\x00iota_dimension\x00compare_type\x00comparison_direction\x00jit__lambda_\x00jit(<lambda>)/jit(main)/pjit[in_shardings=(UnspecifiedValue,) out_shardings=(UnspecifiedValue,) resource_env=None donated_invars=(False,) name=tril in_positional_semantics=(<_PositionalSemantics.GLOBAL: 1>,) out_positional_semantics=_PositionalSemantics.GLOBAL keep_unused=False inline=False]\x00jit(<lambda>)/jit(main)/jit(tril)/iota[dtype=int32 shape=(8, 8) dimension=0]\x00jit(<lambda>)/jit(main)/jit(tril)/add\x00jit(<lambda>)/jit(main)/jit(tril)/iota[dtype=int32 shape=(8, 8) dimension=1]\x00jit(<lambda>)/jit(main)/jit(tril)/ge\x00jit(<lambda>)/jit(main)/jit(tril)/broadcast_in_dim[shape=(8, 8) broadcast_dimensions=()]\x00jit(<lambda>)/jit(main)/jit(tril)/select_n\x00jit(<lambda>)/jit(main)/iota[dtype=complex64 shape=(64,) dimension=0]\x00jit(<lambda>)/jit(main)/reshape[new_sizes=(8, 8) dimensions=None]\x00permutation\x00jit(<lambda>)/jit(main)/transpose[permutation=(1, 0)]\x00jit(<lambda>)/jit(main)/real\x00jit(<lambda>)/jit(main)/imag\x00jit(<lambda>)/jit(main)/neg\x00jit(<lambda>)/jit(main)/complex\x00jit(<lambda>)/jit(main)/add\x00jit(<lambda>)/jit(main)/div\x00callee\x00jit(<lambda>)/jit(main)/eigh[lower=True sort_eigenvalues=True]\x00api_version\x00backend_config\x00call_target_name\x00called_computations\x00has_side_effect\x00operand_layouts\x00output_operand_aliases\x00result_layouts\x00jax.result_info\x00tril\x00[0]\x00[1]\x00main\x00public\x00private\x00\x00lapack_cheevd\x00",
        xla_call_module_version=4,
    ),  # End paste

    # Pasted from the test output (see back_compat_test.py module docstring)
    c128=dict(
        testdata_version=1,
        platform='cpu',
        custom_call_targets=['lapack_zheevd'],
        serialized_date=datetime.date(2023, 3, 17),
        inputs=(),
        expected_outputs=(array([[-6.1857700048412056e-01+0.j,  2.4081403770912022e-01+0.j,
            3.5662489253627483e-01+0.j, -6.3034019033669797e-01+0.j,
            1.0043483479985752e-16+0.j, -2.8842036081919542e-02+0.j,
            7.7164692943283169e-25+0.j, -1.8446994643771725e-01+0.j],
          [-4.7070881487314609e-01+0.j,  4.7473787464450828e-01+0.j,
            -4.8036836210243361e-01+0.j,  4.3802686872516400e-01+0.j,
            1.7961797619639255e-01+0.j,  8.3080980076741355e-03+0.j,
            2.1415294457221759e-01+0.j, -2.2856669794666584e-01+0.j],
          [-3.2284062926217072e-01+0.j, -5.4336490915553370e-01+0.j,
            2.2181041859724987e-01+0.j,  2.9947877954402286e-01+0.j,
            -3.6491813600134637e-01+0.j,  3.2867679819727436e-01+0.j,
            3.8223299448843473e-01+0.j, -2.7266344945561438e-01+0.j],
          [-1.7497244365119527e-01+0.j, -8.9251550609769331e-02+0.j,
            -6.3518515114898352e-02+0.j,  1.9162997359209963e-01+0.j,
            -2.2087281326110142e-01+0.j,  5.9957027043505008e-02+0.j,
            -8.7632498908241274e-01+0.j, -3.1676020096456303e-01+0.j],
          [-2.7104258040220017e-02+0.j, -3.3772873786627688e-01+0.j,
            2.5901386593721754e-01+0.j,  1.7032650752287815e-01+0.j,
            6.7521217612940321e-01+0.j, -4.5036136532965476e-01+0.j,
            -1.2279030059078447e-02+0.j, -3.6085695247351163e-01+0.j],
          [ 1.2076392757075533e-01+0.j, -3.3834734096469249e-01+0.j,
            -6.5506827461665529e-01+0.j, -5.0472498521116760e-01+0.j,
            6.9987430903492132e-02+0.j,  1.0595648906599270e-01+0.j,
            8.3443844143082035e-02+0.j, -4.0495370398246017e-01+0.j],
          [ 2.6863211318173102e-01+0.j,  2.2958613191407312e-01+0.j,
            6.3952843755683969e-02+0.j,  1.8776775771084192e-02+0.j,
            -5.3523731432241317e-01+0.j, -5.9199531677602002e-01+0.j,
            1.7916671834524250e-01+0.j, -4.4905045549140887e-01+0.j],
          [ 4.1650029879270667e-01+0.j,  3.6355449432857068e-01+0.j,
            2.9755313100756148e-01+0.j,  1.6826270392616000e-02+0.j,
            1.9621068035557282e-01+0.j,  5.6830030587314817e-01+0.j,
            2.9607517592514260e-02+0.j, -4.9314720700035747e-01+0.j]]), array([-2.4598804776133626e+01, -4.6567755957874661e-14,
          -1.9932120610662194e-14, -5.7323356091157378e-15,
          -4.5459724251334835e-16,  4.0479851042511616e-14,
            9.2325194924982089e-14,  2.7659880477613365e+02])),
        mlir_module_text=r"""
module @jit__lambda_ {
  func.func public @main() -> (tensor<8x8xcomplex<f64>> {jax.result_info = "[0]"}, tensor<8xf64> {jax.result_info = "[1]"}) {
    %0 = stablehlo.iota dim = 0 : tensor<64xcomplex<f64>>
    %1 = stablehlo.reshape %0 : (tensor<64xcomplex<f64>>) -> tensor<8x8xcomplex<f64>>
    %2 = stablehlo.transpose %1, dims = [1, 0] : (tensor<8x8xcomplex<f64>>) -> tensor<8x8xcomplex<f64>>
    %3 = stablehlo.real %2 : (tensor<8x8xcomplex<f64>>) -> tensor<8x8xf64>
    %4 = stablehlo.imag %2 : (tensor<8x8xcomplex<f64>>) -> tensor<8x8xf64>
    %5 = stablehlo.negate %4 : tensor<8x8xf64>
    %6 = stablehlo.complex %3, %5 : tensor<8x8xcomplex<f64>>
    %7 = stablehlo.add %1, %6 : tensor<8x8xcomplex<f64>>
    %8 = stablehlo.constant dense<(2.000000e+00,0.000000e+00)> : tensor<complex<f64>>
    %9 = stablehlo.broadcast_in_dim %8, dims = [] : (tensor<complex<f64>>) -> tensor<8x8xcomplex<f64>>
    %10 = stablehlo.divide %7, %9 : tensor<8x8xcomplex<f64>>
    %11 = call @tril(%10) : (tensor<8x8xcomplex<f64>>) -> tensor<8x8xcomplex<f64>>
    %12 = stablehlo.constant dense<1> : tensor<i32>
    %13 = stablehlo.constant dense<1> : tensor<i32>
    %14 = stablehlo.constant dense<8> : tensor<i32>
    %15 = stablehlo.custom_call @lapack_zheevd(%12, %13, %14, %11) {api_version = 2 : i32, operand_layouts = [dense<> : tensor<0xindex>, dense<> : tensor<0xindex>, dense<> : tensor<0xindex>, dense<[0, 1]> : tensor<2xindex>], output_operand_aliases = [#stablehlo.output_operand_alias<output_tuple_indices = [0], operand_index = 3, operand_tuple_indices = []>], result_layouts = [dense<[0, 1]> : tensor<2xindex>, dense<0> : tensor<1xindex>, dense<> : tensor<0xindex>, dense<0> : tensor<1xindex>, dense<0> : tensor<1xindex>, dense<0> : tensor<1xindex>]} : (tensor<i32>, tensor<i32>, tensor<i32>, tensor<8x8xcomplex<f64>>) -> tuple<tensor<8x8xcomplex<f64>>, tensor<8xf64>, tensor<i32>, tensor<81xcomplex<f64>>, tensor<169xf64>, tensor<43xi32>>
    %16 = stablehlo.get_tuple_element %15[0] : (tuple<tensor<8x8xcomplex<f64>>, tensor<8xf64>, tensor<i32>, tensor<81xcomplex<f64>>, tensor<169xf64>, tensor<43xi32>>) -> tensor<8x8xcomplex<f64>>
    %17 = stablehlo.get_tuple_element %15[1] : (tuple<tensor<8x8xcomplex<f64>>, tensor<8xf64>, tensor<i32>, tensor<81xcomplex<f64>>, tensor<169xf64>, tensor<43xi32>>) -> tensor<8xf64>
    %18 = stablehlo.get_tuple_element %15[2] : (tuple<tensor<8x8xcomplex<f64>>, tensor<8xf64>, tensor<i32>, tensor<81xcomplex<f64>>, tensor<169xf64>, tensor<43xi32>>) -> tensor<i32>
    %19 = stablehlo.get_tuple_element %15[3] : (tuple<tensor<8x8xcomplex<f64>>, tensor<8xf64>, tensor<i32>, tensor<81xcomplex<f64>>, tensor<169xf64>, tensor<43xi32>>) -> tensor<81xcomplex<f64>>
    %20 = stablehlo.get_tuple_element %15[4] : (tuple<tensor<8x8xcomplex<f64>>, tensor<8xf64>, tensor<i32>, tensor<81xcomplex<f64>>, tensor<169xf64>, tensor<43xi32>>) -> tensor<169xf64>
    %21 = stablehlo.get_tuple_element %15[5] : (tuple<tensor<8x8xcomplex<f64>>, tensor<8xf64>, tensor<i32>, tensor<81xcomplex<f64>>, tensor<169xf64>, tensor<43xi32>>) -> tensor<43xi32>
    %22 = stablehlo.constant dense<0> : tensor<i32>
    %23 = stablehlo.broadcast_in_dim %22, dims = [] : (tensor<i32>) -> tensor<i32>
    %24 = stablehlo.compare  EQ, %18, %23,  SIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1>
    %25 = stablehlo.broadcast_in_dim %24, dims = [] : (tensor<i1>) -> tensor<1x1xi1>
    %26 = stablehlo.constant dense<(0x7FF8000000000000,0x7FF8000000000000)> : tensor<complex<f64>>
    %27 = stablehlo.broadcast_in_dim %26, dims = [] : (tensor<complex<f64>>) -> tensor<8x8xcomplex<f64>>
    %28 = stablehlo.broadcast_in_dim %25, dims = [0, 1] : (tensor<1x1xi1>) -> tensor<8x8xi1>
    %29 = stablehlo.select %28, %16, %27 : tensor<8x8xi1>, tensor<8x8xcomplex<f64>>
    %30 = stablehlo.broadcast_in_dim %24, dims = [] : (tensor<i1>) -> tensor<1xi1>
    %31 = stablehlo.constant dense<0x7FF8000000000000> : tensor<f64>
    %32 = stablehlo.broadcast_in_dim %31, dims = [] : (tensor<f64>) -> tensor<8xf64>
    %33 = stablehlo.broadcast_in_dim %30, dims = [0] : (tensor<1xi1>) -> tensor<8xi1>
    %34 = stablehlo.select %33, %17, %32 : tensor<8xi1>, tensor<8xf64>
    return %29, %34 : tensor<8x8xcomplex<f64>>, tensor<8xf64>
  }
  func.func private @tril(%arg0: tensor<8x8xcomplex<f64>>) -> tensor<8x8xcomplex<f64>> {
    %0 = stablehlo.iota dim = 0 : tensor<8x8xi32>
    %1 = stablehlo.constant dense<0> : tensor<i32>
    %2 = stablehlo.broadcast_in_dim %1, dims = [] : (tensor<i32>) -> tensor<8x8xi32>
    %3 = stablehlo.add %0, %2 : tensor<8x8xi32>
    %4 = stablehlo.iota dim = 1 : tensor<8x8xi32>
    %5 = stablehlo.compare  GE, %3, %4,  SIGNED : (tensor<8x8xi32>, tensor<8x8xi32>) -> tensor<8x8xi1>
    %6 = stablehlo.constant dense<(0.000000e+00,0.000000e+00)> : tensor<complex<f64>>
    %7 = stablehlo.broadcast_in_dim %6, dims = [] : (tensor<complex<f64>>) -> tensor<8x8xcomplex<f64>>
    %8 = stablehlo.select %5, %arg0, %7 : tensor<8x8xi1>, tensor<8x8xcomplex<f64>>
    return %8 : tensor<8x8xcomplex<f64>>
  }
}
""",
        mlir_module_serialized=b"ML\xefR\x03MLIRxxx-trunk\x00\x015\x05\x01\x05\x01\x03\x05\x03%\x07\t\x0b\r\x0f\x11\x13\x15\x17\x19\x1b\x1d\x1f!#%')\x03\xc6\x02\x1e\x02?\x01\xa9\x0f\x17\x13\x0b\x17\x0b\x07\x0f\x0b\x0b\x0b\x0b\x0b\x0b\x0b\x13\x0b\x13\x0f\x0b\x0b\x17\x0f\x13\x13\x0b33\x0b\x0f\x0b\x0b\x13\x0f\x0b\x1b\x0f\x0b\x13\x0f\x0b\x0f\x0b\x0f\x0b\x0f\x0b\x13\x0b\x0f\x0b\x0f\x0b\x0f\x0b\x0f\x0b\x0f\x0b\x0f\x0b\x13\x0b\x13\x0b\x0b\x13K\x0b\x0b\x0b\x0b\x0b\x0b\x0b\x0b\x13\x13\x13\x13\x13\x13\x1b\x17\x03a\x0f/\x0b\x0b\x0f\x0b\x0bO\x0b\x13\x13\x0b\x13\x0b\x0b\x0b\x0b\x0b\x0b\x0f\x1f\x0f\x0f\x0bOOO\x1f\x1f\x0b\x0b\x0b\x0b\x1b\x0f\x17#\x0f\x0f\x0f\x0f\x0f\x0f\x0bOO//\x01\x07\x17\x17\x17\x03?\x17\x0f\x07\x0f\x07\x13\x07\x07\x0b\x17\x17\x07\x17\x13\x17\x17\x13\x0f\x17\x17\x13\x17#\x13\x13\x13\x0f\x17\x13\x13\x13\x02\x96\n\x1d\x83\x03\x17\x13\xf6\x04\x01\x03\x03\x15\xd3\x05+\x17\x13\xf2\x04\x01\x05-\x1f\x1d9\x03\x05/\x051\x053\x055\x057\x059\x05;\x03\x03!\xcf\x05=\x03\x03\x07\xd1\x1d?\x03\x05?\x05A\x17\x13\xea\x04\x01\x1d}\t\x03\x03\x07\xdf\x03\x03\x113\x05C\x03\x0b\x17\xad\x19\xb9\x1b\xbb\x11\xc5\x1d\xc7\x03\x0b\x17\xb1\x19\xcb\x1b\xb1\x11\xb3\x1d\xcd\x05E\x1d=\x03\x05G\x05I\x03\x03!\xd5\x1dE\x03\x05K\x03\x05'\xb5)\xd7\x1dK\x03\x05M\x03\x03\x07\xd9\x1dQ\x03\x05O\x1dU\x03\x05Q\x1dY+\x05S\x1d]+\x05U\x03\x03a\xdb\x05W\x1de\t\x05Y\x1di\t\x05[\x1dm\t\x05]\x1dq\t\x05_\x1du\t\x05a\x1dy\t\x05c\x03\x03\x07\xdd\x05e\x03\x03\x81\xb3\x05g\x05i\x03\x03\x07\xe1\x03\x11\x89\xe3\x8b\xe5\x8d\xe7\x8f\xad\x91\xe9\x93\xeb\x95\xed\x97\xf1\x05k\x05m\x05o\x05q\x05s\x05u\x05w\x05y\x03\x03\x0b\xf3\x03\x03\x0b\xf5\x03\x03\x0b\xf7\x03\x03\x0b\xf9\x03\x03\x0b\xfb\x03\x03\x0b\xfd\x03\x05'\xb5)\xff\x03\x03\x07\x02\x02\x1f/\x01\x1f3\x11\x00\x00\x00\x00\x00\x00\x00\x00\x03\x01\x1d{\x03\x03\xc9\x1d}\t\x07\x1f1!\x00\x00\x00\x00\x00\x00\x00\x00\x01\x00\x00\x00\x00\x00\x00\x00#%\x03\x05\xbd\xc1\r\x03\xaf\xbf\x1d\x7f\r\x03\xaf\xc3\x1d\x81\x1d\x83\x1d\x85\r\x01#'\x1d\x87\x13\r\x01\x1f\x03\t\x00\x00\x00\x00\x1f)\x01\x13\r\x05\x07\x05\x1f\x07!\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x1f\x1b!\x01\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x1f\x07!\x00\x00\x00\x00\x00\x00\x00@\x00\x00\x00\x00\x00\x00\x00\x00\x1f\x03\t\x01\x00\x00\x00\x1f\x03\t\x08\x00\x00\x00\x0b\x05\x1d\x89\x1d\x8b\x05\x01\x03\t\xa9\xa9\xa9\xb7\x03\x03\xef\x15\x03\x01\r\x01\x03\r\xb7\xab\xa9\xab\xab\xab\x13\x05\x01\x13\x05\x05\x13\x05\t\x13\x05\r\x13\x05\x11\x13\x05\x15\x07\x01\x1f\x07!\x00\x00\x00\x00\x00\x00\xf8\x7f\x00\x00\x00\x00\x00\x00\xf8\x7f\x1f\x1b!\x00\x00\x00\x00\x00\x00\x00\x00\x01\x00\x00\x00\x00\x00\x00\x00\x1f#\x11\x00\x00\x00\x00\x00\x00\xf8\x7f\x1f=\x11\x00\x00\x00\x00\x00\x00\x00\x00\x03\x03\x15\x06\x02\x03\x03\x07\n\x02\x03\x03\x15\x0e\x02)\x05!!\x11)\x01\x05\x1b)\x01\x11\x0b)\x03!\t\x1d\x01\x03\t)\x05!!\x05)\x05!!\t\x13)\x05!!\x0f)\x03\t\r)\x03\x8a\x02\x11)\x03J\x05\t)\x03\xad\x05)\x01\t\x11\x01\x05\x01\x0b\x11\x03\x01\x03\x01)\x03\x01\r)\x03\x02\x02\x11/\r\x01\x0b\x03\x1d\x1f!)\x03\x01\x17)\x03\t\x17)\x03\x05\x17)\x01\x0f)\x05\x05\x05\x0f)\x03\x05\x0f)\x03!\x0f)\x03\x05\r\x04\xda\x05\x05\x01\x11\r1\x07\x03\x01\t\r\x11\r5\x05\x03G\x91\t\x03W\x1f\x03+\x15\x06[\x03\x01\x03\x01\x17\x07c_\x03\x01\x03\x03\x19\x06g\x03\x15\x03\x05\x1b\x06k\x03\x15\x03\x05\x1d\x06o\x03\x15\x03\t\x1f\x06s\x03\x01\x05\x07\x0b\x0f\x06w\x03\x01\x05\x03\r\x05\x03\r{\x03\x07\x03\x07-\x05\x03\x01\x03\x11!\x06-\x03\x01\x05\x0f\x13#\x07\x0f\x7f\x03\x01\x03\x15\x05\x03\x01/\x03\x03\x05\x03\x01/\x03\x03\x05\x03\x01\x85\x03\x03%\x07\x01\x87\x03-\t\x19\x1b\x1d\x17\x07\x07\x01\x99\x03\x01\x03\x1f\x07\x07\x01\x9b\x03\x0b\x03\x1f\x07\x07\x01\x9d\x03\x03\x03\x1f\x07\x07\x01\x9f\x03\x1d\x03\x1f\x07\x07\x01\xa1\x03\x1f\x03\x1f\x07\x07\x01\xa3\x03!\x03\x1f\x05\x03\x01#\x03\x03\x03\x07\x01\x05\x03\x03\x03-\x11\x07\x01\xa5\x035\x05%/\x03\x07\x01\x05\x037\x031\x05\x03\x01\xa7\x03\x07\x03\x07\x01\x05\x03\x01\x035\x03\x07\x01\x12\x02\x03\x19\x033\x0b\x06\x01\x03\x01\x079!7\x03\x07\x01\x05\x039\x031\x05\x03\x01\x16\x02\x03#\x03\x07\x01\x05\x03\x0b\x03?\x03\x07\x01\x1a\x02\x03;\x03=\x0b\x06\x01\x03\x0b\x07C#A\x13\x04\r\x05;E\r\x11\x0f7\x05\x03\x15+\x03\x01\r\t\x03;\x1f\x03\x13\x05\x03\x0f#\x03\x03\x03\x07%\x05\x03\x13\x03\x05\x0f\x06%\x03\x13\x05\x03\x07\t\x03CA\x03\x13\x11\x07IG\x03\x19\x05\t\x0b\x05\x03\x0fM\x03\x07\x03\x07O\x05\x03\x01\x03\x0f\x0b\x06S\x03\x01\x07\r\x01\x11\x13\x04\x0f\x03\x13\x06\x03\x01\x05\x01\x00J\x1c\x8d\x1d\x03\x11\x0f\x0b\t\t\x0b!\x1f/!!)#\x1f\x19\x7f\x0f99A9;;m\x19\x85\x8fW\xb3K\x9bM\x9b\x96\x04\x1b+\x1b\x1f\x1f\x15\x1d\x15+\x83\x13\r\r\x1f\x11\x15\x17\x15\x11\x11\x1b\x17\x15\x17\x0f\x11\x15\x11+\x19)\x0f\x0b\x11builtin\x00vhlo\x00module\x00broadcast_in_dim_v1\x00constant_v1\x00get_tuple_element_v1\x00iota_v1\x00select_v1\x00func_v1\x00add_v1\x00compare_v1\x00return_v1\x00reshape_v1\x00transpose_v1\x00real_v1\x00imag_v1\x00negate_v1\x00complex_v1\x00divide_v1\x00call_v1\x00custom_call_v1\x00value\x00index\x00sym_name\x00third_party/py/jax/experimental/jax2tf/tests/back_compat_test.py\x00broadcast_dimensions\x00arg_attrs\x00function_type\x00res_attrs\x00sym_visibility\x00iota_dimension\x00compare_type\x00comparison_direction\x00jit__lambda_\x00jit(<lambda>)/jit(main)/pjit[in_shardings=(UnspecifiedValue,) out_shardings=(UnspecifiedValue,) resource_env=None donated_invars=(False,) name=tril in_positional_semantics=(<_PositionalSemantics.GLOBAL: 1>,) out_positional_semantics=_PositionalSemantics.GLOBAL keep_unused=False inline=False]\x00jit(<lambda>)/jit(main)/jit(tril)/iota[dtype=int32 shape=(8, 8) dimension=0]\x00jit(<lambda>)/jit(main)/jit(tril)/add\x00jit(<lambda>)/jit(main)/jit(tril)/iota[dtype=int32 shape=(8, 8) dimension=1]\x00jit(<lambda>)/jit(main)/jit(tril)/ge\x00jit(<lambda>)/jit(main)/jit(tril)/broadcast_in_dim[shape=(8, 8) broadcast_dimensions=()]\x00jit(<lambda>)/jit(main)/jit(tril)/select_n\x00jit(<lambda>)/jit(main)/iota[dtype=complex128 shape=(64,) dimension=0]\x00jit(<lambda>)/jit(main)/reshape[new_sizes=(8, 8) dimensions=None]\x00permutation\x00jit(<lambda>)/jit(main)/transpose[permutation=(1, 0)]\x00jit(<lambda>)/jit(main)/real\x00jit(<lambda>)/jit(main)/imag\x00jit(<lambda>)/jit(main)/neg\x00jit(<lambda>)/jit(main)/complex\x00jit(<lambda>)/jit(main)/add\x00jit(<lambda>)/jit(main)/div\x00callee\x00jit(<lambda>)/jit(main)/eigh[lower=True sort_eigenvalues=True]\x00api_version\x00backend_config\x00call_target_name\x00called_computations\x00has_side_effect\x00operand_layouts\x00output_operand_aliases\x00result_layouts\x00jax.result_info\x00tril\x00[0]\x00[1]\x00main\x00public\x00private\x00\x00lapack_zheevd\x00",
        xla_call_module_version=4,
    ),  # End paste
)

data_2024_05_31 = {}


# Pasted from the test output (see export_back_compat_test_util.py module docstring)
data_2024_05_31["c128"] = dict(
    testdata_version=1,
    platform='cpu',
    custom_call_targets=['lapack_zheevd_ffi'],
    serialized_date=datetime.date(2024, 5, 31),
    inputs=(),
    expected_outputs=(array([[-6.1857700048412056e-01+0.j,  2.4081403770912022e-01+0.j,
         3.5662489253627483e-01+0.j, -6.3034019033669797e-01+0.j,
         1.0043483479985752e-16+0.j, -2.8842036081919542e-02+0.j,
         7.7164692943283169e-25+0.j, -1.8446994643771725e-01+0.j],
       [-4.7070881487314609e-01+0.j,  4.7473787464450828e-01+0.j,
        -4.8036836210243361e-01+0.j,  4.3802686872516400e-01+0.j,
         1.7961797619639255e-01+0.j,  8.3080980076741355e-03+0.j,
         2.1415294457221759e-01+0.j, -2.2856669794666584e-01+0.j],
       [-3.2284062926217072e-01+0.j, -5.4336490915553370e-01+0.j,
         2.2181041859724987e-01+0.j,  2.9947877954402286e-01+0.j,
        -3.6491813600134637e-01+0.j,  3.2867679819727436e-01+0.j,
         3.8223299448843473e-01+0.j, -2.7266344945561438e-01+0.j],
       [-1.7497244365119527e-01+0.j, -8.9251550609769331e-02+0.j,
        -6.3518515114898352e-02+0.j,  1.9162997359209963e-01+0.j,
        -2.2087281326110142e-01+0.j,  5.9957027043505008e-02+0.j,
        -8.7632498908241274e-01+0.j, -3.1676020096456303e-01+0.j],
       [-2.7104258040220017e-02+0.j, -3.3772873786627688e-01+0.j,
         2.5901386593721754e-01+0.j,  1.7032650752287815e-01+0.j,
         6.7521217612940321e-01+0.j, -4.5036136532965476e-01+0.j,
        -1.2279030059078447e-02+0.j, -3.6085695247351163e-01+0.j],
       [ 1.2076392757075533e-01+0.j, -3.3834734096469249e-01+0.j,
        -6.5506827461665529e-01+0.j, -5.0472498521116760e-01+0.j,
         6.9987430903492132e-02+0.j,  1.0595648906599270e-01+0.j,
         8.3443844143082035e-02+0.j, -4.0495370398246017e-01+0.j],
       [ 2.6863211318173102e-01+0.j,  2.2958613191407312e-01+0.j,
         6.3952843755683969e-02+0.j,  1.8776775771084192e-02+0.j,
        -5.3523731432241317e-01+0.j, -5.9199531677602002e-01+0.j,
         1.7916671834524250e-01+0.j, -4.4905045549140887e-01+0.j],
       [ 4.1650029879270667e-01+0.j,  3.6355449432857068e-01+0.j,
         2.9755313100756148e-01+0.j,  1.6826270392616000e-02+0.j,
         1.9621068035557282e-01+0.j,  5.6830030587314817e-01+0.j,
         2.9607517592514260e-02+0.j, -4.9314720700035747e-01+0.j]]), array([-2.4598804776133626e+01, -4.6567755957874661e-14,
       -1.9932120610662194e-14, -5.7323356091157378e-15,
       -4.5459724251334835e-16,  4.0479851042511616e-14,
        9.2325194924982089e-14,  2.7659880477613365e+02])),
    mlir_module_text=r"""
#loc7 = loc("third_party/py/jax/tests/export_back_compat_test.py":260:27)
#loc18 = loc("jit(<lambda>)/jit(main)/pjit[in_shardings=(UnspecifiedValue,) out_shardings=(UnspecifiedValue,) in_layouts=(None,) out_layouts=(None,) resource_env=None donated_invars=(False,) name=tril keep_unused=False inline=False]"(#loc7))
module @jit__lambda_ attributes {jax.uses_shape_polymorphism = false, mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} {
  func.func public @main() -> (tensor<8x8xcomplex<f64>> {jax.result_info = "[0]", mhlo.layout_mode = "default"}, tensor<8xf64> {jax.result_info = "[1]", mhlo.layout_mode = "default"}) {
    %0 = stablehlo.iota dim = 0 : tensor<64xcomplex<f64>> loc(#loc9)
    %1 = stablehlo.reshape %0 : (tensor<64xcomplex<f64>>) -> tensor<8x8xcomplex<f64>> loc(#loc10)
    %2 = stablehlo.transpose %1, dims = [1, 0] : (tensor<8x8xcomplex<f64>>) -> tensor<8x8xcomplex<f64>> loc(#loc11)
    %3 = stablehlo.real %2 : (tensor<8x8xcomplex<f64>>) -> tensor<8x8xf64> loc(#loc12)
    %4 = stablehlo.imag %2 : (tensor<8x8xcomplex<f64>>) -> tensor<8x8xf64> loc(#loc13)
    %5 = stablehlo.negate %4 : tensor<8x8xf64> loc(#loc14)
    %6 = stablehlo.complex %3, %5 : tensor<8x8xcomplex<f64>> loc(#loc15)
    %7 = stablehlo.add %1, %6 : tensor<8x8xcomplex<f64>> loc(#loc16)
    %cst = stablehlo.constant dense<(2.000000e+00,0.000000e+00)> : tensor<complex<f64>> loc(#loc)
    %8 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<complex<f64>>) -> tensor<8x8xcomplex<f64>> loc(#loc17)
    %9 = stablehlo.divide %7, %8 : tensor<8x8xcomplex<f64>> loc(#loc17)
    %10 = call @tril(%9) : (tensor<8x8xcomplex<f64>>) -> tensor<8x8xcomplex<f64>> loc(#loc18)
    %c = stablehlo.constant dense<8> : tensor<i64> loc(#loc19)
    %c_0 = stablehlo.constant dense<8> : tensor<i64> loc(#loc19)
    %11:6 = stablehlo.custom_call @lapack_zheevd_ffi(%10) {mhlo.backend_config = {mode = 86 : ui8, uplo = 76 : ui8}, operand_layouts = [dense<[0, 1]> : tensor<2xindex>], output_operand_aliases = [#stablehlo.output_operand_alias<output_tuple_indices = [0], operand_index = 0, operand_tuple_indices = []>], result_layouts = [dense<[0, 1]> : tensor<2xindex>, dense<0> : tensor<1xindex>, dense<> : tensor<0xindex>, dense<0> : tensor<1xindex>, dense<0> : tensor<1xindex>, dense<0> : tensor<1xindex>]} : (tensor<8x8xcomplex<f64>>) -> (tensor<8x8xcomplex<f64>>, tensor<8xf64>, tensor<i32>, tensor<80xcomplex<f64>>, tensor<169xf64>, tensor<43xi32>) loc(#loc19)
    %c_1 = stablehlo.constant dense<0> : tensor<i32> loc(#loc19)
    %12 = stablehlo.broadcast_in_dim %c_1, dims = [] : (tensor<i32>) -> tensor<i32> loc(#loc19)
    %13 = stablehlo.compare  EQ, %11#2, %12,  SIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1> loc(#loc19)
    %14 = stablehlo.broadcast_in_dim %13, dims = [] : (tensor<i1>) -> tensor<1x1xi1> loc(#loc19)
    %cst_2 = stablehlo.constant dense<(0x7FF8000000000000,0x7FF8000000000000)> : tensor<complex<f64>> loc(#loc19)
    %15 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<complex<f64>>) -> tensor<8x8xcomplex<f64>> loc(#loc19)
    %16 = stablehlo.broadcast_in_dim %14, dims = [0, 1] : (tensor<1x1xi1>) -> tensor<8x8xi1> loc(#loc19)
    %17 = stablehlo.select %16, %11#0, %15 : tensor<8x8xi1>, tensor<8x8xcomplex<f64>> loc(#loc19)
    %18 = stablehlo.broadcast_in_dim %13, dims = [] : (tensor<i1>) -> tensor<1xi1> loc(#loc19)
    %cst_3 = stablehlo.constant dense<0x7FF8000000000000> : tensor<f64> loc(#loc19)
    %19 = stablehlo.broadcast_in_dim %cst_3, dims = [] : (tensor<f64>) -> tensor<8xf64> loc(#loc19)
    %20 = stablehlo.broadcast_in_dim %18, dims = [0] : (tensor<1xi1>) -> tensor<8xi1> loc(#loc19)
    %21 = stablehlo.select %20, %11#1, %19 : tensor<8xi1>, tensor<8xf64> loc(#loc19)
    return %17, %21 : tensor<8x8xcomplex<f64>>, tensor<8xf64> loc(#loc)
  } loc(#loc)
  func.func private @tril(%arg0: tensor<8x8xcomplex<f64>> {mhlo.layout_mode = "default"} loc("jit(<lambda>)/jit(main)/pjit[in_shardings=(UnspecifiedValue,) out_shardings=(UnspecifiedValue,) in_layouts=(None,) out_layouts=(None,) resource_env=None donated_invars=(False,) name=tril keep_unused=False inline=False]"(#loc7))) -> (tensor<8x8xcomplex<f64>> {mhlo.layout_mode = "default"}) {
    %0 = stablehlo.iota dim = 0 : tensor<8x8xi32> loc(#loc20)
    %c = stablehlo.constant dense<0> : tensor<i32> loc(#loc18)
    %1 = stablehlo.broadcast_in_dim %c, dims = [] : (tensor<i32>) -> tensor<8x8xi32> loc(#loc21)
    %2 = stablehlo.add %0, %1 : tensor<8x8xi32> loc(#loc21)
    %3 = stablehlo.iota dim = 1 : tensor<8x8xi32> loc(#loc22)
    %4 = stablehlo.compare  GE, %2, %3,  SIGNED : (tensor<8x8xi32>, tensor<8x8xi32>) -> tensor<8x8xi1> loc(#loc23)
    %cst = stablehlo.constant dense<(0.000000e+00,0.000000e+00)> : tensor<complex<f64>> loc(#loc18)
    %5 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<complex<f64>>) -> tensor<8x8xcomplex<f64>> loc(#loc24)
    %6 = stablehlo.select %4, %arg0, %5 : tensor<8x8xi1>, tensor<8x8xcomplex<f64>> loc(#loc25)
    return %6 : tensor<8x8xcomplex<f64>> loc(#loc18)
  } loc(#loc18)
} loc(#loc)
#loc = loc(unknown)
#loc1 = loc("third_party/py/jax/tests/export_back_compat_test.py":252:26)
#loc2 = loc("third_party/py/jax/tests/export_back_compat_test.py":252:14)
#loc3 = loc("third_party/py/jax/tests/export_back_compat_test.py":254:34)
#loc4 = loc("third_party/py/jax/tests/export_back_compat_test.py":254:25)
#loc5 = loc("third_party/py/jax/tests/export_back_compat_test.py":254:15)
#loc6 = loc("third_party/py/jax/tests/export_back_compat_test.py":254:14)
#loc8 = loc("third_party/py/jax/tests/export_back_compat_test.py":260:11)
#loc9 = loc("jit(<lambda>)/jit(main)/iota[dtype=complex128 shape=(64,) dimension=0]"(#loc1))
#loc10 = loc("jit(<lambda>)/jit(main)/reshape[new_sizes=(8, 8) dimensions=None]"(#loc2))
#loc11 = loc("jit(<lambda>)/jit(main)/transpose[permutation=(1, 0)]"(#loc3))
#loc12 = loc("jit(<lambda>)/jit(main)/real"(#loc4))
#loc13 = loc("jit(<lambda>)/jit(main)/imag"(#loc4))
#loc14 = loc("jit(<lambda>)/jit(main)/neg"(#loc4))
#loc15 = loc("jit(<lambda>)/jit(main)/complex"(#loc4))
#loc16 = loc("jit(<lambda>)/jit(main)/add"(#loc5))
#loc17 = loc("jit(<lambda>)/jit(main)/div"(#loc6))
#loc19 = loc("jit(<lambda>)/jit(main)/eigh[lower=True sort_eigenvalues=True subset_by_index=None]"(#loc8))
#loc20 = loc("jit(<lambda>)/jit(main)/jit(tril)/iota[dtype=int32 shape=(8, 8) dimension=0]"(#loc7))
#loc21 = loc("jit(<lambda>)/jit(main)/jit(tril)/add"(#loc7))
#loc22 = loc("jit(<lambda>)/jit(main)/jit(tril)/iota[dtype=int32 shape=(8, 8) dimension=1]"(#loc7))
#loc23 = loc("jit(<lambda>)/jit(main)/jit(tril)/ge"(#loc7))
#loc24 = loc("jit(<lambda>)/jit(main)/jit(tril)/broadcast_in_dim[shape=(8, 8) broadcast_dimensions=()]"(#loc7))
#loc25 = loc("jit(<lambda>)/jit(main)/jit(tril)/select_n"(#loc7))
""",
    mlir_module_serialized=b'ML\xefR\x01StableHLO_v0.9.0\x00\x013\x05\x01\x03\x01\x03\x05\x03#\x07\t\x0b\r\x0f\x11\x13\x15\x17\x19\x1b\x1d\x1f!#%\'\x03\xe6\x02*\x02E\x01\xab\x0f\x0b\x13\x17\x0f\x0b\x07\x17\x0b\x0b\x0f\x0b\x0b\x0b\x0b\x13\x0b\x13\x0f\x0b\x0b\x0f\x13+\x0b\x0f\x0b\x0b\x0b33\x0b\x0f\x0b\x0b\x13\x0f\x0b\x1b\x0f\x0b\x13\x0f\x0b\x0f\x0b\x0f\x0b\x17\x0f\x0b\x17\x13\x0b\x0f\x0b\x17\x0f\x0b\x0f\x0b\x0f\x0b\x0f\x0b\x0f\x0b\x17\x13\x0b\x17\x13\x0b\x0b\x17S\x0b\x0b\x0b\x0b\x0b\x0b\x0b\x0b\x0b\x03a/\x0b\x0b\x0b\x0b\x0f\x0b\x0bO\x0b\x13\x1b\x0b\x1b\x0b\x0b\x0b\x13\x0b\x0b\x0f\x1f\x0f\x0f\x0bOOO/\x0b\x0b\x0b\x0b\x1b\x0b\x0f\x0b\x0f\x0f\x0f\x17#\x0f\x0bOO//\x01\x0b\x1f\x17\x17\x17\x17\x01\x05\x0b\x0f\x03A\x17\x07\x0f\x07\x0f\x07\x0b\x13\x17\x07\x17\x0f\x07\x17\x13\x07\x0f\x17\x17\x13\x17\x17\x17\x13\x13\x13\x13\x0f\x17\x13\x13\x13\x02\xea\n\x1d\x93\x95\x05)\x03\x03\x13\xd7\x17\x03\x12\x047\x1d?\x07\x05+\x1f\x17\x03\xfa\x033\x05-\x05/\x11\x03\x05\x051\x053\x055\x057\x03\x03!\xd3\x059\x03\x03\x0b\xd5\x1dE\x07\x05;\x05=\x1d\x8b\x8d\x03\x03\x0b\xe3\x03\t135\x157\x15\x119\x05?\x11\x01\x00\x05A\x05C\x05E\x03\x0b\x17\xb1\x19\xbd\x1b\xbf\x11\xc9\x1d\xcb\x03\x0b\x17\xb5\x19\xcf\x1b\xb5\x11\xb7\x1d\xd1\x05G\x1dC\x07\x05I\x05K\x03\x03!\xd9\x1dK\x07\x05M\x03\x05\'\xb9)\xdb\x1dQ\x07\x05O\x03\x03\x0b\xdd\x1dW\x07\x05Q\x1d[\x07\x05S\x1d_a\x05U\x17\x03\xf2\x035\x1deg\x05W\x17\x03\xf2\x03\x1d\x03\x03k\xdf\x05Y\x1doq\x05[\x17\x03\xfa\x03E\x1du\x0f\x05]\x1dy\x0f\x05_\x1d}\x0f\x05a\x1d\x81\x0f\x05c\x1d\x85\x87\x05e\x17\x03\xfa\x03\x1f\x03\x03\x0b\xe1\x05g\x17\x03\xfa\x03\x1d\x03\x03\x91\xb7\x05i\x05k\x17\x03\x12\x04\x17\x03\x13\x99\xe5\x9b\xe7\x9d\xe9\x9f\xb1\xa1\xeb\xa3\xed\xa5\xf7\xa7\xf9\xa9\xfd\x05m\x05o\x05q\x05s\x05u\x05w\x05y\x05{\x05}\x1f7\x11\x00\x00\x00\x00\x00\x00\x00\x00\x1d\x7f\x1d\x81\x03\x01\x1d\x83\x03\x03\xcd\x1d\x85\t\x07\x1f5!\x00\x00\x00\x00\x00\x00\x00\x00\x01\x00\x00\x00\x00\x00\x00\x00#\'\x03\x05\xc1\xc5\r\x05\xb3\xc3\xad\xaf\x1d\x87\r\x05\xb3\xc7\xad\xaf\x1d\x89\x1d\x8b\x1d\x8d\r\x03\xad\xaf#)\x1d\x8f\x13\x07\x01\x1f\r\t\x00\x00\x00\x00\x1f+\x01\x13\x07\x05\x07\x05\x1f\t!\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x1f!!\x01\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x1f\t!\x00\x00\x00\x00\x00\x00\x00@\x00\x00\x00\x00\x00\x00\x00\x00\x1f\x1b\x11\x08\x00\x00\x00\x00\x00\x00\x00\x0b\x03\x1d\x91\x1d\x93\x05\x01\r\x05\xef\xf1\xf3\xf5\x1d\x95\x13#V\x1d\x97\x13#L\x03\x03\xbb\x03\x03\xfb\x15\x03\x01\x01\x01\x03\r\xbb\xab\xff\xab\xab\xab\x1f9\x01\x07\x01\x1f\t!\x00\x00\x00\x00\x00\x00\xf8\x7f\x00\x00\x00\x00\x00\x00\xf8\x7f\x1f!!\x00\x00\x00\x00\x00\x00\x00\x00\x01\x00\x00\x00\x00\x00\x00\x00\x1f%\x11\x00\x00\x00\x00\x00\x00\xf8\x7f\x1fC\x11\x00\x00\x00\x00\x00\x00\x00\x00\x03\x05\'\xb9)\x02\x02\x03\x03\x0b\x06\x02\x03\x03\x13\n\x02\x03\x03\x0b\x0e\x02\x03\x03\x13\x12\x02\x01\t\x01\x02\x02)\x05!!\x11\x1d)\x01\x11\x0b)\x01\x17\x01\x03\x0b)\x03!\x0b)\x05!!\x17\x1b)\x05!!\x0b)\x01\x07\x13)\x05!!\x0f)\x03\t\x07!)\x01\x0b\x11\x01\x05\x05\x13\x11\x03\x05\x03\x05)\x03\x01\x07)\x03\x02\x02\x11)\x03\x82\x02\x11)\x03J\x05\x0b)\x03\xad\x17)\x03\t\x1d)\x03\x05\x1d)\x03\x01\x1d)\x01\x0f)\x05\x05\x05\x0f)\x03\x05\x0f)\x03!\x0f)\x03\x05\x07\x04\x12\x05\x05\x01\x11\r/\x07\x03\x01\t\x0b\x11\r;\x07\x03Cu\x07\x03]\x1f\x03-\x13\x06c\x03\x05\x03\x01\x15\x07mi\x03\x05\x03\x03\x17\x06s\x03\x19\x03\x05\x19\x06w\x03\x19\x03\x05\x1b\x06{\x03\x19\x03\t\x1d\x06\x7f\x03\x05\x05\x07\x0b\r\x06\x83\x03\x05\x05\x03\r\x05\x03\r\x89\x03\t\x03\x07+\x05\x03\x05\x03\x11\x1f\x06+\x03\x05\x05\x0f\x13!\x07\t\x8f\x03\x05\x03\x15\x05\x03\x01-\x03\x1b\x05\x03\x01-\x03\x1b#\x07\x01\x97\r\x05\x13\r/13\x03\x17\x05\x03\x01#\x03\r\x03\x07\x01\x05\x03\r\x03)\x0f\x07\x01\x16\x02\x03;\x05!+\x03\x07\x01\x05\x03=\x03-\x05\x03\x01\x1a\x02\x03\t\x03\x07\x01\x05\x03\x05\x031\x03\x07\x01\x1e\x02\x03\x1f\x03/\t\x06\x01\x03\x05\x075\x1d3\x03\x07\x01\x05\x03?\x03-\x05\x03\x01"\x02\x03%\x03\x07\x01\x05\x03\x13\x03;\x03\x07\x01&\x02\x03A\x039\t\x06\x01\x03\x13\x07?\x1f=\x11\x04\r\x057A\x0b\x11\t=\x07\x03\x15+\x03\x05\t\x07\x03A\x1f\x03\x15\x05\x03\t#\x03\r\x03\x07%\x05\x03\x15\x03\x05\r\x06%\x03\x15\x05\x03\x07\x07\x03IG\x03\x15\x0f\x07OM\x03\x1f\x05\t\x0b\x05\x03\tS\x03\t\x03\x07U\x05\x03\x05\x03\x0f\t\x06Y\x03\x05\x07\r\x01\x11\x11\x04\t\x03\x13\x06\x03\x01\x05\x01\x00\xe2\x1c\x99\x0b\x0b%\x03\x11\x0f\x0b\t\t\x0b!\x11#\x1f/!)!)#\x1f\x19\xa9\x0f99A9;;m\x19\x85\x8fW\xb3K\x9bM\x9bn\x03\x1b%)9+\x1b\x1f\x1f\x15\x1d\x15+\x13\ri\x1f\x11\x15\x17\x15\x11\x11\x1b\x17\x15\x17\x0f\x11\x15\x11\x19)\x0f\x0b\x11builtin\x00vhlo\x00module\x00broadcast_in_dim_v1\x00constant_v1\x00iota_v1\x00select_v1\x00func_v1\x00add_v1\x00compare_v1\x00return_v1\x00reshape_v1\x00transpose_v1\x00real_v1\x00imag_v1\x00negate_v1\x00complex_v1\x00divide_v1\x00call_v1\x00custom_call_v1\x00third_party/py/jax/tests/export_back_compat_test.py\x00value\x00sym_name\x00broadcast_dimensions\x00arg_attrs\x00function_type\x00res_attrs\x00sym_visibility\x00iota_dimension\x00compare_type\x00comparison_direction\x00jax.uses_shape_polymorphism\x00mhlo.num_partitions\x00mhlo.num_replicas\x00jit__lambda_\x00jit(<lambda>)/jit(main)/pjit[in_shardings=(UnspecifiedValue,) out_shardings=(UnspecifiedValue,) in_layouts=(None,) out_layouts=(None,) resource_env=None donated_invars=(False,) name=tril keep_unused=False inline=False]\x00jit(<lambda>)/jit(main)/jit(tril)/iota[dtype=int32 shape=(8, 8) dimension=0]\x00jit(<lambda>)/jit(main)/jit(tril)/add\x00jit(<lambda>)/jit(main)/jit(tril)/iota[dtype=int32 shape=(8, 8) dimension=1]\x00jit(<lambda>)/jit(main)/jit(tril)/ge\x00jit(<lambda>)/jit(main)/jit(tril)/broadcast_in_dim[shape=(8, 8) broadcast_dimensions=()]\x00jit(<lambda>)/jit(main)/jit(tril)/select_n\x00jit(<lambda>)/jit(main)/iota[dtype=complex128 shape=(64,) dimension=0]\x00jit(<lambda>)/jit(main)/reshape[new_sizes=(8, 8) dimensions=None]\x00permutation\x00jit(<lambda>)/jit(main)/transpose[permutation=(1, 0)]\x00jit(<lambda>)/jit(main)/real\x00jit(<lambda>)/jit(main)/imag\x00jit(<lambda>)/jit(main)/neg\x00jit(<lambda>)/jit(main)/complex\x00jit(<lambda>)/jit(main)/add\x00jit(<lambda>)/jit(main)/div\x00callee\x00jit(<lambda>)/jit(main)/eigh[lower=True sort_eigenvalues=True subset_by_index=None]\x00api_version\x00backend_config\x00call_target_name\x00called_computations\x00has_side_effect\x00mhlo.backend_config\x00operand_layouts\x00output_operand_aliases\x00result_layouts\x00mhlo.layout_mode\x00default\x00jax.result_info\x00tril\x00[0]\x00[1]\x00main\x00public\x00private\x00\x00lapack_zheevd_ffi\x00mode\x00uplo\x00',
    xla_call_module_version=9,
    nr_devices=1,
)  # End paste


# Pasted from the test output (see export_back_compat_test_util.py module docstring)
data_2024_05_31["c64"] = dict(
    testdata_version=1,
    platform='cpu',
    custom_call_targets=['lapack_cheevd_ffi'],
    serialized_date=datetime.date(2024, 5, 31),
    inputs=(),
    expected_outputs=(array([[-0.6185769    +0.j, -0.20142993   +0.j, -0.09725195   +0.j,
         0.62983674   +0.j, -0.07926044   +0.j,  0.3605001    -0.j,
        -0.019093221  +0.j, -0.18446997   +0.j],
       [-0.47070873   +0.j,  0.29325768   +0.j, -0.19454116   +0.j,
        -0.6394365    +0.j,  0.06229549   +0.j,  0.33249345   +0.j,
         0.28112718   +0.j, -0.22856665   +0.j],
       [-0.32284075   +0.j, -0.12361939   +0.j,  0.20547704   +0.j,
        -0.18307868   +0.j,  0.47294614   +0.j, -0.3170349    +0.j,
        -0.6373532    +0.j, -0.27266347   +0.j],
       [-0.17497246   +0.j, -0.079641335  +0.j,  0.15042792   +0.j,
        -0.15416273   +0.j, -0.815209     +0.j, -0.38054234   +0.j,
        -0.083263926  +0.j, -0.31676024   +0.j],
       [-0.027104257  +0.j, -0.26490977   +0.j,  0.32271704   +0.j,
         0.08653544   +0.j,  0.30305928   +0.j, -0.33998996   +0.j,
         0.6926741    +0.j, -0.360857     +0.j],
       [ 0.120763965  +0.j,  0.43288827   +0.j, -0.64385164   +0.j,
         0.2652551    +0.j,  0.094823755  +0.j, -0.37435007   +0.j,
         0.00091664493+0.j, -0.40495378   +0.j],
       [ 0.26863196   +0.j,  0.51607686   +0.j,  0.53846526   +0.j,
         0.16969058   +0.j, -0.0216703    +0.j,  0.35755336   +0.j,
        -0.113144726  +0.j, -0.4490505    +0.j],
       [ 0.4165004    +0.j, -0.57262254   +0.j, -0.28144246   +0.j,
        -0.17463988   +0.j, -0.016984984  +0.j,  0.3613705    +0.j,
        -0.12186296   +0.j, -0.49314725   +0.j]], dtype=complex64), array([-2.4598808e+01, -3.3105560e-05, -3.1002426e-05, -1.0103593e-05,
       -1.0022322e-05,  4.0141886e-06,  9.5510331e-06,  2.7659882e+02],
      dtype=float32)),
    mlir_module_text=r"""
#loc7 = loc("third_party/py/jax/tests/export_back_compat_test.py":260:27)
#loc18 = loc("jit(<lambda>)/jit(main)/pjit[in_shardings=(UnspecifiedValue,) out_shardings=(UnspecifiedValue,) in_layouts=(None,) out_layouts=(None,) resource_env=None donated_invars=(False,) name=tril keep_unused=False inline=False]"(#loc7))
module @jit__lambda_ attributes {jax.uses_shape_polymorphism = false, mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} {
  func.func public @main() -> (tensor<8x8xcomplex<f32>> {jax.result_info = "[0]", mhlo.layout_mode = "default"}, tensor<8xf32> {jax.result_info = "[1]", mhlo.layout_mode = "default"}) {
    %0 = stablehlo.iota dim = 0 : tensor<64xcomplex<f32>> loc(#loc9)
    %1 = stablehlo.reshape %0 : (tensor<64xcomplex<f32>>) -> tensor<8x8xcomplex<f32>> loc(#loc10)
    %2 = stablehlo.transpose %1, dims = [1, 0] : (tensor<8x8xcomplex<f32>>) -> tensor<8x8xcomplex<f32>> loc(#loc11)
    %3 = stablehlo.real %2 : (tensor<8x8xcomplex<f32>>) -> tensor<8x8xf32> loc(#loc12)
    %4 = stablehlo.imag %2 : (tensor<8x8xcomplex<f32>>) -> tensor<8x8xf32> loc(#loc13)
    %5 = stablehlo.negate %4 : tensor<8x8xf32> loc(#loc14)
    %6 = stablehlo.complex %3, %5 : tensor<8x8xcomplex<f32>> loc(#loc15)
    %7 = stablehlo.add %1, %6 : tensor<8x8xcomplex<f32>> loc(#loc16)
    %cst = stablehlo.constant dense<(2.000000e+00,0.000000e+00)> : tensor<complex<f32>> loc(#loc)
    %8 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<complex<f32>>) -> tensor<8x8xcomplex<f32>> loc(#loc17)
    %9 = stablehlo.divide %7, %8 : tensor<8x8xcomplex<f32>> loc(#loc17)
    %10 = call @tril(%9) : (tensor<8x8xcomplex<f32>>) -> tensor<8x8xcomplex<f32>> loc(#loc18)
    %c = stablehlo.constant dense<8> : tensor<i64> loc(#loc19)
    %c_0 = stablehlo.constant dense<8> : tensor<i64> loc(#loc19)
    %11:6 = stablehlo.custom_call @lapack_cheevd_ffi(%10) {mhlo.backend_config = {mode = 86 : ui8, uplo = 76 : ui8}, operand_layouts = [dense<[0, 1]> : tensor<2xindex>], output_operand_aliases = [#stablehlo.output_operand_alias<output_tuple_indices = [0], operand_index = 0, operand_tuple_indices = []>], result_layouts = [dense<[0, 1]> : tensor<2xindex>, dense<0> : tensor<1xindex>, dense<> : tensor<0xindex>, dense<0> : tensor<1xindex>, dense<0> : tensor<1xindex>, dense<0> : tensor<1xindex>]} : (tensor<8x8xcomplex<f32>>) -> (tensor<8x8xcomplex<f32>>, tensor<8xf32>, tensor<i32>, tensor<80xcomplex<f32>>, tensor<169xf32>, tensor<43xi32>) loc(#loc19)
    %c_1 = stablehlo.constant dense<0> : tensor<i32> loc(#loc19)
    %12 = stablehlo.broadcast_in_dim %c_1, dims = [] : (tensor<i32>) -> tensor<i32> loc(#loc19)
    %13 = stablehlo.compare  EQ, %11#2, %12,  SIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1> loc(#loc19)
    %14 = stablehlo.broadcast_in_dim %13, dims = [] : (tensor<i1>) -> tensor<1x1xi1> loc(#loc19)
    %cst_2 = stablehlo.constant dense<(0x7FC00000,0x7FC00000)> : tensor<complex<f32>> loc(#loc19)
    %15 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<complex<f32>>) -> tensor<8x8xcomplex<f32>> loc(#loc19)
    %16 = stablehlo.broadcast_in_dim %14, dims = [0, 1] : (tensor<1x1xi1>) -> tensor<8x8xi1> loc(#loc19)
    %17 = stablehlo.select %16, %11#0, %15 : tensor<8x8xi1>, tensor<8x8xcomplex<f32>> loc(#loc19)
    %18 = stablehlo.broadcast_in_dim %13, dims = [] : (tensor<i1>) -> tensor<1xi1> loc(#loc19)
    %cst_3 = stablehlo.constant dense<0x7FC00000> : tensor<f32> loc(#loc19)
    %19 = stablehlo.broadcast_in_dim %cst_3, dims = [] : (tensor<f32>) -> tensor<8xf32> loc(#loc19)
    %20 = stablehlo.broadcast_in_dim %18, dims = [0] : (tensor<1xi1>) -> tensor<8xi1> loc(#loc19)
    %21 = stablehlo.select %20, %11#1, %19 : tensor<8xi1>, tensor<8xf32> loc(#loc19)
    return %17, %21 : tensor<8x8xcomplex<f32>>, tensor<8xf32> loc(#loc)
  } loc(#loc)
  func.func private @tril(%arg0: tensor<8x8xcomplex<f32>> {mhlo.layout_mode = "default"} loc("jit(<lambda>)/jit(main)/pjit[in_shardings=(UnspecifiedValue,) out_shardings=(UnspecifiedValue,) in_layouts=(None,) out_layouts=(None,) resource_env=None donated_invars=(False,) name=tril keep_unused=False inline=False]"(#loc7))) -> (tensor<8x8xcomplex<f32>> {mhlo.layout_mode = "default"}) {
    %0 = stablehlo.iota dim = 0 : tensor<8x8xi32> loc(#loc20)
    %c = stablehlo.constant dense<0> : tensor<i32> loc(#loc18)
    %1 = stablehlo.broadcast_in_dim %c, dims = [] : (tensor<i32>) -> tensor<8x8xi32> loc(#loc21)
    %2 = stablehlo.add %0, %1 : tensor<8x8xi32> loc(#loc21)
    %3 = stablehlo.iota dim = 1 : tensor<8x8xi32> loc(#loc22)
    %4 = stablehlo.compare  GE, %2, %3,  SIGNED : (tensor<8x8xi32>, tensor<8x8xi32>) -> tensor<8x8xi1> loc(#loc23)
    %cst = stablehlo.constant dense<(0.000000e+00,0.000000e+00)> : tensor<complex<f32>> loc(#loc18)
    %5 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<complex<f32>>) -> tensor<8x8xcomplex<f32>> loc(#loc24)
    %6 = stablehlo.select %4, %arg0, %5 : tensor<8x8xi1>, tensor<8x8xcomplex<f32>> loc(#loc25)
    return %6 : tensor<8x8xcomplex<f32>> loc(#loc18)
  } loc(#loc18)
} loc(#loc)
#loc = loc(unknown)
#loc1 = loc("third_party/py/jax/tests/export_back_compat_test.py":252:26)
#loc2 = loc("third_party/py/jax/tests/export_back_compat_test.py":252:14)
#loc3 = loc("third_party/py/jax/tests/export_back_compat_test.py":254:34)
#loc4 = loc("third_party/py/jax/tests/export_back_compat_test.py":254:25)
#loc5 = loc("third_party/py/jax/tests/export_back_compat_test.py":254:15)
#loc6 = loc("third_party/py/jax/tests/export_back_compat_test.py":254:14)
#loc8 = loc("third_party/py/jax/tests/export_back_compat_test.py":260:11)
#loc9 = loc("jit(<lambda>)/jit(main)/iota[dtype=complex64 shape=(64,) dimension=0]"(#loc1))
#loc10 = loc("jit(<lambda>)/jit(main)/reshape[new_sizes=(8, 8) dimensions=None]"(#loc2))
#loc11 = loc("jit(<lambda>)/jit(main)/transpose[permutation=(1, 0)]"(#loc3))
#loc12 = loc("jit(<lambda>)/jit(main)/real"(#loc4))
#loc13 = loc("jit(<lambda>)/jit(main)/imag"(#loc4))
#loc14 = loc("jit(<lambda>)/jit(main)/neg"(#loc4))
#loc15 = loc("jit(<lambda>)/jit(main)/complex"(#loc4))
#loc16 = loc("jit(<lambda>)/jit(main)/add"(#loc5))
#loc17 = loc("jit(<lambda>)/jit(main)/div"(#loc6))
#loc19 = loc("jit(<lambda>)/jit(main)/eigh[lower=True sort_eigenvalues=True subset_by_index=None]"(#loc8))
#loc20 = loc("jit(<lambda>)/jit(main)/jit(tril)/iota[dtype=int32 shape=(8, 8) dimension=0]"(#loc7))
#loc21 = loc("jit(<lambda>)/jit(main)/jit(tril)/add"(#loc7))
#loc22 = loc("jit(<lambda>)/jit(main)/jit(tril)/iota[dtype=int32 shape=(8, 8) dimension=1]"(#loc7))
#loc23 = loc("jit(<lambda>)/jit(main)/jit(tril)/ge"(#loc7))
#loc24 = loc("jit(<lambda>)/jit(main)/jit(tril)/broadcast_in_dim[shape=(8, 8) broadcast_dimensions=()]"(#loc7))
#loc25 = loc("jit(<lambda>)/jit(main)/jit(tril)/select_n"(#loc7))
""",
    mlir_module_serialized=b'ML\xefR\x01StableHLO_v0.9.0\x00\x013\x05\x01\x03\x01\x03\x05\x03#\x07\t\x0b\r\x0f\x11\x13\x15\x17\x19\x1b\x1d\x1f!#%\'\x03\xe6\x02*\x02E\x01\xab\x0f\x0b\x13\x17\x0f\x0b\x07\x17\x0b\x0b\x0f\x0b\x0b\x0b\x0b\x13\x0b\x13\x0f\x0b\x0b\x0f\x13+\x0b\x0f\x0b\x0b\x0b33\x0b\x0f\x0b\x0b\x13\x0f\x0b\x1b\x0f\x0b\x13\x0f\x0b\x0f\x0b\x0f\x0b\x17\x0f\x0b\x17\x13\x0b\x0f\x0b\x17\x0f\x0b\x0f\x0b\x0f\x0b\x0f\x0b\x0f\x0b\x17\x13\x0b\x17\x13\x0b\x0b\x17S\x0b\x0b\x0b\x0b\x0b\x0b\x0b\x0b\x0b\x03a/\x0b\x0b\x0b\x0b\x0f\x0b\x0bO\x0b\x13\x1b\x0b\x1b\x0b\x0b\x0b\x13\x0b\x0b\x0f\x1f\x0f\x0f\x0b/O//\x0b\x0b\x0b\x0b\x1b\x0b\x0f\x0b\x0f\x0f\x0f\x17#\x0f\x0b/O\x1f/\x01\x0b\x1f\x17\x17\x17\x17\x01\x05\x0b\x0f\x03A\x17\x07\x0f\x07\x0f\x07\x0b\x13\x17\x07\x17\x0f\x07\x17\x13\x07\x0f\x17\x17\x13\x17\x17\x17\x13\x13\x13\x13\x0f\x17\x13\x13\x13\x02z\n\x1d\x93\x95\x05)\x03\x03\x13\xd7\x17\x03\x12\x047\x1d?\x07\x05+\x1f\x17\x03\xfa\x033\x05-\x05/\x11\x03\x05\x051\x053\x055\x057\x03\x03!\xd3\x059\x03\x03\x0b\xd5\x1dE\x07\x05;\x05=\x1d\x8b\x8d\x03\x03\x0b\xe3\x03\t135\x157\x15\x119\x05?\x11\x01\x00\x05A\x05C\x05E\x03\x0b\x17\xb1\x19\xbd\x1b\xbf\x11\xc9\x1d\xcb\x03\x0b\x17\xb5\x19\xcf\x1b\xb5\x11\xb7\x1d\xd1\x05G\x1dC\x07\x05I\x05K\x03\x03!\xd9\x1dK\x07\x05M\x03\x05\'\xb9)\xdb\x1dQ\x07\x05O\x03\x03\x0b\xdd\x1dW\x07\x05Q\x1d[\x07\x05S\x1d_a\x05U\x17\x03\xf2\x035\x1deg\x05W\x17\x03\xf2\x03\x1d\x03\x03k\xdf\x05Y\x1doq\x05[\x17\x03\xfa\x03E\x1du\x0f\x05]\x1dy\x0f\x05_\x1d}\x0f\x05a\x1d\x81\x0f\x05c\x1d\x85\x87\x05e\x17\x03\xfa\x03\x1f\x03\x03\x0b\xe1\x05g\x17\x03\xfa\x03\x1d\x03\x03\x91\xb7\x05i\x05k\x17\x03\x12\x04\x17\x03\x13\x99\xe5\x9b\xe7\x9d\xe9\x9f\xb1\xa1\xeb\xa3\xed\xa5\xf7\xa7\xf9\xa9\xfd\x05m\x05o\x05q\x05s\x05u\x05w\x05y\x05{\x05}\x1f7\x11\x00\x00\x00\x00\x00\x00\x00\x00\x1d\x7f\x1d\x81\x03\x01\x1d\x83\x03\x03\xcd\x1d\x85\t\x07\x1f5!\x00\x00\x00\x00\x00\x00\x00\x00\x01\x00\x00\x00\x00\x00\x00\x00#\'\x03\x05\xc1\xc5\r\x05\xb3\xc3\xad\xaf\x1d\x87\r\x05\xb3\xc7\xad\xaf\x1d\x89\x1d\x8b\x1d\x8d\r\x03\xad\xaf#)\x1d\x8f\x13\x07\x01\x1f\r\t\x00\x00\x00\x00\x1f+\x01\x13\x07\x05\x07\x05\x1f\t\x11\x00\x00\x00\x00\x00\x00\x00\x00\x1f!!\x01\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x1f\t\x11\x00\x00\x00@\x00\x00\x00\x00\x1f\x1b\x11\x08\x00\x00\x00\x00\x00\x00\x00\x0b\x03\x1d\x91\x1d\x93\x05\x01\r\x05\xef\xf1\xf3\xf5\x1d\x95\x13#V\x1d\x97\x13#L\x03\x03\xbb\x03\x03\xfb\x15\x03\x01\x01\x01\x03\r\xbb\xab\xff\xab\xab\xab\x1f9\x01\x07\x01\x1f\t\x11\x00\x00\xc0\x7f\x00\x00\xc0\x7f\x1f!!\x00\x00\x00\x00\x00\x00\x00\x00\x01\x00\x00\x00\x00\x00\x00\x00\x1f%\t\x00\x00\xc0\x7f\x1fC\x11\x00\x00\x00\x00\x00\x00\x00\x00\x03\x05\'\xb9)\x02\x02\x03\x03\x0b\x06\x02\x03\x03\x13\n\x02\x03\x03\x0b\x0e\x02\x03\x03\x13\x12\x02\x01\t\x01\x02\x02)\x05!!\x11\x1d)\x01\x11\t)\x01\x17\x01\x03\x0b)\x03!\x0b)\x05!!\x17\x1b)\x05!!\x0b)\x01\x07\x13)\x05!!\x0f)\x03\t\x07!)\x01\x0b\x11\x01\x05\x05\x13\x11\x03\x05\x03\x05)\x03\x01\x07)\x03\x02\x02\x11)\x03\x82\x02\x11)\x03J\x05\x0b)\x03\xad\x17)\x03\t\x1d)\x03\x05\x1d)\x03\x01\x1d)\x01\x0f)\x05\x05\x05\x0f)\x03\x05\x0f)\x03!\x0f)\x03\x05\x07\x04\x12\x05\x05\x01\x11\r/\x07\x03\x01\t\x0b\x11\r;\x07\x03Cu\x07\x03]\x1f\x03-\x13\x06c\x03\x05\x03\x01\x15\x07mi\x03\x05\x03\x03\x17\x06s\x03\x19\x03\x05\x19\x06w\x03\x19\x03\x05\x1b\x06{\x03\x19\x03\t\x1d\x06\x7f\x03\x05\x05\x07\x0b\r\x06\x83\x03\x05\x05\x03\r\x05\x03\r\x89\x03\t\x03\x07+\x05\x03\x05\x03\x11\x1f\x06+\x03\x05\x05\x0f\x13!\x07\t\x8f\x03\x05\x03\x15\x05\x03\x01-\x03\x1b\x05\x03\x01-\x03\x1b#\x07\x01\x97\r\x05\x13\r/13\x03\x17\x05\x03\x01#\x03\r\x03\x07\x01\x05\x03\r\x03)\x0f\x07\x01\x16\x02\x03;\x05!+\x03\x07\x01\x05\x03=\x03-\x05\x03\x01\x1a\x02\x03\t\x03\x07\x01\x05\x03\x05\x031\x03\x07\x01\x1e\x02\x03\x1f\x03/\t\x06\x01\x03\x05\x075\x1d3\x03\x07\x01\x05\x03?\x03-\x05\x03\x01"\x02\x03%\x03\x07\x01\x05\x03\x13\x03;\x03\x07\x01&\x02\x03A\x039\t\x06\x01\x03\x13\x07?\x1f=\x11\x04\r\x057A\x0b\x11\t=\x07\x03\x15+\x03\x05\t\x07\x03A\x1f\x03\x15\x05\x03\t#\x03\r\x03\x07%\x05\x03\x15\x03\x05\r\x06%\x03\x15\x05\x03\x07\x07\x03IG\x03\x15\x0f\x07OM\x03\x1f\x05\t\x0b\x05\x03\tS\x03\t\x03\x07U\x05\x03\x05\x03\x0f\t\x06Y\x03\x05\x07\r\x01\x11\x11\x04\t\x03\x13\x06\x03\x01\x05\x01\x00\xde\x1c\x99\x0b\x0b%\x03\x11\x0f\x0b\t\t\x0b!\x11#\x1f/!)!)#\x1f\x19\xa9\x0f99A9;;m\x19\x85\x8dW\xb3K\x9bM\x9bn\x03\x1b%)9+\x1b\x1f\x1f\x15\x1d\x15+\x13\ri\x1f\x11\x15\x17\x15\x11\x11\x1b\x17\x15\x17\x0f\x11\x15\x11\x19)\x0f\x0b\x11builtin\x00vhlo\x00module\x00broadcast_in_dim_v1\x00constant_v1\x00iota_v1\x00select_v1\x00func_v1\x00add_v1\x00compare_v1\x00return_v1\x00reshape_v1\x00transpose_v1\x00real_v1\x00imag_v1\x00negate_v1\x00complex_v1\x00divide_v1\x00call_v1\x00custom_call_v1\x00third_party/py/jax/tests/export_back_compat_test.py\x00value\x00sym_name\x00broadcast_dimensions\x00arg_attrs\x00function_type\x00res_attrs\x00sym_visibility\x00iota_dimension\x00compare_type\x00comparison_direction\x00jax.uses_shape_polymorphism\x00mhlo.num_partitions\x00mhlo.num_replicas\x00jit__lambda_\x00jit(<lambda>)/jit(main)/pjit[in_shardings=(UnspecifiedValue,) out_shardings=(UnspecifiedValue,) in_layouts=(None,) out_layouts=(None,) resource_env=None donated_invars=(False,) name=tril keep_unused=False inline=False]\x00jit(<lambda>)/jit(main)/jit(tril)/iota[dtype=int32 shape=(8, 8) dimension=0]\x00jit(<lambda>)/jit(main)/jit(tril)/add\x00jit(<lambda>)/jit(main)/jit(tril)/iota[dtype=int32 shape=(8, 8) dimension=1]\x00jit(<lambda>)/jit(main)/jit(tril)/ge\x00jit(<lambda>)/jit(main)/jit(tril)/broadcast_in_dim[shape=(8, 8) broadcast_dimensions=()]\x00jit(<lambda>)/jit(main)/jit(tril)/select_n\x00jit(<lambda>)/jit(main)/iota[dtype=complex64 shape=(64,) dimension=0]\x00jit(<lambda>)/jit(main)/reshape[new_sizes=(8, 8) dimensions=None]\x00permutation\x00jit(<lambda>)/jit(main)/transpose[permutation=(1, 0)]\x00jit(<lambda>)/jit(main)/real\x00jit(<lambda>)/jit(main)/imag\x00jit(<lambda>)/jit(main)/neg\x00jit(<lambda>)/jit(main)/complex\x00jit(<lambda>)/jit(main)/add\x00jit(<lambda>)/jit(main)/div\x00callee\x00jit(<lambda>)/jit(main)/eigh[lower=True sort_eigenvalues=True subset_by_index=None]\x00api_version\x00backend_config\x00call_target_name\x00called_computations\x00has_side_effect\x00mhlo.backend_config\x00operand_layouts\x00output_operand_aliases\x00result_layouts\x00mhlo.layout_mode\x00default\x00jax.result_info\x00tril\x00[0]\x00[1]\x00main\x00public\x00private\x00\x00lapack_cheevd_ffi\x00mode\x00uplo\x00',
    xla_call_module_version=9,
    nr_devices=1,
)  # End paste


# Pasted from the test output (see export_back_compat_test_util.py module docstring)
data_2024_05_31["f32"] = dict(
    testdata_version=1,
    platform='cpu',
    custom_call_targets=['lapack_ssyevd_ffi'],
    serialized_date=datetime.date(2024, 5, 31),
    inputs=(),
    expected_outputs=(array([[-0.6185769    , -0.20142993   , -0.09725195   ,  0.62983674   ,
        -0.07926044   ,  0.3605001    , -0.019093221  , -0.18446997   ],
       [-0.47070873   ,  0.29325768   , -0.19454119   , -0.6394365    ,
         0.0622955    ,  0.33249345   ,  0.28112718   , -0.22856665   ],
       [-0.32284075   , -0.12361939   ,  0.20547704   , -0.18307868   ,
         0.47294614   , -0.3170349    , -0.6373532    , -0.27266347   ],
       [-0.17497246   , -0.079641335  ,  0.15042791   , -0.15416273   ,
        -0.815209     , -0.38054234   , -0.083263926  , -0.31676024   ],
       [-0.027104253  , -0.26490977   ,  0.32271704   ,  0.08653544   ,
         0.30305928   , -0.33998996   ,  0.6926741    , -0.360857     ],
       [ 0.12076397   ,  0.43288827   , -0.64385164   ,  0.2652551    ,
         0.09482376   , -0.37435007   ,  0.00091664493, -0.40495378   ],
       [ 0.26863196   ,  0.51607686   ,  0.53846526   ,  0.16969058   ,
        -0.021670295  ,  0.35755336   , -0.113144726  , -0.4490505    ],
       [ 0.4165004    , -0.57262254   , -0.2814425    , -0.17463988   ,
        -0.01698498   ,  0.3613705    , -0.12186296   , -0.49314725   ]],
      dtype=float32), array([-2.4598808e+01, -3.3105560e-05, -3.1002426e-05, -1.0103593e-05,
       -1.0022322e-05,  4.0141886e-06,  9.5510331e-06,  2.7659882e+02],
      dtype=float32)),
    mlir_module_text=r"""
#loc6 = loc("third_party/py/jax/tests/export_back_compat_test.py":260:27)
#loc13 = loc("jit(<lambda>)/jit(main)/pjit[in_shardings=(UnspecifiedValue,) out_shardings=(UnspecifiedValue,) in_layouts=(None,) out_layouts=(None,) resource_env=None donated_invars=(False,) name=tril keep_unused=False inline=False]"(#loc6))
module @jit__lambda_ attributes {jax.uses_shape_polymorphism = false, mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} {
  func.func public @main() -> (tensor<8x8xf32> {jax.result_info = "[0]", mhlo.layout_mode = "default"}, tensor<8xf32> {jax.result_info = "[1]", mhlo.layout_mode = "default"}) {
    %0 = stablehlo.iota dim = 0 : tensor<64xf32> loc(#loc8)
    %1 = stablehlo.reshape %0 : (tensor<64xf32>) -> tensor<8x8xf32> loc(#loc9)
    %2 = stablehlo.transpose %1, dims = [1, 0] : (tensor<8x8xf32>) -> tensor<8x8xf32> loc(#loc10)
    %3 = stablehlo.add %1, %2 : tensor<8x8xf32> loc(#loc11)
    %cst = stablehlo.constant dense<2.000000e+00> : tensor<f32> loc(#loc)
    %4 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<8x8xf32> loc(#loc12)
    %5 = stablehlo.divide %3, %4 : tensor<8x8xf32> loc(#loc12)
    %6 = call @tril(%5) : (tensor<8x8xf32>) -> tensor<8x8xf32> loc(#loc13)
    %c = stablehlo.constant dense<8> : tensor<i64> loc(#loc14)
    %c_0 = stablehlo.constant dense<8> : tensor<i64> loc(#loc14)
    %7:5 = stablehlo.custom_call @lapack_ssyevd_ffi(%6) {mhlo.backend_config = {mode = 86 : ui8, uplo = 76 : ui8}, operand_layouts = [dense<[0, 1]> : tensor<2xindex>], output_operand_aliases = [#stablehlo.output_operand_alias<output_tuple_indices = [0], operand_index = 0, operand_tuple_indices = []>], result_layouts = [dense<[0, 1]> : tensor<2xindex>, dense<0> : tensor<1xindex>, dense<> : tensor<0xindex>, dense<0> : tensor<1xindex>, dense<0> : tensor<1xindex>]} : (tensor<8x8xf32>) -> (tensor<8x8xf32>, tensor<8xf32>, tensor<i32>, tensor<177xf32>, tensor<43xi32>) loc(#loc14)
    %c_1 = stablehlo.constant dense<0> : tensor<i32> loc(#loc14)
    %8 = stablehlo.broadcast_in_dim %c_1, dims = [] : (tensor<i32>) -> tensor<i32> loc(#loc14)
    %9 = stablehlo.compare  EQ, %7#2, %8,  SIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1> loc(#loc14)
    %10 = stablehlo.broadcast_in_dim %9, dims = [] : (tensor<i1>) -> tensor<1x1xi1> loc(#loc14)
    %cst_2 = stablehlo.constant dense<0x7FC00000> : tensor<f32> loc(#loc14)
    %11 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<8x8xf32> loc(#loc14)
    %12 = stablehlo.broadcast_in_dim %10, dims = [0, 1] : (tensor<1x1xi1>) -> tensor<8x8xi1> loc(#loc14)
    %13 = stablehlo.select %12, %7#0, %11 : tensor<8x8xi1>, tensor<8x8xf32> loc(#loc14)
    %14 = stablehlo.broadcast_in_dim %9, dims = [] : (tensor<i1>) -> tensor<1xi1> loc(#loc14)
    %cst_3 = stablehlo.constant dense<0x7FC00000> : tensor<f32> loc(#loc14)
    %15 = stablehlo.broadcast_in_dim %cst_3, dims = [] : (tensor<f32>) -> tensor<8xf32> loc(#loc14)
    %16 = stablehlo.broadcast_in_dim %14, dims = [0] : (tensor<1xi1>) -> tensor<8xi1> loc(#loc14)
    %17 = stablehlo.select %16, %7#1, %15 : tensor<8xi1>, tensor<8xf32> loc(#loc14)
    return %13, %17 : tensor<8x8xf32>, tensor<8xf32> loc(#loc)
  } loc(#loc)
  func.func private @tril(%arg0: tensor<8x8xf32> {mhlo.layout_mode = "default"} loc("jit(<lambda>)/jit(main)/pjit[in_shardings=(UnspecifiedValue,) out_shardings=(UnspecifiedValue,) in_layouts=(None,) out_layouts=(None,) resource_env=None donated_invars=(False,) name=tril keep_unused=False inline=False]"(#loc6))) -> (tensor<8x8xf32> {mhlo.layout_mode = "default"}) {
    %0 = stablehlo.iota dim = 0 : tensor<8x8xi32> loc(#loc15)
    %c = stablehlo.constant dense<0> : tensor<i32> loc(#loc13)
    %1 = stablehlo.broadcast_in_dim %c, dims = [] : (tensor<i32>) -> tensor<8x8xi32> loc(#loc16)
    %2 = stablehlo.add %0, %1 : tensor<8x8xi32> loc(#loc16)
    %3 = stablehlo.iota dim = 1 : tensor<8x8xi32> loc(#loc17)
    %4 = stablehlo.compare  GE, %2, %3,  SIGNED : (tensor<8x8xi32>, tensor<8x8xi32>) -> tensor<8x8xi1> loc(#loc18)
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32> loc(#loc13)
    %5 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<8x8xf32> loc(#loc19)
    %6 = stablehlo.select %4, %arg0, %5 : tensor<8x8xi1>, tensor<8x8xf32> loc(#loc20)
    return %6 : tensor<8x8xf32> loc(#loc13)
  } loc(#loc13)
} loc(#loc)
#loc = loc(unknown)
#loc1 = loc("third_party/py/jax/tests/export_back_compat_test.py":252:26)
#loc2 = loc("third_party/py/jax/tests/export_back_compat_test.py":252:14)
#loc3 = loc("third_party/py/jax/tests/export_back_compat_test.py":254:34)
#loc4 = loc("third_party/py/jax/tests/export_back_compat_test.py":254:15)
#loc5 = loc("third_party/py/jax/tests/export_back_compat_test.py":254:14)
#loc7 = loc("third_party/py/jax/tests/export_back_compat_test.py":260:11)
#loc8 = loc("jit(<lambda>)/jit(main)/iota[dtype=float32 shape=(64,) dimension=0]"(#loc1))
#loc9 = loc("jit(<lambda>)/jit(main)/reshape[new_sizes=(8, 8) dimensions=None]"(#loc2))
#loc10 = loc("jit(<lambda>)/jit(main)/transpose[permutation=(1, 0)]"(#loc3))
#loc11 = loc("jit(<lambda>)/jit(main)/add"(#loc4))
#loc12 = loc("jit(<lambda>)/jit(main)/div"(#loc5))
#loc14 = loc("jit(<lambda>)/jit(main)/eigh[lower=True sort_eigenvalues=True subset_by_index=None]"(#loc7))
#loc15 = loc("jit(<lambda>)/jit(main)/jit(tril)/iota[dtype=int32 shape=(8, 8) dimension=0]"(#loc6))
#loc16 = loc("jit(<lambda>)/jit(main)/jit(tril)/add"(#loc6))
#loc17 = loc("jit(<lambda>)/jit(main)/jit(tril)/iota[dtype=int32 shape=(8, 8) dimension=1]"(#loc6))
#loc18 = loc("jit(<lambda>)/jit(main)/jit(tril)/ge"(#loc6))
#loc19 = loc("jit(<lambda>)/jit(main)/jit(tril)/broadcast_in_dim[shape=(8, 8) broadcast_dimensions=()]"(#loc6))
#loc20 = loc("jit(<lambda>)/jit(main)/jit(tril)/select_n"(#loc6))
""",
    mlir_module_serialized=b"ML\xefR\x01StableHLO_v0.9.0\x00\x01+\x05\x01\x03\x01\x03\x05\x03\x1b\x07\t\x0b\r\x0f\x11\x13\x15\x17\x19\x1b\x1d\x1f\x03\x9e\x02\xff=\x01\xa1\x0f\x13\x17\x0b\x0f\x0b\x07\x0b\x0b\x0f\x0b\x0b\x0b\x0b\x13\x0b\x13\x0f\x0b\x0b\x0f\x13\x13+\x0b\x0f\x0b\x0b\x0b33\x0b\x0f\x0b\x0b\x13\x0f\x0b\x1b\x0f\x0b\x13\x0f\x0b\x0f\x0b\x0f\x0b\x17\x0f\x0b\x17\x13\x0b\x0f\x0b\x17\x0f\x0b\x17\x13\x0b\x17\x13\x0b\x0b\x17S\x0b\x0b\x0b\x0b\x0b\x0b\x0b\x0b\x0b\x1b\x13\x13\x03_\x0b\x0b/\x0b\x0b\x0f\x0b\x0bO\x0b\x13\x1b\x0b\x1b\x0b\x0b\x0b\x13\x0b\x0b\x0f\x1f\x0f\x0f\x0b\x1fO\x1f/\x0b\x0b\x0b\x0b\x1b\x0b\x0f\x0b\x0f\x0f\x0f\x17\x1f\x0f\x0b\x1fO/\x01\x05\x0b\x0f\x039\x17\x0f\x07\x07\x0f\x07\x13\x17\x07\x0f\x07\x17\x13\x07\x17\x17\x13\x17\x17\x13\x13\x13\x13\x0f\x17\x13\x13\x13\x02f\t\x1d\x83\x85\x03\x03\x11\xcd\x17\x07\x12\x047\x05!\x1d?\x05\x05#\x1f\x05%\x05'\x11\x03\x05\x05)\x05+\x05-\x05/\x03\x03\x1f\xc9\x051\x03\x03\x0b\xcb\x1dE\x05\x053\x055\x1d{}\x03\x03\x0b\xd9\x03\x03\x0b\xf9\x03\t135\x137\x13\x0f9\x057\x11\x01\x00\x059\x05;\x05=\x03\x0b\x15\xa7\x17\xb3\x19\xb5\x0f\xbf\x1b\xc1\x03\x0b\x15\xab\x17\xc5\x19\xab\x0f\xad\x1b\xc7\x05?\x1dC\x05\x05A\x05C\x03\x03\x1f\xcf\x1dK\x05\x05E\x03\x05%\xaf'\xd1\x1dQ\x05\x05G\x03\x03\x0b\xd3\x1dW\x05\x05I\x1d[\x05\x05K\x1d_a\x05M\x17\x07\xf2\x035\x1deg\x05O\x17\x07\xf2\x03\x1d\x03\x03k\xd5\x05Q\x1doq\x05S\x17\x07\xfa\x03E\x1duw\x05U\x17\x07\xfa\x03\x1f\x03\x03\x0b\xd7\x05W\x17\x07\xfa\x03\x1d\x03\x03\x81\xad\x05Y\x05[\x17\x07\x12\x04\x17\x03\x13\x89\xdb\x8b\xdd\x8d\xdf\x8f\xa7\x91\xe1\x93\xe3\x95\xed\x97\xef\x99\xf3\x05]\x05_\x05a\x05c\x05e\x05g\x05i\x05k\x05m\x03\x05%\xaf'\xf7\x03\x03\x11\xfb\x03\x03\x11\xfd\x1do\x1dq\x1f/\x11\x00\x00\x00\x00\x00\x00\x00\x00\x03\x01\x1ds\x03\x03\xc3\x1du\t\x07\x1f-!\x00\x00\x00\x00\x00\x00\x00\x00\x01\x00\x00\x00\x00\x00\x00\x00#!\x03\x05\xb7\xbb\r\x05\xa9\xb9\xa1\xa3\x1dw\r\x05\xa9\xbd\xa1\xa3\x1dy\x1d{\x1d}\r\x03\xa1\xa3##\x1d\x7f\x13\t\x01\x1f\r\t\x00\x00\x00\x00\x1f%\x01\x13\t\x05\x07\x05\x1f\x07\t\x00\x00\x00\x00\x1f\x1d!\x01\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x1f\x07\t\x00\x00\x00@\x1f\x17\x11\x08\x00\x00\x00\x00\x00\x00\x00\x0b\x03\x1d\x81\x1d\x83\x05\x01\r\x05\xe5\xe7\xe9\xeb\x1d\x85\x13\x1fV\x1d\x87\x13\x1fL\x03\x03\xb1\x03\x03\xf1\x15\x03\x01\x01\x01\x03\x0b\xb1\xa5\xf5\xa5\xa5\x1f1\x01\x07\x01\x1f\x07\t\x00\x00\xc0\x7f\x1f\x1d!\x00\x00\x00\x00\x00\x00\x00\x00\x01\x00\x00\x00\x00\x00\x00\x00\x1f;\x11\x00\x00\x00\x00\x00\x00\x00\x00\x01\t\x01\x02\x02)\x05!!\x0b)\x01\x0b\x1d\t)\x01\x15\x01)\x03!\x0b)\x05!!\x15\x1b)\x01\t\x13)\x05!!\x0f)\x03\t\t!\x11\x01\x05\x05\x11\x11\x03\x05\x03\x05)\x03\x01\t)\x03\x02\x02\x0b)\x03\x8a\x05\x0b)\x03\xad\x15)\x03\t\x19)\x03\x05\x19)\x03\x01\x19)\x01\x0f)\x05\x05\x05\x0f)\x03\x05\x0f)\x03!\x0f)\x03\x05\t\x04\x86\x04\x05\x01\x11\r/\x07\x03\x01\t\x0b\x11\r;\x07\x039e\x07\x03]\x1d\x03'\x13\x06c\x03\x05\x03\x01\x15\x07mi\x03\x05\x03\x03\r\x06s\x03\x05\x05\x03\x05\x05\x03\ry\x03\x07\x03\x07)\x03\x03\x05\x03\t\x17\x06)\x03\x05\x05\x07\x0b\x19\x07\t\x7f\x03\x05\x03\r\x05\x03\x01+\x03\x17\x05\x03\x01+\x03\x17\x1b\x07\x01\x87\x0b\x05\x11\r)+\x03\x0f\x05\x03\x01!\x03\r\x03\x07\x01\x03\x03\r\x03\x1f\x0f\x07\x01\x9b\x033\x05\x19!\x03\x07\x01\x03\x035\x03#\x05\x03\x01-\x03\x07\x03\x07\x01\x03\x03\x05\x03'\x03\x07\x01\x9d\x03\x1b\x03%\t\x06\x01\x03\x05\x07+\x15)\x03\x07\x01\x03\x037\x03#\x05\x03\x01-\x03\x07\x03\x07\x01\x03\x03\x11\x031\x03\x07\x01\x9f\x039\x03/\t\x06\x01\x03\x11\x075\x173\x11\x04\r\x05-7\x0b\x11\t=\x07\x03\x15+\x03\x05\t\x07\x03A\x1d\x03\x13\x05\x03\t!\x03\r\x03\x07#\x03\x03\x13\x03\x05\r\x06#\x03\x13\x05\x03\x07\x07\x03IG\x03\x13\x0f\x07OM\x03\x1b\x05\t\x0b\x05\x03\tS\x03\x07\x03\x07U\x03\x03\x05\x03\x0f\t\x06Y\x03\x05\x07\r\x01\x11\x11\x04\t\x03\x13\x06\x03\x01\x05\x01\x00J\x1a\x89\x0b\x0b%\x03\x11\x0f\x0b\t\t\x0b!\x11#\x1f/!)!)#\x1f\x19\xa9\x0f99m\x19\x85\x89W\xb3K\x9bM\x9bn\x03\x1b%)9+\x1b\x1f\x1f\x15\x1d\x15+\x13\ri\x1f\x11\x15\x1b\x17\x15\x17\x0f\x11\x15\x11\x19)\x0f\x0b\x11builtin\x00vhlo\x00module\x00broadcast_in_dim_v1\x00constant_v1\x00iota_v1\x00select_v1\x00func_v1\x00add_v1\x00compare_v1\x00return_v1\x00reshape_v1\x00transpose_v1\x00divide_v1\x00call_v1\x00custom_call_v1\x00third_party/py/jax/tests/export_back_compat_test.py\x00value\x00sym_name\x00broadcast_dimensions\x00arg_attrs\x00function_type\x00res_attrs\x00sym_visibility\x00iota_dimension\x00compare_type\x00comparison_direction\x00jax.uses_shape_polymorphism\x00mhlo.num_partitions\x00mhlo.num_replicas\x00jit__lambda_\x00jit(<lambda>)/jit(main)/pjit[in_shardings=(UnspecifiedValue,) out_shardings=(UnspecifiedValue,) in_layouts=(None,) out_layouts=(None,) resource_env=None donated_invars=(False,) name=tril keep_unused=False inline=False]\x00jit(<lambda>)/jit(main)/jit(tril)/iota[dtype=int32 shape=(8, 8) dimension=0]\x00jit(<lambda>)/jit(main)/jit(tril)/add\x00jit(<lambda>)/jit(main)/jit(tril)/iota[dtype=int32 shape=(8, 8) dimension=1]\x00jit(<lambda>)/jit(main)/jit(tril)/ge\x00jit(<lambda>)/jit(main)/jit(tril)/broadcast_in_dim[shape=(8, 8) broadcast_dimensions=()]\x00jit(<lambda>)/jit(main)/jit(tril)/select_n\x00jit(<lambda>)/jit(main)/iota[dtype=float32 shape=(64,) dimension=0]\x00jit(<lambda>)/jit(main)/reshape[new_sizes=(8, 8) dimensions=None]\x00permutation\x00jit(<lambda>)/jit(main)/transpose[permutation=(1, 0)]\x00jit(<lambda>)/jit(main)/add\x00jit(<lambda>)/jit(main)/div\x00callee\x00jit(<lambda>)/jit(main)/eigh[lower=True sort_eigenvalues=True subset_by_index=None]\x00api_version\x00backend_config\x00call_target_name\x00called_computations\x00has_side_effect\x00mhlo.backend_config\x00operand_layouts\x00output_operand_aliases\x00result_layouts\x00mhlo.layout_mode\x00default\x00jax.result_info\x00tril\x00[0]\x00[1]\x00main\x00public\x00private\x00\x00lapack_ssyevd_ffi\x00mode\x00uplo\x00",
    xla_call_module_version=9,
    nr_devices=1,
)  # End paste


# Pasted from the test output (see export_back_compat_test_util.py module docstring)
data_2024_05_31["f64"] = dict(
    testdata_version=1,
    platform='cpu',
    custom_call_targets=['lapack_dsyevd_ffi'],
    serialized_date=datetime.date(2024, 5, 31),
    inputs=(),
    expected_outputs=(array([[-6.1857700048412056e-01,  2.4081403770912022e-01,
         3.5662489253627483e-01, -6.3034019033669797e-01,
         1.0043483479985752e-16, -2.8842036081919542e-02,
         7.7164692943283169e-25, -1.8446994643771725e-01],
       [-4.7070881487314614e-01,  4.7473787464450845e-01,
        -4.8036836210243367e-01,  4.3802686872516400e-01,
         1.7961797619639258e-01,  8.3080980076741355e-03,
         2.1415294457221756e-01, -2.2856669794666584e-01],
       [-3.2284062926217072e-01, -5.4336490915553370e-01,
         2.2181041859724990e-01,  2.9947877954402297e-01,
        -3.6491813600134632e-01,  3.2867679819727436e-01,
         3.8223299448843473e-01, -2.7266344945561438e-01],
       [-1.7497244365119530e-01, -8.9251550609769414e-02,
        -6.3518515114898394e-02,  1.9162997359209971e-01,
        -2.2087281326110139e-01,  5.9957027043505064e-02,
        -8.7632498908241274e-01, -3.1676020096456303e-01],
       [-2.7104258040220038e-02, -3.3772873786627672e-01,
         2.5901386593721748e-01,  1.7032650752287815e-01,
         6.7521217612940332e-01, -4.5036136532965476e-01,
        -1.2279030059078447e-02, -3.6085695247351163e-01],
       [ 1.2076392757075530e-01, -3.3834734096469254e-01,
        -6.5506827461665540e-01, -5.0472498521116749e-01,
         6.9987430903492118e-02,  1.0595648906599275e-01,
         8.3443844143082022e-02, -4.0495370398246017e-01],
       [ 2.6863211318173097e-01,  2.2958613191407318e-01,
         6.3952843755683941e-02,  1.8776775771084137e-02,
        -5.3523731432241317e-01, -5.9199531677602002e-01,
         1.7916671834524248e-01, -4.4905045549140887e-01],
       [ 4.1650029879270661e-01,  3.6355449432857079e-01,
         2.9755313100756142e-01,  1.6826270392615944e-02,
         1.9621068035557282e-01,  5.6830030587314817e-01,
         2.9607517592514246e-02, -4.9314720700035747e-01]]), array([-2.4598804776133626e+01, -4.6567755957874661e-14,
       -1.9932120610662194e-14, -5.7323356091157378e-15,
       -4.5459724251334835e-16,  4.0479851042511616e-14,
        9.2325194924982089e-14,  2.7659880477613365e+02])),
    mlir_module_text=r"""
#loc6 = loc("third_party/py/jax/tests/export_back_compat_test.py":260:27)
#loc13 = loc("jit(<lambda>)/jit(main)/pjit[in_shardings=(UnspecifiedValue,) out_shardings=(UnspecifiedValue,) in_layouts=(None,) out_layouts=(None,) resource_env=None donated_invars=(False,) name=tril keep_unused=False inline=False]"(#loc6))
module @jit__lambda_ attributes {jax.uses_shape_polymorphism = false, mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} {
  func.func public @main() -> (tensor<8x8xf64> {jax.result_info = "[0]", mhlo.layout_mode = "default"}, tensor<8xf64> {jax.result_info = "[1]", mhlo.layout_mode = "default"}) {
    %0 = stablehlo.iota dim = 0 : tensor<64xf64> loc(#loc8)
    %1 = stablehlo.reshape %0 : (tensor<64xf64>) -> tensor<8x8xf64> loc(#loc9)
    %2 = stablehlo.transpose %1, dims = [1, 0] : (tensor<8x8xf64>) -> tensor<8x8xf64> loc(#loc10)
    %3 = stablehlo.add %1, %2 : tensor<8x8xf64> loc(#loc11)
    %cst = stablehlo.constant dense<2.000000e+00> : tensor<f64> loc(#loc)
    %4 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f64>) -> tensor<8x8xf64> loc(#loc12)
    %5 = stablehlo.divide %3, %4 : tensor<8x8xf64> loc(#loc12)
    %6 = call @tril(%5) : (tensor<8x8xf64>) -> tensor<8x8xf64> loc(#loc13)
    %c = stablehlo.constant dense<8> : tensor<i64> loc(#loc14)
    %c_0 = stablehlo.constant dense<8> : tensor<i64> loc(#loc14)
    %7:5 = stablehlo.custom_call @lapack_dsyevd_ffi(%6) {mhlo.backend_config = {mode = 86 : ui8, uplo = 76 : ui8}, operand_layouts = [dense<[0, 1]> : tensor<2xindex>], output_operand_aliases = [#stablehlo.output_operand_alias<output_tuple_indices = [0], operand_index = 0, operand_tuple_indices = []>], result_layouts = [dense<[0, 1]> : tensor<2xindex>, dense<0> : tensor<1xindex>, dense<> : tensor<0xindex>, dense<0> : tensor<1xindex>, dense<0> : tensor<1xindex>]} : (tensor<8x8xf64>) -> (tensor<8x8xf64>, tensor<8xf64>, tensor<i32>, tensor<177xf64>, tensor<43xi32>) loc(#loc14)
    %c_1 = stablehlo.constant dense<0> : tensor<i32> loc(#loc14)
    %8 = stablehlo.broadcast_in_dim %c_1, dims = [] : (tensor<i32>) -> tensor<i32> loc(#loc14)
    %9 = stablehlo.compare  EQ, %7#2, %8,  SIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1> loc(#loc14)
    %10 = stablehlo.broadcast_in_dim %9, dims = [] : (tensor<i1>) -> tensor<1x1xi1> loc(#loc14)
    %cst_2 = stablehlo.constant dense<0x7FF8000000000000> : tensor<f64> loc(#loc14)
    %11 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f64>) -> tensor<8x8xf64> loc(#loc14)
    %12 = stablehlo.broadcast_in_dim %10, dims = [0, 1] : (tensor<1x1xi1>) -> tensor<8x8xi1> loc(#loc14)
    %13 = stablehlo.select %12, %7#0, %11 : tensor<8x8xi1>, tensor<8x8xf64> loc(#loc14)
    %14 = stablehlo.broadcast_in_dim %9, dims = [] : (tensor<i1>) -> tensor<1xi1> loc(#loc14)
    %cst_3 = stablehlo.constant dense<0x7FF8000000000000> : tensor<f64> loc(#loc14)
    %15 = stablehlo.broadcast_in_dim %cst_3, dims = [] : (tensor<f64>) -> tensor<8xf64> loc(#loc14)
    %16 = stablehlo.broadcast_in_dim %14, dims = [0] : (tensor<1xi1>) -> tensor<8xi1> loc(#loc14)
    %17 = stablehlo.select %16, %7#1, %15 : tensor<8xi1>, tensor<8xf64> loc(#loc14)
    return %13, %17 : tensor<8x8xf64>, tensor<8xf64> loc(#loc)
  } loc(#loc)
  func.func private @tril(%arg0: tensor<8x8xf64> {mhlo.layout_mode = "default"} loc("jit(<lambda>)/jit(main)/pjit[in_shardings=(UnspecifiedValue,) out_shardings=(UnspecifiedValue,) in_layouts=(None,) out_layouts=(None,) resource_env=None donated_invars=(False,) name=tril keep_unused=False inline=False]"(#loc6))) -> (tensor<8x8xf64> {mhlo.layout_mode = "default"}) {
    %0 = stablehlo.iota dim = 0 : tensor<8x8xi32> loc(#loc15)
    %c = stablehlo.constant dense<0> : tensor<i32> loc(#loc13)
    %1 = stablehlo.broadcast_in_dim %c, dims = [] : (tensor<i32>) -> tensor<8x8xi32> loc(#loc16)
    %2 = stablehlo.add %0, %1 : tensor<8x8xi32> loc(#loc16)
    %3 = stablehlo.iota dim = 1 : tensor<8x8xi32> loc(#loc17)
    %4 = stablehlo.compare  GE, %2, %3,  SIGNED : (tensor<8x8xi32>, tensor<8x8xi32>) -> tensor<8x8xi1> loc(#loc18)
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f64> loc(#loc13)
    %5 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f64>) -> tensor<8x8xf64> loc(#loc19)
    %6 = stablehlo.select %4, %arg0, %5 : tensor<8x8xi1>, tensor<8x8xf64> loc(#loc20)
    return %6 : tensor<8x8xf64> loc(#loc13)
  } loc(#loc13)
} loc(#loc)
#loc = loc(unknown)
#loc1 = loc("third_party/py/jax/tests/export_back_compat_test.py":252:26)
#loc2 = loc("third_party/py/jax/tests/export_back_compat_test.py":252:14)
#loc3 = loc("third_party/py/jax/tests/export_back_compat_test.py":254:34)
#loc4 = loc("third_party/py/jax/tests/export_back_compat_test.py":254:15)
#loc5 = loc("third_party/py/jax/tests/export_back_compat_test.py":254:14)
#loc7 = loc("third_party/py/jax/tests/export_back_compat_test.py":260:11)
#loc8 = loc("jit(<lambda>)/jit(main)/iota[dtype=float64 shape=(64,) dimension=0]"(#loc1))
#loc9 = loc("jit(<lambda>)/jit(main)/reshape[new_sizes=(8, 8) dimensions=None]"(#loc2))
#loc10 = loc("jit(<lambda>)/jit(main)/transpose[permutation=(1, 0)]"(#loc3))
#loc11 = loc("jit(<lambda>)/jit(main)/add"(#loc4))
#loc12 = loc("jit(<lambda>)/jit(main)/div"(#loc5))
#loc14 = loc("jit(<lambda>)/jit(main)/eigh[lower=True sort_eigenvalues=True subset_by_index=None]"(#loc7))
#loc15 = loc("jit(<lambda>)/jit(main)/jit(tril)/iota[dtype=int32 shape=(8, 8) dimension=0]"(#loc6))
#loc16 = loc("jit(<lambda>)/jit(main)/jit(tril)/add"(#loc6))
#loc17 = loc("jit(<lambda>)/jit(main)/jit(tril)/iota[dtype=int32 shape=(8, 8) dimension=1]"(#loc6))
#loc18 = loc("jit(<lambda>)/jit(main)/jit(tril)/ge"(#loc6))
#loc19 = loc("jit(<lambda>)/jit(main)/jit(tril)/broadcast_in_dim[shape=(8, 8) broadcast_dimensions=()]"(#loc6))
#loc20 = loc("jit(<lambda>)/jit(main)/jit(tril)/select_n"(#loc6))
""",
    mlir_module_serialized=b"ML\xefR\x01StableHLO_v0.9.0\x00\x01+\x05\x01\x03\x01\x03\x05\x03\x1b\x07\t\x0b\r\x0f\x11\x13\x15\x17\x19\x1b\x1d\x1f\x03\x9e\x02\xff=\x01\xa1\x0f\x13\x17\x0b\x0f\x0b\x07\x0b\x0b\x0f\x0b\x0b\x0b\x0b\x13\x0b\x13\x0f\x0b\x0b\x0f\x13\x13+\x0b\x0f\x0b\x0b\x0b33\x0b\x0f\x0b\x0b\x13\x0f\x0b\x1b\x0f\x0b\x13\x0f\x0b\x0f\x0b\x0f\x0b\x17\x0f\x0b\x17\x13\x0b\x0f\x0b\x17\x0f\x0b\x17\x13\x0b\x17\x13\x0b\x0b\x17S\x0b\x0b\x0b\x0b\x0b\x0b\x0b\x0b\x0b\x1b\x13\x13\x03_\x0b\x0b/\x0b\x0b\x0f\x0b\x0bO\x0b\x13\x1b\x0b\x1b\x0b\x0b\x0b\x13\x0b\x0b\x0f\x1f\x0f\x0f\x0b/O//\x0b\x0b\x0b\x0b\x1b\x0b\x0f\x0b\x0f\x0f\x0f\x17\x1f\x0f\x0b/O/\x01\x05\x0b\x0f\x039\x17\x0f\x07\x07\x0f\x07\x13\x17\x07\x0f\x07\x17\x13\x07\x17\x17\x13\x17\x17\x13\x13\x13\x13\x0f\x17\x13\x13\x13\x02\x96\t\x1d\x83\x85\x03\x03\x11\xcd\x17\x07\x12\x047\x05!\x1d?\x05\x05#\x1f\x05%\x05'\x11\x03\x05\x05)\x05+\x05-\x05/\x03\x03\x1f\xc9\x051\x03\x03\x0b\xcb\x1dE\x05\x053\x055\x1d{}\x03\x03\x0b\xd9\x03\x03\x0b\xf9\x03\t135\x137\x13\x0f9\x057\x11\x01\x00\x059\x05;\x05=\x03\x0b\x15\xa7\x17\xb3\x19\xb5\x0f\xbf\x1b\xc1\x03\x0b\x15\xab\x17\xc5\x19\xab\x0f\xad\x1b\xc7\x05?\x1dC\x05\x05A\x05C\x03\x03\x1f\xcf\x1dK\x05\x05E\x03\x05%\xaf'\xd1\x1dQ\x05\x05G\x03\x03\x0b\xd3\x1dW\x05\x05I\x1d[\x05\x05K\x1d_a\x05M\x17\x07\xf2\x035\x1deg\x05O\x17\x07\xf2\x03\x1d\x03\x03k\xd5\x05Q\x1doq\x05S\x17\x07\xfa\x03E\x1duw\x05U\x17\x07\xfa\x03\x1f\x03\x03\x0b\xd7\x05W\x17\x07\xfa\x03\x1d\x03\x03\x81\xad\x05Y\x05[\x17\x07\x12\x04\x17\x03\x13\x89\xdb\x8b\xdd\x8d\xdf\x8f\xa7\x91\xe1\x93\xe3\x95\xed\x97\xef\x99\xf3\x05]\x05_\x05a\x05c\x05e\x05g\x05i\x05k\x05m\x03\x05%\xaf'\xf7\x03\x03\x11\xfb\x03\x03\x11\xfd\x1do\x1dq\x1f/\x11\x00\x00\x00\x00\x00\x00\x00\x00\x03\x01\x1ds\x03\x03\xc3\x1du\t\x07\x1f-!\x00\x00\x00\x00\x00\x00\x00\x00\x01\x00\x00\x00\x00\x00\x00\x00#!\x03\x05\xb7\xbb\r\x05\xa9\xb9\xa1\xa3\x1dw\r\x05\xa9\xbd\xa1\xa3\x1dy\x1d{\x1d}\r\x03\xa1\xa3##\x1d\x7f\x13\t\x01\x1f\r\t\x00\x00\x00\x00\x1f%\x01\x13\t\x05\x07\x05\x1f\x07\x11\x00\x00\x00\x00\x00\x00\x00\x00\x1f\x1d!\x01\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x1f\x07\x11\x00\x00\x00\x00\x00\x00\x00@\x1f\x17\x11\x08\x00\x00\x00\x00\x00\x00\x00\x0b\x03\x1d\x81\x1d\x83\x05\x01\r\x05\xe5\xe7\xe9\xeb\x1d\x85\x13\x1fV\x1d\x87\x13\x1fL\x03\x03\xb1\x03\x03\xf1\x15\x03\x01\x01\x01\x03\x0b\xb1\xa5\xf5\xa5\xa5\x1f1\x01\x07\x01\x1f\x07\x11\x00\x00\x00\x00\x00\x00\xf8\x7f\x1f\x1d!\x00\x00\x00\x00\x00\x00\x00\x00\x01\x00\x00\x00\x00\x00\x00\x00\x1f;\x11\x00\x00\x00\x00\x00\x00\x00\x00\x01\t\x01\x02\x02)\x05!!\x0b)\x01\x0b\x1d\x0b)\x01\x15\x01)\x03!\x0b)\x05!!\x15\x1b)\x01\t\x13)\x05!!\x0f)\x03\t\t!\x11\x01\x05\x05\x11\x11\x03\x05\x03\x05)\x03\x01\t)\x03\x02\x02\x0b)\x03\x8a\x05\x0b)\x03\xad\x15)\x03\t\x19)\x03\x05\x19)\x03\x01\x19)\x01\x0f)\x05\x05\x05\x0f)\x03\x05\x0f)\x03!\x0f)\x03\x05\t\x04\x86\x04\x05\x01\x11\r/\x07\x03\x01\t\x0b\x11\r;\x07\x039e\x07\x03]\x1d\x03'\x13\x06c\x03\x05\x03\x01\x15\x07mi\x03\x05\x03\x03\r\x06s\x03\x05\x05\x03\x05\x05\x03\ry\x03\x07\x03\x07)\x03\x03\x05\x03\t\x17\x06)\x03\x05\x05\x07\x0b\x19\x07\t\x7f\x03\x05\x03\r\x05\x03\x01+\x03\x17\x05\x03\x01+\x03\x17\x1b\x07\x01\x87\x0b\x05\x11\r)+\x03\x0f\x05\x03\x01!\x03\r\x03\x07\x01\x03\x03\r\x03\x1f\x0f\x07\x01\x9b\x033\x05\x19!\x03\x07\x01\x03\x035\x03#\x05\x03\x01-\x03\x07\x03\x07\x01\x03\x03\x05\x03'\x03\x07\x01\x9d\x03\x1b\x03%\t\x06\x01\x03\x05\x07+\x15)\x03\x07\x01\x03\x037\x03#\x05\x03\x01-\x03\x07\x03\x07\x01\x03\x03\x11\x031\x03\x07\x01\x9f\x039\x03/\t\x06\x01\x03\x11\x075\x173\x11\x04\r\x05-7\x0b\x11\t=\x07\x03\x15+\x03\x05\t\x07\x03A\x1d\x03\x13\x05\x03\t!\x03\r\x03\x07#\x03\x03\x13\x03\x05\r\x06#\x03\x13\x05\x03\x07\x07\x03IG\x03\x13\x0f\x07OM\x03\x1b\x05\t\x0b\x05\x03\tS\x03\x07\x03\x07U\x03\x03\x05\x03\x0f\t\x06Y\x03\x05\x07\r\x01\x11\x11\x04\t\x03\x13\x06\x03\x01\x05\x01\x00J\x1a\x89\x0b\x0b%\x03\x11\x0f\x0b\t\t\x0b!\x11#\x1f/!)!)#\x1f\x19\xa9\x0f99m\x19\x85\x89W\xb3K\x9bM\x9bn\x03\x1b%)9+\x1b\x1f\x1f\x15\x1d\x15+\x13\ri\x1f\x11\x15\x1b\x17\x15\x17\x0f\x11\x15\x11\x19)\x0f\x0b\x11builtin\x00vhlo\x00module\x00broadcast_in_dim_v1\x00constant_v1\x00iota_v1\x00select_v1\x00func_v1\x00add_v1\x00compare_v1\x00return_v1\x00reshape_v1\x00transpose_v1\x00divide_v1\x00call_v1\x00custom_call_v1\x00third_party/py/jax/tests/export_back_compat_test.py\x00value\x00sym_name\x00broadcast_dimensions\x00arg_attrs\x00function_type\x00res_attrs\x00sym_visibility\x00iota_dimension\x00compare_type\x00comparison_direction\x00jax.uses_shape_polymorphism\x00mhlo.num_partitions\x00mhlo.num_replicas\x00jit__lambda_\x00jit(<lambda>)/jit(main)/pjit[in_shardings=(UnspecifiedValue,) out_shardings=(UnspecifiedValue,) in_layouts=(None,) out_layouts=(None,) resource_env=None donated_invars=(False,) name=tril keep_unused=False inline=False]\x00jit(<lambda>)/jit(main)/jit(tril)/iota[dtype=int32 shape=(8, 8) dimension=0]\x00jit(<lambda>)/jit(main)/jit(tril)/add\x00jit(<lambda>)/jit(main)/jit(tril)/iota[dtype=int32 shape=(8, 8) dimension=1]\x00jit(<lambda>)/jit(main)/jit(tril)/ge\x00jit(<lambda>)/jit(main)/jit(tril)/broadcast_in_dim[shape=(8, 8) broadcast_dimensions=()]\x00jit(<lambda>)/jit(main)/jit(tril)/select_n\x00jit(<lambda>)/jit(main)/iota[dtype=float64 shape=(64,) dimension=0]\x00jit(<lambda>)/jit(main)/reshape[new_sizes=(8, 8) dimensions=None]\x00permutation\x00jit(<lambda>)/jit(main)/transpose[permutation=(1, 0)]\x00jit(<lambda>)/jit(main)/add\x00jit(<lambda>)/jit(main)/div\x00callee\x00jit(<lambda>)/jit(main)/eigh[lower=True sort_eigenvalues=True subset_by_index=None]\x00api_version\x00backend_config\x00call_target_name\x00called_computations\x00has_side_effect\x00mhlo.backend_config\x00operand_layouts\x00output_operand_aliases\x00result_layouts\x00mhlo.layout_mode\x00default\x00jax.result_info\x00tril\x00[0]\x00[1]\x00main\x00public\x00private\x00\x00lapack_dsyevd_ffi\x00mode\x00uplo\x00",
    xla_call_module_version=9,
    nr_devices=1,
)  # End paste

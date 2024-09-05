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

# ruff: noqa

import datetime
from numpy import array, float32, complex64

data_2024_09_03 = {}


# Pasted from the test output (see export_back_compat_test_util.py module docstring)
data_2024_09_03["c128"] = dict(
    testdata_version=1,
    platform='cpu',
    custom_call_targets=['lapack_zhetrd'],
    serialized_date=datetime.date(2024, 9, 4),
    inputs=(),
    expected_outputs=(array([[[-1.6782909868280393 +0.j                 ,
         -0.44670237330570184+4.847000766107959j  ,
          2.05945450900321   -2.2848432268240106j ,
         -1.852046418980849  +1.672382006137275j  ],
        [ 8.516713699516982  +0.j                 ,
         -2.7881860505313174 +0.j                 ,
          0.9238284715039695 -2.3790501284019947j ,
          0.5005102262291599 -1.30066052934836j   ],
        [-0.12132810525381293-0.2963030371159077j ,
         -3.6374350042782893 +0.j                 ,
          0.5605752523031344 +0.j                 ,
         -2.9865099107523174 +0.5492956557924651j ],
        [-0.40379248092949666-0.7813328344426929j ,
         -0.07101654492399719-0.27208840961051617j,
         -7.4654253782049285 +0.j                 ,
         -8.172380353916964  +0.j                 ]],

       [[-3.996403598623405  +0.j                 ,
          0.59408630943699   +2.531609474375295j  ,
         -1.789098034543644  -2.538389274566601j  ,
         -1.291106590337488  +3.1576544511573843j ],
        [10.8950662522622    +0.j                 ,
         -2.8151642043836693 +0.j                 ,
          6.18998567202382   +1.1866537964613415j ,
          3.1900218245393352 +2.7291222716752372j ],
        [-0.3142889671188478 -0.37781876498252764j,
          3.049208563595754  +0.j                 ,
         -2.4383044880335487 +0.j                 ,
          4.075435464493341  -0.6653616942280807j ],
        [ 0.32757687545025194+0.565870910342534j  ,
          0.8177026465997795 -0.15906305615104555j,
          3.3415143060767125 +0.j                 ,
          4.094619408678314  +0.j                 ]]]), array([[-1.6782909868280393, -2.7881860505313174,  0.5605752523031344,
        -8.172380353916964 ],
       [-3.996403598623405 , -2.8151642043836693, -2.4383044880335487,
         4.094619408678314 ]]), array([[ 8.516713699516982 , -3.6374350042782893, -7.4654253782049285],
       [10.8950662522622   ,  3.049208563595754 ,  3.3415143060767125]]), array([[1.0626274644222748+0.06050271598884928j,
        1.834630852474663 +0.18575551495730305j,
        1.981584368497257 +0.19102912741736966j],
       [1.0365789616521406-0.40942548304121656j,
        1.0872592163018966-0.3187050677167622j ,
        1.0458498304770472-0.9989483435319496j ]])),
    mlir_module_text=r"""
#loc1 = loc("third_party/py/jax/tests/export_back_compat_test.py":695:13)
#loc5 = loc("jit(func)/jit(main)/pjit"(#loc1))
module @jit_func attributes {jax.uses_shape_polymorphism = false, mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} {
  func.func public @main() -> (tensor<2x4x4xcomplex<f64>> {jax.result_info = "[0]", mhlo.layout_mode = "default"}, tensor<2x4xf64> {jax.result_info = "[1]", mhlo.layout_mode = "default"}, tensor<2x3xf64> {jax.result_info = "[2]", mhlo.layout_mode = "default"}, tensor<2x3xcomplex<f64>> {jax.result_info = "[3]", mhlo.layout_mode = "default"}) {
    %cst = stablehlo.constant dense<[[[(-1.6782909868280393,-0.44303325034407437), (-0.44670237330570184,4.8470007661079588), (2.0594545090032099,-2.2848432268240106), (-1.852046418980849,1.6723820061372749)], [(-0.53338018421119981,-0.5152843101202178), (-8.6208093221459947,-1.4723511111926109), (0.92382847150396952,-2.3790501284019947), (0.50051022622915986,-1.30066052934836)], [(0.94535043721506584,2.744088772946665), (-5.9178492824175759,-4.3744650461123786), (1.8341291553102983,-4.8378584827626838), (-2.9865099107523174,0.54929565579246509)], [(3.2517513113853891,7.2792034361133062), (-0.09841002311276037,0.88008791818205689), (-0.035759860211603468,2.4677764344580244), (-3.6133109853094476,-2.2833696560058976)]], [[(-3.996403598623405,2.42308766118121), (0.59408630943699003,2.531609474375295), (-1.789098034543644,-2.538389274566601), (-1.2911065903374881,3.1576544511573843)], [(-0.39853021063902833,4.4607177630985086), (1.0742061295773189,-2.6002112528615386), (6.1899856720238198,1.1866537964613415), (3.1900218245393352,2.7291222716752372)], [(5.2347956435718022,2.8649782894514577), (2.3527586611916762,2.4688953673448575), (-2.317572140163894,4.3609023810820053), (4.0754354644933413,-0.66536169422808067)], [(-6.2237114632988675,-4.9294897244018943), (4.2994486027667103,-1.3300494261380422), (-0.51942958410141249,0.60038999428238982), (0.084516726847668963,-7.2944134049318752)]]]> : tensor<2x4x4xcomplex<f64>> loc(#loc)
    %c = stablehlo.constant dense<4> : tensor<i32> loc(#loc2)
    %c_0 = stablehlo.constant dense<1> : tensor<i32> loc(#loc2)
    %c_1 = stablehlo.constant dense<4> : tensor<i32> loc(#loc2)
    %c_2 = stablehlo.constant dense<2> : tensor<i32> loc(#loc2)
    %c_3 = stablehlo.constant dense<128> : tensor<i32> loc(#loc2)
    %0:6 = stablehlo.custom_call @lapack_zhetrd(%c, %c_0, %c_1, %c_2, %c_3, %cst) {api_version = 2 : i32, operand_layouts = [dense<> : tensor<0xindex>, dense<> : tensor<0xindex>, dense<> : tensor<0xindex>, dense<> : tensor<0xindex>, dense<> : tensor<0xindex>, dense<[1, 2, 0]> : tensor<3xindex>], output_operand_aliases = [#stablehlo.output_operand_alias<output_tuple_indices = [0], operand_index = 5, operand_tuple_indices = []>], result_layouts = [dense<[1, 2, 0]> : tensor<3xindex>, dense<[1, 0]> : tensor<2xindex>, dense<[1, 0]> : tensor<2xindex>, dense<[1, 0]> : tensor<2xindex>, dense<0> : tensor<1xindex>, dense<0> : tensor<1xindex>]} : (tensor<i32>, tensor<i32>, tensor<i32>, tensor<i32>, tensor<i32>, tensor<2x4x4xcomplex<f64>>) -> (tensor<2x4x4xcomplex<f64>>, tensor<2x4xf64>, tensor<2x3xf64>, tensor<2x3xcomplex<f64>>, tensor<2xi32>, tensor<128xcomplex<f64>>) loc(#loc2)
    %c_4 = stablehlo.constant dense<0> : tensor<i32> loc(#loc)
    %1 = stablehlo.broadcast_in_dim %c_4, dims = [] : (tensor<i32>) -> tensor<2xi32> loc(#loc3)
    %2 = stablehlo.compare  EQ, %0#4, %1,  SIGNED : (tensor<2xi32>, tensor<2xi32>) -> tensor<2xi1> loc(#loc3)
    %3 = stablehlo.broadcast_in_dim %2, dims = [0] : (tensor<2xi1>) -> tensor<2x1x1xi1> loc(#loc4)
    %cst_5 = stablehlo.constant dense<(0x7FF8000000000000,0x7FF8000000000000)> : tensor<complex<f64>> loc(#loc)
    %4 = call @_where(%3, %0#0, %cst_5) : (tensor<2x1x1xi1>, tensor<2x4x4xcomplex<f64>>, tensor<complex<f64>>) -> tensor<2x4x4xcomplex<f64>> loc(#loc5)
    %c_6 = stablehlo.constant dense<0> : tensor<i32> loc(#loc)
    %5 = stablehlo.broadcast_in_dim %c_6, dims = [] : (tensor<i32>) -> tensor<2xi32> loc(#loc3)
    %6 = stablehlo.compare  EQ, %0#4, %5,  SIGNED : (tensor<2xi32>, tensor<2xi32>) -> tensor<2xi1> loc(#loc3)
    %7 = stablehlo.broadcast_in_dim %6, dims = [0] : (tensor<2xi1>) -> tensor<2x1xi1> loc(#loc4)
    %cst_7 = stablehlo.constant dense<0x7FF8000000000000> : tensor<f64> loc(#loc)
    %8 = call @_where_0(%7, %0#1, %cst_7) : (tensor<2x1xi1>, tensor<2x4xf64>, tensor<f64>) -> tensor<2x4xf64> loc(#loc5)
    %c_8 = stablehlo.constant dense<0> : tensor<i32> loc(#loc)
    %9 = stablehlo.broadcast_in_dim %c_8, dims = [] : (tensor<i32>) -> tensor<2xi32> loc(#loc3)
    %10 = stablehlo.compare  EQ, %0#4, %9,  SIGNED : (tensor<2xi32>, tensor<2xi32>) -> tensor<2xi1> loc(#loc3)
    %11 = stablehlo.broadcast_in_dim %10, dims = [0] : (tensor<2xi1>) -> tensor<2x1xi1> loc(#loc4)
    %cst_9 = stablehlo.constant dense<0x7FF8000000000000> : tensor<f64> loc(#loc)
    %12 = call @_where_1(%11, %0#2, %cst_9) : (tensor<2x1xi1>, tensor<2x3xf64>, tensor<f64>) -> tensor<2x3xf64> loc(#loc5)
    %c_10 = stablehlo.constant dense<0> : tensor<i32> loc(#loc)
    %13 = stablehlo.broadcast_in_dim %c_10, dims = [] : (tensor<i32>) -> tensor<2xi32> loc(#loc3)
    %14 = stablehlo.compare  EQ, %0#4, %13,  SIGNED : (tensor<2xi32>, tensor<2xi32>) -> tensor<2xi1> loc(#loc3)
    %15 = stablehlo.broadcast_in_dim %14, dims = [0] : (tensor<2xi1>) -> tensor<2x1xi1> loc(#loc4)
    %cst_11 = stablehlo.constant dense<(0x7FF8000000000000,0x7FF8000000000000)> : tensor<complex<f64>> loc(#loc)
    %16 = call @_where_2(%15, %0#3, %cst_11) : (tensor<2x1xi1>, tensor<2x3xcomplex<f64>>, tensor<complex<f64>>) -> tensor<2x3xcomplex<f64>> loc(#loc5)
    return %4, %8, %12, %16 : tensor<2x4x4xcomplex<f64>>, tensor<2x4xf64>, tensor<2x3xf64>, tensor<2x3xcomplex<f64>> loc(#loc)
  } loc(#loc)
  func.func private @_where(%arg0: tensor<2x1x1xi1> {mhlo.layout_mode = "default"} loc("jit(func)/jit(main)/pjit"(#loc1)), %arg1: tensor<2x4x4xcomplex<f64>> {mhlo.layout_mode = "default"} loc("jit(func)/jit(main)/pjit"(#loc1)), %arg2: tensor<complex<f64>> {mhlo.layout_mode = "default"} loc("jit(func)/jit(main)/pjit"(#loc1))) -> (tensor<2x4x4xcomplex<f64>> {mhlo.layout_mode = "default"}) {
    %0 = stablehlo.broadcast_in_dim %arg0, dims = [0, 1, 2] : (tensor<2x1x1xi1>) -> tensor<2x4x4xi1> loc(#loc6)
    %1 = stablehlo.broadcast_in_dim %arg2, dims = [] : (tensor<complex<f64>>) -> tensor<2x4x4xcomplex<f64>> loc(#loc6)
    %2 = stablehlo.select %0, %arg1, %1 : tensor<2x4x4xi1>, tensor<2x4x4xcomplex<f64>> loc(#loc7)
    return %2 : tensor<2x4x4xcomplex<f64>> loc(#loc5)
  } loc(#loc5)
  func.func private @_where_0(%arg0: tensor<2x1xi1> {mhlo.layout_mode = "default"} loc("jit(func)/jit(main)/pjit"(#loc1)), %arg1: tensor<2x4xf64> {mhlo.layout_mode = "default"} loc("jit(func)/jit(main)/pjit"(#loc1)), %arg2: tensor<f64> {mhlo.layout_mode = "default"} loc("jit(func)/jit(main)/pjit"(#loc1))) -> (tensor<2x4xf64> {mhlo.layout_mode = "default"}) {
    %0 = stablehlo.broadcast_in_dim %arg0, dims = [0, 1] : (tensor<2x1xi1>) -> tensor<2x4xi1> loc(#loc6)
    %1 = stablehlo.broadcast_in_dim %arg2, dims = [] : (tensor<f64>) -> tensor<2x4xf64> loc(#loc6)
    %2 = stablehlo.select %0, %arg1, %1 : tensor<2x4xi1>, tensor<2x4xf64> loc(#loc7)
    return %2 : tensor<2x4xf64> loc(#loc5)
  } loc(#loc5)
  func.func private @_where_1(%arg0: tensor<2x1xi1> {mhlo.layout_mode = "default"} loc("jit(func)/jit(main)/pjit"(#loc1)), %arg1: tensor<2x3xf64> {mhlo.layout_mode = "default"} loc("jit(func)/jit(main)/pjit"(#loc1)), %arg2: tensor<f64> {mhlo.layout_mode = "default"} loc("jit(func)/jit(main)/pjit"(#loc1))) -> (tensor<2x3xf64> {mhlo.layout_mode = "default"}) {
    %0 = stablehlo.broadcast_in_dim %arg0, dims = [0, 1] : (tensor<2x1xi1>) -> tensor<2x3xi1> loc(#loc6)
    %1 = stablehlo.broadcast_in_dim %arg2, dims = [] : (tensor<f64>) -> tensor<2x3xf64> loc(#loc6)
    %2 = stablehlo.select %0, %arg1, %1 : tensor<2x3xi1>, tensor<2x3xf64> loc(#loc7)
    return %2 : tensor<2x3xf64> loc(#loc5)
  } loc(#loc5)
  func.func private @_where_2(%arg0: tensor<2x1xi1> {mhlo.layout_mode = "default"} loc("jit(func)/jit(main)/pjit"(#loc1)), %arg1: tensor<2x3xcomplex<f64>> {mhlo.layout_mode = "default"} loc("jit(func)/jit(main)/pjit"(#loc1)), %arg2: tensor<complex<f64>> {mhlo.layout_mode = "default"} loc("jit(func)/jit(main)/pjit"(#loc1))) -> (tensor<2x3xcomplex<f64>> {mhlo.layout_mode = "default"}) {
    %0 = stablehlo.broadcast_in_dim %arg0, dims = [0, 1] : (tensor<2x1xi1>) -> tensor<2x3xi1> loc(#loc6)
    %1 = stablehlo.broadcast_in_dim %arg2, dims = [] : (tensor<complex<f64>>) -> tensor<2x3xcomplex<f64>> loc(#loc6)
    %2 = stablehlo.select %0, %arg1, %1 : tensor<2x3xi1>, tensor<2x3xcomplex<f64>> loc(#loc7)
    return %2 : tensor<2x3xcomplex<f64>> loc(#loc5)
  } loc(#loc5)
} loc(#loc)
#loc = loc(unknown)
#loc2 = loc("jit(func)/jit(main)/tridiagonal"(#loc1))
#loc3 = loc("jit(func)/jit(main)/eq"(#loc1))
#loc4 = loc("jit(func)/jit(main)/broadcast_in_dim"(#loc1))
#loc6 = loc("jit(func)/jit(main)/jit(_where)/broadcast_in_dim"(#loc1))
#loc7 = loc("jit(func)/jit(main)/jit(_where)/select_n"(#loc1))
""",
    mlir_module_serialized=b'ML\xefR\rStableHLO_v1.3.0\x00\x01#\x05\x01\x05\x13\x01\x03\x0b\x03\x11\x0f\x13\x17\x1b\x1f#\'+\x03\xf7\x99I\x01-\x0f\x07\x0f\x0f\x17\x0f\x0f\x0f\x0f#\x0b\x0f\x0b\x0b\x0b\x0b\x0b\x0b\x0b\x0b\x0b\x0b\x03m\x0f\x0b\x0b\x0f\x0b\x17\x13\x0f\x0b\x1f\x0b\x0b/OO\x0b\x0b\x0b\x0b\x0b\x1fo/O/\x0b\x1b\x1b\x0b\x1b\x0b\x1b\x0b\x1b\x0b\x0b\x0b\x0b\x0b\x0b\x0bo&\x10\x1f\x1f\x1f\x0b\x0b\x0b\x0b#\x0f\x17#\x01\x05\x0b\x0f\x03E\x0f\x1b\x17\x17\x17\x17\x0f\x0f\x07\x13\x0b\x07\x07\x07\x13\x1b\x17\x07\x1f\x1f\x1f\x1f\x1f\x13\x13\x17\x1b\x13\x17\x13\x13\x13\x13\x13\x02\x12\x10\x1d\x1f\t\x1f\x1d#\t\x1d)\t\x17!\xde\n\x1b\x1d\'\t\x1d%\t\x1d+\t\x11\x03\x05\x03\x07\x15\x17\x19\x11\x1b\x11\x05\x17\x11\x01\x00\x05\x19\x05\x1b\x05\x1d\x05\x1f\x05!\x05#\x05%\x05\'\x05)\x05+\x1f5\x01\x1d-\x1d/\x1f?\x01\x1d1\x03\x07999\r\x03/1\x03\x039\x1d3\x1f\x05\t\x00\x00\x00\x00\t\x07\x07\x01\x1fG\x11\x00\x00\x00\x00\x00\x00\x00\x00\x1f3!\x00\x00\x00\x00\x00\x00\x00\x00\x01\x00\x00\x00\x00\x00\x00\x00\x1fC!\x01\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x03\x01\x1d5\x1d7\x1d9\x1d;\x1f\x05\t\x04\x00\x00\x00\x1fA1\x01\x00\x00\x00\x00\x00\x00\x00\x02\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x1fE\x11\x00\x00\x00\x00\x00\x00\x00\x00\x1f\x11!\x00\x00\x00\x00\x00\x00\xf8\x7f\x00\x00\x00\x00\x00\x00\xf8\x7f\x1f\x13\x11\x00\x00\x00\x00\x00\x00\xf8\x7f#)\x03\tcgko\r\x055e/1\x1d=\r\x055i/1\x1d?\r\x055m/1\x1dA\r\x055q/1\x1dC\x1dE\x1dG#+#-#/#1\x1f;1\x00\x00\x00\x00\x00\x00\x00\x00\x01\x00\x00\x00\x00\x00\x00\x00\x02\x00\x00\x00\x00\x00\x00\x00\x1f\x07\x02\x08d\x91Y\xa6G\xda\xfa\xbf$-Q"\xa8Z\xdc\xbfL0\x19\x8d\xc5\x96\xdc\xbf\x86{8+Tc\x13@\xf0%\x1eI\xc3y\x00@\xe4\x91\xbd\xe2[G\x02\xc0\x85%\x03m\xfb\xa1\xfd\xbf\x9atl\xa2\x13\xc2\xfa?\x9c\xb0\xf0Qs\x11\xe1\xbf\xd8v\x83\x855}\xe0\xbf\x84V/\xb8\xda=!\xc0\n\xd3\xec\t\xc0\x8e\xf7\xbf\x98$\x07\xba\x00\x90\xed?\xd5?\x08oK\x08\x03\xc0>\xf8\x9e\x05.\x04\xe0?\xf2\xfcKj\x81\xcf\xf4\xbf\xe4"c\x8fO@\xee?y\x03\x89\xd0\xe4\xf3\x05@\xee\x8f\xaa\xae\xe0\xab\x17\xc0\xf20\xda\xc3s\x7f\x11\xc0V*+\xd0\x97X\xfd?P\x91\xf8\x92\xf7Y\x13\xc0\x7f\xe3\xdeN_\xe4\x07\xc0\x14\xd5\xae{\xd4\x93\xe1?\xbc\x00\t1\x96\x03\n@`&l\x81\xe7\x1d\x1d@X/\xde6f1\xb9\xbf\x06KF#\xae)\xec?\xcd\x9a<\xcc\x1dO\xa2\xbf\x91\xb1>\x92\x01\xbe\x03@\xf2s\x01\x97\x0f\xe8\x0c\xc0\xf5\xcaiOWD\x02\xc0F\xa2-s\xa2\xf8\x0f\xc0X\xea\xa0\xc8{b\x03@\x0b\x10\xc1J\xc1\x02\xe3?2|\xd5w\xbc@\x04@\xca>\xbbB%\xa0\xfc\xbf\xe8>6\t\x9fN\x04\xc0\xafdRb_\xa8\xf4\xbf\x80Q>V\xe0B\t@UhJ\xdb\x84\x81\xd9\xbf\t\xc7\xb4e\xc6\xd7\x11@<(;\xc4\xf2/\xf1?\x1a\xda\xad\x8e;\xcd\x04\xc0\x1c4\xa0\x9a\x8b\xc2\x18@z\x9c\xf7\xb0\x88\xfc\xf2?\xaea\x8f)*\x85\t@\x00\x0b\xbd\x0e>\xd5\x05@b\x89\xe9Dn\xf0\x14@a\x8d\xc7\xbcy\xeb\x06@\x8a\x97\t"s\xd2\x02@\xc2\xef\xdf6L\xc0\x03@J\xff Cc\x8a\x02\xc0\xd7.\xcfd\x90q\x11@s\xd4S\xf4>M\x10@t\x10\x97\x9b\xa4J\xe5\xbf\x8eo*\x9e\x14\xe5\x18\xc0\xc5\x18\x81\'\xcc\xb7\x13\xc0\x19\xdd\x8e\xa7\xa22\x11@-95\xe8\xe1G\xf5\xbfZK\x89\xca*\x9f\xe0\xbfR;\xc9\x13e6\xe3?\x7f\x94\xc6a\xe3\xa2\xb5?\xe2\xbe&\xb5z-\x1d\xc0\x1f\x05\t\x01\x00\x00\x00\x1f\x05\t\x02\x00\x00\x00\x1f\x05\t\x80\x00\x00\x00\x0b\x05\x1dI\x1dK\x05\x01\x03\r33333W\x03\x03\x95\x15\x03\x01\x15\x01\x03\rWIIIYY\x01\t\x01\x02\x02)\x01\')\x07\t\x11\x11\x19)\x05\t\x05\x15)\x05\t\x11\x1b)\x05\t\r\x1b)\x05\t\r\x19)\x01\x19)\x01\x1b\x01)\x03\t\'\x03\x1b\x0b\x1d\x13)\x03\t\x15)\x07\t\x05\x05\x15)\x05\t\r\x15\x1b\x11\x01\t\x07\x0b\r\x0f\x11\x07#\x07\x11\x03\x07\x11\x07\t\x0b\x13\x03\x0b\x11\x07\t\r\x13\x03\r\x11\x07\t\x0f\x11\x03\x0f)\x03\t\x1d)\x03\x01\x1d)\x05\t\x11\x15)\x07\t\x11\x11\x15)\x03\r\x1d)\x03\x02\x04\x19)\x03\x01\x1f)\x03\r\x1f)\x03\t\x1f)\x03\x05\x1f)\x03\x05\x1d\x04J\x07\x05\x01Q\x03\x13\x01\x07\x04"\x07\x03\x01\x15\x07P\x03\x03\x07\x04\xf6\x03\x03I\x81\x05B\x03\x05\x03\x07\x05B\x0b\x07\x03\x05\x05B\x0b\t\x03\x05\x05B\x0b\x07\x03\x05\x05B\x0b\x0b\x03\x05\x05B\x0b\r\x03\x05\x11F\x0b\x0f\r\x07\x0b\r\x0f\x17=\r\x03\x05\x07\t\x0b\x01\x05B\x03\x11\x03\x05\x03F\x07\x13\x03\x17\x03\x19\rF\x07\x15\x03!\x05\x15\x1b\x03F\x0f\x17\x03#\x03\x1d\x05B\x03\x19\x03\x11\x0fF\x01\x1b\x03\x07\x07\x1f\r!\x05B\x03\x11\x03\x05\x03F\x07\x13\x03\x17\x03%\rF\x07\x15\x03!\x05\x15\'\x03F\x0f\x17\x03\t\x03)\x05B\x03\x1d\x03\x13\x0fF\x01\x1f\x03\x0b\x07+\x0f-\x05B\x03\x11\x03\x05\x03F\x07\x13\x03\x17\x031\rF\x07\x15\x03!\x05\x153\x03F\x0f\x17\x03\t\x035\x05B\x03\x1d\x03\x13\x0fF\x01!\x03\r\x077\x119\x05B\x03\x11\x03\x05\x03F\x07\x13\x03\x17\x03=\rF\x07\x15\x03!\x05\x15?\x03F\x0f\x17\x03\t\x03A\x05B\x03\x19\x03\x11\x0fF\x01#\x03\x0f\x07C\x13E\t\x04\x03\t#/;G\x07P\x01%\x07\x04S\x03\r\x13\x07G\x01\x0f\x01#\x01\x00\x03F\x05\'\x039\x03\x01\x03F\x05\x13\x03\x07\x03\x05\x0b\x06\r\x03\x07\x07\x07\x03\t\t\x04\x01\x03\x0b\x07P\x01)\x07\x04S\x03\r\x13\x07\x13\x01\x17\x01\'\x01\x00\x03F\x05+\x037\x03\x01\x03F\x05\x13\x03\x0b\x03\x05\x0b\x06\r\x03\x0b\x07\x07\x03\t\t\x04\x01\x03\x0b\x07P\x01-\x07\x04S\x03\r\x13\x07\x13\x01\x1b\x01\'\x01\x00\x03F\x05+\x03%\x03\x01\x03F\x05\x13\x03\r\x03\x05\x0b\x06\r\x03\r\x07\x07\x03\t\t\x04\x01\x03\x0b\x07P\x01/\x07\x04S\x03\r\x13\x07\x13\x01\x1f\x01#\x01\x00\x03F\x05+\x03%\x03\x01\x03F\x05\x13\x03\x0f\x03\x05\x0b\x06\r\x03\x0f\x07\x07\x03\t\t\x04\x01\x03\x0b\x06\x03\x01\x05\x01\x00\x96\tM\x1d\x03\x0f\x0b\t\t\t\t\x13\x13\x13\x0f\x11!\x11#K/ASci3\x13%)9\x1f\x11\x17\x15\x15\x11\x19)\x0f\x0b\x11builtin\x00vhlo\x00module\x00broadcast_in_dim_v1\x00constant_v1\x00func_v1\x00return_v1\x00select_v1\x00compare_v1\x00call_v1\x00custom_call_v1\x00jax.uses_shape_polymorphism\x00mhlo.num_partitions\x00mhlo.num_replicas\x00jit_func\x00jit(func)/jit(main)/pjit\x00third_party/py/jax/tests/export_back_compat_test.py\x00jit(func)/jit(main)/jit(_where)/broadcast_in_dim\x00jit(func)/jit(main)/jit(_where)/select_n\x00jit(func)/jit(main)/tridiagonal\x00jit(func)/jit(main)/eq\x00jit(func)/jit(main)/broadcast_in_dim\x00mhlo.layout_mode\x00default\x00jax.result_info\x00private\x00_where\x00_where_0\x00_where_1\x00_where_2\x00[0]\x00[1]\x00[2]\x00[3]\x00main\x00public\x00\x00lapack_zhetrd\x00\x08\x9d1\x05;\x01\x0bK_asu\x03\x81\x03U\x03\x83\x03\x85\x03\x87\x11\x89\x8b\x8dK\x8f\x91\x93\x97\x03?\x03-\x05AC\x03E\x03[\x03M\x03]\x03O\x03Q\x03S\x0b7w;M=\x03\x7f\x0b7y;O=\x03G\x0b7{;Q=\x0b7};S=',
    xla_call_module_version=9,
    nr_devices=1,
)  # End paste


# Pasted from the test output (see export_back_compat_test_util.py module docstring)
data_2024_09_03["c64"] = dict(
    testdata_version=1,
    platform='cpu',
    custom_call_targets=['lapack_chetrd'],
    serialized_date=datetime.date(2024, 9, 4),
    inputs=(),
    expected_outputs=(array([[[ 3.3228416  +0.j         , -1.9756439  +4.593356j   ,
          7.367708   +0.88518727j , -8.659938   +1.6132793j  ],
        [-6.9206004  +0.j         , -3.6362798  +0.j         ,
          3.3011198  -4.644362j   , -4.8589935  -0.61439794j ],
        [ 0.64957    +0.060723424j,  6.620491   +0.j         ,
          0.2882607  +0.j         , -1.0288142  +1.8544064j  ],
        [-0.05458622 +0.10473086j , -0.15611424 +0.06925995j ,
         -4.431866   +0.j         ,  2.364208   +0.j         ]],

       [[-4.1803885  +0.j         ,  0.5670845  +0.6913016j  ,
          2.675204   -0.23881845j , -0.41825035 -1.4060576j  ],
        [ 8.33625    +0.j         ,  2.6144838  +0.j         ,
         -2.4941807  -1.9316154j  ,  0.6687787  -2.209776j   ],
        [ 0.019031923+0.17462212j ,  2.7034955  +0.j         ,
         -0.70924187 +0.j         ,  2.7962255  +1.5316825j  ],
        [-0.057821754+0.023692288j, -0.62805307 -0.0882424j  ,
          6.6364865  +0.j         , -1.698973   +0.j         ]]],
      dtype=complex64), array([[ 3.3228416 , -3.6362798 ,  0.2882607 ,  2.364208  ],
       [-4.1803885 ,  2.6144838 , -0.70924187, -1.698973  ]],
      dtype=float32), array([[-6.9206004,  6.620491 , -4.431866 ],
       [ 8.33625  ,  2.7034955,  6.6364865]], dtype=float32), array([[1.360567 +0.1977107j , 1.7586378-0.56989706j,
        1.5772758-0.8165493j ],
       [1.9152443-0.1834492j , 1.1593437+0.55631363j,
        1.6889225-0.724835j  ]], dtype=complex64)),
    mlir_module_text=r"""
#loc1 = loc("third_party/py/jax/tests/export_back_compat_test.py":695:13)
#loc5 = loc("jit(func)/jit(main)/pjit"(#loc1))
module @jit_func attributes {jax.uses_shape_polymorphism = false, mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} {
  func.func public @main() -> (tensor<2x4x4xcomplex<f32>> {jax.result_info = "[0]", mhlo.layout_mode = "default"}, tensor<2x4xf32> {jax.result_info = "[1]", mhlo.layout_mode = "default"}, tensor<2x3xf32> {jax.result_info = "[2]", mhlo.layout_mode = "default"}, tensor<2x3xcomplex<f32>> {jax.result_info = "[3]", mhlo.layout_mode = "default"}) {
    %cst = stablehlo.constant dense<[[[(3.32284164,1.14621949), (-1.97564387,4.59335613), (7.36770821,0.885187268), (-8.65993785,1.61327934)], [(2.495340e+00,1.36827672), (-3.96969199,-0.636681795), (3.3011198,-4.64436197), (-4.85899353,-0.614397943)], [(6.03322554,1.46055949), (-3.89591122,-4.1833396), (-1.46423841,-0.106284566), (-1.0288142,1.85440636)], [(-0.657281339,0.911450386), (3.18693113,-2.02812219), (-2.64483237,0.351429433), (4.45011663,-1.79112875)]], [[(-4.18038845,-3.65238023), (0.567084491,0.691301584), (2.67520404,-0.238818452), (-0.418250352,-1.4060576)], [(-7.62970591,1.5292784), (0.269325763,2.48722434), (-2.49418068,-1.93161535), (0.668778717,-2.20977592)], [(-0.570908666,-2.75890398), (-0.235837936,3.45861554), (-0.946199476,0.23120968), (2.79622555,1.53168249)], [(0.886947453,-0.466695577), (-3.194850e+00,-0.0176551137), (-4.37602425,-3.7703948), (0.883143305,-4.70016575)]]]> : tensor<2x4x4xcomplex<f32>> loc(#loc)
    %c = stablehlo.constant dense<4> : tensor<i32> loc(#loc2)
    %c_0 = stablehlo.constant dense<1> : tensor<i32> loc(#loc2)
    %c_1 = stablehlo.constant dense<4> : tensor<i32> loc(#loc2)
    %c_2 = stablehlo.constant dense<2> : tensor<i32> loc(#loc2)
    %c_3 = stablehlo.constant dense<128> : tensor<i32> loc(#loc2)
    %0:6 = stablehlo.custom_call @lapack_chetrd(%c, %c_0, %c_1, %c_2, %c_3, %cst) {api_version = 2 : i32, operand_layouts = [dense<> : tensor<0xindex>, dense<> : tensor<0xindex>, dense<> : tensor<0xindex>, dense<> : tensor<0xindex>, dense<> : tensor<0xindex>, dense<[1, 2, 0]> : tensor<3xindex>], output_operand_aliases = [#stablehlo.output_operand_alias<output_tuple_indices = [0], operand_index = 5, operand_tuple_indices = []>], result_layouts = [dense<[1, 2, 0]> : tensor<3xindex>, dense<[1, 0]> : tensor<2xindex>, dense<[1, 0]> : tensor<2xindex>, dense<[1, 0]> : tensor<2xindex>, dense<0> : tensor<1xindex>, dense<0> : tensor<1xindex>]} : (tensor<i32>, tensor<i32>, tensor<i32>, tensor<i32>, tensor<i32>, tensor<2x4x4xcomplex<f32>>) -> (tensor<2x4x4xcomplex<f32>>, tensor<2x4xf32>, tensor<2x3xf32>, tensor<2x3xcomplex<f32>>, tensor<2xi32>, tensor<128xcomplex<f32>>) loc(#loc2)
    %c_4 = stablehlo.constant dense<0> : tensor<i32> loc(#loc)
    %1 = stablehlo.broadcast_in_dim %c_4, dims = [] : (tensor<i32>) -> tensor<2xi32> loc(#loc3)
    %2 = stablehlo.compare  EQ, %0#4, %1,  SIGNED : (tensor<2xi32>, tensor<2xi32>) -> tensor<2xi1> loc(#loc3)
    %3 = stablehlo.broadcast_in_dim %2, dims = [0] : (tensor<2xi1>) -> tensor<2x1x1xi1> loc(#loc4)
    %cst_5 = stablehlo.constant dense<(0x7FC00000,0x7FC00000)> : tensor<complex<f32>> loc(#loc)
    %4 = call @_where(%3, %0#0, %cst_5) : (tensor<2x1x1xi1>, tensor<2x4x4xcomplex<f32>>, tensor<complex<f32>>) -> tensor<2x4x4xcomplex<f32>> loc(#loc5)
    %c_6 = stablehlo.constant dense<0> : tensor<i32> loc(#loc)
    %5 = stablehlo.broadcast_in_dim %c_6, dims = [] : (tensor<i32>) -> tensor<2xi32> loc(#loc3)
    %6 = stablehlo.compare  EQ, %0#4, %5,  SIGNED : (tensor<2xi32>, tensor<2xi32>) -> tensor<2xi1> loc(#loc3)
    %7 = stablehlo.broadcast_in_dim %6, dims = [0] : (tensor<2xi1>) -> tensor<2x1xi1> loc(#loc4)
    %cst_7 = stablehlo.constant dense<0x7FC00000> : tensor<f32> loc(#loc)
    %8 = call @_where_0(%7, %0#1, %cst_7) : (tensor<2x1xi1>, tensor<2x4xf32>, tensor<f32>) -> tensor<2x4xf32> loc(#loc5)
    %c_8 = stablehlo.constant dense<0> : tensor<i32> loc(#loc)
    %9 = stablehlo.broadcast_in_dim %c_8, dims = [] : (tensor<i32>) -> tensor<2xi32> loc(#loc3)
    %10 = stablehlo.compare  EQ, %0#4, %9,  SIGNED : (tensor<2xi32>, tensor<2xi32>) -> tensor<2xi1> loc(#loc3)
    %11 = stablehlo.broadcast_in_dim %10, dims = [0] : (tensor<2xi1>) -> tensor<2x1xi1> loc(#loc4)
    %cst_9 = stablehlo.constant dense<0x7FC00000> : tensor<f32> loc(#loc)
    %12 = call @_where_1(%11, %0#2, %cst_9) : (tensor<2x1xi1>, tensor<2x3xf32>, tensor<f32>) -> tensor<2x3xf32> loc(#loc5)
    %c_10 = stablehlo.constant dense<0> : tensor<i32> loc(#loc)
    %13 = stablehlo.broadcast_in_dim %c_10, dims = [] : (tensor<i32>) -> tensor<2xi32> loc(#loc3)
    %14 = stablehlo.compare  EQ, %0#4, %13,  SIGNED : (tensor<2xi32>, tensor<2xi32>) -> tensor<2xi1> loc(#loc3)
    %15 = stablehlo.broadcast_in_dim %14, dims = [0] : (tensor<2xi1>) -> tensor<2x1xi1> loc(#loc4)
    %cst_11 = stablehlo.constant dense<(0x7FC00000,0x7FC00000)> : tensor<complex<f32>> loc(#loc)
    %16 = call @_where_2(%15, %0#3, %cst_11) : (tensor<2x1xi1>, tensor<2x3xcomplex<f32>>, tensor<complex<f32>>) -> tensor<2x3xcomplex<f32>> loc(#loc5)
    return %4, %8, %12, %16 : tensor<2x4x4xcomplex<f32>>, tensor<2x4xf32>, tensor<2x3xf32>, tensor<2x3xcomplex<f32>> loc(#loc)
  } loc(#loc)
  func.func private @_where(%arg0: tensor<2x1x1xi1> {mhlo.layout_mode = "default"} loc("jit(func)/jit(main)/pjit"(#loc1)), %arg1: tensor<2x4x4xcomplex<f32>> {mhlo.layout_mode = "default"} loc("jit(func)/jit(main)/pjit"(#loc1)), %arg2: tensor<complex<f32>> {mhlo.layout_mode = "default"} loc("jit(func)/jit(main)/pjit"(#loc1))) -> (tensor<2x4x4xcomplex<f32>> {mhlo.layout_mode = "default"}) {
    %0 = stablehlo.broadcast_in_dim %arg0, dims = [0, 1, 2] : (tensor<2x1x1xi1>) -> tensor<2x4x4xi1> loc(#loc6)
    %1 = stablehlo.broadcast_in_dim %arg2, dims = [] : (tensor<complex<f32>>) -> tensor<2x4x4xcomplex<f32>> loc(#loc6)
    %2 = stablehlo.select %0, %arg1, %1 : tensor<2x4x4xi1>, tensor<2x4x4xcomplex<f32>> loc(#loc7)
    return %2 : tensor<2x4x4xcomplex<f32>> loc(#loc5)
  } loc(#loc5)
  func.func private @_where_0(%arg0: tensor<2x1xi1> {mhlo.layout_mode = "default"} loc("jit(func)/jit(main)/pjit"(#loc1)), %arg1: tensor<2x4xf32> {mhlo.layout_mode = "default"} loc("jit(func)/jit(main)/pjit"(#loc1)), %arg2: tensor<f32> {mhlo.layout_mode = "default"} loc("jit(func)/jit(main)/pjit"(#loc1))) -> (tensor<2x4xf32> {mhlo.layout_mode = "default"}) {
    %0 = stablehlo.broadcast_in_dim %arg0, dims = [0, 1] : (tensor<2x1xi1>) -> tensor<2x4xi1> loc(#loc6)
    %1 = stablehlo.broadcast_in_dim %arg2, dims = [] : (tensor<f32>) -> tensor<2x4xf32> loc(#loc6)
    %2 = stablehlo.select %0, %arg1, %1 : tensor<2x4xi1>, tensor<2x4xf32> loc(#loc7)
    return %2 : tensor<2x4xf32> loc(#loc5)
  } loc(#loc5)
  func.func private @_where_1(%arg0: tensor<2x1xi1> {mhlo.layout_mode = "default"} loc("jit(func)/jit(main)/pjit"(#loc1)), %arg1: tensor<2x3xf32> {mhlo.layout_mode = "default"} loc("jit(func)/jit(main)/pjit"(#loc1)), %arg2: tensor<f32> {mhlo.layout_mode = "default"} loc("jit(func)/jit(main)/pjit"(#loc1))) -> (tensor<2x3xf32> {mhlo.layout_mode = "default"}) {
    %0 = stablehlo.broadcast_in_dim %arg0, dims = [0, 1] : (tensor<2x1xi1>) -> tensor<2x3xi1> loc(#loc6)
    %1 = stablehlo.broadcast_in_dim %arg2, dims = [] : (tensor<f32>) -> tensor<2x3xf32> loc(#loc6)
    %2 = stablehlo.select %0, %arg1, %1 : tensor<2x3xi1>, tensor<2x3xf32> loc(#loc7)
    return %2 : tensor<2x3xf32> loc(#loc5)
  } loc(#loc5)
  func.func private @_where_2(%arg0: tensor<2x1xi1> {mhlo.layout_mode = "default"} loc("jit(func)/jit(main)/pjit"(#loc1)), %arg1: tensor<2x3xcomplex<f32>> {mhlo.layout_mode = "default"} loc("jit(func)/jit(main)/pjit"(#loc1)), %arg2: tensor<complex<f32>> {mhlo.layout_mode = "default"} loc("jit(func)/jit(main)/pjit"(#loc1))) -> (tensor<2x3xcomplex<f32>> {mhlo.layout_mode = "default"}) {
    %0 = stablehlo.broadcast_in_dim %arg0, dims = [0, 1] : (tensor<2x1xi1>) -> tensor<2x3xi1> loc(#loc6)
    %1 = stablehlo.broadcast_in_dim %arg2, dims = [] : (tensor<complex<f32>>) -> tensor<2x3xcomplex<f32>> loc(#loc6)
    %2 = stablehlo.select %0, %arg1, %1 : tensor<2x3xi1>, tensor<2x3xcomplex<f32>> loc(#loc7)
    return %2 : tensor<2x3xcomplex<f32>> loc(#loc5)
  } loc(#loc5)
} loc(#loc)
#loc = loc(unknown)
#loc2 = loc("jit(func)/jit(main)/tridiagonal"(#loc1))
#loc3 = loc("jit(func)/jit(main)/eq"(#loc1))
#loc4 = loc("jit(func)/jit(main)/broadcast_in_dim"(#loc1))
#loc6 = loc("jit(func)/jit(main)/jit(_where)/broadcast_in_dim"(#loc1))
#loc7 = loc("jit(func)/jit(main)/jit(_where)/select_n"(#loc1))
""",
    mlir_module_serialized=b'ML\xefR\rStableHLO_v1.3.0\x00\x01#\x05\x01\x05\x13\x01\x03\x0b\x03\x11\x0f\x13\x17\x1b\x1f#\'+\x03\xf7\x99I\x01-\x0f\x07\x0f\x0f\x17\x0f\x0f\x0f\x0f#\x0b\x0f\x0b\x0b\x0b\x0b\x0b\x0b\x0b\x0b\x0b\x0b\x03m\x0f\x0b\x0b\x0f\x0b\x17\x13\x0f\x0b\x1f\x0b\x0b/OO\x0b\x0b\x0b\x0b\x0b\x1fo//\x1f\x0b\x1b\x1b\x0b\x1b\x0b\x1b\x0b\x1b\x0b\x0b\x0b\x0b\x0b\x0b\x0bo&\x08\x1f\x1f\x1f\x0b\x0b\x0b\x0b#\x0f\x17#\x01\x05\x0b\x0f\x03E\x0f\x1b\x17\x17\x17\x17\x0f\x0f\x07\x13\x0b\x07\x07\x07\x13\x1b\x17\x07\x1f\x1f\x1f\x1f\x1f\x13\x13\x17\x1b\x13\x17\x13\x13\x13\x13\x13\x02\xe2\x0b\x1d\x1f\t\x1f\x1d#\t\x1d)\t\x17!\xde\n\x1b\x1d\'\t\x1d%\t\x1d+\t\x11\x03\x05\x03\x07\x15\x17\x19\x11\x1b\x11\x05\x17\x11\x01\x00\x05\x19\x05\x1b\x05\x1d\x05\x1f\x05!\x05#\x05%\x05\'\x05)\x05+\x1f5\x01\x1d-\x1d/\x1f?\x01\x1d1\x03\x07999\r\x03/1\x03\x039\x1d3\x1f\x05\t\x00\x00\x00\x00\t\x07\x07\x01\x1fG\x11\x00\x00\x00\x00\x00\x00\x00\x00\x1f3!\x00\x00\x00\x00\x00\x00\x00\x00\x01\x00\x00\x00\x00\x00\x00\x00\x1fC!\x01\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x03\x01\x1d5\x1d7\x1d9\x1d;\x1f\x05\t\x04\x00\x00\x00\x1fA1\x01\x00\x00\x00\x00\x00\x00\x00\x02\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x1fE\x11\x00\x00\x00\x00\x00\x00\x00\x00\x1f\x11\x11\x00\x00\xc0\x7f\x00\x00\xc0\x7f\x1f\x13\t\x00\x00\xc0\x7f#)\x03\tcgko\r\x055e/1\x1d=\r\x055i/1\x1d?\r\x055m/1\x1dA\r\x055q/1\x1dC\x1dE\x1dG#+#-#/#1\x1f;1\x00\x00\x00\x00\x00\x00\x00\x00\x01\x00\x00\x00\x00\x00\x00\x00\x02\x00\x00\x00\x00\x00\x00\x00\x1f\x07\x02\x04p\xa9T@R\xb7\x92?\xe6\xe1\xfc\xbf\xc6\xfc\x92@D\xc4\xeb@\xa2\x9bb?\x1b\x8f\n\xc1\xf0\x7f\xce?\xa7\xb3\x1f@\xb1#\xaf?o\x0f~\xc0\x94\xfd"\xbf\x8cES@\x9d\x9e\x94\xc0\xe0|\x9b\xc0/I\x1d\xbf/\x10\xc1@\x9d\xf3\xba?\x9cVy\xc0\xeb\xdd\x85\xc0*l\xbb\xbf\xb9\xab\xd9\xbd/\xb0\x83\xbf0]\xed?\x97C(\xbf\xd0Ti?\xae\xf6K@\xc1\xcc\x01\xc0\xefD)\xc0\x8f\xee\xb3>[g\x8e@\xb5C\xe5\xbf\xbe\xc5\x85\xc0\x99\xc0i\xc0s,\x11?$\xf90?\x8b6+@\xd3\x8ct\xbe\xe9$\xd6\xbe\xb2\xf9\xb3\xbf\x8d&\xf4\xc0e\xbf\xc3?\x11\xe5\x89>\xaf.\x1f@\xa8\xa0\x1f\xc0,?\xf7\xbf\x155+?\xf8l\r\xc0\x12\'\x12\xbf\xe2\x910\xc0\x80\x7fq\xbe\xf5Y]@!:r\xbf;\xc2l>\\\xf52@,\x0e\xc4?\xfd\x0ec?\xb9\xf2\xee\xbelxL\xc0u\xa1\x90\xbcd\x08\x8c\xc0&Nq\xc0\xae\x15b?\xc2g\x96\xc0\x1f\x05\t\x01\x00\x00\x00\x1f\x05\t\x02\x00\x00\x00\x1f\x05\t\x80\x00\x00\x00\x0b\x05\x1dI\x1dK\x05\x01\x03\r33333W\x03\x03\x95\x15\x03\x01\x15\x01\x03\rWIIIYY\x01\t\x01\x02\x02)\x01\')\x07\t\x11\x11\x19)\x05\t\x05\x15)\x05\t\x11\x1b)\x05\t\r\x1b)\x05\t\r\x19)\x01\x19)\x01\x1b\x01)\x03\t\'\x03\x1b\t\x1d\x13)\x03\t\x15)\x07\t\x05\x05\x15)\x05\t\r\x15\x1b\x11\x01\t\x07\x0b\r\x0f\x11\x07#\x07\x11\x03\x07\x11\x07\t\x0b\x13\x03\x0b\x11\x07\t\r\x13\x03\r\x11\x07\t\x0f\x11\x03\x0f)\x03\t\x1d)\x03\x01\x1d)\x05\t\x11\x15)\x07\t\x11\x11\x15)\x03\r\x1d)\x03\x02\x04\x19)\x03\x01\x1f)\x03\r\x1f)\x03\t\x1f)\x03\x05\x1f)\x03\x05\x1d\x04J\x07\x05\x01Q\x03\x13\x01\x07\x04"\x07\x03\x01\x15\x07P\x03\x03\x07\x04\xf6\x03\x03I\x81\x05B\x03\x05\x03\x07\x05B\x0b\x07\x03\x05\x05B\x0b\t\x03\x05\x05B\x0b\x07\x03\x05\x05B\x0b\x0b\x03\x05\x05B\x0b\r\x03\x05\x11F\x0b\x0f\r\x07\x0b\r\x0f\x17=\r\x03\x05\x07\t\x0b\x01\x05B\x03\x11\x03\x05\x03F\x07\x13\x03\x17\x03\x19\rF\x07\x15\x03!\x05\x15\x1b\x03F\x0f\x17\x03#\x03\x1d\x05B\x03\x19\x03\x11\x0fF\x01\x1b\x03\x07\x07\x1f\r!\x05B\x03\x11\x03\x05\x03F\x07\x13\x03\x17\x03%\rF\x07\x15\x03!\x05\x15\'\x03F\x0f\x17\x03\t\x03)\x05B\x03\x1d\x03\x13\x0fF\x01\x1f\x03\x0b\x07+\x0f-\x05B\x03\x11\x03\x05\x03F\x07\x13\x03\x17\x031\rF\x07\x15\x03!\x05\x153\x03F\x0f\x17\x03\t\x035\x05B\x03\x1d\x03\x13\x0fF\x01!\x03\r\x077\x119\x05B\x03\x11\x03\x05\x03F\x07\x13\x03\x17\x03=\rF\x07\x15\x03!\x05\x15?\x03F\x0f\x17\x03\t\x03A\x05B\x03\x19\x03\x11\x0fF\x01#\x03\x0f\x07C\x13E\t\x04\x03\t#/;G\x07P\x01%\x07\x04S\x03\r\x13\x07G\x01\x0f\x01#\x01\x00\x03F\x05\'\x039\x03\x01\x03F\x05\x13\x03\x07\x03\x05\x0b\x06\r\x03\x07\x07\x07\x03\t\t\x04\x01\x03\x0b\x07P\x01)\x07\x04S\x03\r\x13\x07\x13\x01\x17\x01\'\x01\x00\x03F\x05+\x037\x03\x01\x03F\x05\x13\x03\x0b\x03\x05\x0b\x06\r\x03\x0b\x07\x07\x03\t\t\x04\x01\x03\x0b\x07P\x01-\x07\x04S\x03\r\x13\x07\x13\x01\x1b\x01\'\x01\x00\x03F\x05+\x03%\x03\x01\x03F\x05\x13\x03\r\x03\x05\x0b\x06\r\x03\r\x07\x07\x03\t\t\x04\x01\x03\x0b\x07P\x01/\x07\x04S\x03\r\x13\x07\x13\x01\x1f\x01#\x01\x00\x03F\x05+\x03%\x03\x01\x03F\x05\x13\x03\x0f\x03\x05\x0b\x06\r\x03\x0f\x07\x07\x03\t\t\x04\x01\x03\x0b\x06\x03\x01\x05\x01\x00\x96\tM\x1d\x03\x0f\x0b\t\t\t\t\x13\x13\x13\x0f\x11!\x11#K/ASci3\x13%)9\x1f\x11\x17\x15\x15\x11\x19)\x0f\x0b\x11builtin\x00vhlo\x00module\x00broadcast_in_dim_v1\x00constant_v1\x00func_v1\x00return_v1\x00select_v1\x00compare_v1\x00call_v1\x00custom_call_v1\x00jax.uses_shape_polymorphism\x00mhlo.num_partitions\x00mhlo.num_replicas\x00jit_func\x00jit(func)/jit(main)/pjit\x00third_party/py/jax/tests/export_back_compat_test.py\x00jit(func)/jit(main)/jit(_where)/broadcast_in_dim\x00jit(func)/jit(main)/jit(_where)/select_n\x00jit(func)/jit(main)/tridiagonal\x00jit(func)/jit(main)/eq\x00jit(func)/jit(main)/broadcast_in_dim\x00mhlo.layout_mode\x00default\x00jax.result_info\x00private\x00_where\x00_where_0\x00_where_1\x00_where_2\x00[0]\x00[1]\x00[2]\x00[3]\x00main\x00public\x00\x00lapack_chetrd\x00\x08\x9d1\x05;\x01\x0bK_asu\x03\x81\x03U\x03\x83\x03\x85\x03\x87\x11\x89\x8b\x8dK\x8f\x91\x93\x97\x03?\x03-\x05AC\x03E\x03[\x03M\x03]\x03O\x03Q\x03S\x0b7w;M=\x03\x7f\x0b7y;O=\x03G\x0b7{;Q=\x0b7};S=',
    xla_call_module_version=9,
    nr_devices=1,
)  # End paste


# Pasted from the test output (see export_back_compat_test_util.py module docstring)
data_2024_09_03["f32"] = dict(
    testdata_version=1,
    platform='cpu',
    custom_call_targets=['lapack_ssytrd'],
    serialized_date=datetime.date(2024, 9, 4),
    inputs=(),
    expected_outputs=(array([[[-0.8395241 ,  0.156272  , -1.6810869 ,  0.23832119],
        [-2.985257  , -5.571     , -0.22652794, -0.83806676],
        [ 0.27237308, -1.6295947 ,  2.0042834 , -1.148861  ],
        [-0.17183593,  0.57464546,  0.5536146 , -4.206357  ]],

       [[ 1.7666914 ,  2.569005  , -0.86576384, -0.1617768 ],
        [-5.143918  ,  5.0426254 , -3.7237067 ,  4.383015  ],
        [ 0.33311516, -1.5299042 , -8.854181  , -2.896776  ],
        [ 0.3419102 ,  0.2669245 , -2.8250606 ,  5.752488  ]]],
      dtype=float32), array([[-0.8395241, -5.571    ,  2.0042834, -4.206357 ],
       [ 1.7666914,  5.0426254, -8.854181 ,  5.752488 ]], dtype=float32), array([[-2.985257 , -1.6295947,  0.5536146],
       [-5.143918 , -1.5299042, -2.8250606]], dtype=float32), array([[1.8120625, 1.5035137, 0.       ],
       [1.6288393, 1.8669801, 0.       ]], dtype=float32)),
    mlir_module_text=r"""
#loc1 = loc("third_party/py/jax/tests/export_back_compat_test.py":695:13)
#loc5 = loc("jit(func)/jit(main)/pjit"(#loc1))
module @jit_func attributes {jax.uses_shape_polymorphism = false, mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} {
  func.func public @main() -> (tensor<2x4x4xf32> {jax.result_info = "[0]", mhlo.layout_mode = "default"}, tensor<2x4xf32> {jax.result_info = "[1]", mhlo.layout_mode = "default"}, tensor<2x3xf32> {jax.result_info = "[2]", mhlo.layout_mode = "default"}, tensor<2x3xf32> {jax.result_info = "[3]", mhlo.layout_mode = "default"}) {
    %cst = stablehlo.constant dense<[[[-0.83952409, 1.562720e-01, -1.6810869, 0.238321185], [2.42421508, -5.17118931, -0.226527944, -0.838066756], [1.47339451, -1.32866347, -3.3505435, -1.14886105], [-0.929541587, -0.955984473, 2.71886253, 0.748659431]], [[1.76669145, 2.56900501, -0.865763843, -0.161776796], [3.23469758, -0.362713158, -3.72370672, 4.38301516], [2.79104376, 7.36582708, -3.04437494, -2.89677596], [2.86473417, 0.981746375, -2.13533139, 5.34802151]]]> : tensor<2x4x4xf32> loc(#loc)
    %c = stablehlo.constant dense<4> : tensor<i32> loc(#loc2)
    %c_0 = stablehlo.constant dense<1> : tensor<i32> loc(#loc2)
    %c_1 = stablehlo.constant dense<4> : tensor<i32> loc(#loc2)
    %c_2 = stablehlo.constant dense<2> : tensor<i32> loc(#loc2)
    %c_3 = stablehlo.constant dense<128> : tensor<i32> loc(#loc2)
    %0:6 = stablehlo.custom_call @lapack_ssytrd(%c, %c_0, %c_1, %c_2, %c_3, %cst) {api_version = 2 : i32, operand_layouts = [dense<> : tensor<0xindex>, dense<> : tensor<0xindex>, dense<> : tensor<0xindex>, dense<> : tensor<0xindex>, dense<> : tensor<0xindex>, dense<[1, 2, 0]> : tensor<3xindex>], output_operand_aliases = [#stablehlo.output_operand_alias<output_tuple_indices = [0], operand_index = 5, operand_tuple_indices = []>], result_layouts = [dense<[1, 2, 0]> : tensor<3xindex>, dense<[1, 0]> : tensor<2xindex>, dense<[1, 0]> : tensor<2xindex>, dense<[1, 0]> : tensor<2xindex>, dense<0> : tensor<1xindex>, dense<0> : tensor<1xindex>]} : (tensor<i32>, tensor<i32>, tensor<i32>, tensor<i32>, tensor<i32>, tensor<2x4x4xf32>) -> (tensor<2x4x4xf32>, tensor<2x4xf32>, tensor<2x3xf32>, tensor<2x3xf32>, tensor<2xi32>, tensor<128xf32>) loc(#loc2)
    %c_4 = stablehlo.constant dense<0> : tensor<i32> loc(#loc)
    %1 = stablehlo.broadcast_in_dim %c_4, dims = [] : (tensor<i32>) -> tensor<2xi32> loc(#loc3)
    %2 = stablehlo.compare  EQ, %0#4, %1,  SIGNED : (tensor<2xi32>, tensor<2xi32>) -> tensor<2xi1> loc(#loc3)
    %3 = stablehlo.broadcast_in_dim %2, dims = [0] : (tensor<2xi1>) -> tensor<2x1x1xi1> loc(#loc4)
    %cst_5 = stablehlo.constant dense<0x7FC00000> : tensor<f32> loc(#loc)
    %4 = call @_where(%3, %0#0, %cst_5) : (tensor<2x1x1xi1>, tensor<2x4x4xf32>, tensor<f32>) -> tensor<2x4x4xf32> loc(#loc5)
    %c_6 = stablehlo.constant dense<0> : tensor<i32> loc(#loc)
    %5 = stablehlo.broadcast_in_dim %c_6, dims = [] : (tensor<i32>) -> tensor<2xi32> loc(#loc3)
    %6 = stablehlo.compare  EQ, %0#4, %5,  SIGNED : (tensor<2xi32>, tensor<2xi32>) -> tensor<2xi1> loc(#loc3)
    %7 = stablehlo.broadcast_in_dim %6, dims = [0] : (tensor<2xi1>) -> tensor<2x1xi1> loc(#loc4)
    %cst_7 = stablehlo.constant dense<0x7FC00000> : tensor<f32> loc(#loc)
    %8 = call @_where_0(%7, %0#1, %cst_7) : (tensor<2x1xi1>, tensor<2x4xf32>, tensor<f32>) -> tensor<2x4xf32> loc(#loc5)
    %c_8 = stablehlo.constant dense<0> : tensor<i32> loc(#loc)
    %9 = stablehlo.broadcast_in_dim %c_8, dims = [] : (tensor<i32>) -> tensor<2xi32> loc(#loc3)
    %10 = stablehlo.compare  EQ, %0#4, %9,  SIGNED : (tensor<2xi32>, tensor<2xi32>) -> tensor<2xi1> loc(#loc3)
    %11 = stablehlo.broadcast_in_dim %10, dims = [0] : (tensor<2xi1>) -> tensor<2x1xi1> loc(#loc4)
    %cst_9 = stablehlo.constant dense<0x7FC00000> : tensor<f32> loc(#loc)
    %12 = call @_where_1(%11, %0#2, %cst_9) : (tensor<2x1xi1>, tensor<2x3xf32>, tensor<f32>) -> tensor<2x3xf32> loc(#loc5)
    %c_10 = stablehlo.constant dense<0> : tensor<i32> loc(#loc)
    %13 = stablehlo.broadcast_in_dim %c_10, dims = [] : (tensor<i32>) -> tensor<2xi32> loc(#loc3)
    %14 = stablehlo.compare  EQ, %0#4, %13,  SIGNED : (tensor<2xi32>, tensor<2xi32>) -> tensor<2xi1> loc(#loc3)
    %15 = stablehlo.broadcast_in_dim %14, dims = [0] : (tensor<2xi1>) -> tensor<2x1xi1> loc(#loc4)
    %cst_11 = stablehlo.constant dense<0x7FC00000> : tensor<f32> loc(#loc)
    %16 = call @_where_1(%15, %0#3, %cst_11) : (tensor<2x1xi1>, tensor<2x3xf32>, tensor<f32>) -> tensor<2x3xf32> loc(#loc5)
    return %4, %8, %12, %16 : tensor<2x4x4xf32>, tensor<2x4xf32>, tensor<2x3xf32>, tensor<2x3xf32> loc(#loc)
  } loc(#loc)
  func.func private @_where(%arg0: tensor<2x1x1xi1> {mhlo.layout_mode = "default"} loc("jit(func)/jit(main)/pjit"(#loc1)), %arg1: tensor<2x4x4xf32> {mhlo.layout_mode = "default"} loc("jit(func)/jit(main)/pjit"(#loc1)), %arg2: tensor<f32> {mhlo.layout_mode = "default"} loc("jit(func)/jit(main)/pjit"(#loc1))) -> (tensor<2x4x4xf32> {mhlo.layout_mode = "default"}) {
    %0 = stablehlo.broadcast_in_dim %arg0, dims = [0, 1, 2] : (tensor<2x1x1xi1>) -> tensor<2x4x4xi1> loc(#loc6)
    %1 = stablehlo.broadcast_in_dim %arg2, dims = [] : (tensor<f32>) -> tensor<2x4x4xf32> loc(#loc6)
    %2 = stablehlo.select %0, %arg1, %1 : tensor<2x4x4xi1>, tensor<2x4x4xf32> loc(#loc7)
    return %2 : tensor<2x4x4xf32> loc(#loc5)
  } loc(#loc5)
  func.func private @_where_0(%arg0: tensor<2x1xi1> {mhlo.layout_mode = "default"} loc("jit(func)/jit(main)/pjit"(#loc1)), %arg1: tensor<2x4xf32> {mhlo.layout_mode = "default"} loc("jit(func)/jit(main)/pjit"(#loc1)), %arg2: tensor<f32> {mhlo.layout_mode = "default"} loc("jit(func)/jit(main)/pjit"(#loc1))) -> (tensor<2x4xf32> {mhlo.layout_mode = "default"}) {
    %0 = stablehlo.broadcast_in_dim %arg0, dims = [0, 1] : (tensor<2x1xi1>) -> tensor<2x4xi1> loc(#loc6)
    %1 = stablehlo.broadcast_in_dim %arg2, dims = [] : (tensor<f32>) -> tensor<2x4xf32> loc(#loc6)
    %2 = stablehlo.select %0, %arg1, %1 : tensor<2x4xi1>, tensor<2x4xf32> loc(#loc7)
    return %2 : tensor<2x4xf32> loc(#loc5)
  } loc(#loc5)
  func.func private @_where_1(%arg0: tensor<2x1xi1> {mhlo.layout_mode = "default"} loc("jit(func)/jit(main)/pjit"(#loc1)), %arg1: tensor<2x3xf32> {mhlo.layout_mode = "default"} loc("jit(func)/jit(main)/pjit"(#loc1)), %arg2: tensor<f32> {mhlo.layout_mode = "default"} loc("jit(func)/jit(main)/pjit"(#loc1))) -> (tensor<2x3xf32> {mhlo.layout_mode = "default"}) {
    %0 = stablehlo.broadcast_in_dim %arg0, dims = [0, 1] : (tensor<2x1xi1>) -> tensor<2x3xi1> loc(#loc6)
    %1 = stablehlo.broadcast_in_dim %arg2, dims = [] : (tensor<f32>) -> tensor<2x3xf32> loc(#loc6)
    %2 = stablehlo.select %0, %arg1, %1 : tensor<2x3xi1>, tensor<2x3xf32> loc(#loc7)
    return %2 : tensor<2x3xf32> loc(#loc5)
  } loc(#loc5)
} loc(#loc)
#loc = loc(unknown)
#loc2 = loc("jit(func)/jit(main)/tridiagonal"(#loc1))
#loc3 = loc("jit(func)/jit(main)/eq"(#loc1))
#loc4 = loc("jit(func)/jit(main)/broadcast_in_dim"(#loc1))
#loc6 = loc("jit(func)/jit(main)/jit(_where)/broadcast_in_dim"(#loc1))
#loc7 = loc("jit(func)/jit(main)/jit(_where)/select_n"(#loc1))
""",
    mlir_module_serialized=b'ML\xefR\rStableHLO_v1.3.0\x00\x01#\x05\x01\x05\x13\x01\x03\x0b\x03\x11\x0f\x13\x17\x1b\x1f#\'+\x03\xe9\x93A\x01-\x0f\x07\x0f\x17\x0f\x0f\x0f\x0f\x0f#\x0b\x0f\x0b\x0b\x0b\x0b\x0b\x0b\x0b\x0b\x0b\x0b\x03g\x0f\x0b\x0b\x0f\x0b\x13\x1f\x0b\x0b/\x1f\x17\x0f\x0b\x0bO\x0b\x0b\x0bO\x1fo/\x0b\x1b\x1b\x0b\x1b\x0b\x1b\x0b\x1b\x0b\x0b\x0b\x0b\x0b\x0bo&\x04\x1f\x1f\x1f\x0b\x0b\x0b\x0b#\x0f\x17#\x01\x05\x0b\x0f\x03=\x0f\x17\x0f\x1b\x17\x17\x07\x07\x13\x07\x07\x13\x1b\x07\x1f\x1f\x1f\x1f\x17\x13\x13\x17\x1b\x13\x17\x13\x13\x13\x13\x13\x02b\t\x1d\x1f\x07\x1f\x1d)\x07\x17!\xde\n\x1b\x1d#\x07\x1d\'\x07\x1d+\x07\x1d%\x07\x11\x03\x05\x03\x07\x15\x17\x19\x11\x1b\x11\x05\x17\x11\x01\x00\x05\x19\x05\x1b\x05\x1d\x05\x1f\x05!\x05#\x05%\x05\'\x05)\x05+\x1f-\x01\x1d-\x1d/\x1f7\x01\x1d1\r\x03/1\x1f\x05\t\x00\x00\x00\x00\t\x07\x07\x01\x1f?\x11\x00\x00\x00\x00\x00\x00\x00\x00\x1f\t\t\x00\x00\xc0\x7f\x03\x07777\x03\x037\x1d3\x1d5\x1f;!\x01\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x03\x01\x1d7\x1d9\x1f+!\x00\x00\x00\x00\x00\x00\x00\x00\x01\x00\x00\x00\x00\x00\x00\x00\x1f\x05\t\x04\x00\x00\x00\x1f91\x01\x00\x00\x00\x00\x00\x00\x00\x02\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x1f=\x11\x00\x00\x00\x00\x00\x00\x00\x00#!\x03\t_cgk\r\x055a/1\x1d;\r\x055e/1\x1d=\r\x055i/1\x1d?\r\x055m/1\x1dA\x1dC\x1dE###%#\'\x1f31\x00\x00\x00\x00\x00\x00\x00\x00\x01\x00\x00\x00\x00\x00\x00\x00\x02\x00\x00\x00\x00\x00\x00\x00\x1f\x0b\x02\x02\r\xebV\xbf\xc4\x05 >\xdb-\xd7\xbfx\nt>W&\x1b@bz\xa5\xc0\xf1\xf6g\xbe\x8b\x8bV\xbf1\x98\xbc?\xa5\x11\xaa\xbfNoV\xc0\xe1\r\x93\xbfp\xf6m\xbff\xbbt\xbf\xd8\x01.@%\xa8??\xf2"\xe2?\x94j$@\xb3\xa2]\xbf\xd1\xa8%\xbeI\x05O@\x8a\xb5\xb9\xbe6Qn\xc0\xa9A\x8c@v\xa02@\xdb\xb4\xeb@\n\xd7B\xc0\xc7d9\xc0\xceW7@\xbbS{?E\xa9\x08\xc0\xfe"\xab@\x1f\x05\t\x01\x00\x00\x00\x1f\x05\t\x02\x00\x00\x00\x1f\x05\t\x80\x00\x00\x00\x0b\x05\x1dG\x1dI\x05\x01\x03\r33333W\x03\x03\x8f\x15\x03\x01\x15\x01\x03\rWKKKYY\x01\t\x01\x02\x02)\x01\x1f)\x05\t\r\x13)\x01\x13)\x07\t\x11\x11\x13)\x05\t\x11\x13)\x05\t\x05\x11\x01\t)\x03\t\x1f\x1d\x13)\x03\t\x11)\x07\t\x05\x05\x11\x1b\x11\x01\t\x0b\r\x07\x07\x11\x07\x1d\x0b\t\x03\x0b\x11\x07\x0f\r\t\x03\r\x11\x07\x0f\x07\t\x03\x07)\x05\t\r\x11)\x03\t\x17)\x03\x01\x17)\x05\t\x11\x11)\x07\t\x11\x11\x11)\x03\r\x17)\x03\x02\x04\x13)\x03\x01\x19)\x03\r\x19)\x03\t\x19)\x03\x05\x19)\x03\x05\x17\x04\x8a\x06\x05\x01Q\x03\x13\x01\x07\x04b\x06\x03\x01\x11\x07P\x03\x03\x07\x04\xf6\x03\x03I\x81\x05B\x03\x05\x03\x0b\x05B\x0b\x07\x03\x05\x05B\x0b\t\x03\x05\x05B\x0b\x07\x03\x05\x05B\x0b\x0b\x03\x05\x05B\x0b\r\x03\x05\x11F\x0b\x0f\r\x0b\r\x07\x07\x155\r\x03\x05\x07\t\x0b\x01\x05B\x03\x11\x03\x05\x03F\x05\x13\x03\x15\x03\x19\x0bF\x05\x15\x03\x1b\x05\x15\x1b\x03F\r\x17\x03\x1d\x03\x1d\x05B\x03\x19\x03\t\rF\x01\x1b\x03\x0b\x07\x1f\r!\x05B\x03\x11\x03\x05\x03F\x05\x13\x03\x15\x03%\x0bF\x05\x15\x03\x1b\x05\x15\'\x03F\r\x17\x03\x0f\x03)\x05B\x03\x19\x03\t\rF\x01\x1d\x03\r\x07+\x0f-\x05B\x03\x11\x03\x05\x03F\x05\x13\x03\x15\x031\x0bF\x05\x15\x03\x1b\x05\x153\x03F\r\x17\x03\x0f\x035\x05B\x03\x19\x03\t\rF\x01\x1f\x03\x07\x077\x119\x05B\x03\x11\x03\x05\x03F\x05\x13\x03\x15\x03=\x0bF\x05\x15\x03\x1b\x05\x15?\x03F\r\x17\x03\x0f\x03A\x05B\x03\x19\x03\t\rF\x01\x1f\x03\x07\x07C\x13E\t\x04\x03\t#/;G\x07P\x01!\x07\x04S\x03\r\x13\x07;\x01\x17\x01\x13\x01\x00\x03F\t#\x031\x03\x01\x03F\t\x13\x03\x0b\x03\x05\x0f\x06\x0f\x03\x0b\x07\x07\x03\t\t\x04\x01\x03\x0b\x07P\x01%\x07\x04S\x03\r\x13\x07\x1f\x01\x1b\x01\x13\x01\x00\x03F\t\'\x03/\x03\x01\x03F\t\x13\x03\r\x03\x05\x0f\x06\x0f\x03\r\x07\x07\x03\t\t\x04\x01\x03\x0b\x07P\x01)\x07\x04S\x03\r\x13\x07\x1f\x01\x0f\x01\x13\x01\x00\x03F\t\'\x03)\x03\x01\x03F\t\x13\x03\x07\x03\x05\x0f\x06\x0f\x03\x07\x07\x07\x03\t\t\x04\x01\x03\x0b\x06\x03\x01\x05\x01\x00n\tK\x1d\x03\x0f\x0b\t\t\t\t\x13\x0f\x13\x11!\x11#K/ASci3\x13%)9\x1f\x15\x11\x17\x15\x11\x19)\x0f\x0b\x11builtin\x00vhlo\x00module\x00broadcast_in_dim_v1\x00constant_v1\x00func_v1\x00return_v1\x00compare_v1\x00call_v1\x00select_v1\x00custom_call_v1\x00jax.uses_shape_polymorphism\x00mhlo.num_partitions\x00mhlo.num_replicas\x00jit_func\x00jit(func)/jit(main)/pjit\x00third_party/py/jax/tests/export_back_compat_test.py\x00jit(func)/jit(main)/jit(_where)/broadcast_in_dim\x00jit(func)/jit(main)/jit(_where)/select_n\x00jit(func)/jit(main)/tridiagonal\x00jit(func)/jit(main)/eq\x00jit(func)/jit(main)/broadcast_in_dim\x00mhlo.layout_mode\x00default\x00jax.result_info\x00private\x00_where_1\x00_where\x00_where_0\x00[0]\x00[1]\x00[2]\x00[3]\x00main\x00public\x00\x00lapack_ssytrd\x00\x08\x89+\x05;\x01\x0bM[]oq\x03{\x03U\x03}\x03\x7f\x03\x81\x11\x83\x85\x87M\x89\x8b\x8d\x91\x039\x03-\x05;=\x03?\x03A\x03O\x03Q\x03I\x0bCsEOG\x03y\x0bCuEQG\x03S\x0bCwEIG',
    xla_call_module_version=9,
    nr_devices=1,
)  # End paste


# Pasted from the test output (see export_back_compat_test_util.py module docstring)
data_2024_09_03["f64"] = dict(
    testdata_version=1,
    platform='cpu',
    custom_call_targets=['lapack_dsytrd'],
    serialized_date=datetime.date(2024, 9, 4),
    inputs=(),
    expected_outputs=(array([[[ 0.8251247184208595 , -2.6963562039892532 ,
          0.8082445002373937 , -1.551980329390836  ],
        [-2.629505060186711  ,  4.427374205796291  ,
         -2.2111093161901074 ,  7.552489598405787  ],
        [ 0.2269453213819231 ,  0.3650586474106988 ,
         -3.5933639667756205 ,  4.828829679372501  ],
        [-0.6415372293575187 , -0.2519326897319508 ,
         -1.7607827845801751 , -3.381311711243865  ]],

       [[-4.000421911405985  ,  3.6303350337601055 ,
          2.8066821235532355 ,  1.099224389184342  ],
        [-4.141622408467332  , -5.276404169116551  ,
         -0.8496056221591237 , -2.275319346221659  ],
        [ 0.5828958067901202 ,  0.9351254869793256 ,
          2.7765603683442177 , -4.339686212557215  ],
        [-0.6391146585297987 ,  0.3129920702652711 ,
         -0.25441692469349864, -1.4155240723557498 ]]]), array([[ 0.8251247184208595,  4.427374205796291 , -3.5933639667756205,
        -3.381311711243865 ],
       [-4.000421911405985 , -5.276404169116551 ,  2.7765603683442177,
        -1.4155240723557498]]), array([[-2.629505060186711  ,  0.3650586474106988 , -1.7607827845801751 ],
       [-4.141622408467332  ,  0.9351254869793256 , -0.25441692469349864]]), array([[1.3669846724688552, 1.8806358893589366, 0.                ],
       [1.1440109149169537, 1.8215532880266878, 0.                ]])),
    mlir_module_text=r"""
#loc1 = loc("third_party/py/jax/tests/export_back_compat_test.py":695:13)
#loc5 = loc("jit(func)/jit(main)/pjit"(#loc1))
module @jit_func attributes {jax.uses_shape_polymorphism = false, mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} {
  func.func public @main() -> (tensor<2x4x4xf64> {jax.result_info = "[0]", mhlo.layout_mode = "default"}, tensor<2x4xf64> {jax.result_info = "[1]", mhlo.layout_mode = "default"}, tensor<2x3xf64> {jax.result_info = "[2]", mhlo.layout_mode = "default"}, tensor<2x3xf64> {jax.result_info = "[3]", mhlo.layout_mode = "default"}) {
    %cst = stablehlo.constant dense<[[[0.82512471842085955, -2.6963562039892532, 0.80824450023739369, -1.5519803293908361], [0.96498805326781766, -4.1313349231964409, -2.2111093161901074, 7.5524895984057867], [0.81575339483804743, 1.0647235400727899, -1.0064296232364345, 4.8288296793725012], [-2.3060011529502993, -2.9182106402942192, -1.7781896154088577, 2.5904630742096817]], [[-4.0004219114059847, 3.6303350337601055, 2.8066821235532355, 1.0992243891843421], [0.59643883228393779, -1.5243235004961249, -0.84960562215912372, -2.275319346221659], [2.7617960295487092, -0.57538970930521982, 0.12559406141906576, -4.3396862125572149], [-3.0281643919760217, 0.38177997229319849, 3.860398204232184, -2.5166384340510231]]]> : tensor<2x4x4xf64> loc(#loc)
    %c = stablehlo.constant dense<4> : tensor<i32> loc(#loc2)
    %c_0 = stablehlo.constant dense<1> : tensor<i32> loc(#loc2)
    %c_1 = stablehlo.constant dense<4> : tensor<i32> loc(#loc2)
    %c_2 = stablehlo.constant dense<2> : tensor<i32> loc(#loc2)
    %c_3 = stablehlo.constant dense<128> : tensor<i32> loc(#loc2)
    %0:6 = stablehlo.custom_call @lapack_dsytrd(%c, %c_0, %c_1, %c_2, %c_3, %cst) {api_version = 2 : i32, operand_layouts = [dense<> : tensor<0xindex>, dense<> : tensor<0xindex>, dense<> : tensor<0xindex>, dense<> : tensor<0xindex>, dense<> : tensor<0xindex>, dense<[1, 2, 0]> : tensor<3xindex>], output_operand_aliases = [#stablehlo.output_operand_alias<output_tuple_indices = [0], operand_index = 5, operand_tuple_indices = []>], result_layouts = [dense<[1, 2, 0]> : tensor<3xindex>, dense<[1, 0]> : tensor<2xindex>, dense<[1, 0]> : tensor<2xindex>, dense<[1, 0]> : tensor<2xindex>, dense<0> : tensor<1xindex>, dense<0> : tensor<1xindex>]} : (tensor<i32>, tensor<i32>, tensor<i32>, tensor<i32>, tensor<i32>, tensor<2x4x4xf64>) -> (tensor<2x4x4xf64>, tensor<2x4xf64>, tensor<2x3xf64>, tensor<2x3xf64>, tensor<2xi32>, tensor<128xf64>) loc(#loc2)
    %c_4 = stablehlo.constant dense<0> : tensor<i32> loc(#loc)
    %1 = stablehlo.broadcast_in_dim %c_4, dims = [] : (tensor<i32>) -> tensor<2xi32> loc(#loc3)
    %2 = stablehlo.compare  EQ, %0#4, %1,  SIGNED : (tensor<2xi32>, tensor<2xi32>) -> tensor<2xi1> loc(#loc3)
    %3 = stablehlo.broadcast_in_dim %2, dims = [0] : (tensor<2xi1>) -> tensor<2x1x1xi1> loc(#loc4)
    %cst_5 = stablehlo.constant dense<0x7FF8000000000000> : tensor<f64> loc(#loc)
    %4 = call @_where(%3, %0#0, %cst_5) : (tensor<2x1x1xi1>, tensor<2x4x4xf64>, tensor<f64>) -> tensor<2x4x4xf64> loc(#loc5)
    %c_6 = stablehlo.constant dense<0> : tensor<i32> loc(#loc)
    %5 = stablehlo.broadcast_in_dim %c_6, dims = [] : (tensor<i32>) -> tensor<2xi32> loc(#loc3)
    %6 = stablehlo.compare  EQ, %0#4, %5,  SIGNED : (tensor<2xi32>, tensor<2xi32>) -> tensor<2xi1> loc(#loc3)
    %7 = stablehlo.broadcast_in_dim %6, dims = [0] : (tensor<2xi1>) -> tensor<2x1xi1> loc(#loc4)
    %cst_7 = stablehlo.constant dense<0x7FF8000000000000> : tensor<f64> loc(#loc)
    %8 = call @_where_0(%7, %0#1, %cst_7) : (tensor<2x1xi1>, tensor<2x4xf64>, tensor<f64>) -> tensor<2x4xf64> loc(#loc5)
    %c_8 = stablehlo.constant dense<0> : tensor<i32> loc(#loc)
    %9 = stablehlo.broadcast_in_dim %c_8, dims = [] : (tensor<i32>) -> tensor<2xi32> loc(#loc3)
    %10 = stablehlo.compare  EQ, %0#4, %9,  SIGNED : (tensor<2xi32>, tensor<2xi32>) -> tensor<2xi1> loc(#loc3)
    %11 = stablehlo.broadcast_in_dim %10, dims = [0] : (tensor<2xi1>) -> tensor<2x1xi1> loc(#loc4)
    %cst_9 = stablehlo.constant dense<0x7FF8000000000000> : tensor<f64> loc(#loc)
    %12 = call @_where_1(%11, %0#2, %cst_9) : (tensor<2x1xi1>, tensor<2x3xf64>, tensor<f64>) -> tensor<2x3xf64> loc(#loc5)
    %c_10 = stablehlo.constant dense<0> : tensor<i32> loc(#loc)
    %13 = stablehlo.broadcast_in_dim %c_10, dims = [] : (tensor<i32>) -> tensor<2xi32> loc(#loc3)
    %14 = stablehlo.compare  EQ, %0#4, %13,  SIGNED : (tensor<2xi32>, tensor<2xi32>) -> tensor<2xi1> loc(#loc3)
    %15 = stablehlo.broadcast_in_dim %14, dims = [0] : (tensor<2xi1>) -> tensor<2x1xi1> loc(#loc4)
    %cst_11 = stablehlo.constant dense<0x7FF8000000000000> : tensor<f64> loc(#loc)
    %16 = call @_where_1(%15, %0#3, %cst_11) : (tensor<2x1xi1>, tensor<2x3xf64>, tensor<f64>) -> tensor<2x3xf64> loc(#loc5)
    return %4, %8, %12, %16 : tensor<2x4x4xf64>, tensor<2x4xf64>, tensor<2x3xf64>, tensor<2x3xf64> loc(#loc)
  } loc(#loc)
  func.func private @_where(%arg0: tensor<2x1x1xi1> {mhlo.layout_mode = "default"} loc("jit(func)/jit(main)/pjit"(#loc1)), %arg1: tensor<2x4x4xf64> {mhlo.layout_mode = "default"} loc("jit(func)/jit(main)/pjit"(#loc1)), %arg2: tensor<f64> {mhlo.layout_mode = "default"} loc("jit(func)/jit(main)/pjit"(#loc1))) -> (tensor<2x4x4xf64> {mhlo.layout_mode = "default"}) {
    %0 = stablehlo.broadcast_in_dim %arg0, dims = [0, 1, 2] : (tensor<2x1x1xi1>) -> tensor<2x4x4xi1> loc(#loc6)
    %1 = stablehlo.broadcast_in_dim %arg2, dims = [] : (tensor<f64>) -> tensor<2x4x4xf64> loc(#loc6)
    %2 = stablehlo.select %0, %arg1, %1 : tensor<2x4x4xi1>, tensor<2x4x4xf64> loc(#loc7)
    return %2 : tensor<2x4x4xf64> loc(#loc5)
  } loc(#loc5)
  func.func private @_where_0(%arg0: tensor<2x1xi1> {mhlo.layout_mode = "default"} loc("jit(func)/jit(main)/pjit"(#loc1)), %arg1: tensor<2x4xf64> {mhlo.layout_mode = "default"} loc("jit(func)/jit(main)/pjit"(#loc1)), %arg2: tensor<f64> {mhlo.layout_mode = "default"} loc("jit(func)/jit(main)/pjit"(#loc1))) -> (tensor<2x4xf64> {mhlo.layout_mode = "default"}) {
    %0 = stablehlo.broadcast_in_dim %arg0, dims = [0, 1] : (tensor<2x1xi1>) -> tensor<2x4xi1> loc(#loc6)
    %1 = stablehlo.broadcast_in_dim %arg2, dims = [] : (tensor<f64>) -> tensor<2x4xf64> loc(#loc6)
    %2 = stablehlo.select %0, %arg1, %1 : tensor<2x4xi1>, tensor<2x4xf64> loc(#loc7)
    return %2 : tensor<2x4xf64> loc(#loc5)
  } loc(#loc5)
  func.func private @_where_1(%arg0: tensor<2x1xi1> {mhlo.layout_mode = "default"} loc("jit(func)/jit(main)/pjit"(#loc1)), %arg1: tensor<2x3xf64> {mhlo.layout_mode = "default"} loc("jit(func)/jit(main)/pjit"(#loc1)), %arg2: tensor<f64> {mhlo.layout_mode = "default"} loc("jit(func)/jit(main)/pjit"(#loc1))) -> (tensor<2x3xf64> {mhlo.layout_mode = "default"}) {
    %0 = stablehlo.broadcast_in_dim %arg0, dims = [0, 1] : (tensor<2x1xi1>) -> tensor<2x3xi1> loc(#loc6)
    %1 = stablehlo.broadcast_in_dim %arg2, dims = [] : (tensor<f64>) -> tensor<2x3xf64> loc(#loc6)
    %2 = stablehlo.select %0, %arg1, %1 : tensor<2x3xi1>, tensor<2x3xf64> loc(#loc7)
    return %2 : tensor<2x3xf64> loc(#loc5)
  } loc(#loc5)
} loc(#loc)
#loc = loc(unknown)
#loc2 = loc("jit(func)/jit(main)/tridiagonal"(#loc1))
#loc3 = loc("jit(func)/jit(main)/eq"(#loc1))
#loc4 = loc("jit(func)/jit(main)/broadcast_in_dim"(#loc1))
#loc6 = loc("jit(func)/jit(main)/jit(_where)/broadcast_in_dim"(#loc1))
#loc7 = loc("jit(func)/jit(main)/jit(_where)/select_n"(#loc1))
""",
    mlir_module_serialized=b'ML\xefR\rStableHLO_v1.3.0\x00\x01#\x05\x01\x05\x13\x01\x03\x0b\x03\x11\x0f\x13\x17\x1b\x1f#\'+\x03\xe9\x93A\x01-\x0f\x07\x0f\x17\x0f\x0f\x0f\x0f\x0f#\x0b\x0f\x0b\x0b\x0b\x0b\x0b\x0b\x0b\x0b\x0b\x0b\x03g\x0f\x0b\x0b\x0f\x0b\x13\x1f\x0b\x0b//\x17\x0f\x0b\x0bO\x0b\x0b\x0bO\x1fo/\x0b\x1b\x1b\x0b\x1b\x0b\x1b\x0b\x1b\x0b\x0b\x0b\x0b\x0b\x0bo&\x08\x1f\x1f\x1f\x0b\x0b\x0b\x0b#\x0f\x17#\x01\x05\x0b\x0f\x03=\x0f\x17\x0f\x1b\x17\x17\x07\x07\x13\x07\x07\x13\x1b\x07\x1f\x1f\x1f\x1f\x17\x13\x13\x17\x1b\x13\x17\x13\x13\x13\x13\x13\x02r\x0b\x1d\x1f\x07\x1f\x1d)\x07\x17!\xde\n\x1b\x1d#\x07\x1d\'\x07\x1d+\x07\x1d%\x07\x11\x03\x05\x03\x07\x15\x17\x19\x11\x1b\x11\x05\x17\x11\x01\x00\x05\x19\x05\x1b\x05\x1d\x05\x1f\x05!\x05#\x05%\x05\'\x05)\x05+\x1f-\x01\x1d-\x1d/\x1f7\x01\x1d1\r\x03/1\x1f\x05\t\x00\x00\x00\x00\t\x07\x07\x01\x1f?\x11\x00\x00\x00\x00\x00\x00\x00\x00\x1f\t\x11\x00\x00\x00\x00\x00\x00\xf8\x7f\x03\x07777\x03\x037\x1d3\x1d5\x1f;!\x01\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x03\x01\x1d7\x1d9\x1f+!\x00\x00\x00\x00\x00\x00\x00\x00\x01\x00\x00\x00\x00\x00\x00\x00\x1f\x05\t\x04\x00\x00\x00\x1f91\x01\x00\x00\x00\x00\x00\x00\x00\x02\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x1f=\x11\x00\x00\x00\x00\x00\x00\x00\x00#!\x03\t_cgk\r\x055a/1\x1d;\r\x055e/1\x1d=\r\x055i/1\x1d?\r\x055m/1\x1dA\x1dC\x1dE###%#\'\x1f31\x00\x00\x00\x00\x00\x00\x00\x00\x01\x00\x00\x00\x00\x00\x00\x00\x02\x00\x00\x00\x00\x00\x00\x00\x1f\x0b\x02\x04A\xa4\x17\xf4kg\xea?\x1f\x01\x943#\x92\x05\xc0\x86 \xf6\x91#\xdd\xe9?\x9dMlS\xe9\xd4\xf8\xbf\x88\x1c:\xa0.\xe1\xee?8\xce\x7f\xa9|\x86\x10\xc0\xe8V\xc7\x14Z\xb0\x01\xc0\xd2!R\xd5\xbf5\x1e@\xbf\xc5\r\xdd\xa6\x1a\xea?\xbcM\xfe\x8c\x1b\t\xf1?\xdbj\xd8\xf2U\x1a\xf0\xbf\xado;\xba\xb8P\x13@\xbb\xad\x83\xbb\xb0r\x02\xc0\x1f9\xf7\xd1~X\x07\xc0)ID\xf4vs\xfc\xbfD\xcfI\xb4D\xb9\x04@\x16\xc3\xfe\x99n\x00\x10\xc0\x82.\x1c\x18\xed\n\r@\x8cn\xd7\xc1\x15t\x06@|2(Pl\x96\xf1?\x88*\xd7\xe3\x06\x16\xe3?F{\xf2\t\xa1c\xf8\xbf8z5!\xf8/\xeb\xbf4\xd3\x1f\xa1\xda3\x02\xc0)\x13I\x84(\x18\x06@\xbcw\xfd\xad\x97i\xe2\xbf\x1e\xf0.Yw\x13\xc0?dW\xd7\xb3\xd6[\x11\xc0\x04\x97\xb3@\xae9\x08\xc0\xbc\x17\xd1C\x15o\xd8?\x02\xb7%t\x18\xe2\x0e@\xac\xd8\xd0T\x13"\x04\xc0\x1f\x05\t\x01\x00\x00\x00\x1f\x05\t\x02\x00\x00\x00\x1f\x05\t\x80\x00\x00\x00\x0b\x05\x1dG\x1dI\x05\x01\x03\r33333W\x03\x03\x8f\x15\x03\x01\x15\x01\x03\rWKKKYY\x01\t\x01\x02\x02)\x01\x1f)\x05\t\r\x13)\x01\x13)\x07\t\x11\x11\x13)\x05\t\x11\x13)\x05\t\x05\x11\x01\x0b)\x03\t\x1f\x1d\x13)\x03\t\x11)\x07\t\x05\x05\x11\x1b\x11\x01\t\x0b\r\x07\x07\x11\x07\x1d\x0b\t\x03\x0b\x11\x07\x0f\r\t\x03\r\x11\x07\x0f\x07\t\x03\x07)\x05\t\r\x11)\x03\t\x17)\x03\x01\x17)\x05\t\x11\x11)\x07\t\x11\x11\x11)\x03\r\x17)\x03\x02\x04\x13)\x03\x01\x19)\x03\r\x19)\x03\t\x19)\x03\x05\x19)\x03\x05\x17\x04\x8a\x06\x05\x01Q\x03\x13\x01\x07\x04b\x06\x03\x01\x11\x07P\x03\x03\x07\x04\xf6\x03\x03I\x81\x05B\x03\x05\x03\x0b\x05B\x0b\x07\x03\x05\x05B\x0b\t\x03\x05\x05B\x0b\x07\x03\x05\x05B\x0b\x0b\x03\x05\x05B\x0b\r\x03\x05\x11F\x0b\x0f\r\x0b\r\x07\x07\x155\r\x03\x05\x07\t\x0b\x01\x05B\x03\x11\x03\x05\x03F\x05\x13\x03\x15\x03\x19\x0bF\x05\x15\x03\x1b\x05\x15\x1b\x03F\r\x17\x03\x1d\x03\x1d\x05B\x03\x19\x03\t\rF\x01\x1b\x03\x0b\x07\x1f\r!\x05B\x03\x11\x03\x05\x03F\x05\x13\x03\x15\x03%\x0bF\x05\x15\x03\x1b\x05\x15\'\x03F\r\x17\x03\x0f\x03)\x05B\x03\x19\x03\t\rF\x01\x1d\x03\r\x07+\x0f-\x05B\x03\x11\x03\x05\x03F\x05\x13\x03\x15\x031\x0bF\x05\x15\x03\x1b\x05\x153\x03F\r\x17\x03\x0f\x035\x05B\x03\x19\x03\t\rF\x01\x1f\x03\x07\x077\x119\x05B\x03\x11\x03\x05\x03F\x05\x13\x03\x15\x03=\x0bF\x05\x15\x03\x1b\x05\x15?\x03F\r\x17\x03\x0f\x03A\x05B\x03\x19\x03\t\rF\x01\x1f\x03\x07\x07C\x13E\t\x04\x03\t#/;G\x07P\x01!\x07\x04S\x03\r\x13\x07;\x01\x17\x01\x13\x01\x00\x03F\t#\x031\x03\x01\x03F\t\x13\x03\x0b\x03\x05\x0f\x06\x0f\x03\x0b\x07\x07\x03\t\t\x04\x01\x03\x0b\x07P\x01%\x07\x04S\x03\r\x13\x07\x1f\x01\x1b\x01\x13\x01\x00\x03F\t\'\x03/\x03\x01\x03F\t\x13\x03\r\x03\x05\x0f\x06\x0f\x03\r\x07\x07\x03\t\t\x04\x01\x03\x0b\x07P\x01)\x07\x04S\x03\r\x13\x07\x1f\x01\x0f\x01\x13\x01\x00\x03F\t\'\x03)\x03\x01\x03F\t\x13\x03\x07\x03\x05\x0f\x06\x0f\x03\x07\x07\x07\x03\t\t\x04\x01\x03\x0b\x06\x03\x01\x05\x01\x00n\tK\x1d\x03\x0f\x0b\t\t\t\t\x13\x0f\x13\x11!\x11#K/ASci3\x13%)9\x1f\x15\x11\x17\x15\x11\x19)\x0f\x0b\x11builtin\x00vhlo\x00module\x00broadcast_in_dim_v1\x00constant_v1\x00func_v1\x00return_v1\x00compare_v1\x00call_v1\x00select_v1\x00custom_call_v1\x00jax.uses_shape_polymorphism\x00mhlo.num_partitions\x00mhlo.num_replicas\x00jit_func\x00jit(func)/jit(main)/pjit\x00third_party/py/jax/tests/export_back_compat_test.py\x00jit(func)/jit(main)/jit(_where)/broadcast_in_dim\x00jit(func)/jit(main)/jit(_where)/select_n\x00jit(func)/jit(main)/tridiagonal\x00jit(func)/jit(main)/eq\x00jit(func)/jit(main)/broadcast_in_dim\x00mhlo.layout_mode\x00default\x00jax.result_info\x00private\x00_where_1\x00_where\x00_where_0\x00[0]\x00[1]\x00[2]\x00[3]\x00main\x00public\x00\x00lapack_dsytrd\x00\x08\x89+\x05;\x01\x0bM[]oq\x03{\x03U\x03}\x03\x7f\x03\x81\x11\x83\x85\x87M\x89\x8b\x8d\x91\x039\x03-\x05;=\x03?\x03A\x03O\x03Q\x03I\x0bCsEOG\x03y\x0bCuEQG\x03S\x0bCwEIG',
    xla_call_module_version=9,
    nr_devices=1,
)  # End paste

data_2024_09_04 = {}


# Pasted from the test output (see export_back_compat_test_util.py module docstring)
data_2024_09_04["c128"] = dict(
    testdata_version=1,
    platform='cpu',
    custom_call_targets=['lapack_zhetrd_ffi'],
    serialized_date=datetime.date(2024, 9, 4),
    inputs=(),
    expected_outputs=(array([[[-1.6782909868280393 +0.j                 ,
         -0.44670237330570184+4.847000766107959j  ,
          2.05945450900321   -2.2848432268240106j ,
         -1.852046418980849  +1.672382006137275j  ],
        [ 8.516713699516982  +0.j                 ,
         -2.7881860505313174 +0.j                 ,
          0.9238284715039695 -2.3790501284019947j ,
          0.5005102262291599 -1.30066052934836j   ],
        [-0.12132810525381293-0.2963030371159077j ,
         -3.6374350042782893 +0.j                 ,
          0.5605752523031344 +0.j                 ,
         -2.9865099107523174 +0.5492956557924651j ],
        [-0.40379248092949666-0.7813328344426929j ,
         -0.07101654492399719-0.27208840961051617j,
         -7.4654253782049285 +0.j                 ,
         -8.172380353916964  +0.j                 ]],

       [[-3.996403598623405  +0.j                 ,
          0.59408630943699   +2.531609474375295j  ,
         -1.789098034543644  -2.538389274566601j  ,
         -1.291106590337488  +3.1576544511573843j ],
        [10.8950662522622    +0.j                 ,
         -2.8151642043836693 +0.j                 ,
          6.18998567202382   +1.1866537964613415j ,
          3.1900218245393352 +2.7291222716752372j ],
        [-0.3142889671188478 -0.37781876498252764j,
          3.049208563595754  +0.j                 ,
         -2.4383044880335487 +0.j                 ,
          4.075435464493341  -0.6653616942280807j ],
        [ 0.32757687545025194+0.565870910342534j  ,
          0.8177026465997795 -0.15906305615104555j,
          3.3415143060767125 +0.j                 ,
          4.094619408678314  +0.j                 ]]]), array([[-1.6782909868280393, -2.7881860505313174,  0.5605752523031344,
        -8.172380353916964 ],
       [-3.996403598623405 , -2.8151642043836693, -2.4383044880335487,
         4.094619408678314 ]]), array([[ 8.516713699516982 , -3.6374350042782893, -7.4654253782049285],
       [10.8950662522622   ,  3.049208563595754 ,  3.3415143060767125]]), array([[1.0626274644222748+0.06050271598884928j,
        1.834630852474663 +0.18575551495730305j,
        1.981584368497257 +0.19102912741736966j],
       [1.0365789616521406-0.40942548304121656j,
        1.0872592163018966-0.3187050677167622j ,
        1.0458498304770472-0.9989483435319496j ]])),
    mlir_module_text=r"""
#loc1 = loc("third_party/py/jax/tests/export_back_compat_test.py":695:13)
#loc5 = loc("jit(func)/jit(main)/pjit"(#loc1))
module @jit_func attributes {jax.uses_shape_polymorphism = false, mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} {
  func.func public @main() -> (tensor<2x4x4xcomplex<f64>> {jax.result_info = "[0]", mhlo.layout_mode = "default"}, tensor<2x4xf64> {jax.result_info = "[1]", mhlo.layout_mode = "default"}, tensor<2x3xf64> {jax.result_info = "[2]", mhlo.layout_mode = "default"}, tensor<2x3xcomplex<f64>> {jax.result_info = "[3]", mhlo.layout_mode = "default"}) {
    %cst = stablehlo.constant dense<[[[(-1.6782909868280393,-0.44303325034407437), (-0.44670237330570184,4.8470007661079588), (2.0594545090032099,-2.2848432268240106), (-1.852046418980849,1.6723820061372749)], [(-0.53338018421119981,-0.5152843101202178), (-8.6208093221459947,-1.4723511111926109), (0.92382847150396952,-2.3790501284019947), (0.50051022622915986,-1.30066052934836)], [(0.94535043721506584,2.744088772946665), (-5.9178492824175759,-4.3744650461123786), (1.8341291553102983,-4.8378584827626838), (-2.9865099107523174,0.54929565579246509)], [(3.2517513113853891,7.2792034361133062), (-0.09841002311276037,0.88008791818205689), (-0.035759860211603468,2.4677764344580244), (-3.6133109853094476,-2.2833696560058976)]], [[(-3.996403598623405,2.42308766118121), (0.59408630943699003,2.531609474375295), (-1.789098034543644,-2.538389274566601), (-1.2911065903374881,3.1576544511573843)], [(-0.39853021063902833,4.4607177630985086), (1.0742061295773189,-2.6002112528615386), (6.1899856720238198,1.1866537964613415), (3.1900218245393352,2.7291222716752372)], [(5.2347956435718022,2.8649782894514577), (2.3527586611916762,2.4688953673448575), (-2.317572140163894,4.3609023810820053), (4.0754354644933413,-0.66536169422808067)], [(-6.2237114632988675,-4.9294897244018943), (4.2994486027667103,-1.3300494261380422), (-0.51942958410141249,0.60038999428238982), (0.084516726847668963,-7.2944134049318752)]]]> : tensor<2x4x4xcomplex<f64>> loc(#loc)
    %0:5 = stablehlo.custom_call @lapack_zhetrd_ffi(%cst) {mhlo.backend_config = {uplo = 76 : ui8}, operand_layouts = [dense<[1, 2, 0]> : tensor<3xindex>], output_operand_aliases = [#stablehlo.output_operand_alias<output_tuple_indices = [0], operand_index = 0, operand_tuple_indices = []>], result_layouts = [dense<[1, 2, 0]> : tensor<3xindex>, dense<[1, 0]> : tensor<2xindex>, dense<[1, 0]> : tensor<2xindex>, dense<[1, 0]> : tensor<2xindex>, dense<0> : tensor<1xindex>]} : (tensor<2x4x4xcomplex<f64>>) -> (tensor<2x4x4xcomplex<f64>>, tensor<2x4xf64>, tensor<2x3xf64>, tensor<2x3xcomplex<f64>>, tensor<2xi32>) loc(#loc2)
    %c = stablehlo.constant dense<0> : tensor<i32> loc(#loc)
    %1 = stablehlo.broadcast_in_dim %c, dims = [] : (tensor<i32>) -> tensor<2xi32> loc(#loc3)
    %2 = stablehlo.compare  EQ, %0#4, %1,  SIGNED : (tensor<2xi32>, tensor<2xi32>) -> tensor<2xi1> loc(#loc3)
    %3 = stablehlo.broadcast_in_dim %2, dims = [0] : (tensor<2xi1>) -> tensor<2x1x1xi1> loc(#loc4)
    %cst_0 = stablehlo.constant dense<(0x7FF8000000000000,0x7FF8000000000000)> : tensor<complex<f64>> loc(#loc)
    %4 = call @_where(%3, %0#0, %cst_0) : (tensor<2x1x1xi1>, tensor<2x4x4xcomplex<f64>>, tensor<complex<f64>>) -> tensor<2x4x4xcomplex<f64>> loc(#loc5)
    %c_1 = stablehlo.constant dense<0> : tensor<i32> loc(#loc)
    %5 = stablehlo.broadcast_in_dim %c_1, dims = [] : (tensor<i32>) -> tensor<2xi32> loc(#loc3)
    %6 = stablehlo.compare  EQ, %0#4, %5,  SIGNED : (tensor<2xi32>, tensor<2xi32>) -> tensor<2xi1> loc(#loc3)
    %7 = stablehlo.broadcast_in_dim %6, dims = [0] : (tensor<2xi1>) -> tensor<2x1xi1> loc(#loc4)
    %cst_2 = stablehlo.constant dense<0x7FF8000000000000> : tensor<f64> loc(#loc)
    %8 = call @_where_0(%7, %0#1, %cst_2) : (tensor<2x1xi1>, tensor<2x4xf64>, tensor<f64>) -> tensor<2x4xf64> loc(#loc5)
    %c_3 = stablehlo.constant dense<0> : tensor<i32> loc(#loc)
    %9 = stablehlo.broadcast_in_dim %c_3, dims = [] : (tensor<i32>) -> tensor<2xi32> loc(#loc3)
    %10 = stablehlo.compare  EQ, %0#4, %9,  SIGNED : (tensor<2xi32>, tensor<2xi32>) -> tensor<2xi1> loc(#loc3)
    %11 = stablehlo.broadcast_in_dim %10, dims = [0] : (tensor<2xi1>) -> tensor<2x1xi1> loc(#loc4)
    %cst_4 = stablehlo.constant dense<0x7FF8000000000000> : tensor<f64> loc(#loc)
    %12 = call @_where_1(%11, %0#2, %cst_4) : (tensor<2x1xi1>, tensor<2x3xf64>, tensor<f64>) -> tensor<2x3xf64> loc(#loc5)
    %c_5 = stablehlo.constant dense<0> : tensor<i32> loc(#loc)
    %13 = stablehlo.broadcast_in_dim %c_5, dims = [] : (tensor<i32>) -> tensor<2xi32> loc(#loc3)
    %14 = stablehlo.compare  EQ, %0#4, %13,  SIGNED : (tensor<2xi32>, tensor<2xi32>) -> tensor<2xi1> loc(#loc3)
    %15 = stablehlo.broadcast_in_dim %14, dims = [0] : (tensor<2xi1>) -> tensor<2x1xi1> loc(#loc4)
    %cst_6 = stablehlo.constant dense<(0x7FF8000000000000,0x7FF8000000000000)> : tensor<complex<f64>> loc(#loc)
    %16 = call @_where_2(%15, %0#3, %cst_6) : (tensor<2x1xi1>, tensor<2x3xcomplex<f64>>, tensor<complex<f64>>) -> tensor<2x3xcomplex<f64>> loc(#loc5)
    return %4, %8, %12, %16 : tensor<2x4x4xcomplex<f64>>, tensor<2x4xf64>, tensor<2x3xf64>, tensor<2x3xcomplex<f64>> loc(#loc)
  } loc(#loc)
  func.func private @_where(%arg0: tensor<2x1x1xi1> {mhlo.layout_mode = "default"} loc("jit(func)/jit(main)/pjit"(#loc1)), %arg1: tensor<2x4x4xcomplex<f64>> {mhlo.layout_mode = "default"} loc("jit(func)/jit(main)/pjit"(#loc1)), %arg2: tensor<complex<f64>> {mhlo.layout_mode = "default"} loc("jit(func)/jit(main)/pjit"(#loc1))) -> (tensor<2x4x4xcomplex<f64>> {mhlo.layout_mode = "default"}) {
    %0 = stablehlo.broadcast_in_dim %arg0, dims = [0, 1, 2] : (tensor<2x1x1xi1>) -> tensor<2x4x4xi1> loc(#loc6)
    %1 = stablehlo.broadcast_in_dim %arg2, dims = [] : (tensor<complex<f64>>) -> tensor<2x4x4xcomplex<f64>> loc(#loc6)
    %2 = stablehlo.select %0, %arg1, %1 : tensor<2x4x4xi1>, tensor<2x4x4xcomplex<f64>> loc(#loc7)
    return %2 : tensor<2x4x4xcomplex<f64>> loc(#loc5)
  } loc(#loc5)
  func.func private @_where_0(%arg0: tensor<2x1xi1> {mhlo.layout_mode = "default"} loc("jit(func)/jit(main)/pjit"(#loc1)), %arg1: tensor<2x4xf64> {mhlo.layout_mode = "default"} loc("jit(func)/jit(main)/pjit"(#loc1)), %arg2: tensor<f64> {mhlo.layout_mode = "default"} loc("jit(func)/jit(main)/pjit"(#loc1))) -> (tensor<2x4xf64> {mhlo.layout_mode = "default"}) {
    %0 = stablehlo.broadcast_in_dim %arg0, dims = [0, 1] : (tensor<2x1xi1>) -> tensor<2x4xi1> loc(#loc6)
    %1 = stablehlo.broadcast_in_dim %arg2, dims = [] : (tensor<f64>) -> tensor<2x4xf64> loc(#loc6)
    %2 = stablehlo.select %0, %arg1, %1 : tensor<2x4xi1>, tensor<2x4xf64> loc(#loc7)
    return %2 : tensor<2x4xf64> loc(#loc5)
  } loc(#loc5)
  func.func private @_where_1(%arg0: tensor<2x1xi1> {mhlo.layout_mode = "default"} loc("jit(func)/jit(main)/pjit"(#loc1)), %arg1: tensor<2x3xf64> {mhlo.layout_mode = "default"} loc("jit(func)/jit(main)/pjit"(#loc1)), %arg2: tensor<f64> {mhlo.layout_mode = "default"} loc("jit(func)/jit(main)/pjit"(#loc1))) -> (tensor<2x3xf64> {mhlo.layout_mode = "default"}) {
    %0 = stablehlo.broadcast_in_dim %arg0, dims = [0, 1] : (tensor<2x1xi1>) -> tensor<2x3xi1> loc(#loc6)
    %1 = stablehlo.broadcast_in_dim %arg2, dims = [] : (tensor<f64>) -> tensor<2x3xf64> loc(#loc6)
    %2 = stablehlo.select %0, %arg1, %1 : tensor<2x3xi1>, tensor<2x3xf64> loc(#loc7)
    return %2 : tensor<2x3xf64> loc(#loc5)
  } loc(#loc5)
  func.func private @_where_2(%arg0: tensor<2x1xi1> {mhlo.layout_mode = "default"} loc("jit(func)/jit(main)/pjit"(#loc1)), %arg1: tensor<2x3xcomplex<f64>> {mhlo.layout_mode = "default"} loc("jit(func)/jit(main)/pjit"(#loc1)), %arg2: tensor<complex<f64>> {mhlo.layout_mode = "default"} loc("jit(func)/jit(main)/pjit"(#loc1))) -> (tensor<2x3xcomplex<f64>> {mhlo.layout_mode = "default"}) {
    %0 = stablehlo.broadcast_in_dim %arg0, dims = [0, 1] : (tensor<2x1xi1>) -> tensor<2x3xi1> loc(#loc6)
    %1 = stablehlo.broadcast_in_dim %arg2, dims = [] : (tensor<complex<f64>>) -> tensor<2x3xcomplex<f64>> loc(#loc6)
    %2 = stablehlo.select %0, %arg1, %1 : tensor<2x3xi1>, tensor<2x3xcomplex<f64>> loc(#loc7)
    return %2 : tensor<2x3xcomplex<f64>> loc(#loc5)
  } loc(#loc5)
} loc(#loc)
#loc = loc(unknown)
#loc2 = loc("jit(func)/jit(main)/tridiagonal"(#loc1))
#loc3 = loc("jit(func)/jit(main)/eq"(#loc1))
#loc4 = loc("jit(func)/jit(main)/broadcast_in_dim"(#loc1))
#loc6 = loc("jit(func)/jit(main)/jit(_where)/broadcast_in_dim"(#loc1))
#loc7 = loc("jit(func)/jit(main)/jit(_where)/select_n"(#loc1))
""",
    mlir_module_serialized=b'ML\xefR\rStableHLO_v1.3.0\x00\x01#\x05\x01\x05\x13\x01\x03\x0b\x03\x11\x0f\x13\x17\x1b\x1f#\'+\x03\xf5\x99G\x011\x0f\x07\x0f\x0f\x17\x0f\x0f\x0f#\x0b\x0f\x0b\x0b\x0b\x0b\x0b\x0b\x0b\x13\x0b\x0f\x0b\x0b\x0b\x03i\x0f\x0b\x0b\x0b\x17\x13\x0f\x0b\x1f\x0b\x0b/OO\x0b\x0b\x0b\x0b\x0boO/\x0b\x1b\x1b\x0b\x1b\x0b\x1b\x0b\x1b\x0b\x0b\x0b\x0b\x0b\x0b\x0bo&\x10\x13\x0b\x0f\x0b\x0b\x0b\x0b\x0f\x0f\x17\x1f/\x01\x05\x0b\x0f\x03C\x1b\x17\x17\x17\x17\x0f\x0f\x07\x13\x0f\x07\x07\x13\x0b\x1b\x07\x17\x07\x1f\x1f\x1f\x1f\x1f\x13\x13\x17\x1b\x13\x07\x13\x13\x13\x13\x02\x9a\x0f\x1d\x1d\t\x1f\x1d!\t\x1d-\t\x17\x1f\xde\n\x1b\x1d#\t\x1d/\t\x11\x03\x05\x03\x07\x13\x15\x17\x0f\x19\x0f\x05\x17\x11\x01\x00\x05\x19\x05\x1b\x05\x1d\x05\x1f\x05!\x05#\x05%\x03\x03\'\x81\x05\'\x1d+\t\x05)\x05+\x05-\x1f5\x01\x1d/\x1d1\x1d3\x03\x07;;;\r\x0335\x03\x03;\x1d5\x1f\x17\t\x00\x00\x00\x00\t\x07\x07\x01\x1fE\x11\x00\x00\x00\x00\x00\x00\x00\x00\x1f3!\x00\x00\x00\x00\x00\x00\x00\x00\x01\x00\x00\x00\x00\x00\x00\x00\x1fA!\x01\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x03\x01\x1d7\x1d9\x1d;\x1d=\x1f?1\x01\x00\x00\x00\x00\x00\x00\x00\x02\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x1f\x0f!\x00\x00\x00\x00\x00\x00\xf8\x7f\x00\x00\x00\x00\x00\x00\xf8\x7f\x1f\x11\x11\x00\x00\x00\x00\x00\x00\xf8\x7f#)\x03\taeim\r\x057c35\x1d?\r\x057g35\x1dA\r\x057k35\x1dC\r\x057o35\x1dE\x1dG\x1dI#+#-#/#1\x1f;1\x00\x00\x00\x00\x00\x00\x00\x00\x01\x00\x00\x00\x00\x00\x00\x00\x02\x00\x00\x00\x00\x00\x00\x00\x1f\x05\x02\x08d\x91Y\xa6G\xda\xfa\xbf$-Q"\xa8Z\xdc\xbfL0\x19\x8d\xc5\x96\xdc\xbf\x86{8+Tc\x13@\xf0%\x1eI\xc3y\x00@\xe4\x91\xbd\xe2[G\x02\xc0\x85%\x03m\xfb\xa1\xfd\xbf\x9atl\xa2\x13\xc2\xfa?\x9c\xb0\xf0Qs\x11\xe1\xbf\xd8v\x83\x855}\xe0\xbf\x84V/\xb8\xda=!\xc0\n\xd3\xec\t\xc0\x8e\xf7\xbf\x98$\x07\xba\x00\x90\xed?\xd5?\x08oK\x08\x03\xc0>\xf8\x9e\x05.\x04\xe0?\xf2\xfcKj\x81\xcf\xf4\xbf\xe4"c\x8fO@\xee?y\x03\x89\xd0\xe4\xf3\x05@\xee\x8f\xaa\xae\xe0\xab\x17\xc0\xf20\xda\xc3s\x7f\x11\xc0V*+\xd0\x97X\xfd?P\x91\xf8\x92\xf7Y\x13\xc0\x7f\xe3\xdeN_\xe4\x07\xc0\x14\xd5\xae{\xd4\x93\xe1?\xbc\x00\t1\x96\x03\n@`&l\x81\xe7\x1d\x1d@X/\xde6f1\xb9\xbf\x06KF#\xae)\xec?\xcd\x9a<\xcc\x1dO\xa2\xbf\x91\xb1>\x92\x01\xbe\x03@\xf2s\x01\x97\x0f\xe8\x0c\xc0\xf5\xcaiOWD\x02\xc0F\xa2-s\xa2\xf8\x0f\xc0X\xea\xa0\xc8{b\x03@\x0b\x10\xc1J\xc1\x02\xe3?2|\xd5w\xbc@\x04@\xca>\xbbB%\xa0\xfc\xbf\xe8>6\t\x9fN\x04\xc0\xafdRb_\xa8\xf4\xbf\x80Q>V\xe0B\t@UhJ\xdb\x84\x81\xd9\xbf\t\xc7\xb4e\xc6\xd7\x11@<(;\xc4\xf2/\xf1?\x1a\xda\xad\x8e;\xcd\x04\xc0\x1c4\xa0\x9a\x8b\xc2\x18@z\x9c\xf7\xb0\x88\xfc\xf2?\xaea\x8f)*\x85\t@\x00\x0b\xbd\x0e>\xd5\x05@b\x89\xe9Dn\xf0\x14@a\x8d\xc7\xbcy\xeb\x06@\x8a\x97\t"s\xd2\x02@\xc2\xef\xdf6L\xc0\x03@J\xff Cc\x8a\x02\xc0\xd7.\xcfd\x90q\x11@s\xd4S\xf4>M\x10@t\x10\x97\x9b\xa4J\xe5\xbf\x8eo*\x9e\x14\xe5\x18\xc0\xc5\x18\x81\'\xcc\xb7\x13\xc0\x19\xdd\x8e\xa7\xa22\x11@-95\xe8\xe1G\xf5\xbfZK\x89\xca*\x9f\xe0\xbfR;\xc9\x13e6\xe3?\x7f\x94\xc6a\xe3\xa2\xb5?\xe2\xbe&\xb5z-\x1d\xc0\r\x03\x83\x85\x1dK\x13=L\x0b\x03\x1dM\x1dO\x05\x01\x03\x03W\x03\x03\x93\x15\x03\x01\x01\x01\x03\x0bWKKK\x97\x1fC\x11\x00\x00\x00\x00\x00\x00\x00\x00\x01\t\x01\x02\x02)\x07\t\x11\x11\x1f)\x05\t\x05\x13)\x05\t\x11\x19)\x05\t\r\x19)\x05\t\r\x1f)\x01\x1f)\x01\x19\x01)\x03\t\')\x01\'\x0b\x1d)\x03\t\x13\x03\x19)\x07\t\x05\x05\x13\x13)\x05\t\r\x13\x1b\x11\x01\t\x05\t\x0b\r\x11\x07!\x05\x0f\x03\x05\x11\x07\x07\t\x11\x03\t\x11\x07\x07\x0b\x11\x03\x0b\x11\x07\x07\r\x0f\x03\r)\x03\t\x1b)\x03\x01\x1b)\x05\t\x11\x13)\x07\t\x11\x11\x13)\x03\r\x1b!)\x03\r#)\x03\t#)\x03\x05#)\x03\x05\x1b\x04\xbe\x06\x05\x01Q\x03\x11\x01\x07\x04\x96\x06\x03\x01\x15\x07P\x03\x03\x07\x04j\x03\x03=m\x05B\x03\x05\x03\x05\x11G)%\x07\x0b\x05\t\x0b\r\x15\x03\x01\x05B\x03\t\x03\x17\x03F\x07\x0b\x03\x15\x03\r\rF\x07\r\x03\x1d\x05\x0b\x0f\x03F\r\x0f\x03!\x03\x11\x05B\x03\x11\x03\x0f\x0fF\x01\x13\x03\x05\x07\x13\x03\x15\x05B\x03\t\x03\x17\x03F\x07\x0b\x03\x15\x03\x19\rF\x07\r\x03\x1d\x05\x0b\x1b\x03F\r\x0f\x03\x07\x03\x1d\x05B\x03\x15\x03\x11\x0fF\x01\x17\x03\t\x07\x1f\x05!\x05B\x03\t\x03\x17\x03F\x07\x0b\x03\x15\x03%\rF\x07\r\x03\x1d\x05\x0b\'\x03F\r\x0f\x03\x07\x03)\x05B\x03\x15\x03\x11\x0fF\x01\x19\x03\x0b\x07+\x07-\x05B\x03\t\x03\x17\x03F\x07\x0b\x03\x15\x031\rF\x07\r\x03\x1d\x05\x0b3\x03F\r\x0f\x03\x07\x035\x05B\x03\x11\x03\x0f\x0fF\x01\x1b\x03\r\x077\t9\t\x04\x03\t\x17#/;\x07P\x01\x1d\x07\x04S\x03\r\x13\x07C\x01\x0b\x01\x1f\x01\x00\x03F\x05\x1f\x039\x03\x01\x03F\x05\x0b\x03\x05\x03\x05\x0b\x06\x0b\x03\x05\x07\x07\x03\t\t\x04\x01\x03\x0b\x07P\x01!\x07\x04S\x03\r\x13\x07\x0f\x01\x13\x01#\x01\x00\x03F\x05#\x037\x03\x01\x03F\x05\x0b\x03\t\x03\x05\x0b\x06\x0b\x03\t\x07\x07\x03\t\t\x04\x01\x03\x0b\x07P\x01%\x07\x04S\x03\r\x13\x07\x0f\x01\x17\x01#\x01\x00\x03F\x05#\x03%\x03\x01\x03F\x05\x0b\x03\x0b\x03\x05\x0b\x06\x0b\x03\x0b\x07\x07\x03\t\t\x04\x01\x03\x0b\x07P\x01\'\x07\x04S\x03\r\x13\x07\x0f\x01\x1b\x01\x1f\x01\x00\x03F\x05#\x03%\x03\x01\x03F\x05\x0b\x03\r\x03\x05\x0b\x06\x0b\x03\r\x07\x07\x03\t\t\x04\x01\x03\x0b\x06\x03\x01\x05\x01\x00\x12\nQ%\x03\x0b\x0f\x0b\t\t\t\t\x13\x13\x13\x0f\x11!\x11#K/A)Sci3\x13%)9\x1f\x11\x17\x15\x15\x11\x19)\x0f\x0b\x11builtin\x00vhlo\x00module\x00broadcast_in_dim_v1\x00constant_v1\x00func_v1\x00return_v1\x00select_v1\x00compare_v1\x00call_v1\x00custom_call_v1\x00jax.uses_shape_polymorphism\x00mhlo.num_partitions\x00mhlo.num_replicas\x00jit_func\x00jit(func)/jit(main)/pjit\x00third_party/py/jax/tests/export_back_compat_test.py\x00jit(func)/jit(main)/jit(_where)/broadcast_in_dim\x00jit(func)/jit(main)/jit(_where)/select_n\x00mhlo.backend_config\x00jit(func)/jit(main)/tridiagonal\x00jit(func)/jit(main)/eq\x00jit(func)/jit(main)/broadcast_in_dim\x00mhlo.layout_mode\x00default\x00jax.result_info\x00private\x00_where\x00_where_0\x00_where_1\x00_where_2\x00[0]\x00[1]\x00[2]\x00[3]\x00main\x00public\x00uplo\x00\x00lapack_zhetrd_ffi\x00\x08\x8d)\x057\x01\x0bM]_qs\x03\x7f\x11\x87\x89\x8bM\x8d\x8f\x91\x95\x03A\x031\x05CE\x03G\x03Y\x03O\x03[\x03Q\x03S\x03U\x0b9u=O?\x03}\x0b9w=Q?\x03I\x0b9y=S?\x0b9{=U?',
    xla_call_module_version=9,
    nr_devices=1,
)  # End paste


# Pasted from the test output (see export_back_compat_test_util.py module docstring)
data_2024_09_04["c64"] = dict(
    testdata_version=1,
    platform='cpu',
    custom_call_targets=['lapack_chetrd_ffi'],
    serialized_date=datetime.date(2024, 9, 4),
    inputs=(),
    expected_outputs=(array([[[ 3.3228416  +0.j         , -1.9756439  +4.593356j   ,
          7.367708   +0.88518727j , -8.659938   +1.6132793j  ],
        [-6.9206004  +0.j         , -3.6362798  +0.j         ,
          3.3011198  -4.644362j   , -4.8589935  -0.61439794j ],
        [ 0.64957    +0.060723424j,  6.620491   +0.j         ,
          0.2882607  +0.j         , -1.0288142  +1.8544064j  ],
        [-0.05458622 +0.10473086j , -0.15611424 +0.06925995j ,
         -4.431866   +0.j         ,  2.364208   +0.j         ]],

       [[-4.1803885  +0.j         ,  0.5670845  +0.6913016j  ,
          2.675204   -0.23881845j , -0.41825035 -1.4060576j  ],
        [ 8.33625    +0.j         ,  2.6144838  +0.j         ,
         -2.4941807  -1.9316154j  ,  0.6687787  -2.209776j   ],
        [ 0.019031923+0.17462212j ,  2.7034955  +0.j         ,
         -0.70924187 +0.j         ,  2.7962255  +1.5316825j  ],
        [-0.057821754+0.023692288j, -0.62805307 -0.0882424j  ,
          6.6364865  +0.j         , -1.698973   +0.j         ]]],
      dtype=complex64), array([[ 3.3228416 , -3.6362798 ,  0.2882607 ,  2.364208  ],
       [-4.1803885 ,  2.6144838 , -0.70924187, -1.698973  ]],
      dtype=float32), array([[-6.9206004,  6.620491 , -4.431866 ],
       [ 8.33625  ,  2.7034955,  6.6364865]], dtype=float32), array([[1.360567 +0.1977107j , 1.7586378-0.56989706j,
        1.5772758-0.8165493j ],
       [1.9152443-0.1834492j , 1.1593437+0.55631363j,
        1.6889225-0.724835j  ]], dtype=complex64)),
    mlir_module_text=r"""
#loc1 = loc("third_party/py/jax/tests/export_back_compat_test.py":695:13)
#loc5 = loc("jit(func)/jit(main)/pjit"(#loc1))
module @jit_func attributes {jax.uses_shape_polymorphism = false, mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} {
  func.func public @main() -> (tensor<2x4x4xcomplex<f32>> {jax.result_info = "[0]", mhlo.layout_mode = "default"}, tensor<2x4xf32> {jax.result_info = "[1]", mhlo.layout_mode = "default"}, tensor<2x3xf32> {jax.result_info = "[2]", mhlo.layout_mode = "default"}, tensor<2x3xcomplex<f32>> {jax.result_info = "[3]", mhlo.layout_mode = "default"}) {
    %cst = stablehlo.constant dense<[[[(3.32284164,1.14621949), (-1.97564387,4.59335613), (7.36770821,0.885187268), (-8.65993785,1.61327934)], [(2.495340e+00,1.36827672), (-3.96969199,-0.636681795), (3.3011198,-4.64436197), (-4.85899353,-0.614397943)], [(6.03322554,1.46055949), (-3.89591122,-4.1833396), (-1.46423841,-0.106284566), (-1.0288142,1.85440636)], [(-0.657281339,0.911450386), (3.18693113,-2.02812219), (-2.64483237,0.351429433), (4.45011663,-1.79112875)]], [[(-4.18038845,-3.65238023), (0.567084491,0.691301584), (2.67520404,-0.238818452), (-0.418250352,-1.4060576)], [(-7.62970591,1.5292784), (0.269325763,2.48722434), (-2.49418068,-1.93161535), (0.668778717,-2.20977592)], [(-0.570908666,-2.75890398), (-0.235837936,3.45861554), (-0.946199476,0.23120968), (2.79622555,1.53168249)], [(0.886947453,-0.466695577), (-3.194850e+00,-0.0176551137), (-4.37602425,-3.7703948), (0.883143305,-4.70016575)]]]> : tensor<2x4x4xcomplex<f32>> loc(#loc)
    %0:5 = stablehlo.custom_call @lapack_chetrd_ffi(%cst) {mhlo.backend_config = {uplo = 76 : ui8}, operand_layouts = [dense<[1, 2, 0]> : tensor<3xindex>], output_operand_aliases = [#stablehlo.output_operand_alias<output_tuple_indices = [0], operand_index = 0, operand_tuple_indices = []>], result_layouts = [dense<[1, 2, 0]> : tensor<3xindex>, dense<[1, 0]> : tensor<2xindex>, dense<[1, 0]> : tensor<2xindex>, dense<[1, 0]> : tensor<2xindex>, dense<0> : tensor<1xindex>]} : (tensor<2x4x4xcomplex<f32>>) -> (tensor<2x4x4xcomplex<f32>>, tensor<2x4xf32>, tensor<2x3xf32>, tensor<2x3xcomplex<f32>>, tensor<2xi32>) loc(#loc2)
    %c = stablehlo.constant dense<0> : tensor<i32> loc(#loc)
    %1 = stablehlo.broadcast_in_dim %c, dims = [] : (tensor<i32>) -> tensor<2xi32> loc(#loc3)
    %2 = stablehlo.compare  EQ, %0#4, %1,  SIGNED : (tensor<2xi32>, tensor<2xi32>) -> tensor<2xi1> loc(#loc3)
    %3 = stablehlo.broadcast_in_dim %2, dims = [0] : (tensor<2xi1>) -> tensor<2x1x1xi1> loc(#loc4)
    %cst_0 = stablehlo.constant dense<(0x7FC00000,0x7FC00000)> : tensor<complex<f32>> loc(#loc)
    %4 = call @_where(%3, %0#0, %cst_0) : (tensor<2x1x1xi1>, tensor<2x4x4xcomplex<f32>>, tensor<complex<f32>>) -> tensor<2x4x4xcomplex<f32>> loc(#loc5)
    %c_1 = stablehlo.constant dense<0> : tensor<i32> loc(#loc)
    %5 = stablehlo.broadcast_in_dim %c_1, dims = [] : (tensor<i32>) -> tensor<2xi32> loc(#loc3)
    %6 = stablehlo.compare  EQ, %0#4, %5,  SIGNED : (tensor<2xi32>, tensor<2xi32>) -> tensor<2xi1> loc(#loc3)
    %7 = stablehlo.broadcast_in_dim %6, dims = [0] : (tensor<2xi1>) -> tensor<2x1xi1> loc(#loc4)
    %cst_2 = stablehlo.constant dense<0x7FC00000> : tensor<f32> loc(#loc)
    %8 = call @_where_0(%7, %0#1, %cst_2) : (tensor<2x1xi1>, tensor<2x4xf32>, tensor<f32>) -> tensor<2x4xf32> loc(#loc5)
    %c_3 = stablehlo.constant dense<0> : tensor<i32> loc(#loc)
    %9 = stablehlo.broadcast_in_dim %c_3, dims = [] : (tensor<i32>) -> tensor<2xi32> loc(#loc3)
    %10 = stablehlo.compare  EQ, %0#4, %9,  SIGNED : (tensor<2xi32>, tensor<2xi32>) -> tensor<2xi1> loc(#loc3)
    %11 = stablehlo.broadcast_in_dim %10, dims = [0] : (tensor<2xi1>) -> tensor<2x1xi1> loc(#loc4)
    %cst_4 = stablehlo.constant dense<0x7FC00000> : tensor<f32> loc(#loc)
    %12 = call @_where_1(%11, %0#2, %cst_4) : (tensor<2x1xi1>, tensor<2x3xf32>, tensor<f32>) -> tensor<2x3xf32> loc(#loc5)
    %c_5 = stablehlo.constant dense<0> : tensor<i32> loc(#loc)
    %13 = stablehlo.broadcast_in_dim %c_5, dims = [] : (tensor<i32>) -> tensor<2xi32> loc(#loc3)
    %14 = stablehlo.compare  EQ, %0#4, %13,  SIGNED : (tensor<2xi32>, tensor<2xi32>) -> tensor<2xi1> loc(#loc3)
    %15 = stablehlo.broadcast_in_dim %14, dims = [0] : (tensor<2xi1>) -> tensor<2x1xi1> loc(#loc4)
    %cst_6 = stablehlo.constant dense<(0x7FC00000,0x7FC00000)> : tensor<complex<f32>> loc(#loc)
    %16 = call @_where_2(%15, %0#3, %cst_6) : (tensor<2x1xi1>, tensor<2x3xcomplex<f32>>, tensor<complex<f32>>) -> tensor<2x3xcomplex<f32>> loc(#loc5)
    return %4, %8, %12, %16 : tensor<2x4x4xcomplex<f32>>, tensor<2x4xf32>, tensor<2x3xf32>, tensor<2x3xcomplex<f32>> loc(#loc)
  } loc(#loc)
  func.func private @_where(%arg0: tensor<2x1x1xi1> {mhlo.layout_mode = "default"} loc("jit(func)/jit(main)/pjit"(#loc1)), %arg1: tensor<2x4x4xcomplex<f32>> {mhlo.layout_mode = "default"} loc("jit(func)/jit(main)/pjit"(#loc1)), %arg2: tensor<complex<f32>> {mhlo.layout_mode = "default"} loc("jit(func)/jit(main)/pjit"(#loc1))) -> (tensor<2x4x4xcomplex<f32>> {mhlo.layout_mode = "default"}) {
    %0 = stablehlo.broadcast_in_dim %arg0, dims = [0, 1, 2] : (tensor<2x1x1xi1>) -> tensor<2x4x4xi1> loc(#loc6)
    %1 = stablehlo.broadcast_in_dim %arg2, dims = [] : (tensor<complex<f32>>) -> tensor<2x4x4xcomplex<f32>> loc(#loc6)
    %2 = stablehlo.select %0, %arg1, %1 : tensor<2x4x4xi1>, tensor<2x4x4xcomplex<f32>> loc(#loc7)
    return %2 : tensor<2x4x4xcomplex<f32>> loc(#loc5)
  } loc(#loc5)
  func.func private @_where_0(%arg0: tensor<2x1xi1> {mhlo.layout_mode = "default"} loc("jit(func)/jit(main)/pjit"(#loc1)), %arg1: tensor<2x4xf32> {mhlo.layout_mode = "default"} loc("jit(func)/jit(main)/pjit"(#loc1)), %arg2: tensor<f32> {mhlo.layout_mode = "default"} loc("jit(func)/jit(main)/pjit"(#loc1))) -> (tensor<2x4xf32> {mhlo.layout_mode = "default"}) {
    %0 = stablehlo.broadcast_in_dim %arg0, dims = [0, 1] : (tensor<2x1xi1>) -> tensor<2x4xi1> loc(#loc6)
    %1 = stablehlo.broadcast_in_dim %arg2, dims = [] : (tensor<f32>) -> tensor<2x4xf32> loc(#loc6)
    %2 = stablehlo.select %0, %arg1, %1 : tensor<2x4xi1>, tensor<2x4xf32> loc(#loc7)
    return %2 : tensor<2x4xf32> loc(#loc5)
  } loc(#loc5)
  func.func private @_where_1(%arg0: tensor<2x1xi1> {mhlo.layout_mode = "default"} loc("jit(func)/jit(main)/pjit"(#loc1)), %arg1: tensor<2x3xf32> {mhlo.layout_mode = "default"} loc("jit(func)/jit(main)/pjit"(#loc1)), %arg2: tensor<f32> {mhlo.layout_mode = "default"} loc("jit(func)/jit(main)/pjit"(#loc1))) -> (tensor<2x3xf32> {mhlo.layout_mode = "default"}) {
    %0 = stablehlo.broadcast_in_dim %arg0, dims = [0, 1] : (tensor<2x1xi1>) -> tensor<2x3xi1> loc(#loc6)
    %1 = stablehlo.broadcast_in_dim %arg2, dims = [] : (tensor<f32>) -> tensor<2x3xf32> loc(#loc6)
    %2 = stablehlo.select %0, %arg1, %1 : tensor<2x3xi1>, tensor<2x3xf32> loc(#loc7)
    return %2 : tensor<2x3xf32> loc(#loc5)
  } loc(#loc5)
  func.func private @_where_2(%arg0: tensor<2x1xi1> {mhlo.layout_mode = "default"} loc("jit(func)/jit(main)/pjit"(#loc1)), %arg1: tensor<2x3xcomplex<f32>> {mhlo.layout_mode = "default"} loc("jit(func)/jit(main)/pjit"(#loc1)), %arg2: tensor<complex<f32>> {mhlo.layout_mode = "default"} loc("jit(func)/jit(main)/pjit"(#loc1))) -> (tensor<2x3xcomplex<f32>> {mhlo.layout_mode = "default"}) {
    %0 = stablehlo.broadcast_in_dim %arg0, dims = [0, 1] : (tensor<2x1xi1>) -> tensor<2x3xi1> loc(#loc6)
    %1 = stablehlo.broadcast_in_dim %arg2, dims = [] : (tensor<complex<f32>>) -> tensor<2x3xcomplex<f32>> loc(#loc6)
    %2 = stablehlo.select %0, %arg1, %1 : tensor<2x3xi1>, tensor<2x3xcomplex<f32>> loc(#loc7)
    return %2 : tensor<2x3xcomplex<f32>> loc(#loc5)
  } loc(#loc5)
} loc(#loc)
#loc = loc(unknown)
#loc2 = loc("jit(func)/jit(main)/tridiagonal"(#loc1))
#loc3 = loc("jit(func)/jit(main)/eq"(#loc1))
#loc4 = loc("jit(func)/jit(main)/broadcast_in_dim"(#loc1))
#loc6 = loc("jit(func)/jit(main)/jit(_where)/broadcast_in_dim"(#loc1))
#loc7 = loc("jit(func)/jit(main)/jit(_where)/select_n"(#loc1))
""",
    mlir_module_serialized=b'ML\xefR\rStableHLO_v1.3.0\x00\x01#\x05\x01\x05\x13\x01\x03\x0b\x03\x11\x0f\x13\x17\x1b\x1f#\'+\x03\xf5\x99G\x011\x0f\x07\x0f\x0f\x17\x0f\x0f\x0f#\x0b\x0f\x0b\x0b\x0b\x0b\x0b\x0b\x0b\x13\x0b\x0f\x0b\x0b\x0b\x03i\x0f\x0b\x0b\x0b\x17\x13\x0f\x0b\x1f\x0b\x0b/OO\x0b\x0b\x0b\x0b\x0bo/\x1f\x0b\x1b\x1b\x0b\x1b\x0b\x1b\x0b\x1b\x0b\x0b\x0b\x0b\x0b\x0b\x0bo&\x08\x13\x0b\x0f\x0b\x0b\x0b\x0b\x0f\x0f\x17\x1f/\x01\x05\x0b\x0f\x03C\x1b\x17\x17\x17\x17\x0f\x0f\x07\x13\x0f\x07\x07\x13\x0b\x1b\x07\x17\x07\x1f\x1f\x1f\x1f\x1f\x13\x13\x17\x1b\x13\x07\x13\x13\x13\x13\x02j\x0b\x1d\x1d\t\x1f\x1d!\t\x1d-\t\x17\x1f\xde\n\x1b\x1d#\t\x1d/\t\x11\x03\x05\x03\x07\x13\x15\x17\x0f\x19\x0f\x05\x17\x11\x01\x00\x05\x19\x05\x1b\x05\x1d\x05\x1f\x05!\x05#\x05%\x03\x03\'\x81\x05\'\x1d+\t\x05)\x05+\x05-\x1f5\x01\x1d/\x1d1\x1d3\x03\x07;;;\r\x0335\x03\x03;\x1d5\x1f\x17\t\x00\x00\x00\x00\t\x07\x07\x01\x1fE\x11\x00\x00\x00\x00\x00\x00\x00\x00\x1f3!\x00\x00\x00\x00\x00\x00\x00\x00\x01\x00\x00\x00\x00\x00\x00\x00\x1fA!\x01\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x03\x01\x1d7\x1d9\x1d;\x1d=\x1f?1\x01\x00\x00\x00\x00\x00\x00\x00\x02\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x1f\x0f\x11\x00\x00\xc0\x7f\x00\x00\xc0\x7f\x1f\x11\t\x00\x00\xc0\x7f#)\x03\taeim\r\x057c35\x1d?\r\x057g35\x1dA\r\x057k35\x1dC\r\x057o35\x1dE\x1dG\x1dI#+#-#/#1\x1f;1\x00\x00\x00\x00\x00\x00\x00\x00\x01\x00\x00\x00\x00\x00\x00\x00\x02\x00\x00\x00\x00\x00\x00\x00\x1f\x05\x02\x04p\xa9T@R\xb7\x92?\xe6\xe1\xfc\xbf\xc6\xfc\x92@D\xc4\xeb@\xa2\x9bb?\x1b\x8f\n\xc1\xf0\x7f\xce?\xa7\xb3\x1f@\xb1#\xaf?o\x0f~\xc0\x94\xfd"\xbf\x8cES@\x9d\x9e\x94\xc0\xe0|\x9b\xc0/I\x1d\xbf/\x10\xc1@\x9d\xf3\xba?\x9cVy\xc0\xeb\xdd\x85\xc0*l\xbb\xbf\xb9\xab\xd9\xbd/\xb0\x83\xbf0]\xed?\x97C(\xbf\xd0Ti?\xae\xf6K@\xc1\xcc\x01\xc0\xefD)\xc0\x8f\xee\xb3>[g\x8e@\xb5C\xe5\xbf\xbe\xc5\x85\xc0\x99\xc0i\xc0s,\x11?$\xf90?\x8b6+@\xd3\x8ct\xbe\xe9$\xd6\xbe\xb2\xf9\xb3\xbf\x8d&\xf4\xc0e\xbf\xc3?\x11\xe5\x89>\xaf.\x1f@\xa8\xa0\x1f\xc0,?\xf7\xbf\x155+?\xf8l\r\xc0\x12\'\x12\xbf\xe2\x910\xc0\x80\x7fq\xbe\xf5Y]@!:r\xbf;\xc2l>\\\xf52@,\x0e\xc4?\xfd\x0ec?\xb9\xf2\xee\xbelxL\xc0u\xa1\x90\xbcd\x08\x8c\xc0&Nq\xc0\xae\x15b?\xc2g\x96\xc0\r\x03\x83\x85\x1dK\x13=L\x0b\x03\x1dM\x1dO\x05\x01\x03\x03W\x03\x03\x93\x15\x03\x01\x01\x01\x03\x0bWKKK\x97\x1fC\x11\x00\x00\x00\x00\x00\x00\x00\x00\x01\t\x01\x02\x02)\x07\t\x11\x11\x1f)\x05\t\x05\x13)\x05\t\x11\x19)\x05\t\r\x19)\x05\t\r\x1f)\x01\x1f)\x01\x19\x01)\x03\t\')\x01\'\t\x1d)\x03\t\x13\x03\x19)\x07\t\x05\x05\x13\x13)\x05\t\r\x13\x1b\x11\x01\t\x05\t\x0b\r\x11\x07!\x05\x0f\x03\x05\x11\x07\x07\t\x11\x03\t\x11\x07\x07\x0b\x11\x03\x0b\x11\x07\x07\r\x0f\x03\r)\x03\t\x1b)\x03\x01\x1b)\x05\t\x11\x13)\x07\t\x11\x11\x13)\x03\r\x1b!)\x03\r#)\x03\t#)\x03\x05#)\x03\x05\x1b\x04\xbe\x06\x05\x01Q\x03\x11\x01\x07\x04\x96\x06\x03\x01\x15\x07P\x03\x03\x07\x04j\x03\x03=m\x05B\x03\x05\x03\x05\x11G)%\x07\x0b\x05\t\x0b\r\x15\x03\x01\x05B\x03\t\x03\x17\x03F\x07\x0b\x03\x15\x03\r\rF\x07\r\x03\x1d\x05\x0b\x0f\x03F\r\x0f\x03!\x03\x11\x05B\x03\x11\x03\x0f\x0fF\x01\x13\x03\x05\x07\x13\x03\x15\x05B\x03\t\x03\x17\x03F\x07\x0b\x03\x15\x03\x19\rF\x07\r\x03\x1d\x05\x0b\x1b\x03F\r\x0f\x03\x07\x03\x1d\x05B\x03\x15\x03\x11\x0fF\x01\x17\x03\t\x07\x1f\x05!\x05B\x03\t\x03\x17\x03F\x07\x0b\x03\x15\x03%\rF\x07\r\x03\x1d\x05\x0b\'\x03F\r\x0f\x03\x07\x03)\x05B\x03\x15\x03\x11\x0fF\x01\x19\x03\x0b\x07+\x07-\x05B\x03\t\x03\x17\x03F\x07\x0b\x03\x15\x031\rF\x07\r\x03\x1d\x05\x0b3\x03F\r\x0f\x03\x07\x035\x05B\x03\x11\x03\x0f\x0fF\x01\x1b\x03\r\x077\t9\t\x04\x03\t\x17#/;\x07P\x01\x1d\x07\x04S\x03\r\x13\x07C\x01\x0b\x01\x1f\x01\x00\x03F\x05\x1f\x039\x03\x01\x03F\x05\x0b\x03\x05\x03\x05\x0b\x06\x0b\x03\x05\x07\x07\x03\t\t\x04\x01\x03\x0b\x07P\x01!\x07\x04S\x03\r\x13\x07\x0f\x01\x13\x01#\x01\x00\x03F\x05#\x037\x03\x01\x03F\x05\x0b\x03\t\x03\x05\x0b\x06\x0b\x03\t\x07\x07\x03\t\t\x04\x01\x03\x0b\x07P\x01%\x07\x04S\x03\r\x13\x07\x0f\x01\x17\x01#\x01\x00\x03F\x05#\x03%\x03\x01\x03F\x05\x0b\x03\x0b\x03\x05\x0b\x06\x0b\x03\x0b\x07\x07\x03\t\t\x04\x01\x03\x0b\x07P\x01\'\x07\x04S\x03\r\x13\x07\x0f\x01\x1b\x01\x1f\x01\x00\x03F\x05#\x03%\x03\x01\x03F\x05\x0b\x03\r\x03\x05\x0b\x06\x0b\x03\r\x07\x07\x03\t\t\x04\x01\x03\x0b\x06\x03\x01\x05\x01\x00\x12\nQ%\x03\x0b\x0f\x0b\t\t\t\t\x13\x13\x13\x0f\x11!\x11#K/A)Sci3\x13%)9\x1f\x11\x17\x15\x15\x11\x19)\x0f\x0b\x11builtin\x00vhlo\x00module\x00broadcast_in_dim_v1\x00constant_v1\x00func_v1\x00return_v1\x00select_v1\x00compare_v1\x00call_v1\x00custom_call_v1\x00jax.uses_shape_polymorphism\x00mhlo.num_partitions\x00mhlo.num_replicas\x00jit_func\x00jit(func)/jit(main)/pjit\x00third_party/py/jax/tests/export_back_compat_test.py\x00jit(func)/jit(main)/jit(_where)/broadcast_in_dim\x00jit(func)/jit(main)/jit(_where)/select_n\x00mhlo.backend_config\x00jit(func)/jit(main)/tridiagonal\x00jit(func)/jit(main)/eq\x00jit(func)/jit(main)/broadcast_in_dim\x00mhlo.layout_mode\x00default\x00jax.result_info\x00private\x00_where\x00_where_0\x00_where_1\x00_where_2\x00[0]\x00[1]\x00[2]\x00[3]\x00main\x00public\x00uplo\x00\x00lapack_chetrd_ffi\x00\x08\x8d)\x057\x01\x0bM]_qs\x03\x7f\x11\x87\x89\x8bM\x8d\x8f\x91\x95\x03A\x031\x05CE\x03G\x03Y\x03O\x03[\x03Q\x03S\x03U\x0b9u=O?\x03}\x0b9w=Q?\x03I\x0b9y=S?\x0b9{=U?',
    xla_call_module_version=9,
    nr_devices=1,
)  # End paste


# Pasted from the test output (see export_back_compat_test_util.py module docstring)
data_2024_09_04["f32"] = dict(
    testdata_version=1,
    platform='cpu',
    custom_call_targets=['lapack_ssytrd_ffi'],
    serialized_date=datetime.date(2024, 9, 4),
    inputs=(),
    expected_outputs=(array([[[-0.8395241 ,  0.156272  , -1.6810869 ,  0.23832119],
        [-2.985257  , -5.571     , -0.22652794, -0.83806676],
        [ 0.27237308, -1.6295947 ,  2.0042834 , -1.148861  ],
        [-0.17183593,  0.57464546,  0.5536146 , -4.206357  ]],

       [[ 1.7666914 ,  2.569005  , -0.86576384, -0.1617768 ],
        [-5.143918  ,  5.0426254 , -3.7237067 ,  4.383015  ],
        [ 0.33311516, -1.5299042 , -8.854181  , -2.896776  ],
        [ 0.3419102 ,  0.2669245 , -2.8250606 ,  5.752488  ]]],
      dtype=float32), array([[-0.8395241, -5.571    ,  2.0042834, -4.206357 ],
       [ 1.7666914,  5.0426254, -8.854181 ,  5.752488 ]], dtype=float32), array([[-2.985257 , -1.6295947,  0.5536146],
       [-5.143918 , -1.5299042, -2.8250606]], dtype=float32), array([[1.8120625, 1.5035137, 0.       ],
       [1.6288393, 1.8669801, 0.       ]], dtype=float32)),
    mlir_module_text=r"""
#loc1 = loc("third_party/py/jax/tests/export_back_compat_test.py":695:13)
#loc5 = loc("jit(func)/jit(main)/pjit"(#loc1))
module @jit_func attributes {jax.uses_shape_polymorphism = false, mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} {
  func.func public @main() -> (tensor<2x4x4xf32> {jax.result_info = "[0]", mhlo.layout_mode = "default"}, tensor<2x4xf32> {jax.result_info = "[1]", mhlo.layout_mode = "default"}, tensor<2x3xf32> {jax.result_info = "[2]", mhlo.layout_mode = "default"}, tensor<2x3xf32> {jax.result_info = "[3]", mhlo.layout_mode = "default"}) {
    %cst = stablehlo.constant dense<[[[-0.83952409, 1.562720e-01, -1.6810869, 0.238321185], [2.42421508, -5.17118931, -0.226527944, -0.838066756], [1.47339451, -1.32866347, -3.3505435, -1.14886105], [-0.929541587, -0.955984473, 2.71886253, 0.748659431]], [[1.76669145, 2.56900501, -0.865763843, -0.161776796], [3.23469758, -0.362713158, -3.72370672, 4.38301516], [2.79104376, 7.36582708, -3.04437494, -2.89677596], [2.86473417, 0.981746375, -2.13533139, 5.34802151]]]> : tensor<2x4x4xf32> loc(#loc)
    %0:5 = stablehlo.custom_call @lapack_ssytrd_ffi(%cst) {mhlo.backend_config = {uplo = 76 : ui8}, operand_layouts = [dense<[1, 2, 0]> : tensor<3xindex>], output_operand_aliases = [#stablehlo.output_operand_alias<output_tuple_indices = [0], operand_index = 0, operand_tuple_indices = []>], result_layouts = [dense<[1, 2, 0]> : tensor<3xindex>, dense<[1, 0]> : tensor<2xindex>, dense<[1, 0]> : tensor<2xindex>, dense<[1, 0]> : tensor<2xindex>, dense<0> : tensor<1xindex>]} : (tensor<2x4x4xf32>) -> (tensor<2x4x4xf32>, tensor<2x4xf32>, tensor<2x3xf32>, tensor<2x3xf32>, tensor<2xi32>) loc(#loc2)
    %c = stablehlo.constant dense<0> : tensor<i32> loc(#loc)
    %1 = stablehlo.broadcast_in_dim %c, dims = [] : (tensor<i32>) -> tensor<2xi32> loc(#loc3)
    %2 = stablehlo.compare  EQ, %0#4, %1,  SIGNED : (tensor<2xi32>, tensor<2xi32>) -> tensor<2xi1> loc(#loc3)
    %3 = stablehlo.broadcast_in_dim %2, dims = [0] : (tensor<2xi1>) -> tensor<2x1x1xi1> loc(#loc4)
    %cst_0 = stablehlo.constant dense<0x7FC00000> : tensor<f32> loc(#loc)
    %4 = call @_where(%3, %0#0, %cst_0) : (tensor<2x1x1xi1>, tensor<2x4x4xf32>, tensor<f32>) -> tensor<2x4x4xf32> loc(#loc5)
    %c_1 = stablehlo.constant dense<0> : tensor<i32> loc(#loc)
    %5 = stablehlo.broadcast_in_dim %c_1, dims = [] : (tensor<i32>) -> tensor<2xi32> loc(#loc3)
    %6 = stablehlo.compare  EQ, %0#4, %5,  SIGNED : (tensor<2xi32>, tensor<2xi32>) -> tensor<2xi1> loc(#loc3)
    %7 = stablehlo.broadcast_in_dim %6, dims = [0] : (tensor<2xi1>) -> tensor<2x1xi1> loc(#loc4)
    %cst_2 = stablehlo.constant dense<0x7FC00000> : tensor<f32> loc(#loc)
    %8 = call @_where_0(%7, %0#1, %cst_2) : (tensor<2x1xi1>, tensor<2x4xf32>, tensor<f32>) -> tensor<2x4xf32> loc(#loc5)
    %c_3 = stablehlo.constant dense<0> : tensor<i32> loc(#loc)
    %9 = stablehlo.broadcast_in_dim %c_3, dims = [] : (tensor<i32>) -> tensor<2xi32> loc(#loc3)
    %10 = stablehlo.compare  EQ, %0#4, %9,  SIGNED : (tensor<2xi32>, tensor<2xi32>) -> tensor<2xi1> loc(#loc3)
    %11 = stablehlo.broadcast_in_dim %10, dims = [0] : (tensor<2xi1>) -> tensor<2x1xi1> loc(#loc4)
    %cst_4 = stablehlo.constant dense<0x7FC00000> : tensor<f32> loc(#loc)
    %12 = call @_where_1(%11, %0#2, %cst_4) : (tensor<2x1xi1>, tensor<2x3xf32>, tensor<f32>) -> tensor<2x3xf32> loc(#loc5)
    %c_5 = stablehlo.constant dense<0> : tensor<i32> loc(#loc)
    %13 = stablehlo.broadcast_in_dim %c_5, dims = [] : (tensor<i32>) -> tensor<2xi32> loc(#loc3)
    %14 = stablehlo.compare  EQ, %0#4, %13,  SIGNED : (tensor<2xi32>, tensor<2xi32>) -> tensor<2xi1> loc(#loc3)
    %15 = stablehlo.broadcast_in_dim %14, dims = [0] : (tensor<2xi1>) -> tensor<2x1xi1> loc(#loc4)
    %cst_6 = stablehlo.constant dense<0x7FC00000> : tensor<f32> loc(#loc)
    %16 = call @_where_1(%15, %0#3, %cst_6) : (tensor<2x1xi1>, tensor<2x3xf32>, tensor<f32>) -> tensor<2x3xf32> loc(#loc5)
    return %4, %8, %12, %16 : tensor<2x4x4xf32>, tensor<2x4xf32>, tensor<2x3xf32>, tensor<2x3xf32> loc(#loc)
  } loc(#loc)
  func.func private @_where(%arg0: tensor<2x1x1xi1> {mhlo.layout_mode = "default"} loc("jit(func)/jit(main)/pjit"(#loc1)), %arg1: tensor<2x4x4xf32> {mhlo.layout_mode = "default"} loc("jit(func)/jit(main)/pjit"(#loc1)), %arg2: tensor<f32> {mhlo.layout_mode = "default"} loc("jit(func)/jit(main)/pjit"(#loc1))) -> (tensor<2x4x4xf32> {mhlo.layout_mode = "default"}) {
    %0 = stablehlo.broadcast_in_dim %arg0, dims = [0, 1, 2] : (tensor<2x1x1xi1>) -> tensor<2x4x4xi1> loc(#loc6)
    %1 = stablehlo.broadcast_in_dim %arg2, dims = [] : (tensor<f32>) -> tensor<2x4x4xf32> loc(#loc6)
    %2 = stablehlo.select %0, %arg1, %1 : tensor<2x4x4xi1>, tensor<2x4x4xf32> loc(#loc7)
    return %2 : tensor<2x4x4xf32> loc(#loc5)
  } loc(#loc5)
  func.func private @_where_0(%arg0: tensor<2x1xi1> {mhlo.layout_mode = "default"} loc("jit(func)/jit(main)/pjit"(#loc1)), %arg1: tensor<2x4xf32> {mhlo.layout_mode = "default"} loc("jit(func)/jit(main)/pjit"(#loc1)), %arg2: tensor<f32> {mhlo.layout_mode = "default"} loc("jit(func)/jit(main)/pjit"(#loc1))) -> (tensor<2x4xf32> {mhlo.layout_mode = "default"}) {
    %0 = stablehlo.broadcast_in_dim %arg0, dims = [0, 1] : (tensor<2x1xi1>) -> tensor<2x4xi1> loc(#loc6)
    %1 = stablehlo.broadcast_in_dim %arg2, dims = [] : (tensor<f32>) -> tensor<2x4xf32> loc(#loc6)
    %2 = stablehlo.select %0, %arg1, %1 : tensor<2x4xi1>, tensor<2x4xf32> loc(#loc7)
    return %2 : tensor<2x4xf32> loc(#loc5)
  } loc(#loc5)
  func.func private @_where_1(%arg0: tensor<2x1xi1> {mhlo.layout_mode = "default"} loc("jit(func)/jit(main)/pjit"(#loc1)), %arg1: tensor<2x3xf32> {mhlo.layout_mode = "default"} loc("jit(func)/jit(main)/pjit"(#loc1)), %arg2: tensor<f32> {mhlo.layout_mode = "default"} loc("jit(func)/jit(main)/pjit"(#loc1))) -> (tensor<2x3xf32> {mhlo.layout_mode = "default"}) {
    %0 = stablehlo.broadcast_in_dim %arg0, dims = [0, 1] : (tensor<2x1xi1>) -> tensor<2x3xi1> loc(#loc6)
    %1 = stablehlo.broadcast_in_dim %arg2, dims = [] : (tensor<f32>) -> tensor<2x3xf32> loc(#loc6)
    %2 = stablehlo.select %0, %arg1, %1 : tensor<2x3xi1>, tensor<2x3xf32> loc(#loc7)
    return %2 : tensor<2x3xf32> loc(#loc5)
  } loc(#loc5)
} loc(#loc)
#loc = loc(unknown)
#loc2 = loc("jit(func)/jit(main)/tridiagonal"(#loc1))
#loc3 = loc("jit(func)/jit(main)/eq"(#loc1))
#loc4 = loc("jit(func)/jit(main)/broadcast_in_dim"(#loc1))
#loc6 = loc("jit(func)/jit(main)/jit(_where)/broadcast_in_dim"(#loc1))
#loc7 = loc("jit(func)/jit(main)/jit(_where)/select_n"(#loc1))
""",
    mlir_module_serialized=b'ML\xefR\rStableHLO_v1.3.0\x00\x01#\x05\x01\x05\x13\x01\x03\x0b\x03\x11\x0f\x13\x17\x1b\x1f#\'+\x03\xe7\x93?\x011\x0f\x07\x0f\x17\x0f\x0f\x0f\x0f#\x0b\x0f\x0b\x0b\x0b\x0b\x0b\x0b\x0b\x13\x0b\x0f\x0b\x0b\x0b\x03c\x0f\x0b\x0b\x0b\x13\x1f\x0b\x0b/\x1f\x17\x0f\x0b\x0bO\x0b\x0b\x0bOo\x0b\x1b\x1b\x0b\x1b\x0b\x1b\x0b\x1b\x0b\x0b\x0b\x0b\x0b\x0bo&\x04\x13\x0b\x0f\x0b\x0b\x0b\x0b\x0f\x0f\x17\x1f/\x01\x05\x0b\x0f\x03;\x17\x0f\x1b\x17\x17\x07\x13\x0f\x07\x07\x13\x1b\x07\x07\x1f\x1f\x1f\x1f\x17\x13\x13\x17\x1b\x13\x07\x13\x13\x13\x13\x02\xea\x08\x1d\x1d\x07\x1f\x1d-\x07\x17\x1f\xde\n\x1b\x1d!\x07\x1d/\x07\x1d#\x07\x11\x03\x05\x03\x07\x13\x15\x17\x0f\x19\x0f\x05\x17\x11\x01\x00\x05\x19\x05\x1b\x05\x1d\x05\x1f\x05!\x05#\x05%\x03\x03\'{\x05\'\x1d+\x07\x05)\x05+\x05-\x1f-\x01\x1d/\x1d1\x1d3\r\x0335\x1f\x13\t\x00\x00\x00\x00\t\x07\x07\x01\x1f=\x11\x00\x00\x00\x00\x00\x00\x00\x00\x1f\x07\t\x00\x00\xc0\x7f\x03\x07999\x03\x039\x1d5\x1d7\x1f9!\x01\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x03\x01\x1d9\x1d;\x1f+!\x00\x00\x00\x00\x00\x00\x00\x00\x01\x00\x00\x00\x00\x00\x00\x00\x1f71\x01\x00\x00\x00\x00\x00\x00\x00\x02\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00#!\x03\t]aei\r\x057_35\x1d=\r\x057c35\x1d?\r\x057g35\x1dA\r\x057k35\x1dC\x1dE\x1dG###%#\'\x1f31\x00\x00\x00\x00\x00\x00\x00\x00\x01\x00\x00\x00\x00\x00\x00\x00\x02\x00\x00\x00\x00\x00\x00\x00\x1f\t\x02\x02\r\xebV\xbf\xc4\x05 >\xdb-\xd7\xbfx\nt>W&\x1b@bz\xa5\xc0\xf1\xf6g\xbe\x8b\x8bV\xbf1\x98\xbc?\xa5\x11\xaa\xbfNoV\xc0\xe1\r\x93\xbfp\xf6m\xbff\xbbt\xbf\xd8\x01.@%\xa8??\xf2"\xe2?\x94j$@\xb3\xa2]\xbf\xd1\xa8%\xbeI\x05O@\x8a\xb5\xb9\xbe6Qn\xc0\xa9A\x8c@v\xa02@\xdb\xb4\xeb@\n\xd7B\xc0\xc7d9\xc0\xceW7@\xbbS{?E\xa9\x08\xc0\xfe"\xab@\r\x03}\x7f\x1dI\x135L\x0b\x03\x1dK\x1dM\x05\x01\x03\x03W\x03\x03\x8d\x15\x03\x01\x01\x01\x03\x0bWMMM\x91\x1f;\x11\x00\x00\x00\x00\x00\x00\x00\x00\x01\t\x01\x02\x02)\x05\t\r\x15)\x01\x15)\x07\t\x11\x11\x15)\x05\t\x11\x15)\x05\t\x05\x0f\x01)\x03\t\x1f)\x01\x1f\t\x1d)\x03\t\x0f)\x07\t\x05\x05\x0f\x13\x1b\x11\x01\t\t\x0b\x05\x05\x11\x07\x1b\t\x07\x03\t\x11\x07\r\x0b\x07\x03\x0b\x11\x07\r\x05\x07\x03\x05)\x05\t\r\x0f)\x03\t\x17)\x03\x01\x17)\x05\t\x11\x0f)\x07\t\x11\x11\x0f)\x03\r\x17!)\x03\r\x1d)\x03\t\x1d)\x03\x05\x1d)\x03\x05\x17\x04\xfe\x05\x05\x01Q\x03\x11\x01\x07\x04\xd6\x05\x03\x01\x11\x07P\x03\x03\x07\x04j\x03\x03=m\x05B\x03\x05\x03\t\x11G)%\x07\x0b\t\x0b\x05\x05\x11\x03\x01\x05B\x03\t\x03\x13\x03F\x05\x0b\x03\x11\x03\r\x0bF\x05\r\x03\x19\x05\x0b\x0f\x03F\x0b\x0f\x03\x1b\x03\x11\x05B\x03\x11\x03\x07\rF\x01\x13\x03\t\x07\x13\x03\x15\x05B\x03\t\x03\x13\x03F\x05\x0b\x03\x11\x03\x19\x0bF\x05\r\x03\x19\x05\x0b\x1b\x03F\x0b\x0f\x03\r\x03\x1d\x05B\x03\x11\x03\x07\rF\x01\x15\x03\x0b\x07\x1f\x05!\x05B\x03\t\x03\x13\x03F\x05\x0b\x03\x11\x03%\x0bF\x05\r\x03\x19\x05\x0b\'\x03F\x0b\x0f\x03\r\x03)\x05B\x03\x11\x03\x07\rF\x01\x17\x03\x05\x07+\x07-\x05B\x03\t\x03\x13\x03F\x05\x0b\x03\x11\x031\x0bF\x05\r\x03\x19\x05\x0b3\x03F\x0b\x0f\x03\r\x035\x05B\x03\x11\x03\x07\rF\x01\x17\x03\x05\x077\t9\t\x04\x03\t\x17#/;\x07P\x01\x19\x07\x04S\x03\r\x13\x077\x01\x13\x01\x0f\x01\x00\x03F\t\x1b\x031\x03\x01\x03F\t\x0b\x03\t\x03\x05\x0f\x06\r\x03\t\x07\x07\x03\t\t\x04\x01\x03\x0b\x07P\x01\x1d\x07\x04S\x03\r\x13\x07\x1b\x01\x17\x01\x0f\x01\x00\x03F\t\x1f\x03/\x03\x01\x03F\t\x0b\x03\x0b\x03\x05\x0f\x06\r\x03\x0b\x07\x07\x03\t\t\x04\x01\x03\x0b\x07P\x01!\x07\x04S\x03\r\x13\x07\x1b\x01\x0b\x01\x0f\x01\x00\x03F\t\x1f\x03)\x03\x01\x03F\t\x0b\x03\x05\x03\x05\x0f\x06\r\x03\x05\x07\x07\x03\t\t\x04\x01\x03\x0b\x06\x03\x01\x05\x01\x00\xea\tO%\x03\x0b\x0f\x0b\t\t\t\t\x13\x0f\x13\x11!\x11#K/A)Sci3\x13%)9\x1f\x15\x11\x17\x15\x11\x19)\x0f\x0b\x11builtin\x00vhlo\x00module\x00broadcast_in_dim_v1\x00constant_v1\x00func_v1\x00return_v1\x00compare_v1\x00call_v1\x00select_v1\x00custom_call_v1\x00jax.uses_shape_polymorphism\x00mhlo.num_partitions\x00mhlo.num_replicas\x00jit_func\x00jit(func)/jit(main)/pjit\x00third_party/py/jax/tests/export_back_compat_test.py\x00jit(func)/jit(main)/jit(_where)/broadcast_in_dim\x00jit(func)/jit(main)/jit(_where)/select_n\x00mhlo.backend_config\x00jit(func)/jit(main)/tridiagonal\x00jit(func)/jit(main)/eq\x00jit(func)/jit(main)/broadcast_in_dim\x00mhlo.layout_mode\x00default\x00jax.result_info\x00private\x00_where_1\x00_where\x00_where_0\x00[0]\x00[1]\x00[2]\x00[3]\x00main\x00public\x00uplo\x00\x00lapack_ssytrd_ffi\x00\x08y#\x057\x01\x0bOY[mo\x03y\x11\x81\x83\x85O\x87\x89\x8b\x8f\x03;\x031\x05=?\x03A\x03C\x03Q\x03S\x03K\x0bEqGQI\x03w\x0bEsGSI\x03U\x0bEuGKI',
    xla_call_module_version=9,
    nr_devices=1,
)  # End paste


# Pasted from the test output (see export_back_compat_test_util.py module docstring)
data_2024_09_04["f64"] = dict(
    testdata_version=1,
    platform='cpu',
    custom_call_targets=['lapack_dsytrd_ffi'],
    serialized_date=datetime.date(2024, 9, 4),
    inputs=(),
    expected_outputs=(array([[[ 0.8251247184208595 , -2.6963562039892532 ,
          0.8082445002373937 , -1.551980329390836  ],
        [-2.629505060186711  ,  4.427374205796291  ,
         -2.2111093161901074 ,  7.552489598405787  ],
        [ 0.2269453213819231 ,  0.3650586474106988 ,
         -3.5933639667756205 ,  4.828829679372501  ],
        [-0.6415372293575187 , -0.2519326897319508 ,
         -1.7607827845801751 , -3.381311711243865  ]],

       [[-4.000421911405985  ,  3.6303350337601055 ,
          2.8066821235532355 ,  1.099224389184342  ],
        [-4.141622408467332  , -5.276404169116551  ,
         -0.8496056221591237 , -2.275319346221659  ],
        [ 0.5828958067901202 ,  0.9351254869793256 ,
          2.7765603683442177 , -4.339686212557215  ],
        [-0.6391146585297987 ,  0.3129920702652711 ,
         -0.25441692469349864, -1.4155240723557498 ]]]), array([[ 0.8251247184208595,  4.427374205796291 , -3.5933639667756205,
        -3.381311711243865 ],
       [-4.000421911405985 , -5.276404169116551 ,  2.7765603683442177,
        -1.4155240723557498]]), array([[-2.629505060186711  ,  0.3650586474106988 , -1.7607827845801751 ],
       [-4.141622408467332  ,  0.9351254869793256 , -0.25441692469349864]]), array([[1.3669846724688552, 1.8806358893589366, 0.                ],
       [1.1440109149169537, 1.8215532880266878, 0.                ]])),
    mlir_module_text=r"""
#loc1 = loc("third_party/py/jax/tests/export_back_compat_test.py":695:13)
#loc5 = loc("jit(func)/jit(main)/pjit"(#loc1))
module @jit_func attributes {jax.uses_shape_polymorphism = false, mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} {
  func.func public @main() -> (tensor<2x4x4xf64> {jax.result_info = "[0]", mhlo.layout_mode = "default"}, tensor<2x4xf64> {jax.result_info = "[1]", mhlo.layout_mode = "default"}, tensor<2x3xf64> {jax.result_info = "[2]", mhlo.layout_mode = "default"}, tensor<2x3xf64> {jax.result_info = "[3]", mhlo.layout_mode = "default"}) {
    %cst = stablehlo.constant dense<[[[0.82512471842085955, -2.6963562039892532, 0.80824450023739369, -1.5519803293908361], [0.96498805326781766, -4.1313349231964409, -2.2111093161901074, 7.5524895984057867], [0.81575339483804743, 1.0647235400727899, -1.0064296232364345, 4.8288296793725012], [-2.3060011529502993, -2.9182106402942192, -1.7781896154088577, 2.5904630742096817]], [[-4.0004219114059847, 3.6303350337601055, 2.8066821235532355, 1.0992243891843421], [0.59643883228393779, -1.5243235004961249, -0.84960562215912372, -2.275319346221659], [2.7617960295487092, -0.57538970930521982, 0.12559406141906576, -4.3396862125572149], [-3.0281643919760217, 0.38177997229319849, 3.860398204232184, -2.5166384340510231]]]> : tensor<2x4x4xf64> loc(#loc)
    %0:5 = stablehlo.custom_call @lapack_dsytrd_ffi(%cst) {mhlo.backend_config = {uplo = 76 : ui8}, operand_layouts = [dense<[1, 2, 0]> : tensor<3xindex>], output_operand_aliases = [#stablehlo.output_operand_alias<output_tuple_indices = [0], operand_index = 0, operand_tuple_indices = []>], result_layouts = [dense<[1, 2, 0]> : tensor<3xindex>, dense<[1, 0]> : tensor<2xindex>, dense<[1, 0]> : tensor<2xindex>, dense<[1, 0]> : tensor<2xindex>, dense<0> : tensor<1xindex>]} : (tensor<2x4x4xf64>) -> (tensor<2x4x4xf64>, tensor<2x4xf64>, tensor<2x3xf64>, tensor<2x3xf64>, tensor<2xi32>) loc(#loc2)
    %c = stablehlo.constant dense<0> : tensor<i32> loc(#loc)
    %1 = stablehlo.broadcast_in_dim %c, dims = [] : (tensor<i32>) -> tensor<2xi32> loc(#loc3)
    %2 = stablehlo.compare  EQ, %0#4, %1,  SIGNED : (tensor<2xi32>, tensor<2xi32>) -> tensor<2xi1> loc(#loc3)
    %3 = stablehlo.broadcast_in_dim %2, dims = [0] : (tensor<2xi1>) -> tensor<2x1x1xi1> loc(#loc4)
    %cst_0 = stablehlo.constant dense<0x7FF8000000000000> : tensor<f64> loc(#loc)
    %4 = call @_where(%3, %0#0, %cst_0) : (tensor<2x1x1xi1>, tensor<2x4x4xf64>, tensor<f64>) -> tensor<2x4x4xf64> loc(#loc5)
    %c_1 = stablehlo.constant dense<0> : tensor<i32> loc(#loc)
    %5 = stablehlo.broadcast_in_dim %c_1, dims = [] : (tensor<i32>) -> tensor<2xi32> loc(#loc3)
    %6 = stablehlo.compare  EQ, %0#4, %5,  SIGNED : (tensor<2xi32>, tensor<2xi32>) -> tensor<2xi1> loc(#loc3)
    %7 = stablehlo.broadcast_in_dim %6, dims = [0] : (tensor<2xi1>) -> tensor<2x1xi1> loc(#loc4)
    %cst_2 = stablehlo.constant dense<0x7FF8000000000000> : tensor<f64> loc(#loc)
    %8 = call @_where_0(%7, %0#1, %cst_2) : (tensor<2x1xi1>, tensor<2x4xf64>, tensor<f64>) -> tensor<2x4xf64> loc(#loc5)
    %c_3 = stablehlo.constant dense<0> : tensor<i32> loc(#loc)
    %9 = stablehlo.broadcast_in_dim %c_3, dims = [] : (tensor<i32>) -> tensor<2xi32> loc(#loc3)
    %10 = stablehlo.compare  EQ, %0#4, %9,  SIGNED : (tensor<2xi32>, tensor<2xi32>) -> tensor<2xi1> loc(#loc3)
    %11 = stablehlo.broadcast_in_dim %10, dims = [0] : (tensor<2xi1>) -> tensor<2x1xi1> loc(#loc4)
    %cst_4 = stablehlo.constant dense<0x7FF8000000000000> : tensor<f64> loc(#loc)
    %12 = call @_where_1(%11, %0#2, %cst_4) : (tensor<2x1xi1>, tensor<2x3xf64>, tensor<f64>) -> tensor<2x3xf64> loc(#loc5)
    %c_5 = stablehlo.constant dense<0> : tensor<i32> loc(#loc)
    %13 = stablehlo.broadcast_in_dim %c_5, dims = [] : (tensor<i32>) -> tensor<2xi32> loc(#loc3)
    %14 = stablehlo.compare  EQ, %0#4, %13,  SIGNED : (tensor<2xi32>, tensor<2xi32>) -> tensor<2xi1> loc(#loc3)
    %15 = stablehlo.broadcast_in_dim %14, dims = [0] : (tensor<2xi1>) -> tensor<2x1xi1> loc(#loc4)
    %cst_6 = stablehlo.constant dense<0x7FF8000000000000> : tensor<f64> loc(#loc)
    %16 = call @_where_1(%15, %0#3, %cst_6) : (tensor<2x1xi1>, tensor<2x3xf64>, tensor<f64>) -> tensor<2x3xf64> loc(#loc5)
    return %4, %8, %12, %16 : tensor<2x4x4xf64>, tensor<2x4xf64>, tensor<2x3xf64>, tensor<2x3xf64> loc(#loc)
  } loc(#loc)
  func.func private @_where(%arg0: tensor<2x1x1xi1> {mhlo.layout_mode = "default"} loc("jit(func)/jit(main)/pjit"(#loc1)), %arg1: tensor<2x4x4xf64> {mhlo.layout_mode = "default"} loc("jit(func)/jit(main)/pjit"(#loc1)), %arg2: tensor<f64> {mhlo.layout_mode = "default"} loc("jit(func)/jit(main)/pjit"(#loc1))) -> (tensor<2x4x4xf64> {mhlo.layout_mode = "default"}) {
    %0 = stablehlo.broadcast_in_dim %arg0, dims = [0, 1, 2] : (tensor<2x1x1xi1>) -> tensor<2x4x4xi1> loc(#loc6)
    %1 = stablehlo.broadcast_in_dim %arg2, dims = [] : (tensor<f64>) -> tensor<2x4x4xf64> loc(#loc6)
    %2 = stablehlo.select %0, %arg1, %1 : tensor<2x4x4xi1>, tensor<2x4x4xf64> loc(#loc7)
    return %2 : tensor<2x4x4xf64> loc(#loc5)
  } loc(#loc5)
  func.func private @_where_0(%arg0: tensor<2x1xi1> {mhlo.layout_mode = "default"} loc("jit(func)/jit(main)/pjit"(#loc1)), %arg1: tensor<2x4xf64> {mhlo.layout_mode = "default"} loc("jit(func)/jit(main)/pjit"(#loc1)), %arg2: tensor<f64> {mhlo.layout_mode = "default"} loc("jit(func)/jit(main)/pjit"(#loc1))) -> (tensor<2x4xf64> {mhlo.layout_mode = "default"}) {
    %0 = stablehlo.broadcast_in_dim %arg0, dims = [0, 1] : (tensor<2x1xi1>) -> tensor<2x4xi1> loc(#loc6)
    %1 = stablehlo.broadcast_in_dim %arg2, dims = [] : (tensor<f64>) -> tensor<2x4xf64> loc(#loc6)
    %2 = stablehlo.select %0, %arg1, %1 : tensor<2x4xi1>, tensor<2x4xf64> loc(#loc7)
    return %2 : tensor<2x4xf64> loc(#loc5)
  } loc(#loc5)
  func.func private @_where_1(%arg0: tensor<2x1xi1> {mhlo.layout_mode = "default"} loc("jit(func)/jit(main)/pjit"(#loc1)), %arg1: tensor<2x3xf64> {mhlo.layout_mode = "default"} loc("jit(func)/jit(main)/pjit"(#loc1)), %arg2: tensor<f64> {mhlo.layout_mode = "default"} loc("jit(func)/jit(main)/pjit"(#loc1))) -> (tensor<2x3xf64> {mhlo.layout_mode = "default"}) {
    %0 = stablehlo.broadcast_in_dim %arg0, dims = [0, 1] : (tensor<2x1xi1>) -> tensor<2x3xi1> loc(#loc6)
    %1 = stablehlo.broadcast_in_dim %arg2, dims = [] : (tensor<f64>) -> tensor<2x3xf64> loc(#loc6)
    %2 = stablehlo.select %0, %arg1, %1 : tensor<2x3xi1>, tensor<2x3xf64> loc(#loc7)
    return %2 : tensor<2x3xf64> loc(#loc5)
  } loc(#loc5)
} loc(#loc)
#loc = loc(unknown)
#loc2 = loc("jit(func)/jit(main)/tridiagonal"(#loc1))
#loc3 = loc("jit(func)/jit(main)/eq"(#loc1))
#loc4 = loc("jit(func)/jit(main)/broadcast_in_dim"(#loc1))
#loc6 = loc("jit(func)/jit(main)/jit(_where)/broadcast_in_dim"(#loc1))
#loc7 = loc("jit(func)/jit(main)/jit(_where)/select_n"(#loc1))
""",
    mlir_module_serialized=b'ML\xefR\rStableHLO_v1.3.0\x00\x01#\x05\x01\x05\x13\x01\x03\x0b\x03\x11\x0f\x13\x17\x1b\x1f#\'+\x03\xe7\x93?\x011\x0f\x07\x0f\x17\x0f\x0f\x0f\x0f#\x0b\x0f\x0b\x0b\x0b\x0b\x0b\x0b\x0b\x13\x0b\x0f\x0b\x0b\x0b\x03c\x0f\x0b\x0b\x0b\x13\x1f\x0b\x0b//\x17\x0f\x0b\x0bO\x0b\x0b\x0bOo\x0b\x1b\x1b\x0b\x1b\x0b\x1b\x0b\x1b\x0b\x0b\x0b\x0b\x0b\x0bo&\x08\x13\x0b\x0f\x0b\x0b\x0b\x0b\x0f\x0f\x17\x1f/\x01\x05\x0b\x0f\x03;\x17\x0f\x1b\x17\x17\x07\x13\x0f\x07\x07\x13\x1b\x07\x07\x1f\x1f\x1f\x1f\x17\x13\x13\x17\x1b\x13\x07\x13\x13\x13\x13\x02\xfa\n\x1d\x1d\x07\x1f\x1d-\x07\x17\x1f\xde\n\x1b\x1d!\x07\x1d/\x07\x1d#\x07\x11\x03\x05\x03\x07\x13\x15\x17\x0f\x19\x0f\x05\x17\x11\x01\x00\x05\x19\x05\x1b\x05\x1d\x05\x1f\x05!\x05#\x05%\x03\x03\'{\x05\'\x1d+\x07\x05)\x05+\x05-\x1f-\x01\x1d/\x1d1\x1d3\r\x0335\x1f\x13\t\x00\x00\x00\x00\t\x07\x07\x01\x1f=\x11\x00\x00\x00\x00\x00\x00\x00\x00\x1f\x07\x11\x00\x00\x00\x00\x00\x00\xf8\x7f\x03\x07999\x03\x039\x1d5\x1d7\x1f9!\x01\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x03\x01\x1d9\x1d;\x1f+!\x00\x00\x00\x00\x00\x00\x00\x00\x01\x00\x00\x00\x00\x00\x00\x00\x1f71\x01\x00\x00\x00\x00\x00\x00\x00\x02\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00#!\x03\t]aei\r\x057_35\x1d=\r\x057c35\x1d?\r\x057g35\x1dA\r\x057k35\x1dC\x1dE\x1dG###%#\'\x1f31\x00\x00\x00\x00\x00\x00\x00\x00\x01\x00\x00\x00\x00\x00\x00\x00\x02\x00\x00\x00\x00\x00\x00\x00\x1f\t\x02\x04A\xa4\x17\xf4kg\xea?\x1f\x01\x943#\x92\x05\xc0\x86 \xf6\x91#\xdd\xe9?\x9dMlS\xe9\xd4\xf8\xbf\x88\x1c:\xa0.\xe1\xee?8\xce\x7f\xa9|\x86\x10\xc0\xe8V\xc7\x14Z\xb0\x01\xc0\xd2!R\xd5\xbf5\x1e@\xbf\xc5\r\xdd\xa6\x1a\xea?\xbcM\xfe\x8c\x1b\t\xf1?\xdbj\xd8\xf2U\x1a\xf0\xbf\xado;\xba\xb8P\x13@\xbb\xad\x83\xbb\xb0r\x02\xc0\x1f9\xf7\xd1~X\x07\xc0)ID\xf4vs\xfc\xbfD\xcfI\xb4D\xb9\x04@\x16\xc3\xfe\x99n\x00\x10\xc0\x82.\x1c\x18\xed\n\r@\x8cn\xd7\xc1\x15t\x06@|2(Pl\x96\xf1?\x88*\xd7\xe3\x06\x16\xe3?F{\xf2\t\xa1c\xf8\xbf8z5!\xf8/\xeb\xbf4\xd3\x1f\xa1\xda3\x02\xc0)\x13I\x84(\x18\x06@\xbcw\xfd\xad\x97i\xe2\xbf\x1e\xf0.Yw\x13\xc0?dW\xd7\xb3\xd6[\x11\xc0\x04\x97\xb3@\xae9\x08\xc0\xbc\x17\xd1C\x15o\xd8?\x02\xb7%t\x18\xe2\x0e@\xac\xd8\xd0T\x13"\x04\xc0\r\x03}\x7f\x1dI\x135L\x0b\x03\x1dK\x1dM\x05\x01\x03\x03W\x03\x03\x8d\x15\x03\x01\x01\x01\x03\x0bWMMM\x91\x1f;\x11\x00\x00\x00\x00\x00\x00\x00\x00\x01\t\x01\x02\x02)\x05\t\r\x15)\x01\x15)\x07\t\x11\x11\x15)\x05\t\x11\x15)\x05\t\x05\x0f\x01)\x03\t\x1f)\x01\x1f\x0b\x1d)\x03\t\x0f)\x07\t\x05\x05\x0f\x13\x1b\x11\x01\t\t\x0b\x05\x05\x11\x07\x1b\t\x07\x03\t\x11\x07\r\x0b\x07\x03\x0b\x11\x07\r\x05\x07\x03\x05)\x05\t\r\x0f)\x03\t\x17)\x03\x01\x17)\x05\t\x11\x0f)\x07\t\x11\x11\x0f)\x03\r\x17!)\x03\r\x1d)\x03\t\x1d)\x03\x05\x1d)\x03\x05\x17\x04\xfe\x05\x05\x01Q\x03\x11\x01\x07\x04\xd6\x05\x03\x01\x11\x07P\x03\x03\x07\x04j\x03\x03=m\x05B\x03\x05\x03\t\x11G)%\x07\x0b\t\x0b\x05\x05\x11\x03\x01\x05B\x03\t\x03\x13\x03F\x05\x0b\x03\x11\x03\r\x0bF\x05\r\x03\x19\x05\x0b\x0f\x03F\x0b\x0f\x03\x1b\x03\x11\x05B\x03\x11\x03\x07\rF\x01\x13\x03\t\x07\x13\x03\x15\x05B\x03\t\x03\x13\x03F\x05\x0b\x03\x11\x03\x19\x0bF\x05\r\x03\x19\x05\x0b\x1b\x03F\x0b\x0f\x03\r\x03\x1d\x05B\x03\x11\x03\x07\rF\x01\x15\x03\x0b\x07\x1f\x05!\x05B\x03\t\x03\x13\x03F\x05\x0b\x03\x11\x03%\x0bF\x05\r\x03\x19\x05\x0b\'\x03F\x0b\x0f\x03\r\x03)\x05B\x03\x11\x03\x07\rF\x01\x17\x03\x05\x07+\x07-\x05B\x03\t\x03\x13\x03F\x05\x0b\x03\x11\x031\x0bF\x05\r\x03\x19\x05\x0b3\x03F\x0b\x0f\x03\r\x035\x05B\x03\x11\x03\x07\rF\x01\x17\x03\x05\x077\t9\t\x04\x03\t\x17#/;\x07P\x01\x19\x07\x04S\x03\r\x13\x077\x01\x13\x01\x0f\x01\x00\x03F\t\x1b\x031\x03\x01\x03F\t\x0b\x03\t\x03\x05\x0f\x06\r\x03\t\x07\x07\x03\t\t\x04\x01\x03\x0b\x07P\x01\x1d\x07\x04S\x03\r\x13\x07\x1b\x01\x17\x01\x0f\x01\x00\x03F\t\x1f\x03/\x03\x01\x03F\t\x0b\x03\x0b\x03\x05\x0f\x06\r\x03\x0b\x07\x07\x03\t\t\x04\x01\x03\x0b\x07P\x01!\x07\x04S\x03\r\x13\x07\x1b\x01\x0b\x01\x0f\x01\x00\x03F\t\x1f\x03)\x03\x01\x03F\t\x0b\x03\x05\x03\x05\x0f\x06\r\x03\x05\x07\x07\x03\t\t\x04\x01\x03\x0b\x06\x03\x01\x05\x01\x00\xea\tO%\x03\x0b\x0f\x0b\t\t\t\t\x13\x0f\x13\x11!\x11#K/A)Sci3\x13%)9\x1f\x15\x11\x17\x15\x11\x19)\x0f\x0b\x11builtin\x00vhlo\x00module\x00broadcast_in_dim_v1\x00constant_v1\x00func_v1\x00return_v1\x00compare_v1\x00call_v1\x00select_v1\x00custom_call_v1\x00jax.uses_shape_polymorphism\x00mhlo.num_partitions\x00mhlo.num_replicas\x00jit_func\x00jit(func)/jit(main)/pjit\x00third_party/py/jax/tests/export_back_compat_test.py\x00jit(func)/jit(main)/jit(_where)/broadcast_in_dim\x00jit(func)/jit(main)/jit(_where)/select_n\x00mhlo.backend_config\x00jit(func)/jit(main)/tridiagonal\x00jit(func)/jit(main)/eq\x00jit(func)/jit(main)/broadcast_in_dim\x00mhlo.layout_mode\x00default\x00jax.result_info\x00private\x00_where_1\x00_where\x00_where_0\x00[0]\x00[1]\x00[2]\x00[3]\x00main\x00public\x00uplo\x00\x00lapack_dsytrd_ffi\x00\x08y#\x057\x01\x0bOY[mo\x03y\x11\x81\x83\x85O\x87\x89\x8b\x8f\x03;\x031\x05=?\x03A\x03C\x03Q\x03S\x03K\x0bEqGQI\x03w\x0bEsGSI\x03U\x0bEuGKI',
    xla_call_module_version=9,
    nr_devices=1,
)  # End paste

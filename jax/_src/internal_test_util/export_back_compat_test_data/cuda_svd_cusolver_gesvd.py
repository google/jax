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

# type: ignore
# ruff: noqa

import datetime
from numpy import array, float32, complex64

data_2024_10_08 = {"jacobi": {}, "qr": {}}

data_2024_10_08["jacobi"]["f32"] = dict(
    testdata_version=1,
    platform='cuda',
    custom_call_targets=['cusolver_gesvdj_ffi'],
    serialized_date=datetime.date(2024, 10, 8),
    inputs=(array([[[ 4.0477114 , -1.6619755 ,  3.410165  , -0.3096872 ],
        [ 1.7056831 ,  0.17458144,  0.12622283,  1.8056804 ],
        [-0.21965337,  1.7691472 , -0.8960539 ,  1.1495945 ],
        [ 1.4272052 ,  3.2477028 , -1.5676605 , -1.7347798 ]],

       [[-3.914366  , -2.8975718 ,  1.5051024 ,  1.7379241 ],
        [-5.045383  ,  1.4189544 , -0.61290324,  1.7456682 ],
        [-4.823716  ,  0.32512662, -1.9951408 ,  3.632175  ],
        [ 0.8716193 , -0.24001008,  1.5933404 ,  1.0177776 ]]],
      dtype=float32),),
    expected_outputs=(array([[[ 0.9018128  ,  0.31748173 ,  0.090800084, -0.27873662 ],
        [ 0.1856214  ,  0.18972664 , -0.78390217 ,  0.56128997 ],
        [-0.25094122 ,  0.20106053 , -0.55834264 , -0.7647598  ],
        [-0.29884213 ,  0.90707415 ,  0.25594372 ,  0.1496738  ]],

       [[-0.45493636 , -0.8541334  , -0.20037504 , -0.15276858 ],
        [-0.57877773 ,  0.30431184 , -0.4479597  ,  0.6097069  ],
        [-0.6747176  ,  0.29090276 ,  0.57102525 , -0.3661442  ],
        [ 0.052960183, -0.30532822 ,  0.65811265 ,  0.68619025 ]]],
      dtype=float32), array([[5.974016 , 4.183989 , 2.6312675, 0.5687128],
       [9.106636 , 3.995191 , 1.8342099, 1.6058134]], dtype=float32), array([[[ 0.60185665 , -0.48223644 ,  0.6347658  ,  0.047846846],
        [ 0.6833445  ,  0.6709123  , -0.1184354  , -0.26246953 ],
        [-0.18304126 , -0.16886286 ,  0.11772618 , -0.96131265 ],
        [ 0.37054697 , -0.53741086 , -0.75444424 , -0.0685464  ]],

       [[ 0.8786726  ,  0.0290856  ,  0.12085139 , -0.46095958 ],
        [ 0.034706447,  0.76957005 , -0.635503   , -0.051896986],
        [ 0.4708456  , -0.014900906, -0.06417345 ,  0.8797526  ],
        [-0.07095488 ,  0.6377255  ,  0.75987667 ,  0.10420583 ]]],
      dtype=float32)),
    mlir_module_text=r"""
#loc1 = loc("operand")
module @jit_func attributes {jax.uses_shape_polymorphism = false, mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} {
  func.func public @main(%arg0: tensor<2x4x4xf32> {mhlo.layout_mode = "default"} loc("operand")) -> (tensor<2x4x4xf32> {jax.result_info = "[0]", mhlo.layout_mode = "default"}, tensor<2x4xf32> {jax.result_info = "[1]", mhlo.layout_mode = "default"}, tensor<2x4x4xf32> {jax.result_info = "[2]", mhlo.layout_mode = "default"}) {
    %cst = stablehlo.constant dense<0x7FC00000> : tensor<f32> loc(#loc)
    %c = stablehlo.constant dense<0> : tensor<i32> loc(#loc)
    %0:5 = stablehlo.custom_call @cusolver_gesvdj_ffi(%arg0) {mhlo.backend_config = {compute_uv = true, full_matrices = true}, operand_layouts = [dense<[1, 2, 0]> : tensor<3xindex>], output_operand_aliases = [#stablehlo.output_operand_alias<output_tuple_indices = [0], operand_index = 0, operand_tuple_indices = []>], result_layouts = [dense<[1, 2, 0]> : tensor<3xindex>, dense<[1, 0]> : tensor<2xindex>, dense<[1, 2, 0]> : tensor<3xindex>, dense<[1, 2, 0]> : tensor<3xindex>, dense<0> : tensor<1xindex>]} : (tensor<2x4x4xf32>) -> (tensor<2x4x4xf32>, tensor<2x4xf32>, tensor<2x4x4xf32>, tensor<2x4x4xf32>, tensor<2xi32>) loc(#loc3)
    %1 = stablehlo.transpose %0#3, dims = [0, 2, 1] : (tensor<2x4x4xf32>) -> tensor<2x4x4xf32> loc(#loc3)
    %2 = stablehlo.broadcast_in_dim %c, dims = [] : (tensor<i32>) -> tensor<2xi32> loc(#loc3)
    %3 = stablehlo.compare  EQ, %0#4, %2,  SIGNED : (tensor<2xi32>, tensor<2xi32>) -> tensor<2xi1> loc(#loc3)
    %4 = stablehlo.broadcast_in_dim %3, dims = [0] : (tensor<2xi1>) -> tensor<2x1xi1> loc(#loc3)
    %5 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<2x4xf32> loc(#loc3)
    %6 = stablehlo.broadcast_in_dim %4, dims = [0, 1] : (tensor<2x1xi1>) -> tensor<2x4xi1> loc(#loc3)
    %7 = stablehlo.select %6, %0#1, %5 : tensor<2x4xi1>, tensor<2x4xf32> loc(#loc3)
    %8 = stablehlo.broadcast_in_dim %3, dims = [0] : (tensor<2xi1>) -> tensor<2x1x1xi1> loc(#loc3)
    %9 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<2x4x4xf32> loc(#loc3)
    %10 = stablehlo.broadcast_in_dim %8, dims = [0, 1, 2] : (tensor<2x1x1xi1>) -> tensor<2x4x4xi1> loc(#loc3)
    %11 = stablehlo.select %10, %0#2, %9 : tensor<2x4x4xi1>, tensor<2x4x4xf32> loc(#loc3)
    %12 = stablehlo.broadcast_in_dim %3, dims = [0] : (tensor<2xi1>) -> tensor<2x1x1xi1> loc(#loc3)
    %13 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<2x4x4xf32> loc(#loc3)
    %14 = stablehlo.broadcast_in_dim %12, dims = [0, 1, 2] : (tensor<2x1x1xi1>) -> tensor<2x4x4xi1> loc(#loc3)
    %15 = stablehlo.select %14, %1, %13 : tensor<2x4x4xi1>, tensor<2x4x4xf32> loc(#loc3)
    return %11, %7, %15 : tensor<2x4x4xf32>, tensor<2x4xf32>, tensor<2x4x4xf32> loc(#loc)
  } loc(#loc)
} loc(#loc)
#loc = loc(unknown)
#loc2 = loc("third_party/py/jax/tests/export_back_compat_test.py":686:13)
#loc3 = loc("jit(func)/jit(main)/svd"(#loc2))
""",
    mlir_module_serialized=b"ML\xefR\rStableHLO_v1.7.0\x00\x01#\x05\x01\x05\x13\x01\x03\x0b\x03\x11\x0f\x13\x17\x1b\x1f#'+\x03\xb7q3\x01!\x0f\x07\x0f#\x0b\x0f\x0b\x0b\x0b\x0f\x0b\x13\x0b\x0b\x17\x0b\x03Q\x0b\x0bo\x0f\x0b/\x0bo\x0f\x13\x0b\x17\x1b\x0b\x1b\x0b\x1b\x0b\x0b\x0b\x1f\x1f\x1b\x0b\x0b\x0b\x0b\x0b\x0b\x0b\x0f\x0f\x17\x1fO/o\x0b\x0bO\x01\x05\x0b\x0f\x03/\x1b\x07\x17\x07\x07\x07\x0f\x0f\x07\x13\x13\x1b\x1b\x1f\x13\x13\x13\x13\x13\x17\x13\x17\x13\x02\x12\x06\x1d\x1b\x1d\x1f\x11\x03\x05\x03\x07\t\x0b\r\x05\x0f\x05\x05\x17\x11\x01\x00\x05\x19\x05\x1b\x05\x1d\x1d\x15\x03\x05\x1f\x03\x03\x19M\x05!\x05#\x17\x1f\xba\n\x1b\x05%\x1d'\x1d)\x1f!1\x01\x00\x00\x00\x00\x00\x00\x00\x02\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x1f'\x01\x1d+\x1f-\x11\x00\x00\x00\x00\x00\x00\x00\x00\x05\x03\x1f\x191\x00\x00\x00\x00\x00\x00\x00\x00\x01\x00\x00\x00\x00\x00\x00\x00\x02\x00\x00\x00\x00\x00\x00\x00\x03\x033\r\x03!##\x1f\x03\x079=A\r\x05);!#\x1d-\r\x05)?!#\x1d/\r\x05)C!#\x1d1\x1d3\x1d5\x1f\x11\t\x00\x00\xc0\x7f\x1f\x13\t\x00\x00\x00\x00\r\x05O-Q-\x1d7\x1d9\x0b\x03\x1d;\x1d=\x03\x01\x05\x01\x03\x03%\x03\x03a\x15\x03\x01\x01\x01\x03\x0b%e%%g\x1f#!\x01\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x1f%\x11\x00\x00\x00\x00\x00\x00\x00\x00\x1f\x191\x00\x00\x00\x00\x00\x00\x00\x00\x02\x00\x00\x00\x00\x00\x00\x00\x01\x00\x00\x00\x00\x00\x00\x00\t\x07\x07\x01\x1f1!\x00\x00\x00\x00\x00\x00\x00\x00\x01\x00\x00\x00\x00\x00\x00\x00\x01\t\x01\x02\x02)\x07\t\x11\x11\r\x01)\x05\t\x11\r\x1d\t\x13)\x01\r)\x01\x15\x1b)\x03\t\x15)\x03\r\x0b)\x07\t\x05\x05\x07)\x07\t\x11\x11\x07\x11\x03\x05\x07\x05\t\x05)\x03\r\x0f)\x03\t\x0f)\x03\x05\x0f)\x03\x01\x0b)\x03\t\x07)\x05\t\x05\x07)\x03\x05\x0b)\x05\t\x11\x07)\x03\t\x0b\x04\xe2\x02\x05\x01Q\x03\x07\x01\x07\x04\xba\x02\x03\x01\x05\tP\x03\x03\x07\x04\x8e\x02\x03/O\x03\x0b\x13\x00\x07B\x03\x05\x03\x11\x07B\x03\x07\x03\x13\x0bG\x01\x17\t\x0b\x05\t\x05\x05\x17\x03\x01\rF\x01\x0b\x03\x05\x03\r\x03F\x01\r\x03\x17\x03\x05\x0fF\x01\x0f\x03)\x05\x0f\x13\x03F\x01\x11\x03+\x03\x15\x03F\x01\r\x03\t\x03\x03\x03F\x01\x13\x03/\x03\x17\x05\x06\x01\x03\t\x07\x1b\t\x19\x03F\x01\x11\x03\x1b\x03\x15\x03F\x01\r\x03\x05\x03\x03\x03F\x01\x15\x03\x1d\x03\x1f\x05\x06\x01\x03\x05\x07#\x0b!\x03F\x01\x11\x03\x1b\x03\x15\x03F\x01\r\x03\x05\x03\x03\x03F\x01\x15\x03\x1d\x03'\x05\x06\x01\x03\x05\x07+\x11)\x11\x04\x03\x07%\x1d-\x06\x03\x01\x05\x01\x00\xe6\x06?)\x03\x1d\x17\x0f\x0b\t\t\t!\x11#i1)\x11\x13%)9\x15\x17\x1b\x1f\x11\x19\x15)\x0f\x0b\x11builtin\x00vhlo\x00module\x00broadcast_in_dim_v1\x00select_v1\x00constant_v1\x00func_v1\x00custom_call_v1\x00transpose_v1\x00compare_v1\x00return_v1\x00jax.uses_shape_polymorphism\x00mhlo.num_partitions\x00mhlo.num_replicas\x00jit_func\x00operand\x00mhlo.backend_config\x00jit(func)/jit(main)/svd\x00third_party/py/jax/tests/export_back_compat_test.py\x00mhlo.layout_mode\x00default\x00jax.result_info\x00[0]\x00[1]\x00[2]\x00main\x00public\x00compute_uv\x00full_matrices\x00\x00cusolver_gesvdj_ffi\x00\x08I\x17\x05#\x01\x0b157EG\x03I\x03K\x11SUWY[]_c\x03i\x03'\x05km\x03+\x03o\x03/",
    xla_call_module_version=9,
    nr_devices=1,
)  # End paste

data_2024_10_08["jacobi"]["f64"] = dict(
    testdata_version=1,
    platform='cuda',
    custom_call_targets=['cusolver_gesvdj_ffi'],
    serialized_date=datetime.date(2024, 10, 8),
    inputs=(array([[[-0.23389781892940775, -0.20108449262485306,
         -0.5666115573270456 , -2.4789757754536694 ],
        [-1.8555613779538866 ,  2.9994543506103533 ,
          1.087201973266891  , -1.012848084871914  ],
        [-3.195786395215201  , -2.483536010628656  ,
         -0.9470206294018368 , -0.4080455549606986 ],
        [ 0.41241574420074084, -8.059831117309347  ,
         -0.7882929058035465 ,  2.8856408696497664 ]],

       [[ 1.7959513084827519 , -6.006170699401665  ,
          0.9365840264144545 , -2.8339481807478486 ],
        [ 3.37344261819673   , -2.809695890050033  ,
          0.9096330403907014 , -0.22091594063236752],
        [-1.8994569852606458 ,  0.8593563734072376 ,
          3.3970755287480623 ,  1.162324876281341  ],
        [ 1.2730096750276905 ,  3.1397664846677484 ,
         -4.625276688205361  , -3.0618323131122303 ]]]),),
    expected_outputs=(array([[[ 0.05538281351237727 , -0.34543122319556363 ,
         -0.875138554715149   ,  0.33427911101376073 ],
        [ 0.35132368873545367 , -0.3583481459523727  ,
          0.4466199100964102  ,  0.7407353967047108  ],
        [-0.2350018651348275  , -0.8649389618871873  ,
          0.17010144637596905 , -0.40953658387679276 ],
        [-0.904587493327162   ,  0.06437754689044517 ,
          0.07568793759482184 ,  0.41454593771389403 ]],

       [[-0.7655776828405998  , -0.3259951618597633  ,
         -0.5487654998241592  ,  0.08046360781859588 ],
        [-0.4502679475178967  , -0.18435823314270097 ,
          0.7918156001863894  ,  0.3691867719894481  ],
        [ 0.011780696812478626,  0.5608039473610089  ,
         -0.23310619898043075 ,  0.7943687102371408  ],
        [ 0.4593591211209926  , -0.7384024166677944  ,
         -0.13246124527990302 ,  0.47561022634191513 ]]]), array([[9.502458469794536 , 4.039322683970868 , 2.3634256270400944,
        0.7309141361765127],
       [8.368723268857098 , 7.018568450518651 , 2.518568829019885 ,
        1.4845675373399827]]), array([[[-0.030192917862639196,  0.9383993407299717  ,
          0.13535558060292174 , -0.31650265690534246 ],
        [ 0.8755039760059299  ,  0.15444343242930395 ,
          0.14222562381109818 ,  0.4352147586063772  ],
        [-0.4808404523560293  ,  0.20440997041138523 ,
          0.32185308272374646 ,  0.7895692601131882  ],
        [ 0.03706258337996993 , -0.23188459951757628 ,
          0.9262080391966009  , -0.2949484116712019  ]],

       [[-0.2785970537508859  ,  0.8741728222022489  ,
         -0.3837203651304145  ,  0.10471026668126972 ],
        [-0.45773005941421924 ,  0.09111437041907021 ,
          0.690652044371576   ,  0.5524320028900885  ],
        [ 0.7781161676166882  ,  0.18065797801542668 ,
          0.010754987459331431,  0.6014833787542634  ],
        [ 0.32772260227744276 ,  0.4414552564025818  ,
          0.612897026615641   , -0.5675142177221092  ]]])),
    mlir_module_text=r"""
#loc1 = loc("operand")
module @jit_func attributes {jax.uses_shape_polymorphism = false, mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} {
  func.func public @main(%arg0: tensor<2x4x4xf64> {mhlo.layout_mode = "default"} loc("operand")) -> (tensor<2x4x4xf64> {jax.result_info = "[0]", mhlo.layout_mode = "default"}, tensor<2x4xf64> {jax.result_info = "[1]", mhlo.layout_mode = "default"}, tensor<2x4x4xf64> {jax.result_info = "[2]", mhlo.layout_mode = "default"}) {
    %cst = stablehlo.constant dense<0x7FF8000000000000> : tensor<f64> loc(#loc)
    %c = stablehlo.constant dense<0> : tensor<i32> loc(#loc)
    %0:5 = stablehlo.custom_call @cusolver_gesvdj_ffi(%arg0) {mhlo.backend_config = {compute_uv = true, full_matrices = true}, operand_layouts = [dense<[1, 2, 0]> : tensor<3xindex>], output_operand_aliases = [#stablehlo.output_operand_alias<output_tuple_indices = [0], operand_index = 0, operand_tuple_indices = []>], result_layouts = [dense<[1, 2, 0]> : tensor<3xindex>, dense<[1, 0]> : tensor<2xindex>, dense<[1, 2, 0]> : tensor<3xindex>, dense<[1, 2, 0]> : tensor<3xindex>, dense<0> : tensor<1xindex>]} : (tensor<2x4x4xf64>) -> (tensor<2x4x4xf64>, tensor<2x4xf64>, tensor<2x4x4xf64>, tensor<2x4x4xf64>, tensor<2xi32>) loc(#loc3)
    %1 = stablehlo.transpose %0#3, dims = [0, 2, 1] : (tensor<2x4x4xf64>) -> tensor<2x4x4xf64> loc(#loc3)
    %2 = stablehlo.broadcast_in_dim %c, dims = [] : (tensor<i32>) -> tensor<2xi32> loc(#loc3)
    %3 = stablehlo.compare  EQ, %0#4, %2,  SIGNED : (tensor<2xi32>, tensor<2xi32>) -> tensor<2xi1> loc(#loc3)
    %4 = stablehlo.broadcast_in_dim %3, dims = [0] : (tensor<2xi1>) -> tensor<2x1xi1> loc(#loc3)
    %5 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f64>) -> tensor<2x4xf64> loc(#loc3)
    %6 = stablehlo.broadcast_in_dim %4, dims = [0, 1] : (tensor<2x1xi1>) -> tensor<2x4xi1> loc(#loc3)
    %7 = stablehlo.select %6, %0#1, %5 : tensor<2x4xi1>, tensor<2x4xf64> loc(#loc3)
    %8 = stablehlo.broadcast_in_dim %3, dims = [0] : (tensor<2xi1>) -> tensor<2x1x1xi1> loc(#loc3)
    %9 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f64>) -> tensor<2x4x4xf64> loc(#loc3)
    %10 = stablehlo.broadcast_in_dim %8, dims = [0, 1, 2] : (tensor<2x1x1xi1>) -> tensor<2x4x4xi1> loc(#loc3)
    %11 = stablehlo.select %10, %0#2, %9 : tensor<2x4x4xi1>, tensor<2x4x4xf64> loc(#loc3)
    %12 = stablehlo.broadcast_in_dim %3, dims = [0] : (tensor<2xi1>) -> tensor<2x1x1xi1> loc(#loc3)
    %13 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f64>) -> tensor<2x4x4xf64> loc(#loc3)
    %14 = stablehlo.broadcast_in_dim %12, dims = [0, 1, 2] : (tensor<2x1x1xi1>) -> tensor<2x4x4xi1> loc(#loc3)
    %15 = stablehlo.select %14, %1, %13 : tensor<2x4x4xi1>, tensor<2x4x4xf64> loc(#loc3)
    return %11, %7, %15 : tensor<2x4x4xf64>, tensor<2x4xf64>, tensor<2x4x4xf64> loc(#loc)
  } loc(#loc)
} loc(#loc)
#loc = loc(unknown)
#loc2 = loc("third_party/py/jax/tests/export_back_compat_test.py":686:13)
#loc3 = loc("jit(func)/jit(main)/svd"(#loc2))
""",
    mlir_module_serialized=b'ML\xefR\rStableHLO_v1.7.0\x00\x01#\x05\x01\x05\x13\x01\x03\x0b\x03\x11\x0f\x13\x17\x1b\x1f#\'+\x03\xb7q3\x01!\x0f\x07\x0f#\x0b\x0f\x0b\x0b\x0b\x0f\x0b\x13\x0b\x0b\x17\x0b\x03Q\x0b\x0bo\x0f\x0b/\x0bo\x0f\x13\x0b\x17\x1b\x0b\x1b\x0b\x1b\x0b\x0b\x0b/\x1f\x1b\x0b\x0b\x0b\x0b\x0b\x0b\x0b\x0f\x0f\x17\x1fO/o\x0b\x0bO\x01\x05\x0b\x0f\x03/\x1b\x07\x17\x07\x07\x07\x0f\x0f\x07\x13\x13\x1b\x1b\x1f\x13\x13\x13\x13\x13\x17\x13\x17\x13\x02"\x06\x1d\x1b\x1d\x1f\x11\x03\x05\x03\x07\t\x0b\r\x05\x0f\x05\x05\x17\x11\x01\x00\x05\x19\x05\x1b\x05\x1d\x1d\x15\x03\x05\x1f\x03\x03\x19M\x05!\x05#\x17\x1f\xba\n\x1b\x05%\x1d\'\x1d)\x1f!1\x01\x00\x00\x00\x00\x00\x00\x00\x02\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x1f\'\x01\x1d+\x1f-\x11\x00\x00\x00\x00\x00\x00\x00\x00\x05\x03\x1f\x191\x00\x00\x00\x00\x00\x00\x00\x00\x01\x00\x00\x00\x00\x00\x00\x00\x02\x00\x00\x00\x00\x00\x00\x00\x03\x033\r\x03!##\x1f\x03\x079=A\r\x05);!#\x1d-\r\x05)?!#\x1d/\r\x05)C!#\x1d1\x1d3\x1d5\x1f\x11\x11\x00\x00\x00\x00\x00\x00\xf8\x7f\x1f\x13\t\x00\x00\x00\x00\r\x05O-Q-\x1d7\x1d9\x0b\x03\x1d;\x1d=\x03\x01\x05\x01\x03\x03%\x03\x03a\x15\x03\x01\x01\x01\x03\x0b%e%%g\x1f#!\x01\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x1f%\x11\x00\x00\x00\x00\x00\x00\x00\x00\x1f\x191\x00\x00\x00\x00\x00\x00\x00\x00\x02\x00\x00\x00\x00\x00\x00\x00\x01\x00\x00\x00\x00\x00\x00\x00\t\x07\x07\x01\x1f1!\x00\x00\x00\x00\x00\x00\x00\x00\x01\x00\x00\x00\x00\x00\x00\x00\x01\t\x01\x02\x02)\x07\t\x11\x11\r\x01)\x05\t\x11\r\x1d\x0b\x13)\x01\r)\x01\x15\x1b)\x03\t\x15)\x03\r\x0b)\x07\t\x05\x05\x07)\x07\t\x11\x11\x07\x11\x03\x05\x07\x05\t\x05)\x03\r\x0f)\x03\t\x0f)\x03\x05\x0f)\x03\x01\x0b)\x03\t\x07)\x05\t\x05\x07)\x03\x05\x0b)\x05\t\x11\x07)\x03\t\x0b\x04\xe2\x02\x05\x01Q\x03\x07\x01\x07\x04\xba\x02\x03\x01\x05\tP\x03\x03\x07\x04\x8e\x02\x03/O\x03\x0b\x13\x00\x07B\x03\x05\x03\x11\x07B\x03\x07\x03\x13\x0bG\x01\x17\t\x0b\x05\t\x05\x05\x17\x03\x01\rF\x01\x0b\x03\x05\x03\r\x03F\x01\r\x03\x17\x03\x05\x0fF\x01\x0f\x03)\x05\x0f\x13\x03F\x01\x11\x03+\x03\x15\x03F\x01\r\x03\t\x03\x03\x03F\x01\x13\x03/\x03\x17\x05\x06\x01\x03\t\x07\x1b\t\x19\x03F\x01\x11\x03\x1b\x03\x15\x03F\x01\r\x03\x05\x03\x03\x03F\x01\x15\x03\x1d\x03\x1f\x05\x06\x01\x03\x05\x07#\x0b!\x03F\x01\x11\x03\x1b\x03\x15\x03F\x01\r\x03\x05\x03\x03\x03F\x01\x15\x03\x1d\x03\'\x05\x06\x01\x03\x05\x07+\x11)\x11\x04\x03\x07%\x1d-\x06\x03\x01\x05\x01\x00\xe6\x06?)\x03\x1d\x17\x0f\x0b\t\t\t!\x11#i1)\x11\x13%)9\x15\x17\x1b\x1f\x11\x19\x15)\x0f\x0b\x11builtin\x00vhlo\x00module\x00broadcast_in_dim_v1\x00select_v1\x00constant_v1\x00func_v1\x00custom_call_v1\x00transpose_v1\x00compare_v1\x00return_v1\x00jax.uses_shape_polymorphism\x00mhlo.num_partitions\x00mhlo.num_replicas\x00jit_func\x00operand\x00mhlo.backend_config\x00jit(func)/jit(main)/svd\x00third_party/py/jax/tests/export_back_compat_test.py\x00mhlo.layout_mode\x00default\x00jax.result_info\x00[0]\x00[1]\x00[2]\x00main\x00public\x00compute_uv\x00full_matrices\x00\x00cusolver_gesvdj_ffi\x00\x08I\x17\x05#\x01\x0b157EG\x03I\x03K\x11SUWY[]_c\x03i\x03\'\x05km\x03+\x03o\x03/',
    xla_call_module_version=9,
    nr_devices=1,
)  # End paste

data_2024_10_08["jacobi"]["c64"] = dict(
    testdata_version=1,
    platform='cuda',
    custom_call_targets=['cusolver_gesvdj_ffi'],
    serialized_date=datetime.date(2024, 10, 8),
    inputs=(array([[[ 2.4649541 -5.8653884j ,  5.5100183 +2.0214202j ,
         -2.4541297 +1.862114j  ,  5.4709225 +4.409564j  ],
        [ 1.1091617 +2.325679j  , -5.0506334 +5.5802264j ,
         -1.7254959 -1.5569435j ,  3.002013  -2.7583091j ],
        [ 0.8154569 +5.66862j   , -0.7711735 +1.8621845j ,
          1.2456422 -1.1770611j ,  0.03156909-0.22670403j],
        [ 3.9012303 -2.0669405j , -2.7752936 -0.71004313j,
         -1.0354352 -5.5713825j ,  1.554125  +0.9581067j ]],

       [[-2.077837  +6.2613506j , -2.0213401 -3.2755377j ,
         -2.1061401 -0.4942127j , -5.098616  +3.4436114j ],
        [ 3.6104472 +0.75928044j,  1.3155019 -3.6494553j ,
          0.58335614-0.8751654j , -1.1484178 -4.0733714j ],
        [-0.4858576 +4.38415j   , -2.3318157 +2.744366j  ,
         -0.7987209 -0.23303579j, -2.6747904 +1.5206293j ],
        [ 4.013358  +0.978174j  , -2.707136  -0.29939318j,
         -2.5241148 -1.44767j   , -1.8191104 +0.26034543j]]],
      dtype=complex64),),
    expected_outputs=(array([[[-7.7744454e-01+0.09018909j  , -2.1625498e-01-0.16622283j  ,
         -1.6743444e-01+0.5261705j   ,  8.9576304e-02+0.01171514j  ],
        [ 1.2709992e-01-0.38680157j  , -3.4196457e-01+0.6261395j   ,
          1.3424198e-01+0.37196818j  , -1.5489596e-01+0.38061285j  ],
        [ 2.6221693e-01-0.25399417j  ,  1.7437853e-01+0.053393736j ,
          7.3115915e-02+0.51672214j  ,  2.4561302e-01-0.7076709j   ],
        [-5.0269447e-02-0.29305094j  , -5.6865913e-01-0.24491112j  ,
          4.3156582e-01-0.28307965j  ,  5.0845784e-01-0.05768533j  ]],

       [[-1.5179910e-02+0.8092423j   , -1.0622249e-01+0.09009284j  ,
         -5.3503644e-01-0.053553585j ,  1.8145569e-01+0.058631808j ],
        [-1.7091817e-01+0.0025223175j,  7.6241887e-01+0.2851526j   ,
         -3.0385127e-02-0.115484476j ,  3.0117261e-01-0.45080122j  ],
        [ 1.7759532e-01+0.4123873j   , -2.5654158e-01+0.052343685j ,
          6.6419083e-01-0.32588708j  , -1.6986026e-04-0.4271902j   ],
        [ 2.4243295e-01+0.23515853j  ,  4.7519770e-01-0.15375356j  ,
          3.5048491e-01-0.16253117j  ,  8.7767310e-02+0.6924699j   ]]],
      dtype=complex64), array([[13.7465725,  9.692749 ,  6.012994 ,  1.7378366],
       [12.048737 ,  7.9871097,  3.9069395,  1.7972969]], dtype=float32), array([[[-0.29246047 +0.582183j   , -0.5259067  -0.27628872j ,
          0.3469344  -0.15329583j , -0.19642796 -0.20042528j ],
        [ 0.02593917 +0.33676162j ,  0.55821204 +0.18806678j ,
          0.20056807 +0.35542676j , -0.5978458  -0.12236632j ],
        [ 0.46109042 -0.034899022j,  0.24078317 -0.19413127j ,
          0.19841644 -0.3350929j  ,  0.17725058 -0.71234804j ],
        [-0.48504043 -0.1111859j  ,  0.31423682 +0.32512894j ,
          0.23620881 -0.69435585j , -0.04027982 +0.091500096j]],

       [[ 0.6148358  +0.14274344j , -0.23763065 +0.3584556j  ,
         -0.13778959 +0.19841002j ,  0.234248   +0.55083406j ],
        [ 0.734292   -0.11843189j , -0.07720101 -0.4717579j  ,
         -0.051302787-0.19603764j , -0.16576114 -0.38695592j ],
        [ 0.019250087+0.17437238j , -0.43637297 +0.6207004j  ,
          0.033975117-0.27825257j ,  0.024782527-0.5606609j  ],
        [-0.060103483+0.11833174j , -0.074751794-0.072441235j,
         -0.5370554  +0.7304642j  ,  0.07712744 -0.37893948j ]]],
      dtype=complex64)),
    mlir_module_text=r"""
#loc1 = loc("operand")
module @jit_func attributes {jax.uses_shape_polymorphism = false, mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} {
  func.func public @main(%arg0: tensor<2x4x4xcomplex<f32>> {mhlo.layout_mode = "default"} loc("operand")) -> (tensor<2x4x4xcomplex<f32>> {jax.result_info = "[0]", mhlo.layout_mode = "default"}, tensor<2x4xf32> {jax.result_info = "[1]", mhlo.layout_mode = "default"}, tensor<2x4x4xcomplex<f32>> {jax.result_info = "[2]", mhlo.layout_mode = "default"}) {
    %cst = stablehlo.constant dense<(0x7FC00000,0x7FC00000)> : tensor<complex<f32>> loc(#loc)
    %cst_0 = stablehlo.constant dense<0x7FC00000> : tensor<f32> loc(#loc)
    %c = stablehlo.constant dense<0> : tensor<i32> loc(#loc)
    %0:5 = stablehlo.custom_call @cusolver_gesvdj_ffi(%arg0) {mhlo.backend_config = {compute_uv = true, full_matrices = true}, operand_layouts = [dense<[1, 2, 0]> : tensor<3xindex>], output_operand_aliases = [#stablehlo.output_operand_alias<output_tuple_indices = [0], operand_index = 0, operand_tuple_indices = []>], result_layouts = [dense<[1, 2, 0]> : tensor<3xindex>, dense<[1, 0]> : tensor<2xindex>, dense<[1, 2, 0]> : tensor<3xindex>, dense<[1, 2, 0]> : tensor<3xindex>, dense<0> : tensor<1xindex>]} : (tensor<2x4x4xcomplex<f32>>) -> (tensor<2x4x4xcomplex<f32>>, tensor<2x4xf32>, tensor<2x4x4xcomplex<f32>>, tensor<2x4x4xcomplex<f32>>, tensor<2xi32>) loc(#loc3)
    %1 = stablehlo.transpose %0#3, dims = [0, 2, 1] : (tensor<2x4x4xcomplex<f32>>) -> tensor<2x4x4xcomplex<f32>> loc(#loc3)
    %2 = stablehlo.real %1 : (tensor<2x4x4xcomplex<f32>>) -> tensor<2x4x4xf32> loc(#loc3)
    %3 = stablehlo.imag %1 : (tensor<2x4x4xcomplex<f32>>) -> tensor<2x4x4xf32> loc(#loc3)
    %4 = stablehlo.negate %3 : tensor<2x4x4xf32> loc(#loc3)
    %5 = stablehlo.complex %2, %4 : tensor<2x4x4xcomplex<f32>> loc(#loc3)
    %6 = stablehlo.broadcast_in_dim %c, dims = [] : (tensor<i32>) -> tensor<2xi32> loc(#loc3)
    %7 = stablehlo.compare  EQ, %0#4, %6,  SIGNED : (tensor<2xi32>, tensor<2xi32>) -> tensor<2xi1> loc(#loc3)
    %8 = stablehlo.broadcast_in_dim %7, dims = [0] : (tensor<2xi1>) -> tensor<2x1xi1> loc(#loc3)
    %9 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<2x4xf32> loc(#loc3)
    %10 = stablehlo.broadcast_in_dim %8, dims = [0, 1] : (tensor<2x1xi1>) -> tensor<2x4xi1> loc(#loc3)
    %11 = stablehlo.select %10, %0#1, %9 : tensor<2x4xi1>, tensor<2x4xf32> loc(#loc3)
    %12 = stablehlo.broadcast_in_dim %7, dims = [0] : (tensor<2xi1>) -> tensor<2x1x1xi1> loc(#loc3)
    %13 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<complex<f32>>) -> tensor<2x4x4xcomplex<f32>> loc(#loc3)
    %14 = stablehlo.broadcast_in_dim %12, dims = [0, 1, 2] : (tensor<2x1x1xi1>) -> tensor<2x4x4xi1> loc(#loc3)
    %15 = stablehlo.select %14, %0#2, %13 : tensor<2x4x4xi1>, tensor<2x4x4xcomplex<f32>> loc(#loc3)
    %16 = stablehlo.broadcast_in_dim %7, dims = [0] : (tensor<2xi1>) -> tensor<2x1x1xi1> loc(#loc3)
    %17 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<complex<f32>>) -> tensor<2x4x4xcomplex<f32>> loc(#loc3)
    %18 = stablehlo.broadcast_in_dim %16, dims = [0, 1, 2] : (tensor<2x1x1xi1>) -> tensor<2x4x4xi1> loc(#loc3)
    %19 = stablehlo.select %18, %5, %17 : tensor<2x4x4xi1>, tensor<2x4x4xcomplex<f32>> loc(#loc3)
    return %15, %11, %19 : tensor<2x4x4xcomplex<f32>>, tensor<2x4xf32>, tensor<2x4x4xcomplex<f32>> loc(#loc)
  } loc(#loc)
} loc(#loc)
#loc = loc(unknown)
#loc2 = loc("third_party/py/jax/tests/export_back_compat_test.py":686:13)
#loc3 = loc("jit(func)/jit(main)/svd"(#loc2))
""",
    mlir_module_serialized=b"ML\xefR\rStableHLO_v1.7.0\x00\x01+\x05\x01\x05\x1b\x01\x03\x0b\x03\x19\x0f\x13\x17\x1b\x1f#'+/37;\x03\xbfs9\x01!\x0f\x07\x0f#\x0b\x0f\x0b\x0b\x0b\x0f\x0b\x13\x0b\x0b\x17\x0b\x03S\x0b\x0bo\x0f\x0b/\x0bo\x0f\x13\x0b\x17\x1b\x0b\x1b\x0b\x1b\x0b\x0b\x0b/\x1f\x1f\x1b\x0b\x0b\x0b\x0b\x0b\x0b\x0b\x0f\x0f\x17\x1fO/o\x0b\x0bO\x01\x05\x0b\x0f\x035\x1b\x07\x07\x17\x07\x07\x1b\x0b\x0f\x0f\x0f\x07\x13\x13\x1b\x1b\x1f\x13\x13\x13\x13\x13\x17\x13\x17\x13\x02j\x06\x1d\x1b\x1d\x1f\x11\x03\x05\x03\x07\t\x0b\r\x05\x0f\x05\x05\x1f\x11\x01\x00\x05!\x05#\x05%\x1d\x15\x03\x05'\x03\x03\x19O\x05)\x05+\x17\x1f\xba\n\x1b\x05-\x1d/\x1d1\x1f'1\x01\x00\x00\x00\x00\x00\x00\x00\x02\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x1f-\x01\x1d3\x1f3\x11\x00\x00\x00\x00\x00\x00\x00\x00\x05\x03\x1f\x1f1\x00\x00\x00\x00\x00\x00\x00\x00\x01\x00\x00\x00\x00\x00\x00\x00\x02\x00\x00\x00\x00\x00\x00\x00\x03\x033\r\x03!##%\x03\x079=A\r\x05);!#\x1d5\r\x05)?!#\x1d7\r\x05)C!#\x1d9\x1d;\x1d=\x1f\x15\x11\x00\x00\xc0\x7f\x00\x00\xc0\x7f\x1f\x17\t\x00\x00\xc0\x7f\x1f\x19\t\x00\x00\x00\x00\r\x05Q-S-\x1d?\x1dA\x0b\x03\x1dC\x1dE\x03\x01\x05\x01\x03\x03%\x03\x03c\x15\x03\x01\x01\x01\x03\x0b%g%%i\x1f)!\x01\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x1f+\x11\x00\x00\x00\x00\x00\x00\x00\x00\x1f\x1f1\x00\x00\x00\x00\x00\x00\x00\x00\x02\x00\x00\x00\x00\x00\x00\x00\x01\x00\x00\x00\x00\x00\x00\x00\t\x07\x07\x01\x1f7!\x00\x00\x00\x00\x00\x00\x00\x00\x01\x00\x00\x00\x00\x00\x00\x00\x01\t\x01\x02\x02)\x07\t\x11\x11\x13\x01\t)\x05\t\x11\t\x1d\x13)\x07\t\x11\x11\t\x03\t)\x01\x13)\x01\t)\x01\x1b\x1b)\x03\t\x1b)\x03\r\r)\x07\t\x05\x05\x07)\x07\t\x11\x11\x07\x11\x03\x05\x07\x05\x0b\x05)\x03\r\x0f)\x03\t\x0f)\x03\x05\x0f)\x03\x01\r)\x03\t\x07)\x05\t\x05\x07)\x03\x05\r)\x05\t\x11\x07)\x03\t\r\x04n\x03\x05\x01Q\x03\x07\x01\x07\x04F\x03\x03\x01\x05\tP\x03\x03\x07\x04\x1a\x03\x039c\x03\x0b\x13\x00\x05B\x03\x05\x03\x15\x05B\x03\x07\x03\x17\x05B\x03\t\x03\x19\x0bG\x01\x17\x0b\x0b\x05\x0b\x05\x05\x1d\x03\x01\rF\x01\r\x03\x05\x03\x0f\x0f\x06\x01\x03\x11\x03\x13\x11\x06\x01\x03\x11\x03\x13\x13\x06\x01\x03\x11\x03\x17\x15\x06\x01\x03\x05\x05\x15\x19\x03F\x01\x0f\x03\x1d\x03\x07\x17F\x01\x11\x03/\x05\x11\x1d\x03F\x01\x13\x031\x03\x1f\x03F\x01\x0f\x03\x0b\x03\x05\x03F\x01\x15\x035\x03!\x07\x06\x01\x03\x0b\x07%\x0b#\x03F\x01\x13\x03!\x03\x1f\x03F\x01\x0f\x03\x05\x03\x03\x03F\x01\x17\x03#\x03)\x07\x06\x01\x03\x05\x07-\r+\x03F\x01\x13\x03!\x03\x1f\x03F\x01\x0f\x03\x05\x03\x03\x03F\x01\x17\x03#\x031\x07\x06\x01\x03\x05\x075\x1b3\x19\x04\x03\x07/'7\x06\x03\x01\x05\x01\x00\x8a\x07G)\x03\x1d\x17\x0f\x0b\t\t\t!\x11#i1)\x11\x13%)9\x15\x17\x17\x15\x11\x11\x1b\x1f\x11\x15\x19)\x0f\x0b\x11builtin\x00vhlo\x00module\x00broadcast_in_dim_v1\x00constant_v1\x00select_v1\x00func_v1\x00custom_call_v1\x00transpose_v1\x00real_v1\x00imag_v1\x00negate_v1\x00complex_v1\x00compare_v1\x00return_v1\x00jax.uses_shape_polymorphism\x00mhlo.num_partitions\x00mhlo.num_replicas\x00jit_func\x00operand\x00mhlo.backend_config\x00jit(func)/jit(main)/svd\x00third_party/py/jax/tests/export_back_compat_test.py\x00mhlo.layout_mode\x00default\x00jax.result_info\x00[0]\x00[1]\x00[2]\x00main\x00public\x00compute_uv\x00full_matrices\x00\x00cusolver_gesvdj_ffi\x00\x08M\x19\x05#\x01\x0b157EG\x03I\x03K\x03M\x11UWY[]_ae\x03k\x03'\x05mo\x03+\x03q\x03/",
    xla_call_module_version=9,
    nr_devices=1,
)  # End paste

data_2024_10_08["jacobi"]["c128"] = dict(
    testdata_version=1,
    platform='cuda',
    custom_call_targets=['cusolver_gesvdj_ffi'],
    serialized_date=datetime.date(2024, 10, 8),
    inputs=(array([[[ 3.796399075115019  +0.6837988589198791j ,
         -3.4038861521636834 -0.8791658549931785j ,
          5.587025360536978  +1.3741531770045181j ,
          2.546764576857623  +1.6809085185078099j ],
        [-0.29341875154987235-0.7960296026166453j ,
         -3.978920434463012  -3.064639731239574j  ,
          7.797646929978315  +0.8348826539375432j ,
          1.3257327511547885 +4.524484078738454j  ],
        [ 2.664074451502439  +3.393033766292215j  ,
         -1.9919844260791377 -0.6279428424058938j ,
          4.9406207893724465 -1.3141491766038624j ,
         -0.09537336365814258-1.6177405195744095j ],
        [-2.4688921567972058 +0.9746213770706899j ,
          0.5921515121270726 +4.164182480017167j  ,
         -2.2589950508277568 +2.6432862086413222j ,
         -2.556559707542412  +8.972441493886869j  ]],

       [[-1.1112549656532051 +1.9704765658574988j ,
          8.989616960518267  -3.6609418049818268j ,
          1.2346549691110358 -0.24414962907971388j,
         -2.9090211908941734 +4.153707428606018j  ],
        [-0.5588638014996116 -1.67573324865607j   ,
          3.149773407631379  +4.604525381223088j  ,
          1.128102476802749  -1.3129091142659617j ,
         -3.9361491309229013 -0.4879709640370399j ],
        [-1.7318669937731295 -2.869679468975394j  ,
         -3.0523599142955913 +8.95268082648917j   ,
         -2.6723736459369936 -2.1677507057699845j ,
          2.2856025738573873 +4.128295675578359j  ],
        [-3.0608685537602445 -0.21903335326027606j,
         -4.833657640993765  -2.184980999873441j  ,
         -1.4399110875167116 +0.23459254553652992j,
          1.9463909302306492 -6.990119453626141j  ]]]),),
    expected_outputs=(array([[[ 0.5058374100694445  +0.08736973325822138j ,
         -0.15357420292088966 +0.15092672888666417j ,
          0.4000099867967776  +0.03343224253498994j ,
         -0.08606623675966052 -0.7222174392115995j  ],
        [ 0.5494982003369341  +0.21627530772570236j ,
         -0.24559859535168804 +0.42796239936478525j ,
         -0.35789689606765984 -0.3818532162746145j  ,
          0.1870806116265344  +0.3144916716626133j  ],
        [ 0.35367519290177574 -0.09104684890511797j ,
         -0.2722754551548636  -0.01520202737842482j ,
          0.20564615603913944 +0.6788851354546692j  ,
         -0.26426814455261194 +0.46823742188539086j ],
        [-0.26578682435990086 +0.42866473683552114j ,
          0.3753385929542211  +0.7035065862655021j  ,
          0.11671177248101379 +0.21948854675620316j ,
         -0.21652589234437378 +0.03351132737269058j ]],

       [[ 0.6447398173517832  -0.15549362721502685j ,
          0.15703159957366286 -0.18175797380505443j ,
         -0.26984447689730545 -0.021902292069557842j,
         -0.5809856180118731  +0.30265058246070115j ],
        [ 0.1676295953197978  +0.35517400244986735j ,
          0.1471535132890569  -0.11945200848513432j ,
         -0.048587988684360706-0.5959669737080945j  ,
          0.4658299942602748  +0.485070920592558j   ],
        [-0.18394077259047054 +0.4607473838424225j  ,
         -0.6953163697460342  -0.28046734855011674j ,
         -0.31852936346663746 +0.1408757043223249j  ,
         -0.18675394341953266 +0.18859188206317357j ],
        [-0.38736135459868454 -0.0985538837287597j  ,
          0.1400965142945759  +0.5697616659026354j  ,
         -0.6360840273269505  -0.20798320176325613j ,
         -0.16572122692263513 +0.14373411782489567j ]]]), array([[14.800219047973494 , 10.208626444252278 ,  5.121442071722992 ,
         2.3317857898198886],
       [16.387010501961203 , 10.85923071644345  ,  4.434400577803048 ,
         0.7913357405499906]]), array([[[ 0.22661733590854113 +0.12716811836891184j ,
         -0.24780255554134717 -0.18478518062614943j ,
          0.7440472412639979  -0.05201706404993294j ,
          0.5257597465394888  +0.0646977585249859j  ],
        [-0.1730295430553622  +0.08448133745769008j ,
          0.3682599006560183  +0.4301598551539888j  ,
         -0.2470430370452839  -0.1549815154245142j  ,
          0.6735915557971368  +0.3217074899922104j  ],
        [ 0.9230886222728419  -0.02650327461707979j ,
          0.26368770031726146 +0.17940633187899951j ,
         -0.07583142147304134 +0.04327009313242354j ,
         -0.11210558432341579 +0.15904958085235385j ],
        [ 0.13986036936906365 +0.15179124656114312j ,
         -0.2301030912215956  -0.6550803059657956j  ,
         -0.4696876553365799  -0.3611221351860597j  ,
          0.15426668090371212 +0.31702832012208193j ]],

       [[-0.09203102957042177 +0.1296290046576001j  ,
          0.9338316046889822  -0.07199509757761581j ,
          0.035650554858596174+0.04949401150838972j ,
         -0.11826036178363568 +0.2824813592971463j  ],
        [ 0.09583662270821797 +0.27782657909919606j ,
         -0.029480552998465036-0.2320825865452519j  ,
          0.27250121923986637 +0.16010847179743545j ,
         -0.7541782211683652  -0.4361420378707925j  ],
        [ 0.77179564007941    -0.03313538642869378j ,
          0.11720446778594507 +0.18064177478753635j ,
          0.40879934622962677 +0.326378840223041j   ,
          0.2808442732051157  -0.06596695716827156j ],
        [ 0.5393502219452392  +0.0262529741146748j  ,
          0.14580725795066302 -0.020389102179641145j,
         -0.6823399052632065  -0.3964340353663506j  ,
         -0.12461789414204025 -0.2201348160267379j  ]]])),
    mlir_module_text=r"""
#loc1 = loc("operand")
module @jit_func attributes {jax.uses_shape_polymorphism = false, mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} {
  func.func public @main(%arg0: tensor<2x4x4xcomplex<f64>> {mhlo.layout_mode = "default"} loc("operand")) -> (tensor<2x4x4xcomplex<f64>> {jax.result_info = "[0]", mhlo.layout_mode = "default"}, tensor<2x4xf64> {jax.result_info = "[1]", mhlo.layout_mode = "default"}, tensor<2x4x4xcomplex<f64>> {jax.result_info = "[2]", mhlo.layout_mode = "default"}) {
    %cst = stablehlo.constant dense<(0x7FF8000000000000,0x7FF8000000000000)> : tensor<complex<f64>> loc(#loc)
    %cst_0 = stablehlo.constant dense<0x7FF8000000000000> : tensor<f64> loc(#loc)
    %c = stablehlo.constant dense<0> : tensor<i32> loc(#loc)
    %0:5 = stablehlo.custom_call @cusolver_gesvdj_ffi(%arg0) {mhlo.backend_config = {compute_uv = true, full_matrices = true}, operand_layouts = [dense<[1, 2, 0]> : tensor<3xindex>], output_operand_aliases = [#stablehlo.output_operand_alias<output_tuple_indices = [0], operand_index = 0, operand_tuple_indices = []>], result_layouts = [dense<[1, 2, 0]> : tensor<3xindex>, dense<[1, 0]> : tensor<2xindex>, dense<[1, 2, 0]> : tensor<3xindex>, dense<[1, 2, 0]> : tensor<3xindex>, dense<0> : tensor<1xindex>]} : (tensor<2x4x4xcomplex<f64>>) -> (tensor<2x4x4xcomplex<f64>>, tensor<2x4xf64>, tensor<2x4x4xcomplex<f64>>, tensor<2x4x4xcomplex<f64>>, tensor<2xi32>) loc(#loc3)
    %1 = stablehlo.transpose %0#3, dims = [0, 2, 1] : (tensor<2x4x4xcomplex<f64>>) -> tensor<2x4x4xcomplex<f64>> loc(#loc3)
    %2 = stablehlo.real %1 : (tensor<2x4x4xcomplex<f64>>) -> tensor<2x4x4xf64> loc(#loc3)
    %3 = stablehlo.imag %1 : (tensor<2x4x4xcomplex<f64>>) -> tensor<2x4x4xf64> loc(#loc3)
    %4 = stablehlo.negate %3 : tensor<2x4x4xf64> loc(#loc3)
    %5 = stablehlo.complex %2, %4 : tensor<2x4x4xcomplex<f64>> loc(#loc3)
    %6 = stablehlo.broadcast_in_dim %c, dims = [] : (tensor<i32>) -> tensor<2xi32> loc(#loc3)
    %7 = stablehlo.compare  EQ, %0#4, %6,  SIGNED : (tensor<2xi32>, tensor<2xi32>) -> tensor<2xi1> loc(#loc3)
    %8 = stablehlo.broadcast_in_dim %7, dims = [0] : (tensor<2xi1>) -> tensor<2x1xi1> loc(#loc3)
    %9 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f64>) -> tensor<2x4xf64> loc(#loc3)
    %10 = stablehlo.broadcast_in_dim %8, dims = [0, 1] : (tensor<2x1xi1>) -> tensor<2x4xi1> loc(#loc3)
    %11 = stablehlo.select %10, %0#1, %9 : tensor<2x4xi1>, tensor<2x4xf64> loc(#loc3)
    %12 = stablehlo.broadcast_in_dim %7, dims = [0] : (tensor<2xi1>) -> tensor<2x1x1xi1> loc(#loc3)
    %13 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<complex<f64>>) -> tensor<2x4x4xcomplex<f64>> loc(#loc3)
    %14 = stablehlo.broadcast_in_dim %12, dims = [0, 1, 2] : (tensor<2x1x1xi1>) -> tensor<2x4x4xi1> loc(#loc3)
    %15 = stablehlo.select %14, %0#2, %13 : tensor<2x4x4xi1>, tensor<2x4x4xcomplex<f64>> loc(#loc3)
    %16 = stablehlo.broadcast_in_dim %7, dims = [0] : (tensor<2xi1>) -> tensor<2x1x1xi1> loc(#loc3)
    %17 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<complex<f64>>) -> tensor<2x4x4xcomplex<f64>> loc(#loc3)
    %18 = stablehlo.broadcast_in_dim %16, dims = [0, 1, 2] : (tensor<2x1x1xi1>) -> tensor<2x4x4xi1> loc(#loc3)
    %19 = stablehlo.select %18, %5, %17 : tensor<2x4x4xi1>, tensor<2x4x4xcomplex<f64>> loc(#loc3)
    return %15, %11, %19 : tensor<2x4x4xcomplex<f64>>, tensor<2x4xf64>, tensor<2x4x4xcomplex<f64>> loc(#loc)
  } loc(#loc)
} loc(#loc)
#loc = loc(unknown)
#loc2 = loc("third_party/py/jax/tests/export_back_compat_test.py":686:13)
#loc3 = loc("jit(func)/jit(main)/svd"(#loc2))
""",
    mlir_module_serialized=b"ML\xefR\rStableHLO_v1.7.0\x00\x01+\x05\x01\x05\x1b\x01\x03\x0b\x03\x19\x0f\x13\x17\x1b\x1f#'+/37;\x03\xbfs9\x01!\x0f\x07\x0f#\x0b\x0f\x0b\x0b\x0b\x0f\x0b\x13\x0b\x0b\x17\x0b\x03S\x0b\x0bo\x0f\x0b/\x0bo\x0f\x13\x0b\x17\x1b\x0b\x1b\x0b\x1b\x0b\x0b\x0bO/\x1f\x1b\x0b\x0b\x0b\x0b\x0b\x0b\x0b\x0f\x0f\x17\x1fO/o\x0b\x0bO\x01\x05\x0b\x0f\x035\x1b\x07\x07\x17\x07\x07\x1b\x0b\x0f\x0f\x0f\x07\x13\x13\x1b\x1b\x1f\x13\x13\x13\x13\x13\x17\x13\x17\x13\x02\x9a\x06\x1d\x1b\x1d\x1f\x11\x03\x05\x03\x07\t\x0b\r\x05\x0f\x05\x05\x1f\x11\x01\x00\x05!\x05#\x05%\x1d\x15\x03\x05'\x03\x03\x19O\x05)\x05+\x17\x1f\xba\n\x1b\x05-\x1d/\x1d1\x1f'1\x01\x00\x00\x00\x00\x00\x00\x00\x02\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x1f-\x01\x1d3\x1f3\x11\x00\x00\x00\x00\x00\x00\x00\x00\x05\x03\x1f\x1f1\x00\x00\x00\x00\x00\x00\x00\x00\x01\x00\x00\x00\x00\x00\x00\x00\x02\x00\x00\x00\x00\x00\x00\x00\x03\x033\r\x03!##%\x03\x079=A\r\x05);!#\x1d5\r\x05)?!#\x1d7\r\x05)C!#\x1d9\x1d;\x1d=\x1f\x15!\x00\x00\x00\x00\x00\x00\xf8\x7f\x00\x00\x00\x00\x00\x00\xf8\x7f\x1f\x17\x11\x00\x00\x00\x00\x00\x00\xf8\x7f\x1f\x19\t\x00\x00\x00\x00\r\x05Q-S-\x1d?\x1dA\x0b\x03\x1dC\x1dE\x03\x01\x05\x01\x03\x03%\x03\x03c\x15\x03\x01\x01\x01\x03\x0b%g%%i\x1f)!\x01\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x1f+\x11\x00\x00\x00\x00\x00\x00\x00\x00\x1f\x1f1\x00\x00\x00\x00\x00\x00\x00\x00\x02\x00\x00\x00\x00\x00\x00\x00\x01\x00\x00\x00\x00\x00\x00\x00\t\x07\x07\x01\x1f7!\x00\x00\x00\x00\x00\x00\x00\x00\x01\x00\x00\x00\x00\x00\x00\x00\x01\t\x01\x02\x02)\x07\t\x11\x11\x13\x01\x0b)\x05\t\x11\t\x1d\x13)\x07\t\x11\x11\t\x03\t)\x01\x13)\x01\t)\x01\x1b\x1b)\x03\t\x1b)\x03\r\r)\x07\t\x05\x05\x07)\x07\t\x11\x11\x07\x11\x03\x05\x07\x05\x0b\x05)\x03\r\x0f)\x03\t\x0f)\x03\x05\x0f)\x03\x01\r)\x03\t\x07)\x05\t\x05\x07)\x03\x05\r)\x05\t\x11\x07)\x03\t\r\x04n\x03\x05\x01Q\x03\x07\x01\x07\x04F\x03\x03\x01\x05\tP\x03\x03\x07\x04\x1a\x03\x039c\x03\x0b\x13\x00\x05B\x03\x05\x03\x15\x05B\x03\x07\x03\x17\x05B\x03\t\x03\x19\x0bG\x01\x17\x0b\x0b\x05\x0b\x05\x05\x1d\x03\x01\rF\x01\r\x03\x05\x03\x0f\x0f\x06\x01\x03\x11\x03\x13\x11\x06\x01\x03\x11\x03\x13\x13\x06\x01\x03\x11\x03\x17\x15\x06\x01\x03\x05\x05\x15\x19\x03F\x01\x0f\x03\x1d\x03\x07\x17F\x01\x11\x03/\x05\x11\x1d\x03F\x01\x13\x031\x03\x1f\x03F\x01\x0f\x03\x0b\x03\x05\x03F\x01\x15\x035\x03!\x07\x06\x01\x03\x0b\x07%\x0b#\x03F\x01\x13\x03!\x03\x1f\x03F\x01\x0f\x03\x05\x03\x03\x03F\x01\x17\x03#\x03)\x07\x06\x01\x03\x05\x07-\r+\x03F\x01\x13\x03!\x03\x1f\x03F\x01\x0f\x03\x05\x03\x03\x03F\x01\x17\x03#\x031\x07\x06\x01\x03\x05\x075\x1b3\x19\x04\x03\x07/'7\x06\x03\x01\x05\x01\x00\x8a\x07G)\x03\x1d\x17\x0f\x0b\t\t\t!\x11#i1)\x11\x13%)9\x15\x17\x17\x15\x11\x11\x1b\x1f\x11\x15\x19)\x0f\x0b\x11builtin\x00vhlo\x00module\x00broadcast_in_dim_v1\x00constant_v1\x00select_v1\x00func_v1\x00custom_call_v1\x00transpose_v1\x00real_v1\x00imag_v1\x00negate_v1\x00complex_v1\x00compare_v1\x00return_v1\x00jax.uses_shape_polymorphism\x00mhlo.num_partitions\x00mhlo.num_replicas\x00jit_func\x00operand\x00mhlo.backend_config\x00jit(func)/jit(main)/svd\x00third_party/py/jax/tests/export_back_compat_test.py\x00mhlo.layout_mode\x00default\x00jax.result_info\x00[0]\x00[1]\x00[2]\x00main\x00public\x00compute_uv\x00full_matrices\x00\x00cusolver_gesvdj_ffi\x00\x08M\x19\x05#\x01\x0b157EG\x03I\x03K\x03M\x11UWY[]_ae\x03k\x03'\x05mo\x03+\x03q\x03/",
    xla_call_module_version=9,
    nr_devices=1,
)  # End paste

data_2024_10_08["qr"]["f32"] = dict(
    testdata_version=1,
    platform='cuda',
    custom_call_targets=['cusolver_gesvd_ffi'],
    serialized_date=datetime.date(2024, 10, 8),
    inputs=(array([[[ 7.064613  ,  4.4742312 , -0.12700312, -0.71483076],
        [-0.59317935,  4.0224333 ,  0.5515773 ,  6.009665  ],
        [-5.193879  , -1.0297644 , -4.388829  ,  2.3485358 ],
        [-0.8724199 , -1.5610907 ,  0.47096923,  0.10478485]],

       [[ 0.7020009 ,  0.0506321 ,  2.5788887 , -0.44895908],
        [ 2.617715  , -1.5580447 , -0.9952533 ,  1.3444504 ],
        [ 0.7077899 , -0.9494638 ,  2.3607216 , -4.8069396 ],
        [-0.9731158 ,  2.762172  , -3.6125846 , -0.59783787]]],
      dtype=float32),),
    expected_outputs=(array([[[-0.7763474  , -0.1832199  , -0.54734766 ,  0.25323072 ],
        [-0.025878029, -0.93325573 ,  0.35778683 ,  0.018767256],
        [ 0.6168491  , -0.29107273 , -0.72102463 ,  0.12205475 ],
        [ 0.12693313 ,  0.10363644 ,  0.22917844 ,  0.959492   ]],

       [[-0.35366213 , -0.12650901 ,  0.30348226 ,  0.8756809  ],
        [ 0.08954996 , -0.56219155 , -0.7897131  ,  0.22863595 ],
        [-0.7685047  ,  0.45641905 , -0.4388071  , -0.09236202 ],
        [ 0.5256468  ,  0.6779513  , -0.30281967 ,  0.41518393 ]]],
      dtype=float32), array([[10.327013  ,  7.659154  ,  3.736682  ,  0.61586076],
       [ 6.3917613 ,  4.5939665 ,  2.8317585 ,  1.2306355 ]],
      dtype=float32), array([[[-0.8505677  , -0.42713356 , -0.24819751 ,  0.18024875 ],
        [ 0.088860154, -0.5791473  ,  0.108991824, -0.8030028  ],
        [-0.14292236 , -0.16727918 ,  0.94716436 ,  0.23338944 ],
        [ 0.49820864 , -0.6739162  , -0.17145996 ,  0.51790595 ]],

       [[-0.16729502 ,  0.31668338 , -0.7375667  ,  0.5724681  ],
        [-0.41296393 ,  0.5025677  , -0.24780482 , -0.7179688  ],
        [-0.6604038  ,  0.29167938 ,  0.5744389  ,  0.38575882 ],
        [ 0.6044337  ,  0.7497071  ,  0.25418177 ,  0.08939337 ]]],
      dtype=float32)),
    mlir_module_text=r"""
#loc1 = loc("operand")
module @jit_func attributes {jax.uses_shape_polymorphism = false, mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} {
  func.func public @main(%arg0: tensor<2x4x4xf32> {mhlo.layout_mode = "default"} loc("operand")) -> (tensor<2x4x4xf32> {jax.result_info = "[0]", mhlo.layout_mode = "default"}, tensor<2x4xf32> {jax.result_info = "[1]", mhlo.layout_mode = "default"}, tensor<2x4x4xf32> {jax.result_info = "[2]", mhlo.layout_mode = "default"}) {
    %cst = stablehlo.constant dense<0x7FC00000> : tensor<f32> loc(#loc)
    %c = stablehlo.constant dense<0> : tensor<i32> loc(#loc)
    %0:5 = stablehlo.custom_call @cusolver_gesvd_ffi(%arg0) {mhlo.backend_config = {compute_uv = true, full_matrices = true, transposed = false}, operand_layouts = [dense<[1, 2, 0]> : tensor<3xindex>], output_operand_aliases = [#stablehlo.output_operand_alias<output_tuple_indices = [0], operand_index = 0, operand_tuple_indices = []>], result_layouts = [dense<[1, 2, 0]> : tensor<3xindex>, dense<[1, 0]> : tensor<2xindex>, dense<[1, 2, 0]> : tensor<3xindex>, dense<[1, 2, 0]> : tensor<3xindex>, dense<0> : tensor<1xindex>]} : (tensor<2x4x4xf32>) -> (tensor<2x4x4xf32>, tensor<2x4xf32>, tensor<2x4x4xf32>, tensor<2x4x4xf32>, tensor<2xi32>) loc(#loc3)
    %1 = stablehlo.broadcast_in_dim %c, dims = [] : (tensor<i32>) -> tensor<2xi32> loc(#loc3)
    %2 = stablehlo.compare  EQ, %0#4, %1,  SIGNED : (tensor<2xi32>, tensor<2xi32>) -> tensor<2xi1> loc(#loc3)
    %3 = stablehlo.broadcast_in_dim %2, dims = [0] : (tensor<2xi1>) -> tensor<2x1xi1> loc(#loc3)
    %4 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<2x4xf32> loc(#loc3)
    %5 = stablehlo.broadcast_in_dim %3, dims = [0, 1] : (tensor<2x1xi1>) -> tensor<2x4xi1> loc(#loc3)
    %6 = stablehlo.select %5, %0#1, %4 : tensor<2x4xi1>, tensor<2x4xf32> loc(#loc3)
    %7 = stablehlo.broadcast_in_dim %2, dims = [0] : (tensor<2xi1>) -> tensor<2x1x1xi1> loc(#loc3)
    %8 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<2x4x4xf32> loc(#loc3)
    %9 = stablehlo.broadcast_in_dim %7, dims = [0, 1, 2] : (tensor<2x1x1xi1>) -> tensor<2x4x4xi1> loc(#loc3)
    %10 = stablehlo.select %9, %0#2, %8 : tensor<2x4x4xi1>, tensor<2x4x4xf32> loc(#loc3)
    %11 = stablehlo.broadcast_in_dim %2, dims = [0] : (tensor<2xi1>) -> tensor<2x1x1xi1> loc(#loc3)
    %12 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<2x4x4xf32> loc(#loc3)
    %13 = stablehlo.broadcast_in_dim %11, dims = [0, 1, 2] : (tensor<2x1x1xi1>) -> tensor<2x4x4xi1> loc(#loc3)
    %14 = stablehlo.select %13, %0#3, %12 : tensor<2x4x4xi1>, tensor<2x4x4xf32> loc(#loc3)
    return %10, %6, %14 : tensor<2x4x4xf32>, tensor<2x4xf32>, tensor<2x4x4xf32> loc(#loc)
  } loc(#loc)
} loc(#loc)
#loc = loc(unknown)
#loc2 = loc("third_party/py/jax/tests/export_back_compat_test.py":690:13)
#loc3 = loc("jit(func)/jit(main)/svd"(#loc2))
""",
    mlir_module_serialized=b"ML\xefR\rStableHLO_v1.7.0\x00\x01!\x05\x01\x05\x11\x01\x03\x0b\x03\x0f\x0f\x13\x17\x1b\x1f#'\x03\xb7q3\x01!\x0f\x07\x0f#\x0b\x0f\x0b\x0b\x0b\x0f\x0b\x13\x0b\x0b\x17\x0b\x03Q\x0b\x0bo\x0f\x0b/\x0b\x0bo\x0f\x13\x0b\x17\x1b\x0b\x1b\x0b\x1b\x0b\x0b\x0b\x1f\x1f#\x0b\x0b\x0b\x0b\x0b\x0b\x0b\x0f\x0f\x17\x1fO/\x0b\x0bO\x01\x05\x0b\x0f\x03/\x1b\x07\x17\x07\x07\x07\x0f\x0f\x07\x13\x1b\x1b\x1f\x13\x13\x13\x13\x13\x17\x13\x17\x13\x13\x02\xb6\x05\x1d\x1b\x1d\x1f\x11\x03\x05\x03\x07\t\x0b\r\x05\x0f\x05\x05\x15\x11\x01\x00\x05\x17\x05\x19\x05\x1b\x1d\x15\x03\x05\x1d\x03\x03\x19O\x05\x1f\x05!\x17\x1f\xca\n\x1b\x05#\x1d%\x1d'\x1f\x1f1\x01\x00\x00\x00\x00\x00\x00\x00\x02\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x1f%\x01\x1d)\x1f+\x11\x00\x00\x00\x00\x00\x00\x00\x00\x05\x03\x05\x01\x1f11\x00\x00\x00\x00\x00\x00\x00\x00\x01\x00\x00\x00\x00\x00\x00\x00\x02\x00\x00\x00\x00\x00\x00\x00\x03\x035\r\x03!##\x1d\x03\x07;?C\r\x05)=!#\x1d+\r\x05)A!#\x1d-\r\x05)E!#\x1d/\x1d1\x1d3\x1f\x11\t\x00\x00\xc0\x7f\x1f\x13\t\x00\x00\x00\x00\r\x07Q-S-U/\x1d5\x1d7\x1d9\x0b\x03\x1d;\x1d=\x03\x01\x03\x03%\x03\x03c\x15\x03\x01\x01\x01\x03\x0b%g%%i\x1f!!\x01\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x1f#\x11\x00\x00\x00\x00\x00\x00\x00\x00\t\x07\x07\x01\x1f/!\x00\x00\x00\x00\x00\x00\x00\x00\x01\x00\x00\x00\x00\x00\x00\x00\x01\t\x01\x02\x02)\x07\t\x11\x11\r\x01)\x05\t\x11\r\x1d\t\x13)\x01\r)\x01\x15\x1b)\x03\t\x15)\x07\t\x05\x05\x07)\x07\t\x11\x11\x07\x11\x03\x05\x07\x05\t\x05)\x03\r\x0f)\x03\t\x0f)\x03\x05\x0f)\x03\x01\x0b)\x03\t\x07)\x05\t\x05\x07)\x03\x05\x0b)\x05\t\x11\x07)\x03\t\x0b)\x03\r\x0b\x04\xc2\x02\x05\x01Q\x03\x07\x01\x07\x04\x9a\x02\x03\x01\x05\tP\x03\x03\x07\x04n\x02\x03-K\x03\x0b\x13\x00\x07B\x03\x05\x03\x11\x07B\x03\x07\x03\x13\x0bG\x01\x17\t\x0b\x05\t\x05\x05\x17\x03\x01\x03F\x01\x0b\x03\x17\x03\x05\rF\x01\r\x03'\x05\x0f\x11\x03F\x01\x0f\x03)\x03\x13\x03F\x01\x0b\x03\t\x03\x03\x03F\x01\x11\x03-\x03\x15\x05\x06\x01\x03\t\x07\x19\t\x17\x03F\x01\x0f\x03\x19\x03\x13\x03F\x01\x0b\x03\x05\x03\x03\x03F\x01\x13\x03\x1b\x03\x1d\x05\x06\x01\x03\x05\x07!\x0b\x1f\x03F\x01\x0f\x03\x19\x03\x13\x03F\x01\x0b\x03\x05\x03\x03\x03F\x01\x13\x03\x1b\x03%\x05\x06\x01\x03\x05\x07)\r'\x0f\x04\x03\x07#\x1b+\x06\x03\x01\x05\x01\x00\xda\x06?'\x03\x17\x1d\x17\x0f\x0b\t\t\t!\x11#i1)\x11\x13%)9\x15\x17\x1f\x11\x19\x15)\x0f\x0b\x11builtin\x00vhlo\x00module\x00broadcast_in_dim_v1\x00select_v1\x00constant_v1\x00func_v1\x00custom_call_v1\x00compare_v1\x00return_v1\x00jax.uses_shape_polymorphism\x00mhlo.num_partitions\x00mhlo.num_replicas\x00jit_func\x00operand\x00mhlo.backend_config\x00jit(func)/jit(main)/svd\x00third_party/py/jax/tests/export_back_compat_test.py\x00mhlo.layout_mode\x00default\x00jax.result_info\x00[0]\x00[1]\x00[2]\x00main\x00public\x00compute_uv\x00full_matrices\x00transposed\x00\x00cusolver_gesvd_ffi\x00\x08E\x15\x05#\x01\x0b379GI\x03K\x03M\x11WY[]/_ae\x03'\x05km\x03+\x03o\x031",
    xla_call_module_version=9,
    nr_devices=1,
)  # End paste

data_2024_10_08["qr"]["f64"] = dict(
    testdata_version=1,
    platform='cuda',
    custom_call_targets=['cusolver_gesvd_ffi'],
    serialized_date=datetime.date(2024, 10, 8),
    inputs=(array([[[ 1.6666730531514784  , -0.283751211500066   ,
          3.0028872858127387  , -3.539066961520449   ],
        [-3.5009517179281326  , -3.0927716333012025  ,
          0.7807364288380038  , -1.7085927853808216  ],
        [ 1.6481964138894265  , -1.0457448512148775  ,
          6.119638350643893   ,  0.6798789946663015  ],
        [ 0.45137958693199876 , -3.0487560288436093  ,
         -3.5653048640383225  ,  3.1078238891060703  ]],

       [[-0.18274271852634477 , -4.107847311422953   ,
         -6.970910660834766   ,  0.026925818090434366],
        [ 0.609826504319294   , -0.08188345529211022 ,
          0.4988730098060886  ,  0.5371129476436916  ],
        [-1.8478868672212718  ,  5.777430685108351   ,
          3.6156805021156426  , -5.4328316768756855  ],
        [ 2.839181461365762   , -5.931652277530413   ,
         -6.898420189145518   ,  0.33823029148249784 ]]]),),
    expected_outputs=(array([[[ 0.5185598212776665  ,  0.08640745162480469 ,
         -0.22604389271595665 ,  0.8200814731634898  ],
        [ 0.058783626848861056,  0.9889125760720038  ,
          0.040834454253227216, -0.13011129638493632 ],
        [ 0.6559139736574813  , -0.09584764795840599 ,
          0.7197537968326447  , -0.20626332559785515 ],
        [-0.5453595659120885  ,  0.07347719082261084 ,
          0.6551268410442749  ,  0.5176802762713159  ]],

       [[ 0.5266925395219686  ,  0.49814307967604127 ,
          0.6468374971059552  ,  0.23674816434446275 ],
        [-0.004398828045962558, -0.15543570506880638 ,
         -0.22846771664609714 ,  0.9610530132891235  ],
        [-0.5473983592728412  ,  0.8074523783002769  ,
         -0.20514413534806356 ,  0.07931946025360624 ],
        [ 0.6503311890022828  ,  0.27516153535306986 ,
         -0.6980828307017146  , -0.11847293172915115 ]]]), array([[ 8.364583718088932  ,  5.052514677781009  ,  4.631967246026687  ,
         2.5284502274249308 ],
       [14.318697073130886  ,  5.219146075652053  ,  2.112106833842507  ,
         0.15087576902549965]]), array([[[ 0.17853631156202626 ,  0.0774459691431368  ,
          0.9039781164136     , -0.3807236167650057  ],
        [-0.6814293626119452  , -0.6346900565025612  ,
          0.036225601414469004, -0.3626434361038585  ],
        [ 0.20775315131153385 , -0.6071183144157039  ,
          0.30699755938819207 ,  0.7028502535752935  ],
        [ 0.6786880265219302  , -0.4717817359132189  ,
         -0.29540441665143263 , -0.47910415040709536 ]],

       [[ 0.19268560067166468 , -0.641350735371731   ,
         -0.7081088748435267  ,  0.22388236844328566 ],
        [-0.1718035739905713  ,  0.19146225969939276 ,
         -0.4845129100140278  , -0.8361058396546505  ],
        [-0.8808414232069264  ,  0.1501707384172312  ,
         -0.25997252143926985 ,  0.3660347313883319  ],
        [ 0.39683016319410813 ,  0.7276401491617723  ,
         -0.4429936224079104  ,  0.34179275213656773 ]]])),
    mlir_module_text=r"""
#loc1 = loc("operand")
module @jit_func attributes {jax.uses_shape_polymorphism = false, mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} {
  func.func public @main(%arg0: tensor<2x4x4xf64> {mhlo.layout_mode = "default"} loc("operand")) -> (tensor<2x4x4xf64> {jax.result_info = "[0]", mhlo.layout_mode = "default"}, tensor<2x4xf64> {jax.result_info = "[1]", mhlo.layout_mode = "default"}, tensor<2x4x4xf64> {jax.result_info = "[2]", mhlo.layout_mode = "default"}) {
    %cst = stablehlo.constant dense<0x7FF8000000000000> : tensor<f64> loc(#loc)
    %c = stablehlo.constant dense<0> : tensor<i32> loc(#loc)
    %0:5 = stablehlo.custom_call @cusolver_gesvd_ffi(%arg0) {mhlo.backend_config = {compute_uv = true, full_matrices = true, transposed = false}, operand_layouts = [dense<[1, 2, 0]> : tensor<3xindex>], output_operand_aliases = [#stablehlo.output_operand_alias<output_tuple_indices = [0], operand_index = 0, operand_tuple_indices = []>], result_layouts = [dense<[1, 2, 0]> : tensor<3xindex>, dense<[1, 0]> : tensor<2xindex>, dense<[1, 2, 0]> : tensor<3xindex>, dense<[1, 2, 0]> : tensor<3xindex>, dense<0> : tensor<1xindex>]} : (tensor<2x4x4xf64>) -> (tensor<2x4x4xf64>, tensor<2x4xf64>, tensor<2x4x4xf64>, tensor<2x4x4xf64>, tensor<2xi32>) loc(#loc3)
    %1 = stablehlo.broadcast_in_dim %c, dims = [] : (tensor<i32>) -> tensor<2xi32> loc(#loc3)
    %2 = stablehlo.compare  EQ, %0#4, %1,  SIGNED : (tensor<2xi32>, tensor<2xi32>) -> tensor<2xi1> loc(#loc3)
    %3 = stablehlo.broadcast_in_dim %2, dims = [0] : (tensor<2xi1>) -> tensor<2x1xi1> loc(#loc3)
    %4 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f64>) -> tensor<2x4xf64> loc(#loc3)
    %5 = stablehlo.broadcast_in_dim %3, dims = [0, 1] : (tensor<2x1xi1>) -> tensor<2x4xi1> loc(#loc3)
    %6 = stablehlo.select %5, %0#1, %4 : tensor<2x4xi1>, tensor<2x4xf64> loc(#loc3)
    %7 = stablehlo.broadcast_in_dim %2, dims = [0] : (tensor<2xi1>) -> tensor<2x1x1xi1> loc(#loc3)
    %8 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f64>) -> tensor<2x4x4xf64> loc(#loc3)
    %9 = stablehlo.broadcast_in_dim %7, dims = [0, 1, 2] : (tensor<2x1x1xi1>) -> tensor<2x4x4xi1> loc(#loc3)
    %10 = stablehlo.select %9, %0#2, %8 : tensor<2x4x4xi1>, tensor<2x4x4xf64> loc(#loc3)
    %11 = stablehlo.broadcast_in_dim %2, dims = [0] : (tensor<2xi1>) -> tensor<2x1x1xi1> loc(#loc3)
    %12 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f64>) -> tensor<2x4x4xf64> loc(#loc3)
    %13 = stablehlo.broadcast_in_dim %11, dims = [0, 1, 2] : (tensor<2x1x1xi1>) -> tensor<2x4x4xi1> loc(#loc3)
    %14 = stablehlo.select %13, %0#3, %12 : tensor<2x4x4xi1>, tensor<2x4x4xf64> loc(#loc3)
    return %10, %6, %14 : tensor<2x4x4xf64>, tensor<2x4xf64>, tensor<2x4x4xf64> loc(#loc)
  } loc(#loc)
} loc(#loc)
#loc = loc(unknown)
#loc2 = loc("third_party/py/jax/tests/export_back_compat_test.py":690:13)
#loc3 = loc("jit(func)/jit(main)/svd"(#loc2))
""",
    mlir_module_serialized=b"ML\xefR\rStableHLO_v1.7.0\x00\x01!\x05\x01\x05\x11\x01\x03\x0b\x03\x0f\x0f\x13\x17\x1b\x1f#'\x03\xb7q3\x01!\x0f\x07\x0f#\x0b\x0f\x0b\x0b\x0b\x0f\x0b\x13\x0b\x0b\x17\x0b\x03Q\x0b\x0bo\x0f\x0b/\x0b\x0bo\x0f\x13\x0b\x17\x1b\x0b\x1b\x0b\x1b\x0b\x0b\x0b/\x1f#\x0b\x0b\x0b\x0b\x0b\x0b\x0b\x0f\x0f\x17\x1fO/\x0b\x0bO\x01\x05\x0b\x0f\x03/\x1b\x07\x17\x07\x07\x07\x0f\x0f\x07\x13\x1b\x1b\x1f\x13\x13\x13\x13\x13\x17\x13\x17\x13\x13\x02\xc6\x05\x1d\x1b\x1d\x1f\x11\x03\x05\x03\x07\t\x0b\r\x05\x0f\x05\x05\x15\x11\x01\x00\x05\x17\x05\x19\x05\x1b\x1d\x15\x03\x05\x1d\x03\x03\x19O\x05\x1f\x05!\x17\x1f\xca\n\x1b\x05#\x1d%\x1d'\x1f\x1f1\x01\x00\x00\x00\x00\x00\x00\x00\x02\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x1f%\x01\x1d)\x1f+\x11\x00\x00\x00\x00\x00\x00\x00\x00\x05\x03\x05\x01\x1f11\x00\x00\x00\x00\x00\x00\x00\x00\x01\x00\x00\x00\x00\x00\x00\x00\x02\x00\x00\x00\x00\x00\x00\x00\x03\x035\r\x03!##\x1d\x03\x07;?C\r\x05)=!#\x1d+\r\x05)A!#\x1d-\r\x05)E!#\x1d/\x1d1\x1d3\x1f\x11\x11\x00\x00\x00\x00\x00\x00\xf8\x7f\x1f\x13\t\x00\x00\x00\x00\r\x07Q-S-U/\x1d5\x1d7\x1d9\x0b\x03\x1d;\x1d=\x03\x01\x03\x03%\x03\x03c\x15\x03\x01\x01\x01\x03\x0b%g%%i\x1f!!\x01\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x1f#\x11\x00\x00\x00\x00\x00\x00\x00\x00\t\x07\x07\x01\x1f/!\x00\x00\x00\x00\x00\x00\x00\x00\x01\x00\x00\x00\x00\x00\x00\x00\x01\t\x01\x02\x02)\x07\t\x11\x11\r\x01)\x05\t\x11\r\x1d\x0b\x13)\x01\r)\x01\x15\x1b)\x03\t\x15)\x07\t\x05\x05\x07)\x07\t\x11\x11\x07\x11\x03\x05\x07\x05\t\x05)\x03\r\x0f)\x03\t\x0f)\x03\x05\x0f)\x03\x01\x0b)\x03\t\x07)\x05\t\x05\x07)\x03\x05\x0b)\x05\t\x11\x07)\x03\t\x0b)\x03\r\x0b\x04\xc2\x02\x05\x01Q\x03\x07\x01\x07\x04\x9a\x02\x03\x01\x05\tP\x03\x03\x07\x04n\x02\x03-K\x03\x0b\x13\x00\x07B\x03\x05\x03\x11\x07B\x03\x07\x03\x13\x0bG\x01\x17\t\x0b\x05\t\x05\x05\x17\x03\x01\x03F\x01\x0b\x03\x17\x03\x05\rF\x01\r\x03'\x05\x0f\x11\x03F\x01\x0f\x03)\x03\x13\x03F\x01\x0b\x03\t\x03\x03\x03F\x01\x11\x03-\x03\x15\x05\x06\x01\x03\t\x07\x19\t\x17\x03F\x01\x0f\x03\x19\x03\x13\x03F\x01\x0b\x03\x05\x03\x03\x03F\x01\x13\x03\x1b\x03\x1d\x05\x06\x01\x03\x05\x07!\x0b\x1f\x03F\x01\x0f\x03\x19\x03\x13\x03F\x01\x0b\x03\x05\x03\x03\x03F\x01\x13\x03\x1b\x03%\x05\x06\x01\x03\x05\x07)\r'\x0f\x04\x03\x07#\x1b+\x06\x03\x01\x05\x01\x00\xda\x06?'\x03\x17\x1d\x17\x0f\x0b\t\t\t!\x11#i1)\x11\x13%)9\x15\x17\x1f\x11\x19\x15)\x0f\x0b\x11builtin\x00vhlo\x00module\x00broadcast_in_dim_v1\x00select_v1\x00constant_v1\x00func_v1\x00custom_call_v1\x00compare_v1\x00return_v1\x00jax.uses_shape_polymorphism\x00mhlo.num_partitions\x00mhlo.num_replicas\x00jit_func\x00operand\x00mhlo.backend_config\x00jit(func)/jit(main)/svd\x00third_party/py/jax/tests/export_back_compat_test.py\x00mhlo.layout_mode\x00default\x00jax.result_info\x00[0]\x00[1]\x00[2]\x00main\x00public\x00compute_uv\x00full_matrices\x00transposed\x00\x00cusolver_gesvd_ffi\x00\x08E\x15\x05#\x01\x0b379GI\x03K\x03M\x11WY[]/_ae\x03'\x05km\x03+\x03o\x031",
    xla_call_module_version=9,
    nr_devices=1,
)  # End paste

data_2024_10_08["qr"]["c64"] = dict(
    testdata_version=1,
    platform='cuda',
    custom_call_targets=['cusolver_gesvd_ffi'],
    serialized_date=datetime.date(2024, 10, 8),
    inputs=(array([[[ 1.0732381 +3.5808065j ,  3.9696057 +0.20698753j,
         -0.6425436 +0.5669031j ,  2.2608232 +3.3495777j ],
        [-0.14261106+1.9452835j ,  0.7328605 -3.497075j  ,
         -0.2833068 +2.5005085j , -5.3383408 -0.13752732j],
        [ 0.378204  +0.31973448j, -0.82705253-1.2925721j ,
          4.6363106 -0.6026158j , -4.2700663 +0.3752707j ],
        [-0.5445609 -2.3843932j , -1.5469118 -0.22753051j,
         -1.2669541 -5.0028024j ,  0.03133653-3.4463193j ]],

       [[ 2.5795584 +0.6289093j ,  1.27082   +3.8879561j ,
          1.9604253 +1.1865004j , -2.399359  +1.3273407j ],
        [-1.4925842 +5.878235j  , -5.6121607 -3.4182298j ,
         -1.9998045 +0.10950515j,  1.9120374 +3.3194423j ],
        [ 0.40499273-2.4316337j ,  1.6648822 +2.6184802j ,
         -4.0471325 +3.133329j  , -0.76832575-1.7445682j ],
        [-1.895143  +0.5848787j ,  2.6240315 +5.021989j  ,
          1.449456  +3.472618j  , -2.0201976 -3.5186274j ]]],
      dtype=complex64),),
    expected_outputs=(array([[[-0.25113735  -0.21140018j ,  0.018947408 -0.64372665j ,
         -0.070043676 -0.5433938j  ,  0.21103051  +0.3643902j  ],
        [ 0.34577918  -0.55255175j , -0.21863852  +0.16678011j ,
         -0.48150375  -0.24859619j , -0.45313844  +0.022904329j],
        [-0.07727728  -0.37749314j , -0.107260175 +0.56208193j ,
          0.30505398  -0.3713933j  ,  0.53557503  -0.07908653j ],
        [ 0.14142953  +0.5467069j  ,  0.13847762  +0.4037597j  ,
         -0.24584742  -0.33873183j , -0.0011076198+0.56897277j ]],

       [[-0.15231489  +0.2637356j  , -0.333622    -0.3903981j  ,
         -0.3809538   -0.1761738j  , -0.40208697  -0.5528941j  ],
        [ 0.08597728  -0.6945265j  ,  0.21902993  -0.445389j   ,
         -0.33454537  -0.015148819j, -0.28255132  +0.26816013j ],
        [-0.27464584  +0.24423108j ,  0.3478235   +0.3221502j  ,
         -0.740009    -0.16378604j ,  0.15596838  +0.20345287j ],
        [-0.25253323  +0.4675811j  ,  0.37721652  -0.35055056j ,
          0.3376375   -0.15247335j , -0.38508245  +0.40851057j ]]],
      dtype=complex64), array([[ 9.668089 ,  8.574805 ,  4.549492 ,  0.5780793],
       [12.876013 ,  7.7651014,  4.0119534,  2.2829206]], dtype=float32), array([[[-0.38075778  +0.j          ,  0.14002012  +0.060418203j ,
         -0.4637056   +0.22876605j  , -0.489978    -0.5695017j   ],
        [-0.32981405  +0.j          , -0.20355047  +0.51292306j  ,
         -0.34164208  -0.4227407j   , -0.19677992  +0.50254196j  ],
        [-0.3292037   +0.j          ,  0.17828293  +0.62404454j  ,
          0.63654786  +0.14848639j  ,  0.07557414  -0.19353445j  ],
        [ 0.7986684   +0.j          ,  0.056182697 +0.4978434j   ,
         -0.099771045 -0.0043060416j, -0.28370282  -0.14375056j  ]],

       [[-0.3410219   +0.j          ,  0.35656637  -0.6787797j   ,
          0.22528769  -0.27213925j  , -0.21556833  +0.35289994j  ],
        [-0.72291875  +0.j          , -0.12834571  -0.11083051j  ,
         -0.3442186   +0.47835365j  , -0.14619395  -0.28275603j  ],
        [-0.3274419   +0.j          , -0.19452412  +0.057822693j ,
          0.53667945  -0.43909317j  ,  0.17421937  -0.5834539j   ],
        [ 0.50385934  -0.j          , -0.06922947  -0.58084977j  ,
          0.0073776054+0.21678242j  , -0.24243501  -0.546006j    ]]],
      dtype=complex64)),
    mlir_module_text=r"""
#loc1 = loc("operand")
module @jit_func attributes {jax.uses_shape_polymorphism = false, mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} {
  func.func public @main(%arg0: tensor<2x4x4xcomplex<f32>> {mhlo.layout_mode = "default"} loc("operand")) -> (tensor<2x4x4xcomplex<f32>> {jax.result_info = "[0]", mhlo.layout_mode = "default"}, tensor<2x4xf32> {jax.result_info = "[1]", mhlo.layout_mode = "default"}, tensor<2x4x4xcomplex<f32>> {jax.result_info = "[2]", mhlo.layout_mode = "default"}) {
    %cst = stablehlo.constant dense<(0x7FC00000,0x7FC00000)> : tensor<complex<f32>> loc(#loc)
    %cst_0 = stablehlo.constant dense<0x7FC00000> : tensor<f32> loc(#loc)
    %c = stablehlo.constant dense<0> : tensor<i32> loc(#loc)
    %0:5 = stablehlo.custom_call @cusolver_gesvd_ffi(%arg0) {mhlo.backend_config = {compute_uv = true, full_matrices = true, transposed = false}, operand_layouts = [dense<[1, 2, 0]> : tensor<3xindex>], output_operand_aliases = [#stablehlo.output_operand_alias<output_tuple_indices = [0], operand_index = 0, operand_tuple_indices = []>], result_layouts = [dense<[1, 2, 0]> : tensor<3xindex>, dense<[1, 0]> : tensor<2xindex>, dense<[1, 2, 0]> : tensor<3xindex>, dense<[1, 2, 0]> : tensor<3xindex>, dense<0> : tensor<1xindex>]} : (tensor<2x4x4xcomplex<f32>>) -> (tensor<2x4x4xcomplex<f32>>, tensor<2x4xf32>, tensor<2x4x4xcomplex<f32>>, tensor<2x4x4xcomplex<f32>>, tensor<2xi32>) loc(#loc3)
    %1 = stablehlo.broadcast_in_dim %c, dims = [] : (tensor<i32>) -> tensor<2xi32> loc(#loc3)
    %2 = stablehlo.compare  EQ, %0#4, %1,  SIGNED : (tensor<2xi32>, tensor<2xi32>) -> tensor<2xi1> loc(#loc3)
    %3 = stablehlo.broadcast_in_dim %2, dims = [0] : (tensor<2xi1>) -> tensor<2x1xi1> loc(#loc3)
    %4 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<2x4xf32> loc(#loc3)
    %5 = stablehlo.broadcast_in_dim %3, dims = [0, 1] : (tensor<2x1xi1>) -> tensor<2x4xi1> loc(#loc3)
    %6 = stablehlo.select %5, %0#1, %4 : tensor<2x4xi1>, tensor<2x4xf32> loc(#loc3)
    %7 = stablehlo.broadcast_in_dim %2, dims = [0] : (tensor<2xi1>) -> tensor<2x1x1xi1> loc(#loc3)
    %8 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<complex<f32>>) -> tensor<2x4x4xcomplex<f32>> loc(#loc3)
    %9 = stablehlo.broadcast_in_dim %7, dims = [0, 1, 2] : (tensor<2x1x1xi1>) -> tensor<2x4x4xi1> loc(#loc3)
    %10 = stablehlo.select %9, %0#2, %8 : tensor<2x4x4xi1>, tensor<2x4x4xcomplex<f32>> loc(#loc3)
    %11 = stablehlo.broadcast_in_dim %2, dims = [0] : (tensor<2xi1>) -> tensor<2x1x1xi1> loc(#loc3)
    %12 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<complex<f32>>) -> tensor<2x4x4xcomplex<f32>> loc(#loc3)
    %13 = stablehlo.broadcast_in_dim %11, dims = [0, 1, 2] : (tensor<2x1x1xi1>) -> tensor<2x4x4xi1> loc(#loc3)
    %14 = stablehlo.select %13, %0#3, %12 : tensor<2x4x4xi1>, tensor<2x4x4xcomplex<f32>> loc(#loc3)
    return %10, %6, %14 : tensor<2x4x4xcomplex<f32>>, tensor<2x4xf32>, tensor<2x4x4xcomplex<f32>> loc(#loc)
  } loc(#loc)
} loc(#loc)
#loc = loc(unknown)
#loc2 = loc("third_party/py/jax/tests/export_back_compat_test.py":690:13)
#loc3 = loc("jit(func)/jit(main)/svd"(#loc2))
""",
    mlir_module_serialized=b"ML\xefR\rStableHLO_v1.7.0\x00\x01!\x05\x01\x05\x11\x01\x03\x0b\x03\x0f\x0f\x13\x17\x1b\x1f#'\x03\xbds7\x01!\x0f\x07\x0f#\x0b\x0f\x0b\x0b\x0b\x0f\x0b\x13\x0b\x0b\x17\x0b\x03S\x0b\x0bo\x0f\x0b/\x0b\x0bo\x0f\x13\x0b\x17\x1b\x0b\x1b\x0b\x1b\x0b\x0b\x0b/\x1f\x1f#\x0b\x0b\x0b\x0b\x0b\x0b\x0b\x0f\x0f\x17\x1fO/\x0b\x0bO\x01\x05\x0b\x0f\x033\x1b\x07\x17\x07\x07\x07\x0b\x0f\x0f\x0f\x07\x13\x1b\x1b\x1f\x13\x13\x13\x13\x13\x17\x13\x17\x13\x13\x02\xf6\x05\x1d\x1b\x1d\x1f\x11\x03\x05\x03\x07\t\x0b\r\x05\x0f\x05\x05\x15\x11\x01\x00\x05\x17\x05\x19\x05\x1b\x1d\x15\x03\x05\x1d\x03\x03\x19Q\x05\x1f\x05!\x17\x1f\xca\n\x1b\x05#\x1d%\x1d'\x1f#1\x01\x00\x00\x00\x00\x00\x00\x00\x02\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x1f)\x01\x1d)\x1f/\x11\x00\x00\x00\x00\x00\x00\x00\x00\x05\x03\x05\x01\x1f51\x00\x00\x00\x00\x00\x00\x00\x00\x01\x00\x00\x00\x00\x00\x00\x00\x02\x00\x00\x00\x00\x00\x00\x00\x03\x035\r\x03!##!\x03\x07;?C\r\x05)=!#\x1d+\r\x05)A!#\x1d-\r\x05)E!#\x1d/\x1d1\x1d3\x1f\x13\x11\x00\x00\xc0\x7f\x00\x00\xc0\x7f\x1f\x15\t\x00\x00\xc0\x7f\x1f\x17\t\x00\x00\x00\x00\r\x07S-U-W/\x1d5\x1d7\x1d9\x0b\x03\x1d;\x1d=\x03\x01\x03\x03%\x03\x03e\x15\x03\x01\x01\x01\x03\x0b%i%%k\x1f%!\x01\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x1f'\x11\x00\x00\x00\x00\x00\x00\x00\x00\t\x07\x07\x01\x1f3!\x00\x00\x00\x00\x00\x00\x00\x00\x01\x00\x00\x00\x00\x00\x00\x00\x01\t\x01\x02\x02)\x07\t\x11\x11\x11\x01)\x05\t\x11\r\x1d\t\x13\x03\r)\x01\x11)\x01\r)\x01\x19\x1b)\x03\t\x19)\x07\t\x05\x05\x07)\x07\t\x11\x11\x07\x11\x03\x05\x07\x05\t\x05)\x03\r\x0f)\x03\t\x0f)\x03\x05\x0f)\x03\x01\x0b)\x03\t\x07)\x05\t\x05\x07)\x03\x05\x0b)\x05\t\x11\x07)\x03\t\x0b)\x03\r\x0b\x04\xda\x02\x05\x01Q\x03\x07\x01\x07\x04\xb2\x02\x03\x01\x05\tP\x03\x03\x07\x04\x86\x02\x03/O\x03\x0b\x13\x00\x05B\x03\x05\x03\x13\x05B\x03\x07\x03\x15\x05B\x03\t\x03\x17\x0bG\x01\x17\x0b\x0b\x05\t\x05\x05\x1b\x03\x01\x03F\x01\r\x03\x1b\x03\x07\rF\x01\x0f\x03+\x05\x11\x13\x03F\x01\x11\x03-\x03\x15\x03F\x01\r\x03\t\x03\x05\x03F\x01\x13\x031\x03\x17\x07\x06\x01\x03\t\x07\x1b\x0b\x19\x03F\x01\x11\x03\x1d\x03\x15\x03F\x01\r\x03\x05\x03\x03\x03F\x01\x15\x03\x1f\x03\x1f\x07\x06\x01\x03\x05\x07#\r!\x03F\x01\x11\x03\x1d\x03\x15\x03F\x01\r\x03\x05\x03\x03\x03F\x01\x15\x03\x1f\x03'\x07\x06\x01\x03\x05\x07+\x0f)\x0f\x04\x03\x07%\x1d-\x06\x03\x01\x05\x01\x00\xda\x06?'\x03\x17\x1d\x17\x0f\x0b\t\t\t!\x11#i1)\x11\x13%)9\x15\x17\x1f\x11\x15\x19)\x0f\x0b\x11builtin\x00vhlo\x00module\x00broadcast_in_dim_v1\x00constant_v1\x00select_v1\x00func_v1\x00custom_call_v1\x00compare_v1\x00return_v1\x00jax.uses_shape_polymorphism\x00mhlo.num_partitions\x00mhlo.num_replicas\x00jit_func\x00operand\x00mhlo.backend_config\x00jit(func)/jit(main)/svd\x00third_party/py/jax/tests/export_back_compat_test.py\x00mhlo.layout_mode\x00default\x00jax.result_info\x00[0]\x00[1]\x00[2]\x00main\x00public\x00compute_uv\x00full_matrices\x00transposed\x00\x00cusolver_gesvd_ffi\x00\x08I\x17\x05#\x01\x0b379GI\x03K\x03M\x03O\x11Y[]_/acg\x03'\x05mo\x03+\x03q\x031",
    xla_call_module_version=9,
    nr_devices=1,
)  # End paste

data_2024_10_08["qr"]["c128"] = dict(
    testdata_version=1,
    platform='cuda',
    custom_call_targets=['cusolver_gesvd_ffi'],
    serialized_date=datetime.date(2024, 10, 8),
    inputs=(array([[[-4.699732537674587  -0.18164091805215468j,
         -2.6529672987457267 +2.1873545441571416j ,
         -4.583623180305504  +0.05141217967419283j,
         -1.4684446379730842 +0.5956695859134695j ],
        [ 2.217429580673316  -1.6820541069935535j ,
         -1.489637886109648  -1.1907523648513954j ,
         -5.37070728884717   +0.3011497067658051j ,
         -3.5377553884933244 +1.560799473477663j  ],
        [ 0.4865985561509131 +4.547548126143047j  ,
          1.9744285723487844 +1.579347193702052j  ,
          3.662108610237921  -3.8947365367486944j ,
         -0.46900368026456773+3.897268760016375j  ],
        [-3.9057171822032837 +0.894017787659835j  ,
         -2.665956542656175  -5.446062606216615j  ,
          6.586068520522582  +7.82920032979931j   ,
          0.2438426632437082 -2.5324000439269967j ]],

       [[ 2.3593407739528036 +0.1518669531658939j ,
          0.6163481796609258 +2.2151855304705617j ,
          1.1710769743888314 -6.27345033430341j   ,
          0.9738490103384626 +0.5395897278168652j ],
        [-2.4788654273898656 +0.4265527313512031j ,
         -1.1807578044484868 -0.0496832499163036j ,
         -4.4976038167764765 +1.058853052811918j  ,
         -1.1727797045618331 -5.283007446632174j  ],
        [ 2.1607883932036422 +0.15328185326939148j,
          0.33959787374719413-0.44019437888510504j,
          5.554548585416958  -5.5054723821239575j ,
          3.6501512907075853 +2.5205805340930167j ],
        [ 1.3385284474824868 -5.630140770855095j  ,
         -0.27414990799969   -0.46452124262376304j,
          1.611578799750626  +8.022764935423794j  ,
         -2.616414597337455  +0.02175053549931295j]]]),),
    expected_outputs=(array([[[ 0.010550471640845436-0.1718885244537855j  ,
          0.7192461888739357  +0.05423301579908536j ,
         -0.3986393309169209  +0.0992550693722462j  ,
          0.2704504005919772  -0.45626573218528016j ],
        [ 0.1757158215796512  -0.288981687849456j   ,
          0.2705184461915344  +0.2220238094575261j  ,
          0.8080741699984124  +0.3193834688705089j  ,
          0.08837920732925486 -0.018389772805120566j],
        [ 0.18559809104086944 +0.39876558214096686j ,
         -0.035649910397314105-0.49144324300371245j ,
          0.15413388674267017 +0.0406886784503357j  ,
          0.7321684611544159  +0.047628802113382905j],
        [-0.8024624385313128  +0.13619820370204241j ,
         -0.23383020362294066 +0.2445505198523447j  ,
          0.12332909849753147 +0.18873939840686926j ,
          0.26625208768341435 -0.3182762352506706j  ]],

       [[-0.19373811197499574 -0.3678054849282306j  ,
          0.2636906578352804  +0.07973830233400031j ,
         -0.22842504880430503 +0.7133040647598473j  ,
         -0.3539448620401726  +0.25502167015286226j ],
        [-0.24171920617553377 +0.3311681855178457j  ,
         -0.516368216465723   +0.1379819660618946j  ,
         -0.45968933418302504 +0.04091891748552965j ,
          0.20304338982129244 +0.5403786083964173j  ],
        [ 0.02755060979123564 -0.5776868042861196j  ,
          0.27416989428331057 -0.1571567241856725j  ,
         -0.2065632715230957  -0.2247313731964578j  ,
          0.630967059955887   +0.27268947023173257j ],
        [ 0.4031484266707497  +0.40258464155812185j ,
          0.22887811112799977 -0.6972670403334557j  ,
         -0.29285414368652807 +0.2170127687269824j  ,
          0.03733951130463506 +0.050775060769343766j]]]), array([[15.105031148122244 ,  9.10491991264034  ,  5.006211740104105 ,
         3.446376589720919 ],
       [15.343823952995173 ,  7.3753715646873195,  3.7496815109995807,
         0.8625145657311305]]), array([[[ 0.398346102099369   +0.j                  ,
          0.1371865428555439  +0.20963212142757817j ,
         -0.40914166828555465 -0.7712183712683253j  ,
         -0.01748398034296661 +0.12678422058548106j ],
        [-0.4705168176319191  +0.j                  ,
         -0.44062591598236445 +0.5013959991714195j  ,
         -0.27697957785810795 +0.006226163227350533j,
         -0.46230452584693416 +0.2063561296099042j  ],
        [ 0.6106772453975169  +0.j                  ,
         -0.2591680391544425  -0.21982460883707244j ,
          0.0568261232179787  +0.2729248746486013j  ,
         -0.41495999423821034 +0.511540202216849j   ],
        [-0.49699860082446057 +0.j                  ,
          0.2086557264813747  -0.5767641952667593j  ,
          0.0041166738594973  -0.2886775449288538j  ,
         -0.0862160321999865  +0.5348021741590255j  ]],

       [[-0.09961676341576099 +0.j                  ,
         -0.045561706463074156+0.0200549951728976j  ,
          0.6993928144861429  +0.5554242311281644j  ,
         -0.27729760005165593 +0.3362411099877399j  ],
        [ 0.9183964339371264  +0.j                  ,
          0.18513592789573766 +0.048643367601494444j,
         -0.07592192157413628 +0.08808182166509272j ,
          0.022653328707496152+0.3253779069593685j  ],
        [-0.36489390866852983 +0.j                  ,
          0.5302626885445338  -0.13646920230359977j ,
         -0.339383962926636   -0.005000573360891423j,
         -0.01709833277905625 +0.6719756248311359j  ],
        [ 0.11609016321265409 +0.j                  ,
          0.16300036810869795 -0.7965607051693728j  ,
          0.13402108640742152 -0.23592994174105483j ,
         -0.4709038361949722  -0.17340699243933513j ]]])),
    mlir_module_text=r"""
#loc1 = loc("operand")
module @jit_func attributes {jax.uses_shape_polymorphism = false, mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} {
  func.func public @main(%arg0: tensor<2x4x4xcomplex<f64>> {mhlo.layout_mode = "default"} loc("operand")) -> (tensor<2x4x4xcomplex<f64>> {jax.result_info = "[0]", mhlo.layout_mode = "default"}, tensor<2x4xf64> {jax.result_info = "[1]", mhlo.layout_mode = "default"}, tensor<2x4x4xcomplex<f64>> {jax.result_info = "[2]", mhlo.layout_mode = "default"}) {
    %cst = stablehlo.constant dense<(0x7FF8000000000000,0x7FF8000000000000)> : tensor<complex<f64>> loc(#loc)
    %cst_0 = stablehlo.constant dense<0x7FF8000000000000> : tensor<f64> loc(#loc)
    %c = stablehlo.constant dense<0> : tensor<i32> loc(#loc)
    %0:5 = stablehlo.custom_call @cusolver_gesvd_ffi(%arg0) {mhlo.backend_config = {compute_uv = true, full_matrices = true, transposed = false}, operand_layouts = [dense<[1, 2, 0]> : tensor<3xindex>], output_operand_aliases = [#stablehlo.output_operand_alias<output_tuple_indices = [0], operand_index = 0, operand_tuple_indices = []>], result_layouts = [dense<[1, 2, 0]> : tensor<3xindex>, dense<[1, 0]> : tensor<2xindex>, dense<[1, 2, 0]> : tensor<3xindex>, dense<[1, 2, 0]> : tensor<3xindex>, dense<0> : tensor<1xindex>]} : (tensor<2x4x4xcomplex<f64>>) -> (tensor<2x4x4xcomplex<f64>>, tensor<2x4xf64>, tensor<2x4x4xcomplex<f64>>, tensor<2x4x4xcomplex<f64>>, tensor<2xi32>) loc(#loc3)
    %1 = stablehlo.broadcast_in_dim %c, dims = [] : (tensor<i32>) -> tensor<2xi32> loc(#loc3)
    %2 = stablehlo.compare  EQ, %0#4, %1,  SIGNED : (tensor<2xi32>, tensor<2xi32>) -> tensor<2xi1> loc(#loc3)
    %3 = stablehlo.broadcast_in_dim %2, dims = [0] : (tensor<2xi1>) -> tensor<2x1xi1> loc(#loc3)
    %4 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f64>) -> tensor<2x4xf64> loc(#loc3)
    %5 = stablehlo.broadcast_in_dim %3, dims = [0, 1] : (tensor<2x1xi1>) -> tensor<2x4xi1> loc(#loc3)
    %6 = stablehlo.select %5, %0#1, %4 : tensor<2x4xi1>, tensor<2x4xf64> loc(#loc3)
    %7 = stablehlo.broadcast_in_dim %2, dims = [0] : (tensor<2xi1>) -> tensor<2x1x1xi1> loc(#loc3)
    %8 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<complex<f64>>) -> tensor<2x4x4xcomplex<f64>> loc(#loc3)
    %9 = stablehlo.broadcast_in_dim %7, dims = [0, 1, 2] : (tensor<2x1x1xi1>) -> tensor<2x4x4xi1> loc(#loc3)
    %10 = stablehlo.select %9, %0#2, %8 : tensor<2x4x4xi1>, tensor<2x4x4xcomplex<f64>> loc(#loc3)
    %11 = stablehlo.broadcast_in_dim %2, dims = [0] : (tensor<2xi1>) -> tensor<2x1x1xi1> loc(#loc3)
    %12 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<complex<f64>>) -> tensor<2x4x4xcomplex<f64>> loc(#loc3)
    %13 = stablehlo.broadcast_in_dim %11, dims = [0, 1, 2] : (tensor<2x1x1xi1>) -> tensor<2x4x4xi1> loc(#loc3)
    %14 = stablehlo.select %13, %0#3, %12 : tensor<2x4x4xi1>, tensor<2x4x4xcomplex<f64>> loc(#loc3)
    return %10, %6, %14 : tensor<2x4x4xcomplex<f64>>, tensor<2x4xf64>, tensor<2x4x4xcomplex<f64>> loc(#loc)
  } loc(#loc)
} loc(#loc)
#loc = loc(unknown)
#loc2 = loc("third_party/py/jax/tests/export_back_compat_test.py":690:13)
#loc3 = loc("jit(func)/jit(main)/svd"(#loc2))
""",
    mlir_module_serialized=b"ML\xefR\rStableHLO_v1.7.0\x00\x01!\x05\x01\x05\x11\x01\x03\x0b\x03\x0f\x0f\x13\x17\x1b\x1f#'\x03\xbds7\x01!\x0f\x07\x0f#\x0b\x0f\x0b\x0b\x0b\x0f\x0b\x13\x0b\x0b\x17\x0b\x03S\x0b\x0bo\x0f\x0b/\x0b\x0bo\x0f\x13\x0b\x17\x1b\x0b\x1b\x0b\x1b\x0b\x0b\x0bO/\x1f#\x0b\x0b\x0b\x0b\x0b\x0b\x0b\x0f\x0f\x17\x1fO/\x0b\x0bO\x01\x05\x0b\x0f\x033\x1b\x07\x17\x07\x07\x07\x0b\x0f\x0f\x0f\x07\x13\x1b\x1b\x1f\x13\x13\x13\x13\x13\x17\x13\x17\x13\x13\x02&\x06\x1d\x1b\x1d\x1f\x11\x03\x05\x03\x07\t\x0b\r\x05\x0f\x05\x05\x15\x11\x01\x00\x05\x17\x05\x19\x05\x1b\x1d\x15\x03\x05\x1d\x03\x03\x19Q\x05\x1f\x05!\x17\x1f\xca\n\x1b\x05#\x1d%\x1d'\x1f#1\x01\x00\x00\x00\x00\x00\x00\x00\x02\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x1f)\x01\x1d)\x1f/\x11\x00\x00\x00\x00\x00\x00\x00\x00\x05\x03\x05\x01\x1f51\x00\x00\x00\x00\x00\x00\x00\x00\x01\x00\x00\x00\x00\x00\x00\x00\x02\x00\x00\x00\x00\x00\x00\x00\x03\x035\r\x03!##!\x03\x07;?C\r\x05)=!#\x1d+\r\x05)A!#\x1d-\r\x05)E!#\x1d/\x1d1\x1d3\x1f\x13!\x00\x00\x00\x00\x00\x00\xf8\x7f\x00\x00\x00\x00\x00\x00\xf8\x7f\x1f\x15\x11\x00\x00\x00\x00\x00\x00\xf8\x7f\x1f\x17\t\x00\x00\x00\x00\r\x07S-U-W/\x1d5\x1d7\x1d9\x0b\x03\x1d;\x1d=\x03\x01\x03\x03%\x03\x03e\x15\x03\x01\x01\x01\x03\x0b%i%%k\x1f%!\x01\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x1f'\x11\x00\x00\x00\x00\x00\x00\x00\x00\t\x07\x07\x01\x1f3!\x00\x00\x00\x00\x00\x00\x00\x00\x01\x00\x00\x00\x00\x00\x00\x00\x01\t\x01\x02\x02)\x07\t\x11\x11\x11\x01)\x05\t\x11\r\x1d\x0b\x13\x03\r)\x01\x11)\x01\r)\x01\x19\x1b)\x03\t\x19)\x07\t\x05\x05\x07)\x07\t\x11\x11\x07\x11\x03\x05\x07\x05\t\x05)\x03\r\x0f)\x03\t\x0f)\x03\x05\x0f)\x03\x01\x0b)\x03\t\x07)\x05\t\x05\x07)\x03\x05\x0b)\x05\t\x11\x07)\x03\t\x0b)\x03\r\x0b\x04\xda\x02\x05\x01Q\x03\x07\x01\x07\x04\xb2\x02\x03\x01\x05\tP\x03\x03\x07\x04\x86\x02\x03/O\x03\x0b\x13\x00\x05B\x03\x05\x03\x13\x05B\x03\x07\x03\x15\x05B\x03\t\x03\x17\x0bG\x01\x17\x0b\x0b\x05\t\x05\x05\x1b\x03\x01\x03F\x01\r\x03\x1b\x03\x07\rF\x01\x0f\x03+\x05\x11\x13\x03F\x01\x11\x03-\x03\x15\x03F\x01\r\x03\t\x03\x05\x03F\x01\x13\x031\x03\x17\x07\x06\x01\x03\t\x07\x1b\x0b\x19\x03F\x01\x11\x03\x1d\x03\x15\x03F\x01\r\x03\x05\x03\x03\x03F\x01\x15\x03\x1f\x03\x1f\x07\x06\x01\x03\x05\x07#\r!\x03F\x01\x11\x03\x1d\x03\x15\x03F\x01\r\x03\x05\x03\x03\x03F\x01\x15\x03\x1f\x03'\x07\x06\x01\x03\x05\x07+\x0f)\x0f\x04\x03\x07%\x1d-\x06\x03\x01\x05\x01\x00\xda\x06?'\x03\x17\x1d\x17\x0f\x0b\t\t\t!\x11#i1)\x11\x13%)9\x15\x17\x1f\x11\x15\x19)\x0f\x0b\x11builtin\x00vhlo\x00module\x00broadcast_in_dim_v1\x00constant_v1\x00select_v1\x00func_v1\x00custom_call_v1\x00compare_v1\x00return_v1\x00jax.uses_shape_polymorphism\x00mhlo.num_partitions\x00mhlo.num_replicas\x00jit_func\x00operand\x00mhlo.backend_config\x00jit(func)/jit(main)/svd\x00third_party/py/jax/tests/export_back_compat_test.py\x00mhlo.layout_mode\x00default\x00jax.result_info\x00[0]\x00[1]\x00[2]\x00main\x00public\x00compute_uv\x00full_matrices\x00transposed\x00\x00cusolver_gesvd_ffi\x00\x08I\x17\x05#\x01\x0b379GI\x03K\x03M\x03O\x11Y[]_/acg\x03'\x05mo\x03+\x03q\x031",
    xla_call_module_version=9,
    nr_devices=1,
)  # End paste

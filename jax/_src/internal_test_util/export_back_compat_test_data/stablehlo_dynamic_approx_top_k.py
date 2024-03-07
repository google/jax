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
from numpy import array, float32, int32

# Pasted from the test output (see export_back_compat_test_util.py module docstring)
data_2024_05_30 = dict(
    testdata_version=1,
    platform='cpu',
    custom_call_targets=['shape_assertion', 'stablehlo.dynamic_approx_top_k'],
    serialized_date=datetime.date(2024, 5, 30),
    inputs=(array([ 0.,  1.,  2.,  3.,  4.,  5.,  6.,  7.,  8.,  9., 10., 11., 12.,
       13., 14., 15., 16., 17., 18., 19., 20., 21., 22., 23.],
      dtype=float32),),
    expected_outputs=(array([23., 22., 21., 20., 19., 18., 17., 16., 15., 14., 13., 12., 11.,
       10.,  9.,  8.,  7.,  6.,  5.,  4.], dtype=float32), array([23, 22, 21, 20, 19, 18, 17, 16, 15, 14, 13, 12, 11, 10,  9,  8,  7,
        6,  5,  4], dtype=int32)),
    mlir_module_text=r"""
#loc = loc(unknown)
#loc2 = loc("third_party/py/jax/tests/export_back_compat_test.py":718:13)
#loc3 = loc("a")
#loc9 = loc("jit(func)/jit(main)/approx_top_k[k=b reduction_dimension=-1 recall_target=0.95 is_max_k=True reduction_input_size_override=-1 aggregate_to_topk=True]"(#loc2))
module @jit_func attributes {jax.uses_shape_polymorphism = true, mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} {
  func.func public @main(%arg0: tensor<?xf32> {mhlo.layout_mode = "default"} loc(unknown)) -> (tensor<?xf32> {jax.result_info = "[0]", mhlo.layout_mode = "default"}, tensor<?xi32> {jax.result_info = "[1]", mhlo.layout_mode = "default"}) {
    %0 = stablehlo.get_dimension_size %arg0, dim = 0 : (tensor<?xf32>) -> tensor<i32> loc(#loc4)
    %1 = stablehlo.convert %0 : (tensor<i32>) -> tensor<i64> loc(#loc4)
    %2 = stablehlo.convert %1 : tensor<i64> loc(#loc5)
    %c = stablehlo.constant dense<-4> : tensor<i64> loc(#loc)
    %3 = stablehlo.add %2, %c : tensor<i64> loc(#loc6)
    %c_0 = stablehlo.constant dense<1> : tensor<i64> loc(#loc)
    %4 = stablehlo.compare  GE, %3, %c_0,  SIGNED : (tensor<i64>, tensor<i64>) -> tensor<i1> loc(#loc7)
    stablehlo.custom_call @shape_assertion(%4, %3, %1) {api_version = 2 : i32, error_message = "Input shapes do not match the polymorphic shapes specification. Expected value >= 1 for dimension variable 'b'. Using the following polymorphic shapes specifications: args[0].shape = (b + 4,). Obtained dimension variables: 'b' = {0} from specification 'b + 4' for dimension args[0].shape[0] (= {1}), . Please see https://github.com/googlexjax/blob/main/jax/experimental/jax2tf/README.md#shape-assertion-errors for more details.", has_side_effect = true} : (tensor<i1>, tensor<i64>, tensor<i64>) -> () loc(#loc8)
    %5:2 = call @_wrapped_jax_export_main(%3, %arg0) : (tensor<i64>, tensor<?xf32>) -> (tensor<?xf32>, tensor<?xi32>) loc(#loc)
    return %5#0, %5#1 : tensor<?xf32>, tensor<?xi32> loc(#loc)
  } loc(#loc)
  func.func @top_k_gt_f32_comparator(%arg0: tensor<f32> loc("jit(func)/jit(main)/approx_top_k[k=b reduction_dimension=-1 recall_target=0.95 is_max_k=True reduction_input_size_override=-1 aggregate_to_topk=True]"(#loc2)), %arg1: tensor<f32> loc("jit(func)/jit(main)/approx_top_k[k=b reduction_dimension=-1 recall_target=0.95 is_max_k=True reduction_input_size_override=-1 aggregate_to_topk=True]"(#loc2)), %arg2: tensor<i32> loc("jit(func)/jit(main)/approx_top_k[k=b reduction_dimension=-1 recall_target=0.95 is_max_k=True reduction_input_size_override=-1 aggregate_to_topk=True]"(#loc2)), %arg3: tensor<i32> loc("jit(func)/jit(main)/approx_top_k[k=b reduction_dimension=-1 recall_target=0.95 is_max_k=True reduction_input_size_override=-1 aggregate_to_topk=True]"(#loc2))) -> tensor<i1> {
    %0 = stablehlo.compare  GT, %arg0, %arg1 : (tensor<f32>, tensor<f32>) -> tensor<i1> loc(#loc9)
    return %0 : tensor<i1> loc(#loc9)
  } loc(#loc9)
  func.func private @_wrapped_jax_export_main(%arg0: tensor<i64> {jax.global_constant = "b", mhlo.layout_mode = "default"} loc(unknown), %arg1: tensor<?xf32> {mhlo.layout_mode = "default"} loc("a")) -> (tensor<?xf32> {jax.result_info = "[0]", mhlo.layout_mode = "default"}, tensor<?xi32> {jax.result_info = "[1]", mhlo.layout_mode = "default"}) {
    %0 = stablehlo.convert %arg0 : tensor<i64> loc(#loc10)
    %c = stablehlo.constant dense<4> : tensor<i64> loc(#loc9)
    %1 = stablehlo.add %0, %c : tensor<i64> loc(#loc11)
    %2 = stablehlo.convert %1 : (tensor<i64>) -> tensor<i32> loc(#loc9)
    %3 = stablehlo.reshape %2 : (tensor<i32>) -> tensor<1xi32> loc(#loc9)
    %4 = stablehlo.dynamic_iota %3, dim = 0 : (tensor<1xi32>) -> tensor<?xi32> loc(#loc9)
    %c_0 = stablehlo.constant dense<-1> : tensor<i32> loc(#loc9)
    %cst = stablehlo.constant dense<0xFF800000> : tensor<f32> loc(#loc9)
    %5 = stablehlo.convert %arg0 : (tensor<i64>) -> tensor<i32> loc(#loc9)
    %6 = stablehlo.reshape %5 : (tensor<i32>) -> tensor<1xi32> loc(#loc9)
    %7 = stablehlo.convert %arg0 : (tensor<i64>) -> tensor<i32> loc(#loc9)
    %8 = stablehlo.reshape %7 : (tensor<i32>) -> tensor<1xi32> loc(#loc9)
    %9 = stablehlo.convert %arg0 : (tensor<i64>) -> tensor<i32> loc(#loc9)
    %10:2 = stablehlo.custom_call @stablehlo.dynamic_approx_top_k(%arg1, %4, %cst, %c_0, %9, %6, %8) {called_computations = [@top_k_gt_f32_comparator], indices_of_shape_operands = dense<[5, 6]> : tensor<2xi64>, mhlo.backend_config = {aggregate_to_topk = true, is_fallback = true, recall_target = 0.949999988 : f32, reduction_dim = 0 : i64, reduction_input_size_override = -1 : i64}} : (tensor<?xf32>, tensor<?xi32>, tensor<f32>, tensor<i32>, tensor<i32>, tensor<1xi32>, tensor<1xi32>) -> (tensor<?xf32>, tensor<?xi32>) loc(#loc9)
    return %10#0, %10#1 : tensor<?xf32>, tensor<?xi32> loc(#loc)
  } loc(#loc)
} loc(#loc)
#loc1 = loc("third_party/py/jax/tests/export_back_compat_test.py":738:4)
#loc4 = loc("/dimension_size[dimension=0]"(#loc1))
#loc5 = loc("/convert_element_type[new_dtype=int64 weak_type=False]"(#loc1))
#loc6 = loc("/add"(#loc1))
#loc7 = loc("/ge"(#loc1))
#loc8 = loc("/shape_assertion[error_message=Input shapes do not match the polymorphic shapes specification. Expected value >= 1 for dimension variable 'b'. Using the following polymorphic shapes specifications: args[0].shape = (b + 4,). Obtained dimension variables: 'b' = {0} from specification 'b + 4' for dimension args[0].shape[0] (= {1}), . Please see https://github.com/googlexjax/blob/main/jax/experimental/jax2tf/README.md#shape-assertion-errors for more details.]"(#loc1))
#loc10 = loc("jit(func)/jit(main)/convert_element_type[new_dtype=int64 weak_type=False]"(#loc2))
#loc11 = loc("jit(func)/jit(main)/add"(#loc2))
""",
    mlir_module_serialized=b'ML\xefR\x01StableHLO_v0.9.0\x00\x01\'\x05\x01\x03\x01\x03\x05\x03\x17\x07\t\x0b\r\x0f\x11\x13\x15\x17\x19\x1b\x03B\x02\xeb#\x01\x85\x0f\x07\x0b\x17\x0b\x0b\x0b\x0b\x0b\x17\x0f\x0b\x0b\x0b\x0b\x0b\x0b\x0b\x0b\x0b\x0b\x0b\x0f+\x0b\x0f\x0b\x0b\x0b33\x0b3\x0f\x0b\x0f\x0b\x13\x0f\x0b\x13\x0b\x13\x13[\x0b\x0b\x1b\x13\x0b\x0b\x0f\x0b\x13\x0f\x0b\x13\x1b\x0f\x0bS\x0b\x0f\x0b\x13\x0b\x03g\x0b\x0b\x0b\x0b\x0f\x0b\x13\x13\x0b\x0b\x0b\x0f\x0b\x1b\x0b\x1b\x0b\x0b\x0b\x0b\x13\x1b\x0b\x0b\x0b\x0b/\x1f\x1f\x0b\x0b\x0f\x0bO3\x0b\x0b\x0b\x1f\x0b\x0b\x0f\x0b\x0b//\x0b\x0b\x0b\x0b\x0b\x01\x05\x0b\x0f\x03\x1f\x0f\x0f3\x0f3\x07\x07\x07\x0f\x13\x1b#\x07\x1f\x13\x02\x0e\x08\x1d?\x13\x1f\x05\x1d\x17\x17\x8a\x0b\t\x05\x1f\x05!\x05#\x05%\x05\'\x17\x17:\x0b\x1b\x11\x03\x05\x05)\x05+\x05-\x05/\x051\x053\x055\x057\x059\x05;\x05=\x1de\x07\x03\t135\x157\x15\t9\x05?\x11\x01\x01\x05A\x05C\x05E\x03\x0b\x0b\x9b\r\x9d\x0f\x93\t\xa7\x11\xa9\x03\x0b\x0b\x85\r\xab\x0f\x85\t\x97\x11\x8b\x05G\x03\x0b\x0b\xad\r\xb5\x0f\x93\t\x99\x11\xb7\x1dE\x03\x05I\x1dI\x13\x05K\x03\x03\x05\xb9\x1dO\x13\x05M\x03\x03S\x8d\x05O\x03\x03\x05\xbb\x03\x03\x05\xbd\x03\x15\x19\xbf\x1b\x8b\x1d\xc1\x1f\xc3!\xc5[\xc7]\xc9#\x85%\x85\'\x85\x05Q\x05S\x03\x05)\xd9+\xdb\x03\x03c\x8d\x05U\x05W\x1di\x07\x05Y\x03\x03\x05\xdd\x1do\x07\x05[\x03\x03\x05\xdf\x03\x05)\xe1+\xe3\x1dw\x07\x05]\x03\x13\x19\xe5\x1b\x8b\x1d\xe7\x1f\x85{\xe9!\x8f#\x85%\x85\'\x85\x05_\x1d\x7f\x07\x05a\x03\x03\x83\x99\x05c\x03\x01\x1de\x1dg\x1di\x13\x0f\x01\x05\x03\r\x03\x87\x89\x03\x05\x9f\xa3\x1dk\x1dm\x1do\x03\x03\x91#\x19\r\x05\x95\xa1\x87\x89\x1dq\r\x05\x95\xa5\x87\x89\x1ds\x1du\x1dw#\x1b\x03\x05\xaf\x91\r\x05\xb1\xb3\x87\x89\x1dy\x1d{#\x1f\x1d}\x1f\x05\x11\x04\x00\x00\x00\x00\x00\x00\x00\x1f\x07\t\xff\xff\xff\xff\x1f\x0b\t\x00\x00\x80\xff\x0b\x03\x1d\x7f\x03\x03\x97\x05\x01\x1f!!\x05\x00\x00\x00\x00\x00\x00\x00\x06\x00\x00\x00\x00\x00\x00\x00\r\x0b\xcb\x8f\xcd\x8f\xcf\xd1\xd3\x8d\xd5\xd7\x1d\x81\x1d\x83\x1d\x85\x11\x11\xd0\xcc\xcc\xdc\x0f\x1d\x87\x1d\x89\x13\x0f\x03\t\x01\x07\x07\x1f\x05\x11\xfc\xff\xff\xff\xff\xff\xff\xff\x1f\x05\x11\x01\x00\x00\x00\x00\x00\x00\x00\t\x07\x07\x05\x0b\x05\x1d\x8b\x1d\x8d\x01\t\x01\x02\x02)\x01\x0f)\x01\x13)\x03\x00\xff\xff\xff\xff\xff\xff\xff\xff\x11)\x01\x11)\x03\x00\xff\xff\xff\xff\xff\xff\xff\xff\x13\x1d\t\x1b)\x01\x1d)\x03\x05\x13\x11\x03\t\x05\t\r\x11\t\x0b\x0b\x07\x07\x03\x15\x01\x11\x05\x05\t\x05\t\r)\x03\t\x0f\x04\xea\x03\x05\x01\x11\x03/\x07\x03\x01\r\x07\x11\x03;\x07\x03\x15+\x03\t\x03\x15\x07-a\x03\x07\x03\x01\x03\x06-\x03\x05\x03\x03\x03\x06g\x03\x05\x03\x05\x05\x03\x03k\x03\x05\r\x06m\x03\x05\x05\x07\t\x05\x03\x03q\x03\x05\x11\x07us\x03\x15\x05\x0b\r\x0f\x05}y\x07\x0f\x0b\x05\x17\x07\x03\x81\x05\t\r\x05\x0b\x01\x0b\x04\x03\x05\x11\x13\x07\x11\x01=\x07\x03\x0b\x0b\t\x0b\x01\x0b\x01\x07\x01\x07\x01\x11\x07\x01_\x03\x15\x05\x01\x03\x0b\x04\x01\x03\t\x07\x11\x03A\x07\x03#?\x05\x05\x03\tC\x03\x06G\x03\x05\x03\x01\x05\x03\x01K\x03\x05\r\x06M\x03\x05\x05\x05\x07\x03\x06\x01\x03\x07\x03\t\t\x06\x01\x03\x17\x03\x0b\x13\x07\x01Q\x03\r\x03\r\x05\x03\x01U\x03\x07\x05\x03\x01W\x03\x0b\x03\x06\x01\x03\x07\x03\x01\t\x06\x01\x03\x17\x03\x15\x03\x06\x01\x03\x07\x03\x01\t\x06\x01\x03\x17\x03\x19\x03\x06\x01\x03\x07\x03\x01\x0f\x07\x01Y\x05\t\r\x0f\x03\x0f\x13\x11\x1d\x17\x1b\x0b\x04\x03\x05\x1f!\x06\x03\x01\x05\x01\x00""\x8f\xb2\x06!=\x1d\x1d\x19%?\x11\x05)\x0f\x0b\t\t31!\x03\x11#\x0f2\x07\x1d\t\x0bo;\x15)5\x1f1\x95\x05Z\x02\x13%)9+\x1b\x1f/!!)#\x1f\x19i\x1f\x15\x1d\x15\x13\r\x11-!\x17\x1f\x0f\x15\x17\x11\x19\x17\x0f\x0b\x11builtin\x00vhlo\x00module\x00convert_v1\x00constant_v1\x00func_v1\x00reshape_v1\x00return_v1\x00add_v1\x00custom_call_v1\x00compare_v1\x00dynamic_iota_v1\x00get_dimension_size_v1\x00call_v1\x00value\x00sym_name\x00arg_attrs\x00function_type\x00res_attrs\x00sym_visibility\x00third_party/py/jax/tests/export_back_compat_test.py\x00api_version\x00backend_config\x00call_target_name\x00called_computations\x00has_side_effect\x00operand_layouts\x00output_operand_aliases\x00result_layouts\x00compare_type\x00comparison_direction\x00jax.uses_shape_polymorphism\x00mhlo.num_partitions\x00mhlo.num_replicas\x00jit_func\x00jit(func)/jit(main)/approx_top_k[k=b reduction_dimension=-1 recall_target=0.95 is_max_k=True reduction_input_size_override=-1 aggregate_to_topk=True]\x00a\x00jit(func)/jit(main)/convert_element_type[new_dtype=int64 weak_type=False]\x00jit(func)/jit(main)/add\x00iota_dimension\x00indices_of_shape_operands\x00mhlo.backend_config\x00dimension\x00/dimension_size[dimension=0]\x00/convert_element_type[new_dtype=int64 weak_type=False]\x00/add\x00/ge\x00error_message\x00/shape_assertion[error_message=Input shapes do not match the polymorphic shapes specification. Expected value >= 1 for dimension variable \'b\'. Using the following polymorphic shapes specifications: args[0].shape = (b + 4,). Obtained dimension variables: \'b\' = {0} from specification \'b + 4\' for dimension args[0].shape[0] (= {1}), . Please see https://github.com/googlexjax/blob/main/jax/experimental/jax2tf/README.md#shape-assertion-errors for more details.]\x00callee\x00mhlo.layout_mode\x00default\x00\x00jax.result_info\x00top_k_gt_f32_comparator\x00_wrapped_jax_export_main\x00[0]\x00[1]\x00main\x00public\x00jax.global_constant\x00b\x00private\x00stablehlo.dynamic_approx_top_k\x00aggregate_to_topk\x00is_fallback\x00recall_target\x00reduction_dim\x00reduction_input_size_override\x00shape_assertion\x00Input shapes do not match the polymorphic shapes specification. Expected value >= 1 for dimension variable \'b\'. Using the following polymorphic shapes specifications: args[0].shape = (b + 4,). Obtained dimension variables: \'b\' = {0} from specification \'b + 4\' for dimension args[0].shape[0] (= {1}), . Please see https://github.com/googlexjax/blob/main/jax/experimental/jax2tf/README.md#shape-assertion-errors for more details.\x00',
    xla_call_module_version=9,
    nr_devices=1,
)  # End paste

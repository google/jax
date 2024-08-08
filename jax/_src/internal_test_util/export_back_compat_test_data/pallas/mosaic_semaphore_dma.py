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
from numpy import array, float32

# Pasted from the test output (see export_back_compat_test_util.py module docstring)
semaphore_and_dma_2024_04_22 = dict(
    testdata_version=1,
    platform='tpu',
    custom_call_targets=['tpu_custom_call'],
    serialized_date=datetime.date(2024, 4, 22),
    inputs=(),
    expected_outputs=(array(1., dtype=float32),),
    mlir_module_text=r"""
#loc2 = loc("third_party/py/jax_triton/googlexpallas_tpu/back_compat_test.py":60:4)
#loc3 = loc("third_party/py/absl/testing/absltest.py":2718:19)
#loc4 = loc("third_party/py/absl/testing/absltest.py":2754:35)
#loc5 = loc("third_party/py/absl/testing/absltest.py":2298:6)
#loc6 = loc("third_party/py/absl/app.py":395:13)
#loc7 = loc("third_party/py/absl/app.py":473:6)
#loc8 = loc("third_party/py/absl/testing/absltest.py":2300:4)
#loc9 = loc("third_party/py/absl/testing/absltest.py":2182:2)
#loc10 = loc("third_party/py/jax_triton/googlexpallas_tpu/back_compat_test.py":64:2)
#loc11 = loc("third_party/py/jax_triton/googlexpallas_tpu/back_compat_test.py":57:10)
#loc14 = loc("PallasKernelTest.test_semaphore_and_dma_22_04_2024"(#loc2))
#loc15 = loc("_run_and_get_tests_result"(#loc3))
#loc16 = loc("run_tests"(#loc4))
#loc17 = loc("_run_in_app.<locals>.main_function"(#loc5))
#loc18 = loc("_run_main"(#loc6))
#loc19 = loc("run"(#loc7))
#loc20 = loc("_run_in_app"(#loc8))
#loc21 = loc("main"(#loc9))
#loc22 = loc("<module>"(#loc10))
#loc23 = loc("PallasKernelTest.test_semaphore_and_dma_22_04_2024.<locals>.func"(#loc11))
#loc25 = loc(callsite(#loc21 at #loc22))
#loc26 = loc(callsite(#loc20 at #loc25))
#loc27 = loc(callsite(#loc19 at #loc26))
#loc28 = loc(callsite(#loc18 at #loc27))
#loc29 = loc(callsite(#loc17 at #loc28))
#loc30 = loc(callsite(#loc16 at #loc29))
#loc31 = loc(callsite(#loc15 at #loc30))
#loc32 = loc(callsite(#loc14 at #loc31))
#loc34 = loc(callsite(#loc23 at #loc32))
#loc38 = loc("jit(func)/jit(main)/pjit[in_shardings=(UnspecifiedValue,) out_shardings=(UnspecifiedValue,) in_layouts=(None,) out_layouts=(None,) resource_env=None donated_invars=(False,) name=wrapped keep_unused=False inline=False]"(#loc34))
#loc42 = loc("jit(func)/jit(main)/jit(wrapped)/pjit[in_shardings=(UnspecifiedValue,) out_shardings=(UnspecifiedValue,) in_layouts=(None,) out_layouts=(None,) resource_env=None donated_invars=(False,) name=apply_kernel keep_unused=False inline=False]"(#loc34))
module @jit_func attributes {jax.uses_shape_polymorphism = false, mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} {
  func.func public @main() -> (tensor<f32> {jax.result_info = "", mhlo.layout_mode = "default"}) {
    %0 = stablehlo.iota dim = 0 : tensor<16384xf32> loc(#loc36)
    %1 = stablehlo.reshape %0 : (tensor<16384xf32>) -> tensor<128x128xf32> loc(#loc37)
    %2 = call @wrapped(%1) : (tensor<128x128xf32>) -> tensor<128x128xf32> loc(#loc38)
    %3 = stablehlo.compare  EQ, %1, %2,  FLOAT : (tensor<128x128xf32>, tensor<128x128xf32>) -> tensor<128x128xi1> loc(#loc39)
    %c = stablehlo.constant dense<true> : tensor<i1> loc(#loc40)
    %4 = stablehlo.reduce(%3 init: %c) applies stablehlo.and across dimensions = [0, 1] : (tensor<128x128xi1>, tensor<i1>) -> tensor<i1> loc(#loc40)
    %5 = stablehlo.convert %4 : (tensor<i1>) -> tensor<f32> loc(#loc41)
    return %5 : tensor<f32> loc(#loc)
  } loc(#loc)
  func.func private @wrapped(%arg0: tensor<128x128xf32> {mhlo.layout_mode = "default"} loc("jit(func)/jit(main)/pjit[in_shardings=(UnspecifiedValue,) out_shardings=(UnspecifiedValue,) in_layouts=(None,) out_layouts=(None,) resource_env=None donated_invars=(False,) name=wrapped keep_unused=False inline=False]"(#loc34))) -> (tensor<128x128xf32> {mhlo.layout_mode = "default"}) {
    %0 = call @apply_kernel(%arg0) : (tensor<128x128xf32>) -> tensor<128x128xf32> loc(#loc42)
    return %0 : tensor<128x128xf32> loc(#loc38)
  } loc(#loc38)
  func.func private @apply_kernel(%arg0: tensor<128x128xf32> {mhlo.layout_mode = "default"} loc("jit(func)/jit(main)/jit(wrapped)/pjit[in_shardings=(UnspecifiedValue,) out_shardings=(UnspecifiedValue,) in_layouts=(None,) out_layouts=(None,) resource_env=None donated_invars=(False,) name=apply_kernel keep_unused=False inline=False]"(#loc34))) -> (tensor<128x128xf32> {mhlo.layout_mode = "default"}) {
    %0 = stablehlo.custom_call @tpu_custom_call(%arg0) {backend_config = "{\22custom_call_config\22: {\22body\22: \22TUzvUgFNTElSZ29vZ2xlMy10cnVuawABJwcBAwUBAwcDFQkLDQ8RExUXGRsD27UTAbELBwsPCw8PCw8PDw8PDw8LDw9VDxMPDxMLDzMLCwsLhQsLCwsPCxMPCxMPCxMPCxcPCxcPCxcPCxcPCxcPCxcPFw8LDxMPDw8PDw8PDwsLDwsPDxMLDw8TBQWFYQEPJw8PFwcXFwUFTT0CzgYFHR8FHx1HSQUhFRGLEQUBBSMdS00dUVMdV1kdXV8dY2UdaWsdb3EFJR11dx17fWFmZmluZV9tYXA8KCkgLT4gKCk+ABWHCwMDnZ8doaMdqasDAzEzBScRBQUDCzc5Oz1BDUMNRQ8FKQEBBSsNB2FmZmluZV9tYXA8KGQwLCBkMSkgLT4gKGQwLCBkMSk+AAUtBS8FMQUzFRFPBTUXAWsRFRNVBTcXAXMVFRVbBTkXAXkJFRdhBTsXBXoqJxUZZwU9FwUKK0cVG20FPxcF6iMNFR1zBUEXHy4GGxUheQVDFx9mBw0VI38FRRcF8iMJHQ+BFwUaIgUdhScFRx0JiRcBZRUVE40VFY8VF5EVGZMVG5UVHZcVISMdmycFSQVLEQMFBU0VpQsdCacXAWcVBU8VrQsdCa8XAWkVI3RwdS5tZW1vcnlfc3BhY2U8c2VtYXBob3JlX21lbT4AI3RwdS5tZW1vcnlfc3BhY2U8dm1lbT4AF7MFAgQCBAk/AQICAQIEBQUBAQELF7EBDyUXsQERJSF0cHUuZG1hX3NlbWFwaG9yZQAhdHB1LnNlbWFwaG9yZQAEpQUBEQMvBwMBBQcRAzUHAwULBQEDAQMJEAcFAwklAwIHAwsDAgcDDQ0EgwcBAwUPBJkFBQMFAyspAwMRBCsFBwkFAy0pAwMTBC0FBwsVAAcLAAMGAwEFAQA+ElFjtQ2XyxkJFUcVNWeDqxkTIyEdKS03C8dRgRUbHxshGRcVHx0PCR0RYnVpbHRpbgBzdGFibGVfbW9zYWljAHRwdQBtb2R1bGUAdHB1LnNlbV9hbGxvYwBhcml0aC5jb25zdGFudABmdW5jLmZ1bmMAdHB1LnJlZ2lvbgBmdW5jLnJldHVybgB0cHUuZW5xdWV1ZV9kbWEAdHB1LndhaXRfZG1hAHRwdS5zZW1fc2lnbmFsAHRwdS5zZW1fd2FpdAB0cHUueWllbGQAdGhpcmRfcGFydHkvcHkvamF4X3RyaXRvbi9nb29nbGUvcGFsbGFzX3RwdS9iYWNrX2NvbXBhdF90ZXN0LnB5AHRoaXJkX3BhcnR5L3B5L2Fic2wvdGVzdGluZy9hYnNsdGVzdC5weQBQYWxsYXNLZXJuZWxUZXN0LnRlc3Rfc2VtYXBob3JlX2FuZF9kbWFfMjJfMDRfMjAyNC48bG9jYWxzPi5mdW5jLjxsb2NhbHM+LmRtYV9rZXJuZWwuPGxvY2Fscz4uYm9keQBtYWluAHRoaXJkX3BhcnR5L3B5L2Fic2wvYXBwLnB5AHN0YWJsZV9tb3NhaWMudmVyc2lvbgBkaW1lbnNpb25fc2VtYW50aWNzAGZ1bmN0aW9uX3R5cGUAc2NhbGFyX3ByZWZldGNoAHNjcmF0Y2hfb3BlcmFuZHMAc3ltX25hbWUAL3J1bl9zY29wZWQAUGFsbGFzS2VybmVsVGVzdC50ZXN0X3NlbWFwaG9yZV9hbmRfZG1hXzIyXzA0XzIwMjQuPGxvY2Fscz4uZnVuYy48bG9jYWxzPi5kbWFfa2VybmVsAFBhbGxhc0tlcm5lbFRlc3QudGVzdF9zZW1hcGhvcmVfYW5kX2RtYV8yMl8wNF8yMDI0Ljxsb2NhbHM+LmZ1bmMAUGFsbGFzS2VybmVsVGVzdC50ZXN0X3NlbWFwaG9yZV9hbmRfZG1hXzIyXzA0XzIwMjQAX3J1bl9hbmRfZ2V0X3Rlc3RzX3Jlc3VsdABydW5fdGVzdHMAX3J1bl9pbl9hcHAuPGxvY2Fscz4ubWFpbl9mdW5jdGlvbgBfcnVuX21haW4AcnVuAF9ydW5faW5fYXBwAC9kbWFfc3RhcnRbdHJlZT1QeVRyZWVEZWYoKCosICgpLCAqLCAoKSwgKiwgKCksIE5vbmUsIE5vbmUsIE5vbmUpKSBkZXZpY2VfaWRfdHlwZT1EZXZpY2VJZFR5cGUuTUVTSF0AL2RtYV93YWl0W3RyZWU9UHlUcmVlRGVmKCgqLCAoKSwgKiwgKCkpKSBkZXZpY2VfaWRfdHlwZT1EZXZpY2VJZFR5cGUuTUVTSF0AdmFsdWUAL3NlbWFwaG9yZV9zaWduYWxbYXJnc190cmVlPVB5VHJlZURlZihbKiwgKCksICosIE5vbmVdKSBkZXZpY2VfaWRfdHlwZT1EZXZpY2VJZFR5cGUuTUVTSF0AL3NlbWFwaG9yZV93YWl0W2FyZ3NfdHJlZT1QeVRyZWVEZWYoWyosICgpLCAqXSldAA==\22, \22serialization_format\22: 1, \22needs_layout_passes\22: true}, \22implicit_sharding\22: {\22type\22: \22MANUAL\22}}", kernel_name = "dma_kernel", operand_layouts = [dense<[1, 0]> : tensor<2xindex>], result_layouts = [dense<[1, 0]> : tensor<2xindex>]} : (tensor<128x128xf32>) -> tensor<128x128xf32> loc(#loc43)
    return %0 : tensor<128x128xf32> loc(#loc42)
  } loc(#loc42)
} loc(#loc)
#loc = loc(unknown)
#loc1 = loc("third_party/py/jax_triton/googlexpallas_tpu/back_compat_test.py":56:10)
#loc12 = loc("third_party/py/jax_triton/googlexpallas_tpu/back_compat_test.py":58:13)
#loc13 = loc("PallasKernelTest.test_semaphore_and_dma_22_04_2024.<locals>.func"(#loc1))
#loc24 = loc("PallasKernelTest.test_semaphore_and_dma_22_04_2024.<locals>.func"(#loc12))
#loc33 = loc(callsite(#loc13 at #loc32))
#loc35 = loc(callsite(#loc24 at #loc32))
#loc36 = loc("jit(func)/jit(main)/iota[dtype=float32 shape=(16384,) dimension=0]"(#loc33))
#loc37 = loc("jit(func)/jit(main)/reshape[new_sizes=(128, 128) dimensions=None]"(#loc33))
#loc39 = loc("jit(func)/jit(main)/eq"(#loc35))
#loc40 = loc("jit(func)/jit(main)/reduce_and[axes=(0, 1)]"(#loc35))
#loc41 = loc("jit(func)/jit(main)/convert_element_type[new_dtype=float32 weak_type=False]"(#loc35))
#loc43 = loc("jit(func)/jit(main)/jit(wrapped)/jit(apply_kernel)/tpu_custom_call[config=CustomCallBackendConfig(<omitted>) kernel_name=dma_kernel kernel_regeneration_metadata=None out_avals=(ShapedArray(float32[128,128]),) input_output_aliases=()]"(#loc34))
""",
    mlir_module_serialized=b'ML\xefR\x01StableHLO_v0.9.0\x00\x01\'\x05\x01\x03\x01\x03\x05\x03\x17\x07\t\x0b\r\x0f\x11\x13\x15\x17\x19\x1b\x03z\x02\n\x02\x1f\x01\xc9\x0f\x0b\x0b\x0b\x0f\x0f\x07\x0b\x0b\x0b\x0b\x0f\x0b\x0f\x0f\x0f\x0b\x0b\x0f+\x0b\x0f\x0b\x0b\x0b33\x0b\x0f\x13\x0f\x0b\x13\x0f\x0f\x0b\x17\x0f\x0f\x0b\x17\x0f\x0f\x0b\x17\x0f\x0f\x0b\x17\x0f\x0f\x0b\x17\x0f\x0f\x0b\x17\x0f\x0f\x0b\x17\x0f\x0b\x133\x0bS\x0b\x0b\x0b\x0b\x0b\x0b\x0b\x0b\x0b\x0f\x0b\x13\x13\x0b\x0f\x0b\x0f\x13\x0f\x0b\x13\x1b\x0b\x0b\x0f\x0b\x0f\x13\x13\x0b\x0b\x13\x0b\x039\x0f\x0b\x0b\x0b\x0b\x0b\x0b\x0b\x0f\x0b\x0f\x1b\x0b\x0b\x0b\x0b\x13\x0b\x0b\x0b\x0b\x0bO\x0f\x0b\x0b\x13O\x01\x05\x13\x0b\x01\x05\x0b\x0f\x03\x1b\x1f\x0f\x07\x0f\x07\x07\x13\x17\x13\x07\x1b\x1f\x13\x02\xb2\x07\x1d\xc3\x1d\x05\x1d\x05\x1f\x05!\x1d7\x17\x1d\x83\x17\x1f\x05#\x05%\x05\'\x05)\x159\x1b\x05+\x15=C\x15\xbb\x1b\x11\x03\x05\x05-\x05/\x15\xa7\x1b\x03\t)+-\x1f/\x1f\x071\x051\x11\x01\x00\x053\x055\x057\x03\x0b\x0f\xcb\x11\xdb\x13\xdd\x07\xe5\x15\xe7\x03\x0b\x0f\xc9\x11\xd1\x13\xc9\x07\xd3\x15\xd5\x059\x1d\x19;\x17\x03s\x15\x1d?A\x05;\x17\x03y\t\x15EK\x1dGI\x05=\x17\x05z*\'\x15MS\x1dOQ\x05?\x17\x05\n+G\x15U[\x1dWY\x05A\x17\x05\xea#\r\x15]c\x1d_a\x05C\x17!.\x06\x1b\x15ek\x1dgi\x05E\x17!f\x07\r\x15ms\x1doq\x05G\x17\x05\xf2#\t\x15u{\x1dwy\x05I\x17\x05\x1a"\x05\x1d}\x7f\x05K\x17\x03\x81\x05\x03\x0b\x0f\xc9\x11\xd1\x13\xc9\x07\xd7\x15\xd5\x05M\x03\x13\x87\xeb\x89\xed\x8b\xef\x8d\xcb\x8f\xf1\x91\xf3\x93\xd9\x95\xcb\x97\xd9\x05O\x05Q\x05S\x05U\x05W\x05Y\x05[\x05]\x05_\x1d\x9b\x17\x05a\x03\x03#\xd7\x03\x03\xa1\xf7\x05c\x1d\xa5%\x05e\x1d\x19\xa9\x17\x03q\x15\x1d\xad%\x05g\x03\x03#\xd3\x03\x05\xb3\xf9\xb5\xfb\x05i\x05k\x1d\xb9\x1d\x05m\x1d\x19\xbd\x17\x03u\x1b\x03\x03\xc1\xfd\x05o\x05q\x03\x03\xc7\xff\x05s\x03\x03\xe9\x03\x01\x1du\x1dw#\x13\x1dy\x1d{\x1d}\x03\x03\xf5#\x11\x03\x03\xdf\r\x05\xe1\xe3\xcd\xcf\x1d\x7f\x1d\x81\x1dI\x1d\x83\r\x03\xcd\xcf\x0b\x03\x1d\x85\x1d\x87\x05\x01\x1d\x89\x1f\x15!\x01\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x13\r\x01\t\x03\x07\x01\x1f\x07\x03\xff\x1f\x1d!\x00\x00\x00\x00\x00\x00\x00\x00\x01\x00\x00\x00\x00\x00\x00\x00\x1d\x06\x02\x1d\x05\x8b\x01\t\x01\x02\x02)\x05\x02\x04\x02\x04\t)\x01\x0f\t)\x01\t\x1d\x01\x11\x01\x03\x0b\x11\x03\x05\x03\x05)\x03\t\x17\x13)\x03\x04\x00\x04\t)\x05\x02\x04\x02\x04\x0f)\x03\t\r\x04F\x02\x05\x01\x11\r\'\x07\x03\x01\r\x05\x11\r3\x07\x03\x0f!\x0b\x03\xa3\x9f\x03\x19\r\x06\xab\x03\x05\x03\x01\x07\x07\t\xaf\x03\x05\x03\x03\x0f\x07\xb7\xb1\x03\x1b\x05\x03\x05\x11\x03\x01\xbf\x03\x07\x13\x17\x01\xc5\x03\x07\x05\x07\t\x07\x03\x07\x0b\x05\x07\x01\x07\x01\x17\x06\x01\x03\x07\x05\x01\x03\x03\x04\x01\x03\x05\x15\x06\x02\x02\x03\x0b\x03\x0b\x03\x04\r\x03\r\x05\x11\t5\x07\x03\x05\x0b\x03\x05\t\x07\x07\x0b\x9d\x03\x05\x03\x01\x03\x04\t\x03\x03\x05\x11\x0b\x81\x07\x03\x05\x0b\x03\x05\x0b\t\x07\x99\x85\x03\x05\x03\x01\x03\x04\x0b\x03\x03\x06\x03\x01\x05\x01\x00\xbeG\x8d\x99\x17!\xba(\x0f\x03!\x1b\x11\x11\x11#\x17Y\r/+\x1b\x85\x87\x1f\xaa\x03\x1f/!\x19!)#\x1f\x19\xb2\x03\x13\x0b\x19\t\x15G\x155gj\x03\x13%)9\x0f7\x83\x1f\x15\x1d\x15\x13Q\x81\x0f\x17\x15\x19\x17\x17\x11\x1f\x11\x11\x15\x0f\x0b\x11builtin\x00vhlo\x00module\x00return_v1\x00func_v1\x00call_v1\x00custom_call_v1\x00iota_v1\x00reshape_v1\x00compare_v1\x00constant_v1\x00reduce_v1\x00convert_v1\x00and_v1\x00third_party/py/jax_triton/googlexpallas_tpu/back_compat_test.py\x00third_party/py/absl/testing/absltest.py\x00sym_name\x00arg_attrs\x00function_type\x00res_attrs\x00sym_visibility\x00PallasKernelTest.test_semaphore_and_dma_22_04_2024.<locals>.func\x00third_party/py/absl/app.py\x00callee\x00jax.uses_shape_polymorphism\x00mhlo.num_partitions\x00mhlo.num_replicas\x00jit_func\x00jit(func)/jit(main)/pjit[in_shardings=(UnspecifiedValue,) out_shardings=(UnspecifiedValue,) in_layouts=(None,) out_layouts=(None,) resource_env=None donated_invars=(False,) name=wrapped keep_unused=False inline=False]\x00PallasKernelTest.test_semaphore_and_dma_22_04_2024\x00_run_and_get_tests_result\x00run_tests\x00_run_in_app.<locals>.main_function\x00_run_main\x00run\x00_run_in_app\x00main\x00<module>\x00jit(func)/jit(main)/jit(wrapped)/pjit[in_shardings=(UnspecifiedValue,) out_shardings=(UnspecifiedValue,) in_layouts=(None,) out_layouts=(None,) resource_env=None donated_invars=(False,) name=apply_kernel keep_unused=False inline=False]\x00api_version\x00backend_config\x00call_target_name\x00called_computations\x00has_side_effect\x00kernel_name\x00operand_layouts\x00output_operand_aliases\x00result_layouts\x00jit(func)/jit(main)/jit(wrapped)/jit(apply_kernel)/tpu_custom_call[config=CustomCallBackendConfig(<omitted>) kernel_name=dma_kernel kernel_regeneration_metadata=None out_avals=(ShapedArray(float32[128,128]),) input_output_aliases=()]\x00iota_dimension\x00jit(func)/jit(main)/iota[dtype=float32 shape=(16384,) dimension=0]\x00jit(func)/jit(main)/reshape[new_sizes=(128, 128) dimensions=None]\x00compare_type\x00comparison_direction\x00jit(func)/jit(main)/eq\x00value\x00jit(func)/jit(main)/reduce_and[axes=(0, 1)]\x00dimensions\x00mhlo.layout_mode\x00default\x00wrapped\x00private\x00apply_kernel\x00jax.result_info\x00\x00public\x00{"custom_call_config": {"body": "TUzvUgFNTElSZ29vZ2xlMy10cnVuawABJwcBAwUBAwcDFQkLDQ8RExUXGRsD27UTAbELBwsPCw8PCw8PDw8PDw8LDw9VDxMPDxMLDzMLCwsLhQsLCwsPCxMPCxMPCxMPCxcPCxcPCxcPCxcPCxcPCxcPFw8LDxMPDw8PDw8PDwsLDwsPDxMLDw8TBQWFYQEPJw8PFwcXFwUFTT0CzgYFHR8FHx1HSQUhFRGLEQUBBSMdS00dUVMdV1kdXV8dY2UdaWsdb3EFJR11dx17fWFmZmluZV9tYXA8KCkgLT4gKCk+ABWHCwMDnZ8doaMdqasDAzEzBScRBQUDCzc5Oz1BDUMNRQ8FKQEBBSsNB2FmZmluZV9tYXA8KGQwLCBkMSkgLT4gKGQwLCBkMSk+AAUtBS8FMQUzFRFPBTUXAWsRFRNVBTcXAXMVFRVbBTkXAXkJFRdhBTsXBXoqJxUZZwU9FwUKK0cVG20FPxcF6iMNFR1zBUEXHy4GGxUheQVDFx9mBw0VI38FRRcF8iMJHQ+BFwUaIgUdhScFRx0JiRcBZRUVE40VFY8VF5EVGZMVG5UVHZcVISMdmycFSQVLEQMFBU0VpQsdCacXAWcVBU8VrQsdCa8XAWkVI3RwdS5tZW1vcnlfc3BhY2U8c2VtYXBob3JlX21lbT4AI3RwdS5tZW1vcnlfc3BhY2U8dm1lbT4AF7MFAgQCBAk/AQICAQIEBQUBAQELF7EBDyUXsQERJSF0cHUuZG1hX3NlbWFwaG9yZQAhdHB1LnNlbWFwaG9yZQAEpQUBEQMvBwMBBQcRAzUHAwULBQEDAQMJEAcFAwklAwIHAwsDAgcDDQ0EgwcBAwUPBJkFBQMFAyspAwMRBCsFBwkFAy0pAwMTBC0FBwsVAAcLAAMGAwEFAQA+ElFjtQ2XyxkJFUcVNWeDqxkTIyEdKS03C8dRgRUbHxshGRcVHx0PCR0RYnVpbHRpbgBzdGFibGVfbW9zYWljAHRwdQBtb2R1bGUAdHB1LnNlbV9hbGxvYwBhcml0aC5jb25zdGFudABmdW5jLmZ1bmMAdHB1LnJlZ2lvbgBmdW5jLnJldHVybgB0cHUuZW5xdWV1ZV9kbWEAdHB1LndhaXRfZG1hAHRwdS5zZW1fc2lnbmFsAHRwdS5zZW1fd2FpdAB0cHUueWllbGQAdGhpcmRfcGFydHkvcHkvamF4X3RyaXRvbi9nb29nbGUvcGFsbGFzX3RwdS9iYWNrX2NvbXBhdF90ZXN0LnB5AHRoaXJkX3BhcnR5L3B5L2Fic2wvdGVzdGluZy9hYnNsdGVzdC5weQBQYWxsYXNLZXJuZWxUZXN0LnRlc3Rfc2VtYXBob3JlX2FuZF9kbWFfMjJfMDRfMjAyNC48bG9jYWxzPi5mdW5jLjxsb2NhbHM+LmRtYV9rZXJuZWwuPGxvY2Fscz4uYm9keQBtYWluAHRoaXJkX3BhcnR5L3B5L2Fic2wvYXBwLnB5AHN0YWJsZV9tb3NhaWMudmVyc2lvbgBkaW1lbnNpb25fc2VtYW50aWNzAGZ1bmN0aW9uX3R5cGUAc2NhbGFyX3ByZWZldGNoAHNjcmF0Y2hfb3BlcmFuZHMAc3ltX25hbWUAL3J1bl9zY29wZWQAUGFsbGFzS2VybmVsVGVzdC50ZXN0X3NlbWFwaG9yZV9hbmRfZG1hXzIyXzA0XzIwMjQuPGxvY2Fscz4uZnVuYy48bG9jYWxzPi5kbWFfa2VybmVsAFBhbGxhc0tlcm5lbFRlc3QudGVzdF9zZW1hcGhvcmVfYW5kX2RtYV8yMl8wNF8yMDI0Ljxsb2NhbHM+LmZ1bmMAUGFsbGFzS2VybmVsVGVzdC50ZXN0X3NlbWFwaG9yZV9hbmRfZG1hXzIyXzA0XzIwMjQAX3J1bl9hbmRfZ2V0X3Rlc3RzX3Jlc3VsdABydW5fdGVzdHMAX3J1bl9pbl9hcHAuPGxvY2Fscz4ubWFpbl9mdW5jdGlvbgBfcnVuX21haW4AcnVuAF9ydW5faW5fYXBwAC9kbWFfc3RhcnRbdHJlZT1QeVRyZWVEZWYoKCosICgpLCAqLCAoKSwgKiwgKCksIE5vbmUsIE5vbmUsIE5vbmUpKSBkZXZpY2VfaWRfdHlwZT1EZXZpY2VJZFR5cGUuTUVTSF0AL2RtYV93YWl0W3RyZWU9UHlUcmVlRGVmKCgqLCAoKSwgKiwgKCkpKSBkZXZpY2VfaWRfdHlwZT1EZXZpY2VJZFR5cGUuTUVTSF0AdmFsdWUAL3NlbWFwaG9yZV9zaWduYWxbYXJnc190cmVlPVB5VHJlZURlZihbKiwgKCksICosIE5vbmVdKSBkZXZpY2VfaWRfdHlwZT1EZXZpY2VJZFR5cGUuTUVTSF0AL3NlbWFwaG9yZV93YWl0W2FyZ3NfdHJlZT1QeVRyZWVEZWYoWyosICgpLCAqXSldAA==", "serialization_format": 1, "needs_layout_passes": true}, "implicit_sharding": {"type": "MANUAL"}}\x00tpu_custom_call\x00dma_kernel\x00jit(func)/jit(main)/convert_element_type[new_dtype=float32 weak_type=False]\x00',
    xla_call_module_version=9,
    nr_devices=1,
)  # End paste

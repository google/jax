/* Copyright 2022 The JAX Authors.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#ifndef JAX_CUSTOM_CALL_H_
#define JAX_CUSTOM_CALL_H_

#include <cstddef>

#ifdef __cplusplus
extern "C" {
#endif

#define JaxFFICallCpu "JaxFFICallCpu"

#define JaxFFIVersion(api) (api->version)
#define JaxFFIStatusSetSuccess(api, status) \
        ((*((void (*)(JaxFFIStatus*))(api->fn_table[0])(status))))
#define JaxFFIStatusSetFailure(api, status, msg) \
        ((*((void (*)(JaxFFIStatus*, const char*))(api->fn_table[1])))(status, msg))

struct JaxFFIStatus;

struct JaxFFIApi {
  int version;
  void** fn_table;
};

/*
void add_one(JaxFFIApi* api, JaxFFIStatus* status, ...) {
  JaxFFIStatusSetFailure(api, status, "...");
}
*/

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // JAX_CUSTOM_CALL_H_

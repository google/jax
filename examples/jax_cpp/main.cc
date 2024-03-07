/* Copyright 2021 The JAX Authors.

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

// An example for reading a HloModule from a HloProto file and execute the
// module on PJRT CPU client.
//
// To build a HloModule,
//
// $ python3 jax/tools/jax_to_hlo.py \
// --fn examples.jax_cpp.prog.fn \
// --input_shapes '[("x", "f32[2,2]"), ("y", "f32[2,2]")]' \
// --constants '{"z": 2.0}' \
// --hlo_text_dest /tmp/fn_hlo.txt \
// --hlo_proto_dest /tmp/fn_hlo.pb
//
// To load and run the HloModule,
//
// $ bazel build examples/jax_cpp:main --experimental_repo_remote_exec \
//    --check_visibility=false
// $ bazel-bin/examples/jax_cpp/main 2021-01-12
// 15:35:28.316880: I examples/jax_cpp/main.cc:65] result = ( f32[2,2] {
//   { 1.5, 1.5 },
//   { 3.5, 3.5 }
// }
// )

#include <memory>
#include <string>
#include <vector>

#include "third_party/absl/status/statusor.h"
#include "xla/literal.h"
#include "xla/literal_util.h"
#include "xla/pjrt/cpu/cpu_client.h"
#include "xla/pjrt/pjrt_client.h"
#include "xla/tools/hlo_module_loader.h"
#include "tsl/platform/init_main.h"
#include "tsl/platform/logging.h"

int main(int argc, char** argv) {
  tsl::port::InitMain("", &argc, &argv);

  // Load HloModule from file.
  std::string hlo_filename = "/tmp/fn_hlo.txt";
  std::function<void(xla::HloModuleConfig*)> config_modifier_hook =
      [](xla::HloModuleConfig* config) { config->set_seed(42); };
  std::unique_ptr<xla::HloModule> test_module =
      LoadModuleFromFile(hlo_filename, /*format=*/"txt",
                         xla::hlo_module_loader_details::Config(),
                         config_modifier_hook)
          .value();
  const xla::HloModuleProto test_module_proto = test_module->ToProto();

  // Run it using JAX C++ Runtime (PJRT).

  // Get a CPU client.
  std::unique_ptr<xla::PjRtClient> client =
      xla::GetTfrtCpuClient(/*asynchronous=*/true).value();

  // Compile XlaComputation to PjRtExecutable.
  xla::XlaComputation xla_computation(test_module_proto);
  xla::CompileOptions compile_options;
  std::unique_ptr<xla::PjRtLoadedExecutable> executable =
      client->Compile(xla_computation, compile_options).value();

  // Prepare inputs.
  xla::Literal literal_x =
      xla::LiteralUtil::CreateR2<float>({{1.0f, 2.0f}, {3.0f, 4.0f}});
  xla::Literal literal_y =
      xla::LiteralUtil::CreateR2<float>({{1.0f, 1.0f}, {1.0f, 1.0f}});
  std::unique_ptr<xla::PjRtBuffer> param_x =
      client->BufferFromHostLiteral(literal_x, client->addressable_devices()[0])
          .value();
  std::unique_ptr<xla::PjRtBuffer> param_y =
      client->BufferFromHostLiteral(literal_y, client->addressable_devices()[0])
          .value();

  // Execute on CPU.
  xla::ExecuteOptions execute_options;
  // One vector<buffer> for each device.
  std::vector<std::vector<std::unique_ptr<xla::PjRtBuffer>>> results =
      executable->Execute({{param_x.get(), param_y.get()}}, execute_options)
          .value();

  // Get result.
  std::shared_ptr<xla::Literal> result_literal =
      results[0][0]->ToLiteralSync().value();
  LOG(INFO) << "result = " << *result_literal;
  return 0;
}

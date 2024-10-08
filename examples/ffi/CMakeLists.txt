cmake_minimum_required(VERSION 3.15...3.30)
project(${SKBUILD_PROJECT_NAME} LANGUAGES CXX)

option(JAX_FFI_EXAMPLE_ENABLE_CUDA "Enable CUDA support" OFF)

find_package(Python 3.10 REQUIRED COMPONENTS Interpreter Development.Module)
execute_process(
  COMMAND "${Python_EXECUTABLE}"
          "-c" "from jax.extend import ffi; print(ffi.include_dir())"
  OUTPUT_STRIP_TRAILING_WHITESPACE OUTPUT_VARIABLE XLA_DIR)
message(STATUS "XLA include directory: ${XLA_DIR}")

find_package(nanobind CONFIG REQUIRED)

nanobind_add_module(_rms_norm NB_STATIC "src/jax_ffi_example/rms_norm.cc")
target_include_directories(_rms_norm PUBLIC ${XLA_DIR})
install(TARGETS _rms_norm LIBRARY DESTINATION ${SKBUILD_PROJECT_NAME})

nanobind_add_module(_attrs NB_STATIC "src/jax_ffi_example/attrs.cc")
target_include_directories(_attrs PUBLIC ${XLA_DIR})
install(TARGETS _attrs LIBRARY DESTINATION ${SKBUILD_PROJECT_NAME})

if(JAX_FFI_EXAMPLE_ENABLE_CUDA)
  enable_language(CUDA)
  add_library(_cuda_e2e SHARED "src/jax_ffi_example/cuda_e2e.cu")
  set_target_properties(_cuda_e2e PROPERTIES POSITION_INDEPENDENT_CODE ON
                                             CUDA_STANDARD 17)
  target_include_directories(_cuda_e2e PUBLIC ${XLA_DIR})
  install(TARGETS _cuda_e2e LIBRARY DESTINATION ${SKBUILD_PROJECT_NAME})
endif()

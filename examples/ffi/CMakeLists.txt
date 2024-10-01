cmake_minimum_required(VERSION 3.15...3.30)
project(${SKBUILD_PROJECT_NAME} LANGUAGES CXX)

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

nanobind_add_module(_counter NB_STATIC "src/jax_ffi_example/counter.cc")
target_include_directories(_counter PUBLIC ${XLA_DIR})
install(TARGETS _counter LIBRARY DESTINATION ${SKBUILD_PROJECT_NAME})

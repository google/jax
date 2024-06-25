cmake_minimum_required(VERSION 3.18...3.27)
project(rms_norm LANGUAGES CXX)
find_package(Python 3.8 COMPONENTS Interpreter Development.Module REQUIRED)

execute_process(
  COMMAND "${Python_EXECUTABLE}" -m nanobind --cmake_dir
  OUTPUT_STRIP_TRAILING_WHITESPACE OUTPUT_VARIABLE NB_DIR)
list(APPEND CMAKE_PREFIX_PATH "${NB_DIR}")
find_package(nanobind CONFIG REQUIRED)

execute_process(
  COMMAND "${Python_EXECUTABLE}"
          "-c" "from jax.extend import ffi; print(ffi.include_dir())"
  OUTPUT_STRIP_TRAILING_WHITESPACE OUTPUT_VARIABLE XLA_DIR)
message(STATUS "XLA include directory: ${XLA_DIR}")

nanobind_add_module(rms_norm NOMINSIZE "rms_norm.cc")
target_include_directories(rms_norm PUBLIC ${XLA_DIR})
install(TARGETS rms_norm LIBRARY DESTINATION ${CMAKE_CURRENT_LIST_DIR})


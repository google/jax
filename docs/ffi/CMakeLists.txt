cmake_minimum_required(VERSION 3.18...3.27)
project(rms_norm LANGUAGES CXX)

find_package(Python 3.8 COMPONENTS Interpreter Development.Module REQUIRED)
execute_process(
  COMMAND "${Python_EXECUTABLE}"
          "-c" "from jax.extend import ffi; print(ffi.include_dir())"
  OUTPUT_STRIP_TRAILING_WHITESPACE OUTPUT_VARIABLE XLA_DIR)
message(STATUS "XLA include directory: ${XLA_DIR}")

add_library(rms_norm SHARED "rms_norm.cc")
target_include_directories(rms_norm PUBLIC ${XLA_DIR})
target_compile_features(rms_norm PUBLIC cxx_std_17)
install(TARGETS rms_norm LIBRARY DESTINATION ${CMAKE_CURRENT_LIST_DIR})

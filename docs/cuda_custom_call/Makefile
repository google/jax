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


# This Makefile is not used by Bazel for this test, it is intended to serve as
# documentation of build instructions for JAX users that are not using Bazel to
# build their custom call code. For that reason, this Makefile is likely subject
# to bitrot over time. Please file a JAX issue on GitHub if typing "make" in
# this directory no longer runs the test to completion.
NVCC = nvcc
NVCCFLAGS += -I$(shell python -c 'from jax.extend import ffi; print(ffi.include_dir())')
NVCCFLAGS += -arch native
# since the file extension is .cu.cc, tell NVCC explicitly to treat it as .cu
NVCCFLAGS += -x cu

# depends on libfoo.so being in the same directory as cuda_custom_call_test.py
check: libfoo.so
	python cuda_custom_call_test.py

lib%.so: %.cu.cc
	$(NVCC) $(NVCCFLAGS) --compiler-options=-shared,-fPIC -o $@ $<

clean:
	rm -rf *.so

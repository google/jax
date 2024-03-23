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

#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include <stdint.h>

#include "platform.h"
#include "cpuid.h"

#ifdef PLATFORM_IS_X86
static void ReportMissingCpuFeature(const char* name) {
  PyErr_Format(
      PyExc_RuntimeError,
      "This version of jaxlib was built using %s instructions, which your "
      "CPU and/or operating system do not support. You may be able work around "
      "this issue by building jaxlib from source.", name);
}

static PyObject *CheckCpuFeatures(PyObject *self, PyObject *args) {
  uint32_t eax, ebx, ecx, edx;

  CpuFeatureFlags cpu_flags = GetCpuFeatureFlags();

#ifdef __AVX__
  if (!cpu_flags.have_avx) {
    ReportMissingCpuFeature("AVX");
    return NULL;
  }
#endif  // __AVX__

#ifdef __AVX2__
  if (!cpu_flags.have_avx2) {
    ReportMissingCpuFeature("AVX2");
    return NULL;
  }
#endif  // __AVX2__

#ifdef __FMA__
  if (!cpu_flags.have_fma) {
    ReportMissingCpuFeature("FMA");
    return NULL;
  }
#endif  // __FMA__

  Py_INCREF(Py_None);
  return Py_None;
}

#else  // PLATFORM_IS_X86

static PyObject *CheckCpuFeatures(PyObject *self, PyObject *args) {
  Py_INCREF(Py_None);
  return Py_None;
}

#endif  // PLATFORM_IS_X86

static PyMethodDef cpu_feature_guard_methods[] = {
    {"check_cpu_features", CheckCpuFeatures, METH_NOARGS,
     "Throws an exception if the CPU is missing instructions used by jaxlib."},
    {NULL, NULL, 0, NULL}};

static struct PyModuleDef cpu_feature_guard_module = {
    PyModuleDef_HEAD_INIT, "cpu_feature_guard", /* name of module */
    NULL, -1, /* size of per-interpreter state of the module,
                 or -1 if the module keeps state in global variables. */
    cpu_feature_guard_methods};

EXPORT_SYMBOL PyMODINIT_FUNC PyInit_cpu_feature_guard(void) {
  return PyModule_Create(&cpu_feature_guard_module);
}

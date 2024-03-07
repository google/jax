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

#if defined(__x86_64__) || defined(__amd64__) || defined(_M_IX86) || \
    defined(_M_X64)
#define PLATFORM_IS_X86
#endif

#if defined(_WIN32)
#define PLATFORM_WINDOWS
#endif

// SIMD extension querying is only available on x86.
#ifdef PLATFORM_IS_X86

#ifdef PLATFORM_WINDOWS
#if defined(_MSC_VER)
#include <intrin.h>
#endif

// Visual Studio defines a builtin function for CPUID, so use that if possible.
#define GETCPUID(a, b, c, d, a_inp, c_inp) \
  {                                        \
    int cpu_info[4] = {-1};                \
    __cpuidex(cpu_info, a_inp, c_inp);     \
    a = cpu_info[0];                       \
    b = cpu_info[1];                       \
    c = cpu_info[2];                       \
    d = cpu_info[3];                       \
  }

// Visual Studio defines a builtin function, so use that if possible.
static int GetXCR0EAX() { return _xgetbv(0); }

#else

// Otherwise use gcc-format assembler to implement the underlying instructions.
#define GETCPUID(a, b, c, d, a_inp, c_inp) \
  asm("mov %%rbx, %%rdi\n"                 \
      "cpuid\n"                            \
      "xchg %%rdi, %%rbx\n"                \
      : "=a"(a), "=D"(b), "=c"(c), "=d"(d) \
      : "a"(a_inp), "2"(c_inp))

static int GetXCR0EAX() {
  int eax, edx;
  asm("XGETBV" : "=a"(eax), "=d"(edx) : "c"(0));
  return eax;
}

#endif
#endif

// TODO(phawkins): technically we should build this module without AVX support
// and use configure-time tests instead of __AVX__, since there is a
// possibility that the compiler will use AVX instructions before we reach this
// point.
#ifdef PLATFORM_IS_X86

static void ReportMissingCpuFeature(const char* name) {
  PyErr_Format(
      PyExc_RuntimeError,
#if defined(__APPLE__)
      "This version of jaxlib was built using %s instructions, which your "
      "CPU and/or operating system do not support. This error is frequently "
      "encountered on macOS when running an x86 Python installation on ARM "
      "hardware. In this case, try installing an ARM build of Python. "
      "Otherwise, you may be able work around this issue by building jaxlib "
      "from source.",
#else
      "This version of jaxlib was built using %s instructions, which your "
      "CPU and/or operating system do not support. You may be able work around "
      "this issue by building jaxlib from source.",
#endif
      name);
}

static PyObject *CheckCpuFeatures(PyObject *self, PyObject *args) {
  uint32_t eax, ebx, ecx, edx;

  // To get general information and extended features we send eax = 1 and
  // ecx = 0 to cpuid.  The response is returned in eax, ebx, ecx and edx.
  // (See Intel 64 and IA-32 Architectures Software Developer's Manual
  // Volume 2A: Instruction Set Reference, A-M CPUID).
  GETCPUID(eax, ebx, ecx, edx, 1, 0);
  const uint64_t xcr0_xmm_mask = 0x2;
  const uint64_t xcr0_ymm_mask = 0x4;
  const uint64_t xcr0_avx_mask = xcr0_xmm_mask | xcr0_ymm_mask;
  const _Bool have_avx =
      // Does the OS support XGETBV instruction use by applications?
      ((ecx >> 27) & 0x1) &&
      // Does the OS save/restore XMM and YMM state?
      ((GetXCR0EAX() & xcr0_avx_mask) == xcr0_avx_mask) &&
      // Is AVX supported in hardware?
      ((ecx >> 28) & 0x1);
  const _Bool have_fma = have_avx && ((ecx >> 12) & 0x1);

  // Get standard level 7 structured extension features (issue CPUID with
  // eax = 7 and ecx= 0), which is required to check for AVX2 support as
  // well as other Haswell (and beyond) features.  (See Intel 64 and IA-32
  // Architectures Software Developer's Manual Volume 2A: Instruction Set
  // Reference, A-M CPUID).
  GETCPUID(eax, ebx, ecx, edx, 7, 0);
  const _Bool have_avx2 = have_avx && ((ebx >> 5) & 0x1);

#ifdef __AVX__
  if (!have_avx) {
    ReportMissingCpuFeature("AVX");
    return NULL;
  }
#endif  // __AVX__

#ifdef __AVX2__
  if (!have_avx2) {
    ReportMissingCpuFeature("AVX2");
    return NULL;
  }
#endif  // __AVX2__

#ifdef __FMA__
  if (!have_fma) {
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

#if defined(WIN32) || defined(_WIN32)
#define EXPORT_SYMBOL __declspec(dllexport)
#else
#define EXPORT_SYMBOL __attribute__ ((visibility("default")))
#endif

EXPORT_SYMBOL PyMODINIT_FUNC PyInit_cpu_feature_guard(void) {
  return PyModule_Create(&cpu_feature_guard_module);
}

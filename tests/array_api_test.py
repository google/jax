# Copyright 2023 The JAX Authors.
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

"""Smoketest for jax.experimental.array_api

The full test suite for the array API is run via the array-api-tests CI;
this is just a minimal smoke test to catch issues early.
"""
from __future__ import annotations

from types import ModuleType

from absl.testing import absltest, parameterized
import jax
import jax.numpy as jnp
from jax._src import config, test_util as jtu
from jax._src.dtypes import _default_types, canonicalize_dtype
from jax.experimental import array_api

config.parse_flags_with_absl()

MAIN_NAMESPACE = {
  'abs',
  'acos',
  'acosh',
  'add',
  'all',
  'annotations',
  'any',
  'arange',
  'argmax',
  'argmin',
  'argsort',
  'asarray',
  'asin',
  'asinh',
  'astype',
  'atan',
  'atan2',
  'atanh',
  'bitwise_and',
  'bitwise_invert',
  'bitwise_left_shift',
  'bitwise_or',
  'bitwise_right_shift',
  'bitwise_xor',
  'bool',
  'broadcast_arrays',
  'broadcast_to',
  'can_cast',
  'ceil',
  'clip',
  'complex128',
  'complex64',
  'concat',
  'conj',
  'copysign',
  'cos',
  'cosh',
  'cumulative_sum',
  'divide',
  'e',
  'empty',
  'empty_like',
  'equal',
  'exp',
  'expand_dims',
  'expm1',
  'eye',
  'fft',
  'finfo',
  'flip',
  'float32',
  'float64',
  'floor',
  'floor_divide',
  'from_dlpack',
  'full',
  'full_like',
  'greater',
  'greater_equal',
  'hypot',
  'iinfo',
  'imag',
  'inf',
  'int16',
  'int32',
  'int64',
  'int8',
  'isdtype',
  'isfinite',
  'isinf',
  'isnan',
  'less',
  'less_equal',
  'linalg',
  'linspace',
  'log',
  'log10',
  'log1p',
  'log2',
  'logaddexp',
  'logical_and',
  'logical_not',
  'logical_or',
  'logical_xor',
  'matmul',
  'matrix_transpose',
  'max',
  'maximum',
  'mean',
  'meshgrid',
  'min',
  'minimum',
  'moveaxis',
  'multiply',
  'nan',
  'negative',
  'newaxis',
  'nonzero',
  'not_equal',
  'ones',
  'ones_like',
  'permute_dims',
  'pi',
  'positive',
  'pow',
  'prod',
  'real',
  'remainder',
  'repeat',
  'reshape',
  'result_type',
  'roll',
  'round',
  'searchsorted',
  'sign',
  'signbit',
  'sin',
  'sinh',
  'sort',
  'sqrt',
  'square',
  'squeeze',
  'stack',
  'std',
  'subtract',
  'sum',
  'take',
  'tan',
  'tanh',
  'tensordot',
  'tile',
  'tril',
  'triu',
  'trunc',
  'uint16',
  'uint32',
  'uint64',
  'uint8',
  'unique_all',
  'unique_counts',
  'unique_inverse',
  'unique_values',
  'unstack',
  'var',
  'vecdot',
  'where',
  'zeros',
  'zeros_like',
}

LINALG_NAMESPACE = {
  'cholesky',
  'cross',
  'det',
  'diagonal',
  'eigh',
  'eigvalsh',
  'inv',
  'matmul',
  'matrix_norm',
  'matrix_power',
  'matrix_rank',
  'matrix_transpose',
  'outer',
  'pinv',
  'qr',
  'slogdet',
  'solve',
  'svd',
  'svdvals',
  'tensordot',
  'trace',
  'vecdot',
  'vector_norm',
}

FFT_NAMESPACE = {
  'fft',
  'fftfreq',
  'fftn',
  'fftshift',
  'hfft',
  'ifft',
  'ifftn',
  'ifftshift',
  'ihfft',
  'irfft',
  'irfftn',
  'rfft',
  'rfftfreq',
  'rfftn',
}


def names(module: ModuleType) -> set[str]:
  return {name for name in dir(module) if not name.startswith('_')}


class ArrayAPISmokeTest(absltest.TestCase):
  """Smoke test for the array API."""

  def test_main_namespace(self):
    self.assertContainsSubset(MAIN_NAMESPACE, names(array_api))

  def test_linalg_namespace(self):
    self.assertContainsSubset(LINALG_NAMESPACE, names(array_api.linalg))

  def test_fft_namespace(self):
    self.assertContainsSubset(FFT_NAMESPACE, names(array_api.fft))

  def test_array_namespace_method(self):
    x = array_api.arange(20)
    self.assertIsInstance(x, jax.Array)
    self.assertIs(x.__array_namespace__(), jax.numpy)

class ArrayAPIInspectionUtilsTest(jtu.JaxTestCase):

  info = array_api.__array_namespace_info__()

  def setUp(self):
    super().setUp()
    self._boolean = self.build_dtype_dict(["bool"])
    self._signed = self.build_dtype_dict(["int8", "int16", "int32"])
    self._unsigned = self.build_dtype_dict(["uint8", "uint16", "uint32"])
    self._floating = self.build_dtype_dict(["float32"])
    self._complex = self.build_dtype_dict(["complex64"])
    if config.enable_x64.value:
      self._signed["int64"] = jnp.dtype("int64")
      self._unsigned["uint64"] = jnp.dtype("uint64")
      self._floating["float64"] = jnp.dtype("float64")
      self._complex["complex128"] = jnp.dtype("complex128")
    self._integral = self._signed | self._unsigned
    self._numeric = (
      self._signed | self._unsigned | self._floating | self._complex
    )
  def build_dtype_dict(self, dtypes):
    out = {}
    for name in dtypes:
        out[name] = jnp.dtype(name)
    return out

  def test_capabilities_info(self):
    capabilities = self.info.capabilities()
    assert capabilities["boolean indexing"]
    assert not capabilities["data-dependent shapes"]

  def test_default_device_info(self):
    assert self.info.default_device() is None

  def test_devices_info(self):
    assert self.info.devices() == jax.devices()

  def test_default_dtypes_info(self):
    _default_dtypes = {
      "real floating": "f",
      "complex floating": "c",
      "integral": "i",
      "indexing": "i",
    }
    target_dict = {
      dtype_name: canonicalize_dtype(
        _default_types.get(kind)
      ) for dtype_name, kind in _default_dtypes.items()
    }
    assert self.info.default_dtypes() == target_dict

  @parameterized.parameters(
    "bool", "signed integer", "real floating",
    "complex floating", "integral", "numeric", None,
    (("real floating", "complex floating"),),
    (("integral", "signed integer"),),
    (("integral", "bool"),),
  )
  def test_dtypes_info(self, kind):

    info_dict = self.info.dtypes(kind=kind)
    control = {
      "bool":self._boolean,
      "signed integer":self._signed,
      "unsigned integer":self._unsigned,
      "real floating":self._floating,
      "complex floating":self._complex,
      "integral": self._integral,
      "numeric": self._numeric
    }
    target_dict = {}
    if kind is None:
      target_dict = control["numeric"] | self._boolean
    elif isinstance(kind, tuple):
      target_dict = {}
      for _kind in kind:
        target_dict |= control[_kind]
    else:
      target_dict = control[kind]
    assert info_dict == target_dict

class ArrayAPIErrors(absltest.TestCase):
  """Test that our array API implementations raise errors where required"""

  # TODO(micky774): Remove when jnp.clip deprecation is completed
  # (began 2024-4-2) and default behavior is Array API 2023 compliant
  def test_clip_complex(self):
    x = array_api.arange(5, dtype=array_api.complex64)
    complex_msg = "Complex values have no ordering and cannot be clipped"
    with self.assertRaisesRegex(ValueError, complex_msg):
      array_api.clip(x)

    with self.assertRaisesRegex(ValueError, complex_msg):
      array_api.clip(x, max=x)

    x = array_api.arange(5, dtype=array_api.int32)
    with self.assertRaisesRegex(ValueError, complex_msg):
      array_api.clip(x, min=-1+5j)

    with self.assertRaisesRegex(ValueError, complex_msg):
      array_api.clip(x, max=-1+5j)


if __name__ == '__main__':
  absltest.main()

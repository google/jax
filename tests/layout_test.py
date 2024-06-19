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

import contextlib
import math
import re
from absl.testing import absltest
import numpy as np

import jax
import jax.numpy as jnp
from jax.sharding import NamedSharding, PartitionSpec as P, SingleDeviceSharding
from jax._src import config
from jax._src.layout import Layout, DeviceLocalLayout as DLL
from jax._src import test_util as jtu
from jax._src.util import safe_zip

config.parse_flags_with_absl()

_exit_stack = contextlib.ExitStack()

def setUpModule():
  _exit_stack.enter_context(jtu.set_host_platform_device_count(8))

def tearDownModule():
  _exit_stack.close()


pattern = re.compile(r"\{(.*?):")

# Extract minor_to_major from str(layout) because layout doesn't have a
# minor_to_major property yet.
def extract_minor_to_major(l):
  match = re.search(pattern, str(l))
  return tuple(int(i) for i in match.groups()[0].split(','))


class LayoutTest(jtu.JaxTestCase):

  def setUp(self):
    if not jtu.test_device_matches(['tpu']):
      self.skipTest("Layouts do not work on CPU and GPU backends yet.")
    super().setUp()

  def test_auto_layout(self):
    mesh = jtu.create_global_mesh((2, 2), ('x', 'y'))
    shape1 = (128, 128)
    shape2 = (128, 128)
    s1 = NamedSharding(mesh, P('x', 'y'))
    s2 = NamedSharding(mesh, P('x'))

    def apply(x, y):
      return x.T, y.T

    def init(x, y):
      return x * 2, y * 2

    np_inp1 = np.arange(math.prod(shape1)).reshape(shape1)
    np_inp2 = np.arange(math.prod(shape2)).reshape(shape2)
    sds1 = jax.ShapeDtypeStruct(np_inp1.shape, np_inp1.dtype, sharding=s1)
    sds2 = jax.ShapeDtypeStruct(np_inp2.shape, np_inp2.dtype, sharding=s2)

    lowered_apply = jax.jit(apply, in_shardings=Layout(DLL.AUTO),
                            out_shardings=Layout(DLL.AUTO)).lower(sds1, sds2)
    compiled_apply = lowered_apply.compile()

    arg_layouts, kw_layouts = compiled_apply.input_layouts()
    self.assertEmpty(kw_layouts)

    for i, o in zip(arg_layouts, compiled_apply.output_layouts()):
      self.assertEqual(extract_minor_to_major(i),
                       extract_minor_to_major(o)[::-1])

    init_compiled = jax.jit(
        init, out_shardings=arg_layouts).lower(sds1, sds2).compile()

    for i, o in zip(init_compiled.input_layouts()[0],
                    init_compiled.output_layouts()):
      self.assertEqual(i, o)

    arr1 = jax.device_put(np_inp1, s1)
    arr2 = jax.device_put(np_inp2, s2)

    with jtu.count_aot_jit_cpp_cache_miss() as init_count:
      init_out = init_compiled(arr1, arr2)
      init_compiled(arr1, arr2)
    self.assertEqual(init_count[0], 1)

    self.assertEqual(init_out[0].layout, init_compiled.output_layouts()[0])
    self.assertEqual(init_out[1].layout, init_compiled.output_layouts()[1])

    with jtu.count_aot_jit_cpp_cache_miss() as apply_count:
      apply_out = compiled_apply(*init_out)
      compiled_apply(*init_out)
    self.assertEqual(apply_count[0], 1)

    self.assertEqual(apply_out[0].layout, compiled_apply.output_layouts()[0])
    self.assertEqual(apply_out[1].layout, compiled_apply.output_layouts()[1])

    self.assertTupleEqual(extract_minor_to_major(apply_out[0].layout),
                          extract_minor_to_major(init_out[0].layout)[::-1])
    self.assertTupleEqual(extract_minor_to_major(apply_out[1].layout),
                          extract_minor_to_major(init_out[1].layout)[::-1])

    self.assertArraysEqual(init_out[0], np_inp1 * 2)
    self.assertArraysEqual(init_out[1], np_inp2 * 2)
    self.assertArraysEqual(apply_out[0], (np_inp1 * 2).T)
    self.assertArraysEqual(apply_out[1], (np_inp2 * 2).T)

  def test_default_layout(self):
    mesh = jtu.create_global_mesh((2, 2), ('x', 'y'))
    shape = (4, 4, 2)
    np_inp = np.arange(math.prod(shape)).reshape(shape)
    s = NamedSharding(mesh, P('x', 'y'))
    sds = jax.ShapeDtypeStruct(np_inp.shape, np_inp.dtype, sharding=s)
    arr = jax.device_put(np_inp, s)

    def f(x):
      return x.T

    lowered = jax.jit(f, in_shardings=None, out_shardings=None).lower(sds)
    self.assertIn("default", lowered.as_text())
    compiled = lowered.compile()
    out = compiled(arr)

    self.assertTupleEqual(
        extract_minor_to_major(compiled.input_layouts()[0][0]), (2, 1, 0))
    self.assertTupleEqual(
        extract_minor_to_major(compiled.output_layouts()), (2, 1, 0))
    self.assertArraysEqual(out, np_inp.T)
    self.assertEqual(out.sharding, NamedSharding(mesh, P(None, 'y', 'x')))

    compiled_auto = jax.jit(f, in_shardings=Layout(DLL.AUTO),
                            out_shardings=Layout(DLL.AUTO)).lower(sds).compile()
    self.assertTupleEqual(
        extract_minor_to_major(compiled_auto.input_layouts()[0][0]), (2, 1, 0))
    self.assertTupleEqual(
        extract_minor_to_major(compiled_auto.output_layouts()), (0, 1, 2))

    with self.assertRaisesRegex(
        ValueError, "jax.jit` does not accept device-local layouts directly"):
      jax.jit(f, in_shardings=DLL.AUTO,
              out_shardings=DLL.AUTO).lower(sds).compile()

  def test_in_layouts_out_layouts(self):
    mesh = jtu.create_global_mesh((2, 2), ('x', 'y'))
    shape = (8, 8)
    np_inp = np.arange(math.prod(shape)).reshape(shape)
    s = NamedSharding(mesh, P('x', 'y'))
    arr = jax.device_put(np_inp, s)

    def f(x):
      return x.T

    compiled = jax.jit(f, in_shardings=Layout(),
                       out_shardings=Layout(DLL.AUTO)).lower(arr).compile()
    self.assertTupleEqual(
        extract_minor_to_major(compiled.input_layouts()[0][0]), (1, 0))
    self.assertTupleEqual(
        extract_minor_to_major(compiled.output_layouts()), (0, 1))

    out = compiled(arr)
    self.assertArraysEqual(out, np_inp.T)
    self.assertEqual(out.layout, compiled.output_layouts())
    self.assertEqual(out.sharding, NamedSharding(mesh, P('y', 'x')))

  def test_sharding_and_layouts(self):
    mesh = jtu.create_global_mesh((2, 2), ('x', 'y'))
    shape = (4, 8)
    np_inp = np.arange(math.prod(shape)).reshape(shape)
    s = NamedSharding(mesh, P('x', 'y'))

    compiled = jax.jit(lambda x: x.T, in_shardings=Layout(DLL.AUTO, s),
                       out_shardings=Layout(DLL.AUTO, s)).lower(np_inp).compile()
    out = compiled(np_inp)
    self.assertTupleEqual(
        extract_minor_to_major(compiled.input_layouts()[0][0]), (1, 0))
    self.assertTupleEqual(
        extract_minor_to_major(compiled.output_layouts()), (0, 1))
    self.assertArraysEqual(out, np_inp.T)
    self.assertEqual(out.sharding, s)

  def test_dce_in_layouts(self):
    def f(x, y, z, a, b, c):
      return z * 2, b.T

    shape = (8, 2)
    inps = [np.arange(math.prod(shape)).reshape(shape)] * 6
    compiled = jax.jit(f, in_shardings=Layout(DLL.AUTO),
                       out_shardings=Layout(DLL.AUTO)).lower(*inps).compile()
    arg_layouts, _ = compiled.input_layouts()
    out1, out2 = compiled(*inps)

    compiled2 = jax.jit(f, in_shardings=arg_layouts).lower(*inps).compile()
    out3, out4 = compiled2(*inps)

    for l1, l2 in safe_zip(arg_layouts, compiled2.input_layouts()[0]):
      self.assertEqual(l1, l2)

    self.assertArraysEqual(out1, out3)
    self.assertArraysEqual(out2, out4)

    arrs = [jax.device_put(i, l) for i, l in zip(inps, arg_layouts)]
    out5, out6 = jax.jit(f)(*arrs)
    self.assertArraysEqual(out1, out5)
    self.assertArraysEqual(out2, out6)

  def test_no_error_dced_args(self):
    mesh = jtu.create_global_mesh((2, 1), ('x', 'y'))
    shape = (8, 2)
    s = NamedSharding(mesh, P('x', 'y'))
    np_inp = np.arange(math.prod(shape)).reshape(shape)
    arr1 = jax.device_put(np_inp, s)
    arr2 = jax.device_put(np_inp, s)
    arrs = [arr1, arr2]

    def f(x, y):
      return x * 2

    jf = jax.jit(f, in_shardings=Layout(DLL.AUTO, s),
                 out_shardings=Layout(DLL.AUTO, s))
    compiled = jf.lower(np_inp, np_inp).compile()
    arg_layouts, _ = compiled.input_layouts()
    arrs = [jax.device_put(i, l) for i, l in zip(arrs, arg_layouts)]
    compiled(*arrs)

  def test_aot_layout_mismatch(self):
    mesh = jtu.create_global_mesh((2, 2), ('x', 'y'))
    shape = (256, 4, 2)
    np_inp = np.arange(math.prod(shape)).reshape(shape)
    s = NamedSharding(mesh, P('x'))

    sds = jax.ShapeDtypeStruct(np_inp.shape, np_inp.dtype, sharding=s)
    arr = jax.device_put(np_inp, s)

    def f(x):
      return (x * 2).T

    with self.assertRaisesRegex(
        ValueError,
        'Layout passed to jit does not match the layout on the respective arg'):
      jax.jit(f, in_shardings=Layout(DLL.AUTO)).lower(arr)

    compiled = jax.jit(f, in_shardings=Layout(DLL.AUTO),
                       out_shardings=Layout(DLL.AUTO)).lower(sds).compile()

    with self.assertRaisesRegex(
        ValueError,
        r'Compiled object called with input layout\(s\) does'
        r' not match the layout\(s\) the computation was'
        ' compiled with'):
      compiled(arr)

  @jtu.ignore_warning(category=DeprecationWarning,
                      message="backend and device argument")
  def test_cpu_default_backend_layout(self):
    inp = jax.device_put(np.ones((8, 8)), device=jax.devices('cpu')[0])
    out_cpu = jax.jit(jnp.dot)(inp, inp)

    jax.jit(jnp.dot, backend=jax.default_backend()).lower(
        out_cpu, out_cpu).compile()  # doesn't crash

  def test_device_put_concrete_layout(self):
    mesh = jtu.create_global_mesh((2, 2), ('x', 'y'))
    shape = (8, 128)
    np_inp = np.arange(math.prod(shape)).reshape(shape)
    s = NamedSharding(mesh, P('x', 'y'))
    arr = jax.device_put(np_inp, s)

    compiled = jax.jit(
        lambda x: x * 2, out_shardings=Layout(DLL.AUTO)).lower(arr).compile()
    col = compiled.output_layouts()

    out = jax.device_put(np_inp, col)
    self.assertEqual(out.layout, col)
    self.assertArraysEqual(out, np_inp)
    for s in out.addressable_shards:
      self.assertEqual(out.layout.device_local_layout,
                       s.data.layout.device_local_layout)

  def test_device_put_non_concrete_layout_error(self):
    np_inp = np.arange(16).reshape(8, 2)

    l1 = Layout(DLL.AUTO, SingleDeviceSharding(jax.devices()[0]))
    with self.assertRaisesRegex(
        ValueError, 'sharding and device_local_layout.*should be concrete'):
      jax.device_put(np_inp, l1)

    l2 = Layout(DLL.AUTO)
    with self.assertRaisesRegex(
        ValueError, 'sharding and device_local_layout.*should be concrete'):
      jax.device_put(np_inp, l2)

    l3 = Layout(None, SingleDeviceSharding(jax.devices()[0]))
    out = jax.device_put(np_inp, l3)
    self.assertArraysEqual(out, np_inp)
    self.assertTrue(out._committed)

  def invalid_layout_spec(self):
    x = np.arange(8)
    compiled = jax.jit(lambda x: x).lower(x).compile()
    with self.assertRaisesRegex(
        ValueError, 'Sharding has to be concrete when layout.*'):
      Layout(compiled.output_layouts()[0], None)

  def test_layout_on_sds(self):
    mesh = jtu.create_global_mesh((2, 1), ('x', 'y'))
    s = NamedSharding(mesh, P('x', 'y'))
    np_inp = np.arange(16).reshape(8, 2)
    arr = jax.device_put(np_inp, s)

    out_layout = jax.jit(jnp.sin, out_shardings=Layout(DLL.AUTO)).lower(
        arr).compile().output_layouts()

    sds = jax.ShapeDtypeStruct(arr.shape, arr.dtype, sharding=out_layout)
    arg_layout, _ = jax.jit(lambda x: x * 2).lower(sds).compile().input_layouts()
    self.assertEqual(arg_layout[0], out_layout)

    with self.assertRaisesRegex(
        TypeError,
        'DeviceLocalLayout.AUTO` cannot be used in place of a device-local'
        ' layout in a `ShapeDtypeStruct`'):
      jax.ShapeDtypeStruct(arr.shape, arr.dtype, sharding=Layout(DLL.AUTO))

  def test_make_array_from_callback(self):
    mesh = jtu.create_global_mesh((2, 1), ('x', 'y'))
    s = NamedSharding(mesh, P('x', 'y'))
    np_inp = np.arange(16).reshape(8, 2)
    sds = jax.ShapeDtypeStruct(np_inp.shape, np_inp.dtype, sharding=s)

    layout = jax.jit(lambda x: x * 2).lower(sds).compile().output_layouts()

    out = jax.make_array_from_callback(np_inp.shape, layout,
                                       lambda idx: np_inp[idx])
    self.assertArraysEqual(out, np_inp)
    self.assertEqual(out.layout, layout)

    with self.assertRaisesRegex(
        TypeError,
        '`DeviceLocalLayout.AUTO` cannot be used in place of a device-local'
        ' layout'):
      jax.make_array_from_callback(np_inp.shape, Layout(DLL.AUTO, s),
                                   lambda idx: np_inp[idx])

    with self.assertRaisesRegex(
        TypeError, 'sharding should be an instance of `jax.sharding`'):
      jax.make_array_from_callback(
          np_inp.shape, Layout(None, None), lambda idx: np_inp[idx])


if __name__ == '__main__':
  absltest.main(testLoader=jtu.JaxTestLoader())

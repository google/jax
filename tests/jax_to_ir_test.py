# Copyright 2019 The JAX Authors.
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

import unittest

from absl.testing import absltest
import jax.numpy as jnp
from jax.tools import jax_to_ir
from jax._src import test_util as jtu

try:
  import tensorflow as tf
except ImportError:
  tf = None  # type: ignore


def axpy(a, x, y):
  return a * x + y[:, jnp.newaxis]


class JaxToIRTest(absltest.TestCase):

  def test_jax_to_hlo_axpy(self):
    hlo_proto, hlo_text = jax_to_ir.jax_to_hlo(axpy, [
        ('y', jax_to_ir.parse_shape_str('f32[128]')),
        ('a', jax_to_ir.parse_shape_str('f32[]')),
        ('x', jax_to_ir.parse_shape_str('f32[128,2]')),
    ])

    # Check that hlo_text contains a broadcast, add, and multiply.
    self.assertIn('broadcast', hlo_text)
    self.assertIn('add', hlo_text)
    self.assertIn('multiply', hlo_text)

    # Check that the HLO parameters are in the order we specified in the
    # jax_to_hlo call.
    self.assertIn('f32[128]{0} parameter(0)', hlo_text)
    self.assertIn('f32[] parameter(1)', hlo_text)
    self.assertIn('f32[128,2]{1,0} parameter(2)', hlo_text)

    # Check that the parameters are in the expected order.

    # TODO(jlebar): Ideally we'd check that hlo_proto can be deserialized to a
    # valid HLO proto, but we don't seem to have access to hlo_pb2 at the
    # moment, so the best we seem to be able to do is check that it's nonempty.
    assert hlo_proto

  def test_jax_to_hlo_with_constants(self):

    def fn(a, b, x, y):
      return a / b * x + y

    _, hlo_text = jax_to_ir.jax_to_hlo(
        fn,
        input_shapes=[
            ('x', jax_to_ir.parse_shape_str('f32[128]')),
            ('y', jax_to_ir.parse_shape_str('f32[128]')),
        ],
        constants={
            'a': 123456,
            'b': 4,
        })
    # Because we passed `a` and `b` as constants, they get constant-folded away
    # by Python/JAX to a/b = 30864.
    self.assertIn('constant(30864)', hlo_text)
    self.assertNotIn('123456', hlo_text)

  def test_parse_shape_str_invalid(self):
    with self.assertRaisesRegex(ValueError, 'Invalid shape.*foo'):
      jax_to_ir.parse_shape_str('foo[]')

  @unittest.skipIf(tf is None, 'TensorFlow not installed.')
  def test_jax_to_tf_axpy(self):
    tf_proto, tf_text = jax_to_ir.jax_to_tf(axpy, [
        ('y', jax_to_ir.parse_shape_str('f32[128]')),
        ('a', jax_to_ir.parse_shape_str('f32[]')),
        ('x', jax_to_ir.parse_shape_str('f32[128,2]')),
    ])

    # Check that tf debug txt contains a broadcast, add, and multiply.
    self.assertIn('BroadcastTo', tf_text)
    self.assertIn('AddV2', tf_text)
    self.assertIn('Mul', tf_text)

    # Check that we can re-import our graphdef.
    gdef = tf.compat.v1.GraphDef()
    gdef.ParseFromString(tf_proto)
    g = tf.Graph()
    with g.as_default():
      tf.import_graph_def(gdef, name='')

    # Check that the HLO parameters are named as we specified.
    ops = {o.name: o for o in g.get_operations()
           if o.name in ('y', 'a', 'x', 'jax2tf_out')}
    self.assertLen(ops, 4)
    self.assertIdentityOp(ops['y'], [128], jnp.float32)
    self.assertIdentityOp(ops['a'], [], jnp.float32)
    self.assertIdentityOp(ops['x'], [128, 2], jnp.float32)
    self.assertIdentityOp(ops['jax2tf_out'], [128, 2], jnp.float32)

  def assertIdentityOp(self, op, expected_shape, expected_dtype):
    self.assertEqual(op.type, 'Identity')
    output, = op.outputs
    self.assertEqual(output.shape, expected_shape)
    self.assertEqual(output.dtype, expected_dtype)

  def test_parse_shape_str(self):
    self.assertParsedShape('f32[]', [], jnp.float32)
    self.assertParsedShape('f32[1,2,3]', [1, 2, 3], jnp.float32)
    self.assertParsedShape('pred[1]', [1], jnp.bool_)
    if hasattr(jnp, 'int2'):
      self.assertParsedShape('s2[1]', [1], jnp.int2)
    self.assertParsedShape('s4[1]', [1], jnp.int4)
    self.assertParsedShape('s8[1]', [1], jnp.int8)
    self.assertParsedShape('s16[1]', [1], jnp.int16)
    self.assertParsedShape('s32[1]', [1], jnp.int32)
    self.assertParsedShape('s64[1]', [1], jnp.int64)
    if hasattr(jnp, 'uint2'):
      self.assertParsedShape('u2[1]', [1], jnp.uint2)
    self.assertParsedShape('u4[1]', [1], jnp.uint4)
    self.assertParsedShape('u8[1]', [1], jnp.uint8)
    self.assertParsedShape('u16[1]', [1], jnp.uint16)
    self.assertParsedShape('u32[1]', [1], jnp.uint32)
    self.assertParsedShape('u64[1]', [1], jnp.uint64)
    self.assertParsedShape('f16[1]', [1], jnp.float16)
    self.assertParsedShape('f32[1]', [1], jnp.float32)
    self.assertParsedShape('f64[1]', [1], jnp.float64)
    self.assertParsedShape('bf16[1]', [1], jnp.bfloat16)
    self.assertParsedShape('c64[1]', [1], jnp.complex64)
    self.assertParsedShape('c128[1]', [1], jnp.complex128)

  def assertParsedShape(self, s: str, expected_shape, expected_dtype):
    p = jax_to_ir.parse_shape_str(s)
    self.assertEqual(p.shape, tuple(expected_shape))
    self.assertEqual(p.dtype, expected_dtype)


if __name__ == '__main__':
  absltest.main(testLoader=jtu.JaxTestLoader())

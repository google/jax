# Copyright 2022 The JAX Authors.
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
"""Converters for jax2tf."""

from collections.abc import Callable
import dataclasses
import functools
import tempfile
from typing import Any

from jax.experimental import jax2tf
import tensorflow as tf
import tensorflowjs as tfjs

from jax.experimental.jax2tf.tests.model_harness import ModelHarness


@dataclasses.dataclass
class Converter:
  name: str
  convert_fn: Callable[..., Any]
  compare_numerics: bool = True


def jax2tf_convert(harness: ModelHarness, enable_xla: bool = True):
  return jax2tf.convert(
      harness.apply_with_vars,
      enable_xla=enable_xla,
      polymorphic_shapes=harness.polymorphic_shapes)


def jax2tfjs(harness: ModelHarness):
  """Converts the given `test_case` using the TFjs converter."""
  with tempfile.TemporaryDirectory() as model_dir:
    tfjs.converters.convert_jax(
        apply_fn=harness.apply,
        params=harness.variables,
        input_signatures=harness.tf_input_signature,
        polymorphic_shapes=harness.polymorphic_shapes,
        model_dir=model_dir)


def jax2tflite(harness: ModelHarness, use_flex_ops: bool = False):
  """Returns a converter with Flex ops linked in iff `use_flex_ops==True`."""
  tf_fn = tf.function(
      jax2tf_convert(harness, enable_xla=False),
      input_signature=harness.tf_input_signature,
      autograph=False)
  apply_tf = tf_fn.get_concrete_function()
  converter = tf.lite.TFLiteConverter.from_concrete_functions([apply_tf], tf_fn)
  supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS]
  if use_flex_ops:
    supported_ops.append(tf.lite.OpsSet.SELECT_TF_OPS)
  converter.target_spec.supported_ops = supported_ops

  # Convert the model.
  tflite_model = converter.convert()

  # Construct an interpreter for doing a numerical comparison.
  interpreter = tf.lite.Interpreter(model_content=tflite_model)
  interpreter.allocate_tensors()

  inputs = interpreter.get_input_details()
  output_details = interpreter.get_output_details()
  outputs = tuple(interpreter.tensor(out["index"]) for out in output_details)

  def apply_tflite(*xs):
    assert len(xs) == len(inputs)
    for i, x in enumerate(xs):
      interpreter.set_tensor(inputs[i]['index'], x)
    interpreter.invoke()
    if len(outputs) > 1:
      return tuple(o() for o in outputs)
    else:
      return outputs[0]()

  return apply_tflite


ALL_CONVERTERS = [
    # jax2tf with XLA support (enable_xla=True).
    Converter(name='jax2tf_xla', convert_fn=jax2tf_convert),
    # jax2tf without XLA support (enable_xla=False).
    Converter(
        name='jax2tf_noxla',
        convert_fn=functools.partial(jax2tf_convert, enable_xla=False)),
    # Convert JAX to Tensorflow.JS.
    Converter(name='jax2tfjs', convert_fn=jax2tfjs, compare_numerics=False),
    # Convert JAX to TFLIte.
    Converter(name='jax2tflite', convert_fn=jax2tflite),
    # Convert JAX to TFLIte with support for Flex ops.
    Converter(
        name='jax2tflite+flex',
        convert_fn=functools.partial(jax2tflite, use_flex_ops=True))
]

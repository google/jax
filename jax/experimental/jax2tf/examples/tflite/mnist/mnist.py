# Copyright 2020 The JAX Authors.
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

import logging
from absl import app
from absl import flags

from jax.experimental import jax2tf
from jax.experimental.jax2tf.examples import mnist_lib

import numpy as np

import tensorflow as tf
import tensorflow_datasets as tfds  # type: ignore[import-not-found]

_TFLITE_FILE_PATH = flags.DEFINE_string(
    'tflite_file_path',
    '/tmp/mnist.tflite',
    'Path where to save the TensorFlow Lite file.',
)
_SERVING_BATCH_SIZE = flags.DEFINE_integer(
    'serving_batch_size',
    4,
    'For what batch size to prepare the serving signature. ',
)
_NUM_EPOCHS = flags.DEFINE_integer(
    'num_epochs', 10, 'For how many epochs to train.'
)


# A helper function to evaluate the TF Lite model using "test" dataset.
def evaluate_tflite_model(tflite_model, test_ds):
  # Initialize TFLite interpreter using the model.
  interpreter = tf.lite.Interpreter(model_content=tflite_model)
  interpreter.allocate_tensors()
  input_tensor_index = interpreter.get_input_details()[0]['index']
  output = interpreter.tensor(interpreter.get_output_details()[0]['index'])

  # Run predictions on every image in the "test" dataset.
  prediction_digits = []
  labels = []
  for image, one_hot_label in test_ds:
    interpreter.set_tensor(input_tensor_index, image)

    # Run inference.
    interpreter.invoke()

    # Post-processing: for each batch dimension and find the digit with highest
    # probability.
    digits = np.argmax(output(), axis=1)
    prediction_digits.extend(digits)
    labels.extend(np.argmax(one_hot_label, axis=1))

  # Compare prediction results with ground truth labels to calculate accuracy.
  accurate_count = 0
  for index in range(len(prediction_digits)):
    if prediction_digits[index] == labels[index]:
      accurate_count += 1
  accuracy = accurate_count * 1.0 / len(prediction_digits)
  return accuracy


def main(_):
  logging.info('Loading the MNIST TensorFlow dataset')
  train_ds = mnist_lib.load_mnist(
      tfds.Split.TRAIN, batch_size=mnist_lib.train_batch_size)
  test_ds = mnist_lib.load_mnist(
      tfds.Split.TEST, batch_size=_SERVING_BATCH_SIZE)

  (flax_predict, flax_params) = mnist_lib.FlaxMNIST.train(
      train_ds, test_ds, _NUM_EPOCHS.value
  )

  def predict(image):
    return flax_predict(flax_params, image)

  # Convert Flax model to TF function.
  tf_predict = tf.function(
      jax2tf.convert(predict, enable_xla=False),
      input_signature=[
          tf.TensorSpec(
              shape=[_SERVING_BATCH_SIZE, 28, 28, 1],
              dtype=tf.float32,
              name='input')
      ],
      autograph=False)

  # Convert TF function to TF Lite format.
  converter = tf.lite.TFLiteConverter.from_concrete_functions(
      [tf_predict.get_concrete_function()], tf_predict)
  converter.target_spec.supported_ops = [
      tf.lite.OpsSet.TFLITE_BUILTINS,  # enable TensorFlow Lite ops.
      tf.lite.OpsSet.SELECT_TF_OPS  # enable TensorFlow ops.
  ]
  tflite_float_model = converter.convert()

  # Show model size in KBs.
  float_model_size = len(tflite_float_model) / 1024
  print('Float model size = %dKBs.' % float_model_size)

  # Re-convert the model to TF Lite using quantization.
  converter.optimizations = [tf.lite.Optimize.DEFAULT]
  tflite_quantized_model = converter.convert()

  # Show model size in KBs.
  quantized_model_size = len(tflite_quantized_model) / 1024
  print('Quantized model size = %dKBs,' % quantized_model_size)
  print('which is about %d%% of the float model size.' %
        (quantized_model_size * 100 / float_model_size))

  # Evaluate the TF Lite float model. You'll find that its accuracy is identical
  # to the original Flax model because they are essentially the same model
  # stored in different format.
  float_accuracy = evaluate_tflite_model(tflite_float_model, test_ds)
  print('Float model accuracy = %.4f' % float_accuracy)

  # Evalualte the TF Lite quantized model.
  # Don't be surprised if you see quantized model accuracy is higher than
  # the original float model. It happens sometimes :)
  quantized_accuracy = evaluate_tflite_model(tflite_quantized_model, test_ds)
  print('Quantized model accuracy = %.4f' % quantized_accuracy)
  print('Accuracy drop = %.4f' % (float_accuracy - quantized_accuracy))

  f = open(_TFLITE_FILE_PATH.value, 'wb')
  f.write(tflite_quantized_model)
  f.close()


if __name__ == '__main__':
  app.run(main)

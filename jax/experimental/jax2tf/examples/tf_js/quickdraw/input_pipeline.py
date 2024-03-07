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
from jax import numpy as jnp

import numpy as np
import os
import requests

def download_dataset(dir_path, nb_classes):

  if not os.path.exists(dir_path):
    os.makedirs(dir_path)
  assert os.path.isdir(dir_path), f"{dir_path} exists and is not a directory"

  classes_path = os.path.join(
    os.path.dirname(__file__),
    'third_party/zaidalyafeai.github.io/class_names.txt')
  with open(classes_path) as classes_file:
    classes = (
      list(map(lambda c: c.strip(), classes_file.readlines()))[:nb_classes])

  url = 'https://storage.googleapis.com/quickdraw_dataset/full/numpy_bitmap/'
  for cls in classes:
    cls_filename = f"{cls}.npy"
    cls_file_path = os.path.join(dir_path, cls_filename)
    if os.path.exists(cls_file_path):
      print(f'{cls_filename} already exists, skipping')
      continue
    with open(cls_file_path, "wb") as save_file:
      try:
        response = requests.get(url + cls_filename.replace('_', ' '))
        save_file.write(response.content)
        print(f'Successfully fetched {cls_filename}')
      except:
        print(f'Failed to fetch {cls_filename}')
  return classes

def get_datasets(dir_path, classes, batch_size=256, test_ratio=0.1,
                 max_items_per_class=4096):
  x, y = np.empty([0, 784]), np.empty([0])
  for idx, cls in enumerate(classes):
    cls_path = os.path.join(dir_path, f"{cls}.npy")
    data = np.load(cls_path)[:max_items_per_class, :]
    labels = np.full(data.shape[0], idx)
    x, y = np.concatenate((x, data), axis=0), np.append(y, labels)

  assert x.shape[0] % batch_size == 0

  x, y = x.astype(jnp.float32) / 255.0, y.astype(jnp.int32)
  # Reshaping to square images
  x = np.reshape(x, (x.shape[0], 28, 28, 1))
  permutation = np.random.permutation(y.shape[0])
  x = x[permutation, :]
  y = y[permutation]

  x = np.reshape(x, [x.shape[0] // batch_size, batch_size] + list(x.shape[1:]))
  y = np.reshape(y, [y.shape[0] // batch_size, batch_size])

  nb_test_elements = int(x.shape[0] * test_ratio)

  x_test, y_test = x[:nb_test_elements], y[:nb_test_elements]
  x_train, y_train = x[nb_test_elements:], y[nb_test_elements:]

  return list(zip(x_train, y_train)), list(zip(x_test, y_test))

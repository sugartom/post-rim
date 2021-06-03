# Copyright 2016 Google Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

#!/usr/bin/env python2.7

"""A client that talks to tensorflow_model_server loaded with mnist model.

The client downloads test images of mnist data set, queries the service with
such test images to get predictions, and calculates the inference error rate.

Typical usage example:

    mnist_client.py --num_tests=100 --server=localhost:9000
"""

from __future__ import print_function

import sys
import threading

# This is a placeholder for a Google-internal import.

import time
import grpc
import numpy
import tensorflow as tf

from tensorflow_serving.apis import predict_pb2
from tensorflow_serving.apis import prediction_service_pb2_grpc
import mnist_input_data
from tensorflow.python.framework import tensor_util

tf.app.flags.DEFINE_integer('concurrency', 1,
                            'maximum number of concurrent inference requests')
tf.app.flags.DEFINE_integer('num_tests', 8192, 'Number of test images')
tf.app.flags.DEFINE_string('server', 'localhost:8500', 'PredictionService host:port')
tf.app.flags.DEFINE_string('work_dir', '/tmp', 'Working directory. ')
FLAGS = tf.app.flags.FLAGS


class _ResultCounter(object):
  """Counter for the prediction results."""

  def __init__(self, num_tests, concurrency):
    self._num_tests = num_tests
    self._concurrency = concurrency
    self._error = 0
    self._done = 0
    self._active = 0
    self._condition = threading.Condition()

  def inc_error(self):
    with self._condition:
      self._error += 1

  def inc_done(self):
    with self._condition:
      self._done += 1
      self._condition.notify()

  def dec_active(self):
    with self._condition:
      self._active -= 1
      self._condition.notify()

  def get_error_rate(self):
    with self._condition:
      while self._done != self._num_tests:
        self._condition.wait()
      return self._error / float(self._num_tests)

  def throttle(self):
    with self._condition:
      while self._active == self._concurrency:
        self._condition.wait()
      self._active += 1

def do_inference(hostport, work_dir, concurrency, batch_size, num_tests):
  """Tests PredictionService with concurrent requests.

  Args:
    hostport: Host:port address of the PredictionService.
    work_dir: The full path of working directory for test data set.
    concurrency: Maximum number of concurrent requests.
    num_tests: Number of test images to use.

  Returns:
    The classification error rate.

  Raises:
    IOError: An error occurred processing test data set.
  """
  test_data_set = mnist_input_data.read_data_sets(work_dir).test
  
  channel = grpc.insecure_channel(hostport)
  stub = prediction_service_pb2_grpc.PredictionServiceStub(channel)

  result_counter = _ResultCounter(num_tests, concurrency)
  
  for _ in range(num_tests / batch_size):
    request = predict_pb2.PredictRequest()
    request.model_spec.name = 'mnist'
    request.model_spec.signature_name = 'predict_images'
    image, label = test_data_set.next_batch(batch_size)

    request.inputs['images'].CopyFrom(
        tf.contrib.util.make_tensor_proto(image, shape=image.shape))

    result = stub.Predict(request, 10.0)
    response = tensor_util.MakeNdarray(result.outputs['scores'])

    for i in range(len(label)):
      this_label = label[i]
      this_prediction = numpy.argmax(response[i])
      if this_label != this_prediction:
        result_counter.inc_error()
      result_counter.inc_done()
      result_counter.dec_active()

  return result_counter.get_error_rate()



def main(_):
  batch_size = int(sys.argv[1])

  if FLAGS.num_tests > 10000:
    print('num_tests should not be greater than 10k')
    return

  # do_inference(FLAGS.server, FLAGS.work_dir, FLAGS.concurrency, batch_size, FLAGS.num_tests)

  t0 = time.time()
  error_rate = do_inference(FLAGS.server, FLAGS.work_dir, FLAGS.concurrency, batch_size, FLAGS.num_tests)
  t1 = time.time()
  print("duration = %.3f sec" % (t1 - t0))
  print('\nInference error rate: %s%%' % (error_rate * 100))


if __name__ == '__main__':
  tf.app.run()

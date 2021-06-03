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
tf.app.flags.DEFINE_integer('num_tests', 32768, 'Number of test images')
tf.app.flags.DEFINE_string('server', 'localhost:8500', 'PredictionService host:port')
tf.app.flags.DEFINE_string('work_dir', '/tmp', 'Working directory. ')
FLAGS = tf.app.flags.FLAGS

def do_inference(hostport, batch_size, run_num, image, label, thread_id):
  durationSum = 0.0

  # # test_data_set = mnist_input_data.read_data_sets(work_dir).test
  # test_data_set = mnist_input_data.read_data_sets(work_dir).train

  channel = grpc.insecure_channel(hostport)
  stub = prediction_service_pb2_grpc.PredictionServiceStub(channel)
  
  for i in range(run_num):
    start = time.time()
    request = predict_pb2.PredictRequest()
    request.model_spec.name = 'mnist'
    request.model_spec.signature_name = 'predict_images'
    # image, label = test_data_set.next_batch(batch_size)

    request.inputs['images'].CopyFrom(
        tf.contrib.util.make_tensor_proto(image, shape=image.shape))

    result = stub.Predict(request, 10.0)
    end = time.time()
    duration = end - start
    durationSum += duration
    # print("[thread %d] for run %d, duration = %f" % (thread_id, i, duration))

  # print("[thread %d] average duration for batch size of %d = %f" % (thread_id, batch_size, durationSum / run_num))

def main(_):
  # thread_num_array = [1, 2, 4, 8, 16, 32, 64, 128]
  # batch_size_2d_array = [[1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024],
  #                        [1, 2, 4, 8, 16, 32, 64, 128, 256, 512],
  #                        [1, 2, 4, 8, 16, 32, 64, 128, 256],
  #                        [1, 2, 4, 8, 16, 32, 64, 128],
  #                        [1, 2, 4, 8, 16, 32, 64],
  #                        [1, 2, 4, 8, 16, 32],
  #                        [1, 2, 4, 8, 16],
  #                        [1, 2, 4, 8]]

  thread_num_array = [1, 2]
  batch_size_2d_array = [[1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024],
                          [1, 2 ,4, 8, 16, 32, 64, 128, 256, 512]]

  test_data_set = mnist_input_data.read_data_sets(FLAGS.work_dir).train

  for i in range(len(thread_num_array)):
    thread_num = thread_num_array[i]
    batch_size_arrary = batch_size_2d_array[i]
    for batch_size in batch_size_arrary:
      time.sleep(2.0)

      image, label = test_data_set.next_batch(batch_size)

      run_num = FLAGS.num_tests / (thread_num * batch_size)
      # print("[INFO] thread_num = %d, batch_size = %d, run_num = %d" % (thread_num, batch_size, run_num))

      # warmup
      warmup_num = 3
      for i in range(warmup_num):
        start = time.time()
        do_inference(FLAGS.server, batch_size, 1, image, label, 0)
        end = time.time()
        duration = end - start
        # print("warmup duration = %f" % duration)

      # raw_input("Press Enter to continue...")

      start = time.time()

      thread_pool = []
      for i in range(thread_num):
        t = threading.Thread(target = do_inference, args = (FLAGS.server, batch_size, run_num, image, label, i))
        thread_pool.append(t)
        t.start()

      for t in thread_pool:
        t.join()

      end = time.time()
      print("overall duration for b = %3d, p = %3d is %.3f" % (batch_size, thread_num, end - start))

if __name__ == '__main__':
  tf.app.run()

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

"""Send JPEG image to tensorflow_model_server loaded with inception model.
"""

from __future__ import print_function

# This is a placeholder for a Google-internal import.

import grpc
import tensorflow as tf

from tensorflow_serving.apis import predict_pb2
from tensorflow_serving.apis import prediction_service_pb2_grpc

import sys
sys.path.append('/home/yitao/Documents/fun-project/tensorflow-related/post-rim/test/test_ssd/')
from image_preprocessor import decode_image_opencv

tf.app.flags.DEFINE_string('server', 'localhost:8500',
                           'PredictionService host:port')
tf.app.flags.DEFINE_string('image', '/home/yitao/Documents/fun-project/tensorflow-related/post-rim/test/test_ssd/lion.jpg', 'path to image in JPEG format')
FLAGS = tf.app.flags.FLAGS

def send_request(model_name, stub, inputs, run_num, batch_size, client_id):
  durationSum = 0.0

  request = predict_pb2.PredictRequest()    
  request.model_spec.name = model_name
  request.model_spec.signature_name = 'serving_default'

  request.inputs['inputs'].CopyFrom(tf.contrib.util.make_tensor_proto(inputs, shape=inputs.shape))

  for i in range(run_num):

    start = time.time()
    result = stub.Predict(request, 10.0)
    end = time.time()
    duration = end - start
    durationSum += duration
    # print("duration = %f" % duration)

  print("[client %d] average duration for batch size of %d = %f" % (client_id, batch_size, durationSum / run_num))

def main(_):
  channel = grpc.insecure_channel(FLAGS.server)
  stub = prediction_service_pb2_grpc.PredictionServiceStub(channel)

  # model_name = 'ssd_inception_v2_coco'
  model_name = 'ssd_resnet50'
  # model_name = 'ssd_mobilenet'
  
  thread_num = 1
  run_num = 100
  batch_size = 1

  image, org = decode_image_opencv(FLAGS.image)
  image = image.astype(np.uint8)
  inputs = image
  for i in range(batch_size - 1):
    inputs = np.append(inputs, image, axis = 0)

  request = predict_pb2.PredictRequest()    
  request.model_spec.name = model_name
  request.model_spec.signature_name = 'serving_default'
  request.inputs['inputs'].CopyFrom(tf.contrib.util.make_tensor_proto(inputs, shape=inputs.shape))

  # warmup
  warmup_num = 3
  for i in range(warmup_num):

    start = time.time()
    result = stub.Predict(request, 10.0)
    end = time.time()
    duration = end - start
    print("warmup duration = %f" % duration)
  time.sleep(2.0)

  start = time.time()

  thread_pool = []
  for i in range(thread_num):
    t = threading.Thread(target = send_request, args = (model_name, stub, inputs, run_num, batch_size, i,))
    thread_pool.append(t)
    t.start()

  for t in thread_pool:
    t.join()

  end = time.time()
  print("overall time = %f" % (end - start))
  
  # # Send request
  # with open(FLAGS.image, 'rb') as f:
  #   # See prediction_service.proto for gRPC request/response details.
  #   data = f.read()
  #   request = predict_pb2.PredictRequest()
  #   request.model_spec.name = 'inception'
  #   request.model_spec.signature_name = 'predict_images'
  #   request.inputs['images'].CopyFrom(
  #       tf.contrib.util.make_tensor_proto(data, shape=[1]))
  #   result = stub.Predict(request, 10.0)  # 10 secs timeout
  #   print(result)

if __name__ == '__main__':
  tf.app.run()
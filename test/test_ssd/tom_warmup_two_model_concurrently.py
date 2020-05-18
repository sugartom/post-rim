# https://towardsdatascience.com/productising-tensorflow-keras-models-via-tensorflow-serving-69e191cb1f37
# python -m grpc.tools.protoc --python_out=. --grpc_python_out=. -I. string_int_label_map.proto

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

import cv2
import numpy as np
import grpc
import time

import threading

# import os
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
# tf.logging.set_verbosity(tf.logging.ERROR)

from tensorflow_serving.apis import predict_pb2
from tensorflow_serving.apis import prediction_service_pb2_grpc

import string_int_label_map_pb2 as labelmap
from google.protobuf import text_format

from image_preprocessor import decode_image_opencv

tf.app.flags.DEFINE_string('server', 'localhost:8500',
                           'PredictionService host:port')
tf.app.flags.DEFINE_string('image', '/home/yitao/Documents/fun-project/tensorflow-related/post-rim/test/test_ssd/lion.jpg', 'path to image in JPEG format')
FLAGS = tf.app.flags.FLAGS

def box_normal_to_pixel(box, dim, scalefactor = 1):
  # https://github.com/tensorflow/models/blob/master/research/object_detection/utils/visualization_utils.py
  height, width = dim[0], dim[1]
  ymin = int(box[0] * height * scalefactor)
  xmin = int(box[1] * width * scalefactor)

  ymax = int(box[2] * height * scalefactor)
  xmax= int(box[3] * width * scalefactor)
  return np.array([xmin, ymin, xmax, ymax])   

def get_label(index):
  global _label_map
  return _label_map.item[index].display_name

def send_request(stub, inputs, model_name, run_num):
  request = predict_pb2.PredictRequest()    
  request.model_spec.name = model_name
  request.model_spec.signature_name = 'serving_default'
  request.inputs['inputs'].CopyFrom(tf.contrib.util.make_tensor_proto(inputs, shape=inputs.shape))

  for i in range(run_num):
    start = time.time()
    result = stub.Predict(request, 60.0)
    end = time.time()
    duration = end - start
    print("[%s] for run %d, duration = %f" % (model_name, i, duration))

def main(_):
  s = open('mscoco_complete_label_map.pbtxt', 'r').read()
  mymap = labelmap.StringIntLabelMap()
  global _label_map 
  _label_map = text_format.Parse(s, mymap)

  channel = grpc.insecure_channel(FLAGS.server)
  stub = prediction_service_pb2_grpc.PredictionServiceStub(channel)

  image, org = decode_image_opencv(FLAGS.image)
  image = image.astype(np.uint8)

  model_1 = 'ssd_inception_v2_coco'
  model_2 = 'ssd_mobilenet'

  # thread_pool = []  
  t1 = threading.Thread(target = send_request, args = (stub, image, model_1, 10, ))
  t2 = threading.Thread(target = send_request, args = (stub, image, model_2, 1, ))

  t1.start()
  t2.start()

  t1.join()
  t2.join()

if __name__ == '__main__':
  tf.app.run()
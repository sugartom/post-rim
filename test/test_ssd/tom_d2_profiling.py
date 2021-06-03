#!/usr/bin/env python2.7
from __future__ import print_function

import sys
import cv2
import numpy as np
import grpc
import time

import threading

import tensorflow as tf

from tensorflow_serving.apis import predict_pb2
from tensorflow_serving.apis import prediction_service_pb2_grpc

import string_int_label_map_pb2 as labelmap
from google.protobuf import text_format

from image_preprocessor import decode_image_opencv

tf.app.flags.DEFINE_string('server', 'localhost:8500',
                           'PredictionService host:port')
tf.app.flags.DEFINE_string('image', '/home/yitao/Documents/fun-project/tensorflow-related/post-rim/test/test_ssd/lion.jpg', 'path to image in JPEG format')
FLAGS = tf.app.flags.FLAGS

model_name = 'ssd_inception_v2_coco'
# model_name = 'ssd_resnet50'
# model_name = 'ssd_mobilenet'

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

def prepareInputs(batch_size):
  image, org = decode_image_opencv(FLAGS.image)
  image = image.astype(np.uint8)

  # much more efficient way to build a batch...
  return np.tile(image, (batch_size, 1, 1, 1))

def send_request(stub, inputs, batch_size, run_num, thread_id):
  durationSum = 0.0

  for i in range(run_num):
    t0 = time.time()

    request = predict_pb2.PredictRequest()    
    request.model_spec.name = model_name
    request.model_spec.signature_name = 'serving_default'

    request.inputs['inputs'].CopyFrom(tf.contrib.util.make_tensor_proto(inputs, shape=inputs.shape))

    result = stub.Predict(request, 10.0)

    t1 = time.time()
    duration = t1 - t0
    durationSum += duration
    # print("[thread-%d] for run %d for batch size %d, duration = %.3f" % (thread_id, i, batch_size, duration))

  # print("[thread-%d] average duration for batch size %d = %.3f" % (thread_id, batch_size, durationSum / run_num))

def main(_):
  s = open('mscoco_complete_label_map.pbtxt', 'r').read()
  mymap = labelmap.StringIntLabelMap()
  global _label_map 
  _label_map = text_format.Parse(s, mymap)

  channel = grpc.insecure_channel(FLAGS.server)
  stub = prediction_service_pb2_grpc.PredictionServiceStub(channel)

  thread_num_array = [1, 2, 4]
  batch_size_2d_array = [[1, 2, 4, 8, 16, 32, 64, 128],
                         [1, 2, 4, 8, 16, 32, 64],
                         [1, 2, 4, 8, 16, 32]]

  # thread_num_array = [1]
  # batch_size_2d_array = [[4, 8]]

  for i in range(len(thread_num_array)):
    thread_num = thread_num_array[i]
    batch_size_array = batch_size_2d_array[i]
    for batch_size in batch_size_array:

      time.sleep(2.0)
      
      run_num = 1024 / (thread_num * batch_size)

      # print("[INFO] thread_num = %d, batch_size = %d, run_num = %d" % (thread_num, batch_size, run_num))

      t0 = time.time()
      inputs = prepareInputs(batch_size)
      t1 = time.time()
      # print("[Debug] it took %.3f sec to prepare image of shape %s" % (t1 - t0, inputs.shape))

      # warmup
      warmup_num = 3
      send_request(stub, inputs, batch_size, warmup_num, 0)
      
      # raw_input("Press Enter to continue...")
      time.sleep(2.0)

      t0 = time.time()

      thread_pool = []
      for thread_id in range(thread_num):
        t = threading.Thread(target = send_request, args = (stub, inputs, batch_size, run_num, thread_id,))
        thread_pool.append(t)
        t.start()

      for t in thread_pool:
        t.join()

      t1 = time.time()
      print("[Summary] thread_num = %d, batch_size = %d, run_num = %d, total duration = %.3f" % (thread_num, batch_size, run_num, t1 - t0))


if __name__ == '__main__':
  tf.app.run()

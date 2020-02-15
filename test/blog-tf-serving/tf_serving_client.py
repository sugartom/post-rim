#!/usr/bin/env python
from __future__ import print_function
import numpy as np
import time

import cv2
import tensorflow as tf

import grpc

from tensorflow_serving.apis import predict_pb2
from tensorflow_serving.apis import prediction_service_pb2_grpc

tf.app.flags.DEFINE_string('server', 'localhost:8500',
                           'PredictionService host:port')
tf.app.flags.DEFINE_string('image', '', 'path to image in JPEG format')
FLAGS = tf.app.flags.FLAGS

def main():  
  # create prediction service client stub
  channel = grpc.insecure_channel(FLAGS.server)
  stub = prediction_service_pb2_grpc.PredictionServiceStub(channel)
  
  for i in range(10):
    tt = time.time()
    # create request
    request = predict_pb2.PredictRequest()
    request.model_spec.name = 'resnet'
    request.model_spec.signature_name = 'serving_default'

    # read image into numpy array
    img = cv2.imread(FLAGS.image).astype(np.float32)
    
    # convert to tensor proto and make request
    # shape is in NHWC (num_samples x height x width x channels) format
    tensor = tf.contrib.util.make_tensor_proto(img, shape=[1]+list(img.shape))
    request.inputs['input'].CopyFrom(tensor)
    resp = stub.Predict(request, 30.0)
    
    print('total time: {}s'.format(time.time() - tt))
    
if __name__ == '__main__':
    main()

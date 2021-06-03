# Darkflow should be installed from: https://github.com/sugartom/darkflow
from darkflow.net.build import TFNet

import numpy as np
from tensorflow_serving.apis import predict_pb2
from tensorflow_serving.apis import prediction_service_pb2_grpc

from tensorflow.python.framework import tensor_util
import cv2
import grpc
import tensorflow as tf

import time

tf.app.flags.DEFINE_string('server', 'localhost:8500',
                           'PredictionService host:port')
tf.app.flags.DEFINE_string('image', '/home/yitao/Documents/fun-project/tensorflow-related/post-rim/test/test_yolo/dog.jpg', 'path to image in JPEG format')
FLAGS = tf.app.flags.FLAGS

# Place your downloaded cfg and weights under "checkpoints/"
YOLO_CONFIG = '/home/yitao/Documents/fun-project/tensorflow-related/traffic-jammer/cfg'
YOLO_MODEL = '/home/yitao/Documents/fun-project/tensorflow-related/traffic-jammer/cfg/yolo.cfg'
YOLO_WEIGHTS = '/home/yitao/Documents/fun-project/tensorflow-related/traffic-jammer/bin/yolo.weights'
YOLO_THRES = 0.4

def main(_):
  opt = { 
          "config": YOLO_CONFIG,  
          "model": YOLO_MODEL, 
          "load": YOLO_WEIGHTS, 
          "threshold": YOLO_THRES
        }
  tfnet = TFNet(opt)

  channel = grpc.insecure_channel(FLAGS.server)
  stub = prediction_service_pb2_grpc.PredictionServiceStub(channel)

  image = cv2.imread(FLAGS.image)
  image = tfnet.framework.resize_input(image)
  image = np.expand_dims(image, 0)

  # # generate batched input
  # batch_size = 4
  # batched_im = image
  # for i in range(batch_size - 1):
  #   batched_im = np.append(batched_im, image, axis = 0)

  # debug only
  image2 = cv2.imread("/home/yitao/Documents/edge/D2-system/data/dog.jpg")
  image2 = tfnet.framework.resize_input(image2)
  image2 = np.expand_dims(image2, 0)
  batched_im = image
  batched_im = np.append(batched_im, image2, axis = 0)

  print(batched_im.shape)

  # print(batched_im[0].shape)

  for i in range(10):
    start = time.time()
    batched_result = tfnet.return_batched_predict(batched_im, "traffic_yolo", stub)
    end = time.time()
    print("duration = %.3f sec" % (end - start))
  
  # print(batched_result)

  batched_ouput = []
  for dets in batched_result:
    output = ""
    for d in dets:
      output += "%s|%s|%s|%s|%s|%s-" % (str(d['topleft']['x']), str(d['topleft']['y']), str(d['bottomright']['x']), str(d['bottomright']['y']), str(d['confidence']), str(d['label']))
    batched_ouput.append(output[:-1])
  print(batched_ouput)

  # output = ""
  # for d in dets:
  #   output += "%s|%s|%s|%s|%s|%s-" % (str(d['topleft']['x']), str(d['topleft']['y']), str(d['bottomright']['x']), str(d['bottomright']['y']), str(d['confidence']), str(d['label']))
  # output = output[:-1]

  # print(output)

if __name__ == '__main__':
  tf.app.run()
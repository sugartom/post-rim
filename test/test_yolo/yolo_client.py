# Darkflow should be installed from: https://github.com/sugartom/darkflow
from darkflow.net.build import TFNet

import numpy as np
from tensorflow_serving.apis import predict_pb2
from tensorflow_serving.apis import prediction_service_pb2_grpc

from tensorflow.python.framework import tensor_util
import cv2
import grpc
import tensorflow as tf

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
          "threshold": YOLO_THRES,
          "imgdir": "/home/yitao/Documents/fun-project/tensorflow-related/post-rim/test/test_yolo/inputs/",
          "batch": 8
        }
  tfnet = TFNet(opt)

  channel = grpc.insecure_channel(FLAGS.server)
  stub = prediction_service_pb2_grpc.PredictionServiceStub(channel)


  # image = cv2.imread(FLAGS.image)

  # dets = tfnet.return_predict(image, "traffic_yolo", stub)

  # To-do: modify predict(), since self.sess.run(self.out, feed_dict) should support batch inputs
  tfnet.batch_predict()

  # output = ""
  # for d in dets:
  #   output += "%s|%s|%s|%s|%s|%s-" % (str(d['topleft']['x']), str(d['topleft']['y']), str(d['bottomright']['x']), str(d['bottomright']['y']), str(d['confidence']), str(d['label']))
  # output = output[:-1]

  # print(output)

if __name__ == '__main__':
  tf.app.run()
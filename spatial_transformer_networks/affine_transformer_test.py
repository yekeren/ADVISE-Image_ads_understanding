
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import cv2
import numpy as np
import tensorflow as tf

from google.protobuf import text_format

from protos import spatial_transformer_networks_pb2
from spatial_transformer_networks import builder
from spatial_transformer_networks import affine_transformer

from utils import vis

slim = tf.contrib.slim
flatten = tf.contrib.layers.flatten


class AffineTransformerTest(tf.test.TestCase):
  def setUp(self):
    tf.logging.set_verbosity(tf.logging.INFO)

    config_str = """
      affine_transformer: {
        conv_hyperparams {
          activation: RELU_6,
          regularizer {
            l2_regularizer {
              weight: 0.00004
            }
          }
          initializer {
            truncated_normal_initializer {
              stddev: 0.03
              mean: 0.0
            }
          }
          batch_norm {
            train: true,
            scale: true,
            center: true,
            decay: 0.999,
            epsilon: 0.001,
          }
        }
      }
    """
    self.default_config = spatial_transformer_networks_pb2.SpatialTransformer()
    text_format.Merge(config_str, self.default_config)

  def test_get_prior(self):
    config = self.default_config

    g = tf.Graph()
    with g.as_default():
      transformer = builder.build(config)
      self.assertIsInstance(transformer, affine_transformer.AffineTransformer)

      detection_box = tf.placeholder(shape=[4], dtype=tf.float32)

      theta = transformer._get_prior(
          detection_boxes=tf.reshape(detection_box, [1, 1, 4]))
      self.assertEqual(theta.get_shape().as_list(), [1, 1, 3, 3])

    with self.test_session(graph=g) as sess:
      mat = sess.run(flatten(theta), 
          feed_dict={ detection_box: np.array([0, 0, 1, 1]) })
      self.assertNDArrayNear(
          mat[0], np.array([1, 0, 0, 0, 1, 0, 0, 0, 1]),
          err=1e-6)

      mat = sess.run(flatten(theta), 
          feed_dict={ detection_box: np.array([0.04972243, 0.0, 0.3760559, 0.97612488]) })
      self.assertNDArrayNear(
          mat[0], np.array([0.97612488, 0, -0.02387512, 0, 0.32633347,
            -0.57422167, 0, 0, 1]),
          err=1e-6)

  def test_get_relative_transformation(self):
    config = self.default_config

    g = tf.Graph()
    with g.as_default():
      transformer = builder.build(config)
      self.assertIsInstance(transformer, affine_transformer.AffineTransformer)

      feature_map = tf.random_uniform(shape=[1, 19, 19, 512])
      detection_box = tf.placeholder(shape=[4], dtype=tf.float32)

      delta, score = transformer._get_relative_transformation(feature_map,
          detection_boxes=tf.reshape(detection_box, [1, 1, 4]), 
          is_training=True)

      tf.logging.info('*' * 128)
      for v in tf.global_variables():
        tf.logging.info('%s: %s', v.op.name, v.get_shape().as_list())

      self.assertEqual(delta.get_shape().as_list(), [1, 1, 3, 3])
      self.assertEqual(score.get_shape().as_list(), [1, 1])

  def test_predict_transformation(self):
    config = self.default_config

    g = tf.Graph()
    with g.as_default():
      transformer = builder.build(config)
      self.assertIsInstance(transformer, affine_transformer.AffineTransformer)

      feature_map = tf.random_uniform(shape=[1, 19, 19, 512])
      num_detection = tf.placeholder(shape=[], dtype=tf.int64)
      detection_box = tf.placeholder(shape=[4], dtype=tf.float32)

      theta, score = transformer.predict_transformation(feature_map,
          num_detections=tf.expand_dims(num_detection, 0),
          detection_boxes=tf.reshape(detection_box, [1, 1, 4]), 
          is_training=True)

      tf.logging.info('*' * 128)
      for v in tf.global_variables():
        tf.logging.info('%s: %s', v.op.name, v.get_shape().as_list())

      self.assertEqual(theta.get_shape().as_list(), [1, 1, 6])
      self.assertEqual(score.get_shape().as_list(), [1, 1])

if __name__ == '__main__':
  tf.test.main()


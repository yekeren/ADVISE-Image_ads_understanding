
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import cv2
import numpy as np
import tensorflow as tf

from google.protobuf import text_format

from protos import spatial_transformer_networks_pb2
from spatial_transformer_networks import builder
from spatial_transformer_networks import simple_transformer

from utils import vis

slim = tf.contrib.slim


class SimpleTransformerTest(tf.test.TestCase):
  def setUp(self):
    tf.logging.set_verbosity(tf.logging.INFO)

    config_str = """
      simple_transformer: {
      }
    """
    self.default_config = spatial_transformer_networks_pb2.SpatialTransformer()
    text_format.Merge(config_str, self.default_config)

  def test_predict_transformation(self):
    config = self.default_config

    g = tf.Graph()
    with g.as_default():
      transformer = builder.build(config)
      self.assertIsInstance(transformer, simple_transformer.SimpleTransformer)

      num_detection = tf.placeholder(shape=[], dtype=tf.float32)
      detection_box = tf.placeholder(shape=[4], dtype=tf.float32)

      theta = transformer.predict_transformation(
          feature_map=None,
          num_detections=tf.expand_dims(num_detection, 0),
          detection_boxes=tf.reshape(detection_box, [1, 1, 4]))
      self.assertEqual(theta.get_shape().as_list(), [1, 1, 6])

    with self.test_session(graph=g) as sess:
      mat = sess.run(theta, 
          feed_dict={ num_detection: 1, detection_box: np.array([0, 0, 1, 1]) })
      self.assertNDArrayNear(
          mat[0], np.array([1, 0, 0, 0, 1, 0]),
          err=1e-6)

      mat = sess.run(theta, 
          feed_dict={ num_detection: 1, detection_box: np.array([0.04972243, 0.0, 0.3760559, 0.97612488]) })
      self.assertNDArrayNear(
          mat[0], np.array([0.97612488, 0, -0.02387512, 0, 0.32633347, -0.57422167]),
          err=1e-6)

  def test_transform_image(self):
    config = self.default_config

    image_data = vis.image_load('testdata/99790.jpg', convert_to_rgb=True)

    g = tf.Graph()
    with g.as_default():
      transformer = builder.build(config)
      self.assertIsInstance(transformer, simple_transformer.SimpleTransformer)

      image = tf.placeholder(shape=[None, None, 3], dtype=tf.uint8)
      num_detection = tf.placeholder(shape=[], dtype=tf.float32)
      detection_box = tf.placeholder(shape=[4], dtype=tf.float32)

      theta = transformer.predict_transformation(
          feature_map=None,
          num_detections=tf.expand_dims(num_detection, 0),
          detection_boxes=tf.reshape(detection_box, [1, 1, 4]))
      transformed_images = transformer.transform_image(
          image=tf.expand_dims(tf.cast(image, tf.float32), 0),
          theta=theta,
          out_size=(256, 256))
      self.assertEqual(transformed_images.get_shape().as_list(), 
          [1, 1, 256, 256, 3])

    with self.test_session(graph=g) as sess:
      roi_image, theta = sess.run([transformed_images[0, 0], theta[0]], 
          feed_dict={ num_detection: 1, detection_box: np.array([0.04972243,
            0.0, 0.3760559, 0.97612488]), image: image_data})
      roi_image = roi_image.astype(np.uint8)
      vis.image_save('testdata/results/simple_transformer.jpg', roi_image, True)


if __name__ == '__main__':
  tf.test.main()

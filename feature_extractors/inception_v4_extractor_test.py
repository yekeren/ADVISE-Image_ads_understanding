
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import cv2
import numpy as np
import tensorflow as tf

from google.protobuf import text_format

from protos import feature_extractors_pb2
from feature_extractors import builder
from feature_extractors import inception_v4_extractor

slim = tf.contrib.slim


class MobilnetV1ExtractorTest(tf.test.TestCase):
  def setUp(self):
    tf.logging.set_verbosity(tf.logging.INFO)

    config_str = """
      inception_v4_extractor: {
      }
    """
    self.default_config = feature_extractors_pb2.FeatureExtractor()
    text_format.Merge(config_str, self.default_config)

  def test_extract_feature(self):
    g = tf.Graph()
    with g.as_default():
      feature_extractor = builder.build(self.default_config)
      self.assertIsInstance(feature_extractor, 
          inception_v4_extractor.InceptionV4Extractor)

      self.assertEqual(299, feature_extractor.default_image_size)
      image = tf.random_uniform(shape=[5, 311, 311, 3], dtype=tf.float32)
      feature = feature_extractor.extract_feature(image)
      self.assertEqual(feature.get_shape().as_list(), [5, 1536])
      
      assign_fn = feature_extractor.assign_from_checkpoint_fn(
          'models/zoo/inception_v4.ckpt')
      invalid_tensor_names = tf.report_uninitialized_variables()

    with self.test_session(graph=g) as sess:
      assign_fn(sess)
      names = sess.run(invalid_tensor_names)
      self.assertListEqual(names.tolist(), [])


if __name__ == '__main__':
    tf.test.main()


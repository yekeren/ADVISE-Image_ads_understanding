
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import cv2
import numpy as np
import tensorflow as tf

from google.protobuf import text_format

from protos import feature_extractors_pb2
from feature_extractors import builder
from feature_extractors import fc_extractor

slim = tf.contrib.slim


class FCExtractorTest(tf.test.TestCase):
  def setUp(self):
    tf.logging.set_verbosity(tf.logging.INFO)

    config_str = """
      fc_extractor: {
        hidden_hyperparams {
          op: FC
          activation: RELU_6
          regularizer {
            l2_regularizer {
              weight: 1e-8
            }
          }
          initializer {
            truncated_normal_initializer {
              stddev: 0.03
              mean: 0.0
            }
          }
          batch_norm {
            train: true
            scale: true
            center: true
            decay: 0.999
            epsilon: 0.001
          }
        }
        output_hyperparams {
          op: FC
          activation: NONE
          regularizer {
            l2_regularizer {
              weight: 0.0
            }
          }
          initializer {
            truncated_normal_initializer {
              stddev: 0.03
              mean: 0.0
            }
          }
          batch_norm {
            train: true
            scale: true
            center: false
            decay: 0.999
            epsilon: 0.001
          }
        }
        hidden_layers: 1
        hidden_units: 128
        output_units: 1024
      }
    """
    self.default_config = feature_extractors_pb2.FeatureExtractor()
    text_format.Merge(config_str, self.default_config)

  def test_extract_feature(self):
    g = tf.Graph()
    with g.as_default():
      feature_extractor = builder.build(self.default_config)
      self.assertIsInstance(feature_extractor, 
          fc_extractor.FCExtractor)

      input_data = tf.random_uniform(shape=[5, 200], dtype=tf.float32)
      feature = feature_extractor.extract_feature(input_data)

      self.assertEqual(feature.get_shape().as_list(), [5, 1024])


if __name__ == '__main__':
    tf.test.main()


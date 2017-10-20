
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import cv2
import numpy as np
import tensorflow as tf

from google.protobuf import text_format

from protos import text_encoders_pb2
from text_encoders import builder
from text_encoders import bow_encoder

slim = tf.contrib.slim


class BOWEncoderTest(tf.test.TestCase):
  def setUp(self):
    tf.logging.set_verbosity(tf.logging.INFO)

    config_str = """
      bow_encoder: {
        vocab_size: 10000
        embedding_size: 200
      }
    """
    self.default_config = text_encoders_pb2.TextEncoder()
    text_format.Merge(config_str, self.default_config)

  def test_encode(self):
    # vocab_size: 10000, embedding_size: 200
    config = self.default_config

    with tf.Graph().as_default():
      encoder = builder.build(config)
      self.assertIsInstance(encoder, bow_encoder.BOWEncoder)

      text_lengths_placeholder = tf.placeholder(shape=[], dtype=tf.int64)
      text_strings_placeholder = tf.placeholder(shape=[30], dtype=tf.int64)

      embeddings = encoder.encode(
          tf.expand_dims(text_lengths_placeholder, 0), 
          tf.expand_dims(text_strings_placeholder, 0))
      self.assertEqual(embeddings.get_shape().as_list(), [1, 200])
      self.assertEqual(
          encoder.embedding_weights.get_shape().as_list(), 
          [10000, 200])

    # vocab_size: 10001, embedding_size: 201
    config = self.default_config
    config.bow_encoder.vocab_size = 10001
    config.bow_encoder.embedding_size = 201

    with tf.Graph().as_default():
      encoder = builder.build(config)
      self.assertIsInstance(encoder, bow_encoder.BOWEncoder)

      text_lengths_placeholder = tf.placeholder(shape=[], dtype=tf.int64)
      text_strings_placeholder = tf.placeholder(shape=[30], dtype=tf.int64)

      embeddings = encoder.encode(
          tf.expand_dims(text_lengths_placeholder, 0), 
          tf.expand_dims(text_strings_placeholder, 0))
      self.assertEqual(embeddings.get_shape().as_list(), [1, 201])
      self.assertEqual(
          encoder.embedding_weights.get_shape().as_list(), 
          [10001, 201])


if __name__ == '__main__':
    tf.test.main()

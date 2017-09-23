
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import cv2
import numpy as np
import tensorflow as tf

from google.protobuf import text_format

from protos import ads_emb_model_pb2
from text_embedders import builder
from text_embedders import bow_embedder

slim = tf.contrib.slim


class BOWEmbedderTest(tf.test.TestCase):
  def setUp(self):
    tf.logging.set_verbosity(tf.logging.INFO)

    config_str = """
      bow_embedder: {
        vocab_size: 10000
        embedding_size: 200
      }
    """
    self.default_config = ads_emb_model_pb2.TextEmbedder()
    text_format.Merge(config_str, self.default_config)

  def test_embed(self):
    # vocab_size: 10000, embedding_size: 200
    config = self.default_config

    with tf.Graph().as_default():
      embedder = builder.build(config)
      self.assertIsInstance(embedder, bow_embedder.BOWEmbedder)

      text_lengths_placeholder = tf.placeholder(shape=[], dtype=tf.int64)
      text_strings_placeholder = tf.placeholder(shape=[30], dtype=tf.int64)

      embeddings = embedder.embed(
          tf.expand_dims(text_lengths_placeholder, 0), 
          tf.expand_dims(text_strings_placeholder, 0))
      self.assertEqual(embeddings.get_shape().as_list(), [1, 200])
      self.assertEqual(
          embedder.embedding_weights.get_shape().as_list(), 
          [10000, 200])

    # vocab_size: 10001, embedding_size: 201
    config = self.default_config
    config.bow_embedder.vocab_size = 10001
    config.bow_embedder.embedding_size = 201

    with tf.Graph().as_default():
      embedder = builder.build(config)
      self.assertIsInstance(embedder, bow_embedder.BOWEmbedder)

      text_lengths_placeholder = tf.placeholder(shape=[], dtype=tf.int64)
      text_strings_placeholder = tf.placeholder(shape=[30], dtype=tf.int64)

      embeddings = embedder.embed(
          tf.expand_dims(text_lengths_placeholder, 0), 
          tf.expand_dims(text_strings_placeholder, 0))
      self.assertEqual(embeddings.get_shape().as_list(), [1, 201])
      self.assertEqual(
          embedder.embedding_weights.get_shape().as_list(), 
          [10001, 201])


if __name__ == '__main__':
    tf.test.main()

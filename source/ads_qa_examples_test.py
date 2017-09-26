
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import cv2
import numpy as np
import tensorflow as tf

from google.protobuf import text_format

from protos import ads_emb_model_pb2
import ads_qa_examples

slim = tf.contrib.slim


class AdsQAExamplesTest(tf.test.TestCase):
  def setUp(self):
    tf.logging.set_verbosity(tf.logging.INFO)

    config_str = """
      input_path: "output/ads_qa.mobilenet_v1.train.record" 
      batch_size: 32
      image_height: 500
      image_width: 500
      max_num_captions: 5
      max_caption_len: 30
      export_feature: true
      max_detections: 10
      feature_dimentions: 1024
      image_level_feature: true
    """
    self.default_mobilenet_v1_config = ads_emb_model_pb2.AdsQAExamples()
    text_format.Merge(config_str, self.default_mobilenet_v1_config)

    config_str = """
      input_path: "output/ads_qa.inception_v4.train.record" 
      batch_size: 32
      image_height: 500
      image_width: 500
      max_num_captions: 5
      max_caption_len: 30
      export_feature: true
      max_detections: 10
      feature_dimentions: 1536
      image_level_feature: true
    """
    self.default_inception_v4_config = ads_emb_model_pb2.AdsQAExamples()
    text_format.Merge(config_str, self.default_inception_v4_config)

  def test_mobilenet_v1_image_level_feature(self):
    # Image level feature.
    config = self.default_mobilenet_v1_config
    config.image_level_feature = True

    g = tf.Graph()
    with g.as_default():
      example = ads_qa_examples.get_examples(config)

    with self.test_session(graph=g) as sess:
      coord = tf.train.Coordinator()
      threads = tf.train.start_queue_runners(coord=coord)
      example = sess.run(example)
      self.assertEqual(example['image'].shape, (32, 500, 500, 3))
      self.assertEqual(example['topic'].shape, (32,))
      self.assertEqual(example['num_captions'].shape, (32,))
      self.assertEqual(example['caption_lengths'].shape, (32, 5))
      self.assertEqual(example['caption_strings'].shape, (32, 5, 30))
      self.assertEqual(example['num_detections'].shape, (32,))
      self.assertEqual(example['proposed_features'].shape, (32, 1, 1024))

      coord.request_stop()
      coord.join(threads, stop_grace_period_secs=10)

  def test_mobilenet_v1_patch_level_feature(self):
    # Patch level feature.
    config = self.default_mobilenet_v1_config
    config.image_level_feature = False

    g = tf.Graph()
    with g.as_default():
      example = ads_qa_examples.get_examples(config)

    with self.test_session(graph=g) as sess:
      coord = tf.train.Coordinator()
      threads = tf.train.start_queue_runners(coord=coord)
      example = sess.run(example)
      self.assertEqual(example['image'].shape, (32, 500, 500, 3))
      self.assertEqual(example['topic'].shape, (32,))
      self.assertEqual(example['num_captions'].shape, (32,))
      self.assertEqual(example['caption_lengths'].shape, (32, 5))
      self.assertEqual(example['caption_strings'].shape, (32, 5, 30))
      self.assertEqual(example['num_detections'].shape, (32,))
      self.assertEqual(example['proposed_features'].shape, (32, 10, 1024))

      coord.request_stop()
      coord.join(threads, stop_grace_period_secs=10)

  def test_mobilenet_v1_export_no_feature(self):
    # Do not export feature.
    config = self.default_mobilenet_v1_config
    config.export_feature = False

    g = tf.Graph()
    with g.as_default():
      example = ads_qa_examples.get_examples(config)

    with self.test_session(graph=g) as sess:
      coord = tf.train.Coordinator()
      threads = tf.train.start_queue_runners(coord=coord)
      example = sess.run(example)
      self.assertEqual(example['image'].shape, (32, 500, 500, 3))
      self.assertEqual(example['topic'].shape, (32,))
      self.assertEqual(example['num_captions'].shape, (32,))
      self.assertEqual(example['caption_lengths'].shape, (32, 5))
      self.assertEqual(example['caption_strings'].shape, (32, 5, 30))
      self.assertNotIn('num_detections', example)
      self.assertNotIn('proposed_features', example)

      coord.request_stop()
      coord.join(threads, stop_grace_period_secs=10)

  def test_mobilenet_v1_densecap_captions(self):
    # Patch level feature.
    config = self.default_mobilenet_v1_config
    config.image_level_feature = False
    config.input_path = "output/ads_qa_with_densecap.mobilenet_v1.train.record"
    config.export_densecap_captions = True
    config.densecap_max_num_captions = 10
    config.densecap_max_caption_len = 10

    g = tf.Graph()
    with g.as_default():
      example = ads_qa_examples.get_examples(config)

    with self.test_session(graph=g) as sess:
      coord = tf.train.Coordinator()
      threads = tf.train.start_queue_runners(coord=coord)
      example = sess.run(example)
      self.assertEqual(example['image'].shape, (32, 500, 500, 3))
      self.assertEqual(example['topic'].shape, (32,))
      self.assertEqual(example['num_captions'].shape, (32,))
      self.assertEqual(example['caption_lengths'].shape, (32, 5))
      self.assertEqual(example['caption_strings'].shape, (32, 5, 30))
      self.assertEqual(example['num_detections'].shape, (32,))
      self.assertEqual(example['proposed_features'].shape, (32, 10, 1024))
      self.assertEqual(example['densecap_num_captions'].shape, (32,))
      self.assertEqual(example['densecap_caption_lengths'].shape, (32, 10))
      self.assertEqual(example['densecap_caption_strings'].shape, (32, 10, 10))

      coord.request_stop()
      coord.join(threads, stop_grace_period_secs=10)

  def test_inception_v4_image_level_feature(self):
    # Image level feature.
    config = self.default_inception_v4_config
    config.image_level_feature = True

    g = tf.Graph()
    with g.as_default():
      example = ads_qa_examples.get_examples(config)

    with self.test_session(graph=g) as sess:
      coord = tf.train.Coordinator()
      threads = tf.train.start_queue_runners(coord=coord)
      example = sess.run(example)
      self.assertEqual(example['image'].shape, (32, 500, 500, 3))
      self.assertEqual(example['topic'].shape, (32,))
      self.assertEqual(example['num_captions'].shape, (32,))
      self.assertEqual(example['caption_lengths'].shape, (32, 5))
      self.assertEqual(example['caption_strings'].shape, (32, 5, 30))
      self.assertEqual(example['num_detections'].shape, (32,))
      self.assertEqual(example['proposed_features'].shape, (32, 1, 1536))

      coord.request_stop()
      coord.join(threads, stop_grace_period_secs=10)

  def test_inception_v4_patch_level_feature(self):
    # Patch level feature.
    config = self.default_inception_v4_config
    config.image_level_feature = False

    g = tf.Graph()
    with g.as_default():
      example = ads_qa_examples.get_examples(config)

    with self.test_session(graph=g) as sess:
      coord = tf.train.Coordinator()
      threads = tf.train.start_queue_runners(coord=coord)
      example = sess.run(example)
      self.assertEqual(example['image'].shape, (32, 500, 500, 3))
      self.assertEqual(example['topic'].shape, (32,))
      self.assertEqual(example['num_captions'].shape, (32,))
      self.assertEqual(example['caption_lengths'].shape, (32, 5))
      self.assertEqual(example['caption_strings'].shape, (32, 5, 30))
      self.assertEqual(example['num_detections'].shape, (32,))
      self.assertEqual(example['proposed_features'].shape, (32, 10, 1536))

      coord.request_stop()
      coord.join(threads, stop_grace_period_secs=10)

  def test_inception_v4_export_no_feature(self):
    # Do not export feature.
    config = self.default_inception_v4_config
    config.export_feature = False

    g = tf.Graph()
    with g.as_default():
      example = ads_qa_examples.get_examples(config)

    with self.test_session(graph=g) as sess:
      coord = tf.train.Coordinator()
      threads = tf.train.start_queue_runners(coord=coord)
      example = sess.run(example)
      self.assertEqual(example['image'].shape, (32, 500, 500, 3))
      self.assertEqual(example['topic'].shape, (32,))
      self.assertEqual(example['num_captions'].shape, (32,))
      self.assertEqual(example['caption_lengths'].shape, (32, 5))
      self.assertEqual(example['caption_strings'].shape, (32, 5, 30))
      self.assertNotIn('num_detections', example)
      self.assertNotIn('proposed_features', example)

      coord.request_stop()
      coord.join(threads, stop_grace_period_secs=10)

if __name__ == '__main__':
    tf.test.main()

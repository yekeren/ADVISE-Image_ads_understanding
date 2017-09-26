
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import cv2
import numpy as np
import tensorflow as tf

from google.protobuf import text_format

from utils import vis

import object_detection
from protos import ads_emb_model_pb2
import feature_extractors
import text_embedders
from ads_emb_model import AdsEmbModel
from ads_emb_model import mine_hard_examples

slim = tf.contrib.slim


def _load_proto(file_path):
  """Loads proto file.

  Args:
    file_path: path to the proto file.

  Returns:
    model_proto: an instance of AdsEmbModel proto.
  """
  model_proto = ads_emb_model_pb2.AdsEmbModel()
  with open(file_path, 'r') as fp:
    text_format.Merge(fp.read(), model_proto)
  return model_proto


class AdsEmbModelTest(tf.test.TestCase):
  def setUp(self):
    tf.logging.set_verbosity(tf.logging.INFO)

    self._default_proto_text = """
      feature_extractor {
        mobilenet_v1_extractor {
        }
      }
      feature_extractor_checkpoint: "./models/zoo/mobilenet_v1_1.0_224/mobilenet_v1_1.0_224.ckpt"
      caption_embedder {
        bow_embedder {
          vocab_size: 10000
        }
      }
    """

  def test_build_region_proposals(self):
    image_data = vis.image_load('testdata/99790.jpg', convert_to_rgb=True)
    
    # Test pre-trained model.
    g = tf.Graph()
    with g.as_default():
      image = tf.placeholder(shape=[None, None, 3], dtype=tf.uint8)
      images = tf.expand_dims(image, 0)

      model = AdsEmbModel(_load_proto('testdata/rpn.pbtxt'))
      self.assertIsInstance(model.detection_model,
          object_detection.meta_architectures.ssd_meta_arch.SSDMetaArch)
      proposals, assign_fn = model._build_region_proposals(images)

      self.assertEqual(
          proposals['num_detections'].get_shape().as_list(), [1])
      self.assertEqual(
          proposals['detection_scores'].get_shape().as_list(), [1, 10])
      self.assertEqual(
          proposals['detection_boxes'].get_shape().as_list(), [1, 10, 4])
      invalid_tensor_names = tf.report_uninitialized_variables()

    with self.test_session(graph=g) as sess:
      assign_fn(sess)
      invalid_tensor_names = sess.run(invalid_tensor_names)
      self.assertListEqual(invalid_tensor_names.tolist(), [])

      proposals = sess.run(proposals, feed_dict={image: image_data})

      self.assertEqual(proposals['num_detections'][0], 1)
      self.assertNDArrayNear(
          proposals['detection_boxes'][0, 0],
          np.array([0.04972243, 0.0, 0.3760559, 0.97612488]),
          err=1e-6)

      vis_data = np.copy(image_data)
      for i in xrange(int(proposals['num_detections'][0])):
        y1, x1, y2, x2 = proposals['detection_boxes'][0, i]
        vis.image_draw_bounding_box(vis_data, [x1, y1, x2, y2])
      vis.image_save('testdata/results/rpn.jpg', 
          vis_data, convert_to_bgr=True)

    # Test full image proposal.
    g = tf.Graph()
    with g.as_default():
      image = tf.placeholder(shape=[None, None, 3], dtype=tf.uint8)
      images = tf.expand_dims(image, 0)

      model_proto = ads_emb_model_pb2.AdsEmbModel()
      text_format.Merge(self._default_proto_text, model_proto)

      model = AdsEmbModel(model_proto)
      proposals, assign_fn = model._build_region_proposals(images)

      self.assertEqual(
          proposals['num_detections'].get_shape().as_list(), [1])
      self.assertEqual(
          proposals['detection_scores'].get_shape().as_list(), [1, 1])
      self.assertEqual(
          proposals['detection_boxes'].get_shape().as_list(), [1, 1, 4])
      invalid_tensor_names = tf.report_uninitialized_variables()

    with self.test_session(graph=g) as sess:
      assign_fn(sess)
      invalid_tensor_names = sess.run(invalid_tensor_names)
      self.assertListEqual(invalid_tensor_names.tolist(), [])

      proposals = sess.run(proposals, feed_dict={image: image_data})

      self.assertEqual(proposals['num_detections'][0], 1)
      self.assertNDArrayNear(
          proposals['detection_boxes'][0, 0],
          np.array([0.0, 0.0, 1.0, 1.0]),
          err=1e-8)

      y1, x1, y2, x2 = proposals['detection_boxes'][0, 0]
      vis_data = np.copy(image_data)
      vis.image_draw_bounding_box(vis_data, [x1, y1, x2, y2])
      vis.image_save('testdata/results/rpn_no.jpg', 
          vis_data, convert_to_bgr=True)

  def test_crop_and_resize_region_proposals(self):
    input_size = 500
    crop_size = 300
    image_data_1 = vis.image_load('testdata/99790.jpg', convert_to_rgb=True)
    image_data_2 = vis.image_load('testdata/158990.jpg', convert_to_rgb=True)
    image_data_3 = vis.image_load('testdata/17961.jpg', convert_to_rgb=True)
    image_data_4 = vis.image_load('testdata/39839.jpg', convert_to_rgb=True)
    image_data = np.stack([
        cv2.resize(image_data_1, (input_size, input_size)), 
        cv2.resize(image_data_2, (input_size, input_size)),
        cv2.resize(image_data_3, (input_size, input_size)),
        cv2.resize(image_data_4, (input_size, input_size))
        ], axis=0)

    g = tf.Graph()
    with g.as_default():
      images = tf.placeholder(shape=[4, None, None, 3], dtype=tf.uint8)

      model = AdsEmbModel(_load_proto('testdata/rpn.pbtxt'))

      region_proposals, assign_fn = model._build_region_proposals(images)
      boolean_masks, proposed_images = model._crop_and_resize_region_proposals(
          images,
          num_detections=region_proposals['num_detections'],
          detection_boxes=region_proposals['detection_boxes'],
          crop_size=(crop_size, crop_size))

      self.assertEqual(boolean_masks.get_shape().as_list(), [4, 10])
      self.assertEqual(proposed_images.get_shape().as_list(), 
          [None, crop_size, crop_size, 3])

    with self.test_session(graph=g) as sess:
      assign_fn(sess)

      num_detections, boolean_masks, proposed_images = sess.run(
          [region_proposals['num_detections'], boolean_masks, proposed_images],
          feed_dict={images: image_data})

      self.assertEqual(
          num_detections.sum(), boolean_masks.astype(np.int32).sum())
      self.assertEqual(
          num_detections.sum(), proposed_images.shape[0])
      for i, proposed_image in enumerate(proposed_images):
        vis.image_save(
            'testdata/results/crop_%02d.jpg' % (i),
            proposed_image,
            convert_to_bgr=True)

  def test_extract_feature(self):
    g = tf.Graph()
    with g.as_default():
      # MobilenetV1.
      model_proto = ads_emb_model_pb2.AdsEmbModel()
      text_format.Merge(self._default_proto_text, model_proto)

      model = AdsEmbModel(model_proto)

      feature_extractor = model.feature_extractor
      self.assertIsInstance(feature_extractor,
          feature_extractors.mobilenet_v1_extractor.MobilenetV1Extractor)
      self.assertEqual(224, feature_extractor.default_image_size)

      image = tf.random_uniform(shape=[5, 311, 311, 3], dtype=tf.float32)
      feature = feature_extractor.extract_feature(image)
      self.assertEqual(feature.get_shape().as_list(), [5, 1024])
      
      assign_fn_1 = feature_extractor.assign_from_checkpoint_fn(
          'models/zoo/mobilenet_v1_1.0_224/mobilenet_v1_1.0_224.ckpt')

      # InceptionV4.
      proto_str = """
        feature_extractor {
          inception_v4_extractor {
          }
        }
        caption_embedder {
          bow_embedder {
            vocab_size: 10000
          }
        }
      """
      model_proto = ads_emb_model_pb2.AdsEmbModel()
      text_format.Merge(proto_str,  model_proto)

      model = AdsEmbModel(model_proto)

      feature_extractor = model.feature_extractor
      self.assertIsInstance(feature_extractor,
          feature_extractors.inception_v4_extractor.InceptionV4Extractor)
      self.assertEqual(299, feature_extractor.default_image_size)

      image = tf.random_uniform(shape=[5, 311, 311, 3], dtype=tf.float32)
      feature = feature_extractor.extract_feature(image)
      self.assertEqual(feature.get_shape().as_list(), [5, 1536])

      assign_fn_2 = feature_extractor.assign_from_checkpoint_fn(
          'models/zoo/inception_v4.ckpt')

      invalid_tensor_names = tf.report_uninitialized_variables()

    with self.test_session(graph=g) as sess:
      assign_fn_1(sess)
      names = sess.run(invalid_tensor_names)
      self.assertGreater(len(names), 0)

      assign_fn_2(sess)
      names = sess.run(invalid_tensor_names)
      self.assertListEqual(names.tolist(), [])

  def test_embed_feature(self):
    model_proto = ads_emb_model_pb2.AdsEmbModel()
    text_format.Merge(self._default_proto_text, model_proto)

    model = AdsEmbModel(model_proto)

    # 200-D.
    with tf.Graph().as_default():
      feature_vectors = tf.random_uniform(shape=[5, 555], dtype=tf.float32)
      embeddings = model._embed_feature(feature_vectors, 200, 0.0, True)
      self.assertEqual(embeddings.get_shape().as_list(), [5, 200])

    # 205-D.
    with tf.Graph().as_default():
      feature_vectors = tf.random_uniform(shape=[5, 555], dtype=tf.float32)
      embeddings = model._embed_feature(feature_vectors, 205, 0.0, True)
      self.assertEqual(embeddings.get_shape().as_list(), [5, 205])

  def test_average_proposed_embs(self):
    max_detections = 5
    embedding_size = 4

    model_proto = ads_emb_model_pb2.AdsEmbModel()
    text_format.Merge(self._default_proto_text, model_proto)

    model = AdsEmbModel(model_proto)

    g = tf.Graph()
    with g.as_default():
      proposed_embs = tf.placeholder(
          shape=[max_detections, embedding_size],
          dtype=tf.float32)
      boolean_masks = tf.placeholder(
          shape=[max_detections], dtype=tf.bool)

      avg_embs = model._average_proposed_embs(
          tf.expand_dims(proposed_embs, 0),
          tf.expand_dims(boolean_masks, 0))

    with self.test_session(graph=g) as sess:
      input_data = np.array([ 
          [1, 1, 1, 1],
          [2, 2, 2, 2],
          [3, 3, 3, 3],
          [4, 4, 4, 4],
          [5, 5, 5, 5]
          ], dtype=np.float32)
      # Average all embedding vectors.
      emb = sess.run(avg_embs, feed_dict={
          proposed_embs: input_data,
          boolean_masks: np.array([True, True, True, True, True])
          })
      self.assertNDArrayNear(
          emb[0], np.array([3, 3, 3, 3]), err=1e-6)

      # Average first three embedding vectors.
      emb = sess.run(avg_embs, feed_dict={
          proposed_embs: input_data,
          boolean_masks: np.array([True, True, True, False, False])
          })
      self.assertNDArrayNear(
          emb[0], np.array([2, 2, 2, 2]), err=1e-6)

      # Average first four embedding vectors.
      emb = sess.run(avg_embs, feed_dict={
          proposed_embs: input_data,
          boolean_masks: np.array([True, True, True, True, False])
          })
      self.assertNDArrayNear(
          emb[0], np.array([2.5, 2.5, 2.5, 2.5]), err=1e-6)

      # Average no embedding vector.
      emb = sess.run(avg_embs, feed_dict={
          proposed_embs: input_data,
          boolean_masks: np.array([False, False, False, False, False])
          })
      self.assertNDArrayNear(
          emb[0], np.array([0, 0, 0, 0]), err=1e-6)

  def test_build_image_model(self):
    model_proto = ads_emb_model_pb2.AdsEmbModel()
    text_format.Merge(self._default_proto_text, model_proto)

    model = AdsEmbModel(model_proto)

    image_data = vis.image_load('testdata/99790.jpg', convert_to_rgb=True)
    
    g = tf.Graph()
    with g.as_default():
      image = tf.placeholder(shape=[None, None, 3], dtype=tf.uint8)
      images = tf.expand_dims(image, 0)

      image_embs, assign_fn = model.build_image_model(
          images, is_training=False) 
      invalid_tensor_names = tf.report_uninitialized_variables()

    with self.test_session(graph=g) as sess:
      assign_fn(sess)
      invalid_tensor_names = sess.run(invalid_tensor_names)
      for name in invalid_tensor_names:
        self.assertTrue('image/embedding/' in name)

  def test_embed_text(self):
    # vocab_size: 1000, embedding_size: 100
    with tf.Graph().as_default():
      model_proto = ads_emb_model_pb2.AdsEmbModel()
      text_format.Merge(self._default_proto_text, model_proto)
      model_proto.caption_embedder.bow_embedder.vocab_size = 1000
      model_proto.caption_embedder.bow_embedder.embedding_size = 100

      model = AdsEmbModel(model_proto)
      self.assertIsInstance(model.caption_embedder,
          text_embedders.bow_embedder.BOWEmbedder)

      caption_lengths = tf.placeholder(shape=[4], dtype=tf.int64)
      caption_strings = tf.placeholder(shape=[4, 5], dtype=tf.int64)

      embs = model.caption_embedder.embed(caption_lengths, caption_strings)
      self.assertEqual(embs.get_shape().as_list(), [4, 100])
      self.assertEqual(
          model.caption_embedder.embedding_weights.get_shape().as_list(),
          [1000, 100])

    # vocab_size: 999, embedding_size: 99
    with tf.Graph().as_default():
      model_proto = ads_emb_model_pb2.AdsEmbModel()
      text_format.Merge(self._default_proto_text, model_proto)
      model_proto.caption_embedder.bow_embedder.vocab_size = 999
      model_proto.caption_embedder.bow_embedder.embedding_size = 99

      model = AdsEmbModel(model_proto)
      self.assertIsInstance(model.caption_embedder,
          text_embedders.bow_embedder.BOWEmbedder)

      caption_lengths = tf.placeholder(shape=[3], dtype=tf.int64)
      caption_strings = tf.placeholder(shape=[3, 5], dtype=tf.int64)

      embs = model.caption_embedder.embed(caption_lengths, caption_strings)
      self.assertEqual(embs.get_shape().as_list(), [3, 99])
      self.assertEqual(
          model.caption_embedder.embedding_weights.get_shape().as_list(),
          [999, 99])

  def test_build_caption_model(self):
    model_proto = ads_emb_model_pb2.AdsEmbModel()
    text_format.Merge(self._default_proto_text, model_proto)

    model = AdsEmbModel(model_proto)

    g = tf.Graph()
    with g.as_default():
      caption_lengths = tf.placeholder(shape=[3], dtype=tf.int64)
      caption_strings = tf.placeholder(shape=[3, 5], dtype=tf.int64)

      caption_embs, assign_fn = model.build_caption_model(
          caption_lengths, caption_strings, is_training=False) 
      invalid_tensor_names = tf.report_uninitialized_variables()

    with self.test_session(graph=g) as sess:
      assign_fn(sess)
      invalid_tensor_names = sess.run(invalid_tensor_names)
      for name in invalid_tensor_names:
        self.assertTrue('BOW/' in name)

  def test_mine_hard(self):
    g = tf.Graph()
    with g.as_default():
      categories = tf.placeholder(shape=[None], dtype=tf.int64)
      pos_indices, neg_indices = mine_hard_examples(categories)

    with self.test_session(graph=g) as sess:
      # Case 1.
      pos, neg = sess.run([pos_indices, neg_indices],
          feed_dict={categories: np.array([0, 1, 2, 0, 2])})
      self.assertAllEqual(pos, 
          np.array([1, 1, 2, 4]))
      self.assertAllEqual(neg, 
          np.array([2, 4, 1, 1]))

      # Case 2.
      pos, neg = sess.run([pos_indices, neg_indices],
          feed_dict={categories: np.array([1, 3, 0, 2, 3, 4])})
      self.assertAllEqual(pos, np.array([0, 0, 0, 0, 1, 1, 1, 3, 3, 3, 3, 4,
            4, 4, 5, 5, 5, 5]))
      self.assertAllEqual(neg, np.array([1, 3, 4, 5, 0, 3, 5, 0, 1, 4, 5, 0,
            3, 5, 0, 1, 3, 4]))


if __name__ == '__main__':
    tf.test.main()

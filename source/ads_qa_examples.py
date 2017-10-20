
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import logging
import tensorflow as tf

from tensorflow.contrib.slim.python.slim.data import dataset
from tensorflow.contrib.slim.python.slim.data import dataset_data_provider
from tensorflow.contrib.slim.python.slim.data import tfexample_decoder

from protos import ads_qa_examples_pb2

_NUM_EXAMPLES=30000

def _create_tfrecord_dataset(config):
  """Create tfrecord dataset for DatasetDataProvider.

  Args:
    config: an instance of AdsQAExample proto.

  Returns:
    dataset: a slim.data.dataset.Dataset instance.
  """
  keys_to_features = {
    'image/source_id': tf.FixedLenFeature(
        shape=(), dtype=tf.string, default_value=''),
    'image/encoded': tf.FixedLenFeature((), tf.string, default_value=''),
    'image/format': tf.FixedLenFeature((), tf.string, default_value='jpeg'),
    'topic/topic_id': tf.FixedLenFeature(
        shape=(), dtype=tf.int64, default_value=0),
    'caption/num_captions': tf.FixedLenFeature(shape=(), dtype=tf.int64),
    'caption/caption_lengths': tf.FixedLenFeature(
        shape=[config.max_num_captions], dtype=tf.int64),
    'caption/caption_strings': tf.FixedLenFeature(
        shape=[config.max_num_captions, config.max_caption_len], dtype=tf.int64)
  }
  items_to_handlers = {
    'image_id': tfexample_decoder.Tensor('image/source_id'),
    'image': tfexample_decoder.Image(
        shape=[config.image_height, config.image_width, 3]),
    'topic': tfexample_decoder.Tensor('topic/topic_id'),
    'num_captions': tfexample_decoder.Tensor('caption/num_captions'),
    'caption_lengths': tfexample_decoder.Tensor('caption/caption_lengths'),
    'caption_strings': tfexample_decoder.Tensor('caption/caption_strings'),
  }

  if config.export_feature:
    if not config.image_level_feature:
      # Treat each entities in an image individually.
      keys_to_features['entity/num_entities'] = tf.FixedLenFeature(
          shape=(), dtype=tf.int64)
      keys_to_features['entity/embeddings'] = tf.FixedLenFeature(
          shape=[config.max_detections, config.feature_dimentions],
          dtype=tf.float32)
      keys_to_features['entity/scores'] = tf.FixedLenFeature(
          shape=[config.max_detections], dtype=tf.float32,
          default_value=[1.0] * config.max_detections)
      items_to_handlers['num_detections'] = tfexample_decoder.Tensor(
          'entity/num_entities')
      items_to_handlers['proposed_features'] = tfexample_decoder.Tensor(
          'entity/embeddings')
      items_to_handlers['proposed_scores'] = tfexample_decoder.Tensor(
          'entity/scores')

    else:
      # Treat the whole image as an single entity.
      keys_to_features['entity/fake_num'] = tf.FixedLenFeature(
          shape=(), dtype=tf.int64, default_value=1)
      keys_to_features['image/embeddings'] =  tf.FixedLenFeature(
          shape=[1, config.feature_dimentions], dtype=tf.float32)
      keys_to_features['entity/fake_score'] =  tf.FixedLenFeature(
          shape=[1], dtype=tf.float32, default_value=[1.0])
      items_to_handlers['num_detections'] = tfexample_decoder.Tensor(
          'entity/fake_num')
      items_to_handlers['proposed_features'] = tfexample_decoder.Tensor(
          'image/embeddings')
      items_to_handlers['proposed_scores'] = tfexample_decoder.Tensor(
          'entity/fake_score')

  if config.export_densecap_captions:
    keys_to_features['densecap_caption/num_captions'] = tf.FixedLenFeature(
        shape=(), dtype=tf.int64)
    keys_to_features['densecap_caption/caption_lengths'] = tf.FixedLenFeature(
        shape=[config.densecap_max_num_captions], dtype=tf.int64)
    keys_to_features['densecap_caption/caption_strings'] = tf.FixedLenFeature(
        shape=[config.densecap_max_num_captions, config.densecap_max_caption_len], 
        dtype=tf.int64)
    items_to_handlers['densecap_num_captions'] = tfexample_decoder.Tensor(
        'densecap_caption/num_captions')
    items_to_handlers['densecap_caption_lengths'] = tfexample_decoder.Tensor(
        'densecap_caption/caption_lengths')
    items_to_handlers['densecap_caption_strings'] = tfexample_decoder.Tensor(
        'densecap_caption/caption_strings')

  decoder = tfexample_decoder.TFExampleDecoder(
      keys_to_features, items_to_handlers)

  return dataset.Dataset(
      data_sources=config.input_path,
      reader=tf.TFRecordReader,
      decoder=decoder,
      num_samples=config.num_examples,
      items_to_descriptions=None)


def get_examples(config):
  """Get batched tensor of training data.

  Args:
    config: an instance of AdsQAExample proto.

  Returns:
    tensor_dict: a dictionary mapping data names to tensors.
      'image_id':          tf.string,  [batch]
      'image':             tf.uint8    [batch, height, width, 3]
      'topic':             tf.int64,   [batch]
      'num_captions':      tf.int64,   [batch]
      'caption_lengths':   tf.int64,   [batch, max_num_captions]
      'caption_strings':   tf.int64,   [batch, max_num_captions, max_caption_len]

      If config.export_feature is True:
      'num_detections':    tf.int64,   [batch]
      'proposed_features': tf.float32, [batch, max_detections, feature_dimentions]

      If config.export_densecap_captions is True:
      'densecap_num_captions':      tf.int64,   [batch]
      'densecap_caption_lengths':   tf.int64,   [batch, densecap_max_num_captions]
      'densecap_caption_strings':   tf.int64,   [batch, densecap_max_num_captions, densecap_max_caption_len]

  Raises:
    ValueError: if config is invalid.
  """
  if not isinstance(config, ads_qa_examples_pb2.AdsQAExamples):
    raise ValueError('config has to be an instance of AdsQAExamples.')

  dataset = _create_tfrecord_dataset(config)
  provider = dataset_data_provider.DatasetDataProvider(
      dataset,
      shuffle=True,
      num_readers=config.data_provider_num_readers,
      common_queue_capacity=config.data_provider_common_queue_capacity,
      common_queue_min=config.data_provider_common_queue_min)

  # TODO(yek@): why record_key is there?
  items = provider.list_items()
  items = filter(lambda x: x != 'record_key', items)

  data = provider.get(items)
  data = tf.train.batch(data,
      config.batch_size, 
      enqueue_many=False, 
      num_threads=config.batch_op_num_threads, 
      capacity=config.batch_op_capacity)

  name_dict = {}
  tensor_dict = {}
  for item, tensor in zip(items, data):
    tensor_dict[name_dict.get(item, item)] = tensor
  return tensor_dict

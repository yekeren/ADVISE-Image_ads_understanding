
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import logging
import tensorflow as tf

from tensorflow.contrib.slim.python.slim.data import dataset
from tensorflow.contrib.slim.python.slim.data import dataset_data_provider
from tensorflow.contrib.slim.python.slim.data import tfexample_decoder

import ads_emb_model_pb2
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
    'topic_id': tfexample_decoder.Tensor('topic/topic_id'),
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
      items_to_handlers['num_entities'] = tfexample_decoder.Tensor(
          'entity/num_entities')
      items_to_handlers['embeddings'] = tfexample_decoder.Tensor(
          'entity/embeddings')

    else:
      # Treat the whole image as an single entity.
      keys_to_features['entity/fake_one'] = tf.FixedLenFeature(
          shape=(), dtype=tf.int64, default_value=1)
      keys_to_features['image/embeddings'] =  tf.FixedLenFeature(
          shape=[1, config.feature_dimentions], dtype=tf.float32)
      items_to_handlers['num_entities'] = tfexample_decoder.Tensor(
          'entity/fake_one')
      items_to_handlers['embeddings'] = tfexample_decoder.Tensor(
          'image/embeddings')

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
      'topic_id':          tf.int64,   [batch]
      'num_captions':      tf.int64,   [batch]
      'caption_lengths':   tf.int64,   [batch, max_num_captions]
      'caption_strings':   tf.int64,   [batch, max_num_captions, max_caption_len]
      'num_detections':    tf.int64,   [batch]
      'proposed_features': tf.float32, [batch, max_detections, feature_dimentions]

  Raises:
    ValueError: if config is invalid.
  """
  if not isinstance(config, ads_emb_model_pb2.AdsQAExamples):
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

  name_dict = {
    'num_entities': 'num_detections',
    'embeddings': 'proposed_features',
  }
  tensor_dict = {}
  for item, tensor in zip(items, data):
    tensor_dict[name_dict.get(item, item)] = tensor
  return tensor_dict

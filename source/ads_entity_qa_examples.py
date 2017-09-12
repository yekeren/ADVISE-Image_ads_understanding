
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import logging
import tensorflow as tf

from tensorflow.contrib.slim.python.slim.data import dataset
from tensorflow.contrib.slim.python.slim.data import dataset_data_provider
from tensorflow.contrib.slim.python.slim.data import tfexample_decoder

_NUM_EXAMPLES=30000

def _create_tfrecord_dataset(filename):
  keys_to_features = {
    'image/source_id': tf.FixedLenFeature(shape=(), dtype=tf.string, default_value=''),
    'entity/num_entities': tf.FixedLenFeature(shape=(), dtype=tf.int64),
    'entity/embeddings': tf.FixedLenFeature(shape=[10, 1536], dtype=tf.float32),
    'caption/num_captions': tf.FixedLenFeature(shape=(), dtype=tf.int64),
    'caption/caption_lengths': tf.FixedLenFeature(shape=[5], dtype=tf.int64),
    'caption/caption_strings': tf.FixedLenFeature(shape=[5, 30], dtype=tf.int64)
  }
  items_to_handlers = {
    'image_id': tfexample_decoder.Tensor('image/source_id'),
    'num_entities': tfexample_decoder.Tensor('entity/num_entities'),
    'embeddings': tfexample_decoder.Tensor('entity/embeddings'),
    'num_captions': tfexample_decoder.Tensor('caption/num_captions'),
    'caption_lengths': tfexample_decoder.Tensor('caption/caption_lengths'),
    'caption_strings': tfexample_decoder.Tensor('caption/caption_strings'),
  }

  decoder = tfexample_decoder.TFExampleDecoder(
      keys_to_features, items_to_handlers)

  return dataset.Dataset(
      data_sources=filename,
      reader=tf.TFRecordReader,
      decoder=decoder,
      num_samples=_NUM_EXAMPLES,
      items_to_descriptions=None)

def get_examples(filename, batch_size=32):
  dataset = _create_tfrecord_dataset(filename)
  provider = dataset_data_provider.DatasetDataProvider(dataset)

  data = provider.get(['image_id', 'num_entities', 'embeddings', 
      'num_captions', 'caption_lengths', 'caption_strings'])
  data = tf.train.batch(data, batch_size, 
      num_threads=4, enqueue_many=False, capacity=320)

  return {
    'image_id': data[0],
    'num_entities': data[1],
    'embeddings': data[2],
    'num_captions': data[3],
    'caption_lengths': data[4],
    'caption_strings': data[5],
  }


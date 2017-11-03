
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import logging
import tensorflow as tf

from tensorflow.contrib.slim.python.slim.data import dataset
from tensorflow.contrib.slim.python.slim.data import dataset_data_provider
from tensorflow.contrib.slim.python.slim.data import tfexample_decoder

from protos import preprocess_pb2
from protos import ads_qa_examples_pb2

_NUM_EXAMPLES=30000

def _random_crop(image, random_crop_min_scale):
  """Random crop image according to the minimum scale requirement.
  Args:
    image: a [height, width, 3] image tensor.
    random_crop_min_scale: minimum scale requirement.
  Returns:
    image: a [crop_height, crop_width, 3] image tensor.
  """
  height, width, _ = image.get_shape().as_list()

  min_height = int(height * random_crop_min_scale + 0.5)
  min_width = int(width * random_crop_min_scale + 0.5)

  # Randomize the crop box.
  target_height = tf.random_uniform(shape=[], 
      dtype=tf.int32, minval=min_height, maxval=height)
  target_width = tf.random_uniform(shape=[], 
      dtype=tf.int32, minval=min_width, maxval=width)

  offset_height = tf.random_uniform(shape=[], 
      dtype=tf.int32, minval=0, maxval=height - target_height)
  offset_width = tf.random_uniform(shape=[], 
      dtype=tf.int32, minval=0, maxval=width - target_width)

  # Crop the image and compute the normalized crop box.
  image = tf.image.crop_to_bounding_box(image, 
      offset_height, offset_width, target_height, target_width)
  image = tf.image.resize_images(image, size=[height, width])
  return image

def _preprocess(image, options):
  """Preprocess image.

  Args:
    image: a [height, width, 3] uint8 tensor.
    options: an instance of PreprocessOptions.

  Returns:
    preprocessed_image: a [height, width, 3] uint8 tensor.

  Raises:
    ValueError: if options is invalid
  """
  if not isinstance(options, preprocess_pb2.PreprocessOptions):
    raise ValueError('invalid options')
  image = tf.saturate_cast(image, dtype=tf.float32)

  # Image process.
  image = tf.cond( 
      tf.less(tf.random_uniform(shape=[]), options.random_flip_left_right_prob),
      true_fn=lambda: tf.image.flip_left_right(image), false_fn=lambda: image)

  image = tf.cond( 
      tf.less(tf.random_uniform(shape=[]), options.random_brightness_prob),
      true_fn=lambda: tf.image.random_brightness(
        image, max_delta=options.random_brightness_max_delta), 
      false_fn=lambda: image)

  image = tf.cond( 
      tf.less(tf.random_uniform(shape=[]), options.random_hue_prob),
      true_fn=lambda: tf.image.random_hue(
        image, max_delta=options.random_hue_max_delta),
      false_fn=lambda: image)

  image = tf.cond( 
      tf.less(tf.random_uniform(shape=[]), options.random_contrast_prob),
      true_fn=lambda: tf.image.random_contrast(image, 
        lower=options.random_contrast_lower, 
        upper=options.random_contrast_upper), 
      false_fn=lambda: image)

  # Random crop.
  image = tf.cond(
      tf.less(tf.random_uniform(shape=[]), options.random_crop_prob),
      true_fn=lambda: _random_crop(image, options.random_crop_min_scale), 
      false_fn=lambda: image)

  image = tf.saturate_cast(image, dtype=tf.uint8)
  return image

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
#    'image/encoded': tf.FixedLenFeature((), tf.string, default_value=''),
#    'image/format': tf.FixedLenFeature((), tf.string, default_value='jpeg'),
    'topic/topic_id': tf.FixedLenFeature(
        shape=(), dtype=tf.int64, default_value=0),
    'caption/num_captions': tf.FixedLenFeature(shape=(), dtype=tf.int64),
    'caption/caption_lengths': tf.FixedLenFeature(
        shape=[config.max_num_captions], dtype=tf.int64),
    'caption/caption_strings': tf.FixedLenFeature(
        shape=[config.max_num_captions, config.max_caption_len], dtype=tf.int64),
    'symbols/num_symbols': tf.FixedLenFeature(
        shape=(), dtype=tf.int64, default_value=0),
    'symbols/symbol_ids': tf.FixedLenFeature(
        shape=[config.max_num_symbols], dtype=tf.int64),

  }
  items_to_handlers = {
    'image_id': tfexample_decoder.Tensor('image/source_id'),
#    'image': tfexample_decoder.Image(
#        shape=[config.image_height, config.image_width, 3]),
    'topic': tfexample_decoder.Tensor('topic/topic_id'),
    'num_captions': tfexample_decoder.Tensor('caption/num_captions'),
    'caption_lengths': tfexample_decoder.Tensor('caption/caption_lengths'),
    'caption_strings': tfexample_decoder.Tensor('caption/caption_strings'),
    'num_symbols': tfexample_decoder.Tensor('symbols/num_symbols'),
    'symbols': tfexample_decoder.Tensor('symbols/symbol_ids'),
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


def _read_oversample_factors(file_path):
  """Read oversample factors from file.

  Args:
    file_path: path to the config file.

  Returns:
    oversample_factors: a python list mapping from key to oversample factors.
  """
  with open(file_path, 'r') as fp:
    lines = fp.readlines()

  oversample_factors = [1.0] * len(lines)
  for line in lines:
    key, factor = line.strip('\n').split('\t')
    oversample_factors[int(key)] = float(factor)
  return oversample_factors


def _oversample(data, key, oversample_factors):
  """Process oversampling on the input data.

  For factor of '2.6', the function puts 2 examples into the queue, 
  and then the other 1 example based on probability 0.6.

  Args:
    data: a list of tensor object.
    key: an [] int64 tensor denoting the key.
    oversample_factors: a python dict mapping from key to oversample factors.

  Returns:
    data: oversampled data.
  """
  factor = tf.constant(oversample_factors)[key]
  base = tf.cast(factor, tf.int32)
  prob = factor - tf.cast(base, tf.float32)

  replicas = base + tf.cond(
      tf.less(tf.random_uniform(shape=[], dtype=tf.float32), prob),
      true_fn=lambda: 1 + base, false_fn=lambda: base)

  for i in xrange(len(data)):
    shape = [replicas] + [1] * (len(data[i].get_shape()) - 1)
    data[i] = tf.tile(data[i], shape)
  return data


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
  if config.HasField('preprocess_options'):
    for i, item in enumerate(items):
      if item == 'image':
        data[i] = _preprocess(data[i], config.preprocess_options)

  # Expand dimensions and process over-sampling if necessary.
  data = [tf.expand_dims(x, 0) for x in data]

  if config.HasField('oversample_config_path'):
    # Oversample data based on the topic.
    oversample_factors = _read_oversample_factors(
        config.oversample_config_path)

    topic_column_id = -1
    for i, item in enumerate(items):
      if item == 'topic':
        topic_column_id = i
    assert topic_column_id >= 0

    data = _oversample(data, data[topic_column_id][0], oversample_factors)

  # Shuffle batch.
  data = tf.train.shuffle_batch(data,
      config.batch_size, 
      enqueue_many=True, 
      num_threads=config.batch_op_num_threads, 
      capacity=config.batch_op_capacity,
      min_after_dequeue=config.batch_op_min_after_dequeue)

  name_dict = {}
  tensor_dict = {}
  for item, tensor in zip(items, data):
    tensor_dict[name_dict.get(item, item)] = tensor

  # TODO(yek@): fix bug when create tf record.
  tensor_dict['num_symbols'] = tf.minimum(
      tensor_dict['num_symbols'], config.max_num_symbols)
  return tensor_dict

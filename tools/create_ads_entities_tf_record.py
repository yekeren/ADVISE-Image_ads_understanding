
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import hashlib
import logging
import cv2

import numpy as np
import tensorflow as tf

from object_detection.utils import dataset_util

from utils import image_coder
from utils.ads_dataset_api import AdsDatasetApi
from utils.ads_dataset_api import is_training_example
from utils.ads_dataset_api import is_validation_example
from utils.ads_dataset_api import is_testing_example

flags = tf.app.flags
flags.DEFINE_string('data_dir', '', 'Directory to ads dataset.')
flags.DEFINE_string('emb_path', '', 'Path to the feature data file.')
flags.DEFINE_string('entity_annot_path', '', 'File path to entity annotations.')

flags.DEFINE_string('output_dir', '', 'Directory to output TFRecord')

FLAGS = flags.FLAGS
coder = image_coder.ImageCoder()


def _get_data(raw_embs, export_type='training'):
  examples = []
  for k, v in raw_embs.iteritems():
    example = {'image_id': k}
    if export_type == 'training' and is_training_example(k):
      example['entity_emb_list'] = v['entity_emb_list']
      examples.append(example)
    if export_type == 'validation' and is_validation_example(k):
      example['entity_emb_list'] = v['entity_emb_list']
      examples.append(example)
    if export_type == 'testing' and is_testing_example(k):
      example['entity_emb_list'] = v['entity_emb_list']
      examples.append(example)

  logging.info('Loaded %s images for %s.', len(examples), export_type)
  return examples


def dict_to_tf_example(data):
  example = tf.train.Example(features=tf.train.Features(feature={
        'image/source_id': dataset_util.bytes_feature(data['image_id'].encode('utf8')),
        'image/entity/source_id': dataset_util.int64_feature(data['entity_id']),
        'image/entity/embedding': dataset_util.float_list_feature(data['entity_emb'].tolist()),
        }))
  return example


def create_tf_record(output_path, emb_list):
  writer = tf.python_io.TFRecordWriter(output_path)
  for elem_index, elem in enumerate(emb_list):
    if elem_index % 100 == 0:
      logging.info('On image %d of %d.' % (elem_index, len(emb_list)))

    image_id = elem['image_id']
    entity_emb_list = elem['entity_emb_list']

    for entity_id, entity_emb in enumerate(entity_emb_list):
      example = {
        'image_id': image_id,
        'entity_id': entity_id,
        'entity_emb': entity_emb
      }
      tf_example = dict_to_tf_example(example)
      writer.write(tf_example.SerializeToString())
  writer.close()


def main(_):
  logging.basicConfig(level=logging.DEBUG)

  # Read data file.
  raw_embs = np.load(FLAGS.emb_path).item()
  train_embs = _get_data(raw_embs, export_type='training')
  valid_embs = _get_data(raw_embs, export_type='validation')
  test_embs = _get_data(raw_embs, export_type='testing')

  train_output_path = os.path.join(FLAGS.output_dir,
      'ads_entities.train.record')
  val_output_path = os.path.join(FLAGS.output_dir,
      'ads_entities.val.record')

  create_tf_record(train_output_path, train_embs)
  create_tf_record(val_output_path, valid_embs)

if __name__ == '__main__':
  tf.app.run()

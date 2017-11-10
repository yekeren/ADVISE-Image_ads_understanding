
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import cv2

import numpy as np
import tensorflow as tf

from utils.ads_dataset_api import AdsDatasetApi
from utils.ads_dataset_api import is_training_example
from utils.ads_dataset_api import is_validation_example
from utils.ads_dataset_api import is_testing_example

flags = tf.app.flags
flags.DEFINE_string('data_dir', 'raw_data/ads/images', 'Directory to ads dataset.')
flags.DEFINE_string('topic_list_file', 'raw_data/ads/annotations/Topics_List.txt', 'File path to ads topics list.')
flags.DEFINE_string('topic_annot_file', 'raw_data/ads/annotations/Topics.json', 'File path to ads topics annotations.')

FLAGS = flags.FLAGS


def main(_):
  tf.logging.set_verbosity(tf.logging.INFO)

  api = AdsDatasetApi()
  api.init(
      images_dir=FLAGS.data_dir,
      topic_list_file=FLAGS.topic_list_file,
      topic_annot_file=FLAGS.topic_annot_file)

  data = {}

  meta_list = api.get_meta_list()
  for meta in meta_list:
    if 'topic_id' in meta:
      topic_id, topic_name = meta['topic_id'], meta['topic_name']
      topic = data.setdefault(topic_id, {
          'topic_id': topic_id, 
          'topic_name': topic_name, 
          'count': 0})
      topic['count'] += 1

  data = sorted(data.values(), lambda x, y: cmp(x['topic_id'], y['topic_id']))
  max_v = max([x['count'] for x in data])
  for x in data:
    factor = 1.0 + float(np.log2(1.0 * max_v / x['count']))
    if x['topic_id'] == 0: factor = 1.0
    #print('%s\t%s\t%s\t%.2lf' % (x['topic_id'], x['topic_name'], x['count'], factor))
    print('%s\t%.2lf' % (x['topic_id'], factor))

if __name__ == '__main__':
  tf.app.run()

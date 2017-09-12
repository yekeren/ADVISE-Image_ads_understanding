from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import cv2
import logging
import numpy as np

from sklearn.externals import joblib
from sklearn.cluster import MiniBatchKMeans
import tensorflow as tf
from tensorflow.python.training import input as input_lib
from tensorflow.python.training import queue_runner

from utils.ads_dataset_api import is_training_example
from utils.ads_dataset_api import is_validation_example

flags = tf.app.flags
flags.DEFINE_integer('num_clusters', 100, 'Number of clusters.')
flags.DEFINE_integer('batch_size', 10000, 'Batch size for training.')
flags.DEFINE_integer('num_epochs', 50, 'Number of epochs to run.')
flags.DEFINE_string('data_path', '', 'Path to the feature data file.')
flags.DEFINE_string('model_path', '', 'Path to the model dir.')

FLAGS = flags.FLAGS
slim = tf.contrib.slim


def _get_data(raw_data, is_training):
  examples = []
  for k, v in raw_data.iteritems():
    if is_training and is_training_example(k):
      examples.append(v['entity_emb_list'])
    if not is_training and is_validation_example(k):
      examples.append(v['entity_emb_list'])

  purpose = 'training' if is_training else 'validation'
  logging.info('Loaded %s images for %s.', len(examples), purpose)
  examples = np.concatenate(examples, axis=0)
  logging.info('Loaded %s patches for %s.', len(examples), purpose)

  return examples

def main(_):
  logging.basicConfig(level=logging.DEBUG)

  # Read data file.
  raw_data = np.load(FLAGS.data_path).item()
  training_data = _get_data(raw_data, is_training=True)
  validation_data = _get_data(raw_data, is_training=False)

  kmeans = MiniBatchKMeans(n_clusters=FLAGS.num_clusters, 
      max_iter=FLAGS.num_epochs, 
      batch_size=FLAGS.batch_size,
      max_no_improvement=None, 
      verbose=1)

  kmeans.fit(training_data)

  joblib.dump(kmeans, FLAGS.model_path)

if __name__ == '__main__':
  tf.app.run()

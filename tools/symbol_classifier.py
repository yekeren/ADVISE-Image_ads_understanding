
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
import random

import numpy as np
import tensorflow as tf
from google.protobuf import text_format

from sklearn.metrics import average_precision_score

from utils.ads_api import AdsApi

from protos import feature_extractors_pb2
from feature_extractors import builder
from feature_extractors import fc_extractor

flags = tf.app.flags
flags.DEFINE_string('feature_path', 'output4/inception_v4_features.coco.npz', 'Path to the feature data file.')
flags.DEFINE_string('api_config', 'configs/ads_api.ads.config.0', 'Path to config file.')
flags.DEFINE_integer('max_iters', 8000, 'Maximum iterations.')
flags.DEFINE_string('output_path', 'symbols/symbols.0.npz', 'Path to the output file.')

FLAGS = flags.FLAGS
slim = tf.contrib.slim

num_categories = None

config_str = """
  fc_extractor: {
    hidden_hyperparams {
      op: FC
      activation: RELU_6
      regularizer {
        l2_regularizer {
          weight: 0.0
        }
      }
      initializer {
        truncated_normal_initializer {
          stddev: 0.03
          mean: 0.0
        }
      }
    }
    output_hyperparams {
      op: FC
      activation: NONE
      regularizer {
        l2_regularizer {
          weight: 0.0
        }
      }
      initializer {
        truncated_normal_initializer {
          stddev: 0.03
          mean: 0.0
        }
      }
    }
    input_dropout_keep_prob: 0.3
    hidden_layers: 1
    hidden_units: 256
    hidden_dropout_keep_prob: 0.3
    output_units: 0
  }
"""

def _get_feature(file_path):
  """Read feature dict from file.
  """
  feature_data = np.load(file_path).item()
  for key in feature_data.keys():
    if type(feature_data[key]) == dict:
      feature_data[key] = feature_data[key]['image_emb']
  tf.logging.info('Load %s feature vectors.', len(feature_data))
  return feature_data


def _get_data(meta_list, feature_data, split):
  """Splits data to get train, valid, or test partition.

  Args:
    meta_list: a python list containing meta info for all images.
    feature_data: a mapping from image_id to embedding vector.
    split: 'train', 'valid', or 'test'

  Returns:
    x: a [batch, emb_size] numpy array denoting features.
    y: a [batch] numpy array indicating labels.
  """
  meta_list = [meta for meta in meta_list \
              if meta['split'] == split and 'symbol_ids' in meta]

  dims = feature_data[meta_list[0]['image_id']].shape[0]
  x = np.zeros((len(meta_list), dims))
  y = np.zeros((len(meta_list), num_categories))

  for meta_index, meta in enumerate(meta_list):
    x[meta_index] = feature_data[meta['image_id']]
    for symbol_id in meta['symbol_ids']:
      y[meta_index, symbol_id] = 1

  tf.logging.info('%s: x_shape=%s, y_shape=%s', split, x.shape, y.shape)
  return x, y

def _get_data_for_inference(meta_list, feature_data):
  """Splits data to get train, valid, or test partition.

  Args:
    meta_list: a python list containing meta info for all images.
    feature_data: a mapping from image_id to embedding vector.

  Returns:
    x: a [batch, emb_size] numpy array denoting features.
    y: a [batch] numpy array indicating labels.
  """
  dims = feature_data[meta_list[0]['image_id']].shape[0]
  image_ids = []
  x = np.zeros((len(meta_list), dims))

  for meta_index, meta in enumerate(meta_list):
    x[meta_index] = feature_data[meta['image_id']]
    image_ids.append(meta['image_id'])
  return image_ids, x

def _default_session_config_proto():
  """Get the default config proto for tensorflow session.

  Returns:
    config: The default config proto for tf.Session.
  """
  config = tf.ConfigProto()
  config.allow_soft_placement = True
  config.gpu_options.allow_growth = True
  config.gpu_options.per_process_gpu_memory_fraction = 0.5
  return config


def _train_tf_model(x_train, y_train, x_valid, y_valid, x_test, y_test,
    image_ids, x_data):
  """Trains a tensorflow mlp model with dropout layer.
  """
  config = feature_extractors_pb2.FeatureExtractor()
  text_format.Merge(config_str, config)
  config.fc_extractor.output_units = num_categories

  # Build training graph.
  g = tf.Graph()
  with g.as_default():
    feature_extractor = builder.build(config)
    assert isinstance(feature_extractor, fc_extractor.FCExtractor)

    input_data = tf.placeholder(
        shape=[None, x_train.shape[1]], dtype=tf.float32)
    label_data = tf.placeholder(
        shape=[None, num_categories], dtype=tf.float32)

    batch_size = tf.shape(input_data)[0]
    logits = feature_extractor.extract_feature(input_data)

    loss_op = tf.nn.sigmoid_cross_entropy_with_logits(
        labels=label_data, logits=logits)
    loss_op = tf.reduce_mean(loss_op)

    optimizer = tf.train.AdamOptimizer(0.0005)
    #optimizer = tf.train.AdagradOptimizer(2.0)
    train_op = slim.learning.create_train_op(loss_op, optimizer=optimizer)
    init_op = tf.global_variables_initializer()

    # Eval tensor.
    tf.get_variable_scope().reuse_variables()
    logits_eval = tf.sigmoid(feature_extractor.extract_feature(
        input_data, is_training=False))

  batch_size = 64
  with tf.Session(graph=g, config=_default_session_config_proto()) as sess:
    sess.run(init_op)
    for i in xrange(FLAGS.max_iters):
      rand_ind = random.randint(0, len(y_train) - 1 - batch_size)
      _, loss = sess.run([train_op, loss_op], 
          feed_dict={
              input_data: x_train[rand_ind: rand_ind+batch_size, :], 
              label_data: y_train[rand_ind: rand_ind+batch_size, :]})
      if i % 500 == 0:
        y_pred = sess.run(logits_eval, feed_dict={input_data: x_train})
        mAP = average_precision_score(
            y_train[:, 1:], y_pred[:, 1:], average='macro')
        tf.logging.info('step=%d, split=train, mAP=%.4lf', i, mAP)

        y_pred = sess.run(logits_eval, feed_dict={input_data: x_valid})
        mAP = average_precision_score(
            y_valid[:, 1:], y_pred[:, 1:], average='macro')
        tf.logging.info('step=%d, split=valid, mAP=%.4lf', i, mAP)
        #precision_valid = 1.0 * (pred.argmax(1) == y_valid).sum() / x_valid.shape[0]

    # Inference.
    results = sess.run(logits_eval, feed_dict={input_data: x_data})
    output = {
      'image_ids': image_ids,
      'symbols_data': results
    }
    tf.logging.info('Inferred data: %s.', results.shape)
    with open(FLAGS.output_path, 'wb') as fp:
      np.save(fp, output)


def main(_):
  global num_categories
  tf.logging.set_verbosity(tf.logging.INFO)

  api = AdsApi(FLAGS.api_config)
  meta_list = api.get_meta_list()
  tf.logging.info('Load %s meta records using ads api.', len(meta_list))

  symbol_to_name = api.get_symbol_to_name()
  num_categories = len(symbol_to_name)
  tf.logging.info('Total #categories: %s', num_categories)

  # Load train, valid, and test data.
  feature_data = _get_feature(FLAGS.feature_path)

  x_train, y_train = _get_data(meta_list, feature_data, split='train')
  x_valid, y_valid = _get_data(meta_list, feature_data, split='valid')
  x_test, y_test = _get_data(meta_list, feature_data, split='test')

  image_ids, x_data = _get_data_for_inference(meta_list, feature_data)

  # Using tensorflow.
  _train_tf_model(x_train, y_train, x_valid, y_valid, x_test, y_test,
      image_ids, x_data)

if __name__ == '__main__':
  tf.app.run()

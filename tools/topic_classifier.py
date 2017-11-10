
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys

import numpy as np
import tensorflow as tf
from google.protobuf import text_format

import sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier

from utils.ads_api import AdsApi

from protos import feature_extractors_pb2
from feature_extractors import builder
from feature_extractors import fc_extractor

flags = tf.app.flags
flags.DEFINE_string('feature_path', 'output/image_emb.npz', 'Path to the feature data file.')
flags.DEFINE_string('api_config', 'configs/ads_api_topics.config', 'Path to config file.')
flags.DEFINE_integer('max_iters', 10000, 'Maximum iterations.')

FLAGS = flags.FLAGS
slim = tf.contrib.slim

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
    input_dropout_keep_prob: 0.8
    hidden_layers: 0
    hidden_units: 100
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
  meta_list = [meta for meta in meta_list if meta['split'] == split]

  x, y = [], []
  for meta in meta_list:
    # Filters out 'unclear' examples.
    if meta.get('topic_id', 0) > 0: 
      x.append(np.expand_dims(feature_data[meta['image_id']], axis=0))
      y.append(meta['topic_id'])

  x = np.concatenate(x, axis=0)
  y = np.array(y)
  tf.logging.info('%s: x_shape=%s, y_shape=%s', split, x.shape, y.shape)

  return x, y

def _train_sklearn_model(num_categories, 
    x_train, y_train, x_valid, y_valid, x_test, y_test):
  """Trains a sklearn mlp model.
  """
  model = MLPClassifier(
      verbose=True, 
      hidden_layer_sizes=(), 
      learning_rate_init=0.001,
      early_stopping=True)

  model.fit(
      np.concatenate([x_train, x_valid], 0), 
      np.concatenate([y_train, y_valid], 0))

  y_valid_predict = model.predict(x_valid)
  y_test_predict = model.predict(x_test)
  score_valid = sklearn.metrics.precision_score(
      y_valid, y_valid_predict, average='micro')
  score_test = sklearn.metrics.precision_score(
      y_test, y_test_predict, average='micro')

  tf.logging.info('Valid precision: %.4lf', score_valid)
  tf.logging.info('Test precision: %.4lf', score_test)


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


def _train_tf_model(num_categories,
    x_train, y_train, x_valid, y_valid, x_test, y_test):
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
    label_data = tf.placeholder(shape=[None], dtype=tf.int32)

    batch_size = tf.shape(input_data)[0]
    logits = feature_extractor.extract_feature(input_data)
    #loss_op = tf.nn.sparse_softmax_cross_entropy_with_logits(
    #    labels=label_data,
    #    logits=logits)
    loss_op = tf.nn.sigmoid_cross_entropy_with_logits(
        labels=tf.sparse_to_dense(
          tf.stack([tf.range(batch_size, dtype=tf.int32), label_data], 1), 
          [batch_size, num_categories], 1.0),
        logits=logits)
    loss_op = tf.reduce_mean(loss_op)

    optimizer = tf.train.AdamOptimizer(0.05)
    #optimizer = tf.train.AdagradOptimizer(2.0)
    train_op = slim.learning.create_train_op(loss_op, optimizer=optimizer)
    init_op = tf.global_variables_initializer()

    # Eval tensor.
    tf.get_variable_scope().reuse_variables()
    logits_eval = feature_extractor.extract_feature(
        input_data, is_training=False)

  with tf.Session(graph=g, config=_default_session_config_proto()) as sess:
    sess.run(init_op)
    max_valid = 0.0
    result = 0.0
    for i in xrange(FLAGS.max_iters):
      _, loss = sess.run([train_op, loss_op], 
          feed_dict={input_data: x_train, label_data: y_train})
      if i % 100 == 0:
        pred = sess.run(logits_eval, feed_dict={input_data: x_train})
        precision_train = 1.0 * (pred.argmax(1) == y_train).sum() / x_train.shape[0]

        pred = sess.run(logits_eval, feed_dict={input_data: x_valid})
        precision_valid = 1.0 * (pred.argmax(1) == y_valid).sum() / x_valid.shape[0]

        pred = sess.run(logits_eval, feed_dict={input_data: x_test})
        precision_test = 1.0 * (pred.argmax(1) == y_test).sum() / x_test.shape[0]
        tf.logging.info('step=%d, train=%.4lf, valid=%.4lf, test=%.4lf', 
            i, precision_train, precision_valid, precision_test)

        if precision_valid > max_valid:
          max_valid = precision_valid
          result = precision_test
    tf.logging.info('Precision: %.4lf', result)


def main(_):
  tf.logging.set_verbosity(tf.logging.INFO)

  api = AdsApi(FLAGS.api_config)
  meta_list = api.get_meta_list()
  tf.logging.info('Load %s meta records using ads api.', len(meta_list))

  topic_to_name = api.get_topic_to_name()
  num_categories = len(topic_to_name)
  tf.logging.info('Total #categories: %s', num_categories)

  # Load train, valid, and test data.
  feature_data = _get_feature(FLAGS.feature_path)

  x_train, y_train = _get_data(meta_list, feature_data, split='train')
  x_valid, y_valid = _get_data(meta_list, feature_data, split='valid')
  x_test, y_test = _get_data(meta_list, feature_data, split='test')

  # Using sklearn.
  #_train_sklearn_model(num_categories, x_train, y_train, x_valid, y_valid, x_test, y_test)

  # Using tensorflow.
  _train_tf_model(num_categories, x_train, y_train, x_valid, y_valid, x_test, y_test)

if __name__ == '__main__':
  tf.app.run()

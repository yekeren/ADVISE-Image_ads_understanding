
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import tensorflow as tf
from tensorflow import logging

from object_detection.builders import hyperparams_builder

from protos import utils_pb2
from protos import train_config_pb2

slim = tf.contrib.slim

def linear(inputs, output_size, use_bias=False, bias_init=0.0, scope=None):
  """Processes linear projection.

  Args:
    inputs: a [batch, input_size] tf.float32 tensor.
    output_size: a int denoting the size of output.
    use_bias: if True, use bias.
    bias_init: initial value of the bias term.
    scope: variable scope.

  Returns:
    outputs: a [batch, output_size] tf.float32 tensor.
  """
  input_size = inputs.get_shape()[-1]

  with tf.variable_scope(scope or 'linear'):
    W = tf.get_variable('weights',
        shape=[input_size, output_size], dtype=tf.float32)
    outputs = tf.matmul(inputs, W)

    if use_bias:
      bias = tf.get_variable('bias',
          shape=[output_size], dtype=tf.float32,
          initializer = tf.constant_initializer(bias_init))
      outputs = tf.nn.bias_add(outputs, bias)
  return outputs


def encode_feature(features, config, is_training=False, reuse=None):
  """Encodes image using the config.

  Args:
    features: a [batch, feature_dimensions] tf.float32 tensor.
    config: an instance of utils_pb2.ImageEncoder.
    is_training: if True, training graph is built.

  Raises:
    ValueError if config is not an instance of ImageEncoder

  Returns:
    features_encoded: a [batch, num_outputs] tf.float32 tensor.
  """
  if not isinstance(config, utils_pb2.FCEncoder):
    raise ValueError('The config has to be an instance of FCEncoder.')

  hp = hyperparams_builder.build(config.fc_hyperparams, is_training=is_training)

  node = features
  node = slim.dropout(node, config.input_dropout_keep_prob, 
      is_training=is_training)
  with slim.arg_scope(hp):
    node = slim.fully_connected(node, config.num_outputs, 
        scope=config.scope, reuse=reuse)
  node = slim.dropout(node, config.output_dropout_keep_prob,
      is_training=is_training)

  return node


def reduce_mean_for_varlen_data(data, lengths):
  """Reduces mean for each row in data.

  Args:
    data: a [batch, max_elems] tf.float32 tensor.
    lengths: a [batch] tf.int64 tensor indicating number of elems in each row.

  Returns:
    mean_vals: a [batch] tf.float32 tensor indicating the mean for each row.
  """
  max_elems = tf.shape(data)[1]
  masks = tf.less(
      tf.range(max_elems, dtype=tf.int32), 
      tf.expand_dims(tf.cast(lengths, tf.int32), 1))

  sum_vals = tf.reduce_sum(
      data * tf.cast(masks, tf.float32), axis=1)
  mean_vals = tf.div(sum_vals, 
      tf.maximum(1e-8, tf.cast(lengths, tf.float32)))
  return mean_vals

def softmax_for_varlen_logits(logits, lengths):
  """Processes softmax on varlen sequences.

  Args:
    logits: a [batch, max_seq_len] tf.float32 tensor.
    lengths: a [batch] tf.int64 tensor indicating the length for each logit.

  Returns:
    proba: a [batch, max_seq_len] tf.float32 tensor indicating the
      probabilities.
  """
  min_val = -1e30

  max_seq_len = tf.shape(logits)[1]
  boolean_masks = tf.greater_equal(
      tf.range(max_seq_len, dtype=tf.int32), 
      tf.expand_dims(tf.cast(lengths, tf.int32), 1))

  padding_values = min_val * tf.cast(boolean_masks, tf.float32)
  return tf.nn.softmax(logits + padding_values)


def sample_from_distribution(probs):
  """Samples data from probability distribution.

  Args:
    probs: a [batch, max_elems] tf.float32 tensor with each value ranging from 0
      to 1.
    lengths: a [batch] tf.int64 tensor indicating number of elems in each row.

  Returns:
    a [batch, max_elems] boolean tensor denoting sampled data.
  """
  dist = tf.distributions.Bernoulli(probs=probs)
  return dist.sample()

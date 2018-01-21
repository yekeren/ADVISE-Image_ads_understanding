
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from object_detection.builders import hyperparams_builder
from object_detection.protos import hyperparams_pb2

from protos import feature_extractors_pb2
from feature_extractors.feature_extractor import FeatureExtractor

slim = tf.contrib.slim


class FCExtractor(FeatureExtractor):

  def __init__(self, model_proto):
    """Initializes BOWEncoder.

    Args:
      model_proto: an instance of BOWEncoder proto.
    """
    if not isinstance(model_proto, feature_extractors_pb2.FCExtractor):
      raise ValueError('model_proto has to be an instance of FCExtractor.')
    self._model_proto = model_proto

  @property
  def scope(self):
    """Returns variable scope."""
    return self._model_proto.scope

  @property
  def default_image_size(self):
    """Returns default image size."""
    raise ValueError('FCExtractor is not an image feature extractor.')

  def extract_feature(self, input_data, is_training=True):
    """Extracts feature vectors using mobilenet v1 model.

    Args:
      input_data: a [batch, input_dims] float32 tensor denoting input feature.
      is_training: if True, update batch norm parameters.

    Returns:
      feature: a [batch, output_dims] tensor denoting extracted feature.
    """
    model_proto = self._model_proto

    output_hyperparams = hyperparams_builder.build(
        model_proto.output_hyperparams, is_training=is_training)

    node = input_data 
    if is_training:
      node = tf.nn.dropout(node, model_proto.input_dropout_keep_prob)

    with tf.variable_scope(self.scope):
      # Build hidden layers.
      if model_proto.hidden_layers > 0:
        hidden_hyperparams = hyperparams_builder.build(
            model_proto.hidden_hyperparams, is_training=is_training)
        with slim.arg_scope(hidden_hyperparams):
          for i in xrange(model_proto.hidden_layers):
            node = slim.fully_connected(node, 
                num_outputs=model_proto.hidden_units, scope='hidden_%d' % (i))
            if is_training:
              node = tf.nn.dropout(node, model_proto.hidden_dropout_keep_prob)

      # Build output layer.
      with slim.arg_scope(output_hyperparams):
        node = slim.fully_connected(node, 
            num_outputs=model_proto.output_units, scope='project')
        if is_training:
          node = tf.nn.dropout(node, model_proto.output_dropout_keep_prob)

    return node

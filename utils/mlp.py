
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from object_detection.builders import hyperparams_builder

from protos import mlp_pb2

slim = tf.contrib.slim


def model(model_proto, input_data, is_training=False):
  """Extracts feature vectors using mobilenet v1 model.

  Args:
    model_proto: an instance of mlp_pb2.MLPModel.
    input_data: a [batch, input_dims] float32 tensor denoting input feature.
    is_training: if True, update batch norm parameters.

  Raises:
    ValueError if model_proto is invalid.

  Returns:
    feature: a [batch, output_dims] tensor denoting extracted feature.
    init_fn: a callable that takes tf.Session as parameter.
  """
  if not isinstance(model_proto, mlp_pb2.MLPModel):
    raise ValueError('The model_proto has to be an instance of MLBModel.')

  output_hyperparams = hyperparams_builder.build(
      model_proto.output_hyperparams, is_training=is_training)

  node = slim.dropout(input_data, 
      model_proto.input_dropout_keep_prob, is_training=is_training)

  with tf.variable_scope(model_proto.scope):
    # Build hidden layers.
    if model_proto.hidden_layers > 0:
      hidden_hyperparams = hyperparams_builder.build(
          model_proto.hidden_hyperparams, is_training=is_training)

      with slim.arg_scope(hidden_hyperparams):
        for i in xrange(model_proto.hidden_layers):
          node = slim.fully_connected(node, 
              num_outputs=model_proto.hidden_units, 
              scope='hidden_%d' % (i))
          node = slim.dropout(node, 
              model_proto.hidden_dropout_keep_prob, 
              is_training=is_training)

    # Build output layer.
    with slim.arg_scope(output_hyperparams):
      node = slim.fully_connected(node, 
          num_outputs=model_proto.output_units, 
          scope='project')
      node = slim.dropout(node, 
          model_proto.output_dropout_keep_prob,
          is_training=is_training)

  init_fn = slim.assign_from_checkpoint_fn(
      model_proto.checkpoint_path,
      slim.get_model_variables(model_proto.scope))
  return node, init_fn

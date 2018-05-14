
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import tensorflow as tf
from tensorflow import logging

from models import utils
from models import image_stmt_model
from losses import triplet_loss
from protos import vse_model_pb2
from text_encoders import builder

slim = tf.contrib.slim


class Model(image_stmt_model.Model):
  """VSEModel."""

  def __init__(self, model_proto, is_training=False):
    """Initializes ads model.

    Args:
      model_proto: an instance of vse_model_pb2.VSEModel.
      is_training: if True, training graph would be built.
    """
    super(Model, self).__init__(model_proto, is_training)

    if not isinstance(model_proto, vse_model_pb2.VSEModel):
      raise ValueError("The model_proto has to be an instance of VSEModel.")

    self._stmt_encoder = builder.build(model_proto.stmt_encoder, is_training)
    self._mining_fn = triplet_loss.build_mining_func(model_proto.triplet_mining)

  def build_inference_graph(self, examples, **kwargs):
    """Builds tensorflow graph for inference.

    Args:
      examples: a python dict involving at least following k-v fields:
        img_features: a [batch, feature_dimensions] tf.float32 tensor.
        statement_strings: a [batch, statement_max_sent_len] tf.int64 tensor.
        statement_lengths: a [batch] tf.int64 tensor.

    Returns:
      predictions: a dict mapping from output names to output tensors.

    Raises:
      ValueError: if model_proto is not properly configured.
    """
    model_proto = self._model_proto
    is_training = self._is_training

    image_id = examples['image_id']

    # Encode image features.
    img_features = examples['img_features']
    roi_features = examples['roi_features']
    img_encoded = utils.encode_feature(
        img_features, model_proto.image_encoder, is_training)

    # Encode statement features.
    statement_strings = examples['statement_strings']
    statement_lengths = examples['statement_lengths']
    (stmt_encoded, _, _ 
     ) = self._stmt_encoder.encode(statement_strings, statement_lengths)
    self._init_fn_list.append(self._stmt_encoder.get_init_fn())

    # Joint embedding and cosine distance computation.
    img_encoded = tf.nn.l2_normalize(img_encoded, 1)
    stmt_encoded = tf.nn.l2_normalize(stmt_encoded, 1)

    predictions = {
      'image_id': image_id,
      'img_encoded': img_encoded,
      'stmt_encoded': stmt_encoded,
    }
    return predictions

  def build_evaluation_graph(self, examples, **kwargs):
    """Builds tensorflow graph for evaluation.

    Args:
      examples: a python dict involving at least following k-v fields:
        img_features: a [batch, feature_dimensions] tf.float32 tensor.
        statement_strings: a [batch, statement_max_sent_len] tf.int64 tensor.
        statement_lengths: a [batch] tf.int64 tensor.

    Returns:
      predictions: a dict mapping from output names to output tensors.

    Raises:
      ValueError: if model_proto is not properly configured.
    """
    model_proto = self._model_proto
    is_training = self._is_training

    image_id = examples['image_id']

    # Encode image features.
    img_features = examples['img_features']
    img_encoded = utils.encode_feature(
        img_features, model_proto.image_encoder, is_training)

    # Encode statement features.
    statement_strings = examples['eval_statement_strings']
    statement_lengths = examples['eval_statement_lengths']

    (number_of_val_stmts_per_image, max_stmt_len
     ) = statement_strings.get_shape().as_list()[1:]
    statement_strings_reshaped = tf.reshape(statement_strings, [-1, max_stmt_len])
    statement_lengths_reshaped = tf.reshape(statement_lengths, [-1])

    (stmt_encoded, _, _ 
     ) = self._stmt_encoder.encode(
       statement_strings_reshaped, statement_lengths_reshaped)
    self._init_fn_list.append(self._stmt_encoder.get_init_fn())

    # Joint embedding and cosine distance computation.
    img_encoded = tf.nn.l2_normalize(img_encoded, 1)
    stmt_encoded = tf.nn.l2_normalize(stmt_encoded, 1)
    stmt_encoded = tf.reshape(
        stmt_encoded, 
        [-1, number_of_val_stmts_per_image, stmt_encoded.get_shape()[-1].value])

    distance = 1 - tf.reduce_sum(tf.multiply(tf.expand_dims(img_encoded, 1), stmt_encoded), axis=2)
    predictions = {
      'image_id': image_id,
      'distance': distance,
      'eval_statement_strings': statement_strings,
      'eval_statement_lengths': statement_lengths,
    }
    return predictions


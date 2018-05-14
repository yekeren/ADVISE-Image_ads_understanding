
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import abc
import tensorflow as tf
from tensorflow import logging

slim = tf.contrib.slim


class Model(object):
  """Model interface."""
  __metaclass__ = abc.ABCMeta

  def __init__(self, model_proto, is_training=False):
    """Initializes ads model.

    Args:
      model_proto: an general object.
      is_training: if True, training graph would be built.
    """
    self._tensors = {}
    self._init_fn_list = []

    self._model_proto = model_proto
    self._is_training = is_training

  @abc.abstractmethod
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
    pass

  @abc.abstractmethod
  def build_loss(self, predictions, **kwargs):
    """Build tensorflow graph for computing loss.

    Args:
      predictions: a dict mapping from names to predicted tensors.

    Returns:
      loss_dict: a dict mapping from names to loss tensors.
    """
    pass

  def get_tensors(self):
    """Returns a dict containing important tensors for debugging.

    Returns:
      tensors: a dict mapping from tensor name to tensor.
    """
    return self._tensors

  def get_init_fn(self):
    """Returns a callable used to initialize the model.

    Returns:
      init_fn: a callable which takes tf.Session as a parameter.
    """
    def _init_fn(sess):
      for init_fn in self._init_fn_list:
        init_fn(sess)
      logging.info('Model is initialized.')
    return _init_fn

  def get_variables_to_train(self):
    """Returns model variables.
      
    Returns:
      variables: a list of model variables.
    """
    return tf.trainable_variables()

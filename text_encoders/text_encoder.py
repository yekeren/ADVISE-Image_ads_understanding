
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import abc


class TextEncoder(object):
  __metaclass__ = abc.ABCMeta

  def __init__(self, model_proto, is_training=False):
    """Initializes TextEncoder."""
    self._model_proto = model_proto
    self._is_training = is_training

    self._init_fn = None

  @abc.abstractmethod
  def encode(self, text_strings, text_lengths):
    """Encodes texts into embedding vectors.

    Args:
      text_strings: a [batch, max_text_len] tensor indicating multiple texts.
      text_lengths: a [batch] tensor indicating lenghts of each text.

    Returns:
      text_encoded: a [batch, embedding_size] tensor indicating final encoding.
      text_embedding: a [batch, max_text_len, embedding_size] tf.float tensor.
    """
    pass

  def get_init_fn(self):
    """Gets init_fn in order to initialize word embedding matrix.

    Returns:
      init_fn: a callable that takes tf.Session as an argument.
    """
    return self._init_fn


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import abc
import tensorflow as tf

slim = tf.contrib.slim


class TextEncoder(object):
  __metaclass__ = abc.ABCMeta

  def __init__(self, config):
    """Initializes TextEncoder."""
    pass

  @abc.abstractproperty
  def scope(self):
    """Returns variable scope."""
    pass

  @property
  def embedding_weights(self):
    """Returns embedding weights."""
    return self._embedding_weights

  @abc.abstractmethod
  def encode(self, text_lengths, text_strings, is_training=True):
    """Encodes texts into embedding vectors.

    Args:
      text_lengths: a [batch] tensor indicating lenghts of each text.
      text_strings: a [batch, max_text_len] tensor indicating multiple texts.
      is_training: if True, update batch norm parameters.
    """
    pass

  @abc.abstractmethod
  def assign_from_checkpoint_fn(self, checkpoint_path):
    """Returns a function to load from checkpoint.

    Args:
      checkpoint_path: path to the checkpoint file.

    Returns:
      assign_fn: a function that that load weights from checkpoint.
    """
    pass

  def build_weights(self, vocab_size, embedding_size, 
      init_width=0.08, weight_decay=0.0):
    """Builds word embedding matrix.

    Args:
      vocab_size: size of vocabulary.
      embedding_size: dimensions of embedding vector.
      init_width: init width of word embedding.
      weight_decay: weight decay of word embedding weights.

    Returns:
      embedding_weights: a [vocab_size, embedding_size] tensor.
    """
    with tf.variable_scope(self.scope):
      self._embedding_weights = tf.get_variable(
          name='weights', 
          shape=[vocab_size, embedding_size], 
          initializer=tf.random_uniform_initializer(-init_width, init_width),
          regularizer=slim.l2_regularizer(weight_decay))
    return self._embedding_weights
